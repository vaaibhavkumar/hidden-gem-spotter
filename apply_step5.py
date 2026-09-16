from pathlib import Path

content_ingestion_data_store_py = r'''"""
Storage layer (answers "reliable and scalable way to store data").

Design choice: Parquet files as the on-disk format (columnar, compressed,
portable, the same format real table formats like Delta Lake / Iceberg are
built on top of), queried through DuckDB — an embedded, serverless SQL
engine (no server to run, just a library + one file), which comfortably
handles the full ~500-ticker x hourly x multi-year universe on a laptop.

This is deliberately NOT Delta Lake / Apache Iceberg. Those add a
transaction log and catalog on top of Parquet for concurrent multi-writer
access, schema evolution, and time travel — real needs for a shared data
lake with multiple pipelines writing at once, not for one person's local
research project. If this ever grows into a team tool backed by cloud
storage, upgrading the Parquet files under here to an Iceberg/Delta table
is a natural next step; it would be premature now.

Two tables live in data/market.duckdb:
    bars    — ticker, timestamp, open, high, low, close, volume
    signals — a persisted history of every fired signal (ticker, timestamp,
              direction, score, price) — this is what makes "track a
              stock's signal history, not just its current state" (section
              2F of the proposal) an actual queryable log instead of just
              an idea.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# This file now lives in ingestion/, one level below the repo root, so
# DATA_DIR/SCHEMA_PATH both need to go up one extra level to land in the
# same place they always did (repo_root/data, repo_root/db/schema.sql) —
# not inside ingestion/ itself.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "market.duckdb"
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"


def _connect() -> duckdb.DuckDBPyConnection:
    DATA_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    # db/schema.sql is the single source of truth for table structure —
    # checked into git so the schema has real history (see that file's
    # header). Every statement in it is CREATE ... IF NOT EXISTS, so
    # applying it on every connection is always safe.
    con.execute(SCHEMA_PATH.read_text())
    return con


def write_raw_parquet(ticker: str, df: pd.DataFrame) -> Path:
    """Landing zone: one Parquet file per ticker, exactly as ingested."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker}.parquet"
    df.to_parquet(path, index=False)
    return path


def upsert_bars(ticker: str, df: pd.DataFrame) -> int:
    """
    Loads a ticker's OHLCV frame into the `bars` table, replacing any
    existing rows for the same (ticker, timestamp) pairs — safe to re-run
    after pulling fresh/overlapping data.
    """
    con = _connect()
    tmp = df.copy()
    tmp["ticker"] = ticker
    con.register("tmp_bars", tmp[["ticker", "timestamp", "open", "high", "low", "close", "volume"]])
    con.execute("DELETE FROM bars WHERE ticker = ? AND timestamp IN (SELECT timestamp FROM tmp_bars)", [ticker])
    con.execute("INSERT INTO bars SELECT * FROM tmp_bars")
    n = con.execute("SELECT COUNT(*) FROM bars WHERE ticker = ?", [ticker]).fetchone()[0]
    con.close()
    return n


def load_bars(ticker: str) -> pd.DataFrame:
    con = _connect()
    df = con.execute(
        "SELECT timestamp, open, high, low, close, volume FROM bars WHERE ticker = ? ORDER BY timestamp",
        [ticker],
    ).df()
    con.close()
    return df


def log_signals(ticker: str, df: pd.DataFrame) -> None:
    """
    Appends every fired bull_signal/bear_signal row from a scored
    DataFrame (see scoring.generate_signals) into the persistent signals
    log. Call this once per backtest/live run so the signal history
    accumulates across runs instead of being recomputed and discarded.
    """
    con = _connect()
    rows = []
    for direction, flag_col, score_col in (("bull", "bull_signal", "bull_score"), ("bear", "bear_signal", "bear_score")):
        fired = df[df[flag_col]]
        for _, row in fired.iterrows():
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": row["timestamp"],
                    "direction": direction,
                    "score": int(row[score_col]),
                    "price": float(row["close"]),
                }
            )
    if not rows:
        con.close()
        return
    tmp = pd.DataFrame(rows)
    con.register("tmp_signals", tmp)
    con.execute(
        "DELETE FROM signals WHERE (ticker, timestamp, direction) IN "
        "(SELECT ticker, timestamp, direction FROM tmp_signals)"
    )
    con.execute("INSERT INTO signals SELECT ticker, timestamp, direction, score, price, current_timestamp FROM tmp_signals")
    con.close()


def load_signal_history(ticker: str | None = None) -> pd.DataFrame:
    con = _connect()
    if ticker:
        df = con.execute("SELECT * FROM signals WHERE ticker = ? ORDER BY timestamp", [ticker]).df()
    else:
        df = con.execute("SELECT * FROM signals ORDER BY ticker, timestamp").df()
    con.close()
    return df


def get_last_bar_timestamp(ticker: str) -> pd.Timestamp | None:
    """
    Max stored `bars.timestamp` for this ticker, or None if it has no rows
    yet. This is what makes ingestion/alpaca_ingest.py's fetch a delta
    (incremental) load on every run after the first: a ticker with a known
    last timestamp only needs bars *after* it, not the full YEARS_OF_HISTORY
    lookback again. A brand-new ticker (None here) still gets the full
    bootstrap fetch.
    """
    con = _connect()
    result = con.execute("SELECT MAX(timestamp) FROM bars WHERE ticker = ?", [ticker]).fetchone()[0]
    con.close()
    return pd.Timestamp(result) if result is not None else None


def log_recommendation(rec) -> None:
    """
    Persists one recommendation.recommend.Recommendation to the
    `recommendations` table, keyed on (ticker, timestamp) — re-running the
    pipeline against the same bar overwrites that row rather than
    duplicating it, the same delete-then-insert upsert pattern as
    upsert_bars()/log_signals() above. `rec` is duck-typed (not imported by
    type) to avoid a circular import between ingestion and recommendation.
    """
    con = _connect()
    lo, hi = rec.confidence_range if rec.confidence_range else (None, None)
    tmp = pd.DataFrame(
        [
            {
                "ticker": rec.ticker,
                "timestamp": rec.timestamp,
                "action": rec.action,
                "composite_score": float(rec.composite_score),
                "pillars_used": ",".join(rec.pillars_used),
                "confidence_pct": rec.confidence_pct,
                "confidence_lo": lo,
                "confidence_hi": hi,
                "confidence_note": rec.confidence_note,
                "reasoning": "\n".join(rec.reasoning),
                "price": float(rec.price),
            }
        ]
    )
    con.register("tmp_recs", tmp)
    con.execute("DELETE FROM recommendations WHERE (ticker, timestamp) IN (SELECT ticker, timestamp FROM tmp_recs)")
    con.execute(
        "INSERT INTO recommendations "
        "SELECT ticker, timestamp, action, composite_score, pillars_used, confidence_pct, "
        "confidence_lo, confidence_hi, confidence_note, reasoning, price, current_timestamp FROM tmp_recs"
    )
    con.close()


def log_recommendations(recs) -> None:
    """Convenience wrapper: log_recommendation() for each item in an iterable."""
    for rec in recs:
        log_recommendation(rec)


def upsert_analyst_consensus(ticker: str, rows: list[dict]) -> int:
    """
    Loads a ticker's analyst-consensus snapshots into the
    `analyst_consensus` table, replacing any existing rows for the same
    (ticker, as_of_date) pairs — safe to re-run. `rows` is a list of dicts
    with keys: as_of_date, strong_buy, buy, hold, sell, strong_sell,
    revision_score (see ingestion/finnhub_ingest.py for how these are
    built from Finnhub's raw API response). Returns the total row count
    now stored for this ticker.
    """
    if not rows:
        return 0
    con = _connect()
    tmp = pd.DataFrame(rows).copy()
    tmp["ticker"] = ticker
    tmp["as_of_date"] = pd.to_datetime(tmp["as_of_date"]).dt.date  # Finnhub sends "YYYY-MM-DD" strings; the
    # schema's as_of_date column is DATE, and DuckDB won't implicitly compare DATE to VARCHAR in the DELETE below.
    # Finnhub's recommendation-trends response isn't always deduplicated
    # per period -- observed in practice (XOM, 2026-09-16) returning two
    # entries for the same "2026-07-01" period in one response. Without
    # this, INSERT below would try to write two rows sharing the same
    # (ticker, as_of_date) primary key and DuckDB would (correctly) raise
    # a ConstraintException. keep="last" means the later entry in
    # Finnhub's own response order wins for a given period -- an
    # arbitrary but deterministic tiebreak, not a guarantee Finnhub
    # orders duplicates any particular way.
    tmp = tmp.drop_duplicates(subset=["as_of_date"], keep="last")
    tmp = tmp[["ticker", "as_of_date", "strong_buy", "buy", "hold", "sell", "strong_sell", "revision_score"]]
    con.register("tmp_consensus", tmp)
    con.execute(
        "DELETE FROM analyst_consensus WHERE ticker = ? AND as_of_date IN (SELECT as_of_date FROM tmp_consensus)",
        [ticker],
    )
    con.execute(
        "INSERT INTO analyst_consensus "
        "SELECT ticker, as_of_date, strong_buy, buy, hold, sell, strong_sell, revision_score, "
        "current_timestamp FROM tmp_consensus"
    )
    n = con.execute("SELECT COUNT(*) FROM analyst_consensus WHERE ticker = ?", [ticker]).fetchone()[0]
    con.close()
    return n


def load_latest_revision_score(ticker: str) -> float | None:
    """
    The most recent analyst-consensus revision_score for `ticker` (0-100),
    or None if there's no Finnhub coverage stored for it yet — the None
    case is the normal, expected state until ingestion/finnhub_ingest.py
    has been run, and recommend.recommend() already treats a None
    revision_score as "pillar not available" rather than erroring.
    """
    con = _connect()
    result = con.execute(
        "SELECT revision_score FROM analyst_consensus WHERE ticker = ? "
        "ORDER BY as_of_date DESC LIMIT 1",
        [ticker],
    ).fetchone()
    con.close()
    if result is None or result[0] is None:
        return None
    return float(result[0])


def load_recommendation_history(ticker: str | None = None) -> pd.DataFrame:
    """
    Every recommendation ever logged for `ticker` (or all tickers if
    omitted), oldest first — the "see current AND previous buy/sell/hold
    recommendations" view, as an actual queryable table rather than
    whatever the latest report.html happens to show.
    """
    con = _connect()
    if ticker:
        df = con.execute("SELECT * FROM recommendations WHERE ticker = ? ORDER BY timestamp", [ticker]).df()
    else:
        df = con.execute("SELECT * FROM recommendations ORDER BY ticker, timestamp").df()
    con.close()
    return df
'''
Path("ingestion/data_store.py").write_text(content_ingestion_data_store_py)
print("wrote ingestion/data_store.py:", len(content_ingestion_data_store_py), "bytes")

content_tests_test_data_store_py = r'''"""
Level 1 tests for ingestion/data_store.py — the Parquet + DuckDB storage
layer. These redirect DATA_DIR/RAW_DIR/DB_PATH into a pytest tmp_path so
no test ever touches the real (git-ignored) data/ directory, but they
still exercise the real db/schema.sql file, since applying that schema
correctly is exactly what needs to be tested.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ingestion import data_store


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point every data_store path constant at a throwaway directory."""
    monkeypatch.setattr(data_store, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(data_store, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(data_store, "DB_PATH", tmp_path / "data" / "market.duckdb")
    # SCHEMA_PATH is left pointing at the real db/schema.sql on purpose —
    # that's the file this test suite is meant to catch breakage in.


def _sample_bars():
    ts = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [10.5, 11.5, 12.5, 13.5, 14.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5],
            "close": [10.2, 11.2, 12.2, 13.2, 14.2],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )


def test_schema_applies_and_creates_both_tables():
    con = data_store._connect()
    tables = {row[0] for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    con.close()
    assert {"bars", "signals"}.issubset(tables)


def test_write_raw_parquet_round_trips():
    df = _sample_bars()
    path = data_store.write_raw_parquet("TEST", df)
    assert path.exists()
    reloaded = pd.read_parquet(path)
    assert len(reloaded) == len(df)


def test_upsert_and_load_bars_round_trip():
    df = _sample_bars()
    n_stored = data_store.upsert_bars("TEST", df)
    assert n_stored == 5

    loaded = data_store.load_bars("TEST")
    assert len(loaded) == 5
    assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert loaded["close"].tolist() == df["close"].tolist()


def test_upsert_bars_is_idempotent_on_rerun():
    df = _sample_bars()
    data_store.upsert_bars("TEST", df)
    n_stored_again = data_store.upsert_bars("TEST", df)  # re-run with identical rows
    assert n_stored_again == 5  # no duplicates from re-ingesting the same bars


def test_log_and_load_signal_history():
    df = _sample_bars()
    df["bull_signal"] = [False, True, False, False, True]
    df["bear_signal"] = [False, False, True, False, False]
    df["bull_score"] = [0, 5, 0, 0, 4]
    df["bear_score"] = [0, 0, 4, 0, 0]

    data_store.log_signals("TEST", df)
    history = data_store.load_signal_history("TEST")

    assert len(history) == 3  # two bull signals + one bear signal
    assert set(history["direction"]) == {"bull", "bear"}


def test_load_signal_history_with_no_ticker_filter_returns_everything():
    df = _sample_bars()
    df["bull_signal"] = [False, True, False, False, False]
    df["bear_signal"] = [False, False, False, False, False]
    df["bull_score"] = [0, 5, 0, 0, 0]
    df["bear_score"] = [0, 0, 0, 0, 0]

    data_store.log_signals("AAA", df)
    data_store.log_signals("BBB", df)

    history = data_store.load_signal_history()
    assert set(history["ticker"]) == {"AAA", "BBB"}


def test_get_last_bar_timestamp_is_none_for_unknown_ticker():
    assert data_store.get_last_bar_timestamp("NOPE") is None


def test_get_last_bar_timestamp_returns_max_stored_timestamp():
    df = _sample_bars()
    data_store.upsert_bars("TEST", df)
    last = data_store.get_last_bar_timestamp("TEST")
    assert last == pd.Timestamp(df["timestamp"].max())


class _FakeRecommendation:
    """Minimal duck-typed stand-in for recommendation.recommend.Recommendation,
    used so this test doesn't need to import the recommendation package
    (which would pull in signals/config for no reason a storage-layer test
    should care about)."""

    def __init__(self, ticker, timestamp, action="BUY", composite_score=72.5,
                 pillars_used=("technical",), confidence_pct=61.0,
                 confidence_range=(0.5, 0.7), confidence_note="from backtest bucket n=40",
                 reasoning=("[Technical/bullish] example",), price=123.45):
        self.ticker = ticker
        self.timestamp = timestamp
        self.action = action
        self.composite_score = composite_score
        self.pillars_used = list(pillars_used)
        self.confidence_pct = confidence_pct
        self.confidence_range = confidence_range
        self.confidence_note = confidence_note
        self.reasoning = list(reasoning)
        self.price = price


def test_log_recommendation_and_load_history_round_trip():
    ts = pd.Timestamp("2024-01-01")
    rec = _FakeRecommendation("TEST", ts)
    data_store.log_recommendation(rec)

    history = data_store.load_recommendation_history("TEST")
    assert len(history) == 1
    row = history.iloc[0]
    assert row["action"] == "BUY"
    assert row["composite_score"] == 72.5
    assert row["pillars_used"] == "technical"
    assert row["confidence_lo"] == 0.5
    assert row["confidence_hi"] == 0.7


def test_log_recommendation_upserts_on_same_ticker_and_timestamp():
    ts = pd.Timestamp("2024-01-01")
    data_store.log_recommendation(_FakeRecommendation("TEST", ts, action="HOLD", composite_score=50.0))
    data_store.log_recommendation(_FakeRecommendation("TEST", ts, action="BUY", composite_score=72.5))

    history = data_store.load_recommendation_history("TEST")
    assert len(history) == 1  # re-run against the same bar overwrites, doesn't duplicate
    assert history.iloc[0]["action"] == "BUY"


def test_log_recommendation_with_no_calibration_stores_nulls():
    ts = pd.Timestamp("2024-01-01")
    rec = _FakeRecommendation("TEST", ts, confidence_pct=None, confidence_range=None,
                               confidence_note="no calibration table supplied")
    data_store.log_recommendation(rec)

    history = data_store.load_recommendation_history("TEST")
    assert pd.isna(history.iloc[0]["confidence_pct"])
    assert pd.isna(history.iloc[0]["confidence_lo"])


def test_log_recommendations_plural_logs_each_item():
    ts = pd.Timestamp("2024-01-01")
    data_store.log_recommendations([_FakeRecommendation("AAA", ts), _FakeRecommendation("BBB", ts)])
    assert set(data_store.load_recommendation_history()["ticker"]) == {"AAA", "BBB"}


def _sample_consensus_rows():
    return [
        {"as_of_date": "2026-09-01", "strong_buy": 13, "buy": 24, "hold": 7, "sell": 0, "strong_sell": 0, "revision_score": 88.6},
        {"as_of_date": "2026-08-01", "strong_buy": 10, "buy": 20, "hold": 10, "sell": 2, "strong_sell": 0, "revision_score": 78.6},
    ]


def test_load_latest_revision_score_is_none_with_no_coverage():
    assert data_store.load_latest_revision_score("NOPE") is None


def test_upsert_and_load_latest_revision_score_round_trip():
    n = data_store.upsert_analyst_consensus("TEST", _sample_consensus_rows())
    assert n == 2
    # ORDER BY as_of_date DESC -> the 2026-09-01 row's score, not 08-01's.
    assert data_store.load_latest_revision_score("TEST") == 88.6


def test_upsert_analyst_consensus_is_idempotent_on_rerun():
    data_store.upsert_analyst_consensus("TEST", _sample_consensus_rows())
    n_again = data_store.upsert_analyst_consensus("TEST", _sample_consensus_rows())
    assert n_again == 2  # no duplicates from re-ingesting the same periods


def test_load_latest_revision_score_handles_null_score_period():
    # A period with zero total analysts stores revision_score=None (see
    # finnhub_ingest._consensus_to_score) -- that shouldn't crash the lookup.
    rows = [{"as_of_date": "2026-09-01", "strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0, "revision_score": None}]
    data_store.upsert_analyst_consensus("TEST", rows)
    assert data_store.load_latest_revision_score("TEST") is None


def test_upsert_analyst_consensus_with_empty_rows_is_a_noop():
    assert data_store.upsert_analyst_consensus("TEST", []) == 0


def test_upsert_analyst_consensus_dedupes_duplicate_period_in_one_batch():
    # Reproduces a real Finnhub response quirk hit in practice (XOM,
    # 2026-09-16): two entries for the same period in a single fetch. This
    # must not raise a PRIMARY KEY constraint violation, and should keep
    # exactly one row per (ticker, as_of_date) -- the later one in the
    # input list, per this function's documented keep="last" tiebreak.
    rows = [
        {"as_of_date": "2026-07-01", "strong_buy": 5, "buy": 10, "hold": 3, "sell": 0, "strong_sell": 0, "revision_score": 84.4},
        {"as_of_date": "2026-07-01", "strong_buy": 4, "buy": 9, "hold": 4, "sell": 1, "strong_sell": 0, "revision_score": 79.4},
    ]
    n = data_store.upsert_analyst_consensus("XOM", rows)
    assert n == 1  # duplicate period collapsed to a single stored row
    assert data_store.load_latest_revision_score("XOM") == 79.4  # the "last" of the two duplicates
'''
Path("tests/test_data_store.py").write_text(content_tests_test_data_store_py)
print("wrote tests/test_data_store.py:", len(content_tests_test_data_store_py), "bytes")