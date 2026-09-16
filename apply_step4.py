from pathlib import Path

content_db_schema_sql = r'''-- Hidden Gem Spotter — DuckDB schema.
--
-- Single source of truth for data/market.duckdb's structure. Checked into
-- git so the database structure has a real history (this is the answer to
-- "have the db schema, tables, views etc. present in git" without needing
-- a Docker/Kubernetes deployment for a single-user local prototype).
--
-- Applied automatically by ingestion/data_store.py._connect() on every
-- connection (every statement is CREATE ... IF NOT EXISTS, so re-running
-- this file against an existing database is always safe).

-- One row per (ticker, timestamp) OHLCV bar. Populated by
-- ingestion/data_store.upsert_bars(), fed by ingestion/alpaca_ingest.py.
CREATE TABLE IF NOT EXISTS bars (
    ticker    VARCHAR,
    timestamp TIMESTAMP,
    open      DOUBLE,
    high      DOUBLE,
    low       DOUBLE,
    close     DOUBLE,
    volume    BIGINT,
    PRIMARY KEY (ticker, timestamp)
);

-- A persisted history of every fired buy/sell signal, so "signal history"
-- is an actual queryable log rather than something recomputed and thrown
-- away each run (section 2F of the project proposal). Populated by
-- ingestion/data_store.log_signals(), fed by signals/scoring.generate_signals().
CREATE TABLE IF NOT EXISTS signals (
    ticker      VARCHAR,
    timestamp   TIMESTAMP,
    direction   VARCHAR,     -- 'bull' | 'bear'
    score       INTEGER,     -- raw condition count at signal time (signals/scoring.py)
    price       DOUBLE,
    logged_at   TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (ticker, timestamp, direction)
);

-- A persisted history of every recommendation.Recommendation the pipeline
-- has ever produced, one row per (ticker, timestamp) it was computed for —
-- this is what makes "see current AND previous buy/sell/hold
-- recommendations" an actual queryable log instead of just today's
-- report.html snapshot. Populated by ingestion/data_store.log_recommendation(),
-- fed by recommendation/recommend.recommend()'s output in run_real_backtest.py.
-- Re-running the pipeline against the same bar overwrites that row instead
-- of duplicating it (same upsert-via-delete-then-insert pattern as `bars`
-- and `signals` above — see data_store.py's module docstring for why this
-- isn't a truly atomic upsert, and why that's one of the arguments for a
-- future Delta Lake migration).
-- One row per (ticker, monthly period) analyst-consensus snapshot from
-- Finnhub's free-tier "recommendation trends" endpoint — the data behind
-- recommendation/recommend.py's "revision" pillar (proposal section 2C).
-- Populated by ingestion/finnhub_ingest.py, read by
-- data_store.load_latest_revision_score() and fed into
-- recommend.recommend()'s revision_score parameter. Storing every period
-- (not just the latest) as its own row, keyed on (ticker, as_of_date),
-- means this table already accumulates real history for free — a future
-- "track revisions over time" upgrade (comparing this month's consensus
-- to last month's) can be built entirely on top of rows already here,
-- without changing how ingestion works.
CREATE TABLE IF NOT EXISTS analyst_consensus (
    ticker          VARCHAR,
    as_of_date      DATE,      -- the monthly period Finnhub reports this snapshot for
    strong_buy      INTEGER,
    buy             INTEGER,
    hold            INTEGER,
    sell            INTEGER,
    strong_sell     INTEGER,
    revision_score  DOUBLE,    -- 0-100, see finnhub_ingest.py's _consensus_to_score()
    fetched_at      TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS recommendations (
    ticker            VARCHAR,
    timestamp         TIMESTAMP,   -- the bar timestamp the recommendation was computed from
    action            VARCHAR,     -- 'STRONG BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG SELL'
    composite_score   DOUBLE,      -- 0-100, see recommendation/recommend.py
    pillars_used      VARCHAR,     -- comma-joined, e.g. 'technical' (only pillar built so far)
    confidence_pct    DOUBLE,      -- NULL when uncalibrated
    confidence_lo     DOUBLE,      -- Wilson interval low bound, NULL when uncalibrated
    confidence_hi     DOUBLE,      -- Wilson interval high bound, NULL when uncalibrated
    confidence_note   VARCHAR,
    reasoning         VARCHAR,     -- newline-joined reasoning bullets
    price             DOUBLE,
    logged_at         TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (ticker, timestamp)
);
'''
Path("db/schema.sql").write_text(content_db_schema_sql)
print("wrote db/schema.sql:", len(content_db_schema_sql), "bytes")

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

content_ingestion_finnhub_ingest_py = r'''"""
Analyst-consensus ingestion via Finnhub's free-tier "recommendation
trends" endpoint — this is the data source for
recommendation/recommend.py's "revision" pillar (proposal section 2C),
which up to now has always been None (composite scores have only ever
used the technical pillar).

Run this on a machine with normal internet access (same constraint as
ingestion/alpaca_ingest.py — Claude's cloud workspace can't reach it).

Setup (one time):
    pip install requests
    # Get a free API key from https://finnhub.io/register (no credit
    # card required for the free tier: 60 API calls/minute).
    # Do NOT paste your key into a chat with Claude or commit it to
    # source control. Set it as an environment variable instead:
    export FINNHUB_API_KEY="your_api_key"

Run (from the repo root):
    python3 -m ingestion.finnhub_ingest

Output: one row per (ticker, monthly period) in data/market.duckdb's
`analyst_consensus` table (see db/schema.sql) — see
data_store.load_latest_revision_score() for how run_real_backtest.py
reads the most recent snapshot back out.

Scope note (2026-09-16): this is a v1 -- a snapshot of *today's* analyst
consensus, mapped to a 0-100 score. It is NOT yet the "trend" version the
original proposal describes (EPS estimates/ratings actually being
revised *upward over time* is the more predictive signal). Every period
Finnhub returns is stored as its own row here rather than just the
latest, specifically so that upgrade can be built later purely as a query
over rows already collected, without changing how ingestion works.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make config.py (repo root, one directory up) importable when run from
# inside ingestion/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from ingestion import data_store  # noqa: E402

try:
    import requests
except ImportError:
    print("Missing dependency. Run: pip install requests", file=sys.stderr)
    raise

FINNHUB_RECOMMENDATION_URL = "https://finnhub.io/api/v1/stock/recommendation"

# Maps Finnhub's rating buckets to a 0-100 scale, used to collapse a
# (strongBuy, buy, hold, sell, strongSell) count breakdown into the single
# revision_score number recommend.py's composite actually consumes.
# Analyst-count-weighted, not just "whatever the majority says" -- a
# 20-buy/1-sell split scores much higher than an 11-buy/10-sell split,
# same principle as scoring.py's condition counts.
_RATING_WEIGHTS = {"strongBuy": 100.0, "buy": 75.0, "hold": 50.0, "sell": 25.0, "strongSell": 0.0}


def _consensus_to_score(period: dict) -> float | None:
    """
    period: one entry from Finnhub's recommendation-trends response, e.g.
        {"symbol": "AAPL", "period": "2026-09-01", "strongBuy": 13,
         "buy": 24, "hold": 7, "sell": 0, "strongSell": 0}
    Returns a 0-100 analyst-count-weighted score, or None if Finnhub
    reports zero analysts covering this ticker for this period (some
    smaller/newer names have no coverage at all -- that's real, not an
    error, and should surface as "no revision pillar" rather than a
    fabricated neutral score).
    """
    total = sum(period.get(k, 0) for k in _RATING_WEIGHTS)
    if total == 0:
        return None
    weighted = sum(period.get(k, 0) * w for k, w in _RATING_WEIGHTS.items())
    return weighted / total


def fetch_recommendation_trends(ticker: str, api_key: str) -> list[dict]:
    resp = requests.get(
        FINNHUB_RECOMMENDATION_URL,
        params={"symbol": ticker, "token": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        print(
            "Set the FINNHUB_API_KEY environment variable first "
            "(see the docstring at the top of this file).",
            file=sys.stderr,
        )
        sys.exit(1)

    # SPY is an ETF, not a stock any analyst issues a buy/sell rating on --
    # Finnhub has no coverage for it and it isn't fed into recommend()
    # anyway (config.BENCHMARK is only used for relative-strength math).
    tickers = config.all_tickers()

    for ticker in tickers:
        print(f"Fetching {ticker}...", end=" ", flush=True)
        try:
            periods = fetch_recommendation_trends(ticker, api_key)
        except Exception as exc:  # noqa: BLE001 - surface any API error and keep going
            print(f"FAILED ({exc})")
            continue

        if not periods:
            print("no analyst coverage returned")
            continue

        rows = []
        for period in periods:
            score = _consensus_to_score(period)
            rows.append(
                {
                    "as_of_date": period["period"],
                    "strong_buy": int(period.get("strongBuy", 0)),
                    "buy": int(period.get("buy", 0)),
                    "hold": int(period.get("hold", 0)),
                    "sell": int(period.get("sell", 0)),
                    "strong_sell": int(period.get("strongSell", 0)),
                    "revision_score": score,
                }
            )
        n_stored = data_store.upsert_analyst_consensus(ticker, rows)
        latest_score = rows[0]["revision_score"] if rows else None  # Finnhub returns newest period first
        print(f"{len(rows)} periods fetched, latest revision_score={latest_score}, {n_stored} total rows stored")
        time.sleep(1.1)  # stay under Finnhub's free-tier 60 calls/minute limit

    print(f"\nDone. Bring {data_store.DATA_DIR} back to your Claude session to run the real backtest.")


if __name__ == "__main__":
    main()
'''
Path("ingestion/finnhub_ingest.py").write_text(content_ingestion_finnhub_ingest_py)
print("wrote ingestion/finnhub_ingest.py:", len(content_ingestion_finnhub_ingest_py), "bytes")

content_run_real_backtest_py = r'''"""
Same pipeline as demo.py, but reading real OHLCV data (produced by
ingestion/alpaca_ingest.py) instead of synthetic series.

Run: python3 run_real_backtest.py

Reads from data/market.duckdb (see ingestion/data_store.py) if present —
this is the path alpaca_ingest.py now writes to. Falls back to
data/<TICKER>.csv for backward compatibility with the very first
prototype run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from evaluation import backtest
from ingestion import data_store
from recommendation import html_report, recommend
from signals import features, scoring

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_ticker(ticker: str) -> pd.DataFrame:
    df = data_store.load_bars(ticker)
    if not df.empty:
        return df.sort_values("timestamp").reset_index(drop=True)

    csv_path = DATA_DIR / f"{ticker}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    raise FileNotFoundError(
        f"No data for {ticker} in {data_store.DB_PATH} or {csv_path} — run "
        "python3 -m ingestion.alpaca_ingest (on a machine with internet "
        "access) and bring its data/ folder here first."
    )


def detect_bars_per_day(df: pd.DataFrame) -> int:
    """
    Infers whether this data holds daily or (regular-session) hourly bars
    from the median gap between consecutive timestamps, and returns the
    BARS_PER_DAY value config.py's window sizes should use. This exists so
    a daily-vs-hourly mismatch (see config.py's BARS_PER_DAY comment)
    can't silently produce a 7x-too-short "50-day" moving average —
    it's detected and applied automatically instead.
    """
    deltas = df["timestamp"].diff().dropna()
    if deltas.empty:
        return config.BARS_PER_DAY
    median_hours = deltas.median().total_seconds() / 3600
    if median_hours >= 20:   # ~1 trading day between bars (weekends inflate the mean, not the median)
        return 1
    return 7                 # hourly bars, ~7 per regular trading session


def main() -> None:
    pd.set_option("display.width", 120)

    benchmark = load_ticker(config.BENCHMARK)

    detected = detect_bars_per_day(benchmark)
    if detected != config.BARS_PER_DAY:
        print(
            f"Detected {'daily' if detected == 1 else 'hourly'} bars from {config.BENCHMARK}'s "
            f"timestamps — calling config.set_bars_per_day({detected}) so moving-average/52-week "
            f"windows are the right length (was {config.BARS_PER_DAY}). If this guess is wrong for "
            "your data, call config.set_bars_per_day(...) yourself before running the pipeline."
        )
        config.set_bars_per_day(detected)

    # Validation tickers keep their known riser/faller/normal label; the
    # broader calibration tickers have no such label (that's the point —
    # they're an unselected sample) and are excluded from the by-role
    # false-positive summary below, but included everywhere else (feature
    # computation, RS-percentile ranking, and confidence calibration) —
    # see config.all_tickers()'s docstring for why the two sets exist.
    roles = {t: meta["role"] for t, meta in config.VALIDATION_UNIVERSE.items()}
    universe = config.all_tickers()
    series = {t: load_ticker(t) for t in universe}

    # Align every ticker's frame to the benchmark's timestamps (inner join)
    # so the cross-sectional RS-percentile ranking in scoring.py compares
    # apples to apples at each bar.
    aligned = {}
    for t, df in series.items():
        merged = pd.merge(df, benchmark[["timestamp", "close"]], on="timestamp", suffixes=("", "_bench"))
        aligned[t] = merged.drop(columns=["close_bench"]).reset_index(drop=True)
        aligned[t]["_bench_close"] = merged["close_bench"].reset_index(drop=True)

    feats = {t: features.compute_features(df, df["_bench_close"]) for t, df in aligned.items()}
    # Ranked across the FULL universe (validation + calibration), not just
    # the 15 hand-picked names — this is the actual fix for "RS-percentile
    # ranking is only correct across whatever tickers you feed it" (see
    # README's "Known open items").
    scoring.attach_rs_percentile(feats)

    signaled = {t: scoring.generate_signals(df) for t, df in feats.items()}

    print(f"=== Fresh signal counts ({len(roles)} validation tickers; "
          f"{len(universe) - len(roles)} calibration-only tickers omitted from this table) ===")
    for t, df in signaled.items():
        data_store.log_signals(t, df)  # persist to data/market.duckdb's `signals` table
        if t not in roles:
            continue
        role = roles[t]
        print(
            f"{t:6s} ({role:6s}) bull_signals={int(df['bull_signal'].sum()):3d}  "
            f"bear_signals={int(df['bear_signal'].sum()):3d}"
        )

    print("\n=== Signal dates (first 5 of each type per validation ticker) ===")
    for t, df in signaled.items():
        if t not in roles:
            continue
        bulls = df.loc[df["bull_signal"], ["timestamp", "close"]].head(5)
        bears = df.loc[df["bear_signal"], ["timestamp", "close"]].head(5)
        if not bulls.empty:
            print(f"{t} BUY signals:\n{bulls.to_string(index=False)}")
        if not bears.empty:
            print(f"{t} SELL signals:\n{bears.to_string(index=False)}")

    horizon = 5 * config.BARS_PER_DAY  # ~5 trading days ahead, in whatever bar size the data uses
    results = {t: backtest.evaluate_signals(df, horizon=horizon) for t, df in signaled.items()}

    print("\n=== Backtest summary by role (this is the false-positive-rate check — ")
    print("    validation tickers only, since 'role' is only meaningful for the hand-picked set) ===")
    validation_results = {t: r for t, r in results.items() if t in roles}
    summary = backtest.summarize_universe(validation_results, roles)
    print(summary.to_string(index=False) if not summary.empty else "(no signals to evaluate)")

    print(f"\n=== Calibrated confidence by score bucket (validation + calibration tickers, "
          f"{len(universe)} total — still a proxy sample, see config.CALIBRATION_UNIVERSE's ")
    print("    point-in-time caveat, but no longer bucketed on just 15 correlated names) ===")
    all_results = pd.concat([r for r in results.values() if not r.empty], ignore_index=True) if any(len(r) for r in results.values()) else pd.DataFrame()
    calibration = backtest.calibrate_confidence(all_results) if not all_results.empty else pd.DataFrame()
    print(calibration.to_string(index=False) if not calibration.empty else "(not enough signals to calibrate)")

    print("\n=== Most recent recommendation per validation ticker (Action / Score / Confidence / Reasoning) ===")
    recommendations = []
    for t, df in signaled.items():
        if t not in roles:
            continue
        latest = df.iloc[-1]
        # None until ingestion/finnhub_ingest.py has been run for this
        # ticker -- recommend() already treats a None revision_score as
        # "pillar not available" (see its "Pillars not yet available"
        # reasoning note), so this is safe to call unconditionally.
        revision_score = data_store.load_latest_revision_score(t)
        rec = recommend.recommend(latest, ticker=t, revision_score=revision_score, calibration_table=calibration)
        print(rec)
        recommendations.append(rec)
        data_store.log_recommendation(rec)  # persist to data/market.duckdb's `recommendations` table

    # One recommendation history per validation ticker (oldest -> newest),
    # so report.html can show "how has this call changed over past runs",
    # not just today's snapshot.
    history = {t: data_store.load_recommendation_history(t) for t in roles}

    report_path = html_report.generate_html_report(recommendations, roles, history=history, out_path="report.html")
    print(f"\nWrote {report_path} -- open it in a browser to see the watchlist summary + per-ticker detail.")


if __name__ == "__main__":
    main()
'''
Path("run_real_backtest.py").write_text(content_run_real_backtest_py)
print("wrote run_real_backtest.py:", len(content_run_real_backtest_py), "bytes")

content_requirements_txt = r'''# Core pipeline (config.py, signals/, evaluation/, recommendation/, demo.py).
pandas>=2.0
numpy>=1.24

# Storage layer (ingestion/data_store.py) — pyarrow is required for
# df.to_parquet()/pd.read_parquet(); without it, write_raw_parquet() fails
# with "Unable to find a usable engine" even though duckdb itself works.
duckdb>=1.0
pyarrow>=15.0

# Real data ingestion (ingestion/alpaca_ingest.py) — only needed on the
# machine that actually pulls from Alpaca, not to run demo.py's synthetic
# smoke test.
alpaca-py>=0.30

# Analyst-consensus ingestion (ingestion/finnhub_ingest.py) — feeds
# recommendation/recommend.py's "revision" pillar. Same "only needed on
# the machine that actually pulls live data" caveat as alpaca-py above.
requests>=2.31

# Testing (tests/).
pytest>=8.0
'''
Path("requirements.txt").write_text(content_requirements_txt)
print("wrote requirements.txt:", len(content_requirements_txt), "bytes")

content_tests_test_finnhub_ingest_py = r'''"""
Level 1 tests for ingestion/finnhub_ingest.py's pure scoring logic
(_consensus_to_score). No network calls here — fetch_recommendation_trends()
itself is a thin requests.get() wrapper not worth mocking extensively;
the real value to test is "does a raw Finnhub response map to the right
0-100 number."
"""
from __future__ import annotations

from ingestion import finnhub_ingest


def test_consensus_to_score_all_strong_buy_is_100():
    period = {"strongBuy": 10, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
    assert finnhub_ingest._consensus_to_score(period) == 100.0


def test_consensus_to_score_all_strong_sell_is_zero():
    period = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 10}
    assert finnhub_ingest._consensus_to_score(period) == 0.0


def test_consensus_to_score_all_hold_is_fifty():
    period = {"strongBuy": 0, "buy": 0, "hold": 7, "sell": 0, "strongSell": 0}
    assert finnhub_ingest._consensus_to_score(period) == 50.0


def test_consensus_to_score_is_weighted_by_analyst_count():
    # 24 buy + 7 hold + 0 sell, lopsided toward buy -- should land well above 50.
    period = {"strongBuy": 13, "buy": 24, "hold": 7, "sell": 0, "strongSell": 0}
    score = finnhub_ingest._consensus_to_score(period)
    assert 75 < score < 100


def test_consensus_to_score_no_coverage_returns_none():
    period = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
    assert finnhub_ingest._consensus_to_score(period) is None


def test_consensus_to_score_missing_keys_treated_as_zero():
    # Finnhub's real payloads always include all five keys, but don't
    # assume that forever -- a partial dict shouldn't raise.
    period = {"buy": 5, "hold": 5}
    score = finnhub_ingest._consensus_to_score(period)
    assert score == 62.5  # (5*75 + 5*50) / 10
'''
Path("tests/test_finnhub_ingest.py").write_text(content_tests_test_finnhub_ingest_py)
print("wrote tests/test_finnhub_ingest.py:", len(content_tests_test_finnhub_ingest_py), "bytes")

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
'''
Path("tests/test_data_store.py").write_text(content_tests_test_data_store_py)
print("wrote tests/test_data_store.py:", len(content_tests_test_data_store_py), "bytes")

content_tests_test_recommendation_py = r'''"""
Level 1 tests for recommendation/recommend.py — the Action/Score/
Confidence/Reasoning output the user explicitly asked this project to
produce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommendation import recommend


@pytest.mark.parametrize(
    "score,expected_action",
    [
        (100, "STRONG BUY"),
        (85, "STRONG BUY"),
        (84.9, "BUY"),
        (70, "BUY"),
        (69.9, "HOLD"),
        (40, "HOLD"),
        (39.9, "SELL"),
        (25, "SELL"),
        (24.9, "STRONG SELL"),
        (0, "STRONG SELL"),
    ],
)
def test_action_thresholds(score, expected_action):
    assert recommend._action_for_score(score) == expected_action


def _neutral_row():
    # No bullish or bearish condition fires -> tilt = 0 -> technical score
    # should land exactly on the 50 (neutral) midpoint.
    return pd.Series(
        {
            "timestamp": pd.Timestamp("2024-06-01"),
            "close": 123.45,
            "trend_template_bull": False,
            "trend_template_bear": False,
            "rs_percentile": 0.5,
            "volume_z": 0.0,
            "vol_contraction_ratio": 1.0,
            "new_52w_high": False,
            "new_52w_low": False,
            "roc_acceleration": 0.0,
        }
    )


def _bullish_row():
    row = _neutral_row()
    row["trend_template_bull"] = True
    row["rs_percentile"] = 0.9
    row["volume_z"] = 2.0
    row["vol_contraction_ratio"] = 0.5
    row["new_52w_high"] = True
    row["roc_acceleration"] = 0.02
    return row


def test_recommend_with_only_technical_pillar_uses_it_unweighted():
    row = _bullish_row()
    rec = recommend.recommend(row, ticker="TEST")

    assert rec.pillars_used == ["technical"]
    # With only one pillar supplied, the renormalized weighted average is
    # just that pillar's own score.
    tech_score, _ = recommend._technical_pillar_score(row)
    assert rec.composite_score == pytest.approx(tech_score)
    assert rec.action == recommend._action_for_score(tech_score)
    assert any("Pillars not yet available" in r for r in rec.reasoning)


def test_recommend_neutral_row_scores_fifty_and_holds():
    row = _neutral_row()
    rec = recommend.recommend(row, ticker="TEST")
    assert rec.composite_score == pytest.approx(50.0)
    assert rec.action == "HOLD"


def test_recommend_renormalizes_weights_across_supplied_pillars():
    row = _neutral_row()  # technical score fixed at 50
    rec = recommend.recommend(row, ticker="TEST", fundamental_score=80.0)

    w_tech = recommend.PILLAR_WEIGHTS["technical"]
    w_fund = recommend.PILLAR_WEIGHTS["fundamental"]
    expected = 50.0 * (w_tech / (w_tech + w_fund)) + 80.0 * (w_fund / (w_tech + w_fund))

    assert set(rec.pillars_used) == {"technical", "fundamental"}
    assert rec.composite_score == pytest.approx(expected)
    # Only revision/alternative are still missing now.
    missing_note = [r for r in rec.reasoning if "Pillars not yet available" in r][0]
    assert "revision" in missing_note and "alternative" in missing_note
    assert "fundamental" not in missing_note.split(":")[1]


def test_recommend_with_revision_score_folds_it_into_composite():
    row = _neutral_row()  # technical score fixed at 50
    rec = recommend.recommend(row, ticker="TEST", revision_score=90.0)

    w_tech = recommend.PILLAR_WEIGHTS["technical"]
    w_rev = recommend.PILLAR_WEIGHTS["revision"]
    expected = 50.0 * (w_tech / (w_tech + w_rev)) + 90.0 * (w_rev / (w_tech + w_rev))

    assert set(rec.pillars_used) == {"technical", "revision"}
    assert rec.composite_score == pytest.approx(expected)
    missing_note = [r for r in rec.reasoning if "Pillars not yet available" in r][0]
    assert "fundamental" in missing_note and "alternative" in missing_note
    assert "revision" not in missing_note.split(":")[1]


def test_recommend_without_calibration_table_reports_uncalibrated():
    row = _bullish_row()
    rec = recommend.recommend(row, ticker="TEST")
    assert rec.confidence_pct is None
    assert rec.confidence_range is None
    assert "no calibration table supplied" in rec.confidence_note


def test_recommend_with_calibration_table_looks_up_confidence():
    row = _bullish_row()  # composite >= 50 -> "bull" direction lookup
    calibration = pd.DataFrame(
        [
            {"direction": "bull", "score_bucket": 0, "n": 20, "hit_rate": 0.75, "ci_low": 0.5, "ci_high": 0.9},
            {"direction": "bear", "score_bucket": 0, "n": 5, "hit_rate": 0.4, "ci_low": 0.1, "ci_high": 0.7},
        ]
    )
    rec = recommend.recommend(row, ticker="TEST", calibration_table=calibration)
    assert rec.confidence_pct == pytest.approx(75.0)
    assert rec.confidence_range == (0.5, 0.9)


def test_recommend_reports_missing_direction_in_calibration_table():
    row = _neutral_row()
    row["trend_template_bear"] = True  # tips composite below 50 -> "bear" lookup
    calibration = pd.DataFrame(
        [{"direction": "bull", "score_bucket": 0, "n": 20, "hit_rate": 0.75, "ci_low": 0.5, "ci_high": 0.9}]
    )
    rec = recommend.recommend(row, ticker="TEST", calibration_table=calibration)
    assert rec.confidence_pct is None
    assert "no backtested bear signals" in rec.confidence_note
'''
Path("tests/test_recommendation.py").write_text(content_tests_test_recommendation_py)
print("wrote tests/test_recommendation.py:", len(content_tests_test_recommendation_py), "bytes")