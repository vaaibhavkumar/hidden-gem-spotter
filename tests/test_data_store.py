"""
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
