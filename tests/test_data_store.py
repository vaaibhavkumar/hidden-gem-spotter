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
