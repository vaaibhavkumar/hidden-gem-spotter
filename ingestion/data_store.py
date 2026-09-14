"""
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
