-- Hidden Gem Spotter — DuckDB schema.
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
