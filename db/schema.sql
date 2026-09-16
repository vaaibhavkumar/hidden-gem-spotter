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
