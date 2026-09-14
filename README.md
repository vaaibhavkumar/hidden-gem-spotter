# Hidden Gem Spotter — prototype scoring engine

Companion code to the project doc `early-momentum-detection-system-proposal.md`
(sections 2-3 and 6). This is a working prototype of the composite bullish/
bearish screening logic, smoke-tested on synthetic data — **not yet running
on real prices**, because of a network constraint explained below.

## What's here

The code is organized into topical packages so a change to one stage
(say, scoring thresholds) can't silently break another (say, storage) —
and so the test suite in `tests/` can pin each stage down independently.

```
hidden_gem_spotter/
    config.py               entry-point config: universe, windows, thresholds
    demo.py                 end-to-end smoke test on synthetic data
    run_real_backtest.py    end-to-end run on real data (once ingested)
    synthetic_data.py       fake riser/faller/normal series for demo.py & tests
    conftest.py             pytest bootstrap (see "Running the tests")
    pytest.ini
    requirements.txt

    ingestion/              getting price data in and stored
        alpaca_ingest.py    pulls real hourly bars from Alpaca
        data_store.py       Parquet + DuckDB storage layer

    signals/                turning prices into technical signals
        features.py         moving averages, 52w high/low, RS, volume z, etc.
        scoring.py          composite bullish/bearish scoring, debounce, warmup

    evaluation/             did the signals actually work
        backtest.py         forward-return evaluation, Wilson-CI calibration

    recommendation/         the final output
        recommend.py        Action / Score / Confidence / Reasoning

    db/
        schema.sql           the DuckDB schema, versioned in git (see below)

    tests/                  see "Running the tests" below
```

- `config.py` — the 15-ticker validation universe (5 risers / 5 fallers / 5
  normal controls, section 6 of the proposal), benchmark, feature windows,
  and scoring thresholds.
- `signals/features.py` — the technical feature pipeline (section 3.3):
  moving averages, 52-week high/low, volatility contraction, volume
  z-score, rate-of-change acceleration, relative strength vs. benchmark.
- `signals/scoring.py` — the composite bullish/bearish scoring engine
  (sections 2E, 2F, 3.4), including cross-sectional RS-percentile ranking,
  a warm-up mask, and a debounce filter so a noisy score doesn't fire
  duplicate "fresh" signals.
- `evaluation/backtest.py` — evaluates each fired signal's forward return
  and rolls results up by role (riser/faller/normal) — the
  false-positive-rate check section 3.5 and 6 call for.
- `synthetic_data.py` + `demo.py` — generates a fake base-then-breakout
  series, a fake topping-then-breakdown series, and a fake random-walk
  "normal" series, and runs the full pipeline end-to-end on them. This is
  a smoke test, proving the mechanics work — **not a real backtest**.
- `ingestion/data_store.py` — the storage layer: Parquet landing files plus
  a local DuckDB database (`data/market.duckdb`) with `bars` and `signals`
  tables, built from `db/schema.sql`. See "Where the data lives" below for
  why this instead of a pile of CSVs.
- `recommendation/recommend.py` — turns a scored row into the Action /
  Score / Confidence / Reasoning output you actually want to read (STRONG
  BUY..STRONG SELL, a 0-100 composite, a confidence figure, and
  plain-language reasoning).
- `ingestion/alpaca_ingest.py` — ready-to-run script that pulls real
  hourly bars for the section-6 universe from Alpaca. **Run this on your
  own computer**, not inside this Claude session (see below).
- `run_real_backtest.py` — same pipeline as `demo.py`, but reads real data
  from `ingestion.data_store` (falls back to `./data/*.csv` for the very
  first prototype run) once you've run the ingestion script.

## The database schema is versioned in git

`db/schema.sql` is the single source of truth for `data/market.duckdb`'s
structure — the `bars` and `signals` tables, with their column types and
primary keys, as plain SQL DDL checked into this repo. `ingestion/
data_store.py._connect()` reads and applies this file on every connection
(every statement is `CREATE ... IF NOT EXISTS`, so that's always safe).

This is deliberately a plain `.sql` file instead of a Docker/Kubernetes
setup with a running database service. A local DuckDB file has no server
to containerize — "the database" here is just `data/market.duckdb` plus
this schema file, and putting the schema in git already gives you the
important thing (a reviewable history of every structural change) without
the operational overhead of a container orchestrator for a single-user
prototype. If this ever grows into a shared, always-on service, that's
the point to reach for Docker/Kubernetes — not before.

## Running the tests

```
pip install -r requirements.txt
python3 -m pytest
```

`tests/` holds **Level 1** tests: fast, pure-Python unit tests against
synthetic data and hand-built rows, with no network access and no real
market data required — they run the same way on any machine, including
inside this Claude session. They cover:

- `test_config.py` — the daily/hourly bar-scaling math (`set_bars_per_day`).
- `test_features.py` — the technical feature pipeline, using the
  synthetic riser/faller series to check trend-template and 52-week-high
  logic actually fires where it should.
- `test_scoring.py` — condition scoring, cross-sectional RS-percentile
  ranking, and **explicit regression tests for the two real bugs** the
  synthetic smoke test caught (warm-up masking and signal debounce — see
  "Why real data isn't wired up yet" below). If either bug comes back,
  one of these tests fails.
- `test_evaluation.py` — the Wilson confidence interval formula and the
  backtest/calibration bucketing logic, including the thin-data edge case
  (all fired signals sharing one score).
- `test_recommendation.py` — action thresholds, pillar-weight
  renormalization when only some pillars are supplied, and confidence
  reporting with and without a calibration table.
- `test_data_store.py` — the Parquet/DuckDB storage layer round-trips
  correctly, against the real `db/schema.sql` but an isolated tmp
  directory (never touching your real `data/` folder).

`conftest.py` at the repo root puts the repo root on `sys.path` so the
package-style imports (`from signals import scoring`, etc.) resolve no
matter which directory you run `pytest` from, and resets
`config.BARS_PER_DAY` before/after every test so one test's hourly-bar
setup can never leak into another test's daily-bar assumptions.

**What's not here yet ("Level 2"/"Level 3"):** tests against recorded/
mocked real Alpaca API responses, and an end-to-end golden-snapshot
regression test on real data — both need real data flowing first (see
"How to get real data flowing" below), so they're the natural next layer
to add once `ingestion/alpaca_ingest.py` has actually been run.

## Where the data lives

`data/raw/<TICKER>.parquet` is the landing zone — exactly what Alpaca
returned, untouched, kept for reproducibility. `data/market.duckdb` is the
queryable store: a `bars` table (ticker, timestamp, OHLCV) and a `signals`
table that persists every fired buy/sell signal over time, so "signal
history" is an actual queryable log, not just recomputed each run.

This is Parquet (an open, columnar file format) queried through DuckDB (an
embedded, serverless SQL engine — no server process, just a library and a
file) rather than a real "table format" like Delta Lake or Apache Iceberg.
Those add a transaction log and catalog on top of Parquet for concurrent
multi-writer access, schema evolution, and time travel — genuinely useful
for a shared data lake with several pipelines writing at once, overkill
for one person's local research project. DuckDB comfortably handles the
full ~500-ticker x hourly x multi-year universe (tens of millions of rows)
on a laptop; if this ever becomes a team tool backed by cloud storage,
upgrading these Parquet files to an Iceberg/Delta table is the natural
next step, not something to build prematurely now.

`data/` (Parquet + DuckDB files) is git-ignored — it's regenerated by
re-running the ingestion script, not something to version like code.

## Recommendation output

`recommend.recommend(row, ticker, ...)` returns a `Recommendation` with:
STRONG BUY/BUY/HOLD/SELL/STRONG SELL, a 0-100 composite score, a
confidence figure, and reasoning bullets. Two things worth knowing before
trusting it:

1. **Only the technical pillar has real data behind it right now.** The
   composite is designed to blend technical (35%), fundamental (30%),
   analyst-revision (20%), and alternative (15%) pillars (matching the
   proposal's sections 2A-2D), but only the technical one is implemented.
   The output always lists which pillars were actually used, and a
   technical-only "STRONG BUY" is a weaker claim than one where all four
   agree — don't read past that caveat.
2. **Confidence is a real Wilson confidence interval on backtested hit
   rates (`backtest.calibrate_confidence`), not a measure of how well
   today's sub-scores agree with each other.** Without a calibration
   table built from a real backtest, `recommend()` reports confidence as
   "uncalibrated" rather than inventing a precise-looking number — a
   tempting shortcut (score dispersion across pillars) measures internal
   consistency, not whether signals like this one actually worked
   historically, and the 15-ticker validation universe here is nowhere
   near enough to calibrate reliably on its own (see `demo.py`'s output
   for what that thin-sample warning looks like in practice).

## What we borrowed from an alternative (Gemini) design, and what we didn't

You also ran this idea past Gemini, which came back with a similarly-shaped
multi-factor system. Worth stealing some of it rather than reinventing:

**Adopted:**
- The output schema — Action (STRONG BUY..STRONG SELL) + a 0-100 composite
  + confidence + reasoning bullets — is a cleaner deliverable than raw
  condition counts. `recommendation/recommend.py` implements this.
- Weighted percentile-rank pillars (35/30/20/15 for
  technical/fundamental/revision/alternative) instead of a flat 0-6
  condition count — more standard and comparable across factors of
  different scales. Used as `recommend.PILLAR_WEIGHTS`.
- "Earnings Quality Drift" (cash-flow growth leading net-income growth —
  the classic Sloan-1996 accruals anomaly) is a legitimate, well-studied
  fundamental signal we hadn't listed; worth adding to section 2C's
  fundamental overlay once that pillar is built.
- The sequential-gate framing (fundamental gate -> technical gate ->
  alternative gate -> shortlist) is a reasonable alternative architecture
  to a pure weighted composite — cheap/high-confidence filters (basic
  quality, liquidity) could eliminate names before scoring the survivors,
  which is more efficient at 500-name scale. Worth revisiting once the
  fundamental pillar exists.

**Deliberately not adopted:**
- **Its "confidence interval" isn't one.** It's `100 - 1.5*std_dev` across
  four sub-scores — that measures how much your own pillars agree with
  each other, not whether signals like this one actually worked
  historically. Presented as "86.2% (80.2%-92.2%)" it reads as far more
  rigorous than it is (false precision). `recommendation/recommend.py`
  instead reports confidence only when a real backtest-calibrated Wilson
  interval is available (`evaluation.backtest.calibrate_confidence`), and
  says "uncalibrated" otherwise rather than inventing a number.
- **Heavy alt-data scraping (Reddit/WSB via `praw`, Glassdoor) is
  deprioritized for now.** WSB sentiment is noisy and more relevant to
  small/meme caps than S&P 500 mega-caps; Glassdoor scraping raises ToS
  questions and moves too slowly (quarterly-ish) to matter for "early
  stage." Matches this project's phased build order (README/proposal
  section 3.7): validate technical, then fundamentals, then alt-data —
  not all four gates on day one.
- **`yfinance`/`.info` fields as the fundamental data source.** Its
  `returnOnEquity` is being used as an ROIC proxy, which conflates two
  meaningfully different metrics (ROE is levered/buyback-sensitive; ROIC
  is capital-structure-neutral) — real ROIC needs NOPAT/invested capital
  from the actual financial statements (SEC EDGAR/XBRL), not a shortcut.
  Also, Yahoo's endpoints are blocked from this cloud workspace anyway
  (see "Why real data isn't wired up yet" above).

## Keeping git history in sync (no more manual zips)

Early on, the only way to get code changes onto your machine was a full
zip re-upload, because your device's local shell was broken by a Windows
update. Going forward: Claude writes changed files directly into your
connected `Projects/hidden-gem-spotter/hidden_gem_spotter/` folder (no zip
needed), and you commit them from VS Code's Source Control panel (or
`git add -A && git commit`) whenever you see new changes — that keeps one
real git history growing on your machine instead of two copies drifting
apart. Once your device's shell is reachable again, Claude can run `git
commit` there directly and this manual step goes away too.

## Why real data isn't wired up yet

This Claude session runs in a cloud workspace whose network policy only
allows connections to package registries (pypi, npm, etc.). Every market
data host tested from here — Yahoo Finance's API, Alpaca, Polygon, EODHD,
Twelve Data, Alpha Vantage, and stooq.com — returned a policy-denied 403.
This is an org-level egress restriction, not a bug to retry around.

**What worked in the smoke test:** synthetic data generated locally
(no network needed) ran through the full pipeline correctly, and running
it surfaced two real, worth-knowing findings:

1. **Burn-in matters.** The 200-period moving average and 252-bar 52-week
   high/low need that much history before they mean anything — the first
   run's "signals" during the synthetic base turned out to be an artifact
   of an incomplete rolling window, not a real breakout. Fixed by masking
   the first `LOOKBACK_52W + SMA_LONG` bars (~1.8 years of daily-equivalent
   bars). **Practical implication for section 3.1's "3-year rolling
   window": pull more like 4.5 years of raw history so you still have a
   full, clean 3-year analysis window *after* burn-in.**
   `ingestion/alpaca_ingest.py` already does this.
2. **A hard score threshold chatters.** A composite score that hovers near
   its cutoff re-fires a "fresh" signal every time it wiggles back across
   the line, producing dozens of near-duplicate alerts around one real
   event. Fixed with a debounce/cooldown (default 20 bars) — see
   `signals/scoring._debounce`. (Both of these bugs now have dedicated
   regression tests — see "Running the tests" above.)

## How to get real data flowing

1. On your own computer (not in this Claude session):
   `pip install -r requirements.txt`
2. Get free API keys from your Alpaca dashboard (Overview → API Keys in the
   left sidebar) — the paper-trading keys are enough for market data, no
   funding required.
3. Set them as environment variables (never paste keys into a chat):
   `export ALPACA_API_KEY=...` / `export ALPACA_SECRET_KEY=...`
4. From the repo root, run `python3 -m ingestion.alpaca_ingest` — it
   writes Parquet + a DuckDB database to `./data/` (see "Where the data
   lives" below).
5. Bring the `data/` folder back into this Claude session (attach the
   files, or through the connected-folder bridge once it's working again —
   it's currently blocked by an unrelated, tracked Windows update issue)
   and run `python3 run_real_backtest.py`, or hand the files to Claude to
   run it for you.

## Known open items (not yet built)

- The RS-percentile ranking here is only correct across whatever tickers
  you feed it — the real system needs the full ~500-name point-in-time
  S&P 500 universe (section 3.1) for a meaningful percentile.
- The fundamental overlay (section 2C: EPS revisions, guidance direction,
  earnings surprises) isn't implemented yet — the case study in section 5
  suggests it's exactly what would have prevented the 2024 MU whipsaw, so
  it matters for real precision, not just as a nice-to-have.
- Thresholds in `config.THRESHOLDS` are untuned starting points — tune
  them against the real backtest (section 3.5), not by eye.
