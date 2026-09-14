# Hidden Gem Spotter — prototype scoring engine

Companion code to the project doc `early-momentum-detection-system-proposal.md`
(sections 2-3 and 6). This is a working prototype of the composite bullish/
bearish screening logic, smoke-tested on synthetic data — **not yet running
on real prices**, because of a network constraint explained below.

## What's here

- `config.py` — the 15-ticker validation universe (5 risers / 5 fallers / 5
  normal controls, section 6 of the proposal), benchmark, feature windows,
  and scoring thresholds.
- `features.py` — the technical feature pipeline (section 3.3): moving
  averages, 52-week high/low, volatility contraction, volume z-score,
  rate-of-change acceleration, relative strength vs. benchmark.
- `scoring.py` — the composite bullish/bearish scoring engine (sections 2E,
  2F, 3.4), including cross-sectional RS-percentile ranking, a warm-up
  mask, and a debounce filter so a noisy score doesn't fire duplicate
  "fresh" signals.
- `backtest.py` — evaluates each fired signal's forward return and rolls
  results up by role (riser/faller/normal) — the false-positive-rate check
  section 3.5 and 6 call for.
- `synthetic_data.py` + `demo.py` — generates a fake base-then-breakout
  series, a fake topping-then-breakdown series, and a fake random-walk
  "normal" series, and runs the full pipeline end-to-end on them. This is
  a smoke test, proving the mechanics work — **not a real backtest**.
- `data_sources/alpaca_ingest.py` — ready-to-run script that pulls real
  hourly bars for the section-6 universe from Alpaca. **Run this on your
  own computer**, not inside this Claude session (see below).
- `run_real_backtest.py` — same pipeline as `demo.py`, but reads real data
  from `./data/*.csv` once you've run the ingestion script.

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
   full, clean 3-year analysis window *after* burn-in.** `alpaca_ingest.py`
   already does this.
2. **A hard score threshold chatters.** A composite score that hovers near
   its cutoff re-fires a "fresh" signal every time it wiggles back across
   the line, producing dozens of near-duplicate alerts around one real
   event. Fixed with a debounce/cooldown (default 20 bars) — see
   `scoring._debounce`.

## How to get real data flowing

1. On your own computer (not in this Claude session): `pip install alpaca-py`
2. Get free API keys from your Alpaca dashboard (Overview → API Keys in the
   left sidebar) — the paper-trading keys are enough for market data, no
   funding required.
3. Set them as environment variables (never paste keys into a chat):
   `export ALPACA_API_KEY=...` / `export ALPACA_SECRET_KEY=...`
4. Run `python3 data_sources/alpaca_ingest.py` — it writes one CSV per
   ticker to `./data/`.
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
