"""
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
