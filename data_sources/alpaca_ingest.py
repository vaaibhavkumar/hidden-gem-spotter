"""
Real-data ingestion via Alpaca Market Data (section 3.1 of the proposal).

Run this on a machine with normal internet access — e.g. your own computer,
NOT inside the Claude cloud workspace, which sits behind an org network
policy that blocks Alpaca's API (tested 2026-09-14: data.alpaca.markets
returns a policy-denied 403 from that sandbox).

Setup (one time):
    pip install alpaca-py
    # Get free keys from https://app.alpaca.markets/dashboard/overview ->
    # "API Keys" in the left sidebar. Use the PAPER TRADING keys — market
    # data access doesn't require funding a live account.
    # Do NOT paste your keys into a chat with Claude or commit them to
    # source control. Set them as environment variables instead:
    export ALPACA_API_KEY="your_key_id"
    export ALPACA_SECRET_KEY="your_secret_key"

Run:
    python3 alpaca_ingest.py

Output:
    Writes one CSV per ticker to ./data/<TICKER>.csv with columns
    [timestamp, open, high, low, close, volume] — the exact schema
    features.compute_features() expects. Also writes ./data/SPY.csv for
    the benchmark.

Then bring the ./data/ folder back into your Claude session (attach the
files, or use the connected-folder bridge once available) so the actual
backtest (backtest.py / demo.py's pattern, applied to real data instead of
synthetic_data.py) can run against it.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make config.py (one directory up) importable when run from data_sources/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    print("Missing dependency. Run: pip install alpaca-py", file=sys.stderr)
    raise

# Pull extra history beyond the 3-year analysis window so the technical
# features (trend template, 52-week high/low, RS windows) have a full
# burn-in period before the window you actually care about starts — the
# prototype backtest on synthetic data showed the first ~1.8 years of any
# series are unusable "warm-up" for 200-period moving averages plus a
# 252-bar 52-week lookback. 4.5 years of raw history gives a clean 3-year
# analysis window after burn-in.
YEARS_OF_HISTORY = 4.5


def fetch_ticker(client: StockHistoricalDataClient, ticker: str, start, end):
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(1, TimeFrameUnit.Hour),
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)
    df = bars.df  # MultiIndex (symbol, timestamp) DataFrame
    if df.empty:
        return None
    df = df.xs(ticker, level="symbol").reset_index()
    df = df.rename(columns={"timestamp": "timestamp"})[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ]
    return df


def main() -> None:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables first "
            "(see the docstring at the top of this file).",
            file=sys.stderr,
        )
        sys.exit(1)

    client = StockHistoricalDataClient(api_key, secret_key)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(365 * YEARS_OF_HISTORY))

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    tickers = list(config.VALIDATION_UNIVERSE.keys()) + [config.BENCHMARK]
    for ticker in tickers:
        print(f"Fetching {ticker}...", end=" ", flush=True)
        try:
            df = fetch_ticker(client, ticker, start, end)
        except Exception as exc:  # noqa: BLE001 - surface any API error and keep going
            print(f"FAILED ({exc})")
            continue
        if df is None:
            print("no data returned")
            continue
        df.to_csv(out_dir / f"{ticker}.csv", index=False)
        print(f"{len(df)} bars -> data/{ticker}.csv")
        time.sleep(0.3)  # stay well under the free tier's 200 req/min limit

    print("\nDone. Bring the data/ folder back to your Claude session to run the real backtest.")


if __name__ == "__main__":
    main()
