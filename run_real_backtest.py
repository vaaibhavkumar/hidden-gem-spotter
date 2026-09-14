"""
Same pipeline as demo.py, but reading real OHLCV data from ./data/*.csv
(produced by data_sources/alpaca_ingest.py) instead of synthetic series.

Run: python3 run_real_backtest.py

Expects:
    data/SPY.csv                 (benchmark)
    data/<TICKER>.csv             for every ticker in config.VALIDATION_UNIVERSE
each with columns [timestamp, open, high, low, close, volume].
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import backtest
import config
import features
import scoring

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_ticker(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run data_sources/alpaca_ingest.py (on a machine with "
            "internet access) and bring its data/ folder here first."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    pd.set_option("display.width", 120)

    benchmark = load_ticker(config.BENCHMARK)

    series = {t: load_ticker(t) for t in config.VALIDATION_UNIVERSE}
    roles = {t: meta["role"] for t, meta in config.VALIDATION_UNIVERSE.items()}

    # Align every ticker's frame to the benchmark's timestamps (inner join)
    # so the cross-sectional RS-percentile ranking in scoring.py compares
    # apples to apples at each bar.
    aligned = {}
    for t, df in series.items():
        merged = pd.merge(df, benchmark[["timestamp", "close"]], on="timestamp", suffixes=("", "_bench"))
        aligned[t] = merged.drop(columns=["close_bench"]).reset_index(drop=True)
        aligned[t]["_bench_close"] = merged["close_bench"].reset_index(drop=True)

    feats = {t: features.compute_features(df, df["_bench_close"]) for t, df in aligned.items()}
    scoring.attach_rs_percentile(feats)

    signaled = {t: scoring.generate_signals(df) for t, df in feats.items()}

    print("=== Fresh signal counts by ticker ===")
    for t, df in signaled.items():
        role = roles[t]
        print(
            f"{t:6s} ({role:6s}) bull_signals={int(df['bull_signal'].sum()):3d}  "
            f"bear_signals={int(df['bear_signal'].sum()):3d}"
        )

    print("\n=== Signal dates (first 5 of each type per ticker) ===")
    for t, df in signaled.items():
        bulls = df.loc[df["bull_signal"], ["timestamp", "close"]].head(5)
        bears = df.loc[df["bear_signal"], ["timestamp", "close"]].head(5)
        if not bulls.empty:
            print(f"{t} BUY signals:\n{bulls.to_string(index=False)}")
        if not bears.empty:
            print(f"{t} SELL signals:\n{bears.to_string(index=False)}")

    print("\n=== Backtest summary by role (this is the false-positive-rate check) ===")
    horizon = 24 * 5  # ~5 trading days ahead, in hourly bars — adjust to taste
    results = {t: backtest.evaluate_signals(df, horizon=horizon) for t, df in signaled.items()}
    summary = backtest.summarize_universe(results, roles)
    print(summary.to_string(index=False) if not summary.empty else "(no signals to evaluate)")


if __name__ == "__main__":
    main()
