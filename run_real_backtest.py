"""
Same pipeline as demo.py, but reading real OHLCV data (produced by
ingestion/alpaca_ingest.py) instead of synthetic series.

Run: python3 run_real_backtest.py

Reads from data/market.duckdb (see ingestion/data_store.py) if present —
this is the path alpaca_ingest.py now writes to. Falls back to
data/<TICKER>.csv for backward compatibility with the very first
prototype run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from evaluation import backtest
from ingestion import data_store
from recommendation import recommend
from signals import features, scoring

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_ticker(ticker: str) -> pd.DataFrame:
    df = data_store.load_bars(ticker)
    if not df.empty:
        return df.sort_values("timestamp").reset_index(drop=True)

    csv_path = DATA_DIR / f"{ticker}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    raise FileNotFoundError(
        f"No data for {ticker} in {data_store.DB_PATH} or {csv_path} — run "
        "python3 -m ingestion.alpaca_ingest (on a machine with internet "
        "access) and bring its data/ folder here first."
    )


def detect_bars_per_day(df: pd.DataFrame) -> int:
    """
    Infers whether this data holds daily or (regular-session) hourly bars
    from the median gap between consecutive timestamps, and returns the
    BARS_PER_DAY value config.py's window sizes should use. This exists so
    a daily-vs-hourly mismatch (see config.py's BARS_PER_DAY comment)
    can't silently produce a 7x-too-short "50-day" moving average —
    it's detected and applied automatically instead.
    """
    deltas = df["timestamp"].diff().dropna()
    if deltas.empty:
        return config.BARS_PER_DAY
    median_hours = deltas.median().total_seconds() / 3600
    if median_hours >= 20:   # ~1 trading day between bars (weekends inflate the mean, not the median)
        return 1
    return 7                 # hourly bars, ~7 per regular trading session


def main() -> None:
    pd.set_option("display.width", 120)

    benchmark = load_ticker(config.BENCHMARK)

    detected = detect_bars_per_day(benchmark)
    if detected != config.BARS_PER_DAY:
        print(
            f"Detected {'daily' if detected == 1 else 'hourly'} bars from {config.BENCHMARK}'s "
            f"timestamps — calling config.set_bars_per_day({detected}) so moving-average/52-week "
            f"windows are the right length (was {config.BARS_PER_DAY}). If this guess is wrong for "
            "your data, call config.set_bars_per_day(...) yourself before running the pipeline."
        )
        config.set_bars_per_day(detected)

    # Validation tickers keep their known riser/faller/normal label; the
    # broader calibration tickers have no such label (that's the point --
    # they're an unselected sample) and are excluded from the by-role
    # false-positive summary below, but included everywhere else.
    roles = {t: meta["role"] for t, meta in config.VALIDATION_UNIVERSE.items()}
    universe = config.all_tickers()
    series = {t: load_ticker(t) for t in universe}

    # Align every ticker's frame to the benchmark's timestamps (inner join)
    # so the cross-sectional RS-percentile ranking in scoring.py compares
    # apples to apples at each bar.
    aligned = {}
    for t, df in series.items():
        merged = pd.merge(df, benchmark[["timestamp", "close"]], on="timestamp", suffixes=("", "_bench"))
        aligned[t] = merged.drop(columns=["close_bench"]).reset_index(drop=True)
        aligned[t]["_bench_close"] = merged["close_bench"].reset_index(drop=True)

    feats = {t: features.compute_features(df, df["_bench_close"]) for t, df in aligned.items()}
    # Ranked across the FULL universe (validation + calibration), not just
    # the 15 hand-picked names.
    scoring.attach_rs_percentile(feats)

    signaled = {t: scoring.generate_signals(df) for t, df in feats.items()}

    print(f"=== Fresh signal counts ({len(roles)} validation tickers; "
          f"{len(universe) - len(roles)} calibration-only tickers omitted from this table) ===")
    for t, df in signaled.items():
        data_store.log_signals(t, df)  # persist to data/market.duckdb's `signals` table
        if t not in roles:
            continue
        role = roles[t]
        print(
            f"{t:6s} ({role:6s}) bull_signals={int(df['bull_signal'].sum()):3d}  "
            f"bear_signals={int(df['bear_signal'].sum()):3d}"
        )

    print("\n=== Signal dates (first 5 of each type per validation ticker) ===")
    for t, df in signaled.items():
        if t not in roles:
            continue
        bulls = df.loc[df["bull_signal"], ["timestamp", "close"]].head(5)
        bears = df.loc[df["bear_signal"], ["timestamp", "close"]].head(5)
        if not bulls.empty:
            print(f"{t} BUY signals:\n{bulls.to_string(index=False)}")
        if not bears.empty:
            print(f"{t} SELL signals:\n{bears.to_string(index=False)}")

    horizon = 5 * config.BARS_PER_DAY  # ~5 trading days ahead, in whatever bar size the data uses
    results = {t: backtest.evaluate_signals(df, horizon=horizon) for t, df in signaled.items()}

    print("\n=== Backtest summary by role (this is the false-positive-rate check -- ")
    print("    validation tickers only) ===")
    validation_results = {t: r for t, r in results.items() if t in roles}
    summary = backtest.summarize_universe(validation_results, roles)
    print(summary.to_string(index=False) if not summary.empty else "(no signals to evaluate)")

    print(f"\n=== Calibrated confidence by score bucket (validation + calibration tickers, "
          f"{len(universe)} total) ===")
    all_results = pd.concat([r for r in results.values() if not r.empty], ignore_index=True) if any(len(r) for r in results.values()) else pd.DataFrame()
    calibration = backtest.calibrate_confidence(all_results) if not all_results.empty else pd.DataFrame()
    print(calibration.to_string(index=False) if not calibration.empty else "(not enough signals to calibrate)")

    print("\n=== Most recent recommendation per validation ticker (Action / Score / Confidence / Reasoning) ===")
    for t, df in signaled.items():
        if t not in roles:
            continue
        latest = df.iloc[-1]
        rec = recommend.recommend(latest, ticker=t, calibration_table=calibration)
        print(rec)


if __name__ == "__main__":
    main()
