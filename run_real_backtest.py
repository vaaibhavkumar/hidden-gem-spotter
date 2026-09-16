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
from recommendation import html_report, recommend
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


def build_signaled_universe() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    The shared "load real data -> align -> compute features -> rank
    RS-percentile across the full universe -> generate signals" pipeline,
    factored out of main() so other entry points (walk_forward_backtest.py)
    can build the exact same signaled dataset without duplicating this
    logic or risking it drifting out of sync with what production actually
    scores. Returns (signaled, roles): signaled is {ticker: DataFrame with
    bull_signal/bear_signal columns} for every ticker in config.all_tickers();
    roles is {ticker: "riser"/"faller"/"normal"} for the validation-only subset.
    """
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
    # broader calibration tickers have no such label (that's the point —
    # they're an unselected sample) and are excluded from the by-role
    # false-positive summary below, but included everywhere else (feature
    # computation, RS-percentile ranking, and confidence calibration) —
    # see config.all_tickers()'s docstring for why the two sets exist.
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
    # the 15 hand-picked names — this is the actual fix for "RS-percentile
    # ranking is only correct across whatever tickers you feed it" (see
    # README's "Known open items").
    scoring.attach_rs_percentile(feats)

    signaled = {t: scoring.generate_signals(df) for t, df in feats.items()}
    return signaled, roles


def main() -> None:
    pd.set_option("display.width", 120)

    signaled, roles = build_signaled_universe()
    universe = config.all_tickers()

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

    print("\n=== Backtest summary by role (this is the false-positive-rate check — ")
    print("    validation tickers only, since 'role' is only meaningful for the hand-picked set) ===")
    validation_results = {t: r for t, r in results.items() if t in roles}
    summary = backtest.summarize_universe(validation_results, roles)
    print(summary.to_string(index=False) if not summary.empty else "(no signals to evaluate)")

    print(f"\n=== Calibrated confidence by score bucket (validation + calibration tickers, "
          f"{len(universe)} total — still a proxy sample, see config.CALIBRATION_UNIVERSE's ")
    print("    point-in-time caveat, but no longer bucketed on just 15 correlated names) ===")
    all_results = pd.concat([r for r in results.values() if not r.empty], ignore_index=True) if any(len(r) for r in results.values()) else pd.DataFrame()
    calibration = backtest.calibrate_confidence(all_results) if not all_results.empty else pd.DataFrame()
    print(calibration.to_string(index=False) if not calibration.empty else "(not enough signals to calibrate)")

    print("\n=== Most recent recommendation per validation ticker (Action / Score / Confidence / Reasoning) ===")
    recommendations = []
    for t, df in signaled.items():
        if t not in roles:
            continue
        latest = df.iloc[-1]
        # None until ingestion/finnhub_ingest.py has been run for this
        # ticker -- recommend() already treats a None revision_score as
        # "pillar not available" (see its "Pillars not yet available"
        # reasoning note), so this is safe to call unconditionally.
        revision_score = data_store.load_latest_revision_score(t)
        rec = recommend.recommend(latest, ticker=t, revision_score=revision_score, calibration_table=calibration)
        print(rec)
        recommendations.append(rec)
        data_store.log_recommendation(rec)  # persist to data/market.duckdb's `recommendations` table

    # One recommendation history per validation ticker (oldest -> newest),
    # so report.html can show "how has this call changed over past runs",
    # not just today's snapshot.
    history = {t: data_store.load_recommendation_history(t) for t in roles}

    report_path = html_report.generate_html_report(recommendations, roles, history=history, out_path="report.html")
    print(f"\nWrote {report_path} -- open it in a browser to see the watchlist summary + per-ticker detail.")


if __name__ == "__main__":
    main()
