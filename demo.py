"""
End-to-end smoke test: synthetic riser / faller / normal series ->
features -> cross-sectional RS percentile -> composite scores -> signals
-> backtest evaluation.

Run: python3 demo.py

This proves the mechanics of the framework (sections 2-3 of the proposal)
are wired correctly, using clearly-labeled SYNTHETIC data — not a real
backtest. Swap synthetic_data.* for real OHLCV frames (config.py has the
target ticker list) once the data-access path is resolved.
"""
from __future__ import annotations

import pandas as pd

import backtest
import config
import features
import scoring
import synthetic_data as synth


def main() -> None:
    pd.set_option("display.width", 120)

    benchmark = synth.make_benchmark()
    series = {
        "SYN_RISER": synth.make_riser(),
        "SYN_FALLER": synth.make_faller(),
        "SYN_NORMAL": synth.make_normal(),
    }
    roles = {"SYN_RISER": "riser", "SYN_FALLER": "faller", "SYN_NORMAL": "normal"}

    feats = {t: features.compute_features(df, benchmark["close"]) for t, df in series.items()}
    scoring.attach_rs_percentile(feats)  # cross-sectional RS rank, no look-ahead

    signaled = {t: scoring.generate_signals(df) for t, df in feats.items()}

    print("=== Fresh signal counts ===")
    for t, df in signaled.items():
        print(f"{t:12s} bull_signals={int(df['bull_signal'].sum()):3d}  bear_signals={int(df['bear_signal'].sum()):3d}")

    print("\n=== First bull signal on the riser (should appear near its breakout) ===")
    riser = signaled["SYN_RISER"]
    first_bull = riser[riser["bull_signal"]].head(1)
    if not first_bull.empty:
        idx = first_bull.index[0]
        print(f"Fired at bar {idx} ({riser.loc[idx, 'timestamp'].date()}), price={riser.loc[idx, 'close']:.2f}")
        print(f"(base ran through roughly bar {int(len(riser) * 0.4)} in this synthetic series)")
    else:
        print("No bull signal fired — thresholds may need loosening for this synthetic pattern.")

    print("\n=== First bear signal on the faller (should appear near its top) ===")
    faller = signaled["SYN_FALLER"]
    first_bear = faller[faller["bear_signal"]].head(1)
    if not first_bear.empty:
        idx = first_bear.index[0]
        print(f"Fired at bar {idx} ({faller.loc[idx, 'timestamp'].date()}), price={faller.loc[idx, 'close']:.2f}")
        print(f"(top formed around roughly bar {int(len(faller) * 0.35)} in this synthetic series)")
    else:
        print("No bear signal fired — thresholds may need loosening for this synthetic pattern.")

    print("\n=== Backtest evaluation (60-bar forward return per fired signal) ===")
    results = {t: backtest.evaluate_signals(df, horizon=60) for t, df in signaled.items()}
    summary = backtest.summarize_universe(results, roles)
    print(summary.to_string(index=False) if not summary.empty else "(no signals to evaluate)")


if __name__ == "__main__":
    main()
