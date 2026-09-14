"""
Backtest harness (section 3.5 of the proposal).

evaluate_signals() replays a ticker's signal history and measures, for
every fresh bull_signal / bear_signal, the forward return over a chosen
horizon — this is the "would it have worked" check.

summarize_universe() rolls that up across every ticker in the validation
universe (section 6) and reports counts by role (riser / faller / normal),
which is exactly the false-positive-rate check the proposal calls for:
a well-tuned screen should rarely fire on the "normal" control group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_signals(df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """
    df must already have bull_signal / bear_signal columns (see scoring.py)
    and a `close` column. Returns one row per fired signal with the
    forward return over `horizon` bars.
    """
    close = df["close"]
    fwd_return = close.shift(-horizon) / close - 1.0

    rows = []
    for flag_col, direction in (("bull_signal", "bull"), ("bear_signal", "bear")):
        fired = df.index[df[flag_col].fillna(False)]
        for i in fired:
            if i + horizon >= len(df):
                continue  # not enough future data yet
            rows.append(
                {
                    "timestamp": df.loc[i, "timestamp"],
                    "direction": direction,
                    "price_at_signal": close.loc[i],
                    f"fwd_return_{horizon}b": fwd_return.loc[i],
                    "correct": (fwd_return.loc[i] > 0) if direction == "bull" else (fwd_return.loc[i] < 0),
                }
            )
    return pd.DataFrame(rows)


def summarize_universe(signal_results_by_ticker: dict[str, pd.DataFrame], roles: dict[str, str]) -> pd.DataFrame:
    """
    signal_results_by_ticker: {ticker: evaluate_signals() output}
    roles: {ticker: "riser"|"faller"|"normal"} from config.VALIDATION_UNIVERSE
    Returns a per-role summary: signal count, hit rate, mean forward return.
    """
    records = []
    for ticker, res in signal_results_by_ticker.items():
        if res.empty:
            continue
        fwd_col = [c for c in res.columns if c.startswith("fwd_return_")][0]
        for direction, group in res.groupby("direction"):
            records.append(
                {
                    "ticker": ticker,
                    "role": roles.get(ticker, "unknown"),
                    "direction": direction,
                    "n_signals": len(group),
                    "hit_rate": group["correct"].mean(),
                    "mean_fwd_return": group[fwd_col].mean(),
                }
            )
    summary = pd.DataFrame(records)
    if summary.empty:
        return summary
    return (
        summary.groupby(["role", "direction"])
        .agg(n_signals=("n_signals", "sum"), avg_hit_rate=("hit_rate", "mean"), avg_fwd_return=("mean_fwd_return", "mean"))
        .reset_index()
    )
