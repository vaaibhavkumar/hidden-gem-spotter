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

    score_col = {"bull_signal": "bull_score", "bear_signal": "bear_score"}

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
                    "score_at_signal": int(df.loc[i, score_col[flag_col]]) if score_col[flag_col] in df.columns else None,
                    "price_at_signal": close.loc[i],
                    f"fwd_return_{horizon}b": fwd_return.loc[i],
                    "correct": (fwd_return.loc[i] > 0) if direction == "bull" else (fwd_return.loc[i] < 0),
                }
            )
    return pd.DataFrame(rows)


def wilson_confidence_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial hit rate — a real, textbook
    confidence interval (default z=1.96 => ~95%), unlike a heuristic that
    just measures agreement across factor scores. Use this on backtested
    signal outcomes (see calibrate_confidence), not on the score itself.
    Returns (lower, upper) as fractions in [0, 1]. n=0 returns (0, 1) —
    "no data yet" rather than a false-precision point estimate.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    return ((center - margin) / denom, (center + margin) / denom)


def calibrate_confidence(signal_results: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """
    Buckets historical fired signals by their score at signal time and
    reports the EMPIRICAL hit rate per bucket, with a Wilson confidence
    interval — this is what a recommendation's "confidence" should be
    grounded in: not how tightly today's sub-scores agree with each
    other, but how often signals that looked like this one actually
    worked, historically. Feed this the concatenation of every ticker's
    evaluate_signals() output (with a large, diverse universe — not just
    the 15-name validation set, which is too small to bucket reliably).
    """
    if signal_results.empty:
        return pd.DataFrame(columns=["direction", "score_bucket", "n", "hit_rate", "ci_low", "ci_high"])

    df = signal_results.copy()
    records = []
    for direction, group in df.groupby("direction"):
        n_unique = group["score_at_signal"].nunique()
        if n_unique <= 1:
            # Not enough distinct score values to bucket meaningfully yet
            # (typical with a small validation universe) — report as one bucket.
            group = group.assign(score_bucket=group["score_at_signal"])
        else:
            try:
                group = group.assign(
                    score_bucket=pd.qcut(
                        group["score_at_signal"], min(n_buckets, n_unique), labels=False, duplicates="drop"
                    )
                )
            except ValueError:
                group = group.assign(score_bucket=group["score_at_signal"])
        for bucket, bgroup in group.groupby("score_bucket"):
            n = len(bgroup)
            successes = int(bgroup["correct"].sum())
            lo, hi = wilson_confidence_interval(successes, n)
            records.append(
                {
                    "direction": direction,
                    "score_bucket": bucket,
                    "n": n,
                    "hit_rate": successes / n if n else np.nan,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(records)


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
