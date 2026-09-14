"""
Composite scoring / ranking engine (sections 2E, 2F, 3.4 of the proposal).

Two things happen here:
1. attach_rs_percentile(): turns each ticker's raw relative-strength return
   into a cross-sectional percentile rank *within the universe*, for a given
   date — this needs every ticker's features, not just one, which is why it
   lives here rather than in features.py.
2. bullish_score() / bearish_score(): the weighted checklist composite score
   (section 2E) with the bearish mirror (section 2F), plus a helper to flag
   a "fresh" signal (a stock newly crossing the threshold, not one that has
   already been flagged for a while) — since "early stage" means the former.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def attach_rs_percentile(features_by_ticker: dict[str, pd.DataFrame], rs_col: str = "rs_return_126d") -> None:
    """
    Mutates each DataFrame in-place, adding `rs_percentile` = this ticker's
    cross-sectional percentile rank of `rs_col` among all tickers in the
    universe, computed independently at each timestamp (no look-ahead:
    only tickers/timestamps already present are used).
    """
    # Align on timestamp across all tickers, rank cross-sectionally per row.
    combined = pd.concat(
        {t: df.set_index("timestamp")[rs_col] for t, df in features_by_ticker.items()},
        axis=1,
    )
    pct_rank = combined.rank(axis=1, pct=True, na_option="keep")
    for t, df in features_by_ticker.items():
        df["rs_percentile"] = df["timestamp"].map(pct_rank[t])


def bullish_score(row: pd.Series) -> int:
    """Count of independent bullish conditions firing (section 2E composite)."""
    score = 0
    if bool(row.get("trend_template_bull", False)):
        score += 1
    if row.get("rs_percentile", np.nan) >= config.THRESHOLDS["rs_percentile_bull"]:
        score += 1
    if row.get("volume_z", np.nan) >= config.THRESHOLDS["volume_z_confirm"]:
        score += 1
    if row.get("vol_contraction_ratio", np.nan) <= config.THRESHOLDS["vol_contraction_max"]:
        score += 1
    if bool(row.get("new_52w_high", False)):
        score += 1
    if row.get("roc_acceleration", np.nan) > 0:
        score += 1
    return score


def bearish_score(row: pd.Series) -> int:
    """Count of independent bearish conditions firing (section 2F mirror)."""
    score = 0
    if bool(row.get("trend_template_bear", False)):
        score += 1
    if row.get("rs_percentile", np.nan) <= config.THRESHOLDS["rs_percentile_bear"]:
        score += 1
    if row.get("volume_z", np.nan) >= config.THRESHOLDS["volume_z_confirm"]:
        # High volume on a down move is a distribution day either way we score it —
        # direction is disambiguated by trend_template_bear / new_52w_low.
        score += 1
    if bool(row.get("new_52w_low", False)):
        score += 1
    if row.get("roc_acceleration", np.nan) < 0:
        score += 1
    return score


def _debounce(qualified: pd.Series, cooldown: int) -> pd.Series:
    """
    Rising-edge detector with a minimum gap between flags, so a score that
    chatters back and forth across the threshold doesn't register as a new
    "early stage" signal every time it re-crosses. Without this, a single
    noisy topping/basing process fires dozens of near-duplicate alerts
    instead of one — the proposal's section 3.4 warns single-bar alerts on
    a noisy composite score are "mostly noise"; this is the concrete fix.
    """
    fresh = pd.Series(False, index=qualified.index)
    last_fire = -10**9
    prev = False
    for i, val in enumerate(qualified.fillna(False).to_numpy()):
        if val and not prev and (i - last_fire) >= cooldown:
            fresh.iloc[i] = True
            last_fire = i
        prev = val
    return fresh


def generate_signals(df: pd.DataFrame, warmup: int | None = None, cooldown: int = 20) -> pd.DataFrame:
    """
    Adds bull_score, bear_score, and "fresh signal" flags (bull_signal /
    bear_signal) to a single ticker's feature DataFrame.

    warmup: number of leading bars to mask out entirely (default:
        config.LOOKBACK_52W + config.SMA_LONG). Before this many bars have
        accumulated, the 52-week-high/low and long moving-average features
        are built on incomplete history and produce spurious "new highs" —
        exclude them rather than let the backtest count false positives
        that are really just a cold-start artifact.
    cooldown: minimum bars between two "fresh" signals of the same type
        (see _debounce) — prevents chattering across the threshold from
        registering as repeated new signals.
    """
    out = df.copy()
    out["bull_score"] = out.apply(bullish_score, axis=1)
    out["bear_score"] = out.apply(bearish_score, axis=1)

    min_score = config.THRESHOLDS["signal_score_min"]
    bull_qualified = out["bull_score"] >= min_score
    bear_qualified = out["bear_score"] >= min_score

    if warmup is None:
        warmup = config.LOOKBACK_52W + config.SMA_LONG
    bull_qualified.iloc[:warmup] = False
    bear_qualified.iloc[:warmup] = False

    out["bull_signal"] = _debounce(bull_qualified, cooldown)
    out["bear_signal"] = _debounce(bear_qualified, cooldown)

    return out
