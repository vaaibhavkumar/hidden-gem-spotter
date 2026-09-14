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


def attach_rs_percentile(features_by_ticker: dict[str, pd.DataFrame], rs_col: str | None = None) -> None:
    """
    Mutates each DataFrame in-place, adding `rs_percentile` = this ticker's
    cross-sectional percentile rank of `rs_col` among all tickers in the
    universe, computed independently at each timestamp (no look-ahead:
    only tickers/timestamps already present are used).

    rs_col: which of features.py's rs_return_{w}d columns to rank on.
        Defaults to the ~6-month window (the middle entry of
        config.RS_WINDOWS). Computed here, not hardcoded as
        "rs_return_126d", because that column name scales with
        config.BARS_PER_DAY (126 for daily bars, 882 for hourly at 7
        bars/day) — a hardcoded default only worked on daily data.
    """
    if rs_col is None:
        rs_col = f"rs_return_{config.RS_WINDOWS[1]}d"

    # Align on timestamp across all tickers, rank cross-sectionally per row.
    combined = pd.concat(
        {t: df.set_index("timestamp")[rs_col] for t, df in features_by_ticker.items()},
        axis=1,
    )
    pct_rank = combined.rank(axis=1, pct=True, na_option="keep")
    for t, df in features_by_ticker.items():
        df["rs_percentile"] = df["timestamp"].map(pct_rank[t])


#: human-readable label for each condition, used both to build the score
#: and to explain *why* a signal fired (see recommend.py's reasoning bullets).
BULLISH_CONDITION_LABELS = {
    "trend_template": "Trend template intact (price > 50/150/200-period MAs, MAs stacked bullishly, 200 rising)",
    "rs_top": "Relative strength vs. benchmark in the top {pct:.0f}% of the universe",
    "volume_confirm": "Volume confirming the move (>{z:.1f} std above trailing average)",
    "vol_contraction": "Volatility was contracting before this move (VCP-style setup)",
    "new_high": "First new 52-week high after a base",
    "roc_accel": "Rate of price change is accelerating",
}
BEARISH_CONDITION_LABELS = {
    "trend_template": "Trend template broken (price < 50/150/200-period MAs, MAs stacked bearishly, 200 falling)",
    "rs_bottom": "Relative strength vs. benchmark in the bottom {pct:.0f}% of the universe",
    "volume_confirm": "Volume confirming the move (>{z:.1f} std above trailing average — a distribution day)",
    "vol_expansion": "Volatility is expanding into widening swings (distribution-style top)",
    "new_low": "First new 52-week low after a topping process",
    "roc_decel": "Rate of price decline is accelerating",
}


def bullish_conditions(row: pd.Series) -> dict[str, bool]:
    """Each independent bullish condition (section 2E composite), named so
    callers (scoring here, recommend.py's reasoning) don't duplicate logic."""
    return {
        "trend_template": bool(row.get("trend_template_bull", False)),
        "rs_top": row.get("rs_percentile", np.nan) >= config.THRESHOLDS["rs_percentile_bull"],
        "volume_confirm": row.get("volume_z", np.nan) >= config.THRESHOLDS["volume_z_confirm"],
        "vol_contraction": row.get("vol_contraction_ratio", np.nan) <= config.THRESHOLDS["vol_contraction_max"],
        "new_high": bool(row.get("new_52w_high", False)),
        "roc_accel": row.get("roc_acceleration", np.nan) > 0,
    }


def bearish_conditions(row: pd.Series) -> dict[str, bool]:
    """Each independent bearish condition (section 2F mirror)."""
    return {
        "trend_template": bool(row.get("trend_template_bear", False)),
        "rs_bottom": row.get("rs_percentile", np.nan) <= config.THRESHOLDS["rs_percentile_bear"],
        # High volume on a down move is a distribution day either way we score it —
        # direction is disambiguated by trend_template/new_52w_low.
        "volume_confirm": row.get("volume_z", np.nan) >= config.THRESHOLDS["volume_z_confirm"],
        "vol_expansion": row.get("vol_contraction_ratio", np.nan) >= config.THRESHOLDS["vol_expansion_min"],
        "new_low": bool(row.get("new_52w_low", False)),
        "roc_decel": row.get("roc_acceleration", np.nan) < 0,
    }


def bullish_score(row: pd.Series) -> int:
    """Count of independent bullish conditions firing (section 2E composite)."""
    return sum(bool(v) for v in bullish_conditions(row).values())


def bearish_score(row: pd.Series) -> int:
    """Count of independent bearish conditions firing (section 2F mirror)."""
    return sum(bool(v) for v in bearish_conditions(row).values())


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
