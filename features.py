"""
Feature engineering pipeline (section 3.3 of the proposal).

compute_features() takes a single ticker's OHLCV DataFrame (columns:
timestamp, open, high, low, close, volume) plus a benchmark close series,
and returns the same DataFrame with technical feature columns appended.

This is written against DAILY bars for clarity/testability with the free
row budget of this environment. Section 3.1 of the proposal calls for
HOURLY bars in production — swap the input cadence and the window
constants in config.py accordingly; the formulas themselves don't change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window // 4)).mean()


def _slope(s: pd.Series, window: int = 10) -> pd.Series:
    """Simple sign-of-trend proxy: value now vs. value `window` bars ago."""
    return s - s.shift(window)


def compute_features(df: pd.DataFrame, benchmark_close: pd.Series) -> pd.DataFrame:
    """
    df: DataFrame with columns [timestamp, open, high, low, close, volume],
        sorted ascending by timestamp, index reset.
    benchmark_close: pd.Series of the benchmark's close, aligned by position
        to df (same length/order) — e.g. SPY close for the same dates.
    """
    out = df.copy().reset_index(drop=True)
    close = out["close"]
    volume = out["volume"]

    # --- Moving averages & trend ---
    out["sma_short"] = _sma(close, config.SMA_SHORT)
    out["sma_mid"] = _sma(close, config.SMA_MID)
    out["sma_long"] = _sma(close, config.SMA_LONG)
    out["sma_long_slope"] = _slope(out["sma_long"], window=10)

    # --- 52-week high/low proximity ---
    roll_max = close.rolling(config.LOOKBACK_52W, min_periods=20).max()
    roll_min = close.rolling(config.LOOKBACK_52W, min_periods=20).min()
    out["dist_from_52w_high"] = close / roll_max - 1.0          # <= 0
    out["dist_from_52w_low"] = close / roll_min - 1.0           # >= 0
    out["new_52w_high"] = close >= roll_max
    out["new_52w_low"] = close <= roll_min

    # --- Volatility contraction ---
    returns = close.pct_change()
    realized_vol = returns.rolling(config.VOL_WINDOW).std()
    baseline_vol = returns.rolling(config.VOL_BASELINE_WINDOW).std()
    out["realized_vol"] = realized_vol
    out["vol_contraction_ratio"] = realized_vol / baseline_vol   # < 1 => contracting

    # --- Volume z-score ---
    vol_mean = volume.rolling(config.VOLUME_WINDOW).mean()
    vol_std = volume.rolling(config.VOLUME_WINDOW).std()
    out["volume_z"] = (volume - vol_mean) / vol_std

    # --- Rate-of-change & acceleration ---
    roc_short = close.pct_change(config.ROC_SHORT)
    roc_long = close.pct_change(config.ROC_LONG)
    out["roc_short"] = roc_short
    out["roc_long_annualized"] = roc_long * (252 / config.ROC_LONG)
    out["roc_acceleration"] = roc_short - out["roc_long_annualized"] * (config.ROC_SHORT / 252)

    # --- Relative strength vs. benchmark ---
    bench = pd.Series(benchmark_close).reset_index(drop=True)
    rel_line = close / bench
    out["rel_strength_line"] = rel_line
    for w in config.RS_WINDOWS:
        out[f"rs_return_{w}d"] = rel_line.pct_change(w)

    # --- Trend-template pass/fail (bullish & bearish mirrors; RS-percentile
    #     is left as NaN here since it needs the full cross-sectional universe —
    #     see scoring.attach_rs_percentile) ---
    out["trend_template_bull"] = (
        (close > out["sma_short"])
        & (close > out["sma_mid"])
        & (close > out["sma_long"])
        & (out["sma_short"] > out["sma_mid"])
        & (out["sma_mid"] > out["sma_long"])
        & (out["sma_long_slope"] > 0)
        & (out["dist_from_52w_high"] > config.THRESHOLDS["distance_from_high_bull_max"])
    )
    out["trend_template_bear"] = (
        (close < out["sma_short"])
        & (close < out["sma_mid"])
        & (close < out["sma_long"])
        & (out["sma_short"] < out["sma_mid"])
        & (out["sma_mid"] < out["sma_long"])
        & (out["sma_long_slope"] < 0)
        & (out["dist_from_52w_low"] < config.THRESHOLDS["distance_from_low_bear_max"])
    )

    return out
