"""
Synthetic OHLCV generators — used only to smoke-test features.py/scoring.py
/backtest.py end-to-end without needing live market data (see the
"Implementation note" in section 3.1 of the project proposal for why real
data isn't wired up yet in this environment).

Each generator produces a daily-bar DataFrame with an intentionally
recognizable pattern so we can check the framework fires (or stays quiet)
where it should:
  - make_riser():   a multi-month base, then a volume-confirmed breakout
                     and sustained advance (the "MU pattern").
  - make_faller():  an uptrend that tops out, then breaks down on volume.
  - make_normal():  a mild random walk with no dramatic pattern — this is
                     the stand-in for the JNJ/KO/PG/V/HD control group.
  - make_benchmark(): a steady moderate uptrend, standing in for SPY.

These are NOT real prices. Once the data pipeline in section 3.1 is wired
up (via a machine with normal internet access, or an allowlisted API),
swap these calls for real OHLCV frames with the same column schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dates(n_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-03", periods=n_days)


def _to_ohlcv(close: np.ndarray, volume: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close)
    noise = np.random.default_rng(0).normal(0, 0.003, size=len(close))
    open_ = close * (1 + noise)
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.astype(int),
        }
    )


def make_benchmark(n_days: int = 750, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0003, 0.009, size=n_days)
    close = 400 * np.cumprod(1 + daily_ret)
    volume = rng.normal(5_000_000, 500_000, size=n_days).clip(min=1_000_000)
    return _to_ohlcv(close, volume, _dates(n_days))


def make_riser(n_days: int = 750, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_len = int(n_days * 0.4)
    breakout_len = n_days - base_len

    # Stage 1: choppy, low-drift base with contracting volatility (VCP-ish).
    base_vol = np.linspace(0.02, 0.006, base_len)
    base_ret = rng.normal(0.0002, 1, size=base_len) * base_vol
    base_close = 60 * np.cumprod(1 + base_ret)

    # Stage 2: strong sustained advance, higher volatility, positive skew.
    breakout_ret = rng.normal(0.006, 0.02, size=breakout_len)
    breakout_close = base_close[-1] * np.cumprod(1 + breakout_ret)

    close = np.concatenate([base_close, breakout_close])

    volume = rng.normal(3_000_000, 400_000, size=n_days).clip(min=500_000)
    # Volume surge right at the breakout bar and briefly after.
    volume[base_len : base_len + 10] *= rng.uniform(2.0, 3.5, size=10)

    return _to_ohlcv(close, volume, _dates(n_days))


def make_faller(n_days: int = 750, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    top_len = int(n_days * 0.35)
    decline_len = n_days - top_len

    # Uptrend into a top.
    up_ret = rng.normal(0.0025, 0.012, size=top_len)
    up_close = 80 * np.cumprod(1 + up_ret)

    # Breakdown: sustained negative drift, higher volatility.
    down_ret = rng.normal(-0.004, 0.02, size=decline_len)
    down_close = up_close[-1] * np.cumprod(1 + down_ret)

    close = np.concatenate([up_close, down_close])

    volume = rng.normal(3_000_000, 400_000, size=n_days).clip(min=500_000)
    volume[top_len : top_len + 10] *= rng.uniform(2.0, 3.5, size=10)

    return _to_ohlcv(close, volume, _dates(n_days))


def make_normal(n_days: int = 750, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0003, 0.010, size=n_days)
    close = 120 * np.cumprod(1 + daily_ret)
    volume = rng.normal(2_000_000, 300_000, size=n_days).clip(min=300_000)
    return _to_ohlcv(close, volume, _dates(n_days))
