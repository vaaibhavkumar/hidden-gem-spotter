"""
Level 1 tests for signals/scoring.py.

The debounce and warmup-masking tests here are explicit regression tests
for the two real bugs the synthetic-data smoke test caught (see README's
"Why real data isn't wired up yet"): a burn-in artifact that fired
spurious signals before enough rolling-window history existed, and a
score hovering near its threshold that re-fired a "fresh" signal on
every re-cross. If either bug comes back, one of these tests should fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from signals import scoring


def test_bullish_conditions_all_true_scores_six():
    row = pd.Series(
        {
            "trend_template_bull": True,
            "rs_percentile": 0.85,
            "volume_z": 2.0,
            "vol_contraction_ratio": 0.5,
            "new_52w_high": True,
            "roc_acceleration": 0.01,
        }
    )
    conditions = scoring.bullish_conditions(row)
    assert all(conditions.values())
    assert scoring.bullish_score(row) == 6


def test_bearish_conditions_all_true_scores_six():
    row = pd.Series(
        {
            "trend_template_bear": True,
            "rs_percentile": 0.1,
            "volume_z": 2.0,
            "vol_contraction_ratio": 1.5,
            "new_52w_low": True,
            "roc_acceleration": -0.01,
        }
    )
    conditions = scoring.bearish_conditions(row)
    assert all(conditions.values())
    assert scoring.bearish_score(row) == 6


def test_conditions_default_to_false_on_a_bare_row():
    # A row missing every feature column (e.g. still warming up) must not
    # crash and must not accidentally count as bullish/bearish.
    row = pd.Series(dtype=object)
    assert scoring.bullish_score(row) == 0
    assert scoring.bearish_score(row) == 0


def test_attach_rs_percentile_ranks_cross_sectionally_per_timestamp():
    ts = pd.date_range("2024-01-01", periods=5, freq="D")
    a = pd.DataFrame({"timestamp": ts, "rs_return_126d": [0.1, 0.2, 0.3, 0.4, 0.5]})
    b = pd.DataFrame({"timestamp": ts, "rs_return_126d": [0.5, 0.4, 0.3, 0.2, 0.1]})
    features_by_ticker = {"A": a, "B": b}

    scoring.attach_rs_percentile(features_by_ticker)

    assert "rs_percentile" in a.columns and "rs_percentile" in b.columns
    # At the first timestamp A has the lower return, so a lower percentile.
    assert a["rs_percentile"].iloc[0] < b["rs_percentile"].iloc[0]
    # The two tickers' returns cross over by the last timestamp.
    assert a["rs_percentile"].iloc[-1] > b["rs_percentile"].iloc[-1]


def test_debounce_collapses_a_chattering_cluster_into_one_fresh_signal():
    # A qualifying flag toggling on/off rapidly (score wiggling around the
    # threshold) followed by one clearly isolated later flag — should
    # register as exactly two "fresh" signals, not one per toggle.
    qualified = pd.Series([False, True, False, True, False, True, False, False, False, False, True])
    fresh = scoring._debounce(qualified, cooldown=5)
    assert fresh.sum() == 2
    assert bool(fresh.iloc[1]) is True
    assert bool(fresh.iloc[10]) is True


def test_debounce_does_not_refire_while_continuously_qualified():
    qualified = pd.Series([True] * 6)
    fresh = scoring._debounce(qualified, cooldown=3)
    assert fresh.tolist() == [True, False, False, False, False, False]


def test_generate_signals_masks_bull_signals_before_warmup():
    warmup = config.LOOKBACK_52W + config.SMA_LONG
    n = warmup + 50
    df = pd.DataFrame(
        {
            "trend_template_bull": [True] * n,
            "trend_template_bear": [False] * n,
            "rs_percentile": [0.9] * n,
            "volume_z": [2.0] * n,
            "vol_contraction_ratio": [0.5] * n,
            "new_52w_high": [True] * n,
            "roc_acceleration": [0.01] * n,
            "close": np.linspace(100, 200, n),
        }
    )

    out = scoring.generate_signals(df)

    # Every bar looks like a qualifying breakout from bar 0 onward, but
    # the first `warmup` bars must be masked out regardless.
    assert not out["bull_signal"].iloc[:warmup].any()
    # Once warmup ends, the (still-qualifying) condition should register
    # as a fresh rising-edge signal exactly at that boundary.
    assert bool(out["bull_signal"].iloc[warmup]) is True


def test_generate_signals_respects_an_explicit_warmup_override():
    n = 40
    df = pd.DataFrame(
        {
            "trend_template_bull": [True] * n,
            "trend_template_bear": [False] * n,
            "rs_percentile": [0.9] * n,
            "volume_z": [2.0] * n,
            "vol_contraction_ratio": [0.5] * n,
            "new_52w_high": [True] * n,
            "roc_acceleration": [0.01] * n,
            "close": np.linspace(10, 20, n),
        }
    )
    out = scoring.generate_signals(df, warmup=10, cooldown=5)
    assert not out["bull_signal"].iloc[:10].any()
    assert bool(out["bull_signal"].iloc[10]) is True

def test_attach_rs_percentile_default_column_scales_with_bars_per_day():
    # Regression test: attach_rs_percentile's default rs_col used to be
    # hardcoded as "rs_return_126d", which only ever matched features.py's
    # actual column name on daily bars (BARS_PER_DAY=1). Against real
    # hourly data (BARS_PER_DAY=7, so the ~6-month column is really named
    # "rs_return_882d"), that hardcoded default raised
    # KeyError('rs_return_126d') the first time this ran on real data.
    config.set_bars_per_day(7)
    assert config.RS_WINDOWS[1] == 882

    ts = pd.date_range("2024-01-01", periods=5, freq="h")
    a = pd.DataFrame({"timestamp": ts, "rs_return_882d": [0.1, 0.2, 0.3, 0.4, 0.5]})
    b = pd.DataFrame({"timestamp": ts, "rs_return_882d": [0.5, 0.4, 0.3, 0.2, 0.1]})
    features_by_ticker = {"A": a, "B": b}

    scoring.attach_rs_percentile(features_by_ticker)  # must not raise KeyError

    assert "rs_percentile" in a.columns and "rs_percentile" in b.columns


def test_generate_signals_default_cooldown_scales_with_bars_per_day():
    # Regression test: generate_signals()'s default cooldown used to be a
    # bare 20 (bars) -- a real month on daily data but only ~3 days on
    # hourly data (20 hourly bars at 7/day), so debounce wasn't actually
    # suppressing chatter at hourly granularity. The first real backtest
    # against hourly Alpaca data fired 5-7x more "fresh" signals per
    # ticker than the daily-bar synthetic smoke test ever showed because
    # of this. Default must now be 20 * config.BARS_PER_DAY.
    config.set_bars_per_day(7)
    try:
        n = config.LOOKBACK_52W + config.SMA_LONG + 60
        df = pd.DataFrame(
            {
                "trend_template_bull": [True] * n,
                "trend_template_bear": [False] * n,
                "rs_percentile": [0.9] * n,
                "volume_z": [2.0] * n,
                "vol_contraction_ratio": [0.5] * n,
                "new_52w_high": [True] * n,
                "roc_acceleration": [0.01] * n,
                "close": np.linspace(100, 200, n),
            }
        )
        # Two qualifying rising edges 30 bars apart: more than the old bare
        # cooldown=20 (so the bug would let both re-fire as "fresh"), but
        # well inside the correctly-scaled default of 20*BARS_PER_DAY=140
        # bars (so the fix must collapse them into a single fresh signal).
        warmup = config.LOOKBACK_52W + config.SMA_LONG
        gap_start, gap_end = warmup, warmup + 30
        df["trend_template_bull"] = (
            [True] * gap_start + [False] * (gap_end - gap_start) + [True] * (n - gap_end)
        )
        out = scoring.generate_signals(df)  # cooldown left as default
        assert int(out["bull_signal"].sum()) == 1
    finally:
        config.set_bars_per_day(1)
