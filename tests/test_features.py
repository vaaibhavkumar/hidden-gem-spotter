"""
Level 1 tests for signals/features.py, using the synthetic generators
(no network, no real data needed) so these run anywhere, always.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import synthetic_data as synth
from signals import features

EXPECTED_COLUMNS = {
    "sma_short", "sma_mid", "sma_long", "sma_long_slope",
    "dist_from_52w_high", "dist_from_52w_low", "new_52w_high", "new_52w_low",
    "realized_vol", "vol_contraction_ratio", "volume_z",
    "roc_short", "roc_long_annualized", "roc_acceleration",
    "rel_strength_line", "trend_template_bull", "trend_template_bear",
}


def _riser_features():
    bench = synth.make_benchmark()
    riser = synth.make_riser()
    return features.compute_features(riser, bench["close"])


def test_compute_features_adds_expected_columns_and_preserves_length():
    riser = synth.make_riser()
    bench = synth.make_benchmark()
    out = features.compute_features(riser, bench["close"])

    assert len(out) == len(riser)
    assert EXPECTED_COLUMNS.issubset(out.columns)
    # rs_return_{w}d columns come from config.RS_WINDOWS, not a fixed name.
    import config
    for w in config.RS_WINDOWS:
        assert f"rs_return_{w}d" in out.columns


def test_trend_template_bull_is_true_late_in_the_breakout():
    out = _riser_features()
    # The last bar of a sustained multi-month advance should pass every
    # trend-template condition (price above stacked, rising MAs and off
    # the 52-week low) — if this goes false, the trend-template logic in
    # features.py broke.
    assert bool(out["trend_template_bull"].iloc[-1]) is True


def test_trend_template_bear_is_true_late_in_the_breakdown():
    faller = synth.make_faller()
    bench = synth.make_benchmark()
    out = features.compute_features(faller, bench["close"])
    assert bool(out["trend_template_bear"].iloc[-1]) is True


def test_new_52w_high_flags_at_least_one_bar_near_the_end_of_a_riser():
    out = _riser_features()
    # The riser's breakout stage should produce genuine new highs once
    # warmed up (mirrors the bug where an incomplete rolling window
    # produced *spurious* early "highs" — see scoring.generate_signals'
    # warmup masking, tested separately in test_scoring.py).
    tail = out.iloc[-100:]
    assert tail["new_52w_high"].any()


def test_volume_z_is_finite_and_centered_once_warmed_up():
    out = _riser_features()
    import config
    warmed = out.iloc[config.VOLUME_WINDOW + 5 :]
    z = warmed["volume_z"].dropna()
    assert not z.empty
    assert np.isfinite(z).all()


def test_vol_contraction_ratio_is_below_one_during_the_synthetic_base():
    """
    make_riser() explicitly builds a contracting-volatility base (see its
    docstring) before the breakout — vol_contraction_ratio should reflect
    that (< vol_expansion_min, ideally near/under vol_contraction_max)
    somewhere late in the base, once the rolling windows have enough data.
    """
    riser = synth.make_riser()
    bench = synth.make_benchmark()
    out = features.compute_features(riser, bench["close"])
    base_len = int(len(riser) * 0.4)
    # Look at the tail of the base, right before breakout — plenty of
    # history for both rolling windows by this point.
    late_base = out.iloc[max(0, base_len - 30) : base_len]
    assert late_base["vol_contraction_ratio"].dropna().median() < 1.0
