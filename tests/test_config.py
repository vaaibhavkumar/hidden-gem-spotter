"""
Level 1 (pure unit) tests for config.py.

These lock in the daily/hourly bar-scaling fix (see README's "Why real
data isn't wired up yet" #1 and the demo.py history) — a regression here
would silently turn "50-day" into "50-bar" again on hourly data.
"""
from __future__ import annotations

import config


def test_daily_bars_is_the_default():
    # Reset to the documented default in case an earlier test in the same
    # session called set_bars_per_day() and left globals mutated.
    config.set_bars_per_day(1)
    assert config.BARS_PER_DAY == 1
    assert config.SMA_SHORT == 50
    assert config.SMA_MID == 150
    assert config.SMA_LONG == 200
    assert config.LOOKBACK_52W == 252


def test_set_bars_per_day_scales_every_window():
    config.set_bars_per_day(7)  # hourly, ~7 bars/session
    try:
        assert config.BARS_PER_DAY == 7
        assert config.SMA_SHORT == 50 * 7
        assert config.SMA_MID == 150 * 7
        assert config.SMA_LONG == 200 * 7
        assert config.LOOKBACK_52W == 252 * 7
        assert config.VOL_WINDOW == 20 * 7
        assert config.VOL_BASELINE_WINDOW == 60 * 7
        assert config.VOLUME_WINDOW == 20 * 7
        assert config.ROC_SHORT == 20 * 7
        assert config.ROC_LONG == 60 * 7
        assert config.RS_WINDOWS == tuple(d * 7 for d in (63, 126, 252))
    finally:
        # Don't leak hourly scaling into other test modules.
        config.set_bars_per_day(1)


def test_validation_universe_has_five_of_each_role():
    roles = [meta["role"] for meta in config.VALIDATION_UNIVERSE.values()]
    assert roles.count("riser") == 5
    assert roles.count("faller") == 5
    assert roles.count("normal") == 5
