"""
Level 1 tests for evaluation/backtest.py.

wilson_confidence_interval is the textbook formula the project deliberately
chose over a "score dispersion" heuristic (see README's "What we borrowed
from... Gemini" section) — these tests just confirm the arithmetic behaves
the way a real Wilson interval should, so a future refactor can't quietly
turn it back into a naive point estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import backtest


def test_wilson_ci_no_data_returns_full_uncertainty():
    lo, hi = backtest.wilson_confidence_interval(0, 0)
    assert (lo, hi) == (0.0, 1.0)


def test_wilson_ci_all_hits_stays_below_one_but_high():
    lo, hi = backtest.wilson_confidence_interval(50, 50)
    assert 0.0 < lo < 1.0
    assert hi <= 1.0
    assert lo > 0.9  # 50/50 hits is strong evidence, interval should say so


def test_wilson_ci_is_centered_near_the_point_estimate():
    lo, hi = backtest.wilson_confidence_interval(5, 10)
    assert lo < 0.5 < hi


def test_wilson_ci_widens_with_fewer_observations():
    lo_small, hi_small = backtest.wilson_confidence_interval(5, 10)
    lo_big, hi_big = backtest.wilson_confidence_interval(50, 100)
    # Same 50% hit rate, but less data should mean a wider interval.
    assert (hi_small - lo_small) > (hi_big - lo_big)


def _toy_signal_df():
    n = 20
    close = pd.Series(np.linspace(100, 100 + n, n))
    bull_signal = pd.Series([False] * n)
    bull_signal.iloc[2] = True   # rises afterward -> correct
    bull_signal.iloc[10] = True  # also rises -> correct (prices are monotonic here)
    bear_signal = pd.Series([False] * n)
    bear_signal.iloc[5] = True   # price still rises afterward -> incorrect for a bear call
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "close": close,
            "bull_signal": bull_signal,
            "bear_signal": bear_signal,
            "bull_score": [4] * n,
            "bear_score": [3] * n,
        }
    )


def test_evaluate_signals_computes_forward_return_and_correctness():
    df = _toy_signal_df()
    results = backtest.evaluate_signals(df, horizon=3)

    bull_rows = results[results["direction"] == "bull"]
    bear_rows = results[results["direction"] == "bear"]
    assert len(bull_rows) == 2
    assert len(bear_rows) == 1
    # Prices are monotonically increasing in this fixture: bull calls are
    # correct, bear calls are not.
    assert bull_rows["correct"].all()
    assert not bear_rows["correct"].any()
    assert (bull_rows["score_at_signal"] == 4).all()


def test_evaluate_signals_drops_signals_too_close_to_the_end():
    df = _toy_signal_df()
    df["bull_signal"] = False
    df["bear_signal"] = False
    df.loc[len(df) - 1, "bull_signal"] = True  # no room for a forward horizon
    results = backtest.evaluate_signals(df, horizon=3)
    assert results.empty


def test_calibrate_confidence_buckets_by_score_and_reports_hit_rate():
    signal_results = pd.DataFrame(
        {
            "direction": ["bull"] * 6 + ["bear"] * 4,
            "score_at_signal": [3, 3, 4, 4, 5, 5, 3, 3, 4, 4],
            "correct": [True, True, True, False, True, True, False, False, True, False],
        }
    )
    calibration = backtest.calibrate_confidence(signal_results, n_buckets=3)

    assert set(calibration["direction"]) == {"bull", "bear"}
    for _, row in calibration.iterrows():
        assert 0.0 <= row["hit_rate"] <= 1.0
        assert row["ci_low"] <= row["hit_rate"] <= row["ci_high"]


def test_calibrate_confidence_handles_a_single_distinct_score():
    # Regression case: a thin validation universe where every fired signal
    # happens to share one score — pd.qcut can't bucket that, so it must
    # fall back to a single bucket per direction instead of raising.
    signal_results = pd.DataFrame(
        {
            "direction": ["bull", "bull", "bull"],
            "score_at_signal": [4, 4, 4],
            "correct": [True, True, False],
        }
    )
    calibration = backtest.calibrate_confidence(signal_results)
    assert len(calibration) == 1
    assert calibration.iloc[0]["n"] == 3


def test_calibrate_confidence_empty_input_returns_empty_frame_with_columns():
    calibration = backtest.calibrate_confidence(pd.DataFrame())
    assert calibration.empty
    assert list(calibration.columns) == ["direction", "score_bucket", "n", "hit_rate", "ci_low", "ci_high"]


def test_summarize_universe_aggregates_by_role_and_direction():
    results = {
        "RISER1": pd.DataFrame(
            {"direction": ["bull"], "correct": [True], "fwd_return_5b": [0.10]}
        ),
        "RISER2": pd.DataFrame(
            {"direction": ["bull"], "correct": [False], "fwd_return_5b": [-0.02]}
        ),
        "NORMAL1": pd.DataFrame(columns=["direction", "correct", "fwd_return_5b"]),
    }
    roles = {"RISER1": "riser", "RISER2": "riser", "NORMAL1": "normal"}

    summary = backtest.summarize_universe(results, roles)

    riser_row = summary[(summary["role"] == "riser") & (summary["direction"] == "bull")].iloc[0]
    assert riser_row["n_signals"] == 2
    assert riser_row["avg_hit_rate"] == 0.5  # mean of per-ticker hit rates (1.0 and 0.0)
    # An empty per-ticker frame (no signals fired) should be skipped, not
    # show up as a spurious "normal" row.
    assert not ((summary["role"] == "normal")).any()
