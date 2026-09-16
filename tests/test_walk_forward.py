"""
Level 1 tests for evaluation/walk_forward.py -- the expanding-window
walk-forward validation (train on year 1..N, test on unseen year N+1,
repeat). These use small synthetic signal-result frames (same shape as
backtest.evaluate_signals()'s output) rather than real market data, so
the fold-boundary and bucket-matching logic can be checked precisely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation import walk_forward


def _signal_row(year: int, direction: str, score: int, correct: bool, month: int = 6) -> dict:
    return {
        "timestamp": pd.Timestamp(year=year, month=month, day=1),
        "direction": direction,
        "score_at_signal": score,
        "price_at_signal": 100.0,
        "fwd_return_5b": 0.02 if correct else -0.02,
        "correct": correct,
    }


def test_expanding_window_folds_empty_input_returns_no_folds():
    assert walk_forward.expanding_window_folds(pd.DataFrame(columns=["timestamp"])) == []


def test_expanding_window_folds_not_enough_years_returns_none():
    # Only 2 distinct years present, but min_train_years=3 needs 4 total.
    df = pd.DataFrame([_signal_row(2022, "bull", 5, True), _signal_row(2023, "bull", 5, True)])
    assert walk_forward.expanding_window_folds(df, min_train_years=3) == []


def test_expanding_window_folds_generates_expected_boundaries():
    years = [2020, 2021, 2022, 2023, 2024]
    df = pd.DataFrame([_signal_row(y, "bull", 5, True) for y in years])
    folds = walk_forward.expanding_window_folds(df, min_train_years=3)
    # First fold trains through the 3rd year (2022), tests on 2023; then
    # expands by one year each time until 2024 is used as a test year.
    assert folds == [(2022, 2023), (2023, 2024)]


def test_expanding_window_folds_uses_years_actually_present_not_a_hardcoded_range():
    # Data starting in a different decade shouldn't matter -- folds are
    # relative to whatever years are actually in the data.
    years = [2015, 2016, 2017, 2018]
    df = pd.DataFrame([_signal_row(y, "bull", 5, True) for y in years])
    folds = walk_forward.expanding_window_folds(df, min_train_years=3)
    assert folds == [(2017, 2018)]


def test_evaluate_fold_perfect_generalization_lands_within_ci():
    # Train and test both have a 100% hit rate at the same score -- the
    # test hit rate should land squarely inside the training CI.
    train = pd.DataFrame([_signal_row(2020, "bull", 80, True) for _ in range(20)])
    test = pd.DataFrame([_signal_row(2021, "bull", 80, True) for _ in range(10)])
    result = walk_forward.evaluate_fold(train, test)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["train_n"] == 20
    assert row["test_n"] == 10
    assert row["test_hit_rate"] == 1.0
    assert row["test_within_train_ci"] == True  # noqa: E712 (explicit bool check reads clearer here)


def test_evaluate_fold_flags_degraded_generalization_outside_ci():
    # Training period looks great (always correct); test period is
    # essentially the opposite (always wrong) -- a textbook "this
    # calibration didn't generalize" case that should NOT land in the CI.
    train = pd.DataFrame([_signal_row(2020, "bull", 80, True) for _ in range(30)])
    test = pd.DataFrame([_signal_row(2021, "bull", 80, False) for _ in range(30)])
    result = walk_forward.evaluate_fold(train, test)

    row = result.iloc[0]
    assert row["test_hit_rate"] == 0.0
    assert row["test_within_train_ci"] == False  # noqa: E712


def test_evaluate_fold_with_no_test_signals_reports_none_for_within_ci():
    train = pd.DataFrame([_signal_row(2020, "bull", 80, True) for _ in range(10)])
    test = pd.DataFrame(columns=train.columns)  # no signals fired in the test year at all
    result = walk_forward.evaluate_fold(train, test)

    row = result.iloc[0]
    assert row["test_n"] == 0
    assert row["test_within_train_ci"] is None


def test_evaluate_fold_separates_bull_and_bear_directions():
    train = pd.DataFrame(
        [_signal_row(2020, "bull", 80, True) for _ in range(10)]
        + [_signal_row(2020, "bear", 20, True) for _ in range(10)]
    )
    test = pd.DataFrame(
        [_signal_row(2021, "bull", 80, True) for _ in range(5)]
        + [_signal_row(2021, "bear", 20, False) for _ in range(5)]
    )
    result = walk_forward.evaluate_fold(train, test)

    assert set(result["direction"]) == {"bull", "bear"}
    bull_row = result[result["direction"] == "bull"].iloc[0]
    bear_row = result[result["direction"] == "bear"].iloc[0]
    assert bull_row["test_hit_rate"] == 1.0
    assert bear_row["test_hit_rate"] == 0.0


def test_run_walk_forward_labels_every_row_with_its_fold():
    years = [2020, 2021, 2022, 2023]
    df = pd.DataFrame([_signal_row(y, "bull", 80, True) for y in years for _ in range(5)])
    folded = walk_forward.run_walk_forward(df, min_train_years=3)

    assert set(folded["train_through_year"]) == {2022}
    assert set(folded["test_year"]) == {2023}


def test_run_walk_forward_with_insufficient_history_returns_empty_frame_with_expected_columns():
    df = pd.DataFrame([_signal_row(2020, "bull", 80, True)])
    folded = walk_forward.run_walk_forward(df, min_train_years=3)
    assert folded.empty
    assert "test_within_train_ci" in folded.columns
