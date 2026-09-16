"""
Expanding-window walk-forward validation for calibrate_confidence().

Why this exists: calibrate_confidence() (backtest.py) is the one part of
this project that's actually derived from data rather than hand-specified
— it looks at every fired signal's historical outcome and reports an
empirical hit rate per score bucket. Feeding it the ENTIRE history (as
run_real_backtest.py does today) answers "how did this system perform on
data it has already seen," which is a much weaker claim than "would this
calibration have held up if I'd only known the past." This module answers
the second, harder question with an expanding-window walk-forward: train
on year 1..N, test (unseen) on year N+1; then train on year 1..N+1, test
on year N+2; and so on until there's no more future year left to test on.

This is the standard, more rigorous alternative to a single static 70/30
split in quantitative finance specifically because a single split can
land in a lucky or unlucky market regime — repeating the split across
every available year gives a much more honest read on whether the
confidence numbers in report.html actually generalize.

A random (shuffled) train/test split would be wrong here: nearby bars are
autocorrelated, and the whole point is "given only what was knowable as
of some date, did the calibration hold up on what happened next" — a
question a shuffle can't answer, since it would let future information
leak backward into the "past."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.backtest import wilson_confidence_interval


def expanding_window_folds(all_results: pd.DataFrame, min_train_years: int = 3) -> list[tuple[int, int]]:
    """
    Determines fold boundaries directly from the data actually available
    (no hardcoded calendar years, so this keeps working as more history
    accumulates). Returns a list of (train_through_year, test_year) pairs,
    e.g. [(2021, 2022), (2022, 2023), (2023, 2024), ...] — fold i trains
    on every signal dated in or before train_through_year and tests on
    signals dated in test_year (== train_through_year + 1).

    min_train_years: the minimum number of calendar years of signal
    history required before the first fold's training window is
    considered large enough to calibrate from at all. Folds requiring
    fewer years than this, or a test year beyond the last year present in
    the data, are not generated.
    """
    if all_results.empty:
        return []
    years = sorted(all_results["timestamp"].dt.year.unique())
    if len(years) < min_train_years + 1:
        return []  # not enough distinct years to build even one fold
    first_year = years[0]
    last_year = years[-1]
    folds = []
    train_through = first_year + min_train_years - 1
    while train_through + 1 <= last_year:
        folds.append((train_through, train_through + 1))
        train_through += 1
    return folds


def _bucket_edges(scores: pd.Series, n_buckets: int) -> np.ndarray | None:
    """
    Quantile bin edges derived from the TRAINING scores only. Returns None
    when there aren't enough distinct values to bucket meaningfully (same
    "too few distinct scores" case backtest.calibrate_confidence() already
    handles for a single-sample calibration).
    """
    n_unique = scores.nunique()
    if n_unique <= 1:
        return None
    try:
        _, edges = pd.qcut(scores, min(n_buckets, n_unique), retbins=True, duplicates="drop")
    except ValueError:
        return None
    return edges


def evaluate_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """
    For one walk-forward fold: builds score buckets from train_df (per
    direction), computes each bucket's train-period hit rate + Wilson
    interval (this is what calibrate_confidence() would have told you,
    knowing only the training period), then applies those SAME bucket
    edges to test_df and reports the test-period's actual hit rate in
    each bucket — the out-of-sample check calibrate_confidence() alone
    can't give you, since it's always fit and evaluated on the same data.

    Returns one row per (direction, score_bucket) with train_n,
    train_hit_rate, train_ci_low/high, test_n, test_hit_rate, and
    test_within_train_ci (whether the test-period hit rate falls inside
    the confidence interval the training period implied -- the headline
    "did this generalize" verdict for that bucket).
    """
    records = []
    for direction, train_group in train_df.groupby("direction"):
        edges = _bucket_edges(train_group["score_at_signal"], n_buckets)
        test_group = test_df[test_df["direction"] == direction]

        if edges is None:
            # Not enough distinct training scores to bucket -- report the
            # whole direction as one bucket, same fallback
            # calibrate_confidence() uses.
            train_buckets = pd.Series(0, index=train_group.index)
            test_buckets = pd.Series(0, index=test_group.index)
        else:
            train_buckets = pd.cut(train_group["score_at_signal"], bins=edges, labels=False, include_lowest=True)
            test_buckets = pd.cut(test_group["score_at_signal"], bins=edges, labels=False, include_lowest=True)

        train_group = train_group.assign(score_bucket=train_buckets)
        test_group = test_group.assign(score_bucket=test_buckets)

        for bucket, t_train in train_group.groupby("score_bucket"):
            train_n = len(t_train)
            train_successes = int(t_train["correct"].sum())
            train_hit_rate = train_successes / train_n if train_n else np.nan
            ci_low, ci_high = wilson_confidence_interval(train_successes, train_n)

            t_test = test_group[test_group["score_bucket"] == bucket]
            test_n = len(t_test)
            test_hit_rate = t_test["correct"].mean() if test_n else np.nan
            # A tiny floating-point tolerance: the Wilson formula's bounds
            # can land a hair off an exact 0.0/1.0 (e.g. 0.9999999999999998
            # instead of 1.0) due to ordinary float arithmetic, which would
            # otherwise flag a genuinely perfect match as "outside" the
            # interval it should trivially be inside.
            eps = 1e-9
            within_ci = (ci_low - eps <= test_hit_rate <= ci_high + eps) if test_n else None

            records.append(
                {
                    "direction": direction,
                    "score_bucket": bucket,
                    "train_n": train_n,
                    "train_hit_rate": train_hit_rate,
                    "train_ci_low": ci_low,
                    "train_ci_high": ci_high,
                    "test_n": test_n,
                    "test_hit_rate": test_hit_rate,
                    "test_within_train_ci": within_ci,
                }
            )
    return pd.DataFrame(records)


def run_walk_forward(all_results: pd.DataFrame, min_train_years: int = 3, n_buckets: int = 5) -> pd.DataFrame:
    """
    Runs every expanding-window fold and returns one combined DataFrame
    (adds train_through_year/test_year columns identifying each fold) --
    see the module docstring for the overall method, and
    walk_forward_backtest.py for how this gets printed as a report.
    """
    folds = expanding_window_folds(all_results, min_train_years=min_train_years)
    all_rows = []
    for train_through_year, test_year in folds:
        train_df = all_results[all_results["timestamp"].dt.year <= train_through_year]
        test_df = all_results[all_results["timestamp"].dt.year == test_year]
        fold_result = evaluate_fold(train_df, test_df, n_buckets=n_buckets)
        fold_result.insert(0, "train_through_year", train_through_year)
        fold_result.insert(1, "test_year", test_year)
        all_rows.append(fold_result)
    if not all_rows:
        return pd.DataFrame(
            columns=[
                "train_through_year", "test_year", "direction", "score_bucket",
                "train_n", "train_hit_rate", "train_ci_low", "train_ci_high",
                "test_n", "test_hit_rate", "test_within_train_ci",
            ]
        )
    return pd.concat(all_rows, ignore_index=True)
