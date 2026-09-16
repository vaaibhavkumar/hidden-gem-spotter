"""
Expanding-window walk-forward validation report -- "does the confidence
calibration actually generalize, or only look good on data it's already
seen." See evaluation/walk_forward.py's module docstring for the method
and why it's the more rigorous alternative to a single static 70/30
split. This is a periodic AUDIT, not a production pipeline step: run it
after a meaningful amount of new data has accumulated (monthly is a
reasonable cadence -- see the prod-readiness roadmap), not on every
6-hourly ingestion refresh, since a few more hours of data barely moves
a multi-year aggregate statistic.

Run: python3 walk_forward_backtest.py
"""
from __future__ import annotations

import pandas as pd

import config
from evaluation import backtest
from evaluation.walk_forward import run_walk_forward
from run_real_backtest import build_signaled_universe


def main() -> None:
    pd.set_option("display.width", 120)

    print("Building the full-universe signaled dataset (same pipeline run_real_backtest.py uses)...")
    signaled, roles = build_signaled_universe()

    horizon = 5 * config.BARS_PER_DAY  # same horizon as run_real_backtest.py, for a fair comparison
    results = {t: backtest.evaluate_signals(df, horizon=horizon) for t, df in signaled.items()}
    all_results = (
        pd.concat([r for r in results.values() if not r.empty], ignore_index=True)
        if any(len(r) for r in results.values())
        else pd.DataFrame()
    )

    if all_results.empty:
        print("No signals fired across the full history -- nothing to walk-forward validate yet.")
        return

    years_present = sorted(all_results["timestamp"].dt.year.unique())
    print(f"Signals span {len(years_present)} calendar year(s): {years_present}\n")

    MIN_TRAIN_YEARS = 3
    folded = run_walk_forward(all_results, min_train_years=MIN_TRAIN_YEARS)

    if folded.empty:
        print(
            f"Not enough distinct years of signal history to build even one walk-forward fold "
            f"(need at least {MIN_TRAIN_YEARS} training years plus 1 test year). Once more history "
            f"has accumulated, re-run this script."
        )
        return

    for (train_through_year, test_year), fold_df in folded.groupby(["train_through_year", "test_year"]):
        print("=" * 100)
        print(f"FOLD: train on signals through {train_through_year}  ->  test on unseen signals from {test_year}")
        print("=" * 100)
        display_cols = [
            "direction", "score_bucket", "train_n", "train_hit_rate", "train_ci_low", "train_ci_high",
            "test_n", "test_hit_rate", "test_within_train_ci",
        ]
        print(fold_df[display_cols].to_string(index=False))
        print()

    # Headline verdict: across every fold and bucket that actually had
    # test-period signals to check, how often did the training-period
    # confidence interval correctly bracket what really happened next?
    checked = folded[folded["test_n"] > 0]
    if checked.empty:
        print("No fold/bucket combination had any test-period signals to check against -- inconclusive.")
        return

    n_checked = len(checked)
    n_within = int(checked["test_within_train_ci"].sum())
    print("=" * 100)
    print(
        f"SUMMARY: {n_within}/{n_checked} (fold, direction, score-bucket) combinations had a "
        f"test-period hit rate that fell inside the training-period's Wilson confidence interval."
    )
    if n_within / n_checked >= 0.8:
        print(
            "That's a solid generalization rate -- the calibration mostly holds up on data it "
            "never saw during training, not just on the period it was built from."
        )
    else:
        print(
            "That's a meaningfully worse rate than you'd want -- treat report.html's confidence "
            "numbers with real skepticism until this improves. Worth checking whether specific "
            "years or score buckets are driving the misses (print `checked` for the detail)."
        )


if __name__ == "__main__":
    main()
