from pathlib import Path

content_run_real_backtest_py = r'''"""
Same pipeline as demo.py, but reading real OHLCV data (produced by
ingestion/alpaca_ingest.py) instead of synthetic series.

Run: python3 run_real_backtest.py

Reads from data/market.duckdb (see ingestion/data_store.py) if present —
this is the path alpaca_ingest.py now writes to. Falls back to
data/<TICKER>.csv for backward compatibility with the very first
prototype run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from evaluation import backtest
from ingestion import data_store
from recommendation import html_report, recommend
from signals import features, scoring

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_ticker(ticker: str) -> pd.DataFrame:
    df = data_store.load_bars(ticker)
    if not df.empty:
        return df.sort_values("timestamp").reset_index(drop=True)

    csv_path = DATA_DIR / f"{ticker}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    raise FileNotFoundError(
        f"No data for {ticker} in {data_store.DB_PATH} or {csv_path} — run "
        "python3 -m ingestion.alpaca_ingest (on a machine with internet "
        "access) and bring its data/ folder here first."
    )


def detect_bars_per_day(df: pd.DataFrame) -> int:
    """
    Infers whether this data holds daily or (regular-session) hourly bars
    from the median gap between consecutive timestamps, and returns the
    BARS_PER_DAY value config.py's window sizes should use. This exists so
    a daily-vs-hourly mismatch (see config.py's BARS_PER_DAY comment)
    can't silently produce a 7x-too-short "50-day" moving average —
    it's detected and applied automatically instead.
    """
    deltas = df["timestamp"].diff().dropna()
    if deltas.empty:
        return config.BARS_PER_DAY
    median_hours = deltas.median().total_seconds() / 3600
    if median_hours >= 20:   # ~1 trading day between bars (weekends inflate the mean, not the median)
        return 1
    return 7                 # hourly bars, ~7 per regular trading session


def build_signaled_universe() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    The shared "load real data -> align -> compute features -> rank
    RS-percentile across the full universe -> generate signals" pipeline,
    factored out of main() so other entry points (walk_forward_backtest.py)
    can build the exact same signaled dataset without duplicating this
    logic or risking it drifting out of sync with what production actually
    scores. Returns (signaled, roles): signaled is {ticker: DataFrame with
    bull_signal/bear_signal columns} for every ticker in config.all_tickers();
    roles is {ticker: "riser"/"faller"/"normal"} for the validation-only subset.
    """
    benchmark = load_ticker(config.BENCHMARK)

    detected = detect_bars_per_day(benchmark)
    if detected != config.BARS_PER_DAY:
        print(
            f"Detected {'daily' if detected == 1 else 'hourly'} bars from {config.BENCHMARK}'s "
            f"timestamps — calling config.set_bars_per_day({detected}) so moving-average/52-week "
            f"windows are the right length (was {config.BARS_PER_DAY}). If this guess is wrong for "
            "your data, call config.set_bars_per_day(...) yourself before running the pipeline."
        )
        config.set_bars_per_day(detected)

    # Validation tickers keep their known riser/faller/normal label; the
    # broader calibration tickers have no such label (that's the point —
    # they're an unselected sample) and are excluded from the by-role
    # false-positive summary below, but included everywhere else (feature
    # computation, RS-percentile ranking, and confidence calibration) —
    # see config.all_tickers()'s docstring for why the two sets exist.
    roles = {t: meta["role"] for t, meta in config.VALIDATION_UNIVERSE.items()}
    universe = config.all_tickers()
    series = {t: load_ticker(t) for t in universe}

    # Align every ticker's frame to the benchmark's timestamps (inner join)
    # so the cross-sectional RS-percentile ranking in scoring.py compares
    # apples to apples at each bar.
    aligned = {}
    for t, df in series.items():
        merged = pd.merge(df, benchmark[["timestamp", "close"]], on="timestamp", suffixes=("", "_bench"))
        aligned[t] = merged.drop(columns=["close_bench"]).reset_index(drop=True)
        aligned[t]["_bench_close"] = merged["close_bench"].reset_index(drop=True)

    feats = {t: features.compute_features(df, df["_bench_close"]) for t, df in aligned.items()}
    # Ranked across the FULL universe (validation + calibration), not just
    # the 15 hand-picked names — this is the actual fix for "RS-percentile
    # ranking is only correct across whatever tickers you feed it" (see
    # README's "Known open items").
    scoring.attach_rs_percentile(feats)

    signaled = {t: scoring.generate_signals(df) for t, df in feats.items()}
    return signaled, roles


def main() -> None:
    pd.set_option("display.width", 120)

    signaled, roles = build_signaled_universe()
    universe = config.all_tickers()

    print(f"=== Fresh signal counts ({len(roles)} validation tickers; "
          f"{len(universe) - len(roles)} calibration-only tickers omitted from this table) ===")
    for t, df in signaled.items():
        data_store.log_signals(t, df)  # persist to data/market.duckdb's `signals` table
        if t not in roles:
            continue
        role = roles[t]
        print(
            f"{t:6s} ({role:6s}) bull_signals={int(df['bull_signal'].sum()):3d}  "
            f"bear_signals={int(df['bear_signal'].sum()):3d}"
        )

    print("\n=== Signal dates (first 5 of each type per validation ticker) ===")
    for t, df in signaled.items():
        if t not in roles:
            continue
        bulls = df.loc[df["bull_signal"], ["timestamp", "close"]].head(5)
        bears = df.loc[df["bear_signal"], ["timestamp", "close"]].head(5)
        if not bulls.empty:
            print(f"{t} BUY signals:\n{bulls.to_string(index=False)}")
        if not bears.empty:
            print(f"{t} SELL signals:\n{bears.to_string(index=False)}")

    horizon = 5 * config.BARS_PER_DAY  # ~5 trading days ahead, in whatever bar size the data uses
    results = {t: backtest.evaluate_signals(df, horizon=horizon) for t, df in signaled.items()}

    print("\n=== Backtest summary by role (this is the false-positive-rate check — ")
    print("    validation tickers only, since 'role' is only meaningful for the hand-picked set) ===")
    validation_results = {t: r for t, r in results.items() if t in roles}
    summary = backtest.summarize_universe(validation_results, roles)
    print(summary.to_string(index=False) if not summary.empty else "(no signals to evaluate)")

    print(f"\n=== Calibrated confidence by score bucket (validation + calibration tickers, "
          f"{len(universe)} total — still a proxy sample, see config.CALIBRATION_UNIVERSE's ")
    print("    point-in-time caveat, but no longer bucketed on just 15 correlated names) ===")
    all_results = pd.concat([r for r in results.values() if not r.empty], ignore_index=True) if any(len(r) for r in results.values()) else pd.DataFrame()
    calibration = backtest.calibrate_confidence(all_results) if not all_results.empty else pd.DataFrame()
    print(calibration.to_string(index=False) if not calibration.empty else "(not enough signals to calibrate)")

    print("\n=== Most recent recommendation per validation ticker (Action / Score / Confidence / Reasoning) ===")
    recommendations = []
    for t, df in signaled.items():
        if t not in roles:
            continue
        latest = df.iloc[-1]
        # None until ingestion/finnhub_ingest.py has been run for this
        # ticker -- recommend() already treats a None revision_score as
        # "pillar not available" (see its "Pillars not yet available"
        # reasoning note), so this is safe to call unconditionally.
        revision_score = data_store.load_latest_revision_score(t)
        rec = recommend.recommend(latest, ticker=t, revision_score=revision_score, calibration_table=calibration)
        print(rec)
        recommendations.append(rec)
        data_store.log_recommendation(rec)  # persist to data/market.duckdb's `recommendations` table

    # One recommendation history per validation ticker (oldest -> newest),
    # so report.html can show "how has this call changed over past runs",
    # not just today's snapshot.
    history = {t: data_store.load_recommendation_history(t) for t in roles}

    report_path = html_report.generate_html_report(recommendations, roles, history=history, out_path="report.html")
    print(f"\nWrote {report_path} -- open it in a browser to see the watchlist summary + per-ticker detail.")


if __name__ == "__main__":
    main()
'''
Path("run_real_backtest.py").write_text(content_run_real_backtest_py)
print("wrote run_real_backtest.py:", len(content_run_real_backtest_py), "bytes")

content_evaluation_walk_forward_py = r'''"""
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
'''
Path("evaluation/walk_forward.py").write_text(content_evaluation_walk_forward_py)
print("wrote evaluation/walk_forward.py:", len(content_evaluation_walk_forward_py), "bytes")

content_walk_forward_backtest_py = r'''"""
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
'''
Path("walk_forward_backtest.py").write_text(content_walk_forward_backtest_py)
print("wrote walk_forward_backtest.py:", len(content_walk_forward_backtest_py), "bytes")

content_tests_test_walk_forward_py = r'''"""
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
'''
Path("tests/test_walk_forward.py").write_text(content_tests_test_walk_forward_py)
print("wrote tests/test_walk_forward.py:", len(content_tests_test_walk_forward_py), "bytes")