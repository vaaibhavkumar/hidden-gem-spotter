"""
Portfolio/trade simulation report -- converts the hit-rate backtest into
an actual annualized return, and compares it against a concrete target:
max(19%, the trailing 5-year S&P 500 CAGR). See
evaluation/portfolio_sim.py's module docstring for the full simulation
design (long-only, technical-only historical actions, equal-weighted
positions capped at 10 concurrent as an explicit stated assumption, exit
on SELL/STRONG SELL or the same 5-trading-day horizon the rest of the
backtest uses -- whichever comes first, idle capital parked in the
benchmark rather than sitting at 0% -- see PARK_IDLE_CASH_IN_BENCHMARK
below).

This is a periodic AUDIT, like walk_forward_backtest.py -- run it after
a meaningful stretch of new data / signals has accumulated, not on every
6-hourly ingestion refresh.

Run: python3 portfolio_backtest.py
"""
from __future__ import annotations

import pandas as pd

import config
from evaluation.portfolio_sim import simulate_portfolio, trailing_n_year_spy_cagr
from run_real_backtest import build_signaled_universe, load_ticker

# ASSUMPTION, disclosed here and in evaluation/portfolio_sim.py's module
# docstring: the user was asked how to size/allocate capital across
# multiple simultaneous BUY signals and did not answer. 10 equal-weighted
# concurrent positions is a reasonable, common default -- change it here
# if a different number is wanted.
MAX_CONCURRENT = 2
TARGET_FLOOR = 0.19

# The first real run (BUY + STRONG BUY, entry_actions left at its
# default) returned a 2.9% trailing-5yr CAGR against an 11.1% SPY
# baseline. Trying STRONG BUY only (fewer, more selective entries) made
# it WORSE (0.7%), which combined with an "avg capital deployed" reading
# of just 7.1% pointed at the real cause: capital was almost never
# actually in a position, regardless of which entries were used --
# change this back to {"STRONG BUY"} to restore "trade every BUY"
# for comparison.
ENTRY_ACTIONS = frozenset({"STRONG BUY"})

# See evaluation/portfolio_sim.py's module docstring ("IDLE CAPITAL CAN
# BE PARKED..."). With this on, capital not currently committed to a
# stock pick is treated as if it were continuously invested in the
# benchmark (SPY) instead of sitting flat at 0% -- a more realistic
# baseline, and the direct fix for the 7.1%-utilization finding above.
# Set to False to go back to the original flat-cash assumption for
# comparison.
PARK_IDLE_CASH_IN_BENCHMARK = True

# The real run at horizon=5 trading days closed 220 of 222 trades by
# simply timing out, not because a SELL/STRONG SELL signal ever fired
# (only 2 trades exited that way). That's a real structural mismatch:
# the entry logic (trend_template's 50/150/200-period moving averages)
# is built in the style of weeks-to-months trend-following, but a 5-day
# clock was forcing every trade closed before a genuine trend reversal
# had any real chance to develop and fire the bearish exit. That 5-day
# number was itself inherited from evaluation.backtest.evaluate_signals()
# -- a *statistics* tool built to measure "is price higher N bars later"
# on a fixed, comparable yardstick, not a number ever chosen as a trading
# rule. Lengthening it here tests whether the SELL/STRONG SELL exit gets
# more chance to actually do its job instead of the timer dominating --
# watch the "Exit reasons" breakdown at the bottom of the report shift
# away from "horizon" as this goes up. Was 5 (see git history for that
# run's numbers); change back to 5 to reproduce it for comparison.
HORIZON_TRADING_DAYS = 20


def main() -> None:
    pd.set_option("display.width", 120)

    print("Building the full-universe signaled dataset (same pipeline run_real_backtest.py uses)...")
    signaled, roles = build_signaled_universe()
    spy_bars = load_ticker(config.BENCHMARK)

    entry_label = " or ".join(sorted(ENTRY_ACTIONS, reverse=True))
    idle_cash_label = (
        "parked in the SPY benchmark between trades" if PARK_IDLE_CASH_IN_BENCHMARK else "sitting flat in cash"
    )
    print(
        f"\nSimulating trades: long-only, technical-only historical actions, equal-weighted, "
        f"capped at {MAX_CONCURRENT} concurrent positions (ASSUMPTION -- position sizing across "
        f"simultaneous BUY signals wasn't specified; see evaluation/portfolio_sim.py's docstring). "
        f"Entering only on: {entry_label}. Idle capital: {idle_cash_label}. Exit horizon: "
        f"{HORIZON_TRADING_DAYS} trading days (or a SELL/STRONG SELL signal, whichever comes first)...\n"
    )
    result = simulate_portfolio(
        signaled,
        max_concurrent=MAX_CONCURRENT,
        entry_actions=ENTRY_ACTIONS,
        benchmark_bars=spy_bars if PARK_IDLE_CASH_IN_BENCHMARK else None,
        horizon_bars=HORIZON_TRADING_DAYS * config.BARS_PER_DAY,
    )
    if result.n_trades == 0 and not result.benchmark_overlay:
        print("No trades were ever opened across the full universe -- nothing to report.")
        return

    curve = result.equity_curve
    years_spanned = (curve["timestamp"].iloc[-1] - curve["timestamp"].iloc[0]).days / 365.25
    full_cagr = result.cagr(years_spanned) if years_spanned > 0 else float("nan")
    trailing_5y_cagr = result.trailing_cagr(5)

    spy_5y_cagr = trailing_n_year_spy_cagr(spy_bars, n_years=5)
    target = max(TARGET_FLOOR, spy_5y_cagr) if spy_5y_cagr is not None else TARGET_FLOOR

    print("=" * 100)
    print("PORTFOLIO SIMULATION RESULTS")
    print("=" * 100)
    utilization = result.average_utilization()
    idle_capital_fate = "tracked the SPY benchmark" if result.benchmark_overlay else "sat in cash earning nothing"
    print(f"Trades opened:         {result.n_trades}")
    if result.win_rate is not None:
        print(f"Win rate:              {result.win_rate * 100:.1f}%")
        print(f"Avg return per trade:  {result.avg_trade_return * 100:.2f}%  (signal quality, independent of sizing)")
    else:
        print("Win rate:              n/a -- no trades were ever opened")
        print("Avg return per trade:  n/a -- no trades were ever opened")
    if utilization is not None:
        print(
            f"Avg capital deployed:  {utilization * 100:.1f}% of the {MAX_CONCURRENT}-slot cap in an individual "
            f"stock pick (the rest {idle_capital_fate} -- this is the other half of the CAGR story, separate "
            f"from whether the trades themselves were any good)"
        )
    else:
        print("Avg capital deployed:  n/a")
    print(f"Starting capital:      ${result.starting_capital:,.2f}")
    print(f"Ending equity:         ${result.ending_equity:,.2f}")
    print(f"Total return:          {result.total_return * 100:.1f}%  over {years_spanned:.1f} years")
    print(f"Full-history CAGR:     {full_cagr * 100:.1f}%")
    if trailing_5y_cagr is not None:
        print(f"Trailing 5yr CAGR:     {trailing_5y_cagr * 100:.1f}%")
    else:
        print("Trailing 5yr CAGR:     n/a -- local history doesn't span 5 years yet")

    print()
    if spy_5y_cagr is not None:
        print(f"Trailing 5yr SPY CAGR: {spy_5y_cagr * 100:.1f}%")
    else:
        print("Trailing 5yr SPY CAGR: n/a -- local SPY history doesn't span 5 years yet")
    print(f"Target return (max of 19% and trailing 5yr SPY CAGR): {target * 100:.1f}%")

    print()
    if trailing_5y_cagr is not None:
        comparison_cagr, label = trailing_5y_cagr, "trailing 5yr"
    else:
        comparison_cagr, label = full_cagr, "full-history (trailing 5yr unavailable)"

    if comparison_cagr >= target:
        print(f"RESULT: strategy's {label} CAGR ({comparison_cagr * 100:.1f}%) MEETS the {target * 100:.1f}% target.")
    else:
        print(f"RESULT: strategy's {label} CAGR ({comparison_cagr * 100:.1f}%) FALLS SHORT of the {target * 100:.1f}% target.")

    if result.trades:
        exit_reasons = pd.Series([t.exit_reason for t in result.trades]).value_counts()
        print("\nExit reasons:")
        print(exit_reasons.to_string())
    else:
        print("\nExit reasons: n/a -- no trades were ever opened")


if __name__ == "__main__":
    main()
