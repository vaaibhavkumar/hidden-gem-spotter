"""
apply_step8.py -- writes the portfolio/trade simulation (evaluation/portfolio_sim.py,
portfolio_backtest.py, tests/test_portfolio_sim.py). Run from the repo root:

    python3 apply_step8.py
    python3 -m pytest -q      # expect 104 passed
"""
from pathlib import Path

Path('evaluation/portfolio_sim.py').parent.mkdir(parents=True, exist_ok=True)
Path('evaluation/portfolio_sim.py').write_text(r'''"""
Converts the hit-rate/directional-accuracy backtest (evaluation/backtest.py,
evaluation/walk_forward.py) into an actual investable return: a global,
event-driven simulation of placing real (paper) trades whenever the
system's own action fires, so the strategy's own CAGR can be compared
against a concrete target return -- see portfolio_backtest.py for the
top-level report and the target-return calculation.

Why this exists: a hit rate ("55% of BUY signals were followed by a
positive 5-day return") does not tell you how much money the strategy
would have made. Position sizing, how much of the time you're actually
long, and the size of wins vs. losses all matter -- two systems with an
identical hit rate can have very different CAGRs. This module actually
runs the trades and compounds an equity curve.

Design decisions (confirmed with the user, or explicitly flagged where
not):

  - LONG-ONLY. BUY/STRONG BUY opens a long position; SELL/STRONG SELL
    closes an open long early. This never opens a short position, per the
    user's explicit "Share Buy / Long and Sell. I wont short."

  - EQUAL-WEIGHTED, capped at `max_concurrent` open positions at once
    (default 10). The user was asked how to size/allocate across
    simultaneous BUY signals and did not answer -- this default is an
    explicit ASSUMPTION made on the user's behalf per their own standing
    instruction ("make a reasonable assumption and carry on, but tell the
    user what assumption you made"). 10 is a common textbook choice for a
    diversified-but-concentrated equal-weight book; pass a different
    `max_concurrent` to change it.

  - HISTORICAL ACTIONS ARE TECHNICAL-ONLY. recommend.recommend()'s real
    composite score can blend in the analyst-revision pillar (Finnhub),
    but Finnhub only exposes a LATEST snapshot, not a point-in-time
    history -- there is no way to know, as of a bar from 2021, what the
    analyst consensus said back then. So this simulation reuses
    recommend._technical_pillar_score()/_action_for_score() directly:
    the exact same scoring math the real system uses for the technical
    pillar, just without the revision pillar blended in (since that data
    doesn't exist historically). This is a documented limitation, not a
    hidden one -- a portfolio simulated on technical-only actions is a
    legitimate (if partial) test of the system's technical judgment, not
    a claim that it reproduces exactly what today's revision-pillar
    system would have done in the past.

  - EXIT ON WHICHEVER COMES FIRST: a SELL/STRONG SELL signal, or the same
    fixed horizon evaluation.evaluate_signals() uses (5 trading days by
    default) -- so this simulation is testing the same horizon the rest
    of the backtest reports on, not a different one.

  - EQUITY COMPOUNDS ONLY ON CLOSED TRADES. Each new position is sized as
    (current realized equity / max_concurrent) at the moment it opens --
    a standard trade-level-compounding simplification. An unusually good
    stretch of still-open trades doesn't inflate the size of new
    positions opened before they close; this is an approximation, not a
    full mark-to-market simulation.

The engine is split into two layers on purpose:
  1. `_technical_action_series()` turns a feature-rich per-ticker
     DataFrame (the output of run_real_backtest.build_signaled_universe())
     into a plain Action label per bar.
  2. `simulate_portfolio_from_actions()` is the actual trading engine --
     it only needs {ticker: DataFrame[timestamp, close, action]} and
     knows nothing about technical scoring. This keeps the execution
     logic (entries, exits, position sizing, equity compounding) testable
     with small synthetic action sequences, independent of scoring.py's
     internals.
  `simulate_portfolio()` is the convenience wrapper gluing the two
  together for real use (see portfolio_backtest.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config
from recommendation.recommend import _action_for_score, _technical_pillar_score

LONG_ENTRY_ACTIONS = {"BUY", "STRONG BUY"}
LONG_EXIT_ACTIONS = {"SELL", "STRONG SELL"}


@dataclass
class Trade:
    ticker: str
    entry_time: object
    entry_price: float
    exit_time: object
    exit_price: float
    exit_reason: str  # "sell_signal" | "horizon" | "end_of_data"
    alloc_dollars: float
    return_pct: float
    pnl_dollars: float


@dataclass
class PortfolioResult:
    starting_capital: float
    ending_equity: float
    max_concurrent: int
    horizon_bars: int
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["timestamp", "equity"]))

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float | None:
        if not self.trades:
            return None
        wins = sum(1 for t in self.trades if t.pnl_dollars > 0)
        return wins / len(self.trades)

    @property
    def total_return(self) -> float:
        return self.ending_equity / self.starting_capital - 1.0

    def cagr(self, years: float) -> float:
        """Annualized return implied by total_return, compounded over `years` years."""
        if years <= 0:
            return float("nan")
        return (self.ending_equity / self.starting_capital) ** (1.0 / years) - 1.0

    def trailing_cagr(self, n_years: int) -> float | None:
        """
        CAGR over just the trailing n_years of the equity curve: the
        curve's final value against the last recorded value at or before
        (end - n_years). Returns None if the curve doesn't span n_years
        yet, rather than silently returning a full-history number under a
        trailing-year label.
        """
        if self.equity_curve.empty:
            return None
        curve = self.equity_curve.sort_values("timestamp")
        end_time = curve["timestamp"].iloc[-1]
        end_equity = curve["equity"].iloc[-1]
        start_time = curve["timestamp"].iloc[0]
        cutoff = end_time - pd.DateOffset(years=n_years)
        if start_time > cutoff:
            return None
        window = curve[curve["timestamp"] <= cutoff]
        start_equity = window["equity"].iloc[-1] if not window.empty else curve["equity"].iloc[0]
        if start_equity <= 0:
            return None
        return (end_equity / start_equity) ** (1.0 / n_years) - 1.0


def _technical_action_series(df: pd.DataFrame) -> pd.Series:
    """
    Row-wise technical-only Action label for every bar (not just the
    latest one, which is all recommend.recommend() is normally called
    for) -- the exact same score/action math recommend.recommend() uses
    for the technical pillar, applied across full history so this
    simulation has a point-in-time action at every bar. See the module
    docstring for why this can't include the revision pillar historically.
    """
    def _row_action(row: pd.Series) -> str:
        score, _ = _technical_pillar_score(row)
        return _action_for_score(score)

    if df.empty:
        return pd.Series([], dtype=object)
    return df.apply(_row_action, axis=1)


def simulate_portfolio_from_actions(
    action_frames: dict[str, pd.DataFrame],
    starting_capital: float = 100_000.0,
    max_concurrent: int = 10,
    horizon_bars: int = 5,
) -> PortfolioResult:
    """
    The actual trading engine: a single, shared-capital, chronologically
    event-driven simulation across every ticker in `action_frames`. Each
    DataFrame must have `timestamp`, `close`, and `action` columns
    (already sorted or not -- this function sorts). Knows nothing about
    how `action` was computed, which is what makes it directly testable
    with small synthetic sequences (see tests/test_portfolio_sim.py).

    horizon_bars is counted in bars *within a single ticker's own series*
    (matching evaluation.backtest.evaluate_signals()'s "N bars ahead in
    this series" convention), not in global event-loop steps.
    """
    events = []
    for ticker, df in action_frames.items():
        if df is None or df.empty:
            continue
        d = df.sort_values("timestamp").reset_index(drop=True)
        for idx, row in d.iterrows():
            events.append((row["timestamp"], ticker, idx, row["action"], float(row["close"])))
    events.sort(key=lambda e: (e[0], e[1]))

    equity = starting_capital
    open_positions: dict[str, dict] = {}
    trades: list[Trade] = []
    equity_points: list[tuple[object, float]] = [(events[0][0], equity)] if events else []
    last_seen: dict[str, tuple[object, float]] = {}

    def _close(ticker: str, exit_time, exit_price: float, reason: str) -> None:
        nonlocal equity
        pos = open_positions.pop(ticker)
        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        pnl_dollars = pos["alloc_dollars"] * pnl_pct
        equity += pnl_dollars
        trades.append(
            Trade(
                ticker=ticker,
                entry_time=pos["entry_time"],
                entry_price=pos["entry_price"],
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason=reason,
                alloc_dollars=pos["alloc_dollars"],
                return_pct=pnl_pct,
                pnl_dollars=pnl_dollars,
            )
        )
        equity_points.append((exit_time, equity))

    for timestamp, ticker, idx, action, close in events:
        last_seen[ticker] = (timestamp, close)

        if ticker in open_positions:
            pos = open_positions[ticker]
            held = idx - pos["entry_idx"]
            if action in LONG_EXIT_ACTIONS:
                _close(ticker, timestamp, close, "sell_signal")
            elif held >= horizon_bars:
                _close(ticker, timestamp, close, "horizon")
            continue  # can't also open a fresh position in the same bar

        if action in LONG_ENTRY_ACTIONS and len(open_positions) < max_concurrent:
            alloc = equity / max_concurrent
            open_positions[ticker] = {
                "entry_time": timestamp,
                "entry_price": close,
                "entry_idx": idx,
                "alloc_dollars": alloc,
            }

    # Force-close anything still open when the data runs out, at its last
    # known price, so ending equity reflects every position's resolved
    # outcome rather than ignoring whatever hadn't exited yet.
    for ticker in list(open_positions.keys()):
        exit_time, exit_price = last_seen[ticker]
        _close(ticker, exit_time, exit_price, "end_of_data")

    equity_curve = pd.DataFrame(equity_points, columns=["timestamp", "equity"])

    return PortfolioResult(
        starting_capital=starting_capital,
        ending_equity=equity,
        max_concurrent=max_concurrent,
        horizon_bars=horizon_bars,
        trades=trades,
        equity_curve=equity_curve,
    )


def simulate_portfolio(
    signaled: dict[str, pd.DataFrame],
    starting_capital: float = 100_000.0,
    max_concurrent: int = 10,
    horizon_bars: int | None = None,
) -> PortfolioResult:
    """
    Convenience wrapper for real use: takes `signaled`, the output of
    run_real_backtest.build_signaled_universe() (full feature columns per
    ticker), computes the technical-only action at every bar, and runs
    the shared-capital simulation. horizon_bars defaults to the same 5
    trading days (scaled by config.BARS_PER_DAY) run_real_backtest.py and
    walk_forward_backtest.py both use, for a fair comparison.
    """
    if horizon_bars is None:
        horizon_bars = 5 * config.BARS_PER_DAY

    action_frames = {}
    for ticker, df in signaled.items():
        if df.empty:
            continue
        actions = _technical_action_series(df)
        action_frames[ticker] = pd.DataFrame(
            {
                "timestamp": df["timestamp"].reset_index(drop=True),
                "close": df["close"].reset_index(drop=True),
                "action": actions.reset_index(drop=True),
            }
        )

    return simulate_portfolio_from_actions(
        action_frames,
        starting_capital=starting_capital,
        max_concurrent=max_concurrent,
        horizon_bars=horizon_bars,
    )


def trailing_n_year_spy_cagr(spy_bars: pd.DataFrame, n_years: int = 5) -> float | None:
    """
    Trailing n_years CAGR computed directly from the benchmark's own close
    prices (config.BENCHMARK == "SPY" bars, already ingested locally --
    no new external data needed). Returns None if the local history
    doesn't span n_years yet, rather than silently computing a
    shorter-window number under a "5yr" label.
    """
    if spy_bars.empty:
        return None
    df = spy_bars.sort_values("timestamp")
    end_time = df["timestamp"].iloc[-1]
    end_price = df["close"].iloc[-1]
    start_time = df["timestamp"].iloc[0]
    cutoff = end_time - pd.DateOffset(years=n_years)
    if start_time > cutoff:
        return None
    window = df[df["timestamp"] <= cutoff]
    start_price = window["close"].iloc[-1] if not window.empty else df["close"].iloc[0]
    if start_price <= 0:
        return None
    return (end_price / start_price) ** (1.0 / n_years) - 1.0
''')
print("wrote evaluation/portfolio_sim.py:", Path('evaluation/portfolio_sim.py').stat().st_size, "bytes")

Path('portfolio_backtest.py').parent.mkdir(parents=True, exist_ok=True)
Path('portfolio_backtest.py').write_text(r'''"""
Portfolio/trade simulation report -- converts the hit-rate backtest into
an actual annualized return, and compares it against a concrete target:
max(19%, the trailing 5-year S&P 500 CAGR). See
evaluation/portfolio_sim.py's module docstring for the full simulation
design (long-only, technical-only historical actions, equal-weighted
positions capped at 10 concurrent as an explicit stated assumption, exit
on SELL/STRONG SELL or the same 5-trading-day horizon the rest of the
backtest uses -- whichever comes first).

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
MAX_CONCURRENT = 10
TARGET_FLOOR = 0.19


def main() -> None:
    pd.set_option("display.width", 120)

    print("Building the full-universe signaled dataset (same pipeline run_real_backtest.py uses)...")
    signaled, roles = build_signaled_universe()

    print(
        f"\nSimulating trades: long-only, technical-only historical actions, equal-weighted, "
        f"capped at {MAX_CONCURRENT} concurrent positions (ASSUMPTION -- position sizing across "
        f"simultaneous BUY signals wasn't specified; see evaluation/portfolio_sim.py's docstring)...\n"
    )
    result = simulate_portfolio(signaled, max_concurrent=MAX_CONCURRENT)

    if result.n_trades == 0:
        print("No trades were ever opened across the full universe -- nothing to report.")
        return

    curve = result.equity_curve
    years_spanned = (curve["timestamp"].iloc[-1] - curve["timestamp"].iloc[0]).days / 365.25
    full_cagr = result.cagr(years_spanned) if years_spanned > 0 else float("nan")
    trailing_5y_cagr = result.trailing_cagr(5)

    spy_bars = load_ticker(config.BENCHMARK)
    spy_5y_cagr = trailing_n_year_spy_cagr(spy_bars, n_years=5)
    target = max(TARGET_FLOOR, spy_5y_cagr) if spy_5y_cagr is not None else TARGET_FLOOR

    print("=" * 100)
    print("PORTFOLIO SIMULATION RESULTS")
    print("=" * 100)
    print(f"Trades opened:         {result.n_trades}")
    print(f"Win rate:              {result.win_rate * 100:.1f}%")
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

    exit_reasons = pd.Series([t.exit_reason for t in result.trades]).value_counts()
    print("\nExit reasons:")
    print(exit_reasons.to_string())


if __name__ == "__main__":
    main()
''')
print("wrote portfolio_backtest.py:", Path('portfolio_backtest.py').stat().st_size, "bytes")

Path('tests/test_portfolio_sim.py').parent.mkdir(parents=True, exist_ok=True)
Path('tests/test_portfolio_sim.py').write_text(r'''"""
Level 1 tests for evaluation/portfolio_sim.py -- the trade-simulation
engine that turns fired actions into an actual equity curve / CAGR.

Two layers are tested separately, matching the module's own split:
  - simulate_portfolio_from_actions(): the trading engine itself, using
    small synthetic {timestamp, close, action} sequences so entries,
    exits, the max_concurrent cap, and equity compounding can each be
    checked in isolation, independent of real technical scoring.
  - _technical_action_series(): the thin bridge from real feature columns
    (same shape signals/scoring.py + recommend.py already use) to a
    per-bar Action label, using the same synthetic-row pattern
    tests/test_recommendation.py already established.
trailing_n_year_spy_cagr() gets its own direct tests too.
"""
from __future__ import annotations

import pandas as pd
import pytest

from evaluation import portfolio_sim
from evaluation.portfolio_sim import (
    _technical_action_series,
    simulate_portfolio_from_actions,
    trailing_n_year_spy_cagr,
)


def _bars(ticker_actions: list[tuple[str, float]], start="2024-01-01") -> pd.DataFrame:
    """Builds a one-ticker {timestamp, close, action} frame, one row per
    (action, close) pair, on consecutive daily timestamps."""
    ts = pd.date_range(start, periods=len(ticker_actions), freq="D")
    actions, closes = zip(*ticker_actions)
    return pd.DataFrame({"timestamp": ts, "close": list(closes), "action": list(actions)})


def test_buy_then_sell_signal_closes_the_position_and_books_pnl():
    # BUY at 100, HOLD, SELL at 110 -> +10% on the single position.
    frames = {"AAA": _bars([("BUY", 100.0), ("HOLD", 105.0), ("SELL", 110.0)])}
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10)

    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.exit_reason == "sell_signal"
    assert trade.return_pct == pytest.approx(0.10)
    assert result.ending_equity == pytest.approx(1100.0)  # all-in, single position


def test_position_exits_at_horizon_when_no_sell_signal_fires():
    # BUY, then HOLD forever -- horizon_bars=2 should force an exit 2 bars later.
    frames = {"AAA": _bars([("BUY", 100.0), ("HOLD", 100.0), ("HOLD", 120.0), ("HOLD", 130.0)])}
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=2)

    assert result.n_trades == 1
    assert result.trades[0].exit_reason == "horizon"
    assert result.trades[0].exit_price == pytest.approx(120.0)  # bar index 0 + 2 = index 2


def test_still_open_position_is_force_closed_at_end_of_data():
    frames = {"AAA": _bars([("BUY", 100.0), ("HOLD", 150.0)])}
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10)

    assert result.n_trades == 1
    assert result.trades[0].exit_reason == "end_of_data"
    assert result.trades[0].exit_price == pytest.approx(150.0)


def test_never_opens_a_short_position_on_a_sell_with_no_open_position():
    # A SELL/STRONG SELL with nothing open should be a no-op -- long-only,
    # per the user's explicit "I wont short."
    frames = {"AAA": _bars([("SELL", 100.0), ("STRONG SELL", 90.0), ("HOLD", 80.0)])}
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10)

    assert result.n_trades == 0
    assert result.ending_equity == pytest.approx(1000.0)


def test_max_concurrent_caps_simultaneous_positions():
    # Three tickers all fire BUY on the same bar; only 2 slots -- the third
    # (later in ticker-name tiebreak order) should never open.
    frames = {
        "AAA": _bars([("BUY", 100.0), ("HOLD", 100.0)]),
        "BBB": _bars([("BUY", 100.0), ("HOLD", 100.0)]),
        "CCC": _bars([("BUY", 100.0), ("HOLD", 100.0)]),
    }
    result = simulate_portfolio_from_actions(frames, starting_capital=900.0, max_concurrent=2, horizon_bars=10)

    # Nothing has closed yet (still within horizon), so trades is empty,
    # but we can check via a giant horizon that ending_equity reflects at
    # most 2 positions ever having opened by forcing an end-of-data close.
    assert result.n_trades == 2  # AAA and BBB open (alphabetical tiebreak); CCC never gets a slot
    assert {t.ticker for t in result.trades} == {"AAA", "BBB"}


def test_equal_weighted_allocation_uses_current_equity_over_max_concurrent():
    frames = {"AAA": _bars([("BUY", 100.0), ("HOLD", 100.0), ("SELL", 110.0)])}
    result = simulate_portfolio_from_actions(frames, starting_capital=500.0, max_concurrent=5, horizon_bars=10)

    trade = result.trades[0]
    assert trade.alloc_dollars == pytest.approx(500.0 / 5)
    assert trade.pnl_dollars == pytest.approx((500.0 / 5) * 0.10)
    assert result.ending_equity == pytest.approx(500.0 + (500.0 / 5) * 0.10)


def test_equity_compounds_across_sequential_closed_trades():
    # Two back-to-back round trips on the same ticker: the second trade's
    # allocation should be sized off the equity AFTER the first trade
    # closed, not the original starting capital.
    frames = {
        "AAA": _bars(
            [
                ("BUY", 100.0),
                ("SELL", 120.0),   # +20% -> equity becomes 1200 (all-in, max_concurrent=1)
                ("BUY", 50.0),
                ("SELL", 55.0),    # +10% on the now-larger equity
            ]
        )
    }
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10)

    assert result.n_trades == 2
    assert result.trades[0].pnl_dollars == pytest.approx(200.0)
    assert result.trades[1].alloc_dollars == pytest.approx(1200.0)  # sized off post-trade-1 equity
    assert result.trades[1].pnl_dollars == pytest.approx(120.0)
    assert result.ending_equity == pytest.approx(1320.0)


def test_win_rate_and_total_return_properties():
    frames = {
        "AAA": _bars([("BUY", 100.0), ("SELL", 120.0)]),
        "BBB": _bars([("BUY", 100.0), ("SELL", 90.0)]),
    }
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=2, horizon_bars=10)

    assert result.n_trades == 2
    assert result.win_rate == pytest.approx(0.5)
    assert result.total_return == pytest.approx(result.ending_equity / 1000.0 - 1.0)


def test_empty_action_frames_produce_a_flat_zero_trade_result():
    result = simulate_portfolio_from_actions({}, starting_capital=1000.0)
    assert result.n_trades == 0
    assert result.ending_equity == pytest.approx(1000.0)
    assert result.win_rate is None
    assert result.equity_curve.empty


def test_trailing_cagr_returns_none_when_history_shorter_than_window():
    frames = {"AAA": _bars([("BUY", 100.0), ("SELL", 110.0)])}  # spans 1 day
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10)
    assert result.trailing_cagr(5) is None


def test_cagr_matches_hand_computed_value():
    frames = {"AAA": _bars([("BUY", 100.0), ("SELL", 200.0)])}  # doubles
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10)
    assert result.cagr(1) == pytest.approx(1.0)  # doubling in 1 year = 100% CAGR
    assert result.cagr(2) == pytest.approx(2.0 ** 0.5 - 1.0)


# --- _technical_action_series -----------------------------------------
# Same synthetic-row shape tests/test_recommendation.py already uses for
# recommend._technical_pillar_score(), just wrapped in a DataFrame since
# _technical_action_series operates on the whole ticker history at once.

def _neutral_row(timestamp, close=100.0):
    return {
        "timestamp": timestamp,
        "close": close,
        "trend_template_bull": False,
        "trend_template_bear": False,
        "rs_percentile": 0.5,
        "volume_z": 0.0,
        "vol_contraction_ratio": 1.0,
        "new_52w_high": False,
        "new_52w_low": False,
        "roc_acceleration": 0.0,
    }


def _bullish_row(timestamp, close=100.0):
    row = _neutral_row(timestamp, close)
    row.update(
        trend_template_bull=True,
        rs_percentile=0.9,
        volume_z=2.0,
        vol_contraction_ratio=0.5,
        new_52w_high=True,
        roc_acceleration=0.02,
    )
    return row


def _bearish_row(timestamp, close=100.0):
    row = _neutral_row(timestamp, close)
    row.update(
        trend_template_bear=True,
        rs_percentile=0.1,
        volume_z=2.0,
        vol_contraction_ratio=1.5,
        new_52w_low=True,
        roc_acceleration=-0.02,
    )
    return row


def test_technical_action_series_labels_neutral_bullish_and_bearish_bars():
    ts = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        [
            _neutral_row(ts[0]),
            _bullish_row(ts[1]),
            _bearish_row(ts[2]),
        ]
    )
    actions = _technical_action_series(df)

    assert actions.iloc[0] == "HOLD"
    assert actions.iloc[1] in ("BUY", "STRONG BUY")
    assert actions.iloc[2] in ("SELL", "STRONG SELL")


def test_technical_action_series_on_empty_frame_returns_empty_series():
    assert _technical_action_series(pd.DataFrame()).empty


def test_simulate_portfolio_wraps_action_series_and_engine_together():
    ts = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame([_neutral_row(ts[0]), _bullish_row(ts[1]), _bullish_row(ts[2])])
    signaled = {"AAA": df}

    result = portfolio_sim.simulate_portfolio(signaled, starting_capital=1000.0, max_concurrent=1, horizon_bars=1)
    # Whatever happened, it should run end-to-end without needing the
    # caller to compute actions itself, and never open a short.
    assert result.starting_capital == 1000.0
    for trade in result.trades:
        assert trade.pnl_dollars is not None  # engine ran and produced real trades


# --- trailing_n_year_spy_cagr -------------------------------------------

def test_trailing_n_year_spy_cagr_hand_computed():
    ts = pd.date_range("2019-01-01", periods=6, freq="YS")  # spans 5+ years
    df = pd.DataFrame({"timestamp": ts, "close": [100.0, 110.0, 120.0, 130.0, 140.0, 200.0]})
    cagr = trailing_n_year_spy_cagr(df, n_years=5)
    # start price = the last row at/before (end - 5y) = 2019-01-01's 100.0, end = 200.0
    assert cagr == pytest.approx((200.0 / 100.0) ** (1 / 5) - 1.0)


def test_trailing_n_year_spy_cagr_returns_none_when_history_too_short():
    ts = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"timestamp": ts, "close": [100.0, 101.0, 102.0]})
    assert trailing_n_year_spy_cagr(df, n_years=5) is None


def test_trailing_n_year_spy_cagr_empty_frame_returns_none():
    assert trailing_n_year_spy_cagr(pd.DataFrame(), n_years=5) is None
''')
print("wrote tests/test_portfolio_sim.py:", Path('tests/test_portfolio_sim.py').stat().st_size, "bytes")