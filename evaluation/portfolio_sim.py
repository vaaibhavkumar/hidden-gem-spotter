"""
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

  - IDLE CAPITAL CAN BE PARKED IN A BENCHMARK INSTEAD OF SITTING AT 0%.
    A real early run surfaced this concretely: with a selective entry
    signal, capital was only ever actually in a stock position ~7% of
    the time -- the other ~93% sat completely flat, earning nothing,
    while the S&P 500 itself returned over 11%/year across the same
    span. That's not how a real account works: uninvested cash isn't
    nothing, at minimum it's sitting in *something*. Pass
    `benchmark_prices`/`benchmark_bars` (e.g. SPY's own close prices) and
    any capital not currently committed to an open stock pick is marked,
    at every event, as if it were continuously invested in that
    benchmark instead. This makes the reported CAGR a much more honest
    "how would this actually have performed" number: the strategy always
    earns at least the benchmark's return, and the stock-picking logic
    only has to justify itself by beating that baseline, not by beating
    literal cash. This overlay is OPT-IN (default: old flat-cash
    behavior, unchanged) specifically so every existing test and result
    stays reproducible; only the *idle* portion gets this treatment --
    open stock positions are still settled only at close, the same
    trade-level-compounding simplification as always.

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
    entry_actions: frozenset[str] = field(default_factory=lambda: frozenset(LONG_ENTRY_ACTIONS))
    benchmark_overlay: bool = False
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["timestamp", "equity"]))
    occupancy_curve: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["timestamp", "open_count"]))

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
    def avg_trade_return(self) -> float | None:
        """Simple (unweighted) mean of each closed trade's return_pct -- the
        per-trade signal-quality number, independent of how much capital was
        actually deployed (see average_utilization() for that side of it)."""
        if not self.trades:
            return None
        return sum(t.return_pct for t in self.trades) / len(self.trades)

    def average_utilization(self) -> float | None:
        """
        Time-weighted average fraction of max_concurrent slots that were
        actually filled across the simulated period -- e.g. 0.15 means
        only 15% of the capital this simulation could have deployed was
        ever actually in an individual stock pick. Without
        benchmark_overlay, the rest sat in cash earning nothing; with it,
        the rest was tracking the benchmark instead (see the module
        docstring) -- either way, this number is about how much of the
        time capital was in a stock *pick specifically*, which is what
        separates two very different explanations for a weak CAGR: bad
        trades (a low avg_trade_return) vs. rarely picking a stock at all
        (a low average_utilization) -- the fix for each is completely
        different, and a hit rate or a raw CAGR alone can't tell them
        apart. Returns None if there's no time span to weight over
        (fewer than 2 recorded occupancy points).
        """
        if self.occupancy_curve.empty or len(self.occupancy_curve) < 2:
            return None
        curve = self.occupancy_curve.sort_values("timestamp").reset_index(drop=True)
        total_span = (curve["timestamp"].iloc[-1] - curve["timestamp"].iloc[0]).total_seconds()
        if total_span <= 0:
            return None
        weighted_open = 0.0
        for i in range(len(curve) - 1):
            dt = (curve["timestamp"].iloc[i + 1] - curve["timestamp"].iloc[i]).total_seconds()
            weighted_open += dt * curve["open_count"].iloc[i]
        return (weighted_open / total_span) / self.max_concurrent

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
    entry_actions: frozenset[str] | set[str] | None = None,
    benchmark_prices: pd.Series | None = None,
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

    entry_actions: which action labels open a new long position. Defaults
    to LONG_ENTRY_ACTIONS ({"BUY", "STRONG BUY"}) -- pass
    {"STRONG BUY"} to only trade the highest-conviction signal, an
    experiment worth running when trading every BUY doesn't clear your
    target return: fewer, higher-conviction trades vs. more, noisier
    ones. SELL/STRONG SELL always closes an open long regardless of this
    setting (exiting on any bearish flip is a different question from
    which signal was strong enough to buy in the first place).

    benchmark_prices: optional Series indexed by timestamp (e.g. SPY's own
    close prices, sorted or not -- this function sorts) giving the price
    of a benchmark asset idle capital is parked in between trades. See
    the module docstring's "IDLE CAPITAL CAN BE PARKED..." section.
    Defaults to None: capital not committed to an open position simply
    sits flat at its last realized value (the original behavior, kept as
    the default so every existing result stays reproducible). When
    provided, it must cover the full span of `action_frames`' timestamps
    -- a timestamp before the benchmark's own first price raises
    ValueError rather than silently mis-sizing the very first trade.
    """
    entry_actions = frozenset(entry_actions) if entry_actions is not None else frozenset(LONG_ENTRY_ACTIONS)
    use_benchmark = benchmark_prices is not None and not benchmark_prices.empty

    events = []
    for ticker, df in action_frames.items():
        if df is None or df.empty:
            continue
        d = df.sort_values("timestamp").reset_index(drop=True)
        for idx, row in d.iterrows():
            events.append((row["timestamp"], ticker, idx, row["action"], float(row["close"])))
    events.sort(key=lambda e: (e[0], e[1]))

    if not events:
        empty = pd.DataFrame(columns=["timestamp", "equity"])
        empty_occ = pd.DataFrame(columns=["timestamp", "open_count"])
        return PortfolioResult(
            starting_capital=starting_capital,
            ending_equity=starting_capital,
            max_concurrent=max_concurrent,
            horizon_bars=horizon_bars,
            entry_actions=entry_actions,
            benchmark_overlay=use_benchmark,
            trades=[],
            equity_curve=empty,
            occupancy_curve=empty_occ,
        )

    if use_benchmark:
        bp = benchmark_prices.sort_index()
        first_price = bp.asof(events[0][0])
        if first_price is None or pd.isna(first_price):
            raise ValueError(
                f"benchmark_prices has no price at or before {events[0][0]!r} -- it must cover "
                "the full span of action_frames' timestamps to price the very first allocation."
            )
        spy_shares = starting_capital / first_price
    else:
        equity = starting_capital

    open_positions: dict[str, dict] = {}
    trades: list[Trade] = []
    occupancy_points: list[tuple[object, int]] = []
    last_seen: dict[str, tuple[object, float]] = {}

    def _committed_dollars() -> float:
        return sum(pos["alloc_dollars"] for pos in open_positions.values())

    def _equity_now(timestamp) -> float:
        # Total portfolio NAV at this instant: for the flat-cash default,
        # `equity` already IS that total (it only moves on a realized
        # close). With the benchmark overlay, it's whatever's parked in
        # the benchmark (marked to its price right now) plus whatever's
        # committed to open stock positions (at their fixed, not
        # continuously marked-to-market, allocation -- see module
        # docstring).
        if use_benchmark:
            return spy_shares * bp.asof(timestamp) + _committed_dollars()
        return equity

    equity_points: list[tuple[object, float]] = [(events[0][0], _equity_now(events[0][0]))]

    def _close(ticker: str, exit_time, exit_price: float, reason: str) -> None:
        nonlocal equity, spy_shares
        pos = open_positions.pop(ticker)
        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        pnl_dollars = pos["alloc_dollars"] * pnl_pct
        proceeds = pos["alloc_dollars"] + pnl_dollars
        if use_benchmark:
            spy_shares += proceeds / bp.asof(exit_time)
        else:
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
        equity_points.append((exit_time, _equity_now(exit_time)))

    for timestamp, ticker, idx, action, close in events:
        last_seen[ticker] = (timestamp, close)

        if ticker in open_positions:
            pos = open_positions[ticker]
            held = idx - pos["entry_idx"]
            if action in LONG_EXIT_ACTIONS:
                _close(ticker, timestamp, close, "sell_signal")
            elif held >= horizon_bars:
                _close(ticker, timestamp, close, "horizon")
            # Recorded on every event, including ones that just held a
            # position open with no state change, so average_utilization()
            # can time-weight correctly across however long the count
            # stayed at its current level.
            occupancy_points.append((timestamp, len(open_positions)))
            if use_benchmark:
                equity_points.append((timestamp, _equity_now(timestamp)))
            continue  # can't also open a fresh position in the same bar

        if action in entry_actions and len(open_positions) < max_concurrent:
            alloc = _equity_now(timestamp) / max_concurrent
            if use_benchmark:
                spy_shares -= alloc / bp.asof(timestamp)
            open_positions[ticker] = {
                "entry_time": timestamp,
                "entry_price": close,
                "entry_idx": idx,
                "alloc_dollars": alloc,
            }

        occupancy_points.append((timestamp, len(open_positions)))
        if use_benchmark:
            equity_points.append((timestamp, _equity_now(timestamp)))

    # Force-close anything still open when the data runs out, at its last
    # known price, so ending equity reflects every position's resolved
    # outcome rather than ignoring whatever hadn't exited yet.
    for ticker in list(open_positions.keys()):
        exit_time, exit_price = last_seen[ticker]
        _close(ticker, exit_time, exit_price, "end_of_data")

    ending_equity = _equity_now(events[-1][0])
    equity_curve = pd.DataFrame(equity_points, columns=["timestamp", "equity"])
    occupancy_curve = pd.DataFrame(occupancy_points, columns=["timestamp", "open_count"])

    return PortfolioResult(
        starting_capital=starting_capital,
        ending_equity=ending_equity,
        max_concurrent=max_concurrent,
        horizon_bars=horizon_bars,
        entry_actions=entry_actions,
        benchmark_overlay=use_benchmark,
        trades=trades,
        equity_curve=equity_curve,
        occupancy_curve=occupancy_curve,
    )


def simulate_portfolio(
    signaled: dict[str, pd.DataFrame],
    starting_capital: float = 100_000.0,
    max_concurrent: int = 10,
    horizon_bars: int | None = None,
    entry_actions: frozenset[str] | set[str] | None = None,
    benchmark_bars: pd.DataFrame | None = None,
) -> PortfolioResult:
    """
    Convenience wrapper for real use: takes `signaled`, the output of
    run_real_backtest.build_signaled_universe() (full feature columns per
    ticker), computes the technical-only action at every bar, and runs
    the shared-capital simulation. horizon_bars defaults to the same 5
    trading days (scaled by config.BARS_PER_DAY) run_real_backtest.py and
    walk_forward_backtest.py both use, for a fair comparison. See
    simulate_portfolio_from_actions()'s docstring for entry_actions.

    benchmark_bars: optional DataFrame with `timestamp`/`close` columns
    (e.g. run_real_backtest.load_ticker(config.BENCHMARK)'s own SPY bars)
    -- pass this to park idle capital in the benchmark instead of flat
    cash between trades (see the module docstring). Omit it to keep the
    original flat-cash behavior.
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

    benchmark_prices = None
    if benchmark_bars is not None and not benchmark_bars.empty:
        benchmark_prices = benchmark_bars.set_index("timestamp")["close"].sort_index()

    return simulate_portfolio_from_actions(
        action_frames,
        starting_capital=starting_capital,
        max_concurrent=max_concurrent,
        horizon_bars=horizon_bars,
        entry_actions=entry_actions,
        benchmark_prices=benchmark_prices,
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
