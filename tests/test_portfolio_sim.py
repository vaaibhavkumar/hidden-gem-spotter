"""
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


def _spy_prices(prices: list[float], start="2024-01-01") -> pd.Series:
    ts = pd.date_range(start, periods=len(prices), freq="D")
    return pd.Series(prices, index=ts)


def test_without_benchmark_prices_idle_capital_stays_flat_between_trades():
    # Default (unchanged) behavior: no benchmark_prices means no overlay,
    # and the result says so via benchmark_overlay.
    frames = {"AAA": _bars([("BUY", 100.0), ("HOLD", 100.0), ("SELL", 110.0)])}
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10)
    assert result.benchmark_overlay is False


def test_benchmark_overlay_flag_is_set_when_prices_supplied():
    frames = {"AAA": _bars([("HOLD", 100.0), ("HOLD", 100.0)])}
    spy = _spy_prices([200.0, 202.0])
    result = simulate_portfolio_from_actions(
        frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10, benchmark_prices=spy
    )
    assert result.benchmark_overlay is True


def test_idle_capital_tracks_benchmark_when_nothing_is_ever_bought():
    # No stock ever gets bought (all HOLD) -- with the overlay, 100% of
    # capital should ride the benchmark's own return over the period.
    frames = {"AAA": _bars([("HOLD", 100.0), ("HOLD", 100.0), ("HOLD", 100.0)])}
    spy = _spy_prices([200.0, 210.0, 220.0])  # +10% over the window
    result = simulate_portfolio_from_actions(
        frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10, benchmark_prices=spy
    )
    assert result.n_trades == 0
    assert result.ending_equity == pytest.approx(1000.0 * (220.0 / 200.0))


def test_capital_committed_to_a_stock_pick_stops_tracking_the_benchmark():
    # BUY immediately (so ~all capital leaves the benchmark on day 0),
    # hold flat, then SELL flat -- ending equity should reflect the flat
    # stock trade, NOT the benchmark's move during the holding window,
    # since that capital was committed to the stock the whole time.
    frames = {"AAA": _bars([("BUY", 100.0), ("HOLD", 100.0), ("SELL", 100.0)])}
    spy = _spy_prices([200.0, 400.0, 600.0])  # benchmark triples while the stock trade is flat
    result = simulate_portfolio_from_actions(
        frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10, benchmark_prices=spy
    )
    assert result.n_trades == 1
    assert result.trades[0].return_pct == pytest.approx(0.0)
    assert result.ending_equity == pytest.approx(1000.0)  # flat trade, capital fully committed throughout


def test_capital_returns_to_tracking_benchmark_after_a_position_closes():
    # BUY and SELL flat on day 0->1 (committed briefly), then the
    # remaining days are idle -- idle capital should track the benchmark
    # for whatever time it's NOT committed to the stock.
    frames = {"AAA": _bars([("BUY", 100.0), ("SELL", 100.0), ("HOLD", 100.0)])}
    spy = _spy_prices([200.0, 200.0, 220.0])  # flat while position is open, +10% after it closes
    result = simulate_portfolio_from_actions(
        frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10, benchmark_prices=spy
    )
    assert result.ending_equity == pytest.approx(1000.0 * (220.0 / 200.0))


def test_benchmark_prices_starting_after_first_event_raises_value_error():
    # The benchmark series must cover the very first event's timestamp so
    # the initial allocation can be priced -- one starting later (no
    # price at-or-before the first event) should fail loudly, not
    # silently mis-size the simulation's starting position.
    frames = {"AAA": _bars([("HOLD", 100.0)], start="2024-06-01")}
    spy_too_late = _spy_prices([200.0], start="2030-01-01")
    with pytest.raises(ValueError):
        simulate_portfolio_from_actions(
            frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10, benchmark_prices=spy_too_late
        )


def test_simulate_portfolio_wrapper_passes_through_benchmark_bars():
    ts = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame([_neutral_row(ts[0]), _neutral_row(ts[1]), _neutral_row(ts[2])])
    signaled = {"AAA": df}
    benchmark_bars = pd.DataFrame({"timestamp": ts, "close": [200.0, 205.0, 210.0]})

    result = portfolio_sim.simulate_portfolio(
        signaled, starting_capital=1000.0, max_concurrent=1, horizon_bars=1, benchmark_bars=benchmark_bars
    )
    assert result.benchmark_overlay is True


def test_average_utilization_matches_hand_computed_occupancy():
    # Open for the full 2-day window (day0 -> day2, exiting via sell
    # signal on day2) at 1 of 2 max_concurrent slots -> 50% utilization
    # the entire time the simulation spans.
    frames = {"AAA": _bars([("BUY", 100.0), ("HOLD", 100.0), ("SELL", 110.0)])}
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=2, horizon_bars=10)
    assert result.average_utilization() == pytest.approx(0.5)


def test_average_utilization_is_zero_when_nothing_ever_opens():
    frames = {"AAA": _bars([("HOLD", 100.0), ("HOLD", 100.0)])}
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=2, horizon_bars=10)
    assert result.average_utilization() == pytest.approx(0.0)


def test_average_utilization_is_none_with_fewer_than_two_events():
    frames = {"AAA": _bars([("HOLD", 100.0)])}
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=2, horizon_bars=10)
    assert result.average_utilization() is None


def test_avg_trade_return_matches_hand_computed_mean():
    frames = {
        "AAA": _bars([("BUY", 100.0), ("SELL", 110.0)]),  # +10%
        "BBB": _bars([("BUY", 100.0), ("SELL", 90.0)]),   # -10%
    }
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=2, horizon_bars=10)
    assert result.avg_trade_return == pytest.approx(0.0)


def test_avg_trade_return_is_none_with_no_trades():
    result = simulate_portfolio_from_actions({}, starting_capital=1000.0)
    assert result.avg_trade_return is None


def test_entry_actions_defaults_to_buy_and_strong_buy():
    frames = {"AAA": _bars([("BUY", 100.0), ("SELL", 110.0)])}
    result = simulate_portfolio_from_actions(frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10)
    assert result.n_trades == 1
    assert result.entry_actions == frozenset({"BUY", "STRONG BUY"})


def test_entry_actions_can_be_restricted_to_strong_buy_only():
    # A plain BUY should be ignored when entry_actions excludes it -- the
    # conviction-filtering experiment tried after the first real run came
    # in well under target (see portfolio_backtest.py's ENTRY_ACTIONS).
    frames = {"AAA": _bars([("BUY", 100.0), ("HOLD", 105.0), ("SELL", 110.0)])}
    result = simulate_portfolio_from_actions(
        frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10, entry_actions={"STRONG BUY"}
    )
    assert result.n_trades == 0
    assert result.ending_equity == pytest.approx(1000.0)


def test_entry_actions_restricted_to_strong_buy_still_opens_on_strong_buy():
    frames = {"AAA": _bars([("STRONG BUY", 100.0), ("HOLD", 105.0), ("SELL", 110.0)])}
    result = simulate_portfolio_from_actions(
        frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10, entry_actions={"STRONG BUY"}
    )
    assert result.n_trades == 1
    assert result.trades[0].return_pct == pytest.approx(0.10)


def test_sell_still_exits_an_open_position_regardless_of_entry_actions_filter():
    # entry_actions only gates what can OPEN a position -- SELL/STRONG
    # SELL must still close an existing one even if "SELL" isn't (and
    # never would be) in entry_actions.
    frames = {"AAA": _bars([("STRONG BUY", 100.0), ("SELL", 90.0)])}
    result = simulate_portfolio_from_actions(
        frames, starting_capital=1000.0, max_concurrent=1, horizon_bars=10, entry_actions={"STRONG BUY"}
    )
    assert result.n_trades == 1
    assert result.trades[0].exit_reason == "sell_signal"


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
