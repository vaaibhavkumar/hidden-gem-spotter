"""
Hidden Gem Spotter — configuration.

Holds the S&P 500 (point-in-time) universe placeholder, the section-6 validation
universe from the project proposal, and the thresholds used by the composite
scoring engine in scoring.py.
"""

# --- Section 6 validation universe (see project doc "early-momentum-detection-system-proposal.md") ---
# role: "riser" | "faller" | "normal"
VALIDATION_UNIVERSE = {
    # Rose a lot — test the bullish screen
    "NVDA": {"role": "riser", "note": "AI-compute mega-winner, multi-leg advance"},
    "PLTR": {"role": "riser", "note": "Joined S&P 500 Sept 2024, continued advance"},
    "VST":  {"role": "riser", "note": "AI/data-center power demand theme, +264% reported"},
    "CEG":  {"role": "riser", "note": "Nuclear power, AI power-demand theme"},
    "GEV":  {"role": "riser", "note": "GE power/grid spinoff, short trading history"},

    # Fell a lot — test the bearish screen
    "INTC": {"role": "faller", "note": "Slow multi-year competitive decline"},
    "BA":   {"role": "faller", "note": "Event-driven decline (2024 safety crisis)"},
    "MRNA": {"role": "faller", "note": "Post-COVID demand cliff"},
    "ENPH": {"role": "faller", "note": "Solar slowdown; removed from S&P 500 Sept 2025"},
    "EL":   {"role": "faller", "note": "China/travel-retail weakness, guidance cuts"},

    # Normal / control group — estimate false-positive rate
    "JNJ": {"role": "normal", "note": "Stable large-cap healthcare"},
    "KO":  {"role": "normal", "note": "Steady consumer staple"},
    "PG":  {"role": "normal", "note": "Steady consumer staple"},
    "V":   {"role": "normal", "note": "Steady large-cap compounder"},
    "HD":  {"role": "normal", "note": "Large-cap, no extreme story either way"},
}

BENCHMARK = "SPY"

# --- Bar frequency ---
# alpaca_ingest.py currently pulls HOURLY bars (matching the original "track
# hourly" goal), but every window below is defined in TRADING DAYS and then
# scaled by BARS_PER_DAY — so the same config.py works whether data/*.csv
# holds daily or hourly bars. Just set this to match what you actually
# fetched before running run_real_backtest.py:
#   daily bars  -> BARS_PER_DAY = 1
#   hourly bars -> BARS_PER_DAY = 7   (~6.5 regular-hours trading day rounds
#                                       up to 7 hourly bars with Alpaca's
#                                       hour-aligned aggregation)
# Getting this wrong doesn't error out — it silently makes "50-day" mean 50
# *bars*, i.e. ~7 trading days on hourly data — a 7x-too-short moving
# average that will over-fire signals. Always double check this matches
# your data before trusting results.
def set_bars_per_day(n: int) -> None:
    """
    Sets BARS_PER_DAY and recomputes every window constant below from it.
    Call this (instead of assigning config.BARS_PER_DAY directly) whenever
    you switch between daily and hourly data — e.g. run_real_backtest.py
    calls it automatically after detecting the cadence from your CSVs.
    """
    global BARS_PER_DAY, SMA_SHORT, SMA_MID, SMA_LONG, LOOKBACK_52W
    global VOL_WINDOW, VOL_BASELINE_WINDOW, VOLUME_WINDOW, ROC_SHORT, ROC_LONG, RS_WINDOWS

    BARS_PER_DAY = n
    SMA_SHORT = 50 * n
    SMA_MID = 150 * n
    SMA_LONG = 200 * n
    LOOKBACK_52W = 252 * n          # ~1 trading year
    VOL_WINDOW = 20 * n             # realized-volatility window
    VOL_BASELINE_WINDOW = 60 * n    # baseline to compare current volatility against (contraction score)
    VOLUME_WINDOW = 20 * n          # trailing average volume window for the volume z-score
    ROC_SHORT = 20 * n
    ROC_LONG = 60 * n
    RS_WINDOWS = tuple(d * n for d in (63, 126, 252))  # ~3m, 6m, 12m


set_bars_per_day(1)  # default: daily bars. Call set_bars_per_day(7) for hourly data.

# --- Composite scoring thresholds (section 2E/2F/3.4 of the proposal) ---
# These are starting points for the checklist version — tune during backtesting (3.5).
THRESHOLDS = {
    "rs_percentile_bull": 0.70,   # top 30% of the universe by relative strength
    "rs_percentile_bear": 0.30,   # bottom 30%
    "distance_from_high_bull_max": -0.25,  # within 25% of the 52-week high
    "distance_from_low_bear_max": 0.25,    # within 25% of the 52-week low
    "volume_z_confirm": 1.0,       # volume at least ~1 std above its trailing average
    "vol_contraction_max": 0.85,   # current vol / baseline vol below this = "contracting"
    "signal_score_min": 3,         # minimum number of independent conditions that must fire
}
