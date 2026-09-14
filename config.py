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

# --- Feature windows (in bars; use trading DAYS for a daily pipeline, or scale up
# by ~7x for hourly-during-market-hours bars, per the proposal's 3.1 data layer) ---
SMA_SHORT = 50
SMA_MID = 150
SMA_LONG = 200
LOOKBACK_52W = 252          # ~1 trading year of daily bars
VOL_WINDOW = 20             # realized-volatility window
VOL_BASELINE_WINDOW = 60    # baseline to compare current volatility against (contraction score)
VOLUME_WINDOW = 20          # trailing average volume window for the volume z-score
ROC_SHORT = 20
ROC_LONG = 60
RS_WINDOWS = (63, 126, 252)  # ~3m, 6m, 12m in trading days

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
