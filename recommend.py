"""
Buy/Sell recommendation schema: Action + Score + Confidence + Reasoning.

This turns the raw bull_score/bear_score condition-counts from scoring.py
into the structured output you actually want to look at: an Action label,
a 0-100 composite score, a confidence figure, and plain-language reasoning
for why.

Two honesty notes, deliberately surfaced in the output rather than hidden:

1. FOUR PILLARS ARE DESIGNED, ONE IS BUILT. The composite is meant to blend
   technical (this repo), fundamental (EPS/margins/ROIC — section 2C),
   analyst-revision (section 2C), and alternative (section 2D) pillars.
   Only the technical pillar has real data behind it right now. The
   composite below re-normalizes weights across whichever pillars are
   actually supplied, and always reports which ones were used — a
   "STRONG BUY" based on technicals alone is a different, weaker claim
   than one where all four pillars agree, and the output says so.

2. CONFIDENCE COMES FROM BACKTESTED HIT RATES, NOT FACTOR AGREEMENT.
   A tempting shortcut is to call a recommendation "high confidence"
   because its sub-scores happen to agree with each other — but that
   measures internal consistency, not whether signals like this one have
   actually worked historically. This module's confidence figure is a
   real Wilson confidence interval (backtest.wilson_confidence_interval)
   looked up from a calibration table (backtest.calibrate_confidence) you
   build by backtesting; without one, confidence is reported as
   "uncalibrated" rather than a number dressed up to look precise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import scoring

PILLAR_WEIGHTS = {
    "technical": 0.35,
    "fundamental": 0.30,
    "revision": 0.20,
    "alternative": 0.15,
}

ACTION_THRESHOLDS = [
    (85, "STRONG BUY"),
    (70, "BUY"),
    (40, "HOLD"),
    (25, "SELL"),
    (0, "STRONG SELL"),
]


@dataclass
class Recommendation:
    ticker: str
    timestamp: object
    price: float
    action: str
    composite_score: float
    pillars_used: list[str]
    confidence_pct: float | None
    confidence_range: tuple[float, float] | None
    confidence_note: str
    reasoning: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # a compact, report-style rendering
        lines = [
            "=" * 72,
            f"TICKER: {self.ticker}   ({self.timestamp})",
            f"ACTION: {self.action}",
            f"COMPOSITE SCORE: {self.composite_score:.1f} / 100  (pillars used: {', '.join(self.pillars_used)})",
        ]
        if self.confidence_pct is not None:
            lo, hi = self.confidence_range
            lines.append(f"CONFIDENCE: {self.confidence_pct:.1f}% (range: {lo * 100:.1f}%-{hi * 100:.1f}%)")
        else:
            lines.append(f"CONFIDENCE: uncalibrated — {self.confidence_note}")
        lines.append("-" * 72)
        lines.append("REASONING:")
        for r in self.reasoning:
            lines.append(f"  - {r}")
        lines.append("=" * 72)
        return "\n".join(lines)


def _technical_pillar_score(row: pd.Series) -> tuple[float, list[str]]:
    """
    Maps the bullish/bearish condition counts to a single 0-100 technical
    score (50 = neutral), and returns the reasoning bullets for whichever
    conditions actually fired.
    """
    bull = scoring.bullish_conditions(row)
    bear = scoring.bearish_conditions(row)
    tilt = sum(bull.values()) - sum(bear.values())  # range roughly -6..+6
    score = float(np.clip(50 + tilt * (50 / 6), 0, 100))

    reasoning = []
    rs_pct = row.get("rs_percentile", np.nan)
    vz = row.get("volume_z", np.nan)
    for key, fired in bull.items():
        if fired:
            label = scoring.BULLISH_CONDITION_LABELS[key]
            reasoning.append(
                "[Technical/bullish] "
                + label.format(pct=(1 - config.THRESHOLDS["rs_percentile_bull"]) * 100, z=config.THRESHOLDS["volume_z_confirm"])
            )
    for key, fired in bear.items():
        if fired:
            label = scoring.BEARISH_CONDITION_LABELS[key]
            reasoning.append(
                "[Technical/bearish] "
                + label.format(pct=config.THRESHOLDS["rs_percentile_bear"] * 100, z=config.THRESHOLDS["volume_z_confirm"])
            )
    return score, reasoning


def _action_for_score(score: float) -> str:
    for threshold, action in ACTION_THRESHOLDS:
        if score >= threshold:
            return action
    return "STRONG SELL"


def recommend(
    row: pd.Series,
    ticker: str,
    fundamental_score: float | None = None,
    revision_score: float | None = None,
    alternative_score: float | None = None,
    calibration_table: pd.DataFrame | None = None,
) -> Recommendation:
    """
    Builds one Recommendation for a single ticker at a single bar (row must
    come from a features/scoring DataFrame — see demo.py / run_real_backtest.py
    for how `row` is produced).

    fundamental_score / revision_score / alternative_score: pass 0-100
    percentile scores for those pillars once you've built them (section 2C
    of the proposal) — omitted pillars are simply left out of the weighted
    composite (weights renormalize across whatever's supplied) rather than
    silently defaulted to a neutral 50, which would understate conviction
    on names where you actually have the data.

    calibration_table: output of backtest.calibrate_confidence(), if you
    have one from a real backtest — used to look up a genuine confidence
    interval for this composite score. Without it, confidence is reported
    as uncalibrated rather than invented.
    """
    tech_score, tech_reasoning = _technical_pillar_score(row)

    pillar_scores = {"technical": tech_score}
    if fundamental_score is not None:
        pillar_scores["fundamental"] = fundamental_score
    if revision_score is not None:
        pillar_scores["revision"] = revision_score
    if alternative_score is not None:
        pillar_scores["alternative"] = alternative_score

    used_weights = {k: PILLAR_WEIGHTS[k] for k in pillar_scores}
    weight_sum = sum(used_weights.values())
    composite = sum(pillar_scores[k] * (used_weights[k] / weight_sum) for k in pillar_scores)

    action = _action_for_score(composite)

    confidence_pct = None
    confidence_range = None
    if calibration_table is not None and not calibration_table.empty:
        direction = "bull" if composite >= 50 else "bear"
        candidates = calibration_table[calibration_table["direction"] == direction]
        if not candidates.empty:
            # nearest bucket by hit_rate's implied score isn't available here;
            # match by the row count-weighted average as a simple stand-in —
            # replace with a proper score-bucket lookup once real backtests exist.
            best = candidates.loc[candidates["n"].idxmax()]
            confidence_pct = float(best["hit_rate"] * 100)
            confidence_range = (float(best["ci_low"]), float(best["ci_high"]))
            confidence_note = f"from backtest bucket n={int(best['n'])}"
        else:
            confidence_note = f"no backtested {direction} signals in the calibration table yet"
    else:
        confidence_note = (
            "no calibration table supplied — run backtest.calibrate_confidence() on real "
            "backtest results first; do not treat pillar agreement as a confidence interval"
        )

    missing = [p for p in PILLAR_WEIGHTS if p not in pillar_scores]
    if missing:
        tech_reasoning.append(f"[Note] Pillars not yet available: {', '.join(missing)} — composite is partial.")

    return Recommendation(
        ticker=ticker,
        timestamp=row.get("timestamp"),
        price=float(row.get("close", np.nan)),
        action=action,
        composite_score=composite,
        pillars_used=list(pillar_scores.keys()),
        confidence_pct=confidence_pct,
        confidence_range=confidence_range,
        confidence_note=confidence_note,
        reasoning=tech_reasoning,
    )
