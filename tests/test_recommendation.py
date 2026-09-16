"""
Level 1 tests for recommendation/recommend.py — the Action/Score/
Confidence/Reasoning output the user explicitly asked this project to
produce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommendation import recommend


@pytest.mark.parametrize(
    "score,expected_action",
    [
        (100, "STRONG BUY"),
        (85, "STRONG BUY"),
        (84.9, "BUY"),
        (70, "BUY"),
        (69.9, "HOLD"),
        (40, "HOLD"),
        (39.9, "SELL"),
        (25, "SELL"),
        (24.9, "STRONG SELL"),
        (0, "STRONG SELL"),
    ],
)
def test_action_thresholds(score, expected_action):
    assert recommend._action_for_score(score) == expected_action


def _neutral_row():
    # No bullish or bearish condition fires -> tilt = 0 -> technical score
    # should land exactly on the 50 (neutral) midpoint.
    return pd.Series(
        {
            "timestamp": pd.Timestamp("2024-06-01"),
            "close": 123.45,
            "trend_template_bull": False,
            "trend_template_bear": False,
            "rs_percentile": 0.5,
            "volume_z": 0.0,
            "vol_contraction_ratio": 1.0,
            "new_52w_high": False,
            "new_52w_low": False,
            "roc_acceleration": 0.0,
        }
    )


def _bullish_row():
    row = _neutral_row()
    row["trend_template_bull"] = True
    row["rs_percentile"] = 0.9
    row["volume_z"] = 2.0
    row["vol_contraction_ratio"] = 0.5
    row["new_52w_high"] = True
    row["roc_acceleration"] = 0.02
    return row


def test_recommend_with_only_technical_pillar_uses_it_unweighted():
    row = _bullish_row()
    rec = recommend.recommend(row, ticker="TEST")

    assert rec.pillars_used == ["technical"]
    # With only one pillar supplied, the renormalized weighted average is
    # just that pillar's own score.
    tech_score, _ = recommend._technical_pillar_score(row)
    assert rec.composite_score == pytest.approx(tech_score)
    assert rec.action == recommend._action_for_score(tech_score)
    assert any("Pillars not yet available" in r for r in rec.reasoning)


def test_recommend_neutral_row_scores_fifty_and_holds():
    row = _neutral_row()
    rec = recommend.recommend(row, ticker="TEST")
    assert rec.composite_score == pytest.approx(50.0)
    assert rec.action == "HOLD"


def test_recommend_renormalizes_weights_across_supplied_pillars():
    row = _neutral_row()  # technical score fixed at 50
    rec = recommend.recommend(row, ticker="TEST", fundamental_score=80.0)

    w_tech = recommend.PILLAR_WEIGHTS["technical"]
    w_fund = recommend.PILLAR_WEIGHTS["fundamental"]
    expected = 50.0 * (w_tech / (w_tech + w_fund)) + 80.0 * (w_fund / (w_tech + w_fund))

    assert set(rec.pillars_used) == {"technical", "fundamental"}
    assert rec.composite_score == pytest.approx(expected)
    # Only revision/alternative are still missing now.
    missing_note = [r for r in rec.reasoning if "Pillars not yet available" in r][0]
    assert "revision" in missing_note and "alternative" in missing_note
    assert "fundamental" not in missing_note.split(":")[1]


def test_recommend_with_revision_score_folds_it_into_composite():
    row = _neutral_row()  # technical score fixed at 50
    rec = recommend.recommend(row, ticker="TEST", revision_score=90.0)

    w_tech = recommend.PILLAR_WEIGHTS["technical"]
    w_rev = recommend.PILLAR_WEIGHTS["revision"]
    expected = 50.0 * (w_tech / (w_tech + w_rev)) + 90.0 * (w_rev / (w_tech + w_rev))

    assert set(rec.pillars_used) == {"technical", "revision"}
    assert rec.composite_score == pytest.approx(expected)
    missing_note = [r for r in rec.reasoning if "Pillars not yet available" in r][0]
    assert "fundamental" in missing_note and "alternative" in missing_note
    assert "revision" not in missing_note.split(":")[1]


def test_recommend_without_calibration_table_reports_uncalibrated():
    row = _bullish_row()
    rec = recommend.recommend(row, ticker="TEST")
    assert rec.confidence_pct is None
    assert rec.confidence_range is None
    assert "no calibration table supplied" in rec.confidence_note


def test_recommend_with_calibration_table_looks_up_confidence():
    row = _bullish_row()  # composite >= 50 -> "bull" direction lookup
    calibration = pd.DataFrame(
        [
            {"direction": "bull", "score_bucket": 0, "n": 20, "hit_rate": 0.75, "ci_low": 0.5, "ci_high": 0.9},
            {"direction": "bear", "score_bucket": 0, "n": 5, "hit_rate": 0.4, "ci_low": 0.1, "ci_high": 0.7},
        ]
    )
    rec = recommend.recommend(row, ticker="TEST", calibration_table=calibration)
    assert rec.confidence_pct == pytest.approx(75.0)
    assert rec.confidence_range == (0.5, 0.9)


def test_recommend_reports_missing_direction_in_calibration_table():
    row = _neutral_row()
    row["trend_template_bear"] = True  # tips composite below 50 -> "bear" lookup
    calibration = pd.DataFrame(
        [{"direction": "bull", "score_bucket": 0, "n": 20, "hit_rate": 0.75, "ci_low": 0.5, "ci_high": 0.9}]
    )
    rec = recommend.recommend(row, ticker="TEST", calibration_table=calibration)
    assert rec.confidence_pct is None
    assert "no backtested bear signals" in rec.confidence_note
