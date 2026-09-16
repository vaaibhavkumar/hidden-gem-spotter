"""
Level 1 tests for recommendation/html_report.py — the self-contained
report.html builder. These check the generated markup contains the right
structural pieces (action badges, per-ticker anchors, history rows), not
exact byte-for-byte HTML, since the styling is expected to keep evolving.
"""
from __future__ import annotations

import pandas as pd
import pytest

from recommendation import html_report
from recommendation.recommend import Recommendation


def _rec(ticker="TEST", score=72.5, action="BUY"):
    return Recommendation(
        ticker=ticker,
        timestamp=pd.Timestamp("2024-06-01"),
        price=123.45,
        action=action,
        composite_score=score,
        pillars_used=["technical"],
        confidence_pct=61.0,
        confidence_range=(0.5, 0.7),
        confidence_note="from backtest bucket n=40",
        reasoning=["[Technical/bullish] example reason"],
    )


def test_generate_html_report_writes_file_with_ticker_and_action(tmp_path):
    out = tmp_path / "report.html"
    path = html_report.generate_html_report([_rec()], out_path=str(out))

    assert path == str(out)
    text = out.read_text()
    assert "TEST" in text
    assert "BUY" in text
    assert 'id="t-test"' in text


def test_generate_html_report_sorts_by_composite_score_descending(tmp_path):
    out = tmp_path / "report.html"
    recs = [_rec("LOW", score=10.0, action="SELL"), _rec("HIGH", score=90.0, action="STRONG BUY")]
    html_report.generate_html_report(recs, out_path=str(out))

    text = out.read_text()
    # The summary table should list HIGH before LOW.
    assert text.index(">HIGH<") < text.index(">LOW<")


def test_generate_html_report_without_history_shows_placeholder(tmp_path):
    out = tmp_path / "report.html"
    html_report.generate_html_report([_rec()], out_path=str(out))
    text = out.read_text()
    assert "No prior runs logged yet" in text


def test_generate_html_report_with_history_renders_past_rows(tmp_path):
    out = tmp_path / "report.html"
    history_df = pd.DataFrame(
        [
            {
                "ticker": "TEST", "timestamp": pd.Timestamp("2024-05-01"), "action": "HOLD",
                "composite_score": 55.0, "pillars_used": "technical", "confidence_pct": None,
                "confidence_lo": None, "confidence_hi": None, "confidence_note": "uncalibrated",
                "reasoning": "", "price": 100.0, "logged_at": pd.Timestamp("2024-05-01"),
            },
            {
                "ticker": "TEST", "timestamp": pd.Timestamp("2024-06-01"), "action": "BUY",
                "composite_score": 72.5, "pillars_used": "technical", "confidence_pct": 61.0,
                "confidence_lo": 0.5, "confidence_hi": 0.7, "confidence_note": "from backtest bucket n=40",
                "reasoning": "", "price": 123.45, "logged_at": pd.Timestamp("2024-06-01"),
            },
        ]
    )
    html_report.generate_html_report([_rec()], history={"TEST": history_df}, out_path=str(out))

    text = out.read_text()
    assert "table history" not in text  # sanity: class attr, not literal text
    assert "class=\"history\"" in text
    assert "HOLD" in text  # the older row shows up
    # The most recent row (2024-06-01) is excluded since it's the same bar
    # already shown as the main card, not duplicated in the history table.
    assert text.count("2024-06-01") == 1  # only in the card meta, not the history table
