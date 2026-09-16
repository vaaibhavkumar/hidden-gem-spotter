"""
Level 1 tests for ingestion/finnhub_ingest.py's pure scoring logic
(_consensus_to_score). No network calls here — fetch_recommendation_trends()
itself is a thin requests.get() wrapper not worth mocking extensively;
the real value to test is "does a raw Finnhub response map to the right
0-100 number."
"""
from __future__ import annotations

from ingestion import finnhub_ingest


def test_consensus_to_score_all_strong_buy_is_100():
    period = {"strongBuy": 10, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
    assert finnhub_ingest._consensus_to_score(period) == 100.0


def test_consensus_to_score_all_strong_sell_is_zero():
    period = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 10}
    assert finnhub_ingest._consensus_to_score(period) == 0.0


def test_consensus_to_score_all_hold_is_fifty():
    period = {"strongBuy": 0, "buy": 0, "hold": 7, "sell": 0, "strongSell": 0}
    assert finnhub_ingest._consensus_to_score(period) == 50.0


def test_consensus_to_score_is_weighted_by_analyst_count():
    # 24 buy + 7 hold + 0 sell, lopsided toward buy -- should land well above 50.
    period = {"strongBuy": 13, "buy": 24, "hold": 7, "sell": 0, "strongSell": 0}
    score = finnhub_ingest._consensus_to_score(period)
    assert 75 < score < 100


def test_consensus_to_score_no_coverage_returns_none():
    period = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
    assert finnhub_ingest._consensus_to_score(period) is None


def test_consensus_to_score_missing_keys_treated_as_zero():
    # Finnhub's real payloads always include all five keys, but don't
    # assume that forever -- a partial dict shouldn't raise.
    period = {"buy": 5, "hold": 5}
    score = finnhub_ingest._consensus_to_score(period)
    assert score == 62.5  # (5*75 + 5*50) / 10
