from pathlib import Path

content_recommendation_html_report_py = r'''"""
Renders a list of recommend.Recommendation objects as a single
self-contained HTML file: a sortable-by-eye "watchlist" summary table up
top (one row per ticker, action/score/confidence/top reason), then a
detail card per ticker below with the full reasoning.

Deliberately plain, dependency-free HTML/CSS in one file (no Jinja, no
external assets) so `open report.html` in any browser just works, same
philosophy as the rest of this project (no server, no build step).
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd

from recommendation.recommend import Recommendation

# Action -> (background, text) colors for the badge. Chosen for readability
# in both a plain white background and don't rely on color alone — the
# action text itself ("STRONG BUY" etc.) is always present too.
_ACTION_COLORS = {
    "STRONG BUY": ("#0f5132", "#d1f5e0"),
    "BUY": ("#1e7e42", "#e3f9ec"),
    "HOLD": ("#5a5a5a", "#eeeeee"),
    "SELL": ("#8a1f1f", "#fbe4e4"),
    "STRONG SELL": ("#5c0d0d", "#f6cfcf"),
}


def _badge(action: str) -> str:
    fg, bg = _ACTION_COLORS.get(action, ("#333", "#eee"))
    return f'<span class="badge" style="color:{fg};background:{bg};">{html.escape(action)}</span>'


def _confidence_text(rec: Recommendation) -> str:
    if rec.confidence_pct is not None:
        lo, hi = rec.confidence_range
        return f"{rec.confidence_pct:.1f}% <span class='muted'>({lo * 100:.1f}%–{hi * 100:.1f}%)</span>"
    return "<span class='muted'>uncalibrated</span>"


def _row_id(ticker: str) -> str:
    return f"t-{ticker.lower()}"


def _history_table(df: pd.DataFrame | None) -> str:
    """
    Renders a ticker's past recommendations (data_store.load_recommendation_history()'s
    output), most recent first, as a small table -- this is the "see
    current AND previous buy/sell/hold recommendations" view, sourced from
    the `recommendations` table rather than recomputed. Excludes the most
    recent row, since that's already shown as this ticker's main card.
    """
    if df is None or df.empty or len(df) <= 1:
        return "<p class='muted'>No prior runs logged yet for this ticker -- history will " \
               "build up as you re-run run_real_backtest.py over time.</p>"

    past = df.sort_values("timestamp", ascending=False).iloc[1:].head(20)  # drop latest, cap at 20 rows
    rows = []
    for _, r in past.iterrows():
        conf = f"{r['confidence_pct']:.1f}%" if pd.notna(r["confidence_pct"]) else "uncalibrated"
        rows.append(
            f"<tr><td>{html.escape(str(r['timestamp']))}</td>"
            f"<td>{_badge(r['action'])}</td>"
            f"<td class='num'>{r['composite_score']:.1f}</td>"
            f"<td>{html.escape(conf)}</td>"
            f"<td class='num'>${r['price']:,.2f}</td></tr>"
        )
    return f"""
    <table class="history">
      <thead><tr><th>As of</th><th>Action</th><th>Score</th><th>Confidence</th><th>Price</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def generate_html_report(
    recommendations: list[Recommendation],
    roles: dict[str, str] | None = None,
    history: dict[str, pd.DataFrame] | None = None,
    out_path: str = "report.html",
) -> str:
    """
    recommendations: one Recommendation per ticker (e.g. the output of
        recommend.recommend() for each ticker's latest bar — see
        run_real_backtest.py for how these get built).
    roles: optional {ticker: "riser"/"faller"/"normal"} from
        config.VALIDATION_UNIVERSE, shown as a column if given.
    history: optional {ticker: DataFrame} from
        data_store.load_recommendation_history(ticker) — when given, each
        ticker's detail card gets a "Previous recommendations" table below
        its reasoning, sourced from the persistent `recommendations` table
        rather than just this run's in-memory result.
    out_path: where to write the HTML file (relative to cwd unless
        absolute). Returns the path written.
    """
    roles = roles or {}
    history = history or {}

    # Sort like a watchlist: highest-conviction buys at the top, sells at
    # the bottom, so scrolling the summary table reads top-to-bottom as
    # "most bullish -> most bearish" rather than alphabetical.
    ordered = sorted(recommendations, key=lambda r: r.composite_score, reverse=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    summary_rows = []
    for rec in ordered:
        role = roles.get(rec.ticker, "")
        top_reason = next((r for r in rec.reasoning if not r.startswith("[Note]")), "—")
        summary_rows.append(
            f"""
            <tr>
              <td><a href="#{_row_id(rec.ticker)}"><strong>{html.escape(rec.ticker)}</strong></a></td>
              <td class="muted">{html.escape(role)}</td>
              <td>{_badge(rec.action)}</td>
              <td class="num">{rec.composite_score:.1f}</td>
              <td>{_confidence_text(rec)}</td>
              <td class="num">${rec.price:,.2f}</td>
              <td class="reason">{html.escape(top_reason)}</td>
            </tr>"""
        )

    detail_cards = []
    for rec in ordered:
        role = roles.get(rec.ticker, "")
        role_txt = f" &middot; {html.escape(role)}" if role else ""
        reasoning_items = "".join(f"<li>{html.escape(r)}</li>" for r in rec.reasoning) or "<li class='muted'>No conditions fired.</li>"
        ts = rec.timestamp if isinstance(rec.timestamp, str) else str(rec.timestamp)
        detail_cards.append(
            f"""
            <section class="card" id="{_row_id(rec.ticker)}">
              <div class="card-head">
                <h2>{html.escape(rec.ticker)}{role_txt}</h2>
                {_badge(rec.action)}
              </div>
              <div class="card-meta">
                <span><strong>Score:</strong> {rec.composite_score:.1f} / 100 (pillars used: {html.escape(', '.join(rec.pillars_used))})</span>
                <span><strong>Confidence:</strong> {_confidence_text(rec)}</span>
                <span><strong>Price:</strong> ${rec.price:,.2f} <span class="muted">as of {html.escape(ts)}</span></span>
              </div>
              <ul class="reasoning">{reasoning_items}</ul>
              <h3 class="history-heading">Previous recommendations</h3>
              {_history_table(history.get(rec.ticker))}
            </section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hidden Gem Spotter — Recommendations</title>
<style>
  :root {{
    --bg: #fafafa; --fg: #1a1a1a; --muted: #6b6b6b; --border: #e2e2e2; --card-bg: #ffffff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px 20px 60px; background: var(--bg); color: var(--fg);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-bottom: 24px; }}
  .disclaimer {{
    font-size: 12.5px; color: var(--muted); background: #f2f2f2; border: 1px solid var(--border);
    border-radius: 6px; padding: 10px 14px; margin-bottom: 24px;
  }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13.5px; }}
  th {{ background: #f2f2f2; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; color: var(--muted); }}
  tr:last-child td {{ border-bottom: none; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.reason {{ color: #333; max-width: 360px; }}
  a {{ color: #1a4fa0; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .muted {{ color: var(--muted); font-size: 12.5px; }}
  .badge {{ display: inline-block; padding: 3px 9px; border-radius: 12px; font-size: 12px; font-weight: 600; white-space: nowrap; }}
  .card {{
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px 18px; margin: 14px 0; scroll-margin-top: 16px;
  }}
  .card-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
  .card-head h2 {{ font-size: 16px; margin: 0; }}
  .card-meta {{ display: flex; flex-wrap: wrap; gap: 6px 20px; color: #333; font-size: 13px; margin: 10px 0 8px; }}
  .reasoning {{ margin: 8px 0 0; padding-left: 20px; font-size: 13.5px; }}
  .reasoning li {{ margin: 3px 0; }}
  section#detail h1 {{ margin-top: 40px; }}
  .history-heading {{ font-size: 12px; text-transform: uppercase; letter-spacing: .03em; color: var(--muted); margin: 16px 0 6px; }}
  table.history {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  table.history th, table.history td {{ padding: 5px 8px; border-bottom: 1px solid var(--border); text-align: left; }}
  table.history td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Hidden Gem Spotter — Recommendations</h1>
    <div class="subtitle">Generated {generated_at} &middot; {len(ordered)} tickers &middot; sorted by composite score (most bullish first)</div>
    <div class="disclaimer">
      Technical pillar only — fundamental/revision/alternative pillars aren't built yet, so a
      composite score here reflects price/volume behavior alone, not company fundamentals.
      Confidence is a backtested Wilson interval where available, otherwise explicitly marked
      "uncalibrated." This is a personal research prototype, not investment advice.
    </div>

    <table>
      <thead>
        <tr><th>Ticker</th><th>Role</th><th>Action</th><th>Score</th><th>Confidence</th><th>Price</th><th>Top reason</th></tr>
      </thead>
      <tbody>{''.join(summary_rows)}</tbody>
    </table>

    <h1 id="detail">Per-ticker detail</h1>
    {''.join(detail_cards)}
  </div>
</body>
</html>
"""

    with open(out_path, "w") as f:
        f.write(page)
    return out_path
'''
Path("recommendation/html_report.py").write_text(content_recommendation_html_report_py)
print("wrote recommendation/html_report.py:", len(content_recommendation_html_report_py), "bytes")

content_tests_test_html_report_py = r'''"""
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
'''
Path("tests/test_html_report.py").write_text(content_tests_test_html_report_py)
print("wrote tests/test_html_report.py:", len(content_tests_test_html_report_py), "bytes")