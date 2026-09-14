"""
Root pytest bootstrap.

The repo's modules use plain absolute imports (`import config`,
`from signals import scoring`, etc.) that assume the repo root is on
sys.path — true automatically when you run a script *at* the root
(`python3 demo.py`), but not guaranteed when pytest collects tests from
`tests/`. This conftest.py, because it sits at the repo root, is loaded
first and makes sure the root is on sys.path no matter what directory
pytest is invoked from.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_bars_per_day():
    """
    config.set_bars_per_day() rewrites module-level globals (SMA_SHORT,
    LOOKBACK_52W, ...) in place — exactly the kind of shared mutable state
    that lets one test's hourly-bars setup silently leak into the next
    test's daily-bars assumptions. Force every test to start and end on
    the daily-bar default so test order never matters.
    """
    import config

    config.set_bars_per_day(1)
    yield
    config.set_bars_per_day(1)
