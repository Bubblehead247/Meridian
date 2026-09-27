"""The curated live book: one strategy pick per sleeve.

``live_picks.json`` (project root) is the single source of truth for what trades
live — exactly one ``{model, symbol}`` per strategy family. Both the execution path
(``execution/live_runner.run_paper_session``) and the fund chart
(``visualization/fund_chart``) read from here, so what trades always matches what
the chart shows.

A family that routes to a *different* sleeve already held by another live family is
monitor-only (0% weight); see ``live_pick_weight``.
"""

from __future__ import annotations

import json
from pathlib import Path

from meridian.portfolio.allocation import FAMILY_TO_SLEEVE, SLEEVE_ALLOCATIONS

LIVE_PICKS_FILE = Path(__file__).resolve().parent.parent.parent / "live_picks.json"


def load_live_picks() -> dict[str, dict]:
    """Return ``{family: {"model": ..., "symbol": ...}}`` (empty dict if file absent)."""
    if LIVE_PICKS_FILE.exists():
        return json.loads(LIVE_PICKS_FILE.read_text(encoding="utf-8"))
    return {}


def live_pick_weight(family: str, present_families: set[str]) -> float:
    """Account-equity weight for one family's live pick.

    A family whose sleeve (per ``FAMILY_TO_SLEEVE``) is named after a *different*
    family that is also live is monitor-only → 0.0 (the named family is the primary
    holder of that sleeve). Otherwise it gets its sleeve's full weight.

    Examples (with all 8 picks present): trend_following → 0.075,
    breakouts → 0.075 (its own sleeve, no longer monitor-only), volatility → 0.05,
    mean_reversion → 0.15.
    """
    target = FAMILY_TO_SLEEVE.get(family, family)
    if target != family and target in present_families:
        return 0.0
    return SLEEVE_ALLOCATIONS.get(target, 0.0)
