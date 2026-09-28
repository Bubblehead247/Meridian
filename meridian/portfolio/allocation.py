"""Target sleeve weights and initial ledger seeding from the fixed allocation table.

This module owns target sleeve weights and ledger initialization at first run;
it does NOT own risk limits or suspension logic (those stay in portfolio/risk_budget.py).

The live allocation (``SLEEVE_ALLOCATIONS``, the trend 6 core) is the split of account equity. At
first run it seeds one ``StrategyLedger`` per sleeve with ``capital_alloc = pct × equity``;
on later runs seeding is idempotent (existing ledgers, with their accumulated state, are
left untouched).
"""

from __future__ import annotations

from meridian.portfolio.ledger import LedgerStore, StrategyLedger

#: Maps a strategy family to the sleeve that funds it. Families not listed here
#: map to themselves (family == sleeve). Use this when a family shares a sleeve
#: with another.
FAMILY_TO_SLEEVE: dict[str, str] = {
    "volatility": "experimental_research",
}

#: The original 9-sleeve split (PLAN.md §12). Kept for the research tools that
#: simulate the whole multi-sleeve fund (``experiments/fund.py``); it no longer
#: sets live capital.
RESEARCH_SLEEVE_ALLOCATIONS: dict[str, float] = {
    "long_term_etf": 0.25,
    "momentum": 0.15,
    "trend_following": 0.075,
    "breakouts": 0.075,
    "mean_reversion": 0.15,
    "pullback_continuation": 0.10,
    "sector_rotation": 0.10,
    "cash_reserve": 0.05,
    "experimental_research": 0.05,
}

#: Live allocation by sleeve (sums to 1.0): the trend 6 core (TREND6-EW-U8), recommended
#: by the pre-registered full sweep (research/plans/full_sweep_2026_10.json): eight
#: asset-class ETF sleeves at 12.5% each, each held while above its 6-month average,
#: otherwise in T-bills via the idle-cash sweep. 2008-2026/06: 7.0%/yr, max DD -9.6%
#: (inside the owner's -15%). Retired sleeves stay listed at 0 so their accounting continues.
TREND6_SLEEVES = tuple(f"core_trend_{etf}" for etf in ("spy", "qqq", "iwm", "efa", "eem", "gld", "ief", "tlt"))

SLEEVE_ALLOCATIONS: dict[str, float] = {
    **{sleeve: 0.125 for sleeve in TREND6_SLEEVES},
    "long_term_etf": 0.0,
    "momentum": 0.0,
    "trend_following": 0.0,
    "breakouts": 0.0,
    "mean_reversion": 0.0,
    "pullback_continuation": 0.0,
    "sector_rotation": 0.0,
    "cash_reserve": 0.0,
    "experimental_research": 0.0,
}

# Guard the invariant at import: the split must be exhaustive.
for _name, _table in (("SLEEVE_ALLOCATIONS", SLEEVE_ALLOCATIONS),
                      ("RESEARCH_SLEEVE_ALLOCATIONS", RESEARCH_SLEEVE_ALLOCATIONS)):
    if abs(sum(_table.values()) - 1.0) > 1e-9:
        raise ValueError(f"{_name} must sum to 1.0, got {sum(_table.values())}")


def seed_ledgers(equity: float, *, stage: str = "research",
                 allocations: dict[str, float] | None = None) -> list[StrategyLedger]:
    """One fresh ``StrategyLedger`` per sleeve, capital = ``pct × equity``."""
    table = SLEEVE_ALLOCATIONS if allocations is None else allocations
    return [
        StrategyLedger(
            name=name, family=name, stage=stage, capital_alloc=pct * float(equity)
        )
        for name, pct in table.items()
    ]


def rebalance_targets(
    ledgers: list[StrategyLedger], equity: float
) -> dict[str, float]:
    """Dollar capital target per sleeve from the fixed table (``pct × equity``).

    Only sleeves present in ``SLEEVE_ALLOCATIONS`` get a target; others map to 0.0.
    """
    return {
        led.name: SLEEVE_ALLOCATIONS.get(led.name, 0.0) * float(equity) for led in ledgers
    }


def initialize_ledgers(
    equity: float, store: LedgerStore | None = None, *, stage: str = "research"
) -> dict[str, StrategyLedger]:
    """First-run seeding: create+persist any missing sleeve ledger, keep existing ones.

    Idempotent — re-running never clobbers a sleeve that already has accumulated state.
    Returns the full ``{name: ledger}`` map (loaded existing + newly created).
    """
    store = store or LedgerStore()
    out: dict[str, StrategyLedger] = {}
    for led in seed_ledgers(equity, stage=stage):
        existing = store.load(led.name)
        if existing is not None:
            out[led.name] = existing
        else:
            store.save(led)
            out[led.name] = led
    return out
