"""Target sleeve weights and initial ledger seeding from the fixed allocation table.

This module owns target sleeve weights and ledger initialization at first run;
it does NOT own risk limits or suspension logic (those stay in portfolio/risk_budget.py).

The live allocation (``SLEEVE_ALLOCATIONS``, the C1 core book) is the split of account equity. At
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

#: How much of the C1 core is invested; the rest of the account sits in T-bills.
#: 0.78 put C1's worst drawdown at the owner's -15% target over 2008-2026/06
#: (research/2026-09-returns-study step 4). Fitted on the same history it was
#: judged on, so revisit it at each review rather than treat it as precise.
CORE_EXPOSURE = 0.78

#: Live allocation by sleeve (sums to 1.0): the C1 core book chosen by the
#: pre-registered structure test (research/plans/meridian_structure_2026_10.json),
#: 50% 60/40 SPY/IEF + 25% long-term ETF rotation + 25% T-bills, scaled to
#: CORE_EXPOSURE. Retired sleeves stay listed at 0 so their accounting continues.
SLEEVE_ALLOCATIONS: dict[str, float] = {
    "core_equity": round(CORE_EXPOSURE * 0.50 * 0.60, 6),   # SPY   0.234
    "core_bonds": round(CORE_EXPOSURE * 0.50 * 0.40, 6),    # IEF   0.156
    "long_term_etf": round(CORE_EXPOSURE * 0.25, 6),        # QQQ/IEF/KMLM rotation 0.195
    "core_tbills": round(1 - CORE_EXPOSURE * 0.75, 6),      # SGOV  0.415
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
