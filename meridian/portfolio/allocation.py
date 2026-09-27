"""Target sleeve weights and initial ledger seeding from the fixed allocation table.

This module owns target sleeve weights and ledger initialization at first run;
it does NOT own risk limits or suspension logic (those stay in portfolio/risk_budget.py).

The 9-sleeve allocation (PLAN.md §12) is the fixed starting split of account equity. At
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

#: Fixed starting allocation by sleeve (sums to 1.0). Keys double as ledger/family names.
#: trend_following and breakouts used to share one 15% "trend-following breakout"
#: sleeve with breakouts monitor-only (0% capital) — breakouts now gets its own
#: real allocation, carved 50/50 out of that combined sleeve.
SLEEVE_ALLOCATIONS: dict[str, float] = {
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

# Guard the invariant at import: the split must be exhaustive.
if abs(sum(SLEEVE_ALLOCATIONS.values()) - 1.0) > 1e-9:
    raise ValueError(f"SLEEVE_ALLOCATIONS must sum to 1.0, got {sum(SLEEVE_ALLOCATIONS.values())}")


def seed_ledgers(equity: float, *, stage: str = "research") -> list[StrategyLedger]:
    """One fresh ``StrategyLedger`` per sleeve, capital = ``pct × equity``."""
    return [
        StrategyLedger(
            name=name, family=name, stage=stage, capital_alloc=pct * float(equity)
        )
        for name, pct in SLEEVE_ALLOCATIONS.items()
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
