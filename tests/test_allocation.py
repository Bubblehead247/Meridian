"""Tests for the 8-sleeve starting allocation + ledger seeding (PLAN.md §12)."""

from __future__ import annotations

import math

from meridian.portfolio import (
    SLEEVE_ALLOCATIONS,
    LedgerStore,
    initialize_ledgers,
    rebalance_targets,
    seed_ledgers,
)


def test_allocation_table_has_eight_sleeves_summing_to_one():
    assert len(SLEEVE_ALLOCATIONS) == 8
    assert math.isclose(sum(SLEEVE_ALLOCATIONS.values()), 1.0)


def test_seed_ledgers_splits_equity_by_table():
    leds = seed_ledgers(100_000.0)
    assert len(leds) == 8
    by_name = {x.name: x for x in leds}
    assert by_name["long_term_etf"].capital_alloc == 25_000.0
    assert by_name["cash_reserve"].capital_alloc == 5_000.0
    assert all(x.stage == "research" for x in leds)
    assert math.isclose(sum(x.capital_alloc for x in leds), 100_000.0)


def test_rebalance_targets_sum_to_equity():
    leds = seed_ledgers(50_000.0)
    targets = rebalance_targets(leds, 50_000.0)
    assert math.isclose(sum(targets.values()), 50_000.0)
    assert targets["momentum"] == 7_500.0


def test_initialize_ledgers_is_idempotent_and_preserves_state(tmp_path):
    store = LedgerStore(tmp_path)
    first = initialize_ledgers(100_000.0, store)
    assert len(first) == 8 and len(store.list()) == 8

    # accrue state on one sleeve, persist it
    mr = first["mean_reversion"]
    mr.record_trade(500.0)
    store.save(mr)

    # re-init must NOT clobber the accumulated ledger
    second = initialize_ledgers(100_000.0, store)
    assert second["mean_reversion"].realized_pnl == 500.0
    assert len(store.list()) == 8       # no duplicates created
