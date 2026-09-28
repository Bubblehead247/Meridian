"""Tests for the live allocation (the C1 core book) and the research 9-sleeve table."""

from __future__ import annotations

import math

import pytest

from meridian.portfolio import (
    SLEEVE_ALLOCATIONS,
    LedgerStore,
    initialize_ledgers,
    rebalance_targets,
    seed_ledgers,
)
from meridian.portfolio.allocation import CORE_EXPOSURE, RESEARCH_SLEEVE_ALLOCATIONS

LIVE = {"core_equity", "core_bonds", "long_term_etf", "core_tbills"}


def test_live_table_is_the_c1_core_at_the_drawdown_exposure():
    funded = {k: v for k, v in SLEEVE_ALLOCATIONS.items() if v > 0}
    assert set(funded) == LIVE
    assert math.isclose(sum(SLEEVE_ALLOCATIONS.values()), 1.0)
    # 50% 60/40 + 25% long-term ETF + 25% T-bills, scaled; the rest in T-bills.
    assert funded["core_equity"] == pytest.approx(CORE_EXPOSURE * 0.30)
    assert funded["core_bonds"] == pytest.approx(CORE_EXPOSURE * 0.20)
    assert funded["long_term_etf"] == pytest.approx(CORE_EXPOSURE * 0.25)
    assert funded["core_tbills"] == pytest.approx(1 - CORE_EXPOSURE * 0.75)


def test_retired_sleeves_stay_listed_at_zero_so_their_accounting_continues():
    for sleeve in RESEARCH_SLEEVE_ALLOCATIONS:
        assert sleeve in SLEEVE_ALLOCATIONS
    assert SLEEVE_ALLOCATIONS["mean_reversion"] == 0.0
    assert SLEEVE_ALLOCATIONS["momentum"] == 0.0


def test_research_table_keeps_the_original_nine_sleeves():
    assert len(RESEARCH_SLEEVE_ALLOCATIONS) == 9
    assert math.isclose(sum(RESEARCH_SLEEVE_ALLOCATIONS.values()), 1.0)
    leds = seed_ledgers(100_000.0, allocations=RESEARCH_SLEEVE_ALLOCATIONS)
    by_name = {x.name: x for x in leds}
    assert by_name["long_term_etf"].capital_alloc == 25_000.0
    assert by_name["breakouts"].capital_alloc == 7_500.0


def test_seed_ledgers_splits_equity_by_the_live_table():
    leds = seed_ledgers(100_000.0)
    by_name = {x.name: x for x in leds}
    assert by_name["core_tbills"].capital_alloc == pytest.approx(41_500.0)
    assert all(x.stage == "research" for x in leds)
    assert math.isclose(sum(x.capital_alloc for x in leds), 100_000.0)


def test_rebalance_targets_sum_to_equity():
    leds = seed_ledgers(50_000.0)
    targets = rebalance_targets(leds, 50_000.0)
    assert math.isclose(sum(targets.values()), 50_000.0)
    assert targets["momentum"] == 0.0


def test_initialize_ledgers_is_idempotent_and_preserves_state(tmp_path):
    store = LedgerStore(tmp_path)
    first = initialize_ledgers(100_000.0, store)
    n = len(SLEEVE_ALLOCATIONS)
    assert len(first) == n and len(store.list()) == n

    mr = first["mean_reversion"]
    mr.record_trade(500.0)
    store.save(mr)

    second = initialize_ledgers(100_000.0, store)
    assert second["mean_reversion"].realized_pnl == 500.0
    assert len(store.list()) == n
