"""Tests for the live allocation (the trend 6 core) and the research 9-sleeve table."""

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
from meridian.portfolio.allocation import RESEARCH_SLEEVE_ALLOCATIONS, TREND6_SLEEVES


def test_live_table_is_trend6_eight_sleeves_at_an_eighth_each():
    funded = {k: v for k, v in SLEEVE_ALLOCATIONS.items() if v > 0}
    assert set(funded) == set(TREND6_SLEEVES) and len(funded) == 8
    assert all(v == pytest.approx(0.125) for v in funded.values())
    assert math.isclose(sum(SLEEVE_ALLOCATIONS.values()), 1.0)


def test_retired_sleeves_stay_listed_at_zero_so_their_accounting_continues():
    for sleeve in RESEARCH_SLEEVE_ALLOCATIONS:
        assert sleeve in SLEEVE_ALLOCATIONS
        assert SLEEVE_ALLOCATIONS[sleeve] == 0.0


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
    assert by_name["core_trend_tlt"].capital_alloc == pytest.approx(12_500.0)
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
