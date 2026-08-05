"""Tests for the per-strategy virtual ledger (PLAN.md §7)."""

from __future__ import annotations

import math

import pytest

import json

from meridian.portfolio import (
    LedgerStore,
    StaleLedgerWrite,
    StrategyLedger,
    load_ledger,
    save_ledger,
)


def _led(**kw) -> StrategyLedger:
    base = dict(name="mr", family="mean_reversion", capital_alloc=1000.0)
    base.update(kw)
    return StrategyLedger(**base)


# --- PnL / drawdown / stats ----------------------------------------------

def test_record_trade_updates_pnl_winrate_expectancy():
    led = _led()
    led.record_trade(-100.0)
    led.record_trade(200.0)
    assert led.realized_pnl == 100.0
    assert led.n_trades == 2 and led.n_wins == 1
    assert led.win_rate == 0.5
    assert led.expectancy == 50.0           # realized / n_trades
    assert led.equity == 1100.0


def test_drawdown_tracks_peak_to_trough():
    led = _led()
    led.record_trade(-100.0)                 # equity 900, peak 1000
    assert math.isclose(led.drawdown_cur, -0.1)
    assert math.isclose(led.drawdown_max, -0.1)
    led.record_trade(300.0)                  # equity 1200, new peak -> dd_cur 0
    assert led.drawdown_cur == 0.0
    assert math.isclose(led.drawdown_max, -0.1)   # max-DD remembers the trough


def test_mark_updates_unrealized_and_drawdown():
    led = _led()
    led.mark(-50.0)                          # equity 950
    assert led.unrealized_pnl == -50.0
    assert math.isclose(led.drawdown_cur, -0.05)


def test_update_metrics_sets_external_fields():
    led = _led()
    led.update_metrics(turnover=3.2, correlations={"momentum": 0.4}, risk_contribution=0.012)
    assert led.turnover == 3.2
    assert led.correlations == {"momentum": 0.4}
    assert led.risk_contribution == 0.012


# --- stage ----------------------------------------------------------------

def test_set_stage_and_validation():
    led = _led()
    assert led.stage == "research"
    led.set_stage("paper", on="2026-06-26")
    assert led.stage == "paper" and led.stage_entered == "2026-06-26"
    with pytest.raises(ValueError, match="unknown stage"):
        led.set_stage("superstar")


def test_invalid_initial_stage_raises():
    with pytest.raises(ValueError, match="unknown stage"):
        _led(stage="bogus")


# --- persistence ----------------------------------------------------------

def test_to_dict_from_dict_round_trip():
    led = _led()
    led.record_trade(120.0)
    led.set_stage("pilot")
    led.update_metrics(turnover=1.1)
    clone = StrategyLedger.from_dict(led.to_dict())
    assert clone == led


def test_ledger_store_save_load_list(tmp_path):
    store = LedgerStore(tmp_path)
    led = _led()
    led.record_trade(75.0)
    store.save(led)
    assert store.exists("mr")
    loaded = store.load("mr")
    assert loaded == led
    assert store.list() == ["mr"]
    assert set(store.all()) == {"mr"}


def test_load_missing_returns_none(tmp_path):
    assert LedgerStore(tmp_path).load("nope") is None


def test_module_level_convenience(tmp_path):
    store = LedgerStore(tmp_path)
    led = _led(name="trend", family="trend_following")
    save_ledger(led, store)
    assert load_ledger("trend", store) == led


# --- ledger integrity: versioning, history, atomic writes (P2-D) ------------

def test_new_ledger_starts_at_version_zero():
    assert _led().version == 0


def test_save_increments_version_and_appends_history(tmp_path):
    store = LedgerStore(tmp_path)
    led = _led()
    store.save(led)
    assert led.version == 1
    store.save(led)
    assert led.version == 2

    history = (tmp_path / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(history) == 2
    rec1, rec2 = (json.loads(line) for line in history)
    assert rec1["version"] == 1 and rec2["version"] == 2
    assert rec1["name"] == "mr" == rec2["name"]


def test_stale_write_is_refused_by_default(tmp_path):
    store = LedgerStore(tmp_path)
    led = _led()
    store.save(led)                     # version 1 on disk
    stale_copy = store.load("mr")       # a second reader loads version 1
    store.save(led)                     # a different writer advances to version 2

    stale_copy.record_trade(10.0)       # the stale copy is edited...
    with pytest.raises(StaleLedgerWrite):
        store.save(stale_copy)          # ...and must not silently overwrite version 2

    assert store.load("mr").version == 2  # on-disk state is untouched by the refused write


def test_stale_write_succeeds_with_force(tmp_path):
    store = LedgerStore(tmp_path)
    led = _led()
    store.save(led)
    stale_copy = store.load("mr")
    store.save(led)

    stale_copy.record_trade(10.0)
    store.save(stale_copy, force=True)
    assert store.load("mr").realized_pnl == 10.0


def test_save_is_atomic_no_temp_files_left_behind(tmp_path):
    store = LedgerStore(tmp_path)
    store.save(_led())
    leftovers = list(tmp_path.glob(".*.tmp"))
    assert leftovers == []


def test_history_survives_repeated_saves_of_different_strategies(tmp_path):
    store = LedgerStore(tmp_path)
    store.save(_led(name="mr", family="mean_reversion"))
    store.save(_led(name="tf", family="trend_following"))
    history = (tmp_path / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(history) == 2
    names = {json.loads(line)["name"] for line in history}
    assert names == {"mr", "tf"}
    # history.jsonl must not be picked up by list()/all() as a strategy ledger
    assert "history" not in store.list()
