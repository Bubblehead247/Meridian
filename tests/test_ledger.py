"""Tests for the per-strategy virtual ledger (PLAN.md §7)."""

from __future__ import annotations

import math

import pytest

from meridian.portfolio import LedgerStore, StrategyLedger, load_ledger, save_ledger


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
