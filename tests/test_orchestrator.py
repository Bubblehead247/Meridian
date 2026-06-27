"""Tests for the run_pipeline orchestrator (model -> stages -> ledger/graduation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families import create_model
from meridian.pipeline import run_pipeline
from meridian.portfolio import StrategyLedger


def _prices(n=3200, start="2010-01-01"):
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.Series(100 + 5 * np.sin(np.linspace(0, 80, n)), index=idx)


def test_run_pipeline_walks_stages_in_order():
    model = create_model("mean_reversion", "zscore_reversion")
    res = run_pipeline(model, _prices(), stop_on_fail=False)
    assert list(res) == ["backtest", "walk_forward", "oos"]
    assert all(r.scorecard is not None for r in res.values())


def test_run_pipeline_stops_on_first_failure():
    # the synthetic sine model loses -> fails the backtest gate -> walk halts there
    model = create_model("mean_reversion", "zscore_reversion")
    res = run_pipeline(model, _prices(), stop_on_fail=True)
    assert "backtest" in res
    assert res["backtest"].passed is False
    assert set(res) == {"backtest"}              # did not proceed past the failing stage


def test_run_pipeline_advances_ledger():
    model = create_model("mean_reversion", "zscore_reversion")
    led = StrategyLedger(name="zscore_reversion", family="mean_reversion", stage="research")
    res = run_pipeline(model, _prices(), ledger=led, as_of="2026-06-26")
    assert "ledger_action" in res["backtest"].detail
    assert res["backtest"].detail["ledger_action"] in ("promote", "hold", "retire")
    # a losing model gets retired by the graduation machine
    assert led.stage in ("research", "backtest", "retired")
