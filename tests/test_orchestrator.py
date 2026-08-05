"""Tests for the run_pipeline orchestrator (model -> stages -> ledger/graduation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families import create_model
from meridian.pipeline import run_cross_sectional_pipeline, run_pipeline
from meridian.portfolio import StrategyLedger


def _prices(n=3200, start="2010-01-01"):
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.Series(100 + 5 * np.sin(np.linspace(0, 80, n)), index=idx)


def _universe(n=400):
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    return {f"S{i}": pd.Series(100 * np.exp(np.cumsum(rng.normal(d, 0.01, n))), index=idx)
            for i, d in enumerate([0.0006, 0.0002, -0.0002, -0.0006])}


def _bars(prices_by_symbol):
    return {s: pd.DataFrame({"open": p.shift(1).fillna(p.iloc[0]), "close": p}, index=p.index)
            for s, p in prices_by_symbol.items()}


def test_cross_sectional_pipeline_without_bars_falls_back_to_close_approx_and_is_flagged():
    # Regression for the P0 look-ahead gap: run_cross_sectional_pipeline used to call
    # model.backtest() without threading bars_by_symbol through at all, so every CLI/
    # interactive cross-sectional run silently filled at the same-bar close used to
    # generate the signal. This must now be visible on every stage's detail, not silent.
    model = create_model("sector_rotation", "relative_strength")
    universe = _universe()
    res = run_cross_sectional_pipeline(model, universe, stop_on_fail=False)
    assert res["backtest"].detail["fill_realism"] == "close_approx"


def test_cross_sectional_pipeline_with_bars_uses_next_open_fill_and_is_flagged():
    model = create_model("sector_rotation", "relative_strength")
    universe = _universe()
    res = run_cross_sectional_pipeline(
        model, universe, bars_by_symbol=_bars(universe), stop_on_fail=False
    )
    assert res["backtest"].detail["fill_realism"] == "next_open"


def test_cross_sectional_pipeline_tracks_oos_runs_by_default(monkeypatch, tmp_path):
    from meridian.pipeline import oos_guard as oos_guard_mod

    monkeypatch.setattr(oos_guard_mod, "DEFAULT_GUARD_DIR", tmp_path / "oos_runs")
    model = create_model("sector_rotation", "relative_strength")
    universe = _universe(n=1600)
    res1 = run_cross_sectional_pipeline(model, universe, stages=("oos",), stop_on_fail=False)
    res2 = run_cross_sectional_pipeline(model, universe, stages=("oos",), stop_on_fail=False)
    assert res1["oos"].detail.get("oos_run_count") == 1
    assert res2["oos"].detail.get("oos_run_count") == 2


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
