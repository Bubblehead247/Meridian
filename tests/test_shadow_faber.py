"""Faber shadow sleeve: the registered rule, logged once per month-end, no rewrites."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from meridian.execution import shadow_faber as sf


def _prices(end="2026-12-15", up=("SPY", "EFA", "IEF"), down=("GSG", "VNQ")):
    idx = pd.bdate_range("2025-01-01", end)
    t = np.arange(len(idx))
    data = {s: 100 * (1 + 0.001) ** t for s in up}
    data.update({s: 100 * (1 - 0.001) ** t for s in down})
    return pd.DataFrame(data, index=idx)


def test_rule_holds_assets_above_their_10_month_average():
    w = sf.month_end_targets(_prices())
    last = w.iloc[-2]  # a completed month
    assert last[["SPY", "EFA", "IEF"]].tolist() == [0.2, 0.2, 0.2]
    assert last[["GSG", "VNQ"]].tolist() == [0.0, 0.0]


def test_each_completed_month_end_is_logged_once(tmp_path):
    log = tmp_path / "faber.jsonl"
    new = sf.update(_prices(), log=log)
    months = [r["month_end"] for r in new]
    # Sep, Oct, Nov month-ends are complete; December (bars to the 15th) is not.
    assert months == ["2026-09-30", "2026-10-30", "2026-11-30"]
    assert new[0]["tbills"] == pytest.approx(0.4)
    assert sf.update(_prices(), log=log) == []
    assert len(log.read_text().splitlines()) == 3


def test_a_logged_decision_is_never_rewritten(tmp_path):
    log = tmp_path / "faber.jsonl"
    sf.update(_prices(end="2026-10-15"), log=log)
    before = log.read_text()
    # Later data that would imply a different September decision changes nothing.
    sf.update(_prices(end="2026-12-15", up=("GSG",), down=("SPY", "EFA", "IEF", "VNQ")), log=log)
    assert log.read_text().startswith(before)


def test_performance_uses_the_logged_weights_from_the_start_date(tmp_path):
    log = tmp_path / "faber.jsonl"
    px = _prices()
    sf.update(px, log=log)
    rf = pd.Series(0.0, index=px.index)
    perf = sf.performance(px, rf, log=log, cost_bps=0.0)
    rets = px.pct_change().loc[sf.SHADOW_START:]  # 10/1 earns its move from the 9/30 close
    expected = (1 + rets[["SPY", "EFA", "IEF"]].mean(axis=1) * 0.6).prod() - 1
    assert perf["shadow_return"] == pytest.approx(expected, rel=1e-3)
    assert json.loads(log.read_text().splitlines()[0])["month_end"] == "2026-09-30"
