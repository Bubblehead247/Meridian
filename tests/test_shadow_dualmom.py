"""Dual momentum pod shadow: the registered rule, logged once per month-end, checked for fidelity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from meridian.execution import shadow_dualmom as sd


def _prices(end="2026-12-15", growth=None):
    """Business-day prices; each ETF compounds at its own daily rate."""
    idx = pd.bdate_range("2025-01-01", end)
    t = np.arange(len(idx))
    growth = growth or {"SPY": 0.0010, "QQQ": 0.0012, "IWM": 0.0008, "EFA": 0.0006,
                        "EEM": -0.0005, "GLD": 0.0004, "IEF": -0.0002, "TLT": -0.0004}
    return pd.DataFrame({s: 100 * (1 + g) ** t for s, g in growth.items()}, index=idx)


def _rf(px, daily=0.0001):
    return pd.Series(daily, index=px.index)


def test_top_four_beating_tbills_get_a_quarter_each():
    px = _prices()
    w = sd._completed_targets(px, _rf(px)).iloc[-1]
    assert w[["QQQ", "SPY", "IWM", "EFA"]].tolist() == [0.25, 0.25, 0.25, 0.25]
    assert w[["EEM", "GLD", "IEF", "TLT"]].sum() == 0.0


def test_slots_that_fail_the_tbill_hurdle_stay_in_tbills():
    growth = {"SPY": 0.0010, "QQQ": 0.0012, "IWM": -0.0003, "EFA": -0.0003,
              "EEM": -0.0005, "GLD": -0.0004, "IEF": -0.0002, "TLT": -0.0004}
    px = _prices(growth=growth)
    w = sd._completed_targets(px, _rf(px)).iloc[-1]
    assert w.sum() == pytest.approx(0.50)            # only SPY and QQQ qualify; 2 slots in T-bills
    assert w[["SPY", "QQQ"]].tolist() == [0.25, 0.25]


def test_each_completed_month_end_is_logged_once_and_never_rewritten(tmp_path):
    log = tmp_path / "dm.jsonl"
    px = _prices(end="2026-10-15")
    first = sd.update(px, _rf(px), log=log)
    assert [r["month_end"] for r in first] == ["2026-09-30"]        # October isn't complete
    before = log.read_text()
    px2 = _prices(end="2026-12-15", growth={s: -g for s, g in
                  {"SPY": 0.0010, "QQQ": 0.0012, "IWM": 0.0008, "EFA": 0.0006,
                   "EEM": -0.0005, "GLD": 0.0004, "IEF": -0.0002, "TLT": -0.0004}.items()})
    sd.update(px2, _rf(px2), log=log)
    assert log.read_text().startswith(before)                          # September untouched
    assert [json.loads(x)["month_end"] for x in log.read_text().splitlines()] == \
        ["2026-09-30", "2026-10-30", "2026-11-30"]


def test_fidelity_catches_a_tampered_log_row(tmp_path):
    log = tmp_path / "dm.jsonl"
    px = _prices()
    sd.update(px, _rf(px), log=log)
    assert sd.fidelity_mismatches(px, _rf(px), log=log) == 0
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    rows[0]["weights"]["TLT"] = 0.25
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert sd.fidelity_mismatches(px, _rf(px), log=log) == 1


def test_performance_uses_logged_weights_and_counts_trades(tmp_path):
    log = tmp_path / "dm.jsonl"
    px = _prices()
    rf = _rf(px, 0.0)
    sd.update(px, rf, log=log)
    perf = sd.performance(px, rf, log=log, cost_bps=0.0)
    rets = px.pct_change().loc[sd.SHADOW_START:]
    expected_pod = (1 + rets[["QQQ", "SPY", "IWM", "EFA"]].mean(axis=1)).prod() - 1
    assert perf["pod_return"] == pytest.approx(expected_pod, rel=1e-6)
    assert perf["trades"] == 0                          # the same four held every month
    assert perf["bench_return"] == pytest.approx((1 + rets.mean(axis=1)).prod() - 1, rel=1e-6)


def test_each_decision_stores_its_inputs_and_follows_from_them(tmp_path):
    log = tmp_path / "dm.jsonl"
    px = _prices()
    sd.update(px, _rf(px), log=log)
    row = json.loads(log.read_text().splitlines()[0])
    assert set(row["inputs"]["returns_6m"]) == set(sd.ETFS)
    assert sd._decide(row["inputs"]["returns_6m"], row["inputs"]["tbill_hurdle"]) == row["weights"]


def test_a_later_data_revision_is_reported_as_a_revision_not_a_fidelity_failure(tmp_path):
    log = tmp_path / "dm.jsonl"
    px = _prices(end="2026-10-15")
    sd.update(px, _rf(px), log=log)
    revised = px.copy()
    revised.loc[:"2026-09-30", "TLT"] *= 3.0 ** np.linspace(0, 1, len(revised.loc[:"2026-09-30"]))
    assert sd.fidelity_mismatches(revised, _rf(revised), log=log) == 0   # the code followed its inputs
    assert sd.data_revisions(revised, _rf(revised), log=log) == 1        # today's data would decide otherwise


def test_performance_reports_the_alpha_pnl_drawdown_the_stop_rule_uses(tmp_path):
    log = tmp_path / "dm.jsonl"
    px = _prices()
    sd.update(px, _rf(px, 0.0), log=log)
    perf = sd.performance(px, _rf(px, 0.0), log=log, cost_bps=0.0)
    assert "alpha_pnl_max_drawdown" in perf and perf["alpha_pnl_max_drawdown"] <= 0.0
