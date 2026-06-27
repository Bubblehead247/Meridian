"""Tests for the interactive menu (driven by a scripted input function)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.data import loader as loader_mod
from meridian.interactive import run_menu


def _frame(n=500, start="2010-01-01"):
    idx = pd.date_range(start, periods=n, freq="B")
    close = pd.Series(100 + 5 * np.sin(np.linspace(0, 40, n)) + np.linspace(0, 12, n), index=idx)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(close.iloc[0]), "high": close + 1, "low": close - 1,
         "close": close, "adj_close": close, "volume": np.linspace(1e6, 2e6, n)},
        index=idx,
    )


def _scripted(responses):
    it = iter(responses)
    return lambda prompt="": next(it)


def _patch(monkeypatch):
    monkeypatch.setattr(loader_mod, "load_ohlcv", lambda *a, **k: _frame())


def _run(monkeypatch, responses):
    _patch(monkeypatch)
    out: list[str] = []
    rc = run_menu(input_fn=_scripted(responses), print_fn=out.append)
    return rc, "\n".join(out)


def test_quit_at_symbol_prompt(monkeypatch):
    rc, text = _run(monkeypatch, ["q"])
    assert rc == 0 and "Goodbye." in text


def test_gauntlet_then_quit(monkeypatch):
    rc, text = _run(monkeypatch, ["1", "2", "q"])          # SPY preset -> gauntlet -> quit
    assert rc == 0
    assert "Gauntlet" in text and "model" in text


def test_single_strategy_flow(monkeypatch):
    rc, text = _run(monkeypatch, ["1", "1", "1", "q"])      # SPY -> single -> model #1 -> quit
    assert rc == 0
    assert "final graduation stage" in text


def test_fund_flow_with_default_equity(monkeypatch):
    rc, text = _run(monkeypatch, ["1", "3", "", "q"])       # SPY -> fund -> default equity -> quit
    assert rc == 0
    assert "Fund — SPY" in text


def test_basket_gauntlet_is_cross_sectional(monkeypatch):
    rc, text = _run(monkeypatch, ["4", "2", "q"])           # sector basket -> gauntlet -> quit
    assert rc == 0
    assert "cross-sectional models" in text


def test_sweep_flow(monkeypatch):
    rc, text = _run(monkeypatch, ["1", "4", "1", "q"])      # SPY -> tune -> model #1 -> quit
    assert rc == 0
    assert "Sweep —" in text and "Multiple-testing" in text


def test_invalid_action_reprompts(monkeypatch):
    rc, text = _run(monkeypatch, ["1", "z", "q"])           # bad action -> reprompt -> quit
    assert rc == 0
    assert "please enter" in text


def test_run_failure_keeps_menu_alive(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("data unavailable")

    monkeypatch.setattr(loader_mod, "load_ohlcv", boom)
    out: list[str] = []
    rc = run_menu(input_fn=_scripted(["1", "2", "q"]), print_fn=out.append)
    assert rc == 0
    assert any("run failed" in line for line in out)        # caught, menu survived
