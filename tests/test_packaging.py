"""Phase 10 tests: config runner, CLI, and state persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from meridian.cli import main
from meridian.execution import PaperTrader, SimulatedBroker
from meridian.experiments import runner
from meridian.signals import SignalConfig, SignalState


def _mr(n=500, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)
    return pd.Series(100 + x, index=pd.RangeIndex(n))


def _bars(prices: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"high": prices + 1, "low": prices - 1, "close": prices}, index=prices.index)


def _cfg(tmp_path) -> dict:
    return {
        "experiment": {"seed": 0},
        "data": {"symbols": ["SPY"]},
        "estimators": ["sma", "ema"],
        "deviation": "zscore",
        "window": 20,
        "cost_bps": 1.0,
        "signal": {"entry_threshold": 1.0},
        "validation": {"mode": "anchored", "min_train": 150, "test_span": 80,
                       "step": 80, "n_boot": 100, "n_mc": 100, "correction": "bh"},
        "report": {"path": str(tmp_path / "r.md")},
        "estimator": "ou",
        "position_size": 10,
        "broker": {"type": "simulated", "cash": 100000, "cost_bps": 1.0},
    }


# --- config -> objects ----------------------------------------------------

def test_resolve_estimators_variants():
    assert runner.resolve_estimators({"estimators": ["sma", "ema"]}) == ["sma", "ema"]
    assert runner.resolve_estimators({"estimators": "kalman"}) == ["kalman"]
    assert len(runner.resolve_estimators({"estimators": "all"})) >= 37


def test_build_signal_and_wfo():
    sig = runner.build_signal({"signal": {"entry_threshold": 1.5, "exit_threshold": 0.2}})
    assert sig.entry_threshold == 1.5 and sig.exit_threshold == 0.2
    spec = runner.build_wfo({"validation": {"mode": "rolling", "train_span": 300}})
    assert spec.mode == "rolling" and spec.train_span == 300


def test_build_broker_simulated_and_unknown():
    b = runner.build_broker({"broker": {"type": "simulated", "cash": 5000}})
    assert isinstance(b, SimulatedBroker) and b.cash == 5000
    with pytest.raises(ValueError):
        runner.build_broker({"broker": {"type": "wat"}})


# --- pure runs ------------------------------------------------------------

def test_run_validation_returns_table(tmp_path):
    px = _mr()
    table = runner.run_validation(_cfg(tmp_path), px, _bars(px))
    assert set(table["estimator"]) == {"sma", "ema"}
    assert "significant" in table.columns


def test_run_validation_report_writes_file(tmp_path):
    px = _mr()
    table, path = runner.run_validation_report(_cfg(tmp_path), px, _bars(px))
    assert (tmp_path / "r.md").exists()
    assert "survivorship" in (tmp_path / "r.md").read_text(encoding="utf-8").lower()


def test_run_paper_dry_run(tmp_path):
    px = _mr()
    log, summary = runner.run_paper_dry_run(_cfg(tmp_path), px, _bars(px))
    assert summary["bars"] > 0
    assert "equity" in summary
    assert len(log) == summary["bars"]


# --- CLI ------------------------------------------------------------------

def test_cli_list_estimators(capsys):
    rc = main(["list", "estimators"])
    assert rc == 0
    assert "sma" in capsys.readouterr().out


def test_cli_validate(tmp_path, monkeypatch, capsys):
    px = _mr()
    monkeypatch.setattr(runner, "load_prices", lambda cfg, cache=None: (px, _bars(px)))
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(tmp_path)), encoding="utf-8")

    rc = main(["validate", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Significant after correction" in out
    assert (tmp_path / "r.md").exists()


def test_cli_paper(tmp_path, monkeypatch, capsys):
    px = _mr()
    monkeypatch.setattr(runner, "load_prices", lambda cfg, cache=None: (px, _bars(px)))
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(tmp_path)), encoding="utf-8")

    rc = main(["paper", str(cfg_path)])
    assert rc == 0
    assert "Session summary" in capsys.readouterr().out


# --- state persistence ----------------------------------------------------

def test_signal_state_dict_roundtrip():
    st = SignalState(SignalConfig(entry_threshold=1.0))
    st.step(-3.0)  # enter long
    st.bars_held = 4
    d = st.to_dict()
    st2 = SignalState(SignalConfig(entry_threshold=1.0))
    st2.load_dict(d)
    assert st2.to_dict() == d


def test_checkpoint_resume_matches_continuous_run(tmp_path):
    px = _mr()
    a, b = px.iloc[:250], px.iloc[250:]
    sig = SignalConfig(entry_threshold=1.0)

    # reference: one continuous session over A+B
    ref = PaperTrader("X", SimulatedBroker(), "sma", "zscore", sig, window=20)
    ref.replay(px)
    ref_b = [d.signal for d in ref.log[len(a):]]

    # checkpoint after A
    ck = PaperTrader("X", SimulatedBroker(), "sma", "zscore", sig, window=20)
    ck.replay(a)
    ck.save_checkpoint(tmp_path / "ck.json")

    # resume: re-derive indicators via warm_up, restore signal state, continue on B
    res = PaperTrader("X", SimulatedBroker(), "sma", "zscore", sig, window=20)
    res.warm_up(a)
    res.load_checkpoint(tmp_path / "ck.json")
    res.replay(b)
    res_b = [d.signal for d in res.log]

    assert res_b == ref_b
