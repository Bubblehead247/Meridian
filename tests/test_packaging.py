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
    return pd.DataFrame(
        {"high": prices + 1, "low": prices - 1, "close": prices}, index=prices.index
    )


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


def test_resolve_symbols_explicit_list():
    assert runner.resolve_symbols({"data": {"symbols": ["AAPL", "MSFT"]}}) == ["AAPL", "MSFT"]
    assert runner.resolve_symbols({"data": {"symbol": "SPY"}}) == ["SPY"]


def test_is_survivorship_biased_default_path_is_biased():
    assert runner.is_survivorship_biased({}) is True
    assert runner.is_survivorship_biased({"data": {"symbols": ["SPY"]}}) is True


def test_is_survivorship_biased_free_variant_is_not_biased():
    cfg = {"data": {"source": "survivorship", "root": "x"}}
    assert runner.is_survivorship_biased(cfg) is False


def test_is_survivorship_biased_survivor_variant_is_still_biased():
    cfg = {"data": {"source": "survivorship", "variant": "survivor", "root": "x"}}
    assert runner.is_survivorship_biased(cfg) is True


class _FakeScreener:
    def __init__(self):
        self.calls = []

    def build_universe(self, name, **kwargs):
        from meridian.data.universe import Universe
        self.calls.append(kwargs)
        return Universe(name, ("AAA", "BBB", "CCC", "DDD"))


def test_resolve_symbols_from_screen_with_limit():
    scr = _FakeScreener()
    cfg = {"data": {"screen": {"country": "United States", "sector": "Information Technology",
                               "limit": 2, "only_primary_listing": True}}}
    syms = runner.resolve_symbols(cfg, screener=scr)
    assert syms == ["AAA", "BBB"]                       # limited to 2
    # filters forwarded; screener-only keys (limit) not passed as a filter
    fwd = scr.calls[0]
    assert fwd["country"] == "United States" and fwd["sector"] == "Information Technology"
    assert fwd["only_primary_listing"] is True
    assert "limit" not in fwd


def test_resolve_symbols_screen_no_matches_raises():
    class _Empty:
        def build_universe(self, name, **kwargs):
            from meridian.data.universe import Universe
            return Universe(name, ())

    with pytest.raises(ValueError, match="matched no symbols"):
        runner.resolve_symbols({"data": {"screen": {"sector": "Nope"}}}, screener=_Empty())


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
    assert (tmp_path / "run_log.jsonl").exists()


def test_run_validation_report_appends_not_overwrites(tmp_path):
    px = _mr()
    cfg = _cfg(tmp_path)
    runner.run_validation_report(cfg, px, _bars(px))
    runner.run_validation_report(cfg, px, _bars(px))
    lines = (tmp_path / "run_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert (tmp_path / "r.md").exists()  # report itself is still overwritten each run


def test_run_log_handles_non_json_native_meta_values(tmp_path):
    """YAML dates parse as datetime.date (e.g. `start: 2010-01-01` unquoted); must not crash."""
    import datetime

    from meridian.experiments.run_log import append_run, new_run_record, read_runs

    rec = new_run_record(
        config_path="c.yaml", raw_config_text="x: 1", kind="validation",
        meta={"start": datetime.date(2010, 1, 1), "symbols": ["SPY"]},
        report_path="r.md", summary={"n_tested": 1, "n_significant": 0},
    )
    log_path = tmp_path / "run_log.jsonl"
    append_run(rec, path=log_path)
    runs = read_runs(log_path)
    assert len(runs) == 1
    assert runs[0].meta["start"] == "2010-01-01"


def test_cli_list_runs_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = main(["list", "runs"])
    assert rc == 0
    assert "No logged runs" in capsys.readouterr().out


def _universe(k=5, n=600) -> dict:
    out = {}
    for i in range(k):
        rng = np.random.default_rng(i)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.8 * x[t - 1] + rng.normal(0, 1)
        out[f"S{i}"] = pd.Series(100 + x, index=pd.RangeIndex(n))
    return out


def _univ_cfg(tmp_path) -> dict:
    cfg = _cfg(tmp_path)
    cfg["estimators"] = ["sma", "ou"]
    cfg["sizing"] = "equal_weight"
    cfg["validation"] = {"mode": "anchored", "min_train": 250, "test_span": 100,
                         "step": 100, "n_boot": 100, "n_mc": 80, "correction": "bh"}
    cfg["report"]["path"] = str(tmp_path / "u.md")
    return cfg


def _survivorship_root(tmp_path):
    """Build a minimal survivorship-free-spy-format dataset; return its root."""
    root = tmp_path / "sf"
    (root / "data").mkdir(parents=True)
    root.joinpath("constituents.csv").write_text(
        "2018-02-28,\"['A', 'B']\"\n2018-01-31,\"['A', 'B', 'C']\"\n", encoding="utf-8"
    )
    dates = pd.date_range("2018-01-02", "2018-02-28", freq="B")
    for i, t in enumerate(["A", "B", "C"]):
        px = 100 + i + np.arange(len(dates)) * 0.1
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": px, "high": px + 1,
                      "low": px - 1, "close": px, "volume": 1000}).to_csv(
            root / "data" / f"{t}.csv", index=False)
    return root


def test_load_universe_prices_survivorship_free(tmp_path):
    cfg = {"data": {"source": "survivorship", "root": str(_survivorship_root(tmp_path)),
                    "variant": "free", "start": "2018-01-01", "end": "2018-03-01"}}
    prices, bars = runner.load_universe_prices(cfg)
    assert set(prices) == {"A", "B", "C"}     # includes removed name C
    assert bars == {}


def test_load_universe_prices_survivor_only(tmp_path):
    cfg = {"data": {"source": "survivorship", "root": str(_survivorship_root(tmp_path)),
                    "variant": "survivor", "start": "2018-01-01", "end": "2018-03-01"}}
    prices, _ = runner.load_universe_prices(cfg)
    assert set(prices) == {"A", "B"}          # survivor-only drops removed C


def test_survivorship_unknown_variant_raises(tmp_path):
    cfg = {"data": {"source": "survivorship", "root": str(_survivorship_root(tmp_path)),
                    "variant": "bogus"}}
    with pytest.raises(ValueError, match="variant"):
        runner.load_universe_prices(cfg)


def test_survivorship_requires_root():
    with pytest.raises(ValueError, match="data.root"):
        runner.load_universe_prices({"data": {"source": "survivorship"}})


def test_run_universe_validation(tmp_path):
    uni = _universe()
    table = runner.run_universe_validation(_univ_cfg(tmp_path), uni)
    assert set(table["estimator"]) == {"sma", "ou"}
    assert (table["n_symbols"] == 5).all()


def test_cli_universe(tmp_path, monkeypatch, capsys):
    uni = _universe()
    bars = {s: pd.DataFrame({"high": p + 1, "low": p - 1, "close": p}) for s, p in uni.items()}
    monkeypatch.setattr(runner, "load_universe_prices", lambda cfg, cache=None: (uni, bars))
    cfg_path = tmp_path / "u.yaml"
    cfg_path.write_text(yaml.safe_dump(_univ_cfg(tmp_path)), encoding="utf-8")

    rc = main(["universe", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Universe: 5 symbols" in out
    assert "Significant after correction" in out
    assert (tmp_path / "u.md").exists()


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
