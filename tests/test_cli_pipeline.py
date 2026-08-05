"""Tests for the `meridian pipeline` + `meridian list models` CLI subcommands (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian import cli
from meridian.data import loader as loader_mod


def _frame(n=1000, start="2010-01-01", base=None):
    idx = pd.date_range(start, periods=n, freq="B")
    close = base if base is not None else 100 + 5 * np.sin(np.linspace(0, 60, n))
    close = pd.Series(close, index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1, "low": close - 1, "close": close,
            "adj_close": close, "volume": np.linspace(1e6, 2e6, n),
        },
        index=idx,
    )


def _patch_loader(monkeypatch, frame_for=None):
    def fake(symbol, start=None, *a, **k):
        if callable(frame_for):
            return frame_for(symbol)
        return _frame()

    monkeypatch.setattr(loader_mod, "load_ohlcv", fake)


# --- list models ----------------------------------------------------------

def test_list_models(capsys):
    assert cli.main(["list", "models"]) == 0
    out = capsys.readouterr().out
    assert "mean_reversion/zscore_reversion" in out
    assert "momentum/relative_strength" in out


# --- single-asset pipeline ------------------------------------------------

def test_pipeline_single_asset(monkeypatch, capsys):
    _patch_loader(monkeypatch)
    rc = cli.main(["pipeline", "mean_reversion/zscore_reversion", "--symbol", "SPY"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mean_reversion/zscore_reversion on SPY" in out
    assert "backtest" in out
    assert "final graduation stage:" in out


def test_pipeline_unknown_model_errors(capsys):
    rc = cli.main(["pipeline", "mean_reversion/nope", "--symbol", "SPY"])
    assert rc == 2
    assert "Unknown model" in capsys.readouterr().err


def test_pipeline_bad_spec_errors(capsys):
    rc = cli.main(["pipeline", "justaname"])
    assert rc == 2
    assert "<family>/<model>" in capsys.readouterr().err


# --- cross-sectional pipeline ---------------------------------------------

def test_pipeline_cross_sectional(monkeypatch, capsys):
    drifts = {"A": 0.0006, "B": 0.0002, "C": -0.0004}

    def frame_for(symbol):
        rng = np.random.default_rng(abs(hash(symbol)) % 1000)
        close = 100 * np.exp(np.cumsum(rng.normal(drifts.get(symbol, 0.0), 0.01, 800)))
        return _frame(n=800, base=close)

    _patch_loader(monkeypatch, frame_for=frame_for)
    rc = cli.main(["pipeline", "momentum/relative_strength", "--symbols", "A", "B", "C"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cross-sectional, 3 symbols" in out
    assert "sharpe" in out


def test_pipeline_cross_sectional_requires_symbols(capsys):
    rc = cli.main(["pipeline", "momentum/relative_strength"])
    assert rc == 2
    assert "cross-sectional" in capsys.readouterr().err


def test_pipeline_single_asset_runs_each_symbol(monkeypatch, capsys):
    _patch_loader(monkeypatch)
    rc = cli.main(["pipeline", "mean_reversion/zscore_reversion", "--symbols", "SPY", "QQQ"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "on SPY" in out and "on QQQ" in out          # one run per symbol, no retyping


def test_gauntlet_single_symbol(monkeypatch, capsys):
    _patch_loader(monkeypatch)
    rc = cli.main(["gauntlet", "--symbol", "SPY"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "single-asset models on SPY" in out and "sharpe" in out


def test_gauntlet_basket(monkeypatch, capsys):
    def frame_for(symbol):
        rng = np.random.default_rng(abs(hash(symbol)) % 1000)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 700)))
        return _frame(n=700, base=close)

    _patch_loader(monkeypatch, frame_for=frame_for)
    rc = cli.main(["gauntlet", "--symbols", "A", "B", "C"])
    assert rc == 0
    assert "cross-sectional models on 3 names" in capsys.readouterr().out


def test_sweep_subcommand(monkeypatch, capsys):
    _patch_loader(monkeypatch)
    rc = cli.main(["sweep", "trend_following/ma_trend", "--symbol", "SPY"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Tuning trend_following/ma_trend on SPY" in out
    assert "Multiple-testing" in out and "Out-of-sample passes:" in out


def test_sweep_confirm_subcommand(monkeypatch, capsys):
    # a frame spanning the 2010-2019 / 2020-2022 holdout windows
    monkeypatch.setattr(loader_mod, "load_ohlcv", lambda *a, **k: _frame(n=3600))
    rc = cli.main(["sweep", "trend_following/ma_trend", "--symbol", "SPY", "--confirm"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Auto-confirmation through the full pipeline" in out
    assert "VERDICT:" in out


def test_no_command_launches_menu(monkeypatch):
    import meridian.interactive as interactive_mod

    called = {}

    def fake_menu(*a, **k):
        called["ran"] = True
        return 0

    monkeypatch.setattr(interactive_mod, "run_menu", fake_menu)
    assert cli.main([]) == 0
    assert called.get("ran") is True


def test_pipeline_cross_sectional_via_universe(monkeypatch, capsys):
    from meridian.data import universe as universe_mod

    def frame_for(symbol):
        rng = np.random.default_rng(abs(hash(symbol)) % 1000)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 800)))
        return _frame(n=800, base=close)

    _patch_loader(monkeypatch, frame_for=frame_for)
    monkeypatch.setattr(universe_mod, "get_universe",
                        lambda name, **k: universe_mod.Universe(name, ("A", "B", "C")))
    rc = cli.main(["pipeline", "momentum/relative_strength", "--universe", "MINI"])
    assert rc == 0
    assert "cross-sectional, 3 symbols" in capsys.readouterr().out


def test_pipeline_index_universe_blocked_without_acknowledgement(monkeypatch, capsys):
    # Regression for the P0 look-ahead gap: SP500/NASDAQ100/RUSSELL1000 resolve to
    # TODAY's constituents with no historical membership data behind them, so using one
    # for a backtest silently leaked future index membership into the past. This must
    # now be blocked unless explicitly acknowledged, not silently resolved.
    _patch_loader(monkeypatch)
    rc = cli.main(["pipeline", "momentum/relative_strength", "--universe", "SP500"])
    assert rc == 2
    assert "survivorship" in capsys.readouterr().err.lower()


def test_pipeline_index_universe_allowed_with_acknowledgement(monkeypatch, capsys):
    from meridian.data import universe as universe_mod

    def frame_for(symbol):
        rng = np.random.default_rng(abs(hash(symbol)) % 1000)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 800)))
        return _frame(n=800, base=close)

    _patch_loader(monkeypatch, frame_for=frame_for)
    monkeypatch.setattr(universe_mod, "get_universe",
                        lambda name, **k: universe_mod.Universe(name, ("A", "B", "C")))
    rc = cli.main([
        "pipeline", "momentum/relative_strength", "--universe", "SP500",
        "--accept-survivorship-bias",
    ])
    assert rc == 0
    assert "cross-sectional, 3 symbols" in capsys.readouterr().out
