"""Tests for the `meridian fund` whole-lifecycle CLI command (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian import cli
from meridian.data import loader as loader_mod


def _frame(n=1200, start="2010-01-01"):
    idx = pd.date_range(start, periods=n, freq="B")
    close = pd.Series(100 + 8 * np.sin(np.linspace(0, 50, n)) + np.linspace(0, 20, n), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1, "low": close - 1, "close": close,
            "adj_close": close, "volume": np.linspace(1e6, 2e6, n),
        },
        index=idx,
    )


def test_fund_runs_end_to_end_and_writes_report(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(loader_mod, "load_ohlcv", lambda *a, **k: _frame())

    rc = cli.main(["fund", "--symbol", "SPY", "--equity", "100000",
                   "--report-dir", str(tmp_path)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Meridian fund run — SPY" in out
    assert "sleeve actions:" in out
    # the strategy sleeves each ran a model; cash/experimental are held
    assert "mean_reversion" in out and "trend_following" in out
    assert "no strategy model — sleeve held" in out      # cash_reserve / experimental_research

    reports = list(tmp_path.glob("monthly_review_SPY_*.md"))
    assert len(reports) == 1
    body = reports[0].read_text(encoding="utf-8")
    assert "# Monthly Review" in body and "## Sleeve actions" in body


def test_fund_runs_each_of_several_symbols(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(loader_mod, "load_ohlcv", lambda *a, **k: _frame())
    rc = cli.main(["fund", "--symbols", "SPY", "QQQ", "--report-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Meridian fund run — SPY" in out and "Meridian fund run — QQQ" in out
    assert {p.name.split("_")[2] for p in tmp_path.glob("monthly_review_*.md")} == {"SPY", "QQQ"}


def test_fund_network_error_is_graceful(monkeypatch, capsys):
    from urllib.error import URLError

    from meridian.data import universe as universe_mod

    def boom(*a, **k):
        raise URLError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(universe_mod, "get_universe", boom)
    rc = cli.main(["fund", "--universe", "SP500"])
    assert rc == 1                                   # clean exit, no traceback
    assert "data fetch failed" in capsys.readouterr().err


def test_fund_unknown_universe_is_graceful(capsys):
    rc = cli.main(["fund", "--universe", "NOPE"])     # offline: unknown name -> KeyError
    assert rc == 2
    assert "Unknown universe" in capsys.readouterr().err


def test_cs_fund_runs_and_produces_monthly_review(monkeypatch):
    """run_cs_fund processes a basket of symbols through CS models end-to-end."""
    from meridian.experiments.fund import run_cs_fund

    n = 3600
    idx = pd.date_range("2010-01-01", periods=n, freq="B")

    def _make_series(drift):
        rng = np.random.default_rng(42)
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), index=idx)

    basket = {s: _make_series(d) for s, d in zip(
        ["XLK", "XLF", "XLE", "XLY", "XLV", "XLI"],
        [0.0006, 0.0003, 0.0, -0.0002, 0.0004, 0.0001],
    )}

    ledgers, review = run_cs_fund(basket, equity=100_000.0)
    assert len(ledgers) > 0
    from meridian.portfolio import MonthlyReview
    assert isinstance(review, MonthlyReview)
    # CS families (momentum, sector_rotation) should have run; others stay at seeded stage.
    sleeve_names = {s.sleeve for s in review.sleeves}
    assert "momentum" in sleeve_names or "sector_rotation" in sleeve_names


def test_fund_universe_expands_to_symbols(monkeypatch, tmp_path):
    from meridian.data import universe as universe_mod

    monkeypatch.setattr(loader_mod, "load_ohlcv", lambda *a, **k: _frame())
    monkeypatch.setattr(universe_mod, "get_universe",
                        lambda name, **k: universe_mod.Universe(name, ("AAA", "BBB")))
    rc = cli.main(["fund", "--universe", "MINI", "--report-dir", str(tmp_path)])
    assert rc == 0
    assert len(list(tmp_path.glob("monthly_review_*.md"))) == 2   # one per universe member
