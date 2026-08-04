"""Tests for reporting/review_runner.py's sleeve-return reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.pipeline.records import StrategyRecord
from meridian.reporting import review_runner
from meridian.reporting.review_runner import _sleeve_returns_from_records


def _frame(n=300, seed=0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)
    close = pd.Series(100 + x, index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1, "low": close - 1, "close": close,
            "adj_close": close, "volume": np.linspace(1e6, 2e6, n),
        },
        index=idx,
    )


def _record(family, model, symbol, sharpe) -> StrategyRecord:
    return StrategyRecord(
        family=family, model=model, symbol=symbol, stage_passed="paper",
        saved_at="2026-01-01", scorecard={"sharpe": sharpe}, ledger={},
    )


def test_sleeve_returns_populated_for_single_asset_records(monkeypatch):
    monkeypatch.setattr(review_runner, "load_ohlcv", lambda *a, **k: _frame())
    records = [
        _record("mean_reversion", "zscore_reversion", "SPY", sharpe=0.5),
        _record("trend_following", "ma_trend", "QQQ", sharpe=0.3),
    ]
    out = _sleeve_returns_from_records(records)
    assert set(out) == {"mean_reversion", "trend_following"}
    assert all(isinstance(s, pd.Series) and len(s) > 0 for s in out.values())


def test_sleeve_returns_picks_best_sharpe_among_multiple_records(monkeypatch):
    monkeypatch.setattr(review_runner, "load_ohlcv", lambda *a, **k: _frame())
    records = [
        _record("mean_reversion", "zscore_reversion", "SPY", sharpe=0.1),
        _record("mean_reversion", "bollinger_reversion", "QQQ", sharpe=0.9),
    ]
    out = _sleeve_returns_from_records(records)
    assert "mean_reversion" in out  # best-Sharpe candidate succeeded


def test_sleeve_returns_skips_cross_sectional_records(monkeypatch):
    monkeypatch.setattr(review_runner, "load_ohlcv", lambda *a, **k: _frame())
    records = [_record("momentum", "relative_strength", "XLK", sharpe=0.5)]
    out = _sleeve_returns_from_records(records)
    assert "momentum" not in out  # relative_strength is cross-sectional, no basket here


def test_sleeve_returns_swallows_data_errors(monkeypatch):
    def boom(*a, **k):
        raise ValueError("no data")

    monkeypatch.setattr(review_runner, "load_ohlcv", boom)
    records = [_record("mean_reversion", "zscore_reversion", "BADSYM", sharpe=0.5)]
    out = _sleeve_returns_from_records(records)
    assert out == {}


def test_sleeve_returns_empty_records_is_safe():
    assert _sleeve_returns_from_records([]) == {}
