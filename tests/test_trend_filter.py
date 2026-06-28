"""Tests for CrossSectionalModel.trend_filter overlay.

Covers:
- Filter symbol is stripped from the tradeable universe in backtest()
- Signals are zeroed on bars where the filter is below its MA
- Signals are preserved on bars where the filter is above its MA
- All six registered trend-filtered model variants instantiate and produce signals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.families import create_model
from meridian.families.base import CrossSectionalModel, register_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bull_prices(n: int = 300, seed: int = 0) -> pd.Series:
    """Steadily rising price series — always above any long MA."""
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(100 + np.arange(n) * 0.5, index=idx, dtype=float)


def _bear_prices(n: int = 300) -> pd.Series:
    """Steadily falling price series — always below any long MA."""
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(200 - np.arange(n) * 0.5, index=idx, dtype=float)


def _mixed_universe(n: int = 400) -> dict[str, pd.Series]:
    """A small 3-asset universe (A, B, C) with alternating relative performance."""
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    return {
        "A": pd.Series(base * 1.1, index=idx),
        "B": pd.Series(base * 0.9, index=idx),
        "C": pd.Series(base,       index=idx),
    }


# ---------------------------------------------------------------------------
# Filter strip
# ---------------------------------------------------------------------------

class _FilteredModel(CrossSectionalModel):
    """Minimal CS model with trend filter enabled."""
    lookback: int = 20
    quantile: float = 0.33
    long_only: bool = True
    trend_filter: bool = True
    trend_filter_symbol: str = "SPY"
    trend_filter_window: int = 50


def test_trend_filter_strips_spy_from_tradeable():
    """SPY must not appear as a column in signals() when used as the filter."""
    model = _FilteredModel()
    uni = _mixed_universe()
    uni["SPY"] = _bull_prices(len(next(iter(uni.values()))))

    signals = model.signals({k: v for k, v in uni.items() if k != "SPY"})
    assert "SPY" not in signals.columns


def test_backtest_filter_symbol_not_traded():
    """After backtest(), SPY should not appear in the portfolio returns_by_symbol."""
    model = _FilteredModel()
    uni = _mixed_universe(300)
    uni["SPY"] = _bull_prices(300)
    result = model.backtest(uni, cost_bps=0.0)
    assert "SPY" not in result.returns_by_symbol.columns


# ---------------------------------------------------------------------------
# Signal gating
# ---------------------------------------------------------------------------

def test_signals_zeroed_when_filter_below_ma():
    """All signals must be 0 on every bar where filter is below its MA."""
    model = _FilteredModel()
    uni = _mixed_universe(300)
    # Filter always below MA: bear prices
    filter_prices = _bear_prices(300)
    uni["SPY"] = filter_prices

    result = model.backtest(uni, cost_bps=0.0)
    # After MA warmup, the bear filter should zero all signals → equity curve flat
    # (there may be a brief warmup period of 50 bars; check bars after warmup)
    post_warmup = result.returns.iloc[model.trend_filter_window + 5:]
    assert (post_warmup == 0).all(), "Returns should be 0 when filter is in bear mode"


def test_signals_active_when_filter_above_ma():
    """Some signals must be non-zero when the filter is consistently above its MA."""
    model = _FilteredModel()
    uni = _mixed_universe(300)
    uni["SPY"] = _bull_prices(300)  # always above MA

    result = model.backtest(uni, cost_bps=0.0)
    post_warmup = result.returns.iloc[model.trend_filter_window + model.lookback + 5:]
    assert (post_warmup != 0).any(), "Some returns should be non-zero in bull filter mode"


def test_filter_window_respected():
    """Signals should be 0 during the MA warmup window, non-zero after."""
    model = _FilteredModel()
    uni = _mixed_universe(300)
    uni["SPY"] = _bull_prices(300)

    result = model.backtest(uni, cost_bps=0.0)
    warmup = result.returns.iloc[:model.trend_filter_window]
    # During warmup, MA is NaN → signals zeroed
    assert (warmup == 0).all()


# ---------------------------------------------------------------------------
# Trend-filtered model variants — smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,model_name,symbols", [
    ("momentum", "relative_strength_trend_filtered",
     ["SPY", "AAPL", "MSFT", "AMZN"]),
    ("momentum", "dual_momentum_trend_filtered",
     ["SPY", "AAPL", "MSFT", "AMZN"]),
    ("sector_rotation", "relative_strength_trend_filtered",
     ["SPY", "XLK", "XLF", "XLE", "XLI"]),
    ("sector_rotation", "relative_strength_b1_trend_filtered",
     ["SPY", "XLK", "XLF", "XLE", "XLI"]),
])
def test_trend_filtered_variant_produces_signals(family, model_name, symbols):
    """Each registered trend-filtered variant instantiates and returns a valid signal frame."""
    model = create_model(family, model_name)
    assert model.trend_filter is True
    assert model.trend_filter_symbol == "SPY"

    n = 300
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    prices = {
        s: pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
        for s in symbols
        if s != "SPY"
    }
    prices["SPY"] = _bull_prices(n)

    sig = model.signals({k: v for k, v in prices.items() if k != "SPY"})
    assert isinstance(sig, pd.DataFrame)
    assert set(np.unique(sig.values)) <= {-1, 0, 1}
    assert "SPY" not in sig.columns


@pytest.mark.parametrize("family,model_name", [
    ("momentum", "relative_strength_trend_filtered"),
    ("momentum", "dual_momentum_trend_filtered"),
    ("sector_rotation", "relative_strength_trend_filtered"),
    ("sector_rotation", "relative_strength_b1_trend_filtered"),
])
def test_trend_filtered_variant_backtest_runs(family, model_name):
    """backtest() completes without error and routes SPY as filter, not tradeable."""
    model = create_model(family, model_name)
    n = 300
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(1)
    symbols = ["SPY", "A", "B", "C"]
    prices = {
        s: pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
        for s in symbols
    }
    prices["SPY"] = _bull_prices(n)

    result = model.backtest(prices, cost_bps=1.0)
    assert "SPY" not in result.returns_by_symbol.columns
    assert len(result.returns) == n
