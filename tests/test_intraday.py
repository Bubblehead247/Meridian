"""Tests for intraday loading, annualization, and session handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.data import AlpacaDataLoader, bars_per_year, load_intraday
from meridian.data import intraday as intraday_mod
from meridian.data.cache import OHLCVCache


def _raw_intraday(n=40, start="2024-01-02 09:30") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h", name="Datetime")
    base = np.linspace(100, 110, n)
    return pd.DataFrame(
        {"Open": base, "High": base + 1, "Low": base - 1, "Close": base,
         "Adj Close": base, "Volume": np.arange(n) * 100},
        index=idx,
    )


# --- annualization --------------------------------------------------------

def test_bars_per_year_values():
    assert bars_per_year("1d") == 252
    assert bars_per_year("1h") == 7 * 252        # ceil(390/60)=7
    assert bars_per_year("5m") == 78 * 252       # ceil(390/5)=78
    assert bars_per_year("1m") == 390 * 252


def test_bars_per_year_unknown_raises():
    with pytest.raises(ValueError):
        bars_per_year("3s")


# --- loader (offline) -----------------------------------------------------

def test_load_intraday_downloads_caches_and_keeps_time(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake(symbol, interval, lookback):
        calls["n"] += 1
        return _raw_intraday()

    monkeypatch.setattr(intraday_mod, "_download_intraday", fake)
    cache = OHLCVCache(tmp_path)

    df = load_intraday("AAPL", "1h", cache=cache)
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    # time-of-day preserved (not normalized to midnight)
    assert (df.index.hour != 0).any()
    assert calls["n"] == 1

    load_intraday("AAPL", "1h", cache=cache)  # second call hits cache
    assert calls["n"] == 1


def test_load_intraday_empty_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(intraday_mod, "_download_intraday", lambda *a: pd.DataFrame())
    with pytest.raises(ValueError, match="No intraday data"):
        load_intraday("BAD", "1h", cache=OHLCVCache(tmp_path))


def test_unknown_source_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown intraday source"):
        load_intraday("AAPL", "1h", source="bogus", cache=OHLCVCache(tmp_path))


def test_alpaca_loader_requires_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="credentials"):
        AlpacaDataLoader()
