"""Tests for crypto loading and 24/7 annualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.data import crypto as crypto_mod
from meridian.data import crypto_bars_per_year, load_crypto, load_crypto_universe
from meridian.data.cache import OHLCVCache


def _raw_candles(n=40, start_ms=1_700_000_000_000, step_ms=86_400_000):
    """Synthetic Crypto.com candle dicts (ms epoch, string-ish OHLCV)."""
    base = np.linspace(100, 110, n)
    return [
        {
            "t": start_ms + i * step_ms,
            "o": float(base[i]), "h": float(base[i] + 1),
            "l": float(base[i] - 1), "c": float(base[i]),
            "v": float(i * 10),
        }
        for i in range(n)
    ]


# --- annualization --------------------------------------------------------

def test_crypto_bars_per_year_values():
    assert crypto_bars_per_year("1D") == 365      # 24/7, daily
    assert crypto_bars_per_year("1d") == 365
    assert crypto_bars_per_year("4h") == 365 * 6  # 2190
    assert crypto_bars_per_year("1h") == 365 * 24  # 8760


def test_crypto_bars_per_year_unknown_raises():
    with pytest.raises(ValueError, match="unknown crypto interval"):
        crypto_bars_per_year("3s")


# --- loader (offline) -----------------------------------------------------

def test_load_crypto_downloads_caches_and_normalizes(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake(instrument, interval, count):
        calls["n"] += 1
        return _raw_candles()

    monkeypatch.setattr(crypto_mod, "_download_crypto_candles", fake)
    cache = OHLCVCache(tmp_path)

    df = load_crypto("BTC_USDT", "1D", cache=cache)
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    # no splits in crypto -> adj_close mirrors close
    assert (df["adj_close"] == df["close"]).all()
    assert df.index.is_monotonic_increasing
    assert calls["n"] == 1

    load_crypto("BTC_USDT", "1D", cache=cache)  # second call hits the cache
    assert calls["n"] == 1


def test_load_crypto_paginates_backward_for_history(tmp_path, monkeypatch):
    """With a ``start`` date the loader walks ``end_ts`` back across pages."""
    day = 86_400_000
    newest = 1_700_000_000_000

    def fake(instrument, interval, count, end_ts=None):
        # 300-candle pages ending at end_ts (newest page when end_ts is None)
        top = newest if end_ts is None else end_ts
        return [
            {"t": top - i * day, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0}
            for i in range(300)
        ]

    monkeypatch.setattr(crypto_mod, "_download_crypto_candles", fake)
    cache = OHLCVCache(tmp_path)
    # ask for ~2 pages of history (~600 days back)
    start = pd.Timestamp(newest - 500 * day, unit="ms").strftime("%Y-%m-%d")
    df = load_crypto("BTC_USDT", "1D", start=start, cache=cache)
    assert len(df) > 300                       # more than one page assembled
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates


def test_load_crypto_empty_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(crypto_mod, "_download_crypto_candles", lambda *a: [])
    with pytest.raises(ValueError, match="No crypto data"):
        load_crypto("BAD_USDT", "1D", cache=OHLCVCache(tmp_path))


def test_load_crypto_universe_skips_failures(tmp_path, monkeypatch):
    def fake(instrument, interval, count):
        if instrument == "BAD_USDT":
            return []
        return _raw_candles()

    monkeypatch.setattr(crypto_mod, "_download_crypto_candles", fake)
    px = load_crypto_universe(
        ["BTC_USDT", "BAD_USDT", "ETH_USDT"], "1D", cache=OHLCVCache(tmp_path)
    )
    assert set(px) == {"BTC_USDT", "ETH_USDT"}
    assert all(isinstance(s, pd.Series) for s in px.values())
