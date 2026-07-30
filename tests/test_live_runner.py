"""Tests that the live price-fetch path bypasses the on-disk OHLCV cache.

Regression coverage for the bug where ``_fetch_prices`` inherited whatever
was already cached on disk, so live sessions kept trading and notifying off
a stale snapshot once the cache stopped being refreshed. All offline — no
network calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.data import loader as loader_mod
from meridian.data.cache import OHLCVCache
from meridian.execution import live_runner


def _synthetic_frame(n: int, start: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="D", name="Date")
    base = np.linspace(100, 130, n)
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 1,
            "Low": base - 1,
            "Close": base,
            "Adj Close": base * 0.99,
            "Volume": np.arange(n) * 1000,
        },
        index=idx,
    )


def test_fetch_prices_ignores_stale_cache_and_does_not_write_it(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Seed a stale cache entry, as if the last live session ran weeks ago.
    stale_cache = OHLCVCache()
    stale_frame = stale_cache.write(
        "SPY", "1d", _synthetic_frame(30, start="2020-01-01")
    )
    stale_mtime = stale_frame.stat().st_mtime

    calls = {"n": 0}

    def fake_download(symbol, start, end, interval):
        calls["n"] += 1
        return _synthetic_frame(5, start="2024-06-01")

    monkeypatch.setattr(loader_mod, "_download", fake_download)

    out = live_runner._fetch_prices(["SPY"], start="2020-01-01")

    # Fresh data came back, not the stale cached snapshot.
    assert calls["n"] == 1
    assert out["SPY"].index.max() == pd.Timestamp("2024-06-05")

    # The stale cache file was neither read as truth nor overwritten.
    assert stale_frame.stat().st_mtime == stale_mtime
    assert OHLCVCache().read("SPY", "1d").index.max() == pd.Timestamp("2020-01-30")


def test_fetch_prices_skips_symbol_on_download_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def failing_download(symbol, start, end, interval):
        raise ConnectionError("network unavailable")

    monkeypatch.setattr(loader_mod, "_download", failing_download)

    out = live_runner._fetch_prices(["SPY"], start="2020-01-01")

    assert out == {}
