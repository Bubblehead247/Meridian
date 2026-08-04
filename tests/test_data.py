"""Phase 1 tests for the data layer. All offline — no network calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.data import (
    OHLCVCache,
    SplitSpec,
    UniverseCache,
    get_universe,
    load_ohlcv,
    load_universe,
    normalize_ohlcv,
    split,
)
from meridian.data import loader as loader_mod
from meridian.data import universe as universe_mod


def _raw_yf_frame(n: int = 30, start: str = "2020-01-01") -> pd.DataFrame:
    """A synthetic yfinance-style frame (capitalized cols, Adj Close)."""
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


# --- schema ---------------------------------------------------------------

def test_normalize_renames_and_orders_columns():
    out = normalize_ohlcv(_raw_yf_frame())
    assert list(out.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert out.index.name == "date"
    assert out.index.is_monotonic_increasing


def test_normalize_missing_column_raises():
    df = _raw_yf_frame().drop(columns=["Volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_ohlcv(df)


# --- OHLCV cache ----------------------------------------------------------

def test_ohlcv_cache_roundtrip(tmp_path):
    cache = OHLCVCache(tmp_path)
    assert not cache.has("SPY", "1d")
    cache.write("SPY", "1d", _raw_yf_frame())
    assert cache.has("SPY", "1d")
    got = cache.read("SPY", "1d")
    assert list(got.columns) == ["open", "high", "low", "close", "adj_close", "volume"]


def test_ohlcv_cache_miss_returns_none(tmp_path):
    assert OHLCVCache(tmp_path).read("NOPE", "1d") is None


# --- cache TTL --------------------------------------------------------------

def test_max_age_none_never_expires(tmp_path):
    """Default behavior unchanged: no TTL means any existing file is fresh."""
    cache = OHLCVCache(tmp_path)  # max_age_days=None
    cache.write("SPY", "1d", _raw_yf_frame())
    assert cache.has_fresh("SPY", "1d")
    assert cache.read("SPY", "1d") is not None


def test_stale_cache_is_treated_as_a_miss(tmp_path, monkeypatch):
    import os
    import time

    cache = OHLCVCache(tmp_path, max_age_days=1)
    cache.write("SPY", "1d", _raw_yf_frame())
    p = cache.path("SPY", "1d")
    old = time.time() - 2 * 86400  # 2 days old
    os.utime(p, (old, old))

    assert not cache.has_fresh("SPY", "1d")
    assert cache.read("SPY", "1d") is None


def test_fresh_cache_within_ttl_is_used(tmp_path):
    cache = OHLCVCache(tmp_path, max_age_days=7)
    cache.write("SPY", "1d", _raw_yf_frame())
    assert cache.has_fresh("SPY", "1d")
    assert cache.read("SPY", "1d") is not None


def test_per_call_max_age_overrides_instance_default(tmp_path):
    import os
    import time

    cache = OHLCVCache(tmp_path)  # no instance TTL
    cache.write("SPY", "1d", _raw_yf_frame())
    p = cache.path("SPY", "1d")
    old = time.time() - 2 * 86400
    os.utime(p, (old, old))

    assert cache.has_fresh("SPY", "1d")                       # instance default: no TTL
    assert not cache.has_fresh("SPY", "1d", max_age_days=1)    # per-call override: stale
    assert cache.read("SPY", "1d", max_age_days=1) is None


def test_load_ohlcv_max_age_days_forces_redownload_when_stale(tmp_path, monkeypatch):
    import os
    import time

    calls = {"n": 0}

    def fake_download(symbol, start, end, interval):
        calls["n"] += 1
        return _raw_yf_frame(n=40, start="2020-01-01")

    monkeypatch.setattr(loader_mod, "_download", fake_download)
    cache = OHLCVCache(tmp_path)

    load_ohlcv("SPY", cache=cache)
    assert calls["n"] == 1

    old = time.time() - 2 * 86400
    os.utime(cache.path("SPY", "1d"), (old, old))

    load_ohlcv("SPY", cache=cache, max_age_days=1)  # stale -> re-fetches
    assert calls["n"] == 2

    load_ohlcv("SPY", cache=cache)  # no TTL passed -> uses the (now fresh) cache
    assert calls["n"] == 2


# --- loader ---------------------------------------------------------------

def test_load_ohlcv_downloads_caches_and_slices(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_download(symbol, start, end, interval):
        calls["n"] += 1
        return _raw_yf_frame(n=40, start="2020-01-01")

    monkeypatch.setattr(loader_mod, "_download", fake_download)
    cache = OHLCVCache(tmp_path)

    df = load_ohlcv("SPY", start="2020-01-10", end="2020-01-20", cache=cache)
    assert df.index.min() >= pd.Timestamp("2020-01-10")
    assert df.index.max() <= pd.Timestamp("2020-01-20")
    assert calls["n"] == 1

    # Second call hits cache: download not called again.
    load_ohlcv("SPY", start="2020-01-05", cache=cache)
    assert calls["n"] == 1


def test_load_ohlcv_empty_download_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_mod, "_download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="No data"):
        load_ohlcv("BAD", cache=OHLCVCache(tmp_path))


def test_load_universe_skips_bad_symbols(tmp_path, monkeypatch):
    def fake_download(symbol, start, end, interval):
        if symbol == "BAD":
            return pd.DataFrame()
        return _raw_yf_frame()

    monkeypatch.setattr(loader_mod, "_download", fake_download)
    out = load_universe(["SPY", "BAD", "QQQ"], cache=OHLCVCache(tmp_path))
    assert set(out) == {"SPY", "QQQ"}


# --- universe -------------------------------------------------------------

def test_etf_universe_is_static():
    u = get_universe("spy")
    assert u.name == "SPY"
    assert u.symbols == ("SPY",)
    assert len(u) == 1


def test_unknown_universe_raises():
    with pytest.raises(KeyError):
        get_universe("DOW30")


def test_index_universe_fetches_then_caches(tmp_path, monkeypatch):
    fetches = {"n": 0}

    def fake_fetch(url, col):
        fetches["n"] += 1
        return ("AAPL", "MSFT", "BRK-B")

    monkeypatch.setattr(universe_mod, "_fetch_constituents", fake_fetch)
    cache = UniverseCache(tmp_path)

    u1 = get_universe("SP500", cache=cache)
    assert u1.symbols == ("AAPL", "MSFT", "BRK-B")
    assert fetches["n"] == 1

    # Second resolve reads the cache instead of refetching.
    u2 = get_universe("SP500", cache=cache)
    assert u2.symbols == ("AAPL", "MSFT", "BRK-B")
    assert fetches["n"] == 1


def test_symbol_normalization():
    assert universe_mod._normalize_symbol("brk.b") == "BRK-B"


# --- splits ---------------------------------------------------------------

def _full_series() -> pd.Series:
    idx = pd.date_range("2010-01-01", "2024-12-31", freq="D", name="date")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)

def test_split_partitions_by_default_windows():
    parts = split(_full_series())
    assert parts["in_sample"].index.min() >= pd.Timestamp("2010-01-01")
    assert parts["in_sample"].index.max() <= pd.Timestamp("2019-12-31")
    assert parts["out_of_sample"].index.min() >= pd.Timestamp("2020-01-01")
    assert parts["out_of_sample"].index.max() <= pd.Timestamp("2022-12-31")
    assert parts["walk_forward"].index.min() >= pd.Timestamp("2023-01-01")


def test_split_windows_do_not_overlap():
    parts = split(_full_series())
    is_idx = set(parts["in_sample"].index)
    oos_idx = set(parts["out_of_sample"].index)
    wf_idx = set(parts["walk_forward"].index)
    assert is_idx.isdisjoint(oos_idx)
    assert oos_idx.isdisjoint(wf_idx)


def test_splitspec_from_config_and_validation():
    spec = SplitSpec.from_config(
        {
            "in_sample": ["2010-01-01", "2019-12-31"],
            "out_of_sample": ["2020-01-01", "2022-12-31"],
            "walk_forward_start": "2023-01-01",
        }
    )
    spec.validate()  # should not raise


def test_splitspec_overlapping_windows_raise():
    bad = SplitSpec(
        in_sample=("2010-01-01", "2021-01-01"),  # overlaps OOS
        out_of_sample=("2020-01-01", "2022-12-31"),
        walk_forward_start="2023-01-01",
    )
    with pytest.raises(ValueError):
        bad.validate()
