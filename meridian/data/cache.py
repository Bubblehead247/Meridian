"""On-disk caching for market data.

Pulling from yfinance is slow and rate-limited, so every download is cached to
disk and reused. Two caches live here:

- ``OHLCVCache``  — one Parquet file per (symbol, interval) of price bars.
- ``UniverseCache`` — one CSV per index of its constituent tickers.

The cache directory defaults to ``data_cache/`` (git-ignored). Caching is a
pure performance concern: clearing the cache must never change results, only
re-trigger downloads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from meridian.data.schema import normalize_ohlcv

DEFAULT_CACHE_DIR = Path("data_cache")


class OHLCVCache:
    """Parquet-backed cache of OHLCV frames, keyed by symbol and interval."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.root = Path(cache_dir) / "ohlcv"

    def path(self, symbol: str, interval: str) -> Path:
        safe = symbol.upper().replace("/", "_")
        return self.root / interval / f"{safe}.parquet"

    def has(self, symbol: str, interval: str) -> bool:
        return self.path(symbol, interval).exists()

    def read(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Return the cached frame, or None if not cached."""
        p = self.path(symbol, interval)
        if not p.exists():
            return None
        return normalize_ohlcv(pd.read_parquet(p))

    def write(self, symbol: str, interval: str, df: pd.DataFrame) -> Path:
        """Write a frame to the cache (normalized first) and return its path."""
        p = self.path(symbol, interval)
        p.parent.mkdir(parents=True, exist_ok=True)
        normalize_ohlcv(df).to_parquet(p)
        return p


class UniverseCache:
    """CSV-backed cache of index constituent tickers."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.root = Path(cache_dir) / "universes"

    def path(self, name: str) -> Path:
        return self.root / f"{name.upper()}.csv"

    def read(self, name: str) -> tuple[str, ...] | None:
        """Return cached symbols for an index, or None if not cached."""
        p = self.path(name)
        if not p.exists():
            return None
        series = pd.read_csv(p)["symbol"].astype(str)
        return tuple(series)

    def write(self, name: str, symbols: tuple[str, ...]) -> Path:
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"symbol": list(symbols)}).to_csv(p, index=False)
        return p
