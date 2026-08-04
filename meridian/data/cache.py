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

import time
from pathlib import Path

import pandas as pd

from meridian.data.schema import normalize_ohlcv

DEFAULT_CACHE_DIR = Path("data_cache")


class OHLCVCache:
    """Parquet-backed cache of OHLCV frames, keyed by symbol and interval.

    Caching is a pure performance concern: clearing the cache must never
    change results, only re-trigger downloads. ``max_age_days`` preserves
    that — it only automates *when* a re-trigger happens (treating a stale
    file as a miss), it never changes what a fresh download would produce.
    Historical/research callers should leave it unset (the default,
    unchanged behavior); live/paper callers that need today's bars can opt in.
    """

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR, max_age_days: int | None = None):
        self.root = Path(cache_dir) / "ohlcv"
        self.max_age_days = max_age_days

    def path(self, symbol: str, interval: str) -> Path:
        safe = symbol.upper().replace("/", "_")
        return self.root / interval / f"{safe}.parquet"

    def has(self, symbol: str, interval: str) -> bool:
        return self.path(symbol, interval).exists()

    def has_fresh(self, symbol: str, interval: str, max_age_days: int | None = None) -> bool:
        """Whether a cached file exists and is within ``max_age_days`` of now.

        Args:
            max_age_days: Overrides the instance default; ``None`` (either
                here or on the instance) means "no TTL" — any existing file
                counts as fresh, matching the pre-TTL behavior.
        """
        p = self.path(symbol, interval)
        if not p.exists():
            return False
        age_limit = max_age_days if max_age_days is not None else self.max_age_days
        if age_limit is None:
            return True
        age_seconds = time.time() - p.stat().st_mtime
        return age_seconds <= age_limit * 86400

    def read(self, symbol: str, interval: str, max_age_days: int | None = None) -> pd.DataFrame | None:
        """Return the cached frame, or None if not cached or stale.

        Args:
            max_age_days: Overrides the instance default TTL for this call.
        """
        if not self.has_fresh(symbol, interval, max_age_days):
            return None
        return normalize_ohlcv(pd.read_parquet(self.path(symbol, interval)))

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
