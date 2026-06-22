"""Price data ingestion.

Loads daily OHLCV bars from yfinance, normalizes them to the canonical schema,
and serves them through the on-disk cache. This is the single entry point the
rest of the platform uses to obtain price history — no other module calls
yfinance directly.

The raw download is isolated in ``_download`` so tests can replace it with
synthetic data and run fully offline.
"""

from __future__ import annotations

import pandas as pd

from meridian.data.cache import OHLCVCache
from meridian.data.schema import normalize_ohlcv


def _download(symbol: str, start: str | None, end: str | None, interval: str) -> pd.DataFrame:
    """Fetch raw OHLCV from yfinance. Isolated for testability.

    The cache stores full history and slices afterward, so when no explicit
    window is given we pull ``period="max"`` rather than yfinance's short
    default (one month). An explicit start/end, if provided, takes precedence.
    """
    import yfinance as yf

    kwargs: dict = dict(interval=interval, auto_adjust=False, progress=False)
    if start is None and end is None:
        kwargs["period"] = "max"
    else:
        kwargs["start"] = start
        kwargs["end"] = end

    return yf.download(symbol, **kwargs)


def load_ohlcv(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
    cache: OHLCVCache | None = None,
) -> pd.DataFrame:
    """Load normalized OHLCV bars for one symbol.

    On a cache hit the full cached history is read and then sliced to
    ``[start, end]``. On a miss the data is downloaded, cached in full, and
    sliced. Caching the full pull (not the slice) means overlapping date ranges
    reuse one file.

    Args:
        symbol: Ticker, e.g. ``"SPY"``.
        start: Inclusive start date (``YYYY-MM-DD``) or None for all available.
        end: Inclusive end date or None for latest.
        interval: Bar size; daily (``1d``) is the platform default.
        use_cache: Read from / write to the on-disk cache.
        cache: Optional cache instance; a default-located one is used if None.

    Returns:
        OHLCV frame in canonical schema, sliced to the requested window.
    """
    cache = cache or OHLCVCache()

    df: pd.DataFrame | None = None
    if use_cache:
        df = cache.read(symbol, interval)

    if df is None:
        raw = _download(symbol, start=None, end=None, interval=interval)
        if raw is None or raw.empty:
            raise ValueError(f"No data returned for {symbol!r}")
        df = normalize_ohlcv(_flatten_columns(raw))
        if use_cache:
            cache.write(symbol, interval, df)

    return _slice(df, start, end)


def load_universe(
    symbols: tuple[str, ...] | list[str],
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
    cache: OHLCVCache | None = None,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV for many symbols, returning ``{symbol: frame}``.

    Symbols that fail to load are skipped (logged via the raised message being
    swallowed) so one bad ticker does not abort a whole universe pull.
    """
    cache = cache or OHLCVCache()
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = load_ohlcv(
                sym, start=start, end=end, interval=interval, use_cache=use_cache, cache=cache
            )
        except ValueError:
            continue
    return out


def _flatten_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop the ticker level yfinance adds when columns are a MultiIndex."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)
    return raw


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Return rows within [start, end] inclusive (None means open-ended)."""
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df
