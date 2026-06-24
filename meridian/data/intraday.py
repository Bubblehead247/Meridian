"""Intraday OHLCV loading.

Daily loading lives in ``loader.py``; intraday has different source constraints
and needs frequency-aware annualization, so it gets its own module.

Free intraday sources (verified):
- **yfinance** — hourly (``1h``) bars go back ~730 days (the only free, no-keys,
  walk-forward-viable option); 5m/15m only ~60 days, 1m only ~7 days. yfinance's
  intraday API is **period-based** (``period="730d"``), unlike the daily path
  which uses ``period="max"`` (rejected for intraday).
- **Alpaca** (`alpaca-py`) — proper 1-minute bars over years (IEX feed on the
  free tier), but needs the user's free API keys. ``AlpacaDataLoader`` is built
  here and runs when ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` are set.

Both normalize to the canonical schema (``normalize_ohlcv``); the DatetimeIndex
keeps the time-of-day, which the portfolio backtester uses to flatten overnight.
"""

from __future__ import annotations

import math
import os

import pandas as pd

from meridian.data.cache import OHLCVCache
from meridian.data.loader import _flatten_columns, _slice
from meridian.data.schema import normalize_ohlcv

# Minutes per bar for each interval; one US equity session is 390 minutes.
_INTERVAL_MINUTES = {
    "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
    "60m": 60, "1h": 60, "90m": 90, "4h": 240, "1d": 390,
}
_MINUTES_PER_SESSION = 390
_SESSIONS_PER_YEAR = 252


def bars_per_year(interval: str, *, sessions_per_year: int = _SESSIONS_PER_YEAR) -> int:
    """Number of bars in a trading year for ``interval`` — the annualization
    factor to pass as ``periods_per_year`` (e.g. 1h ≈ 1764, 5m ≈ 19656)."""
    if interval not in _INTERVAL_MINUTES:
        raise ValueError(f"unknown interval {interval!r}. Known: {', '.join(_INTERVAL_MINUTES)}")
    if interval == "1d":
        return sessions_per_year
    bars_per_session = math.ceil(_MINUTES_PER_SESSION / _INTERVAL_MINUTES[interval])
    return bars_per_session * sessions_per_year


def _download_intraday(symbol: str, interval: str, lookback: str) -> pd.DataFrame:
    """Fetch raw intraday OHLCV from yfinance (period-based). Isolated for tests."""
    import yfinance as yf

    return yf.download(
        symbol, period=lookback, interval=interval,
        auto_adjust=False, progress=False,
    )


def load_intraday(
    symbol: str,
    interval: str = "1h",
    *,
    lookback: str = "730d",
    start: str | None = None,
    end: str | None = None,
    source: str = "yfinance",
    use_cache: bool = True,
    cache: OHLCVCache | None = None,
) -> pd.DataFrame:
    """Load normalized intraday OHLCV for one symbol.

    Args:
        symbol: Ticker.
        interval: Bar size (``1h`` default; ``1m`` etc. for Alpaca).
        lookback: yfinance period window (e.g. ``"730d"``).
        start/end: Optional inclusive slice applied after loading.
        source: ``"yfinance"`` or ``"alpaca"``.
        use_cache/cache: on-disk cache (keyed per interval).

    Returns:
        OHLCV frame in canonical schema, DatetimeIndex with time-of-day.
    """
    cache = cache or OHLCVCache()

    if source == "alpaca":
        df = AlpacaDataLoader().load(symbol, interval=interval, lookback=lookback)
        if use_cache:
            cache.write(symbol, interval, df)
        return _slice(df, start, end)

    if source != "yfinance":
        raise ValueError(f"unknown intraday source {source!r} (use 'yfinance' or 'alpaca')")

    df = cache.read(symbol, interval) if use_cache else None
    if df is None:
        raw = _download_intraday(symbol, interval, lookback)
        if raw is None or raw.empty:
            raise ValueError(f"No intraday data returned for {symbol!r} ({interval})")
        df = normalize_ohlcv(_flatten_columns(raw))
        if use_cache:
            cache.write(symbol, interval, df)
    return _slice(df, start, end)


def load_intraday_universe(
    symbols: tuple[str, ...] | list[str],
    interval: str = "1h",
    *,
    lookback: str = "730d",
    source: str = "yfinance",
    field: str = "close",
    use_cache: bool = True,
    cache: OHLCVCache | None = None,
) -> dict[str, pd.Series]:
    """Load intraday ``field`` series for many symbols (skips ones that fail)."""
    cache = cache or OHLCVCache()
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            out[sym] = load_intraday(
                sym, interval, lookback=lookback, source=source,
                use_cache=use_cache, cache=cache,
            )[field]
        except (ValueError, KeyError):
            continue
    return out


class AlpacaDataLoader:
    """Historical intraday bars from Alpaca (`alpaca-py`).

    Credentials from the constructor or ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``
    (same pattern as ``meridian.execution.broker.AlpacaBroker``). alpaca-py is
    imported lazily; the free tier serves IEX-feed bars over several years.
    """

    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "alpaca-py is required for AlpacaDataLoader. Install with `pip install alpaca-py`."
            ) from exc

        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise ValueError(
                "Alpaca credentials missing: pass api_key/secret_key or set "
                "ALPACA_API_KEY / ALPACA_SECRET_KEY."
            )
        self._client = StockHistoricalDataClient(key, secret)

    def load(  # pragma: no cover - needs live API
        self, symbol: str, interval: str = "1m", lookback: str = "730d", feed: str = "iex"
    ) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        unit = {"1m": (1, TimeFrameUnit.Minute), "5m": (5, TimeFrameUnit.Minute),
                "15m": (15, TimeFrameUnit.Minute), "1h": (1, TimeFrameUnit.Hour)}[interval]
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(lookback.replace("d", " days"))
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame(*unit), start=start,
            feed=DataFeed(feed),  # free tier serves IEX
        )
        raw = self._client.get_stock_bars(req).df
        if raw is None or raw.empty:
            raise ValueError(f"No Alpaca data for {symbol!r} ({interval})")
        df = raw.xs(symbol, level="symbol") if "symbol" in raw.index.names else raw
        # Alpaca has no adjusted close; use close (splits are rare intraday).
        df = df.assign(adj_close=df["close"])
        return normalize_ohlcv(df)
