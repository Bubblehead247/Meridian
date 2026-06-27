"""Crypto OHLCV loading (Crypto.com public market data).

Equities and crypto differ in two ways that matter to the pipeline:

- **No keys, different API.** Crypto.com's public *candlestick* endpoint serves
  OHLCV with no authentication, so unlike Alpaca this needs no credentials. The
  raw HTTP call is isolated in ``_download_crypto_candles`` so tests stub it and
  run fully offline (the same pattern as ``loader._download`` /
  ``intraday._download_intraday``).
- **24/7 trading.** Crypto never closes, so a "year" is 365 days, not 252
  sessions — ``crypto_bars_per_year`` gives the frequency-correct annualization
  factor to pass as ``periods_per_year`` (daily → 365, 4h → 2190, …), distinct
  from ``intraday.bars_per_year`` which assumes a 252-session, 390-minute day.

Everything normalizes to the canonical schema (``normalize_ohlcv``) and caches
through the existing ``OHLCVCache``, so the rest of the platform treats a crypto
series exactly like an equity series.
"""

from __future__ import annotations

import pandas as pd

from meridian.data.cache import OHLCVCache
from meridian.data.loader import _slice
from meridian.data.schema import normalize_ohlcv

#: Crypto.com v1 public candlestick endpoint (no auth required).
_CANDLE_URL = "https://api.crypto.com/exchange/v1/public/get-candlestick"

#: How many days/year each interval covers — crypto trades 365 days, 24h.
_INTERVAL_HOURS = {
    "1m": 1 / 60, "5m": 5 / 60, "15m": 15 / 60, "30m": 0.5,
    "1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1D": 24, "1d": 24,
}
_DAYS_PER_YEAR = 365
_HOURS_PER_DAY = 24


def crypto_bars_per_year(interval: str, *, days_per_year: int = _DAYS_PER_YEAR) -> int:
    """Number of bars in a crypto trading year for ``interval``.

    Crypto markets run 24/7, so a daily bar annualizes by 365 (not the 252
    equity sessions) and a 4h bar by ``365 * 6 = 2190``. Pass the result as
    ``periods_per_year`` so Sharpe is scaled at the right frequency.
    """
    if interval not in _INTERVAL_HOURS:
        raise ValueError(
            f"unknown crypto interval {interval!r}. Known: {', '.join(_INTERVAL_HOURS)}"
        )
    bars_per_day = _HOURS_PER_DAY / _INTERVAL_HOURS[interval]
    return round(bars_per_day * days_per_year)


def _download_crypto_candles(
    instrument: str, interval: str, count: int, end_ts: int | None = None
) -> list[dict]:
    """Fetch one page (≤300) of raw candlesticks from Crypto.com's public REST API.

    Isolated for tests. Returns the ``result.data`` list of ``{t, o, h, l, c, v}``
    dicts (``t`` is a millisecond epoch). ``end_ts`` (ms) caps the newest candle,
    which is how history is paginated backward. Raises on a non-OK response.
    """
    import requests

    params: dict = {"instrument_name": instrument, "timeframe": interval, "count": count}
    if end_ts is not None:
        params["end_ts"] = end_ts
    resp = requests.get(_CANDLE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") not in (0, "0", None):
        raise ValueError(f"Crypto.com error for {instrument!r}: {payload.get('message', payload)}")
    return payload.get("result", {}).get("data", [])


def _download_crypto_history(instrument: str, interval: str, start_ms: int) -> list[dict]:
    """Paginate backward (300-candle pages via ``end_ts``) until ``start_ms`` is covered.

    The public endpoint caps each call at ~300 candles, so multi-year history is
    assembled by walking ``end_ts`` back to the oldest candle of the prior page.
    Stops when a page reaches ``start_ms``, returns empty, or makes no progress.
    """
    by_ts: dict[int, dict] = {}
    end_ts: int | None = None
    while True:
        page = _download_crypto_candles(instrument, interval, 300, end_ts=end_ts)
        if not page:
            break
        oldest = min(int(c["t"]) for c in page)
        for c in page:
            by_ts[int(c["t"])] = c
        if oldest <= start_ms or (end_ts is not None and oldest >= end_ts):
            break
        end_ts = oldest - 1
    return [by_ts[t] for t in sorted(by_ts)]


def _candles_to_frame(candles: list[dict]) -> pd.DataFrame:
    """Map Crypto.com candle dicts to the canonical OHLCV schema.

    Crypto has no splits/dividends, so ``adj_close = close``.
    """
    rows = []
    for c in candles:
        rows.append(
            {
                "open": float(c["o"]), "high": float(c["h"]), "low": float(c["l"]),
                "close": float(c["c"]), "adj_close": float(c["c"]),
                "volume": float(c.get("v", 0.0)),
            }
        )
    idx = pd.to_datetime([int(c["t"]) for c in candles], unit="ms")
    return normalize_ohlcv(pd.DataFrame(rows, index=idx))


def load_crypto(
    instrument: str,
    interval: str = "1D",
    *,
    count: int = 1000,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
    cache: OHLCVCache | None = None,
) -> pd.DataFrame:
    """Load normalized crypto OHLCV for one instrument (e.g. ``"BTC_USDT"``).

    Args:
        instrument: Crypto.com instrument name, ``BASE_QUOTE`` (e.g. ``BTC_USDT``).
        interval: Candle timeframe (``"1D"`` default; ``"4h"``, ``"1h"`` …).
        count: Max candles for a single-page pull (the endpoint caps each call
            at ~300). Ignored when ``start`` is given.
        start: If set, paginate backward to assemble history from this date
            (``YYYY-MM-DD``). Otherwise one page (most recent ~300 candles).
        end: Optional inclusive upper slice applied after loading.
        use_cache/cache: on-disk Parquet cache (keyed per interval).

    Returns:
        OHLCV frame in canonical schema with a DatetimeIndex.
    """
    cache = cache or OHLCVCache()

    df = cache.read(instrument, interval) if use_cache else None
    if df is None:
        if start is not None:
            start_ms = int(pd.Timestamp(start).timestamp() * 1000)
            candles = _download_crypto_history(instrument, interval, start_ms)
        else:
            candles = _download_crypto_candles(instrument, interval, count)
        if not candles:
            raise ValueError(f"No crypto data returned for {instrument!r} ({interval})")
        df = _candles_to_frame(candles)
        if use_cache:
            cache.write(instrument, interval, df)
    return _slice(df, start, end)


def load_crypto_universe(
    instruments: tuple[str, ...] | list[str],
    interval: str = "1D",
    *,
    count: int = 1000,
    start: str | None = None,
    field: str = "close",
    use_cache: bool = True,
    cache: OHLCVCache | None = None,
) -> dict[str, pd.Series]:
    """Load the ``field`` series for many instruments (skips ones that fail)."""
    cache = cache or OHLCVCache()
    out: dict[str, pd.Series] = {}
    for inst in instruments:
        try:
            out[inst] = load_crypto(
                inst, interval, count=count, start=start, use_cache=use_cache, cache=cache
            )[field]
        except (ValueError, KeyError):
            continue
    return out
