"""Index-level regime labeler for daily trend, volatility, and breadth dimensions.

This module owns daily index-level trend/vol/breadth labels; it does NOT own
single-series rolling classifiers (those stay in regimes/classifiers.py).

The three dimensions (per CLAUDE.md / PLAN.md §4):

- **Trend** (SPY by default): ``bear`` if close < 200-day MA; ``bull`` if close ≥
  200-day MA *and* ADX > 25 (a confirmed up-trend); ``neutral`` otherwise (above the
  MA but no trend strength — this subsumes the "near the MA" case).
- **Volatility** (``^VIX`` close): ``low`` < 15, ``normal`` 15–20, ``elevated`` 20–30,
  ``extreme`` > 30.
- **Breadth** (% of S&P 500 above their own 200-day MA): ``expansion`` > 60,
  ``neutral`` 40–60, ``contraction`` < 40.

A pure core (`regime_frame`) computes labels from in-memory series and is fully
offline-testable; `build_regime_frame` is the network orchestrator that loads SPY/VIX
and the constituent breadth via the existing data loaders. `attach_regimes` left-joins
the labels onto any backtest/signal date index for downstream gating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

UNKNOWN = "unknown"
TREND_LABELS = ("bull", "neutral", "bear")
VOL_LABELS = ("low", "normal", "elevated", "extreme")
BREADTH_LABELS = ("expansion", "neutral", "contraction")

# --- default thresholds (all overridable) ------------------------------------
MA_WINDOW = 200
ADX_WINDOW = 14
ADX_BULL = 25.0
VIX_LOW, VIX_NORMAL, VIX_ELEVATED = 15.0, 20.0, 30.0
BREADTH_EXPANSION, BREADTH_CONTRACTION = 60.0, 40.0


@dataclass(frozen=True)
class RegimeLabel:
    """One day's regime across the three dimensions."""

    trend: str
    volatility: str
    breadth: str


# --- indicator + per-dimension labelers (pure, vectorized) -------------------

def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = ADX_WINDOW) -> pd.Series:
    """Wilder's Average Directional Index — trend *strength* (not direction).

    Uses Wilder smoothing (RMA = ``ewm(alpha=1/window)``) of the directional movement
    and true range. NaN until enough history has accumulated.
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    def rma(s) -> pd.Series:  # Wilder smoothing
        return pd.Series(s, index=close.index).ewm(alpha=1 / window, adjust=False).mean()

    atr = rma(tr)
    plus_di = 100 * rma(plus_dm) / atr
    minus_di = 100 * rma(minus_dm) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean()


def trend_label(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    ma_window: int = MA_WINDOW,
    adx_window: int = ADX_WINDOW,
    adx_bull: float = ADX_BULL,
) -> pd.Series:
    """bull / neutral / bear from the 200-day MA side and ADX strength."""
    ma = close.rolling(ma_window).mean()
    strength = adx(high, low, close, adx_window)
    out = pd.Series(UNKNOWN, index=close.index, dtype=object)
    ready = ma.notna() & strength.notna()
    below = ready & (close < ma)
    bull = ready & (close >= ma) & (strength > adx_bull)
    out[ready] = "neutral"
    out[bull] = "bull"
    out[below] = "bear"
    return out


def vol_label(
    vix: pd.Series,
    *,
    low: float = VIX_LOW,
    normal: float = VIX_NORMAL,
    elevated: float = VIX_ELEVATED,
) -> pd.Series:
    """low / normal / elevated / extreme from VIX closes."""
    out = pd.Series(UNKNOWN, index=vix.index, dtype=object)
    v = vix.astype(float)
    out[v.notna()] = "extreme"
    out[v <= elevated] = "elevated"
    out[v < normal] = "normal"
    out[v < low] = "low"
    return out


def breadth_pct(prices_by_symbol: dict[str, pd.Series], *, ma_window: int = MA_WINDOW) -> pd.Series:
    """% of names trading above their own ``ma_window``-day MA, per date.

    Only names with a defined MA *and* a price that day count toward the denominator,
    so warm-up and membership gaps don't distort the percentage. NaN when none qualify.
    """
    df = pd.DataFrame(prices_by_symbol).astype(float)
    ma = df.rolling(ma_window).mean()
    valid = df.notna() & ma.notna()
    above = (df > ma) & valid
    denom = valid.sum(axis=1)
    pct = 100.0 * above.sum(axis=1) / denom.replace(0, np.nan)
    return pct


def breadth_label(
    pct: pd.Series,
    *,
    expansion: float = BREADTH_EXPANSION,
    contraction: float = BREADTH_CONTRACTION,
) -> pd.Series:
    """expansion / neutral / contraction from the breadth percentage."""
    out = pd.Series(UNKNOWN, index=pct.index, dtype=object)
    p = pct.astype(float)
    out[p.notna()] = "neutral"
    out[p > expansion] = "expansion"
    out[p < contraction] = "contraction"
    return out


# --- frame assembly ----------------------------------------------------------

def regime_frame(
    trend_bars: pd.DataFrame,
    vix: pd.Series,
    breadth: pd.Series | None = None,
    *,
    ma_window: int = MA_WINDOW,
    adx_window: int = ADX_WINDOW,
    adx_bull: float = ADX_BULL,
) -> pd.DataFrame:
    """Pure core: build the date × (trend, volatility, breadth) label frame.

    Args:
        trend_bars: OHLCV frame for the trend index (needs ``high/low/close``).
        vix: VIX close series.
        breadth: optional pre-computed breadth percentage series (see `breadth_pct`).
            When None the breadth column is all ``UNKNOWN`` (caller supplied nothing).
    Returns:
        DataFrame indexed by the union calendar with object columns
        ``trend``, ``volatility``, ``breadth`` (``UNKNOWN`` during warm-up / when absent).
    """
    trend = trend_label(
        trend_bars["close"], trend_bars["high"], trend_bars["low"],
        ma_window=ma_window, adx_window=adx_window, adx_bull=adx_bull,
    )
    # The trend benchmark defines the market calendar; VIX and breadth are reindexed
    # onto it (never unioned in), so a source with stray/extra dates — e.g. VIX bars
    # past the last equity session — cannot inject phantom or all-unknown rows.
    idx = trend.index
    volatility = vol_label(vix).reindex(idx)
    if breadth is not None:
        b_label = breadth_label(breadth).reindex(idx)
    else:
        b_label = pd.Series(UNKNOWN, index=idx, dtype=object)

    return pd.DataFrame(
        {
            "trend": trend.fillna(UNKNOWN),
            "volatility": volatility.fillna(UNKNOWN),
            "breadth": b_label.fillna(UNKNOWN),
        }
    )


def build_regime_frame(
    start: str | None = None,
    end: str | None = None,
    *,
    trend_symbol: str = "SPY",
    vix_symbol: str = "^VIX",
    universe: str = "SP500",
    breadth: pd.Series | None = None,
    ma_window: int = MA_WINDOW,
    use_cache: bool = True,
    cache=None,
) -> pd.DataFrame:
    """Network orchestrator: load SPY/VIX (+ S&P 500 breadth) and build the frame.

    Reuses the existing loaders (`data.loader`, `data.universe`); all network access is
    here so the pure `regime_frame` stays offline-testable. When ``breadth`` is supplied
    the constituent pull is skipped; otherwise the full S&P 500 (`universe`) is loaded
    and its %-above-200MA computed (one cached ~500-ticker pull).
    """
    from meridian.data.loader import load_ohlcv, load_universe
    from meridian.data.universe import get_universe

    trend_bars = load_ohlcv(trend_symbol, start, end, use_cache=use_cache, cache=cache)
    vix = load_ohlcv(vix_symbol, start, end, use_cache=use_cache, cache=cache)["close"]

    if breadth is None:
        symbols = get_universe(universe, use_cache=use_cache).symbols
        bars = load_universe(symbols, start, end, use_cache=use_cache, cache=cache)
        prices = {s: df["adj_close"] for s, df in bars.items()}
        breadth = breadth_pct(prices, ma_window=ma_window)

    return regime_frame(trend_bars, vix, breadth, ma_window=ma_window)


def attach_regimes(index: pd.Index | pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Left-join the three regime columns onto any backtest/signal date index.

    Each bar carries its day's ``(trend, volatility, breadth)`` labels for downstream
    permission gating. Dates with no regime row are ``UNKNOWN``.
    """
    idx = index.index if isinstance(index, pd.DataFrame) else index
    return frame.reindex(idx).fillna(UNKNOWN)
