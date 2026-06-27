"""Tests for the index-level regime labeler (trend / volatility / breadth)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.regimes import labeler as lab
from meridian.regimes.labeler import (
    RegimeLabel,
    adx,
    attach_regimes,
    breadth_label,
    breadth_pct,
    build_regime_frame,
    regime_frame,
    trend_label,
    vol_label,
)


def _ohlc(close: np.ndarray, band: float = 0.5) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=len(close), freq="B")
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"high": c + band, "low": c - band, "close": c})


# --- ADX ------------------------------------------------------------------

def test_adx_high_in_strong_trend_low_in_chop():
    n = 300
    trend = _ohlc(np.linspace(100, 200, n))
    chop = _ohlc(100 + (np.arange(n) % 2))            # sawtooth, no net direction
    adx_trend = adx(trend["high"], trend["low"], trend["close"]).iloc[-1]
    adx_chop = adx(chop["high"], chop["low"], chop["close"]).iloc[-1]
    assert adx_trend > 40          # a clean ramp is a very strong trend
    assert adx_chop < 25           # choppy = weak
    assert adx_trend > adx_chop


# --- trend label ----------------------------------------------------------

def test_trend_bull_uptrend():
    bars = _ohlc(np.linspace(100, 200, 300))
    assert trend_label(bars["close"], bars["high"], bars["low"]).iloc[-1] == "bull"


def test_trend_bear_below_ma():
    bars = _ohlc(np.linspace(200, 100, 300))         # falling: close below trailing MA
    assert trend_label(bars["close"], bars["high"], bars["low"]).iloc[-1] == "bear"


def test_trend_neutral_choppy_above_ma():
    # sawtooth around 100.5, ending at 101 (>= its ~100.5 MA) with weak ADX -> neutral
    bars = _ohlc(100 + (np.arange(300) % 2), band=0.2)
    assert trend_label(bars["close"], bars["high"], bars["low"]).iloc[-1] == "neutral"


def test_trend_unknown_during_warmup():
    bars = _ohlc(np.linspace(100, 110, 50))          # < 200 bars
    assert (trend_label(bars["close"], bars["high"], bars["low"]) == "unknown").all()


# --- volatility label -----------------------------------------------------

def test_vol_label_boundaries():
    vix = pd.Series([10.0, 15.0, 17.0, 20.0, 25.0, 30.0, 31.0, np.nan])
    got = vol_label(vix).tolist()
    assert got == ["low", "normal", "normal", "elevated", "elevated",
                   "elevated", "extreme", "unknown"]


# --- breadth --------------------------------------------------------------

def test_breadth_pct_counts_names_above_their_ma():
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    up = pd.Series(np.linspace(10, 20, n), index=idx)      # above its own MA
    down = pd.Series(np.linspace(20, 10, n), index=idx)    # below its own MA
    prices = {"A": up, "B": up * 1.0, "C": up * 1.0, "D": down}  # 3 of 4 up
    pct = breadth_pct(prices, ma_window=20)
    assert abs(pct.iloc[-1] - 75.0) < 1e-9


def test_breadth_label_bins():
    pct = pd.Series([70.0, 50.0, 30.0, np.nan])
    assert breadth_label(pct).tolist() == ["expansion", "neutral", "contraction", "unknown"]


# --- frame assembly -------------------------------------------------------

def test_regime_frame_columns_and_breadth_none_is_unknown():
    bars = _ohlc(np.linspace(100, 200, 300))
    vix = pd.Series(np.full(300, 12.0), index=bars.index)
    frame = regime_frame(bars, vix, breadth=None)
    assert list(frame.columns) == ["trend", "volatility", "breadth"]
    assert (frame["breadth"] == "unknown").all()         # none supplied
    assert frame["volatility"].iloc[-1] == "low"
    assert frame["trend"].iloc[-1] == "bull"


def test_regime_frame_uses_trend_calendar_not_union():
    """VIX bars past the last equity session must not add all-unknown trailing rows."""
    bars = _ohlc(np.linspace(100, 200, 300))                 # trend calendar
    vix_idx = bars.index.append(pd.date_range(bars.index[-1] + pd.Timedelta("1D"),
                                              periods=3, freq="B"))
    vix = pd.Series(12.0, index=vix_idx)                      # 3 extra trailing dates
    frame = regime_frame(bars, vix, breadth=None)
    assert frame.index.equals(bars.index)                    # exactly the trend sessions
    assert frame["trend"].iloc[-1] == "bull"                 # tail stays labeled


def test_attach_regimes_joins_and_fills_unknown():
    idx = pd.date_range("2021-01-01", periods=5, freq="B")
    frame = pd.DataFrame(
        {"trend": "bull", "volatility": "low", "breadth": "expansion"}, index=idx
    )
    target = pd.date_range("2021-01-06", periods=4, freq="B")   # last 2 dates overshoot
    out = attach_regimes(target, frame)
    assert out.loc[idx[3], "trend"] == "bull"            # 2021-01-06 is inside the frame
    assert (out.iloc[-1] == "unknown").all()             # 2021-01-11 is beyond the frame


# --- network orchestrator (offline via monkeypatch) -----------------------

def test_build_regime_frame_offline(monkeypatch):
    n = 300
    bars = _ohlc(np.linspace(100, 200, n))
    vix_df = bars.assign(close=np.full(n, 25.0))         # elevated VIX

    def fake_load_ohlcv(symbol, start=None, end=None, **kw):
        return vix_df if symbol == "^VIX" else bars

    monkeypatch.setattr("meridian.data.loader.load_ohlcv", fake_load_ohlcv)
    breadth = pd.Series(np.full(n, 70.0), index=bars.index)   # supplied -> skips constituents
    frame = build_regime_frame(breadth=breadth)
    assert frame["trend"].iloc[-1] == "bull"
    assert frame["volatility"].iloc[-1] == "elevated"
    assert frame["breadth"].iloc[-1] == "expansion"


def test_regime_label_dataclass_holds_three_dims():
    r = RegimeLabel(trend="bull", volatility="low", breadth="expansion")
    assert (r.trend, r.volatility, r.breadth) == ("bull", "low", "expansion")
    assert lab.TREND_LABELS == ("bull", "neutral", "bear")
