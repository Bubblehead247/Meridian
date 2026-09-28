"""Core-book models: trend 6 (TREND6-EW-U8).

The pre-registered full sweep (research/plans/full_sweep_2026_10.json, results in
research/2026-10-sweep/) recommended this as Meridian's core: eight asset-class
ETFs at 12.5% each, each held while its month-end price is above the mean of its
last 6 month-end prices (current one included), otherwise that 12.5% sits in
T-bills (the idle-cash sweep holds SGOV). 2008-2026/06 backtest: 7.0%/yr, max
drawdown -9.6%, with signals on dividend-adjusted prices.

Live sizing splits a sleeve equally across its longs, so each ETF is its own
sleeve (``core_trend_<etf>``) with one symbol.

Month-end handling: a decision is made only at a month's last session and held
all month. For the final bar, "is this the month's last session?" is answered
from quantcore's trading calendar when it covers the day; otherwise the next
business day starting a new month is used, which decides one session late when a
month ends on a holiday (e.g. Memorial Day on May 31).
"""

from __future__ import annotations

from datetime import date
from typing import Callable

import pandas as pd

from meridian.families.base import Model, register_model

TREND6_ETFS = ("SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "IEF", "TLT")


def is_last_session_of_month(day: date) -> bool:
    """True if ``day`` is the final trading session of its calendar month."""
    try:
        from quantcore import market_calendar as mc

        sessions = mc.load_cache()
        later = sorted(s.day for s in sessions if s.day > day)
        if mc.covers(day, sessions) and later:
            return later[0].month != day.month
    except Exception:  # noqa: BLE001 - calendar unavailable: fall back below
        pass
    return (pd.Timestamp(day) + pd.offsets.BDay(1)).month != day.month


#: Swappable in tests so results don't depend on the calendar cache on disk.
LAST_SESSION_CHECK: Callable[[date], bool] = is_last_session_of_month


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last bar of each month in ``index``; the final month counts only once it is over."""
    s = pd.Series(index, index=index)
    ends = pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)
    if len(ends) and ends[-1] == index[-1] and not LAST_SESSION_CHECK(index[-1].date()):
        ends = ends[:-1]
    return ends


class MonthlySmaTrend(Model):
    """Long while the month-end price is above its N-month average; decided at month-ends only."""

    sma_months: int = 6
    #: The live runner feeds this model dividend-adjusted closes for its signal (sizing
    #: still uses the real close). Plain closes flipped 2.8% of decisions, mostly in the
    #: bond ETFs, and cost 0.25%/yr (research/plans/trend6_signal_input_check.out).
    signal_price_field: str = "adj_close"

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        p = prices.dropna()
        if p.empty:
            return pd.Series(0, index=prices.index)
        ends = month_end_dates(p.index)
        month_px = p.loc[ends]
        decision = (month_px > month_px.rolling(self.sma_months).mean()).astype(int)
        held = decision.reindex(p.index).ffill().fillna(0).astype(int)
        return held.reindex(prices.index).fillna(0).astype(int)


for _sym in TREND6_ETFS:
    register_model(f"core_trend_{_sym.lower()}", "sma6_monthly")(
        type(f"CoreTrend{_sym}", (MonthlySmaTrend,),
             {"__doc__": f"Trend 6 core sleeve for {_sym}.", "__module__": __name__}))
