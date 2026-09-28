"""Trend 6 core model: decides only at month-ends, holds all month, and treats the
final bar as a month-end only when it is the month's last session."""

from __future__ import annotations

import pandas as pd
import pytest

from meridian.families import create_model
from meridian.families.core import models as core


@pytest.fixture(autouse=True)
def _fixed_calendar(monkeypatch):
    # Business-day rule only, so results don't depend on the calendar cache on disk.
    monkeypatch.setattr(core, "LAST_SESSION_CHECK",
                        lambda d: (pd.Timestamp(d) + pd.offsets.BDay(1)).month != d.month)


def _prices(values_by_month_end, start="2025-01-01"):
    """Daily business-day series whose month-end closes are the given values (flat within months)."""
    idx = pd.bdate_range(start, periods=len(values_by_month_end) * 23)
    s = pd.Series(index=idx, dtype=float)
    months = sorted({(d.year, d.month) for d in idx})
    for (y, m), v in zip(months, values_by_month_end):
        s[(idx.year == y) & (idx.month == m)] = v
    return s.dropna()


def test_long_only_when_the_month_end_is_above_its_six_month_average():
    model = create_model("core_trend_spy", "sma6_monthly")
    # Six flat months at 100, then a month-end at 110: above the average -> long.
    p = _prices([100, 100, 100, 100, 100, 100, 110, 110])
    sig = model.signals(p)
    seventh_end = p.index[p.index.month == p.index[0].month + 6][-1]
    assert sig.loc[seventh_end] == 1
    assert sig.loc[: p.index[p.index.month == p.index[0].month + 5][-1]].eq(0).all()


def test_the_decision_is_held_all_month_even_if_prices_fall_mid_month():
    model = create_model("core_trend_spy", "sma6_monthly")
    p = _prices([100, 100, 100, 100, 100, 100, 110, 110, 110])
    # Crash the middle of the 8th month (not its month-end): the signal must not change mid-month.
    eighth = p.index[p.index.month == p.index[0].month + 7]
    p.loc[eighth[5:10]] = 50.0
    sig = model.signals(p)
    assert sig.loc[eighth[:-1]].eq(1).all()


def test_an_unfinished_final_month_does_not_trigger_a_decision():
    model = create_model("core_trend_spy", "sma6_monthly")
    p = _prices([100, 100, 100, 100, 100, 100, 110])
    # Cut the series mid-month: the final bar is not the month's last session.
    cut = p.index[p.index.month == p.index[0].month + 6][10]
    q = p.loc[:cut].copy()
    q.iloc[-1] = 200.0          # a huge move on an unfinished month changes nothing
    assert model.signals(q).iloc[-1] == 0


def test_month_end_dates_skips_the_final_bar_unless_it_is_the_last_session():
    idx = pd.bdate_range("2026-09-01", "2026-09-25")
    assert len(core.month_end_dates(idx)) == 0
    idx2 = pd.bdate_range("2026-09-01", "2026-09-30")
    assert list(core.month_end_dates(idx2)) == [pd.Timestamp("2026-09-30")]


def test_the_model_asks_for_dividend_adjusted_signal_prices():
    assert create_model("core_trend_tlt", "sma6_monthly").signal_price_field == "adj_close"
