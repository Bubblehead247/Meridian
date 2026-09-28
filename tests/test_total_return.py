"""Total-return tracker: logs the income paper doesn't pay, once per day."""

from __future__ import annotations

import pandas as pd
import pytest

from meridian.execution import total_return as tr


def _px():
    idx = pd.bdate_range("2026-10-01", periods=4)
    close = pd.DataFrame({"SGOV": [100.0, 100.0, 99.6, 99.6], "SPY": [500.0] * 4}, index=idx)
    # SGOV pays 0.4 on day 3: the price drops by it; the adjusted series is
    # scaled back before the ex-date, so it doesn't.
    adj = pd.DataFrame({"SGOV": [99.6, 99.6, 99.6, 99.6], "SPY": [500.0] * 4}, index=idx)
    return close, adj


def test_book_shares_sum_across_sleeves():
    assert tr.book_shares({"a": {"SGOV": 10.0}, "b": {"SGOV": 5.0, "X": 0.0}}) == {"SGOV": 15.0}


def test_a_dividend_is_logged_as_income_once(tmp_path):
    close, adj = _px()
    log = tmp_path / "tr.jsonl"
    # Pretend the log already covers day 1.
    log.write_text('{"date": "2026-10-01", "income": 0, "cumulative_income": 0, '
                   '"account_equity": 1000, "tr_equity": 1000}\n')
    new = tr.update(1000.0, {"SGOV": 100.0}, close, adj, log=log)
    assert [r["date"] for r in new] == ["2026-10-02", "2026-10-05", "2026-10-06"]
    # 100 shares x $100 x (0% adjusted - (-0.4%) price) = $40 on the ex-date.
    assert new[1]["income"] == pytest.approx(40.0, rel=1e-3)
    assert new[-1]["tr_equity"] == pytest.approx(1040.0, rel=1e-3)
    assert tr.update(1000.0, {"SGOV": 100.0}, close, adj, log=log) == []
