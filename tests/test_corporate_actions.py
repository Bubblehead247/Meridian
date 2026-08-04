"""Tests for data/corporate_actions.py."""

from __future__ import annotations

import pandas as pd
import pytest

from meridian.data.corporate_actions import detect_adjustment_anomalies


def _frame(close: list[float], adj_close: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    return pd.DataFrame({"close": close, "adj_close": adj_close}, index=idx)


def test_no_anomalies_on_a_flat_ratio():
    close = [100.0, 101.0, 99.0, 102.0, 100.5]
    adj_close = [100.0, 101.0, 99.0, 102.0, 100.5]  # ratio == 1.0 throughout
    out = detect_adjustment_anomalies(_frame(close, adj_close))
    assert out.empty


def test_normal_dividend_sized_change_is_not_flagged():
    # ~1% ratio change day-over-day -- well under the default 15% threshold.
    close = [100.0, 100.0, 100.0]
    adj_close = [100.0, 100.0, 99.0]  # ratio: 1.0, 1.0, 1.0101 (~1% jump)
    out = detect_adjustment_anomalies(_frame(close, adj_close))
    assert out.empty


def test_clean_two_for_one_split_is_tagged_likely_split():
    # Ratio doubles on day 2 -- a clean 2:1 split factor.
    close = [100.0, 100.0, 100.0]
    adj_close = [100.0, 50.0, 50.0]  # ratio: 1.0, 2.0, 2.0
    out = detect_adjustment_anomalies(_frame(close, adj_close))
    assert len(out) == 1
    assert out.iloc[0]["tag"] == "likely_split"
    assert out.iloc[0]["factor_change"] == pytest.approx(2.0)


def test_unclean_large_jump_is_tagged_unclean_adjustment():
    # A jump too large for a dividend but not close to any clean split ratio.
    close = [100.0, 100.0, 100.0]
    adj_close = [100.0, 73.0, 73.0]  # ratio: 1.0, 1.370, unchanged after
    out = detect_adjustment_anomalies(_frame(close, adj_close))
    assert len(out) == 1
    assert out.iloc[0]["tag"] == "unclean_adjustment"


def test_reverse_split_factor_is_tagged_likely_split():
    # 1-for-3 reverse split -> ratio shrinks to 1/3.
    close = [90.0, 90.0]
    adj_close = [30.0, 90.0]  # ratio: 3.0 -> 1.0, factor_change = 1/3
    out = detect_adjustment_anomalies(_frame(close, adj_close))
    assert len(out) == 1
    assert out.iloc[0]["tag"] == "likely_split"
    assert out.iloc[0]["factor_change"] == pytest.approx(1 / 3, rel=1e-2)


def test_missing_columns_raises():
    with pytest.raises(ValueError, match="close"):
        detect_adjustment_anomalies(pd.DataFrame({"open": [1.0], "high": [1.0]}))


def test_never_modifies_the_input_frame():
    frame = _frame([100.0, 100.0], [100.0, 50.0])
    before = frame.copy()
    detect_adjustment_anomalies(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_custom_thresholds_change_sensitivity():
    close = [100.0, 100.0]
    adj_close = [100.0, 92.0]  # ~8.7% ratio jump
    assert detect_adjustment_anomalies(_frame(close, adj_close), jump_threshold=0.15).empty
    out = detect_adjustment_anomalies(_frame(close, adj_close), jump_threshold=0.05)
    assert len(out) == 1
