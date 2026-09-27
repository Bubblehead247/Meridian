"""Tests for the monthly resize throttle (execution/rebalance_schedule.py)."""

from __future__ import annotations

from datetime import date

from meridian.execution.rebalance_schedule import (
    is_rebalance_due,
    load_last_rebalance,
    record_rebalance,
)


def test_never_recorded_is_due(tmp_path):
    path = tmp_path / "last_rebalance.json"
    assert is_rebalance_due("momentum", "AAPL", date(2026, 9, 15), path=path)


def test_recorded_this_month_is_not_due_again(tmp_path):
    path = tmp_path / "last_rebalance.json"
    record_rebalance("momentum", "AAPL", date(2026, 9, 1), path=path)
    assert not is_rebalance_due("momentum", "AAPL", date(2026, 9, 28), path=path)


def test_recorded_last_month_is_due_again(tmp_path):
    path = tmp_path / "last_rebalance.json"
    record_rebalance("momentum", "AAPL", date(2026, 8, 30), path=path)
    assert is_rebalance_due("momentum", "AAPL", date(2026, 9, 1), path=path)


def test_recorded_last_year_same_month_number_is_still_due():
    # year must match too, not just month-of-year, or Jan 2026 would wrongly
    # look "not due" again in Jan 2027.
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "last_rebalance.json"
        record_rebalance("momentum", "AAPL", date(2025, 9, 15), path=path)
        assert is_rebalance_due("momentum", "AAPL", date(2026, 9, 15), path=path)


def test_tracking_is_per_family_and_symbol(tmp_path):
    path = tmp_path / "last_rebalance.json"
    record_rebalance("momentum", "AAPL", date(2026, 9, 1), path=path)

    # Different symbol, same family: independently due.
    assert is_rebalance_due("momentum", "MSFT", date(2026, 9, 15), path=path)
    # Same symbol, different family (shared-symbol sleeves): independently due.
    assert is_rebalance_due("sector_rotation", "AAPL", date(2026, 9, 15), path=path)
    # The original pairing is still not due.
    assert not is_rebalance_due("momentum", "AAPL", date(2026, 9, 15), path=path)


def test_record_rebalance_persists_across_loads(tmp_path):
    path = tmp_path / "last_rebalance.json"
    record_rebalance("trend_following", "XLK", date(2026, 9, 1), path=path)
    data = load_last_rebalance(path)
    assert data["trend_following/XLK"] == "2026-09-01"


def test_corrupt_file_is_treated_as_never_recorded(tmp_path):
    path = tmp_path / "last_rebalance.json"
    path.write_text("not valid json", encoding="utf-8")
    assert load_last_rebalance(path) == {}
    assert is_rebalance_due("momentum", "AAPL", date(2026, 9, 15), path=path)
