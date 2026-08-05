"""Tests for the survivorship-bias-free dataset loader (offline fixture)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.data import SurvivorshipDataset
from meridian.data.survivorship import coverage_warning


@pytest.fixture
def dataset(tmp_path):
    """A tiny survivorship-free-spy-format dataset.

    Membership: January = {A, B, C}; February = {A, B} (C is removed). So C is a
    point-in-time member that a survivor-only view would wrongly drop.
    """
    root = tmp_path / "sf"
    (root / "data").mkdir(parents=True)

    # constituents.csv: descending dates, python-list-literal members (like the real file)
    (root / "constituents.csv").write_text(
        "2018-02-28,\"['A', 'B']\"\n"
        "2018-01-31,\"['A', 'B', 'C', 'NODATA']\"\n",
        encoding="utf-8",
    )

    dates = pd.date_range("2018-01-02", "2018-02-28", freq="B")
    for i, t in enumerate(["A", "B", "C"]):
        px = 100 + i * 10 + np.arange(len(dates)) * 0.1
        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": px, "high": px + 1, "low": px - 1, "close": px,
            "volume": 1_000_000,
        })
        # C uses US-style dates to exercise mixed-format parsing
        if t == "C":
            df["date"] = dates.strftime("%m/%d/%Y")
        df.to_csv(root / "data" / f"{t}.csv", index=False)

    return SurvivorshipDataset(root)


def test_available_symbols(dataset):
    assert dataset.available_symbols() == {"A", "B", "C"}  # NODATA has no file


def test_members_asof(dataset):
    assert dataset.members_asof("2018-01-15") == set()                  # before first snapshot
    assert dataset.members_asof("2018-02-10") == {"A", "B", "C", "NODATA"}
    assert dataset.members_asof("2018-03-01") == {"A", "B"}             # C removed


def test_members_ever(dataset):
    assert dataset.members_ever("2018-01-01", "2018-03-01") == {"A", "B", "C", "NODATA"}


def test_load_symbol_parses_mixed_date_formats(dataset):
    df = dataset.load_symbol("C")  # C.csv uses M/D/Y
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert "close" in df.columns


def test_survivor_only_drops_removed_names(dataset):
    # asof latest snapshot (Feb) -> only the survivors A, B; C is missing (the bias)
    surv = dataset.survivor_only_universe("2018-01-01", "2018-03-01")
    assert set(surv) == {"A", "B"}


# --- coverage_warning (P1-E) -------------------------------------------------

def test_coverage_warning_none_when_fully_within_coverage():
    assert coverage_warning("2014-01-01", "2018-01-01") is None


def test_coverage_warning_flags_a_window_starting_before_coverage():
    w = coverage_warning("2005-01-01", "2015-01-01")
    assert w is not None and "2010-01-04" in w


def test_coverage_warning_flags_a_window_ending_after_coverage():
    w = coverage_warning("2015-01-01", "2019-12-31")
    assert w is not None and "2018-03-27" in w


def test_coverage_warning_none_when_start_and_end_omitted():
    # Omitted bounds default to the dataset's own coverage — nothing to warn about.
    assert coverage_warning(None, None) is None


def test_survivorship_free_includes_removed_names(dataset):
    free = dataset.survivorship_free_universe("2018-01-01", "2018-03-01")
    assert set(free) == {"A", "B", "C"}  # C included


def test_membership_gating_masks_non_member_dates(dataset):
    free = dataset.survivorship_free_universe("2018-01-01", "2018-03-01", membership_gated=True)
    c = free["C"]
    # C is a member only in [2018-01-31, 2018-02-28): tradable in early Feb, NaN at month end
    assert np.isfinite(c.loc["2018-02-01"])
    assert np.isnan(c.loc["2018-02-28"])      # removed at the Feb snapshot
    # A remains a member throughout, so it is tradable on 2018-02-28
    assert np.isfinite(free["A"].loc["2018-02-28"])


def test_ungated_keeps_all_prices(dataset):
    free = dataset.survivorship_free_universe("2018-01-01", "2018-03-01", membership_gated=False)
    assert free["C"].notna().all()  # no masking when gating disabled


def test_missing_constituents_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SurvivorshipDataset(tmp_path)
