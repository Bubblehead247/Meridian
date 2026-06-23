"""Phase 1 tests for FinanceDatabase-backed screening. Offline via a fake.

A tiny fake stands in for ``financedatabase.Equities`` so these tests never
download the real database. The fake mimics ``select(...)`` returning a
symbol-indexed metadata frame and honoring the filters the screener uses.
"""

from __future__ import annotations

import pandas as pd
import pytest

from meridian.data import EquityScreener
from meridian.data.screener import _to_yf_symbol


class _FakeEquities:
    """Minimal stand-in for financedatabase.Equities."""

    def __init__(self):
        # Includes a slash share-class (BRK/B), a dot-suffix foreign cross
        # listing (RY.TO), a primary US name, and a delisted name.
        self._df = pd.DataFrame(
            {
                "name": ["Apple", "Berkshire B", "Royal Bank", "Defunct Co"],
                "country": ["United States"] * 4,
                "sector": ["Information Technology", "Financials", "Financials", "Energy"],
                "market_cap": ["Mega Cap", "Mega Cap", "Large Cap", "Micro Cap"],
                "primary": [True, True, False, True],
                "delisted": [False, False, False, True],
            },
            index=pd.Index(["AAPL", "BRK/B", "RY.TO", "DEFNCT"], name="symbol"),
        )

    def select(self, only_primary_listing=False, exclude_delisted=True, **filters):
        df = self._df
        if exclude_delisted:
            df = df[~df["delisted"]]
        if only_primary_listing:
            df = df[df["primary"]]
        for key, value in filters.items():
            if value is None:
                continue
            wanted = {value} if isinstance(value, str) else set(value)
            df = df[df[key].isin(wanted)]
        return df.drop(columns=["primary"])


@pytest.fixture
def screener():
    return EquityScreener(equities=_FakeEquities())


def test_to_yf_symbol_converts_slash_not_dot():
    assert _to_yf_symbol("BRK/B") == "BRK-B"      # share class slash -> dash
    assert _to_yf_symbol("002990.SZ") == "002990.SZ"  # exchange-suffix dot kept
    assert _to_yf_symbol(" aapl ") == "AAPL"


def test_screen_passes_filters_through(screener):
    df = screener.screen(sector="Financials")
    assert set(df.index) == {"BRK/B", "RY.TO"}  # delisted excluded by default


def test_build_universe_primary_only_and_yf_safe(screener):
    u = screener.build_universe("us-fin", sector="Financials")
    # only_primary_listing defaults True -> RY.TO (non-primary) dropped;
    # BRK/B normalized to BRK-B.
    assert u.name == "us-fin"
    assert u.symbols == ("BRK-B",)


def test_build_universe_can_keep_all_listings(screener):
    u = screener.build_universe(
        "us-fin-all", sector="Financials", only_primary_listing=False
    )
    assert u.symbols == ("BRK-B", "RY.TO")


def test_build_universe_excludes_delisted_by_default(screener):
    u = screener.build_universe("us-energy", sector="Energy")
    assert u.symbols == ()  # the only Energy name (DEFNCT) is delisted


def test_build_universe_can_include_delisted(screener):
    u = screener.build_universe(
        "us-energy", sector="Energy", exclude_delisted=False
    )
    assert u.symbols == ("DEFNCT",)


def test_metadata_aligns_to_requested_symbols(screener):
    md = screener.metadata(["AAPL", "NOPE"])
    assert list(md.index) == ["AAPL", "NOPE"]
    assert md.loc["AAPL", "sector"] == "Information Technology"
    assert pd.isna(md.loc["NOPE", "sector"])  # unknown symbol -> NaN row
