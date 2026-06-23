"""Metadata-driven universe construction via FinanceDatabase.

`FinanceDatabase <https://github.com/JerBouma/FinanceDatabase>`_ ships a static
database of 300k+ symbols with rich metadata — sector, industry, country,
exchange, market-cap bucket, and a ``delisted`` flag. yfinance gives prices but
no such metadata, so this module is how Meridian builds *custom* universes by
screening on fundamentals-adjacent attributes (e.g. "US, Health Care, Large
Cap"), complementing the fixed ETF and index universes in ``universe.py``.

Symbol formatting note: FinanceDatabase uses dots for exchange suffixes
(``002990.SZ``) and slashes for some share classes (``BRK/B``). yfinance wants
``BRK-B`` and keeps the dot suffix. So the yfinance-safe normalizer here
converts only ``/`` -> ``-`` and must NOT touch dots — the opposite of the
share-class normalizer used for Wikipedia index lists in ``universe.py``.

Survivorship-bias note: FinanceDatabase *does* carry delisted names (its
``delisted`` column), but ``select`` defaults to ``exclude_delisted=True``.
Excluding them reintroduces survivorship bias; including them does not fix
yfinance's own omission of delisted price history. Document the choice.
"""

from __future__ import annotations

import pandas as pd

from meridian.data.universe import Universe


def _to_yf_symbol(sym: str) -> str:
    """Make a FinanceDatabase ticker yfinance-friendly.

    Converts share-class slashes to dashes (``BRK/B`` -> ``BRK-B``) and leaves
    exchange-suffix dots intact (``002990.SZ`` stays as-is).
    """
    return str(sym).strip().upper().replace("/", "-")


class EquityScreener:
    """Build universes and look up metadata from FinanceDatabase equities.

    The underlying ``financedatabase.Equities`` object is created lazily on
    first use (it loads a large dataset), and may be injected for testing.
    """

    def __init__(self, equities=None):
        self._equities = equities

    @property
    def equities(self):
        """The lazily-instantiated ``financedatabase.Equities`` handle."""
        if self._equities is None:
            import financedatabase as fd

            self._equities = fd.Equities()
        return self._equities

    def screen(self, **filters) -> pd.DataFrame:
        """Return the metadata frame for equities matching ``filters``.

        Filters are passed straight to ``financedatabase.Equities.select`` —
        e.g. ``country``, ``sector``, ``industry``, ``exchange``, ``mic``,
        ``market``, ``market_cap``, ``only_primary_listing``,
        ``exclude_delisted``. The result is indexed by symbol.
        """
        return self.equities.select(**filters)

    def build_universe(
        self,
        name: str,
        *,
        only_primary_listing: bool = True,
        yfinance_safe: bool = True,
        **filters,
    ) -> Universe:
        """Screen FinanceDatabase and return a :class:`Universe`.

        Defaults to ``only_primary_listing=True`` so the result is the clean,
        single primary listing per company (no foreign cross-listings), which
        is what yfinance expects.

        Args:
            name: Name to give the resulting universe.
            only_primary_listing: Keep only each name's primary listing.
            yfinance_safe: Convert symbols to yfinance form and de-duplicate.
            **filters: Any ``financedatabase`` select filter.

        Returns:
            A Universe of the screened symbols, order-preserving and de-duped.
        """
        df = self.screen(only_primary_listing=only_primary_listing, **filters)
        raw = list(df.index)
        symbols = [_to_yf_symbol(s) for s in raw] if yfinance_safe else [str(s) for s in raw]
        deduped = tuple(dict.fromkeys(symbols))  # preserve order, drop dups
        return Universe(name, deduped)

    def metadata(self, symbols: tuple[str, ...] | list[str]) -> pd.DataFrame:
        """Return the metadata rows for the given symbols.

        Symbols absent from the database appear as all-NaN rows so the result
        always aligns 1:1 with ``symbols``. Useful downstream for grouping by
        sector/country (e.g. regime or portfolio construction).
        """
        full = self.equities.select()
        return full.reindex([str(s) for s in symbols])
