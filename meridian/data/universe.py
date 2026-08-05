"""Universe management.

A "universe" is the named set of symbols an experiment trades over. Meridian's
universes in scope: the broad ETFs (SPY, QQQ, IWM) and three index constituent
sets (S&P 500, Nasdaq-100, Russell 1000).

The ETF universes are single, fixed symbols — defined statically here. The
index constituent sets change over time; they are fetched from Wikipedia on
demand and cached on disk, because yfinance does not provide constituents.

Survivorship-bias note: constituent lists fetched today reflect *today's* index
membership, not the membership at any past date. Combined with yfinance's
omission of delisted tickers, backtests over these universes are
survivorship-biased. This must be documented in every report.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SurvivorshipBiasError(ValueError):
    """Raised when an index universe is resolved without acknowledging the bias.

    Index universes (SP500/NASDAQ100/RUSSELL1000) are resolved from *today's*
    constituent list, not the historical membership at any past date — using one
    for a backtest is both survivorship-biased and look-ahead-biased. This is a
    ``ValueError`` subclass so existing ``except (KeyError, ValueError)`` callers
    (e.g. the CLI's top-level error handler) catch it without changes.
    """


@dataclass(frozen=True)
class Universe:
    """A named set of tradable symbols."""

    name: str
    symbols: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.symbols)


# --- Static ETF universes -------------------------------------------------

_ETF_UNIVERSES: dict[str, Universe] = {
    "SPY": Universe("SPY", ("SPY",)),
    "QQQ": Universe("QQQ", ("QQQ",)),
    "IWM": Universe("IWM", ("IWM",)),
}

# --- Index constituent sources -------------------------------------------
# Each entry: (Wikipedia URL, column holding the ticker symbol).
_INDEX_SOURCES: dict[str, tuple[str, str]] = {
    "SP500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
    "NASDAQ100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
    "RUSSELL1000": ("https://en.wikipedia.org/wiki/Russell_1000_Index", "Symbol"),
}

#: Names of every universe Meridian knows about.
KNOWN_UNIVERSES: tuple[str, ...] = tuple(_ETF_UNIVERSES) + tuple(_INDEX_SOURCES)

#: Universes backed by a Wikipedia constituent list (as opposed to a fixed ETF ticker).
#: ``get_universe`` always returns *today's* membership for these — there is no date
#: parameter — so using one of these names for a historical backtest is both a
#: survivorship-bias and a look-ahead leak (see the module docstring). Callers that run
#: research/backtests against a named universe should gate on this and require an
#: explicit acknowledgement rather than silently resolving today's list.
INDEX_UNIVERSES: tuple[str, ...] = tuple(_INDEX_SOURCES)


def _normalize_symbol(sym: str) -> str:
    """Make a raw ticker yfinance-friendly (e.g. ``BRK.B`` -> ``BRK-B``)."""
    return sym.strip().upper().replace(".", "-")


# Wikipedia (and many sites) reject the default ``Python-urllib`` agent with HTTP 403;
# send a browser-like User-Agent so the constituent fetch succeeds.
_USER_AGENT = "Mozilla/5.0 (compatible; Meridian/1.0; +https://example.com/meridian)"


def _fetch_constituents(url: str, symbol_col: str) -> tuple[str, ...]:
    """Read an index's constituent tickers from a Wikipedia table.

    Scans every table on the page for one containing ``symbol_col`` and returns
    its symbols. Kept thin so tests can monkeypatch ``pandas.read_html``.
    """
    tables = pd.read_html(url, storage_options={"User-Agent": _USER_AGENT})
    for table in tables:
        if symbol_col in table.columns:
            symbols = table[symbol_col].astype(str).map(_normalize_symbol)
            return tuple(dict.fromkeys(symbols))  # de-dup, preserve order
    raise ValueError(f"No table with column {symbol_col!r} found at {url}")


def get_universe(
    name: str, use_cache: bool = True, cache=None, *, accept_survivorship_bias: bool = False
) -> Universe:
    """Resolve a universe by name.

    Args:
        name: One of ``KNOWN_UNIVERSES`` (case-insensitive).
        use_cache: For index universes, read/write the constituent list to the
            on-disk cache instead of refetching. Ignored for ETF universes.
        cache: Optional ``UniverseCache`` for constituent persistence. If None,
            a default-located cache is used.
        accept_survivorship_bias: Required ``True`` to resolve an index universe
            (see ``INDEX_UNIVERSES``) — it always reflects *today's* membership,
            never the historical membership at any past date. This is the API-level
            enforcement of that gate; callers must pass it explicitly rather than
            relying on a UI layer (e.g. a CLI flag) to have already checked it,
            since ``get_universe`` is reachable directly and from config-driven
            entry points that don't go through the CLI at all.

    Returns:
        The resolved Universe.

    Raises:
        KeyError: If ``name`` is not a known universe.
        SurvivorshipBiasError: If ``name`` is an index universe and
            ``accept_survivorship_bias`` is not True.
    """
    key = name.strip().upper()

    if key in _ETF_UNIVERSES:
        return _ETF_UNIVERSES[key]

    if key not in _INDEX_SOURCES:
        raise KeyError(f"Unknown universe {name!r}. Known: {', '.join(KNOWN_UNIVERSES)}")

    if not accept_survivorship_bias:
        raise SurvivorshipBiasError(
            f"{key!r} resolves to TODAY's index constituents, not the historical "
            "membership at any backtest date — this is a survivorship-bias AND "
            "look-ahead leak, not just a survivorship gap. Pass "
            "accept_survivorship_bias=True to proceed anyway, or use "
            "data.source=survivorship (point-in-time) for a bias-controlled run."
        )

    if cache is None:
        from meridian.data.cache import UniverseCache

        cache = UniverseCache()

    if use_cache:
        cached = cache.read(key)
        if cached is not None:
            return Universe(key, cached)

    url, col = _INDEX_SOURCES[key]
    symbols = _fetch_constituents(url, col)
    cache.write(key, symbols)
    return Universe(key, symbols)
