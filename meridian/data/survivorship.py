"""Survivorship-bias-free S&P 500 dataset loader.

yfinance and FinanceDatabase are both survivor snapshots: delisted/removed names
are simply gone, which biases any historical backtest optimistically. This
loader reads a *point-in-time* dataset instead — originally Teddy Koker's
`survivorship-free-spy` (https://github.com/teddykoker/survivorship-free-spy),
regenerated in-place (P1-E follow-up) to widen its price coverage — which ships:

- ``constituents.csv`` — monthly-ish S&P 500 membership snapshots (each row:
  ``date,"[ticker, ...]"``, 2006-09-29 to 2019-04-29), so you know which names
  were in the index on any past date, *including ones since removed*. This file
  is vendored as-is and has at least one known gap: AAPL and GE are incorrectly
  absent from several snapshots in 2010 (verified — both have been S&P 500
  constituents since well before 2010; AAPL since 1982) despite genuinely
  being members. This is a data-quality issue in the third-party source file
  itself, not corrected here — treat any single-name absence in the early
  (2010-2013) snapshots as unverified until cross-checked against another
  source, even though membership presence elsewhere is reliable.
- ``data/<TICKER>.csv`` — per-ticker OHLCV (from Nasdaq Data Link's WIKI/PRICES
  table, the Quandl dataset's current home), covering delisted/removed names
  too. Coverage is 2010-01-04 to 2018-03-27 — widened from the originally
  shipped 2013-01 to 2018-02 slice by regenerating against the full available
  WIKI/PRICES range (see ``external/survivorship-free-spy/survivorship-free/
  generate.py``, which now requires a free NASDAQ_DATA_LINK_API_KEY rather than
  a manually-placed bulk CSV and no longer depends on iShares' holdings page,
  which stopped publishing historical data around 2020).

The dataset is not vendored into this repo (it is regenerated locally under
``external/``); point ``SurvivorshipDataset`` at wherever you generated it.

The headline use is the bias measurement: run the *same* strategy on the
point-in-time universe (`survivorship_free_universe`) vs the end-of-period
survivors (`survivor_only_universe`) and compare — the difference is the
survivorship bias, quantified on identical data.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

#: Actual price-data coverage of the regenerated survivorship-free-spy dataset,
#: verified directly against external/survivorship-free-spy/survivorship-free/
#: data/*.csv. This is 82% of the project's DEFAULT_IN_SAMPLE window (2010-01-01 to
#: 2019-12-31, see data/splits.py) — up from 44% before the P1-E follow-up regen. The
#: remaining 2018-04 to 2019-12 gap is a hard limit of WIKI/PRICES itself (frozen
#: since Quandl's April 2018 acquisition by Nasdaq), not something more re-fetching
#: can close for free. A report that calls a run over this dataset "in-sample
#: validated" without saying so is still overstating what was actually tested. See
#: research_integrity_gap_analysis.md §3.3.
KNOWN_PRICE_COVERAGE: tuple[str, str] = ("2010-01-04", "2018-03-27")


def coverage_warning(start: str | None, end: str | None) -> str | None:
    """Warn when a requested [start, end] window extends beyond the dataset's actual
    coverage, so callers don't silently believe they validated a wider period than
    the data supports. Returns None when the request is fully within coverage.
    """
    cov_start, cov_end = pd.Timestamp(KNOWN_PRICE_COVERAGE[0]), pd.Timestamp(KNOWN_PRICE_COVERAGE[1])
    req_start = pd.Timestamp(start) if start else cov_start
    req_end = pd.Timestamp(end) if end else cov_end
    if req_start >= cov_start and req_end <= cov_end:
        return None
    return (
        f"requested window [{req_start.date()}, {req_end.date()}] extends beyond this "
        f"dataset's actual price coverage [{cov_start.date()}, {cov_end.date()}] — dates "
        "outside coverage contribute no data, so the effective tested window is narrower "
        "than requested."
    )


def _clean_ticker(t: str) -> str:
    """Strip the dataset's annotations (``AAPL*`` -> ``AAPL``)."""
    return t.strip().rstrip("*").strip()


class SurvivorshipDataset:
    """Reader for the survivorship-free-spy on-disk dataset.

    Args:
        root: Path to the ``survivorship-free`` directory containing
            ``constituents.csv`` and the ``data/`` folder.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        if not (self.root / "constituents.csv").exists():
            raise FileNotFoundError(f"constituents.csv not found under {self.root}")
        self._members: list[tuple[pd.Timestamp, set[str]]] | None = None
        self._available: set[str] | None = None

    # --- membership -------------------------------------------------------

    def _membership(self) -> list[tuple[pd.Timestamp, set[str]]]:
        """Sorted [(snapshot_date, member set)] from constituents.csv."""
        if self._members is None:
            df = pd.read_csv(self.root / "constituents.csv", header=None, names=["date", "members"])
            rows = []
            for _, r in df.iterrows():
                members = {_clean_ticker(t) for t in ast.literal_eval(r["members"])}
                rows.append((pd.Timestamp(r["date"]), members))
            self._members = sorted(rows, key=lambda x: x[0])
        return self._members

    def available_symbols(self) -> set[str]:
        """Tickers that have a price file in ``data/``."""
        if self._available is None:
            self._available = {p.stem for p in self.data_dir.glob("*.csv")}
        return self._available

    def members_asof(self, date) -> set[str]:
        """Index membership as of the latest snapshot on/before ``date``."""
        d = pd.Timestamp(date)
        chosen: set[str] = set()
        for snap_date, members in self._membership():
            if snap_date <= d:
                chosen = members
            else:
                break
        return chosen

    def members_ever(self, start=None, end=None) -> set[str]:
        """All tickers that were members in any snapshot within [start, end]."""
        lo = pd.Timestamp(start) if start else pd.Timestamp.min
        hi = pd.Timestamp(end) if end else pd.Timestamp.max
        out: set[str] = set()
        for snap_date, members in self._membership():
            if lo <= snap_date <= hi:
                out |= members
        return out

    # --- prices -----------------------------------------------------------

    def load_symbol(self, ticker: str) -> pd.DataFrame | None:
        """Load one ticker's OHLCV frame (date-indexed), or None if absent."""
        path = self.data_dir / f"{ticker}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df = df.assign(date=pd.to_datetime(df["date"], format="mixed", errors="coerce"))
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        df.index.name = "date"
        return df

    def _prices_for(self, tickers, start, end, field) -> dict[str, pd.Series]:
        lo = pd.Timestamp(start) if start else None
        hi = pd.Timestamp(end) if end else None
        out: dict[str, pd.Series] = {}
        for t in tickers:
            df = self.load_symbol(t)
            if df is None or field not in df.columns:
                continue
            s = df[field].astype(float)
            if lo is not None:
                s = s[s.index >= lo]
            if hi is not None:
                s = s[s.index <= hi]
            if len(s) > 1:
                out[t] = s
        return out

    # --- universes for the study -----------------------------------------

    def survivor_only_universe(
        self, start=None, end=None, *, field="close", asof=None
    ) -> dict[str, pd.Series]:
        """Prices for only the index members at ``asof`` (default: latest snapshot).

        This is the **survivorship-biased** view — "use the names that were in the
        index at the end, backtested over the whole window" — the very mistake the
        bias-free universe avoids.
        """
        asof = asof or self._membership()[-1][0]
        members = self.members_asof(asof) & self.available_symbols()
        return self._prices_for(members, start, end, field)

    def survivorship_free_universe(
        self, start=None, end=None, *, field="close", membership_gated=True
    ) -> dict[str, pd.Series]:
        """Point-in-time universe: every name that was *ever* a member in the
        window (including ones later removed), each priced only while it was
        actually in the index.

        When ``membership_gated`` is True (default), prices on dates a name was
        not an index member are set to NaN, so the portfolio study treats it as
        untradable then — true point-in-time membership.
        """
        members = self.members_ever(start, end) & self.available_symbols()
        prices = self._prices_for(members, start, end, field)
        if not membership_gated:
            return prices
        return {t: self._gate_to_membership(t, s) for t, s in prices.items()}

    def _gate_to_membership(self, ticker: str, series: pd.Series) -> pd.Series:
        """NaN out the dates ``ticker`` was not an index member."""
        snaps = self._membership()
        mask = np.zeros(len(series), dtype=bool)
        idx = series.index
        for i, (snap_date, members) in enumerate(snaps):
            nxt = snaps[i + 1][0] if i + 1 < len(snaps) else pd.Timestamp.max
            if ticker in members:
                mask |= (idx >= snap_date) & (idx < nxt)
        return series.where(mask)
