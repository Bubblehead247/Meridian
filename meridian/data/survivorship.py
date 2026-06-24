"""Survivorship-bias-free S&P 500 dataset loader.

yfinance and FinanceDatabase are both survivor snapshots: delisted/removed names
are simply gone, which biases any historical backtest optimistically. This
loader reads a *point-in-time* dataset instead — Teddy Koker's
`survivorship-free-spy` (https://github.com/teddykoker/survivorship-free-spy) —
which ships:

- ``constituents.csv`` — monthly S&P 500 membership snapshots (each row:
  ``date,"[ticker, ...]"``), so you know which names were in the index on any
  past date, *including ones since removed*.
- ``data/<TICKER>.csv`` — per-ticker OHLCV (from Quandl WIKI Prices), covering
  delisted/removed names too. Coverage is roughly 2013–Feb 2018.

The dataset is not vendored into this repo (it is ~19 MB and lives under
``external/``); point ``SurvivorshipDataset`` at wherever you cloned it.

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
