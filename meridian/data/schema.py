"""Standardized OHLCV schema.

Every data source (yfinance now, alpaca later) is normalized into one canonical
shape so the rest of the platform never sees source-specific column names. A
"bar" is one period's price summary: open, high, low, close, plus an
adjusted close (corrected for splits/dividends) and traded volume.
"""

from __future__ import annotations

import pandas as pd

#: Canonical column order for an OHLCV frame.
OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "adj_close", "volume"]

#: Name given to the DatetimeIndex of every OHLCV frame.
INDEX_NAME = "date"

# Maps the various column spellings sources use onto the canonical names.
_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Adj_Close": "adj_close",
    "Adjclose": "adj_close",
    "Volume": "volume",
}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw OHLCV frame into the canonical schema.

    - Renames source columns to canonical lowercase names.
    - Ensures all six canonical columns exist (missing ones raise).
    - Sorts by ascending date and names the index ``date``.
    - Drops any extra columns and fixes column order.

    Args:
        df: Raw OHLCV frame with a DatetimeIndex.

    Returns:
        A new frame with exactly ``OHLCV_COLUMNS`` in order.

    Raises:
        ValueError: If a required column is missing after renaming.
    """
    out = df.rename(columns=_RENAME).copy()

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing required columns: {missing}")

    out = out[OHLCV_COLUMNS]
    out.index = pd.to_datetime(out.index)
    out.index.name = INDEX_NAME
    out = out.sort_index()
    return out
