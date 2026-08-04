"""Corporate-action anomaly detection on the adjustment ratio.

This is **not** independent verification against a second data source — no
second provider is integrated in this environment, and building genuine
independence would mean adding one, which is out of scope here. What this
does instead, honestly: yfinance's ``adj_close`` already encodes every split
and dividend it applied; the ratio ``close / adj_close`` is that cumulative
adjustment factor, and it only changes on a bar where a corporate action's
ex-date fell. A sudden jump in that ratio *is* the signal of a corporate
action already present in the trusted source — this flags jumps that don't
look like a clean split factor (and are too large for a plausible dividend)
as worth a human look, rather than trusting the adjustment blindly. It never
modifies the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Day-over-day ratio multipliers a clean stock split would produce
#: (k-for-1 forward splits and 1-for-k reverse splits).
_CLEAN_SPLIT_FACTORS: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5)


def _is_clean_split_factor(factor: float, tolerance: float) -> bool:
    return any(abs(factor - c) / c <= tolerance for c in _CLEAN_SPLIT_FACTORS)


def detect_adjustment_anomalies(
    frame: pd.DataFrame,
    *,
    jump_threshold: float = 0.15,
    round_tolerance: float = 0.03,
) -> pd.DataFrame:
    """Flag day-over-day jumps in the close/adj_close ratio.

    Args:
        frame: OHLCV frame with ``close`` and ``adj_close`` columns (the
            canonical schema — see ``data/schema.py``).
        jump_threshold: Minimum absolute fractional change in the ratio to
            flag (default 15% — well above a normal dividend's effect, so
            routine dividends aren't flagged).
        round_tolerance: How close the jump must be to a clean split factor
            (2, 3, 4, 5, or their reciprocals) to be tagged ``likely_split``
            rather than ``unclean_adjustment``.

    Returns:
        A DataFrame (one row per flagged date) with ``date``, ``ratio``
        (close/adj_close on that date), ``factor_change`` (the day-over-day
        multiplier of the ratio), and ``tag`` (``likely_split`` or
        ``unclean_adjustment`` — the latter is the one worth a closer look).
        Empty if nothing is flagged. Never modifies ``frame``.
    """
    if not {"close", "adj_close"}.issubset(frame.columns):
        raise ValueError("frame must have 'close' and 'adj_close' columns")

    ratio = (frame["close"] / frame["adj_close"]).replace([np.inf, -np.inf], np.nan)
    factor = ratio / ratio.shift(1)
    pct_change = (factor - 1.0).abs()
    flagged = pct_change[(pct_change > jump_threshold) & pct_change.notna()]

    rows = []
    for ts in flagged.index:
        f = float(factor.loc[ts])
        rows.append(
            {
                "date": ts,
                "ratio": float(ratio.loc[ts]),
                "factor_change": f,
                "tag": "likely_split" if _is_clean_split_factor(f, round_tolerance) else "unclean_adjustment",
            }
        )
    return pd.DataFrame(rows, columns=["date", "ratio", "factor_change", "tag"])
