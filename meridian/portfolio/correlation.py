"""Inter-sleeve return correlation matrix computation and monitoring.

This module owns the inter-sleeve return correlation matrix; it does NOT own
position sizing or allocation policy (those stay in sizing.py and allocation.py).

Diversification is the whole point of running many sleeves, so the fund tracks how
correlated their return streams are: a full matrix, the per-sleeve correlations written
back onto each ledger, and a flag for any pair that has crept too high (a hidden
concentration). All correlations are Pearson on the sleeves' common calendar.
"""

from __future__ import annotations

import pandas as pd

from meridian.portfolio.ledger import StrategyLedger


def build_correlation_matrix(sleeve_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise Pearson correlation matrix of sleeve return streams.

    Returns a symmetric ``DataFrame`` indexed/columned by sleeve name (aligned on the
    common calendar; pandas handles the pairwise overlap). Empty input → empty frame.
    """
    if not sleeve_returns:
        return pd.DataFrame()
    return pd.DataFrame(sleeve_returns).corr()


def pairwise_correlations(name: str, matrix: pd.DataFrame) -> dict[str, float]:
    """One sleeve's correlation to each *other* sleeve, from a prebuilt matrix."""
    if name not in matrix:
        return {}
    row = matrix[name].drop(labels=[name], errors="ignore")
    return {other: float(v) for other, v in row.items()}


def average_correlation(matrix: pd.DataFrame) -> float:
    """Mean of the off-diagonal correlations — a single diversification gauge.

    Lower is better-diversified. NaN for a 0/1-sleeve matrix (no pairs).
    """
    if matrix.shape[0] < 2:
        return float("nan")
    m = matrix.to_numpy(dtype=float)
    n = m.shape[0]
    off = (m.sum() - n) / (n * n - n)   # subtract the n diagonal 1.0s
    return float(off)


def flag_high_correlation(
    matrix: pd.DataFrame, threshold: float = 0.7
) -> list[tuple[str, str]]:
    """Unordered sleeve pairs whose correlation exceeds ``threshold`` (hidden concentration)."""
    cols = list(matrix.columns)
    flagged: list[tuple[str, str]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = matrix.at[a, b]
            if pd.notna(v) and v > threshold:
                flagged.append((a, b))
    return flagged


def update_ledger_correlations(
    ledgers: list[StrategyLedger], sleeve_returns: dict[str, pd.Series]
) -> pd.DataFrame:
    """Write each ledger's correlations-to-other-sleeves (via update_metrics).

    Returns the full matrix it built, so callers can also flag/monitor it.
    """
    matrix = build_correlation_matrix(sleeve_returns)
    for led in ledgers:
        led.update_metrics(correlations=pairwise_correlations(led.name, matrix))
    return matrix
