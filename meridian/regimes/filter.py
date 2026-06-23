"""Apply regime labels as a downstream gate on positions.

This is where regimes meet the rest of the platform. The signal engine stays
regime-agnostic; here we compute a causal label per bar and flatten any position
taken in a disallowed regime. Mean-reversion, for instance, can be restricted to
``range`` / ``mean_reverting`` / ``low`` states.

Kept separate from `meridian.signals` so the estimator→signal pipeline is never
entangled with regime logic, per the project's separation principle.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from meridian.regimes.registry import create as make_regime
from meridian.signals import (
    BacktestResult,
    SignalConfig,
    backtest,
    compute_scores,
    generate_positions,
)


def classify(
    prices: pd.Series,
    regime,
    *,
    window: int | None = None,
    bars: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute a causal regime label for every bar.

    Args:
        prices: Price series.
        regime: Regime name or instance.
        window: Lookback when given by name (uses the classifier default if None).
        bars: Optional OHLC frame aligned to ``prices``.

    Returns:
        Series of label strings aligned to ``prices`` (``UNKNOWN`` during warmup).
    """
    if isinstance(regime, str):
        clf = make_regime(regime, window=window) if window else make_regime(regime)
    else:
        clf = regime

    labels = []
    for i, p in enumerate(prices.to_numpy(dtype=float)):
        bar = bars.iloc[i] if bars is not None else None
        clf.update(float(p), bar=bar)
        labels.append(clf.label())
    return pd.Series(labels, index=prices.index, name="regime")


def gate_positions(
    positions: pd.Series,
    labels: pd.Series,
    allowed: Iterable[str],
) -> pd.Series:
    """Force positions to flat (0) on any bar whose regime is not ``allowed``.

    Labels are used as decided at each bar; the backtester applies the usual
    one-bar lag, so no look-ahead is introduced.
    """
    allowed_set = set(allowed)
    mask = labels.isin(allowed_set)
    gated = positions.where(mask.to_numpy(), other=0)
    return gated.astype(int)


def run_gated_backtest(
    prices: pd.Series,
    estimator: str,
    deviation: str,
    regime: str,
    allowed: Iterable[str],
    signal: SignalConfig | None = None,
    *,
    window: int = 20,
    regime_window: int | None = None,
    cost_bps: float = 1.0,
    bars: pd.DataFrame | None = None,
) -> BacktestResult:
    """End-to-end pipeline with a regime gate applied to the positions.

    Same controlled pipeline as `signals.run_backtest`, plus: classify each bar
    and zero out positions taken outside ``allowed`` regimes.
    """
    scores = compute_scores(prices, estimator, deviation, window=window, bars=bars)
    positions = generate_positions(scores, signal)
    labels = classify(prices, regime, window=regime_window, bars=bars)
    gated = gate_positions(positions, labels, allowed)

    result = backtest(prices, gated, cost_bps=cost_bps)
    result.meta.update(
        {
            "estimator": estimator,
            "deviation": deviation,
            "window": window,
            "regime": regime,
            "allowed": list(allowed),
        }
    )
    return result
