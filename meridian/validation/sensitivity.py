"""Parameter-sensitivity diagnostic for the estimator comparison.

CLAUDE.md's "no estimator-specific optimization" principle holds every
estimator's window fixed (20, by convention) for the headline comparison —
correctly, since tuning per estimator would break the fair-comparison premise
the whole project is built on. But that also means nobody currently asks
whether the fixed choice is a knife-edge one: does window=16 or window=25
behave roughly the same, or does the estimator's OOS Sharpe flip sign a few
bars either side of 20?

This module only *diagnoses* that — it reruns the exact same walk-forward +
block-bootstrap pipeline ``validation/pipeline.py::validate()`` already uses,
at a few neighboring windows, and reports how stable the result is. It never
selects a different window for the headline ranking; that would reopen the
door to the estimator-specific optimization the design principle rules out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.signals import SignalConfig
from meridian.validation.bootstrap import block_bootstrap_sharpe
from meridian.validation.walkforward import WalkForwardSpec, walk_forward


def parameter_sensitivity(
    prices: pd.Series,
    estimator: str,
    deviation: str,
    signal: SignalConfig | None = None,
    window: int = 20,
    *,
    bars: pd.DataFrame | None = None,
    cost_bps: float = 1.0,
    spec: WalkForwardSpec | None = None,
    multipliers: tuple[float, ...] = (0.8, 1.0, 1.25),
    n_boot: int = 500,
    block: int = 20,
    periods_per_year: int = 252,
    seed: int = 0,
) -> dict:
    """OOS Sharpe of ``estimator``/``deviation`` at ``window`` and its neighbors.

    Args:
        multipliers: Window multipliers to test, always including 1.0 (the
            actual value used for the headline ranking). Windows are
            ``round(window * m)``, deduplicated and clamped to >= 2.

    Returns:
        dict with ``windows`` (the actual int windows tested, sorted),
        ``oos_sharpes`` (aligned list, NaN where a window's walk-forward
        collapsed to zero folds), ``base_window``, ``base_sharpe``, ``cv``
        (coefficient of variation of the non-NaN Sharpes — NaN if fewer than
        2 are finite or the mean is ~0), and ``sign_stable`` (True if every
        finite neighbor Sharpe has the same sign as ``base_sharpe``, or if
        there aren't enough finite points to judge).
    """
    spec = spec or WalkForwardSpec()
    windows = sorted({max(2, round(window * m)) for m in set(multipliers) | {1.0}})

    sharpes: dict[int, float] = {}
    for w in windows:
        wf = walk_forward(prices, [estimator], [deviation], signal, spec=spec, window=w, bars=bars)
        rets = wf.stitched_returns((estimator, deviation))
        boot = block_bootstrap_sharpe(
            rets, n_boot=n_boot, block=block, periods_per_year=periods_per_year, seed=seed
        )
        sharpes[w] = boot["point"]

    base_sharpe = sharpes.get(window, sharpes.get(round(window)))
    finite = [s for s in sharpes.values() if s == s]  # drop NaN

    cv = float("nan")
    if len(finite) >= 2:
        mean = float(np.mean(finite))
        if mean != 0:
            cv = float(np.std(finite, ddof=0) / abs(mean))

    if base_sharpe is not None and base_sharpe == base_sharpe and finite:
        base_sign = base_sharpe >= 0
        sign_stable = all((s >= 0) == base_sign for s in finite)
    else:
        sign_stable = True  # nothing to contradict

    return {
        "windows": windows,
        "oos_sharpes": [sharpes[w] for w in windows],
        "base_window": window,
        "base_sharpe": base_sharpe,
        "cv": cv,
        "sign_stable": bool(sign_stable),
    }
