"""Monte-Carlo significance test by position rotation.

Question: is a strategy's Sharpe real skill, or could a no-skill strategy with
the same trading footprint have done as well by luck?

To build the no-skill null, we keep the exact position path (same number of
trades, same exposure, same holding lengths) but **circularly rotate** it by a
random offset against the market returns. Rotation destroys the alignment
between when the strategy is positioned and how the market actually moved, while
preserving everything else about the position series. The fraction of random
rotations that match or beat the real Sharpe is the p-value.
"""

from __future__ import annotations

import numpy as np

from meridian.validation.stats import sharpe, strategy_net


def monte_carlo_pvalue(
    market_returns,
    held_positions,
    *,
    cost_bps: float = 0.0,
    n: int = 2000,
    periods_per_year: int = 252,
    seed: int = 0,
) -> dict:
    """One-sided Monte-Carlo p-value that the strategy beats no-skill timing.

    Args:
        market_returns: Per-bar market (asset) returns.
        held_positions: Per-bar held (already-lagged) positions aligned to
            ``market_returns``.
        cost_bps: Transaction cost for both real and null (usually 0 for a pure
            timing test).
        n: Number of random rotations.
        periods_per_year: Annualization factor.
        seed: RNG seed.

    Returns:
        dict with ``actual_sharpe``, ``p_value`` (fraction of rotations with
        Sharpe >= actual, with the standard +1/+1 correction), ``null_mean``,
        ``null_std``, ``n``.
    """
    mret = np.asarray(market_returns, dtype=float)
    held = np.asarray(held_positions, dtype=float)
    mask = ~(np.isnan(mret) | np.isnan(held))
    mret, held = mret[mask], held[mask]
    m = held.size

    actual = sharpe(strategy_net(held, mret, cost_bps), periods_per_year)
    nan_out = {"actual_sharpe": actual, "p_value": float("nan"),
               "null_mean": float("nan"), "null_std": float("nan"), "n": m}
    if m < 3 or np.isnan(actual) or np.all(held == 0):
        return nan_out

    rng = np.random.default_rng(seed)
    null = np.empty(n)
    for i in range(n):
        shift = int(rng.integers(1, m))  # nonzero rotation
        null[i] = sharpe(strategy_net(np.roll(held, shift), mret, cost_bps), periods_per_year)

    null = null[~np.isnan(null)]
    if null.size == 0:
        return nan_out
    p = float((np.sum(null >= actual) + 1) / (null.size + 1))
    return {
        "actual_sharpe": actual,
        "p_value": p,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=0)),
        "n": m,
    }
