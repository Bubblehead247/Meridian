"""Block bootstrap for Sharpe-ratio uncertainty.

A single Sharpe number hides how uncertain it is. The block bootstrap resamples
the return series many times to build a sampling distribution of the Sharpe,
giving a confidence interval. Resampling is done in **blocks** (not single bars)
to preserve short-term autocorrelation in returns — independent-bar resampling
would understate the uncertainty.

If the confidence interval excludes zero, the Sharpe is distinguishable from no
skill at that confidence level.
"""

from __future__ import annotations

import numpy as np

from meridian.validation.stats import sharpe


def block_bootstrap_sharpe(
    returns,
    *,
    n_boot: int = 2000,
    block: int = 20,
    periods_per_year: int = 252,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap confidence interval and one-sided p-value for the Sharpe.

    Args:
        returns: Per-bar return series.
        n_boot: Number of bootstrap resamples.
        block: Block length in bars (preserves autocorrelation).
        periods_per_year: Annualization factor.
        ci: Confidence level for the interval (e.g. 0.95).
        seed: RNG seed (reproducibility).

    Returns:
        dict with ``point`` (observed Sharpe), ``ci_low``/``ci_high``,
        ``p_value`` (bootstrap fraction with Sharpe <= 0), and ``n``.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    nan_out = {"point": float("nan"), "ci_low": float("nan"),
               "ci_high": float("nan"), "p_value": float("nan"), "n": n}
    if n < block + 1:
        return nan_out

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block
    sharpes = np.empty(n_boot)

    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        sharpes[b] = sharpe(r[idx], periods_per_year)

    sharpes = sharpes[~np.isnan(sharpes)]
    if sharpes.size == 0:
        return nan_out
    lo = float(np.percentile(sharpes, (1 - ci) / 2 * 100))
    hi = float(np.percentile(sharpes, (1 + ci) / 2 * 100))
    return {
        "point": sharpe(r, periods_per_year),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": float(np.mean(sharpes <= 0.0)),
        "n": n,
    }
