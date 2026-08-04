"""Effective number of independent tests, from the trials' return correlation.

Most of Meridian's 42 estimators are cascaded/adaptive variants of one
moving-average idea (see ``estimators/moving_average.py``) — correlated, not
independent, trials. Correcting multiple testing (BH/Bonferroni) as if all 42
were independent is conservative in a way that muddies the "no alpha"
conclusion's precision. This derives an effective trial count from the actual
correlation of the trials' return series, not a hand-maintained "family" tag.

Formula: ``n_eff = mean_corr + (1 - mean_corr) * m``, where ``mean_corr`` is
the average pairwise Pearson correlation among all ``m`` trials' return
series. This is the same formula used by ``num_independent_trials`` in a
commonly-cited Deflated Sharpe Ratio reference implementation
(https://github.com/rubenbriones/Probabilistic-Sharpe-Ratio/blob/master/src/sharpe_ratio_stats.py) —
adopted here (over the Cheverud/Nyholt eigenvalue-variance formula from
genomics multiple-testing) so this correction and ``deflated_sharpe.py`` share
one methodology instead of two unrelated ones. At the boundaries: all trials
identical (mean_corr=1) -> n_eff=1; all trials uncorrelated (mean_corr=0) ->
n_eff=m.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def effective_num_tests(returns_by_key: dict) -> float:
    """Effective number of independent trials among ``returns_by_key``.

    Args:
        returns_by_key: ``{key: return_series}`` for every trial (e.g. one per
            (estimator, deviation) pair). Series are aligned on their common
            index; pandas ``.corr()`` handles pairwise overlap.

    Returns:
        The effective trial count (``1 <= n_eff <= m``). ``m`` (the raw count)
        if there are fewer than 2 trials or every pairwise correlation is
        undefined (e.g. a degenerate zero-variance series).
    """
    m = len(returns_by_key)
    if m < 2:
        return float(m)
    frame = pd.DataFrame(dict(returns_by_key))
    corr = frame.corr().to_numpy(dtype=float)
    n = corr.shape[0]
    off_diagonal = corr[~np.eye(n, dtype=bool)]
    if np.all(np.isnan(off_diagonal)):
        return float(m)
    mean_corr = float(np.nanmean(off_diagonal))
    return float(mean_corr + (1.0 - mean_corr) * m)
