"""External statistical skepticism benchmarks (P3).

Harvey, Liu & Zhu, "...and the Cross-Section of Expected Returns" (2016, Review of
Financial Studies), argue that decades of factor/strategy research in finance amounts
to a huge, uncoordinated multiple-testing problem across the whole literature — not
just within one paper's own trial count — and that the conventional t>2 ("95%
significant") threshold is far too lenient given how many factors have been tried
across the field. They recommend a stricter t-statistic hurdle of approximately 3.0 for
a *newly proposed* factor/strategy to be taken seriously.

**How this is used here.** This is an external, independently-sourced skepticism
check, not a replacement for Meridian's own multiple-testing machinery (BH/Bonferroni
with ``m``/``m_eff``, DSR, CPCV/PBO) — those answer "is this result significant given
the specific trials run in *this* research program," while t>3 asks a cruder, blunter
question: "would this survive a much stricter bar calibrated to how much data-mining
has occurred *across the whole field* historically?" Per the audit brief: do not stack
this on top of the existing corrections as if they compound (t>3 AND Bonferroni AND
DSR AND PBO all simultaneously would be needlessly conservative, answering the same
underlying "is this real" question four different ways rather than four different
questions) — report it alongside the others as an independent cross-check, not a
required additional gate.

The formula itself (t = SR_per_period * sqrt(n)) is the elementary Sharpe-ratio
t-statistic, not something requiring Rule 5's "verify against a primary source"
treatment — it follows directly from SR = mean/std and the standard error of the mean
being std/sqrt(n) under i.i.d. sampling. It does NOT correct for serial correlation
(unlike the block bootstrap) or non-normality (unlike ``deflated_sharpe.py``'s
``sharpe_ratio_stdev``) — it is deliberately the crude, textbook version the original
recommendation is calibrated against.
"""

from __future__ import annotations

import numpy as np

from meridian.signals.backtest import PERIODS_PER_YEAR

#: Harvey, Liu & Zhu (2016)'s recommended t-statistic hurdle for a newly proposed
#: factor/strategy, given the scale of multiple testing across the finance literature
#: as a whole. Deliberately not "3" exactly in their paper (they give a range roughly
#: 3.0-3.4 depending on assumptions) — 3.0 is the commonly-cited round figure.
HARVEY_LIU_ZHU_T_THRESHOLD = 3.0


def classical_t_stat(returns, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Classical t-statistic for "true per-period Sharpe > 0": ``SR * sqrt(n)``.

    Works entirely in per-period terms (``t = mean/std * sqrt(n)``, the standard
    t-statistic of the mean), the same convention ``deflated_sharpe.py``'s
    ``sharpe_ratio_stdev`` uses — annualizing would require an "annualized n" that
    isn't a meaningful quantity, so this never annualizes. ``periods_per_year`` is
    accepted only for signature symmetry with the rest of this package; it's unused.

    Returns NaN if fewer than 2 finite observations or the return series is degenerate
    (zero variance).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    if n < 2:
        return float("nan")
    std = r.std(ddof=1)
    if std != std or std <= 0:
        return float("nan")
    sr = r.mean() / std
    return float(sr * np.sqrt(n))


def clears_harvey_liu_zhu_bar(
    returns, *, threshold: float = HARVEY_LIU_ZHU_T_THRESHOLD
) -> bool:
    """Whether ``returns`` clears the (default 3.0) t-statistic hurdle, either
    direction (``abs(t) > threshold``) — a strategy with a strongly negative Sharpe is
    just as "not noise" as a strongly positive one, even though only the positive case
    is normally tradable.
    """
    t = classical_t_stat(returns)
    return bool(t == t and abs(t) > threshold)
