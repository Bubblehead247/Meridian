"""Multiple-testing corrections.

Testing 38 estimators on the same data means some will look good by chance. If
each test uses alpha = 0.05, you expect ~2 false winners out of 38 even if none
has real skill. These corrections adjust for that, as mandated by CLAUDE.md.

- **Bonferroni**: control the chance of *any* false positive; divide alpha by the
  number of tests. Strict, low power.
- **Benjamini-Hochberg (BH)**: control the *false discovery rate* — the expected
  fraction of false positives among the winners. Less strict, more power; the
  recommended default for a screen of many strategies.
"""

from __future__ import annotations

import numpy as np


def bonferroni(pvalues, alpha: float = 0.05) -> dict:
    """Bonferroni correction.

    Returns dict with ``reject`` (bool array), ``adjusted`` (p*m capped at 1),
    and the corrected per-test threshold ``alpha_adj``.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    adjusted = np.minimum(p * m, 1.0)
    return {"reject": p <= alpha / m, "adjusted": adjusted, "alpha_adj": alpha / m}


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR correction.

    Returns dict with ``reject`` (bool array, original order) and ``qvalues``
    (BH-adjusted p-values, original order).
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return {"reject": np.array([], dtype=bool), "qvalues": np.array([])}

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    # BH-adjusted q-values: enforce monotonicity from the largest p down.
    raw_q = ranked * m / ranks
    qvals_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    qvals_sorted = np.minimum(qvals_sorted, 1.0)

    # Largest rank passing p_(k) <= k/m * alpha determines rejections.
    passing = ranked <= ranks / m * alpha
    k = np.max(np.where(passing)[0]) + 1 if passing.any() else 0
    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[:k] = True

    qvalues = np.empty(m)
    reject = np.empty(m, dtype=bool)
    qvalues[order] = qvals_sorted
    reject[order] = reject_sorted
    return {"reject": reject, "qvalues": qvalues}


def correct(pvalues, method: str = "bh", alpha: float = 0.05) -> dict:
    """Apply a correction by name ('bh' or 'bonferroni')."""
    if method == "bh":
        return benjamini_hochberg(pvalues, alpha)
    if method == "bonferroni":
        return bonferroni(pvalues, alpha)
    raise ValueError("method must be 'bh' or 'bonferroni'")
