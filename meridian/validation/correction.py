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


def bonferroni(pvalues, alpha: float = 0.05, m_eff: float | None = None) -> dict:
    """Bonferroni correction.

    Args:
        pvalues: Per-test p-values.
        alpha: Family-wise error rate to control.
        m_eff: Effective number of independent tests (see
            ``validation/effective_tests.py``). When given, it replaces the
            raw test count ``m`` in the threshold/adjustment math only — the
            actual number of tests run (``p.size``) is unchanged. Must be
            ``<= p.size``; a smaller effective count means a *less*
            conservative (higher) threshold.

    Returns dict with ``reject`` (bool array), ``adjusted`` (p*m capped at 1),
    and the corrected per-test threshold ``alpha_adj``.
    """
    p = np.asarray(pvalues, dtype=float)
    m = m_eff if m_eff is not None else p.size
    adjusted = np.minimum(p * m, 1.0)
    return {"reject": p <= alpha / m, "adjusted": adjusted, "alpha_adj": alpha / m}


def benjamini_hochberg(pvalues, alpha: float = 0.05, m_eff: float | None = None) -> dict:
    """Benjamini-Hochberg FDR correction.

    Args:
        pvalues: Per-test p-values.
        alpha: False discovery rate to control.
        m_eff: Effective number of independent tests (see
            ``validation/effective_tests.py``). When given, it replaces the
            raw test count in the *threshold* math (``ranks/m_eff * alpha``
            and ``ranked * m_eff / ranks``) — rank order and the actual number
            of tests (``p.size``) are unchanged, only the multiple-testing
            penalty uses the effective count.

    Returns dict with ``reject`` (bool array, original order) and ``qvalues``
    (BH-adjusted p-values, original order).
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return {"reject": np.array([], dtype=bool), "qvalues": np.array([])}
    denom = m_eff if m_eff is not None else m

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    # BH-adjusted q-values: enforce monotonicity from the largest p down.
    raw_q = ranked * denom / ranks
    qvals_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    qvals_sorted = np.minimum(qvals_sorted, 1.0)

    # Largest rank passing p_(k) <= k/denom * alpha determines rejections.
    passing = ranked <= ranks / denom * alpha
    k = np.max(np.where(passing)[0]) + 1 if passing.any() else 0
    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[:k] = True

    qvalues = np.empty(m)
    reject = np.empty(m, dtype=bool)
    qvalues[order] = qvals_sorted
    reject[order] = reject_sorted
    return {"reject": reject, "qvalues": qvalues}


def correct(pvalues, method: str = "bh", alpha: float = 0.05, m_eff: float | None = None) -> dict:
    """Apply a correction by name ('bh' or 'bonferroni').

    ``m_eff``, when given, is the effective (collinearity-adjusted) test
    count used for the correction's threshold — see ``bonferroni``/
    ``benjamini_hochberg`` for exactly where it applies.
    """
    if method == "bh":
        return benjamini_hochberg(pvalues, alpha, m_eff=m_eff)
    if method == "bonferroni":
        return bonferroni(pvalues, alpha, m_eff=m_eff)
    raise ValueError("method must be 'bh' or 'bonferroni'")
