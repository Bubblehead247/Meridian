"""Combinatorial Purged Cross-Validation (CPCV) and Probability of Backtest
Overfitting (PBO) — Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of
Backtest Overfitting" (2015), and Lopez de Prado, "Advances in Financial Machine
Learning" (2018), ch. 11-12.

**What this deliberately is not.** The original method operates on raw features and
labels, purging any training sample whose *evaluation window* overlaps a test sample
(a training label computed from data that extends into the test period is leakage,
even if the training sample's own timestamp precedes the test period). Meridian's
validation pipeline works from already-computed per-bar trial return series (the
output of a walk-forward run), not raw labels, so there is no per-sample evaluation
window to purge against here. What this module purges instead is *proximity in time*:
bars within ``purge_bars`` of a test block's boundary are dropped from training, plus a
further ``embargo_bars`` immediately after each test block — a bar-count-based stand-in
for the original's label-overlap purge, sized to a strategy's holding period rather than
derived from label construction. This is a legitimate, commonly-used adaptation when
working from strategy returns rather than raw features, but it is an adaptation, not a
literal reproduction of the original algorithm — treat ``purge_bars``/``embargo_bars``
as "at least your typical holding period," not as a formally derived quantity.

**Verification.** The PBO formula's *numerical value* was not checked against the
paper's own worked example (no network access this session — see Rule 5 in
research_integrity_gap_analysis.md: this is explicitly flagged as unverified against
the primary source, not silently presumed correct). What IS verified here, in
``tests/test_cpcv.py``, is the estimator's defining statistical property, which follows
directly from the rank-based definition regardless of any paper's specific numbers:
under a null of zero true skill (trials are IID noise), the in-sample winner's OOS rank
is uniformly distributed, so PBO must converge to ~0.5; when one trial has a large,
persistent true edge over the others, it should almost always also win OOS, so PBO must
be low. Both properties are checked numerically. Treat this module's absolute PBO
values as "not independently verified against the primary source," while its relative
behavior (skilled dominance -> low PBO, pure noise -> PBO ~0.5) is confirmed correct.

CPCV/PBO sit downstream of walk-forward/bootstrap/DSR, not in place of them (the audit
brief's own instruction): this reassesses many resampled train/test partitions for
*rank stability* — is the in-sample winner usually also the out-of-sample winner? — which
neither walk-forward (one fixed path) nor the block bootstrap (resamples one trial's own
returns, not a competition among trials) answers. The final fixed holdout
(``data/splits.py``) remains a separate, untouched judge; CPCV/PBO must not replace it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from meridian.validation.stats import sharpe as _sharpe


@dataclass(frozen=True)
class CPCVSplit:
    """One combinatorial train/test partition."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple[int, ...]


def make_cpcv_splits(
    n: int,
    *,
    n_groups: int = 8,
    n_test_groups: int = 2,
    purge_bars: int = 5,
    embargo_bars: int = 5,
) -> list[CPCVSplit]:
    """Every C(``n_groups``, ``n_test_groups``) combinatorial train/test partition of
    ``n`` sequential observations, purged and embargoed around each test block.

    Args:
        n: Number of observations (bars).
        n_groups: Number of equal-length contiguous blocks to partition the data into.
        n_test_groups: How many of those blocks form the test set in each combination
            (the remaining ``n_groups - n_test_groups`` form the training set).
        purge_bars: Training bars immediately *before* a test block's start are
            dropped, to the extent they fall within this many bars of it.
        embargo_bars: Training bars immediately *after* a test block's end are dropped
            for this many bars — guards against the reverse-direction leak the plain
            purge above doesn't cover (a training sample shortly after a test block
            whose own signal still reflects information from inside the test window).

    Returns:
        One ``CPCVSplit`` per combination — ``C(n_groups, n_test_groups)`` splits when
        every one has at least one training and one test observation left after
        purging (a split degenerates to empty and is dropped only in pathological
        cases, e.g. ``purge_bars``/``embargo_bars`` larger than a whole group).
    """
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if not (1 <= n_test_groups < n_groups):
        raise ValueError("n_test_groups must be between 1 and n_groups - 1")
    if n < n_groups:
        raise ValueError(f"n={n} is smaller than n_groups={n_groups}")

    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]

    splits: list[CPCVSplit] = []
    for test_group_ids in itertools.combinations(range(n_groups), n_test_groups):
        test_idx = np.sort(np.concatenate([groups[g] for g in test_group_ids]))
        exclude = set(test_idx.tolist())
        for g in test_group_ids:
            lo, hi = int(groups[g][0]), int(groups[g][-1])
            exclude.update(range(max(0, lo - purge_bars), lo))
            exclude.update(range(hi + 1, min(n, hi + 1 + embargo_bars)))
        train_idx = np.array([i for i in range(n) if i not in exclude], dtype=int)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        splits.append(CPCVSplit(train_idx=train_idx, test_idx=test_idx, test_groups=test_group_ids))
    return splits


def probability_of_backtest_overfitting(
    returns_by_trial: dict[str, pd.Series],
    *,
    n_groups: int = 8,
    n_test_groups: int = 2,
    purge_bars: int = 5,
    embargo_bars: int = 5,
    periods_per_year: int = 252,
) -> dict:
    """PBO: how often does the in-sample-best trial rank at or below the out-of-sample
    median, across every CPCV train/test combination?

    Args:
        returns_by_trial: ``{trial_name: per-bar return series}`` for every candidate
            (e.g. one per estimator) — must share a common index; rows with any NaN
            are dropped (joint alignment, same reasoning as
            ``risk_budget._covariance_diagnostics``: a partial-overlap comparison
            across trials is not a fair ranking).
        n_groups/n_test_groups/purge_bars/embargo_bars: see ``make_cpcv_splits``.
        periods_per_year: Annualization factor for the per-split Sharpe ranking metric.

    Returns:
        dict with ``pbo`` (fraction of valid splits where the in-sample winner's OOS
        rank was at or below the median — high means the selection procedure is likely
        picking up overfit noise, not real skill), ``n_splits`` (total CPCV
        combinations), ``n_valid_splits`` (how many actually contributed a logit — a
        split can be skipped if every trial's Sharpe is degenerate in that partition),
        and ``logits`` (the per-split logit values PBO is the fraction of that are
        <= 0). ``pbo`` is NaN if fewer than 2 trials or no split was valid.
    """
    frame = pd.DataFrame(returns_by_trial).dropna()
    trial_names = list(frame.columns)
    n = len(frame)
    if len(trial_names) < 2 or n == 0:
        return {"pbo": float("nan"), "n_splits": 0, "n_valid_splits": 0, "logits": []}

    splits = make_cpcv_splits(
        n, n_groups=n_groups, n_test_groups=n_test_groups,
        purge_bars=purge_bars, embargo_bars=embargo_bars,
    )
    arr = {t: frame[t].to_numpy(dtype=float) for t in trial_names}

    logits: list[float] = []
    for split in splits:
        is_sharpes = {t: _sharpe(arr[t][split.train_idx], periods_per_year) for t in trial_names}
        finite_is = {t: v for t, v in is_sharpes.items() if v == v}
        if not finite_is:
            continue
        best_is = max(finite_is, key=finite_is.get)

        oos_sharpes = {t: _sharpe(arr[t][split.test_idx], periods_per_year) for t in trial_names}
        best_is_oos = oos_sharpes[best_is]
        if best_is_oos != best_is_oos:
            continue
        oos_vals = np.array(
            [v for v in oos_sharpes.values() if v == v], dtype=float
        )
        if oos_vals.size < 2:
            continue

        # Relative rank (percentile) of the in-sample winner's OOS Sharpe among every
        # trial's OOS Sharpe: 1.0 = it was also the OOS best, ~0 = it was the OOS worst.
        rank = float(np.mean(oos_vals <= best_is_oos))
        rank = min(max(rank, 1e-6), 1 - 1e-6)  # keep the logit finite at the boundaries
        logits.append(float(np.log(rank / (1.0 - rank))))

    if not logits:
        return {"pbo": float("nan"), "n_splits": len(splits), "n_valid_splits": 0, "logits": []}
    pbo = float(np.mean(np.array(logits) <= 0.0))
    return {"pbo": pbo, "n_splits": len(splits), "n_valid_splits": len(logits), "logits": logits}
