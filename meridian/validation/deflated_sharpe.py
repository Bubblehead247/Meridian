"""Deflated Sharpe Ratio (DSR) — is the best of N trials still good?

Testing many estimators and picking the best Sharpe is a selection procedure:
even N zero-skill trials produce an expected *maximum* Sharpe well above zero,
purely from noise. DSR (Bailey & Lopez de Prado, 2014) answers whether an
observed Sharpe is still distinguishable from that expected maximum, after
also correcting for non-normal returns (skew/kurtosis inflate or deflate a
Sharpe's true uncertainty relative to the textbook Gaussian standard error).

Formulas here are pinned to a verified, code-checked source — not transcribed
from memory — after an earlier draft of this module's design used a different
(incorrect) closed form. Cross-checked against a secondary explainer
(https://quantdare.com/deflated-sharpe-ratio-how-to-avoid-been-fooled-by-randomness/)
and the literal source of a commonly-cited reference implementation
(https://github.com/rubenbriones/Probabilistic-Sharpe-Ratio/blob/master/src/sharpe_ratio_stats.py),
which agree with each other. A PDF-to-text extraction of the primary paper
disagreed with both on the expected-max-Sharpe term and was discarded as
likely math-OCR noise.

All quantities here are **per-period** (not annualized) — annualize only for
display, same convention as ``analytics/metrics.py``. Kurtosis is **raw**
(``scipy.stats.kurtosis(..., fisher=False)``, 3.0 = normal), computed directly
in this module rather than reusing ``analytics.metrics``'s kurtosis field,
which reports *excess* (Fisher) kurtosis — mixing the two silently shifts
every result by a factor involving 3.0.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _sps

_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio_stdev(returns, sr: float | None = None) -> float:
    """Standard deviation of the estimated Sharpe ratio, adjusted for non-normality.

    Bailey & Lopez de Prado's generalization of the Sharpe ratio's standard
    error to non-normal returns:

        sqrt((1 + 0.5*sr^2 - skew*sr + ((kurtosis - 3) / 4) * sr^2) / (n - 1))

    Args:
        returns: Per-period return series (array-like).
        sr: Per-period Sharpe ratio (mean/std, ddof=1). Computed from
            ``returns`` if not given.

    Returns:
        The standard deviation of the Sharpe estimate, in per-period units.
        NaN if fewer than 4 observations (skew/kurtosis undefined).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 4:
        return float("nan")
    if sr is None:
        std = arr.std(ddof=1)
        sr = float(arr.mean() / std) if std > 0 else float("nan")
    skew = float(_sps.skew(arr))
    kurtosis = float(_sps.kurtosis(arr, fisher=False))  # raw kurtosis, 3.0 = normal
    inside = 1.0 + 0.5 * sr**2 - skew * sr + ((kurtosis - 3.0) / 4.0) * sr**2
    if inside < 0:
        return float("nan")
    return float(np.sqrt(inside / (n - 1)))


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, sr_std: float) -> float:
    """PSR: P(true Sharpe > sr_benchmark), given the estimated Sharpe and its std dev."""
    if sr_std != sr_std or sr_std <= 0:  # NaN or non-positive
        return float("nan")
    return float(_sps.norm.cdf((sr - sr_benchmark) / sr_std))


def expected_max_sharpe(trial_sharpes, n_trials: int | None = None) -> float:
    """Expected maximum Sharpe among ``n_trials`` independent zero-skill trials.

    Args:
        trial_sharpes: Per-period Sharpe ratios observed across all trials
            (used only for their standard deviation, per the reference
            formula — the expected maximum scales with how spread out the
            trial outcomes are).
        n_trials: Number of independent trials. Defaults to ``len(trial_sharpes)``.

    Returns:
        The expected maximum per-period Sharpe under the null that every
        trial has zero true skill. NaN if fewer than 2 trials or the trial
        Sharpes are degenerate (std == 0 or undefined).
    """
    arr = np.asarray(trial_sharpes, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = n_trials if n_trials is not None else arr.size
    if n < 2 or arr.size < 2:
        return float("nan")
    std = float(arr.std(ddof=1))
    if std != std or std <= 0:
        return float("nan")
    g = _EULER_MASCHERONI
    max_z = (1 - g) * _sps.norm.ppf(1 - 1.0 / n) + g * _sps.norm.ppf(1 - 1.0 / (n * np.e))
    return float(std * max_z)


def deflated_sharpe_ratio(
    returns, *, trial_sharpes, n_trials: int | None = None
) -> dict:
    """DSR: PSR benchmarked against the expected maximum Sharpe under N trials.

    Args:
        returns: The selected trial's per-period return series.
        trial_sharpes: Per-period Sharpe ratios across all trials (including
            the selected one) — used to estimate the expected maximum under
            the null.
        n_trials: Number of independent trials. Defaults to ``len(trial_sharpes)``.

    Returns:
        dict with ``sr`` (observed per-period Sharpe), ``sr_std``,
        ``expected_max_sr``, and ``dsr_pvalue`` — the DSR itself, a
        probability in [0, 1] that the true Sharpe exceeds the expected
        noise-only maximum. Named to match the correction pipeline's other
        ``*_pvalue`` fields, but reads the **opposite** direction from a
        classical p-value: high (e.g. > 0.95) means significant, not low.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    std = arr.std(ddof=1) if arr.size > 1 else float("nan")
    sr = float(arr.mean() / std) if std == std and std > 0 else float("nan")
    sr_std = sharpe_ratio_stdev(arr, sr=sr)
    exp_max = expected_max_sharpe(trial_sharpes, n_trials=n_trials)
    dsr = probabilistic_sharpe_ratio(sr, exp_max, sr_std) if exp_max == exp_max else float("nan")
    return {"sr": sr, "sr_std": sr_std, "expected_max_sr": exp_max, "dsr_pvalue": dsr}
