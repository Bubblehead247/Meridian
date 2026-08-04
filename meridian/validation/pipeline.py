"""End-to-end validation: WFO -> bootstrap + Monte-Carlo -> correction.

`validate` produces the project's verdict table: for each estimator, its honest
out-of-sample Sharpe, a bootstrap confidence interval, a Monte-Carlo p-value vs
a no-skill null, and a multiple-testing-corrected significance flag. This is the
chain described in the design — honest curves, then uncertainty, then verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.signals import SignalConfig
from meridian.validation.bootstrap import block_bootstrap_sharpe
from meridian.validation.correction import correct
from meridian.validation.deflated_sharpe import deflated_sharpe_ratio
from meridian.validation.effective_tests import effective_num_tests
from meridian.validation.montecarlo import monte_carlo_pvalue
from meridian.validation.sensitivity import parameter_sensitivity
from meridian.validation.stats import total_return
from meridian.validation.walkforward import (
    WalkForwardSpec,
    walk_forward,
)


def validate(
    prices: pd.Series,
    estimators: list[str],
    deviation: str = "zscore",
    signal: SignalConfig | None = None,
    *,
    spec: WalkForwardSpec | None = None,
    window: int = 20,
    cost_bps: float = 1.0,
    bars: pd.DataFrame | None = None,
    n_boot: int = 2000,
    n_mc: int = 2000,
    block: int = 20,
    alpha: float = 0.05,
    method: str = "bh",
    periods_per_year: int = 252,
    seed: int = 0,
    sensitivity: bool = False,
) -> pd.DataFrame:
    """Run the full validation and return a ranked verdict table.

    Columns: estimator, oos_sharpe, oos_return, boot_ci_low, boot_ci_high,
    boot_p, mc_pvalue, q_value/significant (raw multiple-testing correction,
    m = number of estimators tested), q_value_eff/significant_eff (same
    correction using ``m_eff`` — the collinearity-adjusted effective test
    count from ``effective_tests.py``, since most estimators are correlated
    variants of a few underlying ideas, not independent trials), m_eff,
    dsr_pvalue (Deflated Sharpe Ratio — probability the true Sharpe beats the
    expected noise-only maximum across all trials; unlike a normal p-value,
    *higher* means more significant). With ``sensitivity=True``, also
    sharpe_cv/sharpe_sign_stable (see ``validation/sensitivity.py`` — a
    read-only robustness check that never changes which window is used here).
    Sorted by oos_sharpe.

    ``q_value``/``significant`` remain the raw-m correction for backward
    compatibility — a result only clears the bar under the effective-m
    correction is visible via the ``_eff`` columns, not silently promoted.
    """
    wf = walk_forward(
        prices, estimators, [deviation], signal,
        spec=spec, window=window, cost_bps=cost_bps, bars=bars,
    )
    market = wf.oos_market_returns()

    rows = []
    returns_by_estimator: dict[str, pd.Series] = {}
    per_period_sharpes: dict[str, float] = {}
    for est in estimators:
        key = (est, deviation)
        rets = wf.stitched_returns(key)
        held = wf.stitched_positions(key)
        mret = market.reindex(held.index).fillna(0.0)
        returns_by_estimator[est] = rets

        arr = rets.to_numpy(dtype=float)
        arr = arr[~np.isnan(arr)]
        std = arr.std(ddof=1) if arr.size > 1 else float("nan")
        per_period_sharpes[est] = float(arr.mean() / std) if std == std and std > 0 else float("nan")

        boot = block_bootstrap_sharpe(
            rets, n_boot=n_boot, block=block, periods_per_year=periods_per_year, seed=seed
        )
        mc = monte_carlo_pvalue(
            mret, held, cost_bps=0.0, n=n_mc, periods_per_year=periods_per_year, seed=seed
        )
        row = {
            "estimator": est,
            "oos_sharpe": boot["point"],
            "oos_return": total_return(rets),
            "boot_ci_low": boot["ci_low"],
            "boot_ci_high": boot["ci_high"],
            "boot_p": boot["p_value"],
            "mc_pvalue": mc["p_value"],
        }
        if sensitivity:
            sens = parameter_sensitivity(
                prices, est, deviation, signal, window,
                bars=bars, cost_bps=cost_bps, spec=spec,
                n_boot=min(n_boot, 500), block=block,
                periods_per_year=periods_per_year, seed=seed,
            )
            row["sharpe_cv"] = sens["cv"]
            row["sharpe_sign_stable"] = sens["sign_stable"]
        rows.append(row)

    # Deflated Sharpe Ratio: is the best-of-N trial still good given N chances to be lucky?
    trial_sharpes = list(per_period_sharpes.values())
    dsr_pvalues = []
    for est in estimators:
        dsr = deflated_sharpe_ratio(
            returns_by_estimator[est], trial_sharpes=trial_sharpes, n_trials=len(estimators)
        )
        dsr_pvalues.append(dsr["dsr_pvalue"])

    df = pd.DataFrame(rows)
    df["dsr_pvalue"] = dsr_pvalues

    pvals = df["mc_pvalue"].fillna(1.0).to_numpy()
    res_raw = correct(pvals, method=method, alpha=alpha)
    q_col_raw = res_raw["qvalues"] if method == "bh" else res_raw["adjusted"]
    df = df.assign(q_value=q_col_raw, significant=res_raw["reject"])

    m_eff = effective_num_tests(returns_by_estimator)
    res_eff = correct(pvals, method=method, alpha=alpha, m_eff=m_eff)
    q_col_eff = res_eff["qvalues"] if method == "bh" else res_eff["adjusted"]
    df = df.assign(m_eff=m_eff, q_value_eff=q_col_eff, significant_eff=res_eff["reject"])

    return df.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
