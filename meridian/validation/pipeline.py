"""End-to-end validation: WFO -> bootstrap + Monte-Carlo -> correction.

`validate` produces the project's verdict table: for each estimator, its honest
out-of-sample Sharpe, a bootstrap confidence interval, a Monte-Carlo p-value vs
a no-skill null, and a multiple-testing-corrected significance flag. This is the
chain described in the design — honest curves, then uncertainty, then verdict.
"""

from __future__ import annotations

import pandas as pd

from meridian.signals import SignalConfig
from meridian.validation.bootstrap import block_bootstrap_sharpe
from meridian.validation.correction import correct
from meridian.validation.montecarlo import monte_carlo_pvalue
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
) -> pd.DataFrame:
    """Run the full validation and return a ranked verdict table.

    Columns: estimator, oos_sharpe, oos_return, boot_ci_low, boot_ci_high,
    boot_p, mc_pvalue, q_value (corrected), significant. Sorted by oos_sharpe.
    """
    wf = walk_forward(
        prices, estimators, [deviation], signal,
        spec=spec, window=window, cost_bps=cost_bps, bars=bars,
    )
    market = wf.oos_market_returns()

    rows = []
    for est in estimators:
        key = (est, deviation)
        rets = wf.stitched_returns(key)
        held = wf.stitched_positions(key)
        mret = market.reindex(held.index).fillna(0.0)

        boot = block_bootstrap_sharpe(
            rets, n_boot=n_boot, block=block, periods_per_year=periods_per_year, seed=seed
        )
        mc = monte_carlo_pvalue(
            mret, held, cost_bps=0.0, n=n_mc, periods_per_year=periods_per_year, seed=seed
        )
        rows.append(
            {
                "estimator": est,
                "oos_sharpe": boot["point"],
                "oos_return": total_return(rets),
                "boot_ci_low": boot["ci_low"],
                "boot_ci_high": boot["ci_high"],
                "boot_p": boot["p_value"],
                "mc_pvalue": mc["p_value"],
            }
        )

    df = pd.DataFrame(rows)
    pvals = df["mc_pvalue"].fillna(1.0).to_numpy()
    res = correct(pvals, method=method, alpha=alpha)
    q_col = res["qvalues"] if method == "bh" else res["adjusted"]
    df = df.assign(q_value=q_col, significant=res["reject"])
    return df.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
