"""Universe-wide validation.

Runs the same honest evaluation as the single-asset engine — anchored
walk-forward, block bootstrap, Monte-Carlo, multiple-testing correction — but on
the *portfolio* return series built across the whole universe. Averaging over
many names shrinks idiosyncratic noise, so the out-of-sample Sharpe is estimated
far more precisely: this is the test with real statistical power.

The Monte-Carlo null is the portfolio-level analogue of the single-asset one:
each symbol's held-weight path is independently circularly rotated against its
own returns, preserving every name's exposure footprint while destroying its
timing — then re-aggregated into a portfolio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.portfolio.universe import common_index, run_universe_backtest, union_index
from meridian.signals import SignalConfig
from meridian.validation.bootstrap import block_bootstrap_sharpe
from meridian.validation.correction import correct
from meridian.validation.stats import sharpe, total_return
from meridian.validation.walkforward import WalkForwardSpec, make_folds


def _portfolio_mc_pvalue(
    held_weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    n: int = 2000,
    periods_per_year: int = 252,
    seed: int = 0,
) -> float:
    """Monte-Carlo p-value vs a per-symbol rotation null (no-skill timing)."""
    W = np.nan_to_num(held_weights.to_numpy(dtype=float))
    R = np.nan_to_num(returns.to_numpy(dtype=float))
    m, k = W.shape
    actual = sharpe((W * R).sum(axis=1), periods_per_year)
    if m < 3 or np.isnan(actual) or not np.any(W):
        return float("nan")

    rng = np.random.default_rng(seed)
    null = np.empty(n)
    for i in range(n):
        rolled = np.empty_like(W)
        for j in range(k):
            rolled[:, j] = np.roll(W[:, j], int(rng.integers(1, m)))
        null[i] = sharpe((rolled * R).sum(axis=1), periods_per_year)
    null = null[~np.isnan(null)]
    if null.size == 0:
        return float("nan")
    return float((np.sum(null >= actual) + 1) / (null.size + 1))


def validate_universe(
    prices_by_symbol: dict[str, pd.Series],
    estimators: list[str],
    deviation: str = "zscore",
    signal: SignalConfig | None = None,
    *,
    sizing: str = "equal_weight",
    spec: WalkForwardSpec | None = None,
    window: int = 20,
    cost_bps: float = 1.0,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    n_boot: int = 2000,
    n_mc: int = 2000,
    block: int = 20,
    alpha: float = 0.05,
    method: str = "bh",
    periods_per_year: int = 252,
    seed: int = 0,
    align: str = "union",
) -> pd.DataFrame:
    """Walk-forward validate each estimator as a universe-wide portfolio.

    ``align="union"`` (default) uses the full calendar so staggered listings
    (recent IPOs) are handled — a name simply sits inactive before it lists.
    ``align="intersection"`` restricts to dates every symbol shares.

    Returns a ranked verdict table (same columns as the single-asset
    `validate`, plus `n_symbols`), sorted by out-of-sample Sharpe.
    """
    idx = union_index(prices_by_symbol) if align == "union" else common_index(prices_by_symbol)
    spec = spec or WalkForwardSpec()
    folds = make_folds(len(idx), spec)

    rows = []
    for est in estimators:
        pr = run_universe_backtest(
            prices_by_symbol, est, deviation, signal,
            sizing=sizing, window=window, cost_bps=cost_bps,
            bars_by_symbol=bars_by_symbol, index=idx,
        )
        net = pr.returns
        stitched = pd.concat([net.iloc[f.test_start:f.test_end] for f in folds])
        held_oos = pd.concat([pr.weights.iloc[f.test_start:f.test_end] for f in folds])
        ret_oos = pd.concat([pr.returns_by_symbol.iloc[f.test_start:f.test_end] for f in folds])

        boot = block_bootstrap_sharpe(
            stitched, n_boot=n_boot, block=block, periods_per_year=periods_per_year, seed=seed
        )
        mc_p = _portfolio_mc_pvalue(
            held_oos, ret_oos, n=n_mc, periods_per_year=periods_per_year, seed=seed
        )
        rows.append(
            {
                "estimator": est,
                "oos_sharpe": boot["point"],
                "oos_return": total_return(stitched),
                "boot_ci_low": boot["ci_low"],
                "boot_ci_high": boot["ci_high"],
                "mc_pvalue": mc_p,
                "n_symbols": len(prices_by_symbol),
            }
        )

    df = pd.DataFrame(rows)
    res = correct(df["mc_pvalue"].fillna(1.0).to_numpy(), method=method, alpha=alpha)
    q_col = res["qvalues"] if method == "bh" else res["adjusted"]
    df = df.assign(q_value=q_col, significant=res["reject"])
    return df.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
