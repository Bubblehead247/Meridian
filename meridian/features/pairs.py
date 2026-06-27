"""Pairs / spread construction for cointegration mean-reversion.

Absolute MR asks "is a price far from its *own* average?"; cross-sectional MR
asks "far from its *peers*?". **Pairs** MR asks "is the *spread* between two
related instruments far from its average?" — the classic statistical-arbitrage
trade (share-class twins, sector substitutes, cap- vs equal-weight indices).

A pair is reduced to **one synthetic instrument** so the entire universe
backtester/validator is reused unchanged. ``build_pairs`` returns two aligned
dicts, exactly the inputs ``run_universe_backtest`` / ``validate_universe``
already accept:

- ``signal_prices`` — the **spread** ``log(P_a) - β·log(P_b)``. It crosses zero,
  so (like a cross-sectional relative series) it feeds the estimator → deviation
  → signal stage as the *signal* input only.
- ``spread_prices`` — a **synthetic price** whose ``pct_change`` equals the
  long-spread portfolio return ``r_a - β·r_b``. P&L flows through the existing
  price-based path untouched.

The hedge ratio β is estimated by a **rolling, one-bar-lagged** OLS so the spread
at bar *t* uses only data through *t-1* — the same causal discipline the
estimators follow (no lookahead).

Cost caveat: the synthetic instrument's turnover counts one leg, but a real
spread trade moves both legs. The cost sweep therefore scales the per-trade cost
by ``(1 + |β|)`` — see the study summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hedge_ratio(log_a: pd.Series, log_b: pd.Series, *, lookback: int = 60) -> pd.Series:
    """Causal rolling OLS slope β of ``log_a`` on ``log_b`` (no intercept drift).

    β_t is the covariance/variance estimate over the trailing ``lookback`` bars
    ending at *t-1*, so it is known before bar *t* trades. Returns a Series
    aligned to the inputs (NaN until enough history, then shifted by one bar).
    """
    a, b = log_a.align(log_b, join="inner")
    cov = a.rolling(lookback).cov(b)
    var = b.rolling(lookback).var()
    beta = (cov / var).shift(1)  # known one bar before it is used
    return beta.reindex(log_a.index)


def build_pair(
    price_a: pd.Series, price_b: pd.Series, *, lookback: int = 60
) -> tuple[pd.Series, pd.Series]:
    """Build one pair's ``(spread_signal, synthetic_spread_price)``.

    Args:
        price_a/price_b: the two legs' tradable price Series (positive prices).
        lookback: rolling window for the hedge ratio β.

    Returns:
        ``(spread, spread_price)`` aligned on the shared calendar:
        - ``spread = log(P_a) - β·log(P_b)`` — the signal series (zero-crossing).
        - ``spread_price = cumprod(1 + (r_a - β·r_b))`` — a synthetic price whose
          ``pct_change`` reproduces the long-spread *simple* return
          ``r_a - β·r_b`` (β lagged, so the weight on leg B's return is known in
          advance); the backtester's P&L path uses ``pct_change`` directly.
        Bars before β is available (or with a missing leg) are NaN/flat.
    """
    a, b = price_a.align(price_b, join="inner")
    log_a, log_b = np.log(a.where(a > 0)), np.log(b.where(b > 0))
    beta = hedge_ratio(log_a, log_b, lookback=lookback)

    spread = log_a - beta * log_b

    r_a = a.pct_change()
    r_b = b.pct_change()
    spread_ret = (r_a - beta * r_b).where(beta.notna())
    spread_price = (1.0 + spread_ret.fillna(0.0)).cumprod()
    spread_price = spread_price.where(beta.notna())
    return spread, spread_price


def build_pairs(
    prices_by_symbol: dict[str, pd.Series],
    pairs: list[tuple[str, str]],
    *,
    lookback: int = 60,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Build ``(signal_prices, spread_prices)`` dicts keyed ``"A/B"`` for a list of pairs.

    Hand the results straight to ``validate_universe`` /
    ``run_universe_backtest`` as ``signal_prices_by_symbol=signal_prices`` and
    ``prices_by_symbol=spread_prices``. Pairs missing a leg are skipped.
    """
    signal_prices: dict[str, pd.Series] = {}
    spread_prices: dict[str, pd.Series] = {}
    for a, b in pairs:
        if a not in prices_by_symbol or b not in prices_by_symbol:
            continue
        spread, spread_price = build_pair(
            prices_by_symbol[a], prices_by_symbol[b], lookback=lookback
        )
        key = f"{a}/{b}"
        signal_prices[key] = spread
        spread_prices[key] = spread_price
    return signal_prices, spread_prices


def screen_cointegrated_pairs(
    prices_by_symbol: dict[str, pd.Series],
    candidates: list[str],
    *,
    max_pvalue: float = 0.05,
) -> list[tuple[str, str]]:
    """Mechanically select cointegrated pairs from a candidate list (no hand-picking).

    Forms **all** unordered pairs of ``candidates`` present in ``prices_by_symbol``,
    runs an Engle-Granger cointegration test on each pair's **log-price** series, and
    returns those with test p-value < ``max_pvalue``, sorted most- to least-cointegrated.

    Pass only the **in-sample** price slice so selection never sees out-of-sample data
    (the test uses the whole series it is given — lookahead-free is the caller's job).
    """
    from itertools import combinations

    from statsmodels.tsa.stattools import coint

    present = [c for c in candidates if c in prices_by_symbol]
    scored: list[tuple[float, tuple[str, str]]] = []
    for a, b in combinations(present, 2):
        joint = pd.concat(
            [np.log(prices_by_symbol[a]), np.log(prices_by_symbol[b])], axis=1
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(joint) < 30:
            continue
        pvalue = coint(joint.iloc[:, 0], joint.iloc[:, 1])[1]
        if pvalue < max_pvalue:
            scored.append((pvalue, (a, b)))
    scored.sort(key=lambda t: t[0])
    return [pair for _, pair in scored]


def leg_cost(
    held_weights: pd.DataFrame,
    beta_by_pair: dict[str, pd.Series],
    half_spread_bps: float,
) -> pd.Series:
    """Honest per-leg transaction cost for a basket of synthetic-spread instruments.

    Each pair's synthetic held weight ``w`` (already lagged) is a position of ``w`` in
    leg A and ``−β·w`` in leg B, so a real trade moves **both** legs. The per-bar cost is
    ``Σ_pairs (|Δw_A| + |Δw_B|) · half_spread_bps/1e4`` — replacing the flat ``(1+|β|)``
    multiplier with the actual two-leg turnover. The first bar is charged as entry from
    flat (``|w_0|``). ``beta_by_pair`` is the lagged hedge ratio per pair column.

    Returns a per-bar cost Series (aligned to ``held_weights.index``) to subtract from the
    **gross** portfolio return.
    """
    total = pd.Series(0.0, index=held_weights.index)
    for key in held_weights.columns:
        w_a = held_weights[key].fillna(0.0)
        beta = beta_by_pair[key].reindex(held_weights.index).fillna(0.0)
        w_b = beta * w_a
        for leg in (w_a, w_b):
            d = leg.diff()
            d.iloc[0] = leg.iloc[0]  # entering from flat
            total = total + d.abs()
    return total * (half_spread_bps / 1e4)
