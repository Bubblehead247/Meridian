"""Portfolio heat, per-strategy risk contribution, and suspension trigger logic.

This module owns portfolio heat computation, heat budget allocation, and suspension
flags; it does NOT own order execution (that stays in execution/).

"Heat" is risk-at-stop, not dollar exposure: a position's heat is the fraction of account
equity it would lose if stopped out — ``|entry − stop| × size / equity``. Portfolio heat
is the sum across all open positions; each strategy's heat budget is its capital share of
the portfolio-heat cap. A strategy is flagged for suspension when its drawdown breaches
the limit or its heat exceeds its budget (PLAN.md §8). Limits come from the project's
risk decisions (risk/trade 0.25–0.50%, portfolio heat 3–5%, position ≤5–10%, sector
≤20–25%, strategy DD 8–12%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from meridian.portfolio.ledger import StrategyLedger


@dataclass(frozen=True)
class RiskLimits:
    """Aggregate and per-strategy risk limits (fractions of equity)."""

    max_risk_per_trade: float = 0.005   # 0.50%
    max_portfolio_heat: float = 0.05    # 5%
    max_position: float = 0.10          # 10% of equity in one name
    max_sector: float = 0.25            # 25% in one sector
    max_drawdown: float = 0.10          # suspend a strategy past 10% DD


def position_risk(position: dict, equity: float) -> float:
    """Risk-at-stop of one position as a fraction of ``equity``.

    ``|entry_price − stop_price| × size / equity``. A position with no stop (missing
    ``stop_price``) contributes 0 here — its risk is unbounded and must be caught by the
    position-weight / drawdown checks instead.
    """
    if equity <= 0 or "stop_price" not in position:
        return 0.0
    size = abs(float(position.get("size", position.get("qty", 0.0))))
    dist = abs(float(position["entry_price"]) - float(position["stop_price"]))
    return dist * size / equity


def position_weight(position: dict, equity: float) -> float:
    """Gross notional weight of one position (``size × price / equity``)."""
    if equity <= 0:
        return 0.0
    size = abs(float(position.get("size", position.get("qty", 0.0))))
    price = float(position.get("price", position.get("entry_price", 0.0)))
    return size * price / equity


def strategy_heat(ledger: StrategyLedger, account_equity: float) -> float:
    """Sum of a strategy's open-position risk, as a fraction of account equity."""
    return sum(position_risk(p, account_equity) for p in ledger.open_positions)


def compute_portfolio_heat(
    ledgers: list[StrategyLedger], account_equity: float
) -> float:
    """Total open-position risk across all strategies (fraction of account equity)."""
    return sum(strategy_heat(led, account_equity) for led in ledgers)


def heat_budget(
    ledger: StrategyLedger, account_equity: float, limits: RiskLimits | None = None
) -> float:
    """A strategy's heat budget = its capital share × the portfolio-heat cap."""
    limits = limits or RiskLimits()
    if account_equity <= 0:
        return 0.0
    cap_share = ledger.capital_alloc / account_equity
    return cap_share * limits.max_portfolio_heat


def allocate_heat_budget(
    ledgers: list[StrategyLedger], account_equity: float, limits: RiskLimits | None = None
) -> dict[str, float]:
    """Per-strategy heat budget (PLAN.md §8), keyed by strategy name."""
    return {led.name: heat_budget(led, account_equity, limits) for led in ledgers}


def risk_contributions(
    ledgers: list[StrategyLedger], account_equity: float
) -> dict[str, float]:
    """Each strategy's share of total portfolio heat (sums to ~1, or 0 when no heat)."""
    total = compute_portfolio_heat(ledgers, account_equity)
    if total <= 0:
        return {led.name: 0.0 for led in ledgers}
    return {led.name: strategy_heat(led, account_equity) / total for led in ledgers}


def apply_risk_contributions(
    ledgers: list[StrategyLedger], account_equity: float
) -> None:
    """Write each strategy's risk contribution back onto its ledger (via update_metrics)."""
    contrib = risk_contributions(ledgers, account_equity)
    for led in ledgers:
        led.update_metrics(risk_contribution=contrib[led.name])


#: Below this many *jointly-observed* bars (all sleeves aligned), the sample
#: covariance is too noisy to trust for a risk decomposition — return the degenerate
#: (all-zero) result instead of a number built on too little data.
DEFAULT_MIN_COV_PERIODS = 60

#: Condition number above which the covariance matrix is treated as numerically
#: unreliable (near-singular — e.g. two sleeves with almost-identical, or a small
#: joint-observation count relative to the number of sleeves) and ridge-regularized
#: before use. 1e10 is a standard rule-of-thumb "double precision is running out"
#: threshold (a well-conditioned financial covariance matrix is usually << 1e6).
DEFAULT_COND_THRESHOLD = 1e10


def _covariance_diagnostics(
    frame: pd.DataFrame, *, min_periods: int, cond_threshold: float,
) -> tuple[np.ndarray | None, dict]:
    """Build a PSD-safe covariance matrix from ``frame``, or None if unusable.

    Returns ``(cov_or_None, diagnostics)``. ``diagnostics`` always has ``n_periods``,
    ``condition_number`` (NaN if not computed) and ``regularized`` (bool), so a caller
    can see *why* a result is degenerate rather than just getting zeros.
    """
    # Joint (not pairwise) dropna: pandas' default .cov() uses pairwise-complete
    # observations per column pair, which — under differential missingness across
    # sleeves — is not guaranteed to be positive-semi-definite (each entry can come
    # from a different subset of rows). Aligning on the common observed window first
    # guarantees a proper empirical covariance matrix.
    aligned = frame.dropna()
    n = len(aligned)
    diag = {"n_periods": n, "condition_number": float("nan"), "regularized": False}
    if n < min_periods:
        return None, diag

    cov = aligned.cov().to_numpy()
    if not np.all(np.isfinite(cov)):
        return None, diag

    cond = float(np.linalg.cond(cov))
    diag["condition_number"] = cond
    if not np.isfinite(cond) or cond > cond_threshold:
        # Ridge/diagonal loading: a standard numerical-stability fix for a
        # near-singular matrix, NOT a bias-variance shrinkage estimator (that would
        # be Ledoit-Wolf or similar, which needs its own verified derivation before
        # use — see research_integrity_gap_analysis.md's Rule 5). This only nudges
        # the matrix off a singularity; it does not attempt to improve out-of-sample
        # covariance estimation accuracy.
        avg_var = float(np.trace(cov)) / cov.shape[0] if cov.shape[0] else 0.0
        eps = max(avg_var, 1e-12) * 1e-6
        cov = cov + eps * np.eye(cov.shape[0])
        diag["regularized"] = True
        diag["condition_number"] = float(np.linalg.cond(cov))
    return cov, diag


def marginal_risk_contributions(
    sleeve_returns: dict[str, "pd.Series"], weights: dict[str, float],
    *, min_periods: int = DEFAULT_MIN_COV_PERIODS, cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> dict[str, float]:
    """Each sleeve's share of *portfolio volatility* (correlation-adjusted), not heat.

    Standard risk-parity / component-VaR decomposition: portfolio volatility
    ``sigma_p = sqrt(w'*Sigma*w)`` is homogeneous of degree 1 in the weights,
    so by Euler's theorem it decomposes exactly into per-sleeve component
    contributions ``CCTR_i = w_i * (Sigma*w)_i / sigma_p``, which sum exactly
    to ``sigma_p``. Returned as shares of that sum (so they sum to ~1, the
    same convention as ``risk_contributions()``), and both are meant to sit
    side by side — heat measures stop-distance risk, this measures how much
    of the portfolio's actual variance each sleeve explains once correlation
    with the others is accounted for. A sleeve with low heat can still
    dominate this if it's the uncorrelated one carrying most of the swings.

    That the shares sum to exactly 1.0 is a property of Euler decomposition
    arithmetic — it does NOT mean the covariance matrix itself is statistically
    reliable. This function now guards two failure modes that can silently produce a
    misleading (but internally-consistent-looking) decomposition: too little jointly-
    observed history (``min_periods``, joint not pairwise dropna — see
    ``_covariance_diagnostics``) and a near-singular matrix (``cond_threshold`` — ridge-
    regularized rather than trusted as-is). Use ``marginal_risk_contributions_diagnostics``
    to see the condition number / regularization flag / n_periods actually used.

    Args:
        sleeve_returns: ``{sleeve: return_series}`` (e.g. from
            ``experiments/fund.py``'s ``run_fund``/``run_cs_fund``). Sleeves
            with no return series (no model ran there) are excluded from the
            covariance matrix and returned with a 0.0 contribution.
        weights: ``{sleeve: capital_alloc / account_equity}`` for every
            sleeve — only entries also present in ``sleeve_returns`` are used
            to build the covariance matrix.
        min_periods: Minimum jointly-observed bars required to trust the covariance
            estimate; below this, returns the degenerate (all-zero) result.
        cond_threshold: Condition-number ceiling above which the matrix is
            ridge-regularized before use (see ``_covariance_diagnostics``).

    Returns:
        ``{sleeve: share}`` for every key in ``weights`` (0.0 for sleeves
        without a return series, or all sleeves when fewer than 2 have
        returns, too little joint history, or the portfolio variance is degenerate).
    """
    shares, _ = marginal_risk_contributions_diagnostics(
        sleeve_returns, weights, min_periods=min_periods, cond_threshold=cond_threshold,
    )
    return shares


def marginal_risk_contributions_diagnostics(
    sleeve_returns: dict[str, "pd.Series"], weights: dict[str, float],
    *, min_periods: int = DEFAULT_MIN_COV_PERIODS, cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> tuple[dict[str, float], dict]:
    """Same as ``marginal_risk_contributions``, plus the covariance diagnostics.

    Returns ``(shares, diagnostics)`` where ``diagnostics`` has ``n_periods`` (jointly-
    observed bars actually used), ``condition_number`` (post-regularization if
    applied), and ``regularized`` (bool). ``diagnostics`` is ``{}`` when fewer than 2
    sleeves have a return series at all (the decomposition never got as far as
    building a covariance matrix).
    """
    names = [n for n in weights if n in sleeve_returns]
    out = {n: 0.0 for n in weights}
    if len(names) < 2:
        return out, {}

    frame = pd.DataFrame({n: sleeve_returns[n] for n in names})
    cov, diag = _covariance_diagnostics(
        frame, min_periods=min_periods, cond_threshold=cond_threshold
    )
    if cov is None:
        return out, diag

    w = np.array([weights[n] for n in names], dtype=float)
    cov_w = cov @ w
    port_var = float(w @ cov_w)
    if port_var <= 0 or port_var != port_var:
        return out, diag

    sigma_p = np.sqrt(port_var)
    cctr = w * cov_w / sigma_p        # component contributions, sum to sigma_p
    shares = cctr / sigma_p           # normalize to shares summing to ~1
    for n, s in zip(names, shares, strict=True):
        out[n] = float(s)
    return out, diag


def check_suspension(
    ledger: StrategyLedger, account_equity: float, limits: RiskLimits | None = None
) -> tuple[bool, list[str]]:
    """Whether a strategy should be suspended, with the reason(s).

    Triggers (PLAN.md §8): current drawdown beyond ``max_drawdown``, or heat above the
    strategy's ``heat_budget``. Returns ``(suspend, reasons)``.
    """
    limits = limits or RiskLimits()
    reasons: list[str] = []
    if abs(ledger.drawdown_cur) > limits.max_drawdown:
        reasons.append("drawdown")
    if strategy_heat(ledger, account_equity) > heat_budget(ledger, account_equity, limits):
        reasons.append("heat")
    return (bool(reasons), reasons)


def portfolio_heat_breached(
    ledgers: list[StrategyLedger], account_equity: float, limits: RiskLimits | None = None
) -> bool:
    """True when aggregate portfolio heat exceeds the cap."""
    limits = limits or RiskLimits()
    return compute_portfolio_heat(ledgers, account_equity) > limits.max_portfolio_heat


def sector_exposures(
    positions: list[dict], equity: float, sector_of: dict[str, str] | None = None
) -> dict[str, float]:
    """Gross weight by sector (``positions`` carry a ``sector`` key or use ``sector_of``)."""
    out: dict[str, float] = {}
    for p in positions:
        sector = p.get("sector") or (sector_of or {}).get(p.get("symbol", ""), "unknown")
        out[sector] = out.get(sector, 0.0) + position_weight(p, equity)
    return out
