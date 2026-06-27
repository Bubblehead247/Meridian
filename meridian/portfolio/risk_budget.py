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
