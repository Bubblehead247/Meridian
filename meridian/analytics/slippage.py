"""Realized slippage on live fills: fill price vs. decision-time price.

``scoring/scorecard.py``'s ``slippage_sensitivity`` is a backtest-time
*hypothetical* — it charges an assumed extra bps against historical returns to
see how sensitive a strategy's Sharpe is to costs. This module measures what
actually happened: for each real fill, the decision-time price is the prior
close ``live_runner`` sized the order against (it submits after the close,
expecting a next-day-open fill), and slippage is how far the real fill price
strayed from that.

Sign convention: positive bps/dollars means the fill cost more than the
decision price implied (paid more on a buy, received less on a sell) — a real
drag on the strategy. Negative means the fill was favorable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

#: (symbol, submitted_date) -> decision-time price, or None if unresolvable.
PriceLookup = Callable[[str, str], "float | None"]


def fill_slippage_bps(side: str, decision_price: float, fill_price: float) -> float:
    """Cost, in bps of the decision price, of one fill. Positive = costly."""
    if side == "buy":
        return (fill_price - decision_price) / decision_price * 10_000
    return (decision_price - fill_price) / decision_price * 10_000


def slippage_by_fill(records: list[dict], price_lookup: PriceLookup) -> list[dict]:
    """One row per fill with its slippage in bps and $ notional.

    Records missing a ``submitted`` date/symbol, or whose decision price can't
    be resolved (e.g. no price history), are silently skipped — partial
    coverage is better than none, and a KeyError here must never break the
    review that calls this.
    """
    rows = []
    for r in records:
        submitted = r.get("submitted")
        symbol = r.get("symbol")
        price = r.get("price")
        qty = r.get("qty")
        side = r.get("side")
        if not submitted or not symbol or not side or price is None or qty is None:
            continue
        decision_price = price_lookup(symbol, submitted)
        if decision_price is None or decision_price <= 0:
            continue
        fill_price = float(price)
        bps = fill_slippage_bps(side, decision_price, fill_price)
        rows.append({
            "date": r.get("date"),
            "symbol": symbol,
            "family": r.get("family", "unknown"),
            "side": side,
            "decision_price": decision_price,
            "fill_price": fill_price,
            "cost_bps": bps,
            "notional": abs(float(qty)) * fill_price,
        })
    return rows


def _weighted_bps(rows: list[dict]) -> float:
    total_notional = sum(r["notional"] for r in rows)
    if not total_notional:
        return 0.0
    return sum(r["cost_bps"] * r["notional"] for r in rows) / total_notional


def slippage_summary(rows: list[dict]) -> dict:
    """Aggregate slippage stats: overall notional-weighted bps/$ plus a per-sleeve split."""
    if not rows:
        return {"n": 0, "weighted_bps": 0.0, "total_cost_dollars": 0.0, "by_family": {}}

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["family"]].append(r)

    by_family = {
        fam: {
            "n": len(frows),
            "weighted_bps": _weighted_bps(frows),
            "total_cost_dollars": sum(r["cost_bps"] / 10_000 * r["notional"] for r in frows),
        }
        for fam, frows in grouped.items()
    }

    return {
        "n": len(rows),
        "weighted_bps": _weighted_bps(rows),
        "total_cost_dollars": sum(r["cost_bps"] / 10_000 * r["notional"] for r in rows),
        "by_family": by_family,
    }
