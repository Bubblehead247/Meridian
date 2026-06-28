"""Run all paper-stage strategies for today and reconcile with a broker.

Works with any Model or CrossSectionalModel — the same families and models
used in backtesting, driven against a real (Alpaca) or simulated broker.

Typical daily use:
    from meridian.execution.live_runner import run_paper_session
    from meridian.execution import AlpacaBroker
    broker = AlpacaBroker()          # reads ALPACA_API_KEY / ALPACA_SECRET_KEY
    decisions = run_paper_session(broker, account_equity=100_000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from meridian.data import load_ohlcv
from meridian.execution.broker import BaseBroker, Fill
from meridian.families import create_model
from meridian.pipeline.records import load_records
from meridian.portfolio.allocation import SLEEVE_ALLOCATIONS

# Symbols we cannot trade as equities through Alpaca (handled separately later)
_PLACEHOLDER_SYMBOLS = {"SECTORS"}
_CRYPTO_SUFFIXES = ("_USDT", "_USD")


@dataclass
class StrategyDecision:
    """Record of one strategy's signal and execution for a single day."""

    family: str
    model: str
    symbol: str
    as_of: date
    signals: dict[str, int]          # tradeable symbol → {-1, 0, +1}
    target_shares: dict[str, float]  # symbol → target signed shares
    orders: list[Fill]               # fills sent to broker
    skipped: bool = False
    skip_reason: str = ""


def _is_tradeable(symbol_str: str) -> bool:
    """False for placeholder labels and crypto symbols."""
    if symbol_str in _PLACEHOLDER_SYMBOLS:
        return False
    return not any(symbol_str.endswith(s) for s in _CRYPTO_SUFFIXES)


def _parse_symbols(symbol_str: str) -> list[str]:
    """Split a composite symbol string (e.g. 'SPY_TLT_IEF') into individual tickers."""
    return symbol_str.split("_")


def _fetch_prices(symbols: list[str], start: str = "2023-01-01") -> dict[str, pd.Series]:
    """Load closing prices for each symbol from ``start`` to today."""
    out = {}
    for sym in symbols:
        try:
            out[sym] = load_ohlcv(sym, start)["close"]
        except Exception:
            pass
    return out


def _today_signals(model, prices_by_symbol: dict[str, pd.Series]) -> dict[str, int]:
    """Return {symbol: signal} for the most recent bar."""
    cross = getattr(model, "cross_sectional", False)
    if cross:
        sig_df = model.signals(prices_by_symbol)
        if sig_df.empty:
            return {}
        row = sig_df.iloc[-1]
        return {sym: int(row[sym]) for sym in sig_df.columns if sym in row.index}
    else:
        # Single-asset model — prices_by_symbol has exactly one key
        sym, prices = next(iter(prices_by_symbol.items()))
        sig_series = model.signals(prices)
        return {sym: int(sig_series.iloc[-1])} if len(sig_series) else {sym: 0}


def run_paper_session(
    broker: BaseBroker,
    *,
    account_equity: float = 100_000.0,
    price_start: str = "2023-01-01",
    dry_run: bool = False,
) -> list[StrategyDecision]:
    """Run every paper-stage strategy for today; return one StrategyDecision per strategy.

    Args:
        broker: A BaseBroker implementation (AlpacaBroker for real paper trading,
            SimulatedBroker for offline testing).
        account_equity: Total account equity used for position sizing.
        price_start: Earliest date to fetch for indicator warmup. Must be at least
            252 trading days before today to cover 12-month momentum lookbacks.
        dry_run: If True, compute signals and sizes but send no orders to the broker.

    Returns:
        One StrategyDecision per paper record. Check ``decision.skipped`` and
        ``decision.skip_reason`` for records that could not be executed.
    """
    paper_records = [r for r in load_records() if r.stage_passed == "paper"]

    # Count strategies per family for equal intra-sleeve allocation
    from collections import Counter
    family_counts: Counter[str] = Counter(r.family for r in paper_records
                                          if _is_tradeable(r.symbol))

    today = date.today()
    decisions: list[StrategyDecision] = []

    for rec in paper_records:
        # --- skip non-equity symbols ---
        if not _is_tradeable(rec.symbol):
            decisions.append(StrategyDecision(
                family=rec.family, model=rec.model, symbol=rec.symbol, as_of=today,
                signals={}, target_shares={}, orders=[],
                skipped=True, skip_reason="placeholder or crypto symbol",
            ))
            continue

        symbols = _parse_symbols(rec.symbol)

        # --- load prices ---
        prices = _fetch_prices(symbols, start=price_start)
        missing = [s for s in symbols if s not in prices or prices[s].empty]
        if missing:
            decisions.append(StrategyDecision(
                family=rec.family, model=rec.model, symbol=rec.symbol, as_of=today,
                signals={}, target_shares={}, orders=[],
                skipped=True, skip_reason=f"price fetch failed for {missing}",
            ))
            continue

        # --- instantiate model and compute signals ---
        try:
            model = create_model(rec.family, rec.model)
        except KeyError:
            decisions.append(StrategyDecision(
                family=rec.family, model=rec.model, symbol=rec.symbol, as_of=today,
                signals={}, target_shares={}, orders=[],
                skipped=True, skip_reason="model not found in registry",
            ))
            continue

        try:
            sigs = _today_signals(model, prices)
        except Exception as exc:
            decisions.append(StrategyDecision(
                family=rec.family, model=rec.model, symbol=rec.symbol, as_of=today,
                signals={}, target_shares={}, orders=[],
                skipped=True, skip_reason=f"signal error: {exc}",
            ))
            continue

        # --- position sizing ---
        sleeve_weight = SLEEVE_ALLOCATIONS.get(rec.family, 0.0)
        n_strategies = max(family_counts[rec.family], 1)
        per_strategy_equity = account_equity * sleeve_weight / n_strategies

        n_long = sum(1 for v in sigs.values() if v > 0)
        per_position_equity = per_strategy_equity / max(n_long, 1) if n_long else 0.0

        target_shares: dict[str, float] = {}
        for sym, sig in sigs.items():
            if sig == 0 or sym not in prices or prices[sym].empty:
                target_shares[sym] = 0.0
                continue
            price = float(prices[sym].iloc[-1])
            if price <= 0:
                target_shares[sym] = 0.0
                continue
            # Only long positions for paper stage; short signals go flat
            shares = (per_position_equity / price) if sig > 0 else 0.0
            target_shares[sym] = round(shares, 6)

        # --- reconcile with broker ---
        fills: list[Fill] = []
        if not dry_run:
            for sym, target in target_shares.items():
                if sym not in prices or prices[sym].empty:
                    continue
                price = float(prices[sym].iloc[-1])
                if hasattr(broker, "set_price"):
                    broker.set_price(sym, price)
                current = broker.get_position(sym)
                delta = target - current
                if abs(delta) > 0.001:
                    fill = broker.market_order(sym, delta)
                    if fill is not None:
                        fills.append(fill)

        decisions.append(StrategyDecision(
            family=rec.family, model=rec.model, symbol=rec.symbol, as_of=today,
            signals=sigs, target_shares=target_shares, orders=fills,
        ))

    return decisions


def print_session_report(decisions: list[StrategyDecision]) -> None:
    """Print a human-readable summary of a paper session to stdout."""
    from meridian.portfolio.allocation import SLEEVE_ALLOCATIONS

    active = [d for d in decisions if not d.skipped]
    skipped = [d for d in decisions if d.skipped]

    print(f"\n{'='*60}")
    print(f"Meridian paper session  {date.today()}")
    print(f"{'='*60}")
    print(f"  Strategies:  {len(active)} active  |  {len(skipped)} skipped")

    # Group active by family
    by_family: dict[str, list[StrategyDecision]] = {}
    for d in active:
        by_family.setdefault(d.family, []).append(d)

    for family in sorted(by_family):
        pct = SLEEVE_ALLOCATIONS.get(family, 0.0)
        print(f"\n  [{family}]  {pct:.0%} sleeve")
        for d in by_family[family]:
            n_long = sum(1 for v in d.signals.values() if v > 0)
            n_flat = sum(1 for v in d.signals.values() if v == 0)
            order_count = len(d.orders)
            sig_str = f"long={n_long} flat={n_flat}"
            print(f"    {d.model}/{d.symbol[:30]:<30}  {sig_str}  orders={order_count}")
            for sym, shares in d.target_shares.items():
                if shares != 0:
                    print(f"      {sym}: {shares:+.2f} shares")

    if skipped:
        print(f"\n  Skipped ({len(skipped)}):")
        for d in skipped:
            print(f"    {d.family}/{d.model}/{d.symbol}  — {d.skip_reason}")

    print()
