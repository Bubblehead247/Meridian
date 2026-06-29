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

from dataclasses import dataclass
from datetime import date

import pandas as pd

from meridian.data import load_ohlcv
from meridian.execution.broker import BaseBroker, Fill
from meridian.execution.notify import notify_daily_status, notify_entry, notify_exit
from meridian.families import create_model
from meridian.portfolio.allocation import FAMILY_TO_SLEEVE, SLEEVE_ALLOCATIONS
from meridian.portfolio.live_picks import live_pick_weight, load_live_picks

# Crypto suffixes that Alpaca routes through its crypto endpoint (not equity)
_CRYPTO_SUFFIXES = ("_USDT", "_USD")

# The 11 SPDR sector ETFs — what "SECTORS" expands to at execution time
_SECTOR_UNIVERSE = [
    "XLC", "XLY", "XLP", "XLE", "XLF",
    "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU",
]


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
    weight: float = 0.0              # sleeve weight; 0 = monitor-only
    skipped: bool = False
    skip_reason: str = ""


def _is_tradeable(symbol_str: str) -> bool:
    """False for crypto symbols (routed separately).

    SECTORS and composite tickers are equity-tradeable.
    """
    return not any(symbol_str.endswith(s) for s in _CRYPTO_SUFFIXES)


def _expand_symbol(symbol_str: str) -> list[str]:
    """Resolve a symbol string to a list of individual tickers.

    'SECTORS' expands to the full 11-ETF sector universe.
    Composite strings like 'SPY_TLT_IEF' split on '_'.
    Single tickers are returned as a one-element list.
    """
    if symbol_str == "SECTORS":
        return list(_SECTOR_UNIVERSE)
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
    """Run every curated live pick for today; return one StrategyDecision per family.

    Picks come from ``live_picks.json`` (the source of truth) — exactly one strategy
    per family, so no two strategies in a sleeve fight over the same symbol. If that
    file is missing/empty, nothing trades: a live bot never trades an uncurated book.

    Args:
        broker: A BaseBroker implementation (AlpacaBroker for real paper trading,
            SimulatedBroker for offline testing).
        account_equity: Total account equity used for position sizing.
        price_start: Earliest date to fetch for indicator warmup. Must be at least
            252 trading days before today to cover 12-month momentum lookbacks.
        dry_run: If True, compute signals and sizes but send no orders to the broker.

    Returns:
        One StrategyDecision per live pick. Check ``decision.skipped`` and
        ``decision.skip_reason`` for picks that could not be executed.
    """
    picks = load_live_picks()
    present = set(picks)
    today = date.today()
    decisions: list[StrategyDecision] = []

    for family in sorted(picks):
        model_name = picks[family]["model"]
        symbol = picks[family]["symbol"]

        def _skip(reason: str, *, family=family, model_name=model_name, symbol=symbol) -> None:
            decisions.append(StrategyDecision(
                family=family, model=model_name, symbol=symbol, as_of=today,
                signals={}, target_shares={}, orders=[],
                skipped=True, skip_reason=reason,
            ))

        # --- skip non-equity symbols ---
        if not _is_tradeable(symbol):
            _skip("placeholder or crypto symbol")
            continue

        symbols = _expand_symbol(symbol)

        # --- load prices ---
        prices = _fetch_prices(symbols, start=price_start)
        missing = [s for s in symbols if s not in prices or prices[s].empty]
        if missing:
            _skip(f"price fetch failed for {missing}")
            continue

        # --- instantiate model and compute signals ---
        try:
            model = create_model(family, model_name)
        except KeyError:
            _skip("model not found in registry")
            continue

        try:
            sigs = _today_signals(model, prices)
        except Exception as exc:
            _skip(f"signal error: {exc}")
            continue

        # --- position sizing: full sleeve weight (monitor-only families → 0) ---
        weight = live_pick_weight(family, present)
        per_strategy_equity = account_equity * weight

        n_long = sum(1 for v in sigs.values() if v > 0)
        per_position_equity = per_strategy_equity / max(n_long, 1) if n_long else 0.0

        target_shares: dict[str, float] = {}
        for sym, sig in sigs.items():
            price = (
                float(prices[sym].iloc[-1])
                if sym in prices and not prices[sym].empty else 0.0
            )
            # Only long positions for the paper book; shorts and bad prices go flat.
            if sig <= 0 or price <= 0:
                target_shares[sym] = 0.0
            else:
                target_shares[sym] = round(per_position_equity / price, 6)

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
                        if fill.qty > 0:
                            notify_entry(fill, family, model_name)
                        else:
                            notify_exit(fill, family, model_name)

        decisions.append(StrategyDecision(
            family=family, model=model_name, symbol=symbol, as_of=today,
            signals=sigs, target_shares=target_shares, orders=fills,
            weight=weight,
        ))

    if not dry_run:
        notify_daily_status(decisions)

    return decisions


def print_session_report(decisions: list[StrategyDecision]) -> None:
    """Print a human-readable summary of a paper session to stdout."""
    active = [d for d in decisions if not d.skipped and d.weight > 0]
    skipped = [d for d in decisions if d.skipped]

    print(f"\n{'='*60}")
    print(f"Meridian paper session  {date.today()}")
    print(f"{'='*60}")
    print(f"  Strategies:  {len(active)} active  |  {len(skipped)} skipped")

    # Group active by sleeve (breakouts shares trend_following)
    by_family: dict[str, list[StrategyDecision]] = {}
    for d in active:
        sleeve = FAMILY_TO_SLEEVE.get(d.family, d.family)
        by_family.setdefault(sleeve, []).append(d)

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
