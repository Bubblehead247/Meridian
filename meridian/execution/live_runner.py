"""Run all paper-stage strategies for today and reconcile with a broker.

Works with any Model or CrossSectionalModel — the same families and models
used in backtesting, driven against a real (Alpaca) or simulated broker.

Typical daily use:
    from meridian.execution.live_runner import run_paper_session
    from meridian.execution import AlpacaBroker
    broker = AlpacaBroker()          # reads ALPACA_API_KEY / ALPACA_SECRET_KEY
    decisions = run_paper_session(broker, account_equity=100_000)

2026-08-07 — order netting: sleeve universes overlap (SECTORS ⊇ XLK, which
trend_following also holds), so on several sessions between 2026-07-22 and
2026-07-30 sector_rotation and trend_following each bought XLK the same day
as two separate broker orders — two spreads paid for one net exposure
change. ``run_paper_session`` now computes every family's approved order
first and nets them per symbol before touching the broker (see the
docstring on ``run_paper_session`` for the two-pass structure), so
offsetting sleeve deltas settle against each other internally instead of
crossing the spread twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from meridian.data import load_ohlcv
from meridian.execution.broker import BaseBroker, Fill
from meridian.execution.notify import notify_daily_status, notify_entry, notify_exit
from meridian.execution import rebalance_schedule
from meridian.execution.positions import POSITIONS_FILE, adjust_position, get_position
from meridian.execution.reconcile import add_pending_order, record_fill
from meridian.families import create_model
from meridian.portfolio.allocation import FAMILY_TO_SLEEVE, SLEEVE_ALLOCATIONS
from meridian.portfolio.live_picks import live_pick_weight, load_live_picks
from meridian.portfolio.sleeve_ledgers import update_sleeve_ledgers

# Crypto suffixes that Alpaca routes through its crypto endpoint (not equity)
_CRYPTO_SUFFIXES = ("_USDT", "_USD")

#: Alpaca's minimum order value in dollars. An order below this is rejected with
#: "cost basis must be >= minimal amount of order 1", so there is no point
#: sending one. Rebalance deltas smaller than this are held back and reoffered on
#: later sessions, accumulating until they clear the floor.
MIN_ORDER_NOTIONAL = 1.0

#: Rebalance band. An adjustment to a position already held is skipped unless it
#: moves at least this fraction of the position's target value.
#:
#: Why: XLK was bought on six consecutive sessions, ratcheting 8.43 → 9.005
#: shares while the price fell, because the target moves a little every day and
#: any difference at all produced an order. The drift is noise, the commission
#: and spread are not.
#:
#: This gates **adjustments only**. Opening a position and closing one are
#: decisions, not drift, so they always go through however small they are —
#: otherwise a signal to exit could be silently swallowed.
#:
#: Adjustments are additionally throttled to once a calendar month (see
#: ``rebalance_schedule.py``) — this band is what still applies *within* that
#: monthly check, so a monthly resize doesn't fire on pure noise either.
REBALANCE_BAND_PCT = 0.05

# The 11 SPDR sector ETFs — what "SECTORS" expands to at execution time
_SECTOR_UNIVERSE = [
    "XLC", "XLY", "XLP", "XLE", "XLF",
    "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU",
]


@dataclass
class _Leg:
    """One family's approved-but-not-yet-sent order, pending symbol netting."""

    family: str
    model: str
    symbol: str
    delta: float
    price: float


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
    day_changes: dict[str, float] = field(default_factory=dict)  # symbol → 1-day % change


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
    """Load closing prices for each symbol from ``start`` to today.

    Bypasses the on-disk cache (``use_cache=False``): the live path must
    always see today's bar, and must never read stale data left behind by a
    previous session or write into the cache backtests rely on for
    reproducibility.
    """
    out = {}
    for sym in symbols:
        try:
            out[sym] = load_ohlcv(sym, start, use_cache=False)["close"]
        except Exception:
            pass
    return out


def _today_signals(model, prices_by_symbol: dict[str, pd.Series]) -> dict[str, int]:
    """Return {symbol: signal} for the most recent bar."""
    cross = getattr(model, "cross_sectional", False)
    if cross:
        # filtered_signals (not signals) strips the trend-filter symbol (e.g. SPY)
        # from the tradeable universe first — otherwise it gets ranked and traded
        # like any other name instead of only gating entries. See base.py.
        sig_df = model.filtered_signals(prices_by_symbol)
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
    positions_path: Path = POSITIONS_FILE,
    rebalance_schedule_path: Path | None = None,
) -> list[StrategyDecision]:
    """Run every curated live pick for today; return one StrategyDecision per family.

    Picks come from ``live_picks.json`` (the source of truth) — exactly one strategy
    per family, so no two strategies in a sleeve fight over the same symbol. If that
    file is missing/empty, nothing trades: a live bot never trades an uncurated book.

    Two passes: pass 1 computes every family's signals, sizing, and gated order
    (unchanged from the original per-family logic) but defers execution — the
    approved leg is collected instead of sent. Pass 2 nets every symbol's legs
    across families into a single broker order (or, if the net is below the
    notional floor, settles the legs internally with no broker order at all)
    before applying each leg's own delta to that family's position book. This
    means two sleeves whose deltas fully or partially offset never each pay
    their own spread for what is one net exposure change at the account level.

    Args:
        broker: A BaseBroker implementation (AlpacaBroker for real paper trading,
            SimulatedBroker for offline testing).
        account_equity: Total account equity used for position sizing.
        price_start: Earliest date to fetch for indicator warmup. Must be at least
            252 trading days before today to cover 12-month momentum lookbacks.
        dry_run: If True, compute signals and sizes but send no orders to the broker.

    Entries and exits still happen the day a model's signal actually opens or
    closes a position — only *resizing an already-open position* toward a
    drifted target is throttled to once a calendar month (see
    ``rebalance_schedule.py``); pass ``rebalance_schedule_path`` to isolate
    that state in tests, same as ``positions_path``.

    Returns:
        One StrategyDecision per live pick. Check ``decision.skipped`` and
        ``decision.skip_reason`` for picks that could not be executed.
    """
    picks = load_live_picks()
    present = set(picks)
    today = date.today()
    # Resolved here (not as a bound default) so a test's monkeypatch of
    # rebalance_schedule.REBALANCE_SCHEDULE_FILE is actually honored when this
    # is called without an explicit path — a bound default captures the value
    # at import time, before any monkeypatch can apply.
    if rebalance_schedule_path is None:
        rebalance_schedule_path = rebalance_schedule.REBALANCE_SCHEDULE_FILE
    decisions: list[StrategyDecision] = []

    # Pass 1 output: per-family decision fields (minus `orders`, filled in by
    # pass 2) plus the approved legs, grouped by symbol for netting.
    pending: list[dict] = []
    legs_by_symbol: dict[str, list[_Leg]] = {}

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

        # --- 1-day % change per symbol (for the daily status message) ---
        day_changes: dict[str, float] = {}
        for sym in sigs:
            series = prices.get(sym)
            if series is not None and len(series) >= 2 and float(series.iloc[-2]) != 0:
                day_changes[sym] = float(series.iloc[-1] / series.iloc[-2] - 1.0)

        # --- gate this family's per-symbol deltas against its own position
        # book; approved legs are queued for pass 2, not sent yet ---
        # Deltas are taken against the per-sleeve virtual book, never the
        # account-level broker position: universes overlap (SECTORS contains
        # XLK), and a sleeve must not flatten another sleeve's holding.
        decision = StrategyDecision(
            family=family, model=model_name, symbol=symbol, as_of=today,
            signals=sigs, target_shares=target_shares, orders=[],
            weight=weight, day_changes=day_changes,
        )
        decisions.append(decision)
        pending.append({"family": family, "model": model_name, "decision": decision})

        if not dry_run:
            for sym, target in target_shares.items():
                if sym not in prices or prices[sym].empty:
                    continue
                price = float(prices[sym].iloc[-1])
                if hasattr(broker, "set_price"):
                    broker.set_price(sym, price)
                current = get_position(family, sym, path=positions_path)
                delta = target - current
                # Gate on notional, not share count. A share-count band of 0.001
                # is $0.05 of XLRE — two orders of magnitude below the broker's
                # $1 minimum, so orders were submitted only to be rejected with
                # "cost basis must be >= minimal amount of order 1". Skipping
                # here leaves `current` untouched, so the delta simply carries
                # into tomorrow and goes out once it is worth placing.
                if abs(delta) * price < MIN_ORDER_NOTIONAL:
                    continue
                # Adjusting a position already held (not opening or closing
                # one) is throttled to once a calendar month, then still
                # subject to the noise band within that month.
                if current != 0.0 and target != 0.0:
                    if not rebalance_schedule.is_rebalance_due(
                        family, sym, today, path=rebalance_schedule_path
                    ):
                        continue
                    rebalance_schedule.record_rebalance(
                        family, sym, today, path=rebalance_schedule_path
                    )
                    if abs(delta) < REBALANCE_BAND_PCT * abs(target):
                        print(f"  hold: {family} {sym} {delta:+.4f} sh is inside "
                              f"the {REBALANCE_BAND_PCT:.0%} rebalance band")
                        continue
                if abs(delta) > 0.001:
                    legs_by_symbol.setdefault(sym, []).append(
                        _Leg(family=family, model=model_name, symbol=sym,
                             delta=delta, price=price)
                    )

    # --- pass 2: net every symbol's approved legs into one broker order ---
    decision_by_family = {p["family"]: p["decision"] for p in pending}
    for sym, legs in legs_by_symbol.items():
        price = legs[-1].price
        net_qty = round(sum(leg.delta for leg in legs), 6)

        if abs(net_qty) * price < MIN_ORDER_NOTIONAL:
            # Net exposure change isn't worth a broker order at all — legs
            # offset each other (fully or partially) without ever crossing
            # the spread. Settle each leg internally at the last-known price.
            for leg in legs:
                synthetic = Fill(symbol=sym, qty=leg.delta, price=price,
                                  order_id="", filled=True)
                adjust_position(leg.family, sym, leg.delta, path=positions_path)
                decision_by_family[leg.family].orders.append(synthetic)
                if synthetic.qty > 0:
                    notify_entry(synthetic, leg.family, leg.model, last_close=price)
                else:
                    notify_exit(synthetic, leg.family, leg.model, last_close=price)
            continue

        try:
            real_fill = broker.market_order(sym, net_qty)
        except Exception as exc:
            # One rejected net order must not abort the session, and must not
            # apply any of the legs that fed into it.
            print(f"  order failed: {sym} {net_qty:+.4f} (netted) — {exc}")
            continue

        if real_fill is None:
            continue

        for leg in legs:
            fill = Fill(symbol=sym, qty=leg.delta, price=real_fill.price,
                         order_id=real_fill.order_id, filled=real_fill.filled)
            adjust_position(leg.family, sym, leg.delta, path=positions_path)
            decision_by_family[leg.family].orders.append(fill)
            # Real broker orders carry an order_id: log actual fills now,
            # queue unfilled ones (e.g. placed after the close) for the
            # morning reconcile. Simulated fills stay out of the permanent
            # trade log.
            if fill.order_id:
                if fill.filled and fill.price > 0:
                    record_fill(fill, leg.family, leg.model)
                else:
                    add_pending_order(fill, leg.family, leg.model)
            if fill.qty > 0:
                notify_entry(fill, leg.family, leg.model, last_close=price)
            else:
                notify_exit(fill, leg.family, leg.model, last_close=price)

    if not dry_run:
        # BaseBroker declares get_account_equity with a (None, None) default, so
        # this needs no hasattr sniffing — but a live API call can still fail, and
        # a missing balance line must never cost us the daily status message.
        try:
            equity, last_equity = broker.get_account_equity()
        except Exception:
            equity = last_equity = None
        notify_daily_status(decisions, equity=equity, last_equity=last_equity)

        # Persist per-sleeve accounting so sleeve performance is measurable
        # without reconstructing it from the fill log by hand. Recomputed from
        # that log every session, so it is safe to run twice.
        try:
            last_prices = {
                sym: float(series.iloc[-1])
                for sym, series in prices.items() if not series.empty
            }
            update_sleeve_ledgers(equity or account_equity, last_prices)
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a session
            print(f"  sleeve ledger update failed: {exc}")

    return decisions


def print_session_report(decisions: list[StrategyDecision]) -> None:
    """Print a human-readable summary of a paper session to stdout."""
    active = [d for d in decisions if not d.skipped and d.weight > 0]
    skipped = [d for d in decisions if d.skipped]

    print(f"\n{'='*60}")
    print(f"Meridian paper session  {date.today()}")
    print(f"{'='*60}")
    print(f"  Strategies:  {len(active)} active  |  {len(skipped)} skipped")

    # Group active by sleeve (volatility shares experimental_research)
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
