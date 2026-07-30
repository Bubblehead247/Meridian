"""Broker abstraction for paper/live execution.

One interface, two implementations:

- `SimulatedBroker` — an in-memory paper broker used for tests and offline
  dry-runs. Deterministic: fills market orders at the last price it was told,
  applies flat-bps costs, and tracks cash and positions.
- `AlpacaBroker` — a thin wrapper over alpaca-py's paper/live trading API. Lazy
  imported and constructed from API keys (or env vars); never called in tests.

The trader (`meridian.execution.trader`) talks only to this interface, so the
same strategy code runs against a simulation or a real paper account.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Fill:
    """A completed order fill — or a submitted order awaiting its fill.

    Orders sent after market close (the 16:30 ET session) don't fill until the
    next open, so ``filled`` is False and ``price`` is 0.0 until the morning
    reconcile job (`meridian reconcile`) fetches the real fill price.
    """

    symbol: str
    qty: float           # signed: + buy, - sell
    price: float         # actual fill price; 0.0 while not yet filled
    order_id: str = ""   # broker order id ("" for the simulated broker)
    filled: bool = True  # False = accepted by broker but not yet executed


class BaseBroker(ABC):
    """Minimal broker interface the trader depends on."""

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Current signed position (shares/units) in ``symbol``."""

    @abstractmethod
    def market_order(self, symbol: str, qty: float) -> Fill | None:
        """Submit a signed market order; return the Fill (or None for qty≈0)."""

    def get_account_equity(self) -> tuple[float | None, float | None]:
        """Return ``(current equity, prior-close equity)``.

        Not abstract: a broker that cannot report account value returns
        ``(None, None)`` and callers simply omit the balance. Declared here so
        callers can call it directly instead of sniffing for it with ``hasattr``.
        """
        return None, None


class SimulatedBroker(BaseBroker):
    """Deterministic in-memory paper broker.

    Call ``set_price`` before ``market_order`` so the fill has a price (the
    trader does this each bar from the incoming price).
    """

    def __init__(self, cash: float = 100_000.0, cost_bps: float = 1.0):
        self.cash = float(cash)
        self.cost_bps = float(cost_bps)
        self.positions: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.fills: list[Fill] = []
        self._initial_cash = float(cash)

    def set_price(self, symbol: str, price: float) -> None:
        self.last_price[symbol] = float(price)

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def market_order(self, symbol: str, qty: float) -> Fill | None:
        if abs(qty) < 1e-12:
            return None
        price = self.last_price[symbol]
        cost = abs(qty) * price * self.cost_bps / 1e4
        self.cash -= qty * price + cost
        self.positions[symbol] = self.get_position(symbol) + qty
        fill = Fill(symbol, float(qty), float(price))
        self.fills.append(fill)
        return fill

    def equity(self) -> float:
        """Mark-to-market account equity at the last seen prices."""
        mtm = sum(q * self.last_price.get(s, 0.0) for s, q in self.positions.items())
        return self.cash + mtm

    def get_account_equity(self) -> tuple[float, float]:
        """Return (current equity, starting equity).

        The simulated broker has no real "prior trading day close" to compare
        against, so the starting cash it was constructed with stands in.
        """
        return self.equity(), self._initial_cash


class AlpacaBroker(BaseBroker):
    """alpaca-py-backed broker for paper or live trading.

    Credentials come from the constructor or the ``ALPACA_API_KEY`` /
    ``ALPACA_SECRET_KEY`` environment variables. Defaults to the **paper**
    endpoint. alpaca-py is imported lazily so the rest of the platform does not
    depend on it.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        *,
        paper: bool = True,
    ):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "alpaca-py is required for AlpacaBroker. Install with `pip install alpaca-py`."
            ) from exc

        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise ValueError(
                "Alpaca credentials missing: pass api_key/secret_key or set "
                "ALPACA_API_KEY / ALPACA_SECRET_KEY."
            )
        self.paper = paper
        self._client = TradingClient(key, secret, paper=paper)

    def get_position(self, symbol: str) -> float:  # pragma: no cover - needs live API
        try:
            pos = self._client.get_open_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0  # no open position

    def market_order(self, symbol: str, qty: float) -> Fill | None:  # pragma: no cover
        if abs(qty) < 1e-12:
            return None
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY
        )
        order = self._client.submit_order(req)
        order_id = str(order.id)

        # During market hours a market order fills in well under a second; poll
        # briefly for the real price. After hours it stays queued until the next
        # open — return a pending Fill and let `meridian reconcile` pick it up.
        for _ in range(3):
            status, price = self.order_status(order_id)
            if status == "filled" and price > 0:
                return Fill(symbol, float(qty), price, order_id=order_id, filled=True)
            if status in ("canceled", "expired", "rejected"):
                break
            time.sleep(1)
        return Fill(symbol, float(qty), 0.0, order_id=order_id, filled=False)

    def order_status(self, order_id: str) -> tuple[str, float]:  # pragma: no cover
        """Return (status, filled_avg_price) for an order; ("unknown", 0.0) on error."""
        try:
            order = self._client.get_order_by_id(order_id)
            status = str(getattr(order.status, "value", order.status)).lower()
            price = float(getattr(order, "filled_avg_price", None) or 0.0)
            return status, price
        except Exception:
            return "unknown", 0.0

    def get_account_equity(self) -> tuple[float, float]:  # pragma: no cover - needs live API
        """Return (current equity, prior trading-day closing equity)."""
        account = self._client.get_account()
        return float(account.equity), float(account.last_equity)
