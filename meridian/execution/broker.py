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
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Fill:
    """A completed order fill."""

    symbol: str
    qty: float       # signed: + buy, - sell
    price: float


class BaseBroker(ABC):
    """Minimal broker interface the trader depends on."""

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Current signed position (shares/units) in ``symbol``."""

    @abstractmethod
    def market_order(self, symbol: str, qty: float) -> Fill | None:
        """Submit a signed market order; return the Fill (or None for qty≈0)."""


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
        price = float(getattr(order, "filled_avg_price", None) or 0.0)
        return Fill(symbol, float(qty), price)
