"""Live/paper trading loop.

`PaperTrader` runs the *identical* causal pipeline used in backtesting — update
the estimator with the new price, take the residual, update the deviation metric,
score it, and step the shared `SignalState` — then reconciles the resulting
target position with the broker by sending a market order for the difference.

Because it reuses `compute_scores`' per-bar steps and the very same
`SignalState` the backtester uses, a live session reproduces the backtest's
decisions bar for bar. `warm_up` replays history so the indicators start live in
the same state a backtest would have at that point (the book itself starts flat).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from meridian.deviations import create as make_deviation
from meridian.estimators import create as make_estimator
from meridian.execution.broker import BaseBroker, Fill
from meridian.regimes import create as make_regime
from meridian.signals import SignalConfig, SignalState


@dataclass
class BarDecision:
    """Record of one bar's processing."""

    time: object
    price: float
    score: float
    signal: int          # raw {-1,0,+1} from the signal state machine
    target_position: float  # signal * size, after any regime gate
    order: Fill | None
    regime: str | None = None


@dataclass
class PaperTrader:
    """Drive one symbol's mean-reversion strategy against a broker.

    Args:
        symbol: Ticker to trade.
        broker: A `BaseBroker` (simulated or Alpaca).
        estimator: Estimator name (or instance).
        deviation: Deviation-metric name (or instance).
        signal: Signal parameters (defaults if None).
        window: Lookback for estimator/deviation/regime when given by name.
        position_size: Shares/units per unit of signal (±size or 0).
        regime: Optional regime classifier name to gate trading.
        allowed_regimes: Labels in which trading is permitted (when regime set).
    """

    symbol: str
    broker: BaseBroker
    estimator: object = "ou"
    deviation: object = "zscore"
    signal: SignalConfig | None = None
    window: int = 20
    position_size: float = 1.0
    regime: str | None = None
    allowed_regimes: tuple[str, ...] = ()
    log: list[BarDecision] = field(default_factory=list)

    def __post_init__(self):
        self._est = (
            make_estimator(self.estimator, window=self.window)
            if isinstance(self.estimator, str) else self.estimator
        )
        self._dev = (
            make_deviation(self.deviation, window=self.window)
            if isinstance(self.deviation, str) else self.deviation
        )
        self._regime = make_regime(self.regime, window=self.window) if self.regime else None
        self._state = SignalState(self.signal or SignalConfig())

    # --- indicator step (shared with the backtest path) -------------------

    def _score_bar(self, price: float, bar: Mapping | None):
        """Causal update of estimator + deviation (+ regime); return (score, label)."""
        self._est.update(float(price))
        resid = self._est.residual(float(price))
        self._dev.update(resid, bar=bar)
        score = self._dev.value(resid)
        label = None
        if self._regime is not None:
            self._regime.update(float(price), bar=bar)
            label = self._regime.label()
        return score, label

    def warm_up(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> None:
        """Replay history through the indicators so they start live warmed.

        Advances only the estimator/deviation/regime — the signal state machine
        and the book start flat (you go live flat, not mid-trade).
        """
        for i, p in enumerate(prices.to_numpy(dtype=float)):
            self._score_bar(p, None if bars is None else bars.iloc[i])

    # --- live step --------------------------------------------------------

    def on_bar(self, price: float, bar: Mapping | None = None, time=None) -> BarDecision:
        """Process one new bar: score, decide target, reconcile with broker."""
        score, label = self._score_bar(price, bar)
        signal = self._state.step(score)

        gated = signal
        if self._regime is not None and label not in set(self.allowed_regimes):
            gated = 0  # regime gate: flatten when not in an allowed regime

        target = gated * self.position_size

        if hasattr(self.broker, "set_price"):
            self.broker.set_price(self.symbol, float(price))
        current = self.broker.get_position(self.symbol)
        delta = target - current
        order = self.broker.market_order(self.symbol, delta) if abs(delta) > 1e-12 else None

        decision = BarDecision(
            time=time, price=float(price), score=float(score),
            signal=int(signal), target_position=float(target), order=order, regime=label,
        )
        self.log.append(decision)
        return decision

    def replay(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.DataFrame:
        """Feed a whole price series bar by bar (offline dry-run); return the log.

        With a `SimulatedBroker` this is a full paper-trading simulation whose
        per-bar signals match a backtest of the same estimator/deviation/signal.
        """
        for i, p in enumerate(prices.to_numpy(dtype=float)):
            self.on_bar(
                float(p),
                bar=None if bars is None else bars.iloc[i],
                time=prices.index[i],
            )
        return self.log_frame()

    # --- state persistence ------------------------------------------------

    def save_checkpoint(self, path) -> None:
        """Persist the path-dependent signal state to JSON.

        Only the signal state machine is saved — the estimator/deviation/regime
        are re-derived by `warm_up` on the same history (reproducible from data),
        and live positions live at the broker. This is the minimal state needed
        to resume a session without double-counting trades.
        """
        import json
        from pathlib import Path

        payload = {"symbol": self.symbol, "signal_state": self._state.to_dict()}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_checkpoint(self, path) -> None:
        """Restore signal state from a JSON checkpoint (after `warm_up`)."""
        import json
        from pathlib import Path

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._state.load_dict(payload["signal_state"])

    def log_frame(self) -> pd.DataFrame:
        """The decision log as a DataFrame."""
        return pd.DataFrame(
            {
                "time": [d.time for d in self.log],
                "price": [d.price for d in self.log],
                "score": [d.score for d in self.log],
                "signal": [d.signal for d in self.log],
                "target_position": [d.target_position for d in self.log],
                "regime": [d.regime for d in self.log],
                "order_qty": [d.order.qty if d.order else 0.0 for d in self.log],
            }
        )
