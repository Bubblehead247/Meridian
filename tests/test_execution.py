"""Phase 9 tests: broker abstraction and the paper-trading loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.execution import AlpacaBroker, BaseBroker, PaperTrader, SimulatedBroker
from meridian.signals import SignalConfig, compute_scores, generate_positions


def _mean_reverting(n=500, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)
    return pd.Series(100 + x, index=pd.RangeIndex(n))


# --- simulated broker -----------------------------------------------------

def test_simulated_broker_fills_and_accounting():
    b = SimulatedBroker(cash=1000.0, cost_bps=0.0)
    b.set_price("X", 10.0)
    fill = b.market_order("X", 5.0)
    assert fill.qty == 5.0 and fill.price == 10.0
    assert b.get_position("X") == 5.0
    assert b.cash == pytest.approx(950.0)
    assert b.equity() == pytest.approx(1000.0)  # 950 cash + 50 mark-to-market


def test_simulated_broker_zero_order_is_noop():
    b = SimulatedBroker()
    b.set_price("X", 10.0)
    assert b.market_order("X", 0.0) is None
    assert b.get_position("X") == 0.0


def test_simulated_broker_charges_costs():
    b = SimulatedBroker(cash=1000.0, cost_bps=10.0)
    b.set_price("X", 100.0)
    b.market_order("X", 1.0)  # notional 100, cost = 100 * 10/1e4 = 0.1
    assert b.cash == pytest.approx(1000.0 - 100.0 - 0.1)


# --- the key equivalence: live replay == backtest signals -----------------

def test_replay_signals_match_backtest_exactly():
    """A paper-trading replay reproduces the backtester's decisions bar for bar."""
    px = _mean_reverting()
    sig = SignalConfig(entry_threshold=1.0)

    scores = compute_scores(px, "sma", "zscore", window=20)
    batch = generate_positions(scores, sig)

    trader = PaperTrader("SPY", SimulatedBroker(), "sma", "zscore", sig, window=20)
    trader.replay(px)
    live = [d.signal for d in trader.log]

    assert live == list(batch)


def test_broker_position_tracks_target():
    px = _mean_reverting()
    b = SimulatedBroker(cost_bps=0.0)
    trader = PaperTrader("SPY", b, "ou", "zscore", SignalConfig(entry_threshold=1.0),
                        window=20, position_size=10.0)
    trader.replay(px)
    # final broker position equals the last decided target...
    assert b.get_position("SPY") == pytest.approx(trader.log[-1].target_position)
    # ...and every order summed equals the net position (reconciliation is exact)
    total = sum(d.order.qty for d in trader.log if d.order)
    assert total == pytest.approx(b.get_position("SPY"))
    # target is always signal * size when ungated
    for d in trader.log:
        assert d.target_position == pytest.approx(d.signal * 10.0)


def test_orders_only_on_position_change():
    px = _mean_reverting()
    trader = PaperTrader("SPY", SimulatedBroker(), "sma", "zscore",
                        SignalConfig(entry_threshold=1.0), window=20)
    trader.replay(px)
    signals = [d.signal for d in trader.log]
    changes = sum(1 for a, b in zip(signals, signals[1:], strict=False) if a != b)
    if signals[0] != 0:
        changes += 1  # initial entry from flat
    n_orders = sum(1 for d in trader.log if d.order is not None)
    assert n_orders == changes


# --- regime gating --------------------------------------------------------

def test_regime_gate_flattens_disallowed_bars():
    px = _mean_reverting()
    trader = PaperTrader(
        "SPY", SimulatedBroker(), "sma", "zscore", SignalConfig(entry_threshold=1.0),
        window=20, position_size=1.0, regime="trend", allowed_regimes=("range",),
    )
    trader.replay(px)
    for d in trader.log:
        if d.regime is not None and d.regime != "range":
            assert d.target_position == 0.0  # gated flat outside allowed regime


# --- warm-up --------------------------------------------------------------

def test_warm_up_primes_indicators():
    px = _mean_reverting()
    warm, live = px.iloc[:200], px.iloc[200:]

    cold = PaperTrader("SPY", SimulatedBroker(), "sma", "zscore", window=20)
    d_cold = cold.on_bar(float(live.iloc[0]))
    assert np.isnan(d_cold.score)  # nothing seen yet

    warmed = PaperTrader("SPY", SimulatedBroker(), "sma", "zscore", window=20)
    warmed.warm_up(warm)
    d_warm = warmed.on_bar(float(live.iloc[0]))
    assert np.isfinite(d_warm.score)  # indicators already primed


# --- alpaca broker guards -------------------------------------------------

def test_alpaca_broker_requires_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="credentials"):
        AlpacaBroker()


def test_base_broker_is_abstract():
    with pytest.raises(TypeError):
        BaseBroker()
