"""Tests for the curated live book: live_picks-driven paper sessions.

These cover the fix that wires ``run_paper_session`` to ``live_picks.json`` (one
pick per family) instead of trading every paper-stage record. Network and the
strategy registry are stubbed so the test is fast and deterministic.
"""

from __future__ import annotations

import pandas as pd

from meridian.execution import live_runner
from meridian.execution.broker import SimulatedBroker
from meridian.portfolio.live_picks import live_pick_weight

# --- weighting rule -------------------------------------------------------

def test_live_pick_weight_primary_and_monitor():
    present = {
        "long_term_etf", "momentum", "trend_following", "mean_reversion",
        "pullback_continuation", "sector_rotation", "breakouts", "volatility",
    }
    # Primary holders get their sleeve's full weight.
    assert live_pick_weight("trend_following", present) == 0.15
    assert live_pick_weight("mean_reversion", present) == 0.15
    assert live_pick_weight("long_term_etf", present) == 0.25
    # breakouts routes to the trend_following sleeve, which is already live → monitor.
    assert live_pick_weight("breakouts", present) == 0.0
    # volatility routes to experimental_research (no competing family) → full 5%.
    assert live_pick_weight("volatility", present) == 0.05


def test_live_pick_weight_breakouts_funded_when_trend_absent():
    # With no trend_following pick present, breakouts becomes the sleeve's holder.
    assert live_pick_weight("breakouts", {"breakouts"}) == 0.15


# --- session driven by live_picks -----------------------------------------

class _FakeModel:
    """Single-asset model that always emits a long signal on the last bar."""

    cross_sectional = False

    def signals(self, prices: pd.Series) -> pd.Series:
        return pd.Series(1, index=prices.index)


def _patch_session(monkeypatch, picks):
    """Stub picks, price fetch, and model registry for an offline session."""
    monkeypatch.setattr(live_runner, "load_live_picks", lambda: picks)

    def fake_fetch(symbols, start="2023-01-01"):
        idx = pd.RangeIndex(300)
        return {s: pd.Series(100.0, index=idx) for s in symbols}

    monkeypatch.setattr(live_runner, "_fetch_prices", fake_fetch)
    monkeypatch.setattr(live_runner, "create_model", lambda family, model: _FakeModel())


def test_run_paper_session_sizes_primary_and_flattens_monitor(monkeypatch):
    picks = {
        "trend_following": {"model": "ma_trend_long_only", "symbol": "XLK"},
        "breakouts":       {"model": "turtle_ma_exit",     "symbol": "TRGP"},  # monitor → 0%
        "mean_reversion":  {"model": "rsi_exhaustion",     "symbol": "SNOW"},
    }
    _patch_session(monkeypatch, picks)
    broker = SimulatedBroker(cash=100_000.0, cost_bps=0.0)

    decisions = live_runner.run_paper_session(broker, account_equity=100_000.0)

    by_family = {d.family: d for d in decisions}
    assert set(by_family) == set(picks)            # exactly one decision per pick

    # Primary holder sizes to full sleeve weight: 100k * 0.15 / 1 long / $100 = 150 sh.
    assert by_family["trend_following"].target_shares["XLK"] == 150.0
    assert by_family["mean_reversion"].target_shares["SNOW"] == 150.0

    # Monitor-only family holds nothing and sends no orders, despite a long signal.
    assert by_family["breakouts"].target_shares["TRGP"] == 0.0
    assert by_family["breakouts"].orders == []

    # No symbol is traded by two families (stomping is structurally impossible).
    traded: dict[str, str] = {}
    for d in decisions:
        for sym, sh in d.target_shares.items():
            if sh != 0:
                assert sym not in traded, f"{sym} traded by {traded.get(sym)} and {d.family}"
                traded[sym] = d.family


def test_run_paper_session_empty_picks_trades_nothing(monkeypatch):
    monkeypatch.setattr(live_runner, "load_live_picks", lambda: {})
    broker = SimulatedBroker(cash=100_000.0)
    assert live_runner.run_paper_session(broker) == []
