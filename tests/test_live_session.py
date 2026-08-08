"""Tests for the curated live book: live_picks-driven paper sessions.

These cover the fix that wires ``run_paper_session`` to ``live_picks.json`` (one
pick per family) instead of trading every paper-stage record. Network and the
strategy registry are stubbed so the test is fast and deterministic.
"""

from __future__ import annotations

import pandas as pd
import pytest

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


def test_run_paper_session_sizes_primary_and_flattens_monitor(monkeypatch, tmp_path):
    picks = {
        "trend_following": {"model": "ma_trend_long_only", "symbol": "XLK"},
        "breakouts":       {"model": "turtle_ma_exit",     "symbol": "TRGP"},  # monitor → 0%
        "mean_reversion":  {"model": "rsi_exhaustion",     "symbol": "SNOW"},
    }
    _patch_session(monkeypatch, picks)
    broker = SimulatedBroker(cash=100_000.0, cost_bps=0.0)

    decisions = live_runner.run_paper_session(
        broker, account_equity=100_000.0,
        positions_path=tmp_path / "positions.json")

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


# --- per-sleeve position book (no cross-sleeve stomping) --------------------


class _FlatModel:
    """Single-asset model that always signals flat."""

    cross_sectional = False

    def signals(self, prices: pd.Series) -> pd.Series:
        return pd.Series(0, index=prices.index)


def test_sleeve_does_not_flatten_another_sleeves_holding(monkeypatch, tmp_path):
    """A flat signal sells only what THIS sleeve holds — not account shares.

    Regression: sector_rotation (SECTORS ⊇ XLK) used to sell trend_following's
    XLK because deltas were taken against the account-level broker position.
    """
    import json
    picks = {"sector_rotation": {"model": "rs", "symbol": "XLK"}}
    _patch_session(monkeypatch, picks)
    monkeypatch.setattr(live_runner, "create_model", lambda f, m: _FlatModel())

    # Account holds 8.19 XLK — but it belongs to trend_following's book.
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(json.dumps({"trend_following": {"XLK": 8.19}}))
    broker = SimulatedBroker(cash=100_000.0, cost_bps=0.0)
    broker.positions["XLK"] = 8.19

    decisions = live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)

    assert decisions[0].orders == []          # no sell of someone else's shares
    assert broker.positions["XLK"] == 8.19    # holding untouched


def test_sleeve_flattens_its_own_holding(monkeypatch, tmp_path):
    import json
    picks = {"sector_rotation": {"model": "rs", "symbol": "XLK"}}
    _patch_session(monkeypatch, picks)
    monkeypatch.setattr(live_runner, "create_model", lambda f, m: _FlatModel())

    positions_path = tmp_path / "positions.json"
    positions_path.write_text(json.dumps({"sector_rotation": {"XLK": 5.0}}))
    broker = SimulatedBroker(cash=100_000.0, cost_bps=0.0)
    broker.positions["XLK"] = 5.0

    decisions = live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)

    assert len(decisions[0].orders) == 1
    assert decisions[0].orders[0].qty == -5.0
    # Sleeve book is emptied after the sell.
    assert json.loads(positions_path.read_text()).get("sector_rotation", {}) == {}


def test_order_updates_sleeve_position_book(monkeypatch, tmp_path):
    import json
    picks = {"trend_following": {"model": "ma", "symbol": "XLK"}}
    _patch_session(monkeypatch, picks)

    positions_path = tmp_path / "positions.json"
    broker = SimulatedBroker(cash=100_000.0, cost_bps=0.0)

    live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)

    book = json.loads(positions_path.read_text())
    # 100k * 15% sleeve / $100 = 150 shares recorded to trend_following's book.
    assert book["trend_following"]["XLK"] == 150.0


# --- order rejection is non-fatal -------------------------------------------


class _RejectingBroker(SimulatedBroker):
    """Broker whose first market_order raises (like an Alpaca APIError)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempts = 0

    def market_order(self, symbol, qty):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("insufficient qty available for order")
        return super().market_order(symbol, qty)


def test_rejected_order_does_not_abort_session(monkeypatch, tmp_path):
    import json
    picks = {
        "mean_reversion":  {"model": "z",  "symbol": "SNOW"},
        "trend_following": {"model": "ma", "symbol": "XLK"},
    }
    _patch_session(monkeypatch, picks)
    broker = _RejectingBroker(cash=100_000.0, cost_bps=0.0)
    positions_path = tmp_path / "positions.json"

    decisions = live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)

    # Both families processed; first order raised, second went through.
    assert len(decisions) == 2
    assert broker.attempts == 2
    all_fills = [f for d in decisions for f in d.orders]
    assert len(all_fills) == 1
    # The failed order must NOT be recorded in the position book.
    book = json.loads(positions_path.read_text())
    assert sum(len(v) for v in book.values()) == 1


# --- D4: rebalance band (XLK churn) -----------------------------------------


class _BandBroker(SimulatedBroker):
    """Records every order so a test can assert what was actually sent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent: list[tuple[str, float]] = []

    def market_order(self, symbol, shares, **kwargs):
        self.sent.append((symbol, shares))
        return super().market_order(symbol, shares, **kwargs)


def _run_with_holding(monkeypatch, tmp_path, held: float, equity: float):
    """Run one session for a single family already holding `held` shares."""
    from meridian.execution.positions import adjust_position

    picks = {"trend_following": {"model": "ma_trend_long_only", "symbol": "XLK"}}
    _patch_session(monkeypatch, picks)
    positions = tmp_path / "positions.json"
    if held:
        adjust_position("trend_following", "XLK", held, path=positions)

    broker = _BandBroker(cash=equity, cost_bps=0.0)
    live_runner.run_paper_session(
        broker, account_equity=equity, positions_path=positions)
    return broker.sent


def test_a_small_drift_inside_the_band_sends_no_order(monkeypatch, tmp_path):
    """XLK ratcheted 8.43 → 9.005 shares over six sessions of drift like this."""
    # Sleeve target is 100_000 * 0.15 / $100 = 150 shares. Holding 148 leaves a
    # 2-share delta = 1.3% of target, well inside the 5% band.
    sent = _run_with_holding(monkeypatch, tmp_path, held=148.0, equity=100_000.0)

    assert sent == []


def test_a_drift_outside_the_band_still_trades(monkeypatch, tmp_path):
    """The band suppresses noise, not real divergence."""
    # Holding 100 against a 150 target is a 50-share delta = 33% of target.
    sent = _run_with_holding(monkeypatch, tmp_path, held=100.0, equity=100_000.0)

    assert [s for s, _ in sent] == ["XLK"]
    assert sent[0][1] == 50.0


def test_opening_a_position_ignores_the_band(monkeypatch, tmp_path):
    """current == 0 is a decision to enter, not drift."""
    sent = _run_with_holding(monkeypatch, tmp_path, held=0.0, equity=100_000.0)

    assert [s for s, _ in sent] == ["XLK"]


# --- order netting across families (same symbol) ----------------------------
#
# 2026-08-07: sector_rotation and trend_following each bought XLK the same
# day as two separate broker orders on several sessions — two spreads paid
# for one net account-level exposure change. These cover the fix: every
# family's approved order is collected first, then netted per symbol before
# a single broker.market_order call (or none, if the net washes out).


def _model_for(models_by_family: dict[str, object]):
    """create_model stub that dispatches on family, ignoring model name."""
    def _create(family, model):
        return models_by_family[family]
    return _create


def test_two_families_same_direction_send_one_summed_order(monkeypatch, tmp_path):
    picks = {
        "trend_following": {"model": "ma", "symbol": "XLK"},
        "mean_reversion":  {"model": "z",  "symbol": "XLK"},
    }
    _patch_session(monkeypatch, picks)
    monkeypatch.setattr(
        live_runner, "create_model",
        _model_for({"trend_following": _FakeModel(), "mean_reversion": _FakeModel()}),
    )
    broker = _BandBroker(cash=100_000.0, cost_bps=0.0)
    positions_path = tmp_path / "positions.json"

    decisions = live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)

    # Both families open a 150-share position (100k * 15% / $100) — one
    # broker call for the summed 300 shares, not two of 150 each.
    assert broker.sent == [("XLK", 300.0)]

    import json
    book = json.loads(positions_path.read_text())
    assert book["trend_following"]["XLK"] == 150.0
    assert book["mean_reversion"]["XLK"] == 150.0


def test_two_families_opposite_direction_full_offset_sends_no_order(monkeypatch, tmp_path):
    import json
    picks = {
        "trend_following":       {"model": "ma", "symbol": "XLK"},
        "pullback_continuation": {"model": "z",  "symbol": "XLK"},
    }
    _patch_session(monkeypatch, picks)
    monkeypatch.setattr(
        live_runner, "create_model",
        _model_for({"trend_following": _FakeModel(), "pullback_continuation": _FakeModel()}),
    )
    positions_path = tmp_path / "positions.json"
    # trend_following (15% sleeve) already holds 250; its target is 150 →
    # delta -100. pullback_continuation (10% sleeve) holds nothing; its
    # target is 100 → delta +100. Net = 0.
    positions_path.write_text(json.dumps({"trend_following": {"XLK": 250.0}}))
    broker = _BandBroker(cash=100_000.0, cost_bps=0.0)

    decisions = live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)

    assert broker.sent == []  # net exposure change is zero — no spread paid

    # Each family's own book still moves by its own requested delta.
    book = json.loads(positions_path.read_text())
    assert book["trend_following"]["XLK"] == 150.0
    assert book["pullback_continuation"]["XLK"] == 100.0

    by_family = {d.family: d for d in decisions}
    assert len(by_family["trend_following"].orders) == 1
    assert by_family["trend_following"].orders[0].qty == -100.0
    assert len(by_family["pullback_continuation"].orders) == 1
    assert by_family["pullback_continuation"].orders[0].qty == 100.0


def test_two_families_opposite_direction_partial_offset_sends_residual_order(monkeypatch, tmp_path):
    import json
    picks = {
        "trend_following": {"model": "ma", "symbol": "XLK"},
        "mean_reversion":  {"model": "z",  "symbol": "XLK"},
    }
    _patch_session(monkeypatch, picks)
    monkeypatch.setattr(
        live_runner, "create_model",
        _model_for({"trend_following": _FakeModel(), "mean_reversion": _FakeModel()}),
    )
    positions_path = tmp_path / "positions.json"
    # trend_following holds 200 against a 150 target → delta -50.
    # mean_reversion holds nothing against a 150 target → delta +150. Net = +100.
    positions_path.write_text(json.dumps({"trend_following": {"XLK": 200.0}}))
    broker = _BandBroker(cash=100_000.0, cost_bps=0.0)

    live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)

    assert broker.sent == [("XLK", 100.0)]

    book = json.loads(positions_path.read_text())
    assert book["trend_following"]["XLK"] == 150.0
    assert book["mean_reversion"]["XLK"] == 150.0


def test_net_order_failure_applies_no_leg(monkeypatch, tmp_path):
    import json
    picks = {
        "trend_following": {"model": "ma", "symbol": "XLK"},
        "mean_reversion":  {"model": "z",  "symbol": "XLK"},
    }
    _patch_session(monkeypatch, picks)
    monkeypatch.setattr(
        live_runner, "create_model",
        _model_for({"trend_following": _FakeModel(), "mean_reversion": _FakeModel()}),
    )
    positions_path = tmp_path / "positions.json"

    class _ExplodingBroker(SimulatedBroker):
        def market_order(self, symbol, qty):
            raise RuntimeError("insufficient buying power")

    broker = _ExplodingBroker(cash=100_000.0, cost_bps=0.0)

    decisions = live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)

    # Neither leg's position book updates — the batch failed as a unit.
    book = json.loads(positions_path.read_text()) if positions_path.exists() else {}
    assert sum(len(v) for v in book.values()) == 0

    by_family = {d.family: d for d in decisions}
    assert by_family["trend_following"].orders == []
    assert by_family["mean_reversion"].orders == []


def test_leg_clears_notional_alone_but_net_does_not_settles_internally(monkeypatch, tmp_path):
    """Two legs, each individually above MIN_ORDER_NOTIONAL, net below it.

    No broker order is sent; both legs still settle against their own
    sleeve's position book at the last-known price.
    """
    import json
    picks = {
        "trend_following": {"model": "ma", "symbol": "XLK"},  # opens: sig=1
        "volatility":      {"model": "v",  "symbol": "XLK"},  # closes: sig=0
    }
    _patch_session(monkeypatch, picks)
    monkeypatch.setattr(
        live_runner, "create_model",
        _model_for({"trend_following": _FakeModel(), "volatility": _FlatModel()}),
    )
    positions_path = tmp_path / "positions.json"
    # volatility already holds 0.012 sh; its target is 0 (flat signal) →
    # delta -0.012, notional $1.20 — clears MIN_ORDER_NOTIONAL alone.
    positions_path.write_text(json.dumps({"volatility": {"XLK": 0.012}}))
    broker = _BandBroker(cash=100_000.0, cost_bps=0.0)

    # trend_following: equity 10 * weight 0.15 / $100 price = 0.015 sh target,
    # opening from 0 → delta +0.015, notional $1.50 — also clears alone.
    # Net = 0.015 - 0.012 = 0.003 sh → $0.30, below the $1 floor.
    decisions = live_runner.run_paper_session(
        broker, account_equity=10.0, positions_path=positions_path)

    assert broker.sent == []  # net didn't clear the floor — no broker order

    book = json.loads(positions_path.read_text())
    assert book["trend_following"]["XLK"] == pytest.approx(0.015)
    # volatility's position nets to ~0 and is dropped from the book entirely
    # (set_position prunes empty entries), same as any other full close.
    assert book.get("volatility", {}).get("XLK", 0.0) == pytest.approx(0.0)

    by_family = {d.family: d for d in decisions}
    assert len(by_family["trend_following"].orders) == 1
    assert by_family["trend_following"].orders[0].qty == pytest.approx(0.015)
    assert by_family["trend_following"].orders[0].order_id == ""
    assert len(by_family["volatility"].orders) == 1
    assert by_family["volatility"].orders[0].qty == pytest.approx(-0.012)
