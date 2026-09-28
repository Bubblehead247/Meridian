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
from meridian.portfolio.allocation import SLEEVE_ALLOCATIONS
from meridian.portfolio.live_picks import live_pick_weight

# --- weighting rule -------------------------------------------------------

def test_live_pick_weight_primary_and_monitor():
    present = {
        "long_term_etf", "momentum", "trend_following", "mean_reversion",
        "pullback_continuation", "sector_rotation", "breakouts", "volatility",
    }
    # Primary holders get their sleeve's full weight.
    assert live_pick_weight("mean_reversion", present) == 0.15
    assert live_pick_weight("long_term_etf", present) == 0.25
    # trend_following and breakouts each have their own sleeve now (split 50/50
    # out of the old combined 15% "trend-following breakout" sleeve).
    assert live_pick_weight("trend_following", present) == 0.075
    assert live_pick_weight("breakouts", present) == 0.075
    # volatility routes to experimental_research (no competing family) → full 5%.
    assert live_pick_weight("volatility", present) == 0.05


def test_live_pick_weight_monitor_mechanism(monkeypatch):
    """A family whose mapped sleeve is another *live* family's own name is monitor-only.

    breakouts no longer exercises this (it has its own sleeve), but the
    mechanism itself — for whenever two families are made to share a sleeve —
    still needs coverage.
    """
    import meridian.portfolio.live_picks as live_picks

    monkeypatch.setattr(live_picks, "FAMILY_TO_SLEEVE", {"shadow": "trend_following"})
    assert live_pick_weight("shadow", {"shadow", "trend_following"}) == 0.0
    # With no trend_following pick present, "shadow" becomes the sleeve's holder.
    assert live_pick_weight("shadow", {"shadow"}) == SLEEVE_ALLOCATIONS["trend_following"]


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


def test_run_paper_session_sizes_each_familys_own_sleeve(monkeypatch, tmp_path):
    picks = {
        "trend_following": {"model": "ma_trend_long_only", "symbol": "XLK"},
        "breakouts":       {"model": "turtle_ma_exit",     "symbol": "TRGP"},
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
    assert by_family["mean_reversion"].target_shares["SNOW"] == 150.0

    # trend_following and breakouts now each hold their own real 7.5% sleeve:
    # 100k * 0.075 / 1 long / $100 = 75 sh, not monitor-only.
    assert by_family["trend_following"].target_shares["XLK"] == 75.0
    assert by_family["breakouts"].target_shares["TRGP"] == 75.0

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


class _StrictBroker(SimulatedBroker):
    """Rejects a sell larger than the holding, as Alpaca does."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent: list[tuple[str, float]] = []

    def market_order(self, symbol, shares, **kwargs):
        if shares < 0 and -shares > self.positions.get(symbol, 0.0):
            raise RuntimeError("insufficient qty available for order")
        self.sent.append((symbol, shares))
        return super().market_order(symbol, shares, **kwargs)


def _exit_against_broker_holding(monkeypatch, tmp_path, booked, held):
    import json
    _patch_session(monkeypatch, {"momentum": {"model": "m", "symbol": "AVGO"}})
    monkeypatch.setattr(live_runner, "create_model", lambda f, m: _FlatModel())
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(json.dumps({"momentum": {"AVGO": booked}}))
    broker = _StrictBroker(cash=100_000.0, cost_bps=0.0)
    broker.positions["AVGO"] = held
    live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)
    return broker, json.loads(positions_path.read_text())


def test_an_exit_is_not_rejected_over_a_rounding_difference(monkeypatch, tmp_path):
    """AVGO: booked 0.051939, Alpaca held 0.051938999 — the sell was rejected
    for 11 straight sessions from 2026-09-11."""
    broker, book = _exit_against_broker_holding(
        monkeypatch, tmp_path, booked=0.051939, held=0.051938999)
    assert broker.sent == [("AVGO", -0.051938999)]
    assert broker.positions["AVGO"] == 0.0
    assert book.get("momentum", {}).get("AVGO", 0.0) == 0.0


def test_a_real_shortfall_is_not_papered_over(monkeypatch, tmp_path):
    """LLY: booked 0.251332, Alpaca held 0.244511 — that is drift, not rounding."""
    broker, book = _exit_against_broker_holding(
        monkeypatch, tmp_path, booked=0.251332, held=0.244511)
    assert broker.positions["AVGO"] == 0.244511    # rejected, nothing sold
    assert book["momentum"]["AVGO"] == 0.251332    # book untouched


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
    # 100k * 7.5% sleeve / $100 = 75 shares recorded to trend_following's book.
    assert book["trend_following"]["XLK"] == 75.0


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


def _run_with_holding(monkeypatch, tmp_path, held: float, equity: float,
                       positions=None, rebalance_schedule_path=None):
    """Run one session for a single family already holding `held` shares."""
    from meridian.execution.positions import adjust_position

    picks = {"trend_following": {"model": "ma_trend_long_only", "symbol": "XLK"}}
    _patch_session(monkeypatch, picks)
    if positions is None:
        positions = tmp_path / "positions.json"
        if held:
            adjust_position("trend_following", "XLK", held, path=positions)

    broker = _BandBroker(cash=equity, cost_bps=0.0)
    live_runner.run_paper_session(
        broker, account_equity=equity, positions_path=positions,
        rebalance_schedule_path=rebalance_schedule_path)
    return broker.sent


def test_a_small_drift_inside_the_band_sends_no_order(monkeypatch, tmp_path):
    """XLK ratcheted 8.43 → 9.005 shares over six sessions of drift like this."""
    # Sleeve target is 100_000 * 0.075 / $100 = 75 shares. Holding 74 leaves a
    # 1-share delta = 1.3% of target, well inside the 5% band.
    sent = _run_with_holding(monkeypatch, tmp_path, held=74.0, equity=100_000.0)

    assert sent == []


def test_a_drift_outside_the_band_still_trades(monkeypatch, tmp_path):
    """The band suppresses noise, not real divergence."""
    # Holding 50 against a 75 target is a 25-share delta = 33% of target.
    sent = _run_with_holding(monkeypatch, tmp_path, held=50.0, equity=100_000.0)

    assert [s for s, _ in sent] == ["XLK"]
    assert sent[0][1] == 25.0


def test_opening_a_position_ignores_the_band(monkeypatch, tmp_path):
    """current == 0 is a decision to enter, not drift."""
    sent = _run_with_holding(monkeypatch, tmp_path, held=0.0, equity=100_000.0)

    assert [s for s, _ in sent] == ["XLK"]


# --- monthly resize throttle -------------------------------------------------


def test_second_resize_same_month_is_suppressed_even_outside_the_band(monkeypatch, tmp_path):
    """Resizing an already-open position is throttled to once a calendar month.

    Unlike the band (which only ever suppresses noise), this suppresses a
    real divergence too if the position was already reconsidered this month —
    that's the point of "monthly, not daily" resizing.
    """
    from meridian.execution.positions import adjust_position

    positions = tmp_path / "positions.json"
    schedule = tmp_path / "last_rebalance.json"
    adjust_position("trend_following", "XLK", 50.0, path=positions)  # 33% short of 75

    first = _run_with_holding(
        monkeypatch, tmp_path, held=None, equity=100_000.0,
        positions=positions, rebalance_schedule_path=schedule)
    assert [s for s, _ in first] == ["XLK"], "first check this month: real divergence trades"

    # Still 25 shares short of the 75 target (the first session's own order
    # isn't reflected here since _BandBroker doesn't update `positions` on
    # fill — mirrors the other band tests' pattern of one session per call).
    second = _run_with_holding(
        monkeypatch, tmp_path, held=None, equity=100_000.0,
        positions=positions, rebalance_schedule_path=schedule)
    assert second == [], "already reconsidered this month — waits for next month regardless of size"


def test_resize_due_again_after_a_month_has_passed(monkeypatch, tmp_path):
    from datetime import date, timedelta

    from meridian.execution.positions import adjust_position
    from meridian.execution.rebalance_schedule import record_rebalance

    positions = tmp_path / "positions.json"
    schedule = tmp_path / "last_rebalance.json"
    adjust_position("trend_following", "XLK", 50.0, path=positions)

    # Pretend this position was last resized over a month ago.
    last_month = date.today() - timedelta(days=35)
    record_rebalance("trend_following", "XLK", last_month, path=schedule)

    sent = _run_with_holding(
        monkeypatch, tmp_path, held=None, equity=100_000.0,
        positions=positions, rebalance_schedule_path=schedule)
    assert [s for s, _ in sent] == ["XLK"], "a month has passed — due for reconsideration again"


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

    # trend_following opens 75 (100k * 7.5% / $100), mean_reversion opens 150
    # (100k * 15% / $100) — one broker call for the summed 225 shares, not two.
    assert broker.sent == [("XLK", 225.0)]

    import json
    book = json.loads(positions_path.read_text())
    assert book["trend_following"]["XLK"] == 75.0
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
    # trend_following (7.5% sleeve) already holds 175; its target is 75 →
    # delta -100. pullback_continuation (10% sleeve) holds nothing; its
    # target is 100 → delta +100. Net = 0.
    positions_path.write_text(json.dumps({"trend_following": {"XLK": 175.0}}))
    broker = _BandBroker(cash=100_000.0, cost_bps=0.0)

    decisions = live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)

    assert broker.sent == []  # net exposure change is zero — no spread paid

    # Each family's own book still moves by its own requested delta.
    book = json.loads(positions_path.read_text())
    assert book["trend_following"]["XLK"] == 75.0
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
    # trend_following holds 125 against a 75 target → delta -50.
    # mean_reversion holds nothing against a 150 target → delta +150. Net = +100.
    positions_path.write_text(json.dumps({"trend_following": {"XLK": 125.0}}))
    broker = _BandBroker(cash=100_000.0, cost_bps=0.0)

    live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path)

    assert broker.sent == [("XLK", 100.0)]

    book = json.loads(positions_path.read_text())
    assert book["trend_following"]["XLK"] == 75.0
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

    # trend_following: equity 20 * weight 0.075 / $100 price = 0.015 sh target,
    # opening from 0 → delta +0.015, notional $1.50 — also clears alone.
    # Net = 0.015 - 0.012 = 0.003 sh → $0.30, below the $1 floor.
    decisions = live_runner.run_paper_session(
        broker, account_equity=20.0, positions_path=positions_path)

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


# --- idle-cash sweep (option B, 2026-09-27) ---------------------------------
#
# The single-stock sleeves sit flat most days; while flat their capital is
# held in SGOV, and the SGOV is sold the day the sleeve's signal goes long.


def _sweep_session(monkeypatch, tmp_path, family, model_cls, book=None):
    import json
    _patch_session(monkeypatch, {family: {"model": "m", "symbol": "SNOW"}})
    monkeypatch.setattr(live_runner, "create_model", lambda f, m: model_cls())
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(json.dumps(book or {}))
    broker = SimulatedBroker(cash=100_000.0, cost_bps=0.0)
    for sym, qty in (book or {}).get(family, {}).items():
        broker.positions[sym] = qty
    decisions = live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path,
        rebalance_schedule_path=tmp_path / "last_rebalance.json")
    return decisions[0], json.loads(positions_path.read_text())


def test_a_flat_single_stock_sleeve_parks_its_capital_in_sgov(monkeypatch, tmp_path):
    decision, book = _sweep_session(monkeypatch, tmp_path, "mean_reversion", _FlatModel)
    # 15% of $100k at the stubbed $100 price.
    assert decision.target_shares["SGOV"] == pytest.approx(150.0)
    assert book["mean_reversion"]["SGOV"] == pytest.approx(150.0)
    assert book["mean_reversion"].get("SNOW", 0.0) == 0.0


def test_the_sgov_is_sold_the_day_the_sleeve_goes_long(monkeypatch, tmp_path):
    decision, book = _sweep_session(
        monkeypatch, tmp_path, "mean_reversion", _FakeModel,
        book={"mean_reversion": {"SGOV": 150.0}})
    assert decision.target_shares["SGOV"] == 0.0
    assert book["mean_reversion"].get("SGOV", 0.0) == 0.0
    assert book["mean_reversion"]["SNOW"] == pytest.approx(150.0)


def test_sleeves_outside_the_sweep_hold_no_sgov(monkeypatch, tmp_path):
    decision, book = _sweep_session(monkeypatch, tmp_path, "trend_following", _FlatModel)
    assert "SGOV" not in decision.target_shares
    assert "SGOV" not in book.get("trend_following", {})


# --- retired sleeves (owner decision 2026-09-27) -----------------------------


def _retired_session(monkeypatch, tmp_path, book=None):
    import json
    picks = {"mean_reversion": {"model": "m", "symbol": "SNOW", "retired": True}}
    _patch_session(monkeypatch, picks)
    # A retired sleeve must not consult its model, even one that would buy.
    monkeypatch.setattr(live_runner, "create_model", lambda f, m: _FakeModel())
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(json.dumps(book or {}))
    broker = SimulatedBroker(cash=100_000.0, cost_bps=0.0)
    for sym, qty in (book or {}).get("mean_reversion", {}).items():
        broker.positions[sym] = qty
    decisions = live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=positions_path,
        rebalance_schedule_path=tmp_path / "last_rebalance.json")
    return decisions[0], json.loads(positions_path.read_text())


def test_a_retired_sleeve_holds_its_capital_in_sgov(monkeypatch, tmp_path):
    decision, book = _retired_session(monkeypatch, tmp_path)
    assert decision.signals == {"SNOW": 0}
    assert book["mean_reversion"]["SGOV"] == pytest.approx(150.0)
    assert book["mean_reversion"].get("SNOW", 0.0) == 0.0


def test_a_retired_sleeve_sells_what_it_still_holds(monkeypatch, tmp_path):
    decision, book = _retired_session(monkeypatch, tmp_path, book={"mean_reversion": {"SNOW": 40.0}})
    assert book["mean_reversion"].get("SNOW", 0.0) == 0.0
    assert book["mean_reversion"]["SGOV"] == pytest.approx(150.0)
