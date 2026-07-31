"""Regression tests for the issues raised in the 2026-07-28 broker review.

Covers, in order:

* **G** — rebalance orders below the broker's $1 minimum were submitted and
  rejected with "cost basis must be >= minimal amount of order 1". Confirmed live
  on 2026-07-22, 07-23 and again 07-29 for XLRE.
* **H** — nothing stopped two schedulers running at once. Two
  ``run-paper: starting`` lines appear 77 ms apart on 2026-07-14, and the session
  for 07-15 is missing entirely.
* **I** — ``LedgerStore`` existed but had no caller, so no sleeve ledger was ever
  written and no sleeve's profit was recorded anywhere.
* **J** — ``event_driven`` registers no models and cannot trade, yet was counted
  toward "nine strategy families".
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from meridian.execution import live_runner
from meridian.execution.broker import BaseBroker, SimulatedBroker


class _FakeModel:
    """Single-asset model that always emits a long signal."""

    cross_sectional = False

    def signals(self, prices: pd.Series) -> pd.Series:
        return pd.Series(1, index=prices.index)


def _patch_session(monkeypatch, picks, price=100.0):
    monkeypatch.setattr(live_runner, "load_live_picks", lambda: picks)

    def fake_fetch(symbols, start="2023-01-01"):
        idx = pd.RangeIndex(300)
        return {s: pd.Series(price, index=idx) for s in symbols}

    monkeypatch.setattr(live_runner, "_fetch_prices", fake_fetch)
    monkeypatch.setattr(live_runner, "create_model", lambda family, model: _FakeModel())
    # Never touch the real ntfy topic or the real sleeve ledgers from a test.
    monkeypatch.setattr(live_runner, "notify_entry", lambda *a, **k: None)
    monkeypatch.setattr(live_runner, "notify_exit", lambda *a, **k: None)
    monkeypatch.setattr(live_runner, "notify_daily_status", lambda *a, **k: None)
    monkeypatch.setattr(live_runner, "update_sleeve_ledgers", lambda *a, **k: {})


class _RecordingBroker(SimulatedBroker):
    """Records every order that reaches the broker."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orders: list[tuple[str, float]] = []

    def market_order(self, symbol, qty):
        self.orders.append((symbol, qty))
        return super().market_order(symbol, qty)


# --------------------------------------------------------------------------
# G — sub-$1 orders must never be submitted
# --------------------------------------------------------------------------


def test_a_sub_one_dollar_delta_is_never_submitted(monkeypatch, tmp_path):
    """0.0059 shares of XLRE at $46 is $0.27 — the broker rejects it."""
    picks = {"sector_rotation": {"model": "relative_strength_b05", "symbol": "XLRE"}}
    _patch_session(monkeypatch, picks, price=46.02)

    positions = tmp_path / "positions.json"
    # Book already holds all but 0.0059 shares of the target.
    broker = _RecordingBroker(cash=10_000.0, cost_bps=0.0)
    target = round(10_000.0 * 0.10 / 46.02, 6)
    positions.write_text(json.dumps({"sector_rotation": {"XLRE": target - 0.0059}}))

    live_runner.run_paper_session(
        broker, account_equity=10_000.0, positions_path=positions)

    assert broker.orders == [], (
        f"a sub-$1 order was submitted: {broker.orders}")


def test_the_skipped_delta_is_retried_once_it_is_worth_placing(monkeypatch, tmp_path):
    """The delta must stay in the target, not be dropped.

    Skipping leaves the position book untouched, so the shortfall carries into
    the next session and goes out as soon as it is worth placing.

    Updated for the D4 rebalance band (2026-07-30): "worth placing" used to mean
    just clearing Alpaca's $1 minimum. It now also has to clear
    ``REBALANCE_BAND_PCT`` of the target, so a shortfall of 0.5 shares against a
    21.7-share target — 2.3%, about $23 — is deliberately left alone now. That is
    the churn the band exists to stop. The carry-over behaviour this test guards
    is unchanged: the book is never quietly adjusted for an order not sent.
    """
    picks = {"sector_rotation": {"model": "relative_strength_b05", "symbol": "XLRE"}}
    _patch_session(monkeypatch, picks, price=46.02)
    positions = tmp_path / "positions.json"
    target = round(10_000.0 * 0.10 / 46.02, 6)

    # Session 1: shortfall worth $0.27 — skipped, and the book is unchanged.
    broker = _RecordingBroker(cash=10_000.0, cost_bps=0.0)
    positions.write_text(json.dumps({"sector_rotation": {"XLRE": target - 0.0059}}))
    live_runner.run_paper_session(
        broker, account_equity=10_000.0, positions_path=positions)
    assert broker.orders == []
    unchanged = json.loads(positions.read_text())["sector_rotation"]["XLRE"]
    assert unchanged == pytest.approx(target - 0.0059), (
        "the skipped delta was silently absorbed into the book")

    # Session 2: a shortfall of 0.5 shares clears $1 but sits inside the band.
    broker = _RecordingBroker(cash=10_000.0, cost_bps=0.0)
    positions.write_text(json.dumps({"sector_rotation": {"XLRE": target - 0.5}}))
    live_runner.run_paper_session(
        broker, account_equity=10_000.0, positions_path=positions)
    assert broker.orders == [], "2.3% of the target is inside the rebalance band"

    # Session 3: 3 shares short — 13.8% of the target, outside the band.
    broker = _RecordingBroker(cash=10_000.0, cost_bps=0.0)
    positions.write_text(json.dumps({"sector_rotation": {"XLRE": target - 3.0}}))
    live_runner.run_paper_session(
        broker, account_equity=10_000.0, positions_path=positions)
    assert len(broker.orders) == 1, "a real divergence should have been placed"
    assert broker.orders[0][1] == pytest.approx(3.0, abs=1e-6)


def test_an_order_comfortably_over_the_minimum_still_goes_through(monkeypatch, tmp_path):
    picks = {"trend_following": {"model": "ma_trend_long_only", "symbol": "XLK"}}
    _patch_session(monkeypatch, picks, price=171.0)
    broker = _RecordingBroker(cash=100_000.0, cost_bps=0.0)

    live_runner.run_paper_session(
        broker, account_equity=100_000.0, positions_path=tmp_path / "p.json")

    assert len(broker.orders) == 1
    symbol, qty = broker.orders[0]
    assert symbol == "XLK"
    assert abs(qty) * 171.0 >= live_runner.MIN_ORDER_NOTIONAL


def test_the_threshold_is_the_brokers_actual_minimum():
    assert live_runner.MIN_ORDER_NOTIONAL == 1.0


# --------------------------------------------------------------------------
# H — two schedulers must not run at once
# --------------------------------------------------------------------------


def test_a_second_lock_holder_is_refused():
    """The core guarantee: the second acquirer must be told no."""
    from quantcore.single_instance import SingleInstance

    with SingleInstance("meridian-test-lock") as first:
        assert first.acquired
        with SingleInstance("meridian-test-lock") as second:
            assert not second.acquired, "two holders acquired the same lock"


def test_the_lock_is_released_when_the_holder_exits():
    from quantcore.single_instance import SingleInstance

    with SingleInstance("meridian-test-lock-2") as first:
        assert first.acquired
    with SingleInstance("meridian-test-lock-2") as again:
        assert again.acquired, "the lock was not released"


def test_different_names_do_not_block_each_other():
    """SeykotaBot and Meridian must not lock each other out."""
    from quantcore.single_instance import SingleInstance

    with SingleInstance("SeykotaBot-test") as a, SingleInstance("Meridian-test") as b:
        assert a.acquired and b.acquired


def test_the_scheduler_and_the_session_take_a_lock():
    """Both routes into placing orders must be guarded, with distinct names."""
    from meridian.cli import SESSION_LOCK
    from meridian.tray.main import SCHEDULER_LOCK

    assert SCHEDULER_LOCK and SESSION_LOCK
    assert SCHEDULER_LOCK != SESSION_LOCK, (
        "the scheduler holds its lock for its whole life, so the session it "
        "launches must use a different name or it would block itself")


def test_seykotabot_and_meridian_share_one_implementation():
    """Two near-copies of a lock is how they drift apart."""
    import sys

    sys.path.insert(0, r"C:\Users\Cody\claude\Seykotabot")
    try:
        from execution.single_instance import SingleInstance as SeykotaLock
    finally:
        sys.path.pop(0)
    from quantcore.single_instance import SingleInstance as SharedLock

    assert SeykotaLock is SharedLock


# --------------------------------------------------------------------------
# I — sleeve ledgers must actually be written
# --------------------------------------------------------------------------


def test_sleeve_ledgers_are_written_to_disk(tmp_path):
    from meridian.portfolio.ledger import LedgerStore
    from meridian.portfolio.sleeve_ledgers import update_sleeve_ledgers

    log = tmp_path / "trade_log.jsonl"
    log.write_text(
        json.dumps({"date": "2026-07-22", "symbol": "XLK", "qty": 1.0,
                    "side": "buy", "price": 100.0, "family": "trend_following"}) + "\n",
        encoding="utf-8",
    )
    store = LedgerStore(tmp_path / "sleeves")

    saved = update_sleeve_ledgers(10_000.0, {"XLK": 110.0},
                                  store=store, trade_log_path=log)

    assert saved, "no sleeve ledgers were produced"
    assert (tmp_path / "sleeves" / "trend_following.json").exists()
    trend = store.load("trend_following")
    assert trend.capital_alloc == pytest.approx(1500.0)   # 15% of 10k
    assert trend.unrealized_pnl == pytest.approx(10.0)    # 1 share, +$10


def test_realized_profit_is_matched_first_in_first_out(tmp_path):
    from meridian.portfolio.sleeve_ledgers import pnl_by_sleeve

    fills = [
        {"symbol": "XLK", "qty": 1.0, "side": "buy", "price": 100.0,
         "family": "trend_following"},
        {"symbol": "XLK", "qty": 1.0, "side": "buy", "price": 120.0,
         "family": "trend_following"},
        {"symbol": "XLK", "qty": -1.0, "side": "sell", "price": 130.0,
         "family": "trend_following"},
    ]
    stats = pnl_by_sleeve(fills, {"XLK": 125.0})["trend_following"]
    # FIFO closes the $100 lot: +$30 realized. The $120 lot stays open: +$5.
    assert stats["realized"] == pytest.approx(30.0)
    assert stats["unrealized"] == pytest.approx(5.0)


def test_a_sell_with_no_recorded_buy_is_not_booked_as_pure_profit(tmp_path):
    """The fill log starts partway through the account's life.

    Treating an unmatched sell as a zero cost basis would invent a large profit.
    """
    from meridian.portfolio.sleeve_ledgers import pnl_by_sleeve

    fills = [{"symbol": "XLI", "qty": -1.36, "side": "sell", "price": 180.58,
              "family": "sector_rotation"}]
    stats = pnl_by_sleeve(fills, {})["sector_rotation"]
    assert stats["realized"] == pytest.approx(0.0)


def test_breakouts_profit_lands_in_the_trend_following_sleeve(tmp_path):
    """breakouts is funded by the trend_following sleeve, not its own."""
    from meridian.portfolio.sleeve_ledgers import pnl_by_sleeve, sleeve_for

    assert sleeve_for("breakouts") == "trend_following"
    fills = [
        {"symbol": "TRGP", "qty": 1.0, "side": "buy", "price": 10.0,
         "family": "breakouts"},
        {"symbol": "TRGP", "qty": -1.0, "side": "sell", "price": 12.0,
         "family": "breakouts"},
    ]
    stats = pnl_by_sleeve(fills, {})
    assert stats["trend_following"]["realized"] == pytest.approx(2.0)
    assert "breakouts" not in stats


def test_rerunning_does_not_double_count_profit(tmp_path):
    """A double session must not double the recorded profit."""
    from meridian.portfolio.ledger import LedgerStore
    from meridian.portfolio.sleeve_ledgers import update_sleeve_ledgers

    log = tmp_path / "trade_log.jsonl"
    log.write_text(
        json.dumps({"symbol": "XLK", "qty": 1.0, "side": "buy", "price": 100.0,
                    "family": "trend_following"}) + "\n"
        + json.dumps({"symbol": "XLK", "qty": -1.0, "side": "sell", "price": 130.0,
                      "family": "trend_following"}) + "\n",
        encoding="utf-8",
    )
    store = LedgerStore(tmp_path / "sleeves")

    update_sleeve_ledgers(10_000.0, {}, store=store, trade_log_path=log)
    first = store.load("trend_following").realized_pnl
    update_sleeve_ledgers(10_000.0, {}, store=store, trade_log_path=log)
    second = store.load("trend_following").realized_pnl

    assert first == pytest.approx(30.0)
    assert second == pytest.approx(first), "profit was counted twice"


def test_sleeve_ledgers_live_apart_from_the_position_book():
    """LedgerStore.list() globs *.json, and ledger/ holds two other schemas."""
    from meridian.execution.positions import POSITIONS_FILE
    from meridian.portfolio.sleeve_ledgers import SLEEVE_LEDGER_DIR

    assert SLEEVE_LEDGER_DIR != POSITIONS_FILE.parent, (
        "sleeve ledgers share a directory with positions.json, so listing them "
        "would try to load the position book as a StrategyLedger")


# --------------------------------------------------------------------------
# J — event_driven must not be counted as a live family
# --------------------------------------------------------------------------


def test_event_driven_registers_no_models():
    from meridian.families.registry import list_families, list_models

    assert "event_driven" not in list_families()
    assert list_models("event_driven") == []


def test_event_driven_is_explicitly_marked_inactive():
    from meridian.families import event_driven

    assert event_driven.ACTIVE is False
    assert "deferred" in event_driven.STATUS.lower()


def test_event_driven_cannot_trade():
    from meridian.families.permissions import is_permitted

    # Deferred families are blocked in every regime, so no label is needed.
    assert is_permitted("event_driven", None) is False


def test_the_family_status_list_includes_breakouts():
    """breakouts has 7 models and a live pick, and never appeared in the chart."""
    from meridian.interactive import _active_families

    families = _active_families()
    assert "breakouts" in families
    assert "event_driven" not in families
    assert "experimental_research" not in families, (
        "experimental_research is a funding sleeve, not a strategy family")


# --- D3: sleeves are sized off the real account, not a hardcoded $10,000 -----


def _fake_args(**kwargs):
    from argparse import Namespace

    base = dict(equity=None, dry_run=False, live=False, start=None)
    base.update(kwargs)
    return Namespace(**base)


class _EquityBroker:
    """Stands in for AlpacaBroker, reporting an account that is not $10,000."""

    def __init__(self, *args, **kwargs):
        pass

    def get_account_equity(self):
        return 9_853.25, 9_779.89


def _capture_equity(monkeypatch):
    """Run _run_paper with the broker and session stubbed; return the equity used."""
    from meridian import cli
    from meridian.execution import broker as broker_mod
    from meridian.execution import live_runner

    seen = {}

    def fake_session(broker, account_equity=None, **kwargs):
        seen["equity"] = account_equity
        return []

    monkeypatch.setattr(broker_mod, "AlpacaBroker", _EquityBroker)
    monkeypatch.setattr(live_runner, "run_paper_session", fake_session)
    monkeypatch.setattr(live_runner, "print_session_report", lambda *a, **k: None)
    return cli, seen


def test_run_paper_sizes_off_live_account_equity(monkeypatch):
    """The old default sized every sleeve against $10,000 while the account held
    ~$9,730, and the gap grew with P&L."""
    cli, seen = _capture_equity(monkeypatch)

    cli._run_paper(_fake_args())

    assert seen["equity"] == 9_853.25


def test_an_explicit_equity_still_wins(monkeypatch):
    """--equity is how you deliberately size off a notional instead."""
    cli, seen = _capture_equity(monkeypatch)

    cli._run_paper(_fake_args(equity=25_000.0))

    assert seen["equity"] == 25_000.0


def test_equity_falls_back_when_the_broker_cannot_report_it(monkeypatch):
    """A broker returning (None, None) must not size everything off zero."""
    from meridian import cli

    cli_mod, seen = _capture_equity(monkeypatch)
    monkeypatch.setattr(_EquityBroker, "get_account_equity", lambda self: (None, None))

    cli_mod._run_paper(_fake_args())

    assert seen["equity"] == cli.DEFAULT_EQUITY


# --- O5: the ntfy topic is a credential, not a constant ---------------------


def test_the_ntfy_topic_is_not_hardcoded_anywhere():
    """ntfy.sh is public and unauthenticated: the topic name IS the credential.

    It was committed in this file until 2026-07-30, so anyone with the repo could
    read every trade alert and publish forged ones.
    """
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "meridian" / "execution" / "notify.py"
    text = source.read_text(encoding="utf-8")

    assert "ntfy.sh/Meridian" not in text, "a topic is hardcoded again"
    assert "NTFY_TOPIC" in text, "the topic must come from the environment"


def test_no_topic_configured_means_no_request(monkeypatch):
    """Better silent than falling back to a default topic published in the repo."""
    from meridian.execution import notify

    monkeypatch.setenv("NTFY_TOPIC", "")
    assert notify._ntfy_url() is None

    sent = []
    monkeypatch.setattr(notify.urllib.request, "urlopen",
                        lambda *a, **k: sent.append(1))
    notify._send("title", "message")
    assert sent == [], "posted somewhere with no topic configured"


def test_the_project_env_beats_an_ambient_variable(monkeypatch):
    """This machine has a user-level NTFY_TOPIC=WeeklyAI from another project.

    `load_dotenv()` does not override an existing OS variable, so without
    override=True that stray value silently captured Meridian's alerts.
    """
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "meridian" / "cli.py"
    assert "load_dotenv(override=True)" in source.read_text(encoding="utf-8")
