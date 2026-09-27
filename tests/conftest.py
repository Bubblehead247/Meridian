"""Shared test guards.

**Why this file exists.** ``POSITIONS_FILE``, ``PENDING_ORDERS_FILE`` and
``TRADE_LOG_FILE`` all default to a *relative* ``ledger/…`` path, and
``notify._send`` posts to a real public ntfy topic. So a test that calls a
production function without an explicit ``path=`` reads and appends to the live
trading ledger, and a test that reaches a notify path publishes a real push
notification.

Both risks are live today: ``test_live_session.py`` calls
``run_paper_session(broker)`` with no ``positions_path``, and several of its cases
run with ``dry_run=False`` and non-empty picks, which reaches ``notify_entry`` and
``notify_daily_status``.

This is not a theoretical concern. Writing the SeykotaBot ledger tests polluted
that bot's real, never-rotated trade ledger with 48 rows of fake orders before it
was caught. These fixtures are ``autouse`` so protection is the default.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _never_touch_the_real_ledger(monkeypatch, tmp_path):
    """Redirect every default ledger path to a temp directory."""
    from meridian.execution import positions, reconcile, rebalance_schedule
    from meridian.experiments import run_log
    from meridian.pipeline import oos_guard
    from meridian.portfolio import sleeve_ledgers

    ledger = tmp_path / "ledger"
    ledger.mkdir(exist_ok=True)

    monkeypatch.setattr(positions, "POSITIONS_FILE", ledger / "positions.json")
    monkeypatch.setattr(
        rebalance_schedule, "REBALANCE_SCHEDULE_FILE", ledger / "last_rebalance.json"
    )
    monkeypatch.setattr(reconcile, "PENDING_ORDERS_FILE", ledger / "pending_orders.json")
    monkeypatch.setattr(reconcile, "TRADE_LOG_FILE", ledger / "trade_log.jsonl")
    monkeypatch.setattr(sleeve_ledgers, "TRADE_LOG_FILE", ledger / "trade_log.jsonl")
    monkeypatch.setattr(sleeve_ledgers, "SLEEVE_LEDGER_DIR", ledger / "sleeves")
    # OOSGuard now defaults to on (P1-B) — without this, every test that runs the OOS
    # stage would write real run-count files into the repo's ledger/oos_runs/.
    monkeypatch.setattr(oos_guard, "DEFAULT_GUARD_DIR", ledger / "oos_runs")
    # A graduation-criteria-version override (P1-C) logs to run_log.jsonl — redirect it
    # too, same reasoning as above.
    monkeypatch.setattr(run_log, "DEFAULT_RUN_LOG", ledger / "run_log.jsonl")


@pytest.fixture(autouse=True)
def _never_send_a_real_notification(monkeypatch):
    """Block the network under ntfy, leaving the notify code itself intact.

    The topic is a hardcoded public one that needs no auth, so an accidental send
    is a real push to a real phone — and ``_send`` swallows its own errors, so a
    test would never notice it had happened.

    This stubs the transport rather than ``_send``, because ``test_notify.py``
    exists precisely to assert on what ``_send`` builds. Stubbing ``_send`` would
    silently gut those tests. A test that patches ``urlopen`` itself still wins,
    since its patch is applied after this one.

    Captured requests are returned as a list, so a test can assert on them
    without stubbing anything itself. Requests are absorbed rather than rejected:
    ``run_paper_session`` legitimately notifies, and a test about position sizing
    should not have to stub notifications to stay offline.
    """
    import urllib.request

    sent: list = []

    class _Swallowed:
        """Stands in for the response object, which callers only use as a context."""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _capture(req, *args, **kwargs):
        sent.append(req)
        return _Swallowed()

    monkeypatch.setattr(urllib.request, "urlopen", _capture)
    return sent
