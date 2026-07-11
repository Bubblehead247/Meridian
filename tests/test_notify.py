"""Tests for meridian.execution.notify.

Patches urllib.request.urlopen so no real HTTP traffic happens.
Verifies headers and message content for each notification type.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from meridian.execution.broker import Fill
from meridian.execution.notify import (
    _NTFY_URL,
    notify_daily_status,
    notify_entry,
    notify_exit,
    notify_fill,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_request() -> tuple[MagicMock, list[dict]]:
    """Return (mock_urlopen, captured_calls).

    Each captured call is {"title": ..., "message": ..., "tags": ...}.
    """
    calls: list[dict] = []

    def fake_urlopen(req, timeout=5):
        calls.append({
            "url": req.full_url,
            "title": req.get_header("Title"),
            "message": req.data.decode("utf-8"),
            "tags": req.get_header("Tags") or "",
            "priority": req.get_header("Priority"),
        })
        return MagicMock().__enter__.return_value

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = lambda s: s
    mock_ctx.__exit__ = MagicMock(return_value=False)

    def fake_urlopen2(req, timeout=5):
        calls.append({
            "url": req.full_url,
            "title": req.get_header("Title"),
            "message": req.data.decode("utf-8"),
            "tags": req.get_header("Tags") or "",
            "priority": req.get_header("Priority"),
        })
        return mock_ctx

    return fake_urlopen2, calls


def _make_decision(
    family: str = "mean_reversion",
    model: str = "zscore_21",
    symbol: str = "SPY",
    signals: dict | None = None,
    target_shares: dict | None = None,
    orders: list | None = None,
    skipped: bool = False,
    skip_reason: str = "",
    weight: float = 0.15,
    day_changes: dict | None = None,
):
    from meridian.execution.live_runner import StrategyDecision
    return StrategyDecision(
        family=family,
        model=model,
        symbol=symbol,
        as_of=date.today(),
        signals=signals or {"SPY": 1},
        target_shares=target_shares if target_shares is not None else {"SPY": 10.5},
        orders=orders or [],
        skipped=skipped,
        skip_reason=skip_reason,
        weight=weight,
        day_changes=day_changes or {},
    )


# ---------------------------------------------------------------------------
# notify_entry
# ---------------------------------------------------------------------------

def test_notify_entry_posts_to_correct_url():
    fake_open, calls = _capture_request()
    fill = Fill(symbol="SPY", qty=10.5, price=450.00)
    with patch("urllib.request.urlopen", fake_open):
        notify_entry(fill, "mean_reversion", "zscore_21")
    assert len(calls) == 1
    assert calls[0]["url"] == _NTFY_URL


def test_notify_entry_title_contains_symbol():
    fake_open, calls = _capture_request()
    fill = Fill(symbol="QQQ", qty=5.0, price=370.00)
    with patch("urllib.request.urlopen", fake_open):
        notify_entry(fill, "momentum", "dual_momentum")
    assert "QQQ" in calls[0]["title"]


def test_notify_entry_message_contains_key_fields():
    fake_open, calls = _capture_request()
    fill = Fill(symbol="SPY", qty=8.0, price=450.00)
    with patch("urllib.request.urlopen", fake_open):
        notify_entry(fill, "mean_reversion", "zscore_21")
    msg = calls[0]["message"]
    assert "BUY" in msg
    assert "8.00" in msg
    assert "450.00" in msg
    assert "mean_reversion" in msg
    assert "zscore_21" in msg


def test_notify_entry_priority_is_high():
    fake_open, calls = _capture_request()
    fill = Fill(symbol="SPY", qty=5.0, price=450.00)
    with patch("urllib.request.urlopen", fake_open):
        notify_entry(fill, "mean_reversion", "zscore_21")
    assert calls[0]["priority"] == "high"


def test_notify_entry_pending_order_says_order_placed():
    """After-hours orders have no fill price — message must not show $0.00."""
    fake_open, calls = _capture_request()
    fill = Fill(symbol="SPY", qty=10.0, price=0.0, order_id="abc", filled=False)
    with patch("urllib.request.urlopen", fake_open):
        notify_entry(fill, "mean_reversion", "zscore_21", last_close=450.12)
    assert "Order Placed" in calls[0]["title"]
    msg = calls[0]["message"]
    assert "$0.00" not in msg
    assert "last close $450.12" in msg
    assert "fills at next market open" in msg


def test_notify_entry_pending_without_last_close():
    fake_open, calls = _capture_request()
    fill = Fill(symbol="SPY", qty=10.0, price=0.0, order_id="abc", filled=False)
    with patch("urllib.request.urlopen", fake_open):
        notify_entry(fill, "mean_reversion", "zscore_21")
    msg = calls[0]["message"]
    assert "$0.00" not in msg
    assert "fills at next market open" in msg


# ---------------------------------------------------------------------------
# notify_exit
# ---------------------------------------------------------------------------

def test_notify_exit_posts_to_correct_url():
    fake_open, calls = _capture_request()
    fill = Fill(symbol="SPY", qty=-10.5, price=455.00)
    with patch("urllib.request.urlopen", fake_open):
        notify_exit(fill, "mean_reversion", "zscore_21")
    assert len(calls) == 1
    assert calls[0]["url"] == _NTFY_URL


def test_notify_exit_title_contains_symbol():
    fake_open, calls = _capture_request()
    fill = Fill(symbol="IWM", qty=-3.0, price=200.00)
    with patch("urllib.request.urlopen", fake_open):
        notify_exit(fill, "momentum", "dual_momentum")
    assert "IWM" in calls[0]["title"]


def test_notify_exit_message_contains_sell():
    fake_open, calls = _capture_request()
    fill = Fill(symbol="SPY", qty=-10.0, price=455.00)
    with patch("urllib.request.urlopen", fake_open):
        notify_exit(fill, "mean_reversion", "zscore_21")
    msg = calls[0]["message"]
    assert "SELL" in msg
    assert "10.00" in msg
    assert "455.00" in msg


def test_notify_exit_priority_is_high():
    fake_open, calls = _capture_request()
    fill = Fill(symbol="SPY", qty=-5.0, price=450.00)
    with patch("urllib.request.urlopen", fake_open):
        notify_exit(fill, "mean_reversion", "zscore_21")
    assert calls[0]["priority"] == "high"


def test_notify_exit_pending_order_says_order_placed():
    fake_open, calls = _capture_request()
    fill = Fill(symbol="SPY", qty=-10.0, price=0.0, order_id="abc", filled=False)
    with patch("urllib.request.urlopen", fake_open):
        notify_exit(fill, "mean_reversion", "zscore_21", last_close=455.00)
    assert "Order Placed" in calls[0]["title"]
    assert "$0.00" not in calls[0]["message"]
    assert "last close $455.00" in calls[0]["message"]


# ---------------------------------------------------------------------------
# notify_fill (morning confirmation)
# ---------------------------------------------------------------------------

def test_notify_fill_message_contains_actual_price():
    fake_open, calls = _capture_request()
    record = {
        "symbol": "SPY", "qty": 10.0, "price": 450.30,
        "family": "mean_reversion", "model": "zscore_21",
        "submitted": "2026-07-09",
    }
    with patch("urllib.request.urlopen", fake_open):
        notify_fill(record)
    assert "Order Filled" in calls[0]["title"]
    assert "SPY" in calls[0]["title"]
    msg = calls[0]["message"]
    assert "BUY" in msg
    assert "450.30" in msg
    assert "2026-07-09" in msg


def test_notify_fill_survives_network_error():
    import urllib.error
    record = {"symbol": "SPY", "qty": -5.0, "price": 450.30}
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
        notify_fill(record)  # must not raise


# ---------------------------------------------------------------------------
# notify_daily_status
# ---------------------------------------------------------------------------

def test_daily_status_sends_exactly_one_message():
    fake_open, calls = _capture_request()
    decisions = [_make_decision(), _make_decision(family="momentum", model="dm", symbol="QQQ")]
    with patch("urllib.request.urlopen", fake_open):
        notify_daily_status(decisions)
    assert len(calls) == 1


def test_daily_status_lists_held_tickers_with_day_change():
    fake_open, calls = _capture_request()
    decisions = [
        _make_decision(target_shares={"SPY": 10.5}, day_changes={"SPY": 0.0123}),
        _make_decision(
            family="momentum", model="dm", symbol="QQQ",
            target_shares={"QQQ": 5.0}, day_changes={"QQQ": -0.004},
        ),
    ]
    with patch("urllib.request.urlopen", fake_open):
        notify_daily_status(decisions)
    msg = calls[0]["message"]
    assert "SPY  +1.23%" in msg
    assert "QQQ  -0.40%" in msg


def test_daily_status_excludes_flat_and_skipped():
    fake_open, calls = _capture_request()
    decisions = [
        _make_decision(target_shares={"SPY": 10.5}, day_changes={"SPY": 0.01}),
        _make_decision(family="mean_reversion", symbol="SNOW",
                       target_shares={"SNOW": 0.0}),                    # flat: no position
        _make_decision(family="sector_rotation", symbol="XLE",
                       target_shares={"XLE": 3.0},
                       skipped=True, skip_reason="price fetch failed"),  # skipped
    ]
    with patch("urllib.request.urlopen", fake_open):
        notify_daily_status(decisions)
    msg = calls[0]["message"]
    assert "SPY" in msg
    assert "SNOW" not in msg
    assert "XLE" not in msg


def test_daily_status_missing_day_change_shows_na():
    fake_open, calls = _capture_request()
    decisions = [_make_decision(target_shares={"SPY": 10.5}, day_changes={})]
    with patch("urllib.request.urlopen", fake_open):
        notify_daily_status(decisions)
    assert "SPY  n/a" in calls[0]["message"]


def test_daily_status_no_positions_message():
    fake_open, calls = _capture_request()
    decisions = [_make_decision(target_shares={"SPY": 0.0})]
    with patch("urllib.request.urlopen", fake_open):
        notify_daily_status(decisions)
    assert "No open positions." in calls[0]["message"]
    assert "(0 positions)" in calls[0]["title"]


def test_daily_status_title_shows_position_count():
    fake_open, calls = _capture_request()
    decisions = [
        _make_decision(target_shares={"SPY": 10.5}, day_changes={"SPY": 0.01}),
        _make_decision(family="momentum", model="dm", symbol="QQQ",
                       target_shares={"QQQ": 5.0}, day_changes={"QQQ": 0.02}),
    ]
    with patch("urllib.request.urlopen", fake_open):
        notify_daily_status(decisions)
    assert "(2 positions)" in calls[0]["title"]


# ---------------------------------------------------------------------------
# Network failure is silent (no raise)
# ---------------------------------------------------------------------------

def test_notify_entry_survives_network_error():
    fill = Fill(symbol="SPY", qty=5.0, price=450.00)
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
        notify_entry(fill, "mean_reversion", "zscore_21")  # must not raise


def test_notify_exit_survives_network_error():
    fill = Fill(symbol="SPY", qty=-5.0, price=450.00)
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
        notify_exit(fill, "mean_reversion", "zscore_21")  # must not raise


def test_daily_status_survives_network_error():
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
        notify_daily_status([_make_decision()])  # must not raise
