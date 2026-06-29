"""Push notifications to ntfy for trade entries, exits, and daily status.

Topic: Meridian-62926
Sends via the public ntfy.sh server (no auth required for this topic).

Three public functions:
- notify_entry(fill, family, model)
- notify_exit(fill, family, model)
- notify_daily_status(decisions)

Each is a no-op on network failure — never raise in the hot path.
"""

from __future__ import annotations

import urllib.request
import urllib.error
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meridian.execution.broker import Fill
    from meridian.execution.live_runner import StrategyDecision

_NTFY_URL = "https://ntfy.sh/Meridian-62926"


def _send(title: str, message: str, priority: str = "default", tags: str = "") -> None:
    """Fire-and-forget POST to ntfy. Silently swallows network errors."""
    headers: dict[str, str] = {
        "Title": title.encode("utf-8").decode("latin-1", errors="replace"),
        "Priority": priority,
        "Content-Type": "text/plain; charset=utf-8",
    }
    if tags:
        headers["Tags"] = tags

    try:
        data = message.encode("utf-8")
        req = urllib.request.Request(_NTFY_URL, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5):
            pass
    except (urllib.error.URLError, OSError):
        pass


def notify_entry(fill: "Fill", family: str, model: str) -> None:
    """Notify that a position was entered."""
    direction = "BUY" if fill.qty > 0 else "SELL"
    title = f"Trade Entry — {fill.symbol}"
    body = (
        f"{direction} {abs(fill.qty):.2f} shares of {fill.symbol} @ ${fill.price:.2f}\n"
        f"Family: {family}  |  Model: {model}\n"
        f"Date: {date.today()}"
    )
    _send(title, body, priority="high", tags="arrow_up,chart_with_upwards_trend")


def notify_exit(fill: "Fill", family: str, model: str) -> None:
    """Notify that a position was exited (qty is negative = sell-to-close)."""
    direction = "SELL" if fill.qty < 0 else "BUY TO COVER"
    title = f"Trade Exit — {fill.symbol}"
    body = (
        f"{direction} {abs(fill.qty):.2f} shares of {fill.symbol} @ ${fill.price:.2f}\n"
        f"Family: {family}  |  Model: {model}\n"
        f"Date: {date.today()}"
    )
    _send(title, body, priority="high", tags="arrow_down,white_check_mark")


def notify_daily_status(decisions: list["StrategyDecision"]) -> None:
    """Send one daily summary message covering all active strategies."""
    active = [d for d in decisions if not d.skipped and d.weight > 0]
    skipped = [d for d in decisions if d.skipped]

    lines: list[str] = [f"Meridian Daily Status — {date.today()}", ""]

    for d in active:
        n_long = sum(1 for v in d.signals.values() if v > 0)
        n_flat = sum(1 for v in d.signals.values() if v == 0)
        order_count = len(d.orders)
        lines.append(f"{d.family}/{d.model}")
        lines.append(f"  {d.symbol}  |  long={n_long} flat={n_flat}  |  orders={order_count}")
        for sym, shares in d.target_shares.items():
            if shares != 0:
                lines.append(f"    {sym}: {shares:+.2f} sh")

    if skipped:
        lines.append("")
        lines.append(f"Skipped ({len(skipped)}):")
        for d in skipped:
            lines.append(f"  {d.family}/{d.model} — {d.skip_reason}")

    title = f"Meridian Daily Status ({len(active)} active)"
    _send(title, "\n".join(lines), priority="default", tags="bar_chart")
