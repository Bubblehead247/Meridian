"""
main.py — Meridian scheduler loop (child process of tray.py).

Fires twice on every US equities trading day:
  9:45 ET — meridian reconcile  — fetch actual fill prices for orders placed
                                  after yesterday's close; log and notify them
 16:30 ET — meridian run-paper  — compute signals and send orders to Alpaca
            meridian review     — monthly report (first trading day of month only)

Logs to meridian.log at the project root.

Run directly for testing:  python -m meridian.tray.main
The tray app launches it as a child process via subprocess.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET           = ZoneInfo("America/New_York")
HERE         = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
LOG_FILE     = PROJECT_ROOT / "meridian.log"

from quantcore.single_instance import SingleInstance

# Log to the file, and additionally to the console only when there is a real
# console. When the tray launches this as a child it redirects stdout into
# LOG_FILE, so an unconditional StreamHandler wrote every line twice — through
# two independent, unsynchronized append handles on one file. That is the source
# of the duplicated lines around 2026-07-14, separate from the genuine
# double-scheduler run the lock below prevents.
_handlers: list[logging.Handler] = [logging.FileHandler(LOG_FILE, encoding="utf-8")]
if sys.stdout is not None and getattr(sys.stdout, "isatty", lambda: False)():
    _handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=_handlers,
)
logger = logging.getLogger("meridian.scheduler")


# ---------------------------------------------------------------------------
# Trading-day check
# ---------------------------------------------------------------------------

def is_trading_day(dt: date) -> bool:
    """True if ``dt`` is a US equities trading day, per the Alpaca calendar.

    Delegates to the shared gate, which **fails closed**: if the calendar cannot
    be established, the answer is "no" and the session is skipped.

    This used to fall back to ``dt.weekday() < 5`` on any failure, so a bad key,
    a missing key or an Alpaca outage silently turned every weekday into a
    trading day — the bot would have run a session on Thanksgiving. Skipping a
    day costs one day of signals; trading into a closed or misjudged market costs
    money. The shared gate also caches the calendar, so a network blip falls back
    to the cached answer rather than stopping trading.
    """
    from quantcore.market_calendar import trading_session

    return trading_session("meridian", dt, root=PROJECT_ROOT) is not None


def _is_first_trading_day_of_month(today: date) -> bool:
    """True if no earlier date in this month is a trading day."""
    d = date(today.year, today.month, 1)
    while d < today:
        if is_trading_day(d):
            return False
        d += timedelta(days=1)
    return True


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def _run_cmd(label: str, args: list[str], timeout: int = 300) -> bool:
    """Run a meridian CLI command; log all output. Returns True on success."""
    logger.info(f"{label}: starting")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "meridian", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        for line in result.stdout.splitlines():
            if line.strip():
                logger.info(f"  {line}")
        if result.returncode != 0:
            logger.error(f"{label}: exited {result.returncode}")
            for line in result.stderr.splitlines():
                if line.strip():
                    logger.error(f"  {line}")
            return False
        logger.info(f"{label}: done")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"{label}: timed out after {timeout}s")
        return False
    except Exception as exc:
        logger.error(f"{label}: {exc}")
        return False


def run_paper_session() -> None:
    """Run meridian run-paper for today."""
    today = datetime.now(ET).date()
    if not is_trading_day(today):
        logger.info("Market closed today — skipping paper session.")
        return
    _run_cmd("run-paper", ["run-paper"])


def run_reconcile() -> None:
    """Run meridian reconcile — price yesterday's pending orders."""
    today = datetime.now(ET).date()
    if not is_trading_day(today):
        logger.info("Market closed today — skipping reconcile.")
        return
    _run_cmd("reconcile", ["reconcile"])


def run_monthly_review() -> None:
    """Run meridian review on the first trading day of each month."""
    today = datetime.now(ET).date()
    if not is_trading_day(today):
        return
    if not _is_first_trading_day_of_month(today):
        return
    _run_cmd("review", ["review"])


def fire() -> None:
    """The 4:30 PM ET job: paper session then (maybe) monthly review."""
    run_paper_session()
    run_monthly_review()


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------

FIRE_HOUR   = 16
FIRE_MINUTE = 30

RECON_HOUR   = 9
RECON_MINUTE = 45


#: Lock name. Any second scheduler process using this name refuses to start.
SCHEDULER_LOCK = "Meridian-scheduler"


def run() -> None:
    """Block forever: reconcile at 9:45 ET, paper session at 16:30 ET, daily.

    Holds a machine-wide lock for its whole life. ``last_run``/``last_recon``
    below are ordinary local variables, so they only stop *this* process firing
    twice — a second scheduler has its own copies and fires the same session
    again. That is what happened on 2026-07-14: two ``run-paper: starting`` lines
    77 ms apart, and every order placed twice.
    """
    with SingleInstance(SCHEDULER_LOCK) as lock:
        if not lock.acquired:
            logger.error(
                "Another Meridian scheduler is already running — refusing to "
                "start a second one. Stop the other process first."
            )
            return

        logger.info(
            "Meridian scheduler started (reconcile 9:45 ET, session 16:30 ET, trading days)."
        )
        last_run:   date | None = None
        last_recon: date | None = None

        while True:
            now_et = datetime.now(ET)
            today  = now_et.date()

            past_recon_time = (now_et.hour, now_et.minute) >= (RECON_HOUR, RECON_MINUTE)
            if past_recon_time and last_recon != today:
                last_recon = today
                run_reconcile()

            past_fire_time = (now_et.hour, now_et.minute) >= (FIRE_HOUR, FIRE_MINUTE)
            if past_fire_time and last_run != today:
                last_run = today
                fire()

            time.sleep(30)


if __name__ == "__main__":
    run()
