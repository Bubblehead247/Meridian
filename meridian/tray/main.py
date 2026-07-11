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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("meridian.scheduler")


# ---------------------------------------------------------------------------
# Trading-day check
# ---------------------------------------------------------------------------

def is_trading_day(dt: date) -> bool:
    """True if ``dt`` is a US equities trading day per the Alpaca calendar.

    Falls back to weekday-only when Alpaca credentials are absent or the
    API is unreachable.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")

        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetCalendarRequest

        client = TradingClient(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
            paper=True,
        )
        cal = client.get_calendar(
            GetCalendarRequest(start=dt.isoformat(), end=dt.isoformat())
        )
        return len(cal) > 0
    except Exception:
        return dt.weekday() < 5  # weekday fallback


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


def run() -> None:
    """Block forever: reconcile at 9:45 ET, paper session at 16:30 ET, daily."""
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
