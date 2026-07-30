"""
tray.py — Meridian system-tray status icon.

Shows what the bot has actually done today, rather than merely that a process
exists. Those are not the same thing: a tray icon can sit green over a scheduler
that stopped hours ago.

The scheduled jobs now run as separate short-lived processes under Windows Task
Scheduler, whether or not anyone is logged in. This icon only reports on them.
Closing it does not stop the bot, and a failing job cannot take it down.

    grey    not scheduled yet, or no recent reconciliation
    green   every job ran, and the books agree with the broker
    amber   a job that should have run today has not
    red     a job failed, or the books disagree with the broker

Right-click for a per-job breakdown, "Run now", and the log.

Run with:  py -3.14 tray.py   (or double-click run_tray.bat)

The implementation is shared by all three bots — see ``quantcore/tray.py`` — so
they look and behave identically and there is one place to fix.
"""

from __future__ import annotations

from quantcore.tray import run_tray

if __name__ == "__main__":
    run_tray("meridian", label="Meridian", letter="M", colour=(34, 197, 94))
