"""
tray.py — Meridian system-tray application.

Green "M" icon in the Windows taskbar. Opens idle — does NOT auto-start
the scheduler. Use the menu to start/stop, trigger a manual run, view the
log, or quit. The scheduler runs as a child process so quitting the tray
app also stops the scheduler.

Run with:  py -3.14 tray.py   (or double-click run_tray.bat)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pystray

HERE         = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
MAIN_SCRIPT  = HERE / "main.py"
LOG_FILE     = PROJECT_ROOT / "meridian.log"

GREEN = (34, 197, 94)    # Tailwind green-500
WHITE = (255, 255, 255)

_scheduler: "subprocess.Popen | None" = None


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def scheduler_running() -> bool:
    return _scheduler is not None and _scheduler.poll() is None


def start_scheduler(icon, _item=None) -> None:
    global _scheduler
    if scheduler_running():
        return
    log = open(LOG_FILE, "a", buffering=1, encoding="utf-8")
    _scheduler = subprocess.Popen(
        [sys.executable, str(MAIN_SCRIPT)],
        cwd=PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    icon.update_menu()


def stop_scheduler(icon, _item=None) -> None:
    global _scheduler
    if scheduler_running():
        _scheduler.terminate()
        try:
            _scheduler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _scheduler.kill()
    _scheduler = None
    if icon is not None:
        icon.update_menu()


def run_now(icon, _item=None) -> None:
    """Fire a paper session immediately, regardless of the schedule."""
    log = open(LOG_FILE, "a", buffering=1, encoding="utf-8")
    subprocess.Popen(
        [sys.executable, "-m", "meridian", "run-paper"],
        cwd=PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def view_log(_icon=None, _item=None) -> None:
    if LOG_FILE.exists():
        os.startfile(LOG_FILE)  # noqa: S606


def quit_app(icon, _item=None) -> None:
    stop_scheduler(None)
    icon.stop()


# ---------------------------------------------------------------------------
# Icon
# ---------------------------------------------------------------------------

def make_icon() -> Image.Image:
    """64×64 green square with a bold white 'M' centred."""
    size = 64
    img  = Image.new("RGB", (size, size), GREEN)
    draw = ImageDraw.Draw(img)

    text = "M"
    font = None
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, 42)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    x = (size - (r - l)) / 2 - l
    y = (size - (b - t)) / 2 - t
    draw.text((x, y), text, fill=WHITE, font=font)
    return img


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

def _status(_item) -> str:
    return "● Scheduler: running" if scheduler_running() else "○ Scheduler: stopped"


def build_menu() -> pystray.Menu:
    return pystray.Menu(
        pystray.MenuItem(_status, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Start scheduler", start_scheduler,
                         visible=lambda _: not scheduler_running()),
        pystray.MenuItem("Stop scheduler",  stop_scheduler,
                         visible=lambda _: scheduler_running()),
        pystray.MenuItem("Run now",         run_now),
        pystray.MenuItem("View log",        view_log),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    icon = pystray.Icon(
        "meridian",
        icon=make_icon(),
        title="Meridian",
        menu=build_menu(),
    )
    icon.run()


if __name__ == "__main__":
    main()
