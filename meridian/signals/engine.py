"""Signal engine: deviation score -> position.

This is the **constant** entry/exit logic the whole platform shares. Per the
core principle, signal logic never changes across estimators or deviation
metrics — only the estimator and metric vary. That is what makes the comparison
fair: any performance difference is attributable to the fair-value estimator,
not to bespoke trade rules.

The logic is textbook mean reversion: when price sits far *below* fair value
(deviation score very negative), go long expecting reversion up; far *above*
(score very positive), go short. Close the position when the score reverts back
through the exit level (fair value, by default).

Scores are read at each bar's close; the resulting position is applied to the
*next* bar's return by the backtester, so there is no look-ahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    """Parameters of the shared mean-reversion rule (in deviation-score units).

    Args:
        entry_threshold: Enter when the score is at least this far from zero
            (long if score <= -entry, short if score >= +entry). Must be > 0.
        exit_threshold: Signed score at which to close. Default 0.0 = exit at
            fair value. Negative exits early (before full reversion); positive
            waits for overshoot. Long exits when score >= exit_threshold; short
            exits when score <= -exit_threshold.
        stop_threshold: Optional score level (> 0) for a stop-loss: close a long
            if score <= -stop_threshold (deviation widened against us), a short
            if score >= +stop_threshold. None disables.
        max_holding: Optional cap on bars held before forced exit. None disables.
        allow_short: If False, only long positions are taken (long-only assets).
    """

    entry_threshold: float = 2.0
    exit_threshold: float = 0.0
    stop_threshold: float | None = None
    max_holding: int | None = None
    allow_short: bool = True

    def __post_init__(self):
        if self.entry_threshold <= 0:
            raise ValueError("entry_threshold must be > 0")
        if self.stop_threshold is not None and self.stop_threshold <= 0:
            raise ValueError("stop_threshold must be > 0 or None")

    @classmethod
    def from_config(cls, signal_cfg: dict) -> SignalConfig:
        """Build from an experiment config's ``signal`` section (missing keys
        fall back to defaults)."""
        return cls(
            entry_threshold=float(signal_cfg.get("entry_threshold", 2.0)),
            exit_threshold=float(signal_cfg.get("exit_threshold", 0.0)),
            stop_threshold=(
                None if signal_cfg.get("stop_threshold") is None
                else float(signal_cfg["stop_threshold"])
            ),
            max_holding=(
                None if signal_cfg.get("max_holding") is None
                else int(signal_cfg["max_holding"])
            ),
            allow_short=bool(signal_cfg.get("allow_short", True)),
        )


class SignalState:
    """The mean-reversion state machine for one instrument.

    Holds the running position and processes one score at a time via `step`.
    Both the backtester (`generate_positions`) and the live paper trader drive
    this same class, so a live session and a backtest can never diverge in their
    trade logic — they share this exact code.
    """

    def __init__(self, config: SignalConfig | None = None):
        self.cfg = config or SignalConfig()
        self.position = 0
        self.bars_held = 0
        # After a stop-out, block re-entry on that side until the score reverts
        # to neutral — otherwise a still-extreme score would re-enter next bar
        # and the stop would be meaningless.
        self.blocked_long = False
        self.blocked_short = False

    def to_dict(self) -> dict:
        """Serialize the path-dependent state (for checkpoint/resume)."""
        return {
            "position": self.position,
            "bars_held": self.bars_held,
            "blocked_long": self.blocked_long,
            "blocked_short": self.blocked_short,
        }

    def load_dict(self, d: dict) -> None:
        """Restore state previously produced by `to_dict`."""
        self.position = int(d["position"])
        self.bars_held = int(d["bars_held"])
        self.blocked_long = bool(d["blocked_long"])
        self.blocked_short = bool(d["blocked_short"])

    def step(self, score: float) -> int:
        """Advance the state with one score and return the new position."""
        cfg = self.cfg
        s = float(score)
        if math.isnan(s):
            if self.position != 0:
                self.bars_held += 1
            return self.position

        if self.blocked_long and s >= cfg.exit_threshold:
            self.blocked_long = False
        if self.blocked_short and s <= -cfg.exit_threshold:
            self.blocked_short = False

        if self.position == 0:
            if s <= -cfg.entry_threshold and not self.blocked_long:
                self.position, self.bars_held = 1, 0
            elif cfg.allow_short and s >= cfg.entry_threshold and not self.blocked_short:
                self.position, self.bars_held = -1, 0
        elif self.position == 1:
            self.bars_held += 1
            if cfg.stop_threshold is not None and s <= -cfg.stop_threshold:
                self.position, self.bars_held, self.blocked_long = 0, 0, True
            elif s >= cfg.exit_threshold or (
                cfg.max_holding is not None and self.bars_held >= cfg.max_holding
            ):
                self.position, self.bars_held = 0, 0
        else:  # position == -1
            self.bars_held += 1
            if cfg.stop_threshold is not None and s >= cfg.stop_threshold:
                self.position, self.bars_held, self.blocked_short = 0, 0, True
            elif s <= -cfg.exit_threshold or (
                cfg.max_holding is not None and self.bars_held >= cfg.max_holding
            ):
                self.position, self.bars_held = 0, 0

        return self.position


def generate_positions(scores: pd.Series, config: SignalConfig | None = None) -> pd.Series:
    """Turn a stream of deviation scores into a position path of {-1, 0, +1}.

    Stateful and path-dependent: a position persists until an exit/stop/timeout
    condition fires. A ``NaN`` score (warmup, or undefined scale) takes no
    action — an open position is held, a flat book stays flat.

    Args:
        scores: Deviation scores indexed by time (e.g. from a deviation metric).
        config: Signal parameters; defaults if None.

    Returns:
        Position series aligned to ``scores``: +1 long, -1 short, 0 flat. The
        position at bar t reflects the decision made from the score at bar t;
        the backtester lags it one bar before applying returns.
    """
    state = SignalState(config)
    out = [state.step(s) for s in scores.to_numpy(dtype=float)]
    return pd.Series(out, index=scores.index, name="position", dtype=int)
