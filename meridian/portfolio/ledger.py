"""Per-strategy virtual accounting ledger tracking PnL, positions, and drawdown.

This module owns per-strategy virtual accounting (load/update/save of StrategyLedger);
it does NOT own allocation policy (that stays in portfolio/allocation.py).

One brokerage account backs many *virtual* ledgers — one per strategy/sleeve — whose
capital allocations sum to the account. Each ledger carries its graduation stage (the
state the graduation pipeline reads/advances), its PnL and drawdown, and the
externally-computed performance/risk fields (win rate, expectancy, turnover, sleeve
correlations, risk contribution) set from the scorecard and risk budget. Ledgers persist
as one JSON file per strategy under ``ledger/`` (override via ``LedgerStore``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

#: The graduation stages a strategy moves through (PLAN.md §6). ``research`` is the start.
STAGES: tuple[str, ...] = (
    "research", "backtest", "walk_forward", "oos", "paper",
    "pilot", "proven", "core", "elite", "retired",
)

DEFAULT_LEDGER_DIR = Path("ledger")


def _today() -> str:
    return date.today().isoformat()


@dataclass
class StrategyLedger:
    """Virtual accounting record for one strategy/sleeve."""

    name: str
    family: str
    stage: str = "research"
    stage_entered: str = field(default_factory=_today)
    capital_alloc: float = 0.0
    open_positions: list[dict] = field(default_factory=list)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    drawdown_cur: float = 0.0
    drawdown_max: float = 0.0
    # None = not yet measured (JSON-clean null, distinct from a real 0.0)
    win_rate: float | None = None
    expectancy: float | None = None
    turnover: float | None = None
    correlations: dict[str, float] = field(default_factory=dict)
    risk_contribution: float = 0.0
    # internal counters / bookkeeping (persisted)
    n_trades: int = 0
    n_wins: int = 0
    equity_peak: float = 0.0

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}. Known: {', '.join(STAGES)}")
        if self.equity_peak == 0.0:
            self.equity_peak = self.capital_alloc

    # --- derived ----------------------------------------------------------

    @property
    def equity(self) -> float:
        """Virtual equity = allocated capital + realized + unrealized PnL."""
        return self.capital_alloc + self.realized_pnl + self.unrealized_pnl

    def _recompute_drawdown(self) -> None:
        eq = self.equity
        self.equity_peak = max(self.equity_peak, eq)
        self.drawdown_cur = (eq / self.equity_peak - 1.0) if self.equity_peak > 0 else 0.0
        self.drawdown_max = min(self.drawdown_max, self.drawdown_cur)

    # --- updates ----------------------------------------------------------

    def record_trade(self, pnl: float) -> None:
        """Book a closed trade's realized PnL and refresh win rate / expectancy / drawdown."""
        self.realized_pnl += float(pnl)
        self.n_trades += 1
        if pnl > 0:
            self.n_wins += 1
        self.win_rate = self.n_wins / self.n_trades
        self.expectancy = self.realized_pnl / self.n_trades
        self._recompute_drawdown()

    def mark(self, unrealized_pnl: float) -> None:
        """Update open-position mark-to-market PnL and refresh drawdown."""
        self.unrealized_pnl = float(unrealized_pnl)
        self._recompute_drawdown()

    def update_metrics(
        self,
        *,
        turnover: float | None = None,
        correlations: dict[str, float] | None = None,
        risk_contribution: float | None = None,
    ) -> None:
        """Set externally-computed fields (from the scorecard / risk budget)."""
        if turnover is not None:
            self.turnover = float(turnover)
        if correlations is not None:
            self.correlations = dict(correlations)
        if risk_contribution is not None:
            self.risk_contribution = float(risk_contribution)

    def set_stage(self, stage: str, *, on: str | None = None) -> None:
        """Move to a graduation ``stage`` and stamp the entry date."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}. Known: {', '.join(STAGES)}")
        self.stage = stage
        self.stage_entered = on or _today()

    # --- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StrategyLedger:
        return cls(**d)


class LedgerStore:
    """JSON-backed store of strategy ledgers (one file per strategy under ``root``)."""

    def __init__(self, root: str | Path = DEFAULT_LEDGER_DIR):
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def save(self, ledger: StrategyLedger) -> Path:
        p = self.path(ledger.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ledger.to_dict(), indent=2), encoding="utf-8")
        return p

    def load(self, name: str) -> StrategyLedger | None:
        p = self.path(name)
        if not p.exists():
            return None
        return StrategyLedger.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))

    def all(self) -> dict[str, StrategyLedger]:
        return {name: self.load(name) for name in self.list()}


# --- module-level convenience (default store) --------------------------------

def save_ledger(ledger: StrategyLedger, store: LedgerStore | None = None) -> Path:
    return (store or LedgerStore()).save(ledger)


def load_ledger(name: str, store: LedgerStore | None = None) -> StrategyLedger | None:
    return (store or LedgerStore()).load(name)


def list_ledgers(store: LedgerStore | None = None) -> list[str]:
    return (store or LedgerStore()).list()
