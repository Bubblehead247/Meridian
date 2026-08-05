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
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
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
    # Which configs/graduation_criteria.yaml version this strategy was first judged
    # under (bound on the first pipeline.graduation.advance() call). None until then.
    # See pipeline/graduation.py's module docstring (P1-C) for why this exists.
    criteria_version: str | None = None
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
    # Save counter, maintained by LedgerStore.save() (not by this class itself — it
    # has no persistence knowledge). Used for optimistic-locking: a save() based on a
    # ledger loaded before a concurrent writer's save() is detected and refused rather
    # than silently clobbering the concurrent write. See LedgerStore.save().
    version: int = 0

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


class StaleLedgerWrite(RuntimeError):
    """Raised when ``save()`` is called on a ledger older than what's on disk.

    Means a concurrent writer already saved a newer version since this ledger object
    was loaded — proceeding would silently clobber that write. Reload
    (``LedgerStore.load``) and re-apply the intended change, or pass ``force=True`` to
    ``save()`` if overwriting is genuinely intended (e.g. a deliberate manual repair).
    """


class LedgerStore:
    """JSON-backed store of strategy ledgers (one file per strategy under ``root``).

    ``save()`` writes atomically (temp file + rename — never leaves a half-written
    JSON file even on a crash mid-write) and appends every save to an append-only
    JSONL history log under ``root/history.jsonl`` (mirrors the pattern already used
    for ``experiments/run_log.py``), so the current-state overwrite-on-save file is no
    longer the only record — a reviewer can reconstruct how a ledger's stage/PnL/
    drawdown evolved, not just its latest snapshot. See
    research_integrity_gap_analysis.md §3.8/P2-D.
    """

    def __init__(self, root: str | Path = DEFAULT_LEDGER_DIR):
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def history_path(self) -> Path:
        return self.root / "history.jsonl"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def save(self, ledger: StrategyLedger, *, force: bool = False) -> Path:
        """Persist ``ledger``, bumping and stamping its save ``version``.

        Raises ``StaleLedgerWrite`` if the on-disk copy has a higher ``version`` than
        ``ledger`` (a concurrent writer saved after this object was loaded), unless
        ``force=True``. On success, mutates ``ledger.version`` in place so a caller
        that keeps using the same object can save again without reloading.
        """
        p = self.path(ledger.name)
        p.parent.mkdir(parents=True, exist_ok=True)

        on_disk = self.load(ledger.name)
        if on_disk is not None and on_disk.version > ledger.version and not force:
            raise StaleLedgerWrite(
                f"{ledger.name!r}: in-memory version {ledger.version} is behind the "
                f"on-disk version {on_disk.version} — reload before saving, or pass "
                f"force=True to overwrite deliberately."
            )
        ledger.version = (on_disk.version if on_disk is not None else 0) + 1

        payload = json.dumps(ledger.to_dict(), indent=2)
        fd, tmp_name = tempfile.mkstemp(dir=p.parent, prefix=f".{ledger.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, p)  # atomic on POSIX and Windows
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

        with self.history_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "name": ledger.name, "version": ledger.version, "ledger": ledger.to_dict(),
            }) + "\n")
        return p

    def load(self, name: str) -> StrategyLedger | None:
        p = self.path(name)
        if not p.exists():
            return None
        return StrategyLedger.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))  # history.jsonl doesn't match

    def all(self) -> dict[str, StrategyLedger]:
        return {name: self.load(name) for name in self.list()}


# --- module-level convenience (default store) --------------------------------

def save_ledger(ledger: StrategyLedger, store: LedgerStore | None = None) -> Path:
    return (store or LedgerStore()).save(ledger)


def load_ledger(name: str, store: LedgerStore | None = None) -> StrategyLedger | None:
    return (store or LedgerStore()).load(name)


def list_ledgers(store: LedgerStore | None = None) -> list[str]:
    return (store or LedgerStore()).list()
