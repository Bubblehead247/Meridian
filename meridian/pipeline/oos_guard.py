"""OOS-holdout run counter.

This module owns persisting how many times a (family, model, symbol) has been
run through the OOS stage; it does NOT own the OOS math or the pass/fail
decision (those stay in pipeline/oos.py).

The fixed OOS holdout is meant to be evaluated once per strategy (pipeline/oos.py's
own docstring says so), but until now nothing enforced or even recorded that
discipline — a researcher could re-run OOS an unlimited number of times against the
same holdout and only the eventual pass would survive in saved_strategies/. This
guard is intentionally non-blocking: it counts and flags rather than refusing a run
(a hard refusal would also break legitimate re-runs, e.g. a bug fix). The recorded
count is the signal a reviewer needs — see it on the StageResult and on any saved
StrategyRecord — to judge whether a "pass" followed a single evaluation or the tenth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

DEFAULT_GUARD_DIR = Path("ledger") / "oos_runs"


@dataclass
class OOSRunRecord:
    """Run-count record for one (family, model, symbol) against the OOS stage."""

    family: str
    model: str
    symbol: str
    run_count: int = 0
    first_run: str | None = None
    last_run: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OOSRunRecord:
        return cls(**d)


class OOSGuard:
    """JSON-backed run counter, one file per (family, model, symbol)."""

    def __init__(self, root: str | Path | None = None):
        # Resolved at call time (not bound as a default-argument value) so tests can
        # monkeypatch the module-level DEFAULT_GUARD_DIR and have it actually take
        # effect — see tests/conftest.py's ledger-redirection fixture.
        self.root = Path(root) if root is not None else DEFAULT_GUARD_DIR

    def _path(self, family: str, model: str, symbol: str) -> Path:
        return self.root / f"{family}__{model}__{symbol}.json"

    def record_run(
        self, family: str, model: str, symbol: str, *, on: str | None = None
    ) -> OOSRunRecord:
        """Increment and persist the run count for this (family, model, symbol)."""
        on = on or date.today().isoformat()
        p = self._path(family, model, symbol)
        if p.exists():
            rec = OOSRunRecord.from_dict(json.loads(p.read_text(encoding="utf-8")))
            rec.run_count += 1
            rec.last_run = on
        else:
            rec = OOSRunRecord(
                family=family, model=model, symbol=symbol,
                run_count=1, first_run=on, last_run=on,
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
        return rec

    def peek(self, family: str, model: str, symbol: str) -> OOSRunRecord | None:
        """Read the current count without incrementing it, or None if never run."""
        p = self._path(family, model, symbol)
        if not p.exists():
            return None
        return OOSRunRecord.from_dict(json.loads(p.read_text(encoding="utf-8")))
