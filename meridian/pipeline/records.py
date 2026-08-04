"""Persistent records for strategies that clear the graduation metric bar.

This module owns saving and loading StrategyRecord snapshots; it does NOT own
graduation logic (that stays in pipeline/graduation.py).

A record captures the family, model, symbol, which stage passed, the full
scorecard from that stage, and a ledger snapshot — everything needed to
review a validated strategy later without re-running the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

DEFAULT_RECORDS_DIR = Path("saved_strategies")


@dataclass
class StrategyRecord:
    """Snapshot of a strategy that cleared the metric bar on at least one stage."""

    family: str
    model: str
    symbol: str
    stage_passed: str   # highest stage that returned passed=True
    saved_at: str       # ISO date (YYYY-MM-DD)
    scorecard: dict     # full scorecard from that stage
    ledger: dict        # ledger.to_dict() at the moment of save
    oos_run_count: int = 0   # >1 means this OOS pass followed prior attempts (see oos_guard.py)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StrategyRecord:
        d = dict(d)
        sc = d.get("scorecard")
        if isinstance(sc, dict) and "slippage_sensitivity" in sc:
            sc = dict(sc)
            sc["slippage_sensitivity"] = {int(k): v for k, v in sc["slippage_sensitivity"].items()}
            d["scorecard"] = sc
        return cls(**d)


def _filename(rec: StrategyRecord) -> str:
    return f"{rec.symbol}_{rec.family}_{rec.model}_{rec.stage_passed}_{rec.saved_at}.json"


def save_record(
    record: StrategyRecord,
    save_dir: str | Path | None = None,
) -> Path:
    """Write ``record`` as JSON under ``save_dir`` (default: ``saved_strategies/``)."""
    root = Path(save_dir) if save_dir else DEFAULT_RECORDS_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / _filename(record)
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def load_records(save_dir: str | Path | None = None) -> list[StrategyRecord]:
    """Return all saved records, newest first. Skips corrupt files silently."""
    root = Path(save_dir) if save_dir else DEFAULT_RECORDS_DIR
    if not root.exists():
        return []
    out = []
    for p in root.glob("*.json"):
        try:
            out.append(StrategyRecord.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            pass
    return sorted(out, key=lambda r: r.saved_at, reverse=True)


def record_from_pipeline(
    family: str,
    model_name: str,
    symbol: str,
    results: dict,
    ledger,
) -> StrategyRecord | None:
    """Build a StrategyRecord from pipeline results if any stage passed; else None.

    ``results`` is the ``{stage_name: StageResult}`` dict returned by ``run_pipeline``.
    Uses the scorecard of the last stage that passed (highest evidence level).
    """
    last_passing = None
    for stage, res in results.items():
        if res.passed:
            last_passing = (stage, res)
    if last_passing is None:
        return None
    stage_name, stage_result = last_passing
    oos_result = results.get("oos")
    oos_run_count = oos_result.detail.get("oos_run_count", 0) if oos_result is not None else 0
    return StrategyRecord(
        family=family,
        model=model_name,
        symbol=symbol,
        stage_passed=stage_name,
        saved_at=date.today().isoformat(),
        scorecard=stage_result.scorecard,
        ledger=ledger.to_dict() if ledger is not None else {},
        oos_run_count=oos_run_count,
    )
