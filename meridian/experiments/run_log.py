"""Append-only experiment run log.

This module owns recording that a validation run happened — what config, what
code/data version, and where its report landed; it does NOT own running the
experiment or writing the report itself (those stay in experiments/runner.py).

Every ``meridian validate``/``meridian universe`` run overwrote a single fixed
report path with no run ID, code version, or data snapshot captured anywhere —
so "reproducible from config alone" had no record of *what was actually run,
when*. This closes that gap the same minimal way ``pipeline/records.py`` and
``portfolio/ledger.py`` already do: one append-only JSON-lines file, no
database. ``reports/run_log.jsonl`` accumulates one line per run; nothing here
is ever overwritten or deleted.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RUN_LOG = Path("reports") / "run_log.jsonl"


def _git_sha() -> str | None:
    """Best-effort current commit hash, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def config_sha256(raw_yaml_text: str) -> str:
    """Hash of the raw config text — detects a config edited between runs."""
    return hashlib.sha256(raw_yaml_text.encode("utf-8")).hexdigest()


@dataclass
class RunRecord:
    """One logged run: what was run, against what config/code/data, and the result."""

    run_id: str
    timestamp: str
    config_path: str
    config_sha256: str | None
    git_sha: str | None
    kind: str                 # "validation" | "universe_validation"
    meta: dict = field(default_factory=dict)   # universe/symbols/date-range/estimators
    report_path: str | None = None
    summary: dict = field(default_factory=dict)  # e.g. {"n_significant": 0, "n_tested": 42}

    def to_dict(self) -> dict:
        return asdict(self)


def new_run_record(
    *,
    config_path: str | Path,
    raw_config_text: str | None,
    kind: str,
    meta: dict,
    report_path: str | Path | None,
    summary: dict,
) -> RunRecord:
    """Build a RunRecord for the run about to be logged."""
    return RunRecord(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_path=str(config_path),
        config_sha256=config_sha256(raw_config_text) if raw_config_text else None,
        git_sha=_git_sha(),
        kind=kind,
        meta=meta,
        report_path=str(report_path) if report_path else None,
        summary=summary,
    )


def append_run(record: RunRecord, path: str | Path = DEFAULT_RUN_LOG) -> Path:
    """Append ``record`` as one JSON line — never overwrites prior runs.

    ``meta`` comes from an arbitrary config's YAML values (e.g. unquoted YAML
    dates parse as ``datetime.date``, not ``str``), so this serializes with
    ``default=str`` rather than assuming every value is already JSON-native.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), default=str) + "\n")
    return p


def read_runs(path: str | Path = DEFAULT_RUN_LOG) -> list[RunRecord]:
    """Read every logged run, oldest first. Skips corrupt lines silently."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[RunRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(RunRecord(**json.loads(line)))
        except Exception:
            pass
    return out
