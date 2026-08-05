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


#: Run kinds that represent an estimator search (as opposed to e.g. a graduation
#: criteria override) and therefore contribute to cumulative trial counting below.
_SEARCH_KINDS = ("validation", "universe_validation")


def cumulative_trial_count(
    path: str | Path = DEFAULT_RUN_LOG, *, scope: object = None,
) -> int:
    """Count of distinct (estimator, deviation, window) trials ever logged.

    DSR and the BH/Bonferroni correction (``validation/pipeline.py``,
    ``portfolio/validation.py``) only know about the estimators passed into a single
    ``validate()``/``validate_universe()`` call — repeating that call across separate
    research sessions faces no additional multiple-testing penalty, because nothing
    tracked how many distinct trials had already been run against the same data. This
    reads the append-only run log and counts the union of distinct trials across every
    matching logged run, so a caller can see the *true* cumulative search size instead
    of just the current call's ``len(estimators)`` — see
    research_integrity_gap_analysis.md §3.5/P1-A.

    This is deliberately a narrower, single-number complement to full research-lineage
    tracking (parent experiments, hypotheses, promotion decisions, etc.) — that is a
    separate, larger project (the gap analysis's Priority 1 item), not something this
    function attempts.

    Args:
        path: Run log to read.
        scope: When given, only runs whose logged ``meta["scope"]`` matches exactly
            (JSON-comparable equality, e.g. a sorted symbol list) are counted — so a
            SPY-only research line isn't inflated by an unrelated universe's trials.
            When None, every logged run counts (an upper bound across all research,
            not scoped to one dataset).

    Returns:
        The number of distinct (estimator, deviation, window) tuples across all
        matching runs. 0 if the log is empty/missing or no run recorded an
        ``estimators`` list in its meta (older logs predate this field).
    """
    trials: set[tuple[str, str | None, object]] = set()
    for rec in read_runs(path):
        if rec.kind not in _SEARCH_KINDS:
            continue
        meta = rec.meta or {}
        if scope is not None and meta.get("scope") != scope:
            continue
        estimators = meta.get("estimators")
        if not estimators:
            continue
        deviation = meta.get("deviation")
        window = meta.get("window")
        for est in estimators:
            trials.add((est, deviation, window))
    return len(trials)
