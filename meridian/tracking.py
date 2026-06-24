"""Experiment tracking via MLflow.

Wraps MLflow so experiments are logged consistently across the platform. The
goal is reproducibility: every run records the config that produced it, so a
result can always be traced back to the exact inputs.

MLflow is optional at import time — if it is not installed, the helpers degrade
to no-ops so that early phases (which may not log anything yet) still run.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:  # MLflow is a heavy dependency; tolerate its absence in minimal installs.
    import mlflow

    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover - exercised only without mlflow
    _HAS_MLFLOW = False


DEFAULT_TRACKING_DIR = Path("mlruns")


def is_available() -> bool:
    """Return True if MLflow is installed and tracking can be used."""
    return _HAS_MLFLOW


@contextlib.contextmanager
def start_run(
    experiment: str,
    config: dict[str, Any] | None = None,
    run_name: str | None = None,
    tracking_dir: str | Path = DEFAULT_TRACKING_DIR,
) -> Iterator[Any]:
    """Context manager for a single tracked experiment run.

    Logs the full config as params so the run is reproducible from its record
    alone. If MLflow is not installed, yields None and does nothing else, so
    callers can wrap work unconditionally.

    Args:
        experiment: MLflow experiment name to group runs under.
        config: Config dict to record as run params (flattened).
        run_name: Optional human-readable name for the run.
        tracking_dir: Local directory backing the MLflow store.

    Yields:
        The active MLflow run, or None when MLflow is unavailable.
    """
    if not _HAS_MLFLOW:
        yield None
        return

    mlflow.set_tracking_uri(Path(tracking_dir).resolve().as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        if config:
            for key, value in _flatten(config).items():
                mlflow.log_param(key, value)
        yield run


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted keys for param logging."""
    flat: dict[str, Any] = {}
    for key, value in d.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{full}."))
        else:
            flat[full] = value
    return flat
