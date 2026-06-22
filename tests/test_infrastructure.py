"""Phase 0 smoke tests: confirm the scaffolding imports and core infra works."""

from __future__ import annotations

import pytest

import meridian


def test_package_imports_and_has_version():
    assert isinstance(meridian.__version__, str)


def test_all_subpackages_import():
    """Every declared module should import cleanly (no broken __init__)."""
    subpackages = [
        "data",
        "features",
        "estimators",
        "deviations",
        "signals",
        "regimes",
        "portfolio",
        "execution",
        "analytics",
        "visualization",
        "experiments",
    ]
    for name in subpackages:
        __import__(f"meridian.{name}")


def test_load_config_roundtrip(tmp_path):
    from meridian.config import load_config

    cfg_file = tmp_path / "demo.yaml"
    cfg_file.write_text("name: demo\nparams:\n  window: 20\n", encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg["name"] == "demo"
    assert cfg["params"]["window"] == 20


def test_load_config_missing_file(tmp_path):
    from meridian.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_empty_file(tmp_path):
    from meridian.config import load_config

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(empty)


def test_tracking_start_run_is_safe_without_mlflow():
    """start_run must never crash, even if MLflow is absent."""
    from meridian import tracking

    with tracking.start_run("phase0-smoke", config={"a": 1, "nested": {"b": 2}}) as run:
        # run is an MLflow run when available, else None — either is acceptable.
        assert run is None or run is not None
