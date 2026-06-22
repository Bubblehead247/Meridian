# Phase 0 Summary — Project Infrastructure

**Status:** Complete
**Date:** 2026-06-22

Phase 0 establishes the skeleton every later phase builds on: package layout,
the estimator interface contract, config loading, experiment tracking, tests,
and CI. No research logic yet — just the controlled pipeline's frame.

---

## What was built

### 1. Package structure
A `meridian` package with one subpackage per module from the project spec, each
an importable package with a docstring:

    meridian/
      data/  features/  estimators/  deviations/  signals/
      regimes/  portfolio/  execution/  analytics/  visualization/  experiments/
    configs/   # YAML experiment configs
    reports/   # generated reports (git-ignored output)
    tests/

### 2. Estimator interface contract — `meridian/estimators/base.py`
`BaseEstimator` is an abstract base class (ABC) defining the exact API from
CLAUDE.md: `fit`, `update`, `predict_mean`, `residual`, `zscore`, `state`.
Exported as `from meridian.estimators import BaseEstimator`.

- All six methods are `@abstractmethod`, so an incomplete subclass cannot be
  instantiated (enforced by test).
- Regime logic is deliberately **excluded** from the contract, per spec — it
  lives in `meridian/regimes/` and is applied downstream.
- **No concrete estimators exist yet** — that is Phase 2. A throwaway
  `_ConstantMean` reference implementation lives only inside the test file to
  prove the contract is satisfiable.

### 3. Config loading — `meridian/config.py`
`load_config(path) -> dict` reads a YAML file (validates existence, non-empty,
mapping root). Single source of truth so every entry point loads configs
identically. Reproducibility principle: an experiment is reproducible from its
config alone.

- Skeleton config at `configs/example_experiment.yaml` fixes the intended shape
  (universe, in/out-of-sample windows, seed). Fields are consumed by later phases.

### 4. Experiment tracking — `meridian/tracking.py`
`start_run(experiment, config, ...)` context manager wrapping MLflow. Logs the
flattened config as run params. **Degrades to a no-op if MLflow is not
installed** (`is_available()` reports this), so early phases run without the
heavy dependency. Local store defaults to `./mlruns/` (git-ignored).

### 5. Tests — `tests/`
- `test_infrastructure.py` — package imports, all subpackages import, config
  round-trip + error cases, tracking is safe without MLflow.
- `test_base_estimator.py` — ABC cannot be instantiated, incomplete subclass
  rejected, complete subclass satisfies the full contract.
- **Result: 9 passed** locally.

### 6. Packaging & tooling
- `pyproject.toml` — setuptools build, `requires-python>=3.12`, full dependency
  set (yfinance, alpaca-py, numpy, pandas, scipy, statsmodels, scikit-learn,
  plotly, matplotlib, pyyaml, mlflow) + `[dev]` extras (pytest, pytest-cov,
  hypothesis, ruff). Pytest and ruff configured here.
- `requirements-dev.txt` → `-e .[dev]`.
- `.gitignore`, `.gitattributes` (LF normalization), `README.md`.

### 7. CI — `.github/workflows/ci.yml`
On push to `main` and all PRs: set up Python 3.12 → `pip install -e .[dev]` →
`ruff check` → `pytest --cov`. Single-version matrix (3.12) for now.

### 8. Version control
`git init` run; all 25 files staged. **Not yet committed** — left for the user
to make the initial commit.

---

## Interface contracts handed to later phases

```python
# Every estimator (Phase 2+) subclasses this:
from meridian.estimators import BaseEstimator   # ABC: fit/update/predict_mean/residual/zscore/state

# Config loading (all phases):
from meridian.config import load_config          # (path) -> dict

# Experiment tracking (all phases):
from meridian import tracking
with tracking.start_run("my-exp", config=cfg) as run:
    ...   # no-op if mlflow absent
```

---

## Known limitations / notes for Phase 1

- **Python version mismatch locally.** Spec and CI target 3.12; the dev machine
  runs **3.14.4**. Tests pass on 3.14, but CI is the source of truth at 3.12.
  Heavy deps (mlflow, alpaca-py) are **not installed locally** — only pyyaml,
  pandas, pytest were available, which is why `tracking` has a no-op fallback.
- **No data layer yet.** Phase 1 builds `meridian/data/` — yfinance ingestion,
  caching, universe management (S&P 500, Nasdaq-100, Russell 1000, SPY/QQQ/IWM).
- **Survivorship bias** (yfinance) remains an accepted, must-document limitation.
- `configs/example_experiment.yaml` is a placeholder; its `estimators`,
  `signal`, and `regime` fields stay empty until the relevant phases.

---

## Next phase

**Phase 1 — Data engineering:** universe management, yfinance ingestion, and a
caching layer, with the 2010–2019 / 2020–2022 / 2023+ split enforced.
