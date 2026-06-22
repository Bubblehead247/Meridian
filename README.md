# Meridian

A quantitative research platform for testing, validating, and ranking
**mean-reversion strategies** across many definitions of "fair value."

The core research question: *which estimator of fair value produces the most
robust, exploitable mean-reversion signal?* Meridian is **not** a single-strategy
backtest. It is a comparative framework that runs 37+ fair-value estimators
through one shared, controlled pipeline so that only the estimator varies.

## Quick start

```bash
# Python 3.12 is the supported version.
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[dev]

pytest                            # run the test suite
ruff check meridian tests         # lint
```

## Layout

| Path | Purpose |
|------|---------|
| `meridian/data/` | Data ingestion, caching, universe management |
| `meridian/features/` | Feature engineering |
| `meridian/estimators/` | Fair-value estimators (all implement `BaseEstimator`) |
| `meridian/deviations/` | Deviation metrics (z-score, ATR-normalized, …) |
| `meridian/signals/` | Entry/exit signal logic (held constant across estimators) |
| `meridian/regimes/` | Market regime classifiers (applied downstream) |
| `meridian/portfolio/` | Position sizing and portfolio construction |
| `meridian/execution/` | Order execution and paper trading |
| `meridian/analytics/` | Performance metrics |
| `meridian/visualization/` | Charts, heatmaps, dashboards |
| `meridian/experiments/` | Experiment configs and runners |
| `configs/` | YAML experiment configs |
| `reports/` | Generated reports |
| `tests/` | Test suite |

## Core principles

- **No estimator-specific optimization.** Signal logic is identical for every
  estimator; only the estimator changes.
- **Reproducibility first.** Every experiment is reproducible from its config
  alone. Runs are tracked with MLflow.
- **Out-of-sample discipline.** In-sample 2010–2019, OOS 2020–2022,
  walk-forward 2023–present.
- **Multiple-testing awareness.** 37+ estimators on shared data is a serious
  multiple-comparison problem; rankings apply corrections (Bonferroni or BH).

## Known limitations

- **Survivorship bias.** yfinance omits delisted companies; every report must
  document this.
- **PyTorch deferred.** Neural-net and autoencoder estimators come in Phase 2.
- **Flat transaction costs.** Modeled as flat basis points; real slippage varies.

## Development phases

Work proceeds in phases (0–10). Each phase ends by writing a
`phase_N_summary.md` that is the primary input for the next phase. This repo is
currently at **Phase 0 (infrastructure)** — see `phase_0_summary.md`.
