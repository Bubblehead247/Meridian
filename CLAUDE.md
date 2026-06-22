# CLAUDE.md — Meridian Research Platform

## Project Overview

Meridian is an institutional-grade quantitative research platform for testing, validating, and ranking mean-reversion strategies across multiple definitions of "fair value." The core research question is: which estimator of fair value produces the most robust, exploitable mean-reversion signal?

This is not a single-strategy backtest. It is a comparative research framework testing 37+ estimators through a shared, controlled pipeline.

---

## Stack

- **Python:** 3.12
- **Data:** yfinance, alpaca-py
- **Numerics:** numpy, pandas
- **Stats:** scipy, statsmodels
- **ML:** scikit-learn (PyTorch deferred to Phase 2 neural net estimators)
- **Testing:** pytest (primary), hypothesis (estimator math validation)
- **Visualization:** plotly, matplotlib
- **Config:** YAML via pyyaml
- **Experiment tracking:** MLflow

---

## Repo Structure

    meridian/
    ├── data/               # Data ingestion, caching, universe management
    ├── features/           # Feature engineering
    ├── estimators/         # All fair-value estimators
    ├── deviations/         # Deviation metrics (z-score, ATR-norm, etc.)
    ├── signals/            # Entry/exit signal logic
    ├── regimes/            # Market regime classifiers (separate module)
    ├── portfolio/          # Position sizing and portfolio construction
    ├── execution/          # Order execution and paper trading
    ├── analytics/          # Performance metrics
    ├── visualization/      # Charts, heatmaps, dashboards
    ├── experiments/        # Experiment configs and runners
    ├── reports/            # Generated markdown/PDF reports
    ├── tests/              # All tests
    └── configs/            # YAML config files

---

## Estimator Interface Contract

Every estimator must implement this exact API. No exceptions.

```python
class BaseEstimator:
    def fit(self, prices: pd.Series) -> None:
        """Fit estimator to historical price data."""

    def update(self, price: float) -> None:
        """Incremental update with a single new price."""

    def predict_mean(self) -> float:
        """Return current fair-value estimate."""

    def residual(self, price: float) -> float:
        """Raw difference: price - predicted_mean."""

    def zscore(self, price: float) -> float:
        """Standardized deviation from fair value."""

    def state(self) -> dict:
        """Return full internal state for reproducibility."""
```

Regime classification is **not** part of the estimator interface.
Regime logic lives in `regimes/` and is applied downstream.

---

## Core Design Principles

- **No estimator-specific optimization.** Signal logic is held constant across all estimators. Only the estimator varies.
- **Reproducibility first.** Every experiment must be fully reproducible from config alone.
- **Out-of-sample discipline.** In-sample: 2010–2019. OOS: 2020–2022. Walk-forward: 2023–present.
- **Multiple testing awareness.** 37+ estimators on shared data is a significant multiple comparison problem. All rankings must apply corrections (Bonferroni or BH).
- **Survivorship bias acknowledged.** yfinance is survivorship-biased. Document this limitation in every report. Do not overstate robustness.

---

## Data Sources

- **yfinance** — primary historical OHLCV, daily bars
- **alpaca-py** — live and paper trading execution, real-time data

Universes in scope: S&P 500, Nasdaq-100, Russell 1000, SPY, QQQ, IWM

---

## Phases

| Phase | Deliverable |
|-------|-------------|
| 0 | Project infrastructure, experiment tracking, CI |
| 1 | Data engineering, universe management, caching |
| 2 | Estimator library (37+ estimators, unit tested) |
| 3 | Deviation metrics module |
| 4 | Signal engine, backtester |
| 5 | Regime classifiers |
| 6 | Validation engine (WFO, Monte Carlo, bootstrap) |
| 7 | Analytics and reporting |
| 8 | Adaptive meta-model and ensemble |
| 9 | Paper trading deployment |
| 10 | Production packaging |

Each phase produces a `phase_N_summary.md` documenting what was built, interface contracts, and known limitations. This summary is the primary input context for the next phase.

---

## Testing Standards

- Every estimator must have a unit test covering `fit()`, `update()`, `predict_mean()`, `residual()`, `zscore()`, and `state()`
- hypothesis used for mathematical property validation (e.g. z-score mean ≈ 0 over large normal samples)
- No phase is considered complete until all tests pass

---

## Known Limitations

- yfinance data is survivorship-biased — delisted companies are absent
- PyTorch not yet integrated — neural net and autoencoder estimators deferred
- Transaction costs modeled as flat bps — real slippage will vary

---

## Session Convention

Each Claude Code session must:

1. Load this file first
2. Load the most recent `phase_N_summary.md`
3. Work only within the current phase scope
4. End by writing or updating `phase_N_summary.md`
