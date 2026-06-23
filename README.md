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

## Running (CLI)

Everything is driven from a YAML config — a run is reproducible from the config
alone. Installing the package adds a `meridian` command:

```bash
meridian list estimators                       # 42 estimators (incl. ensembles)
meridian validate configs/validation_spy.yaml  # walk-forward + significance -> report
meridian backtest configs/validation_spy.yaml  # ranked table, no report file
meridian paper    configs/paper_spy.yaml        # paper-trading dry-run
```

- **validate** runs anchored walk-forward over the config's estimators, applies
  bootstrap + Monte-Carlo + multiple-testing correction, and writes a markdown
  report (with the mandatory survivorship / multiple-testing disclosures) to
  `report.path`.
- **paper** warms the indicators on the first half of history and replays the
  rest through a `SimulatedBroker`. For **live** paper trading, set
  `broker.type: alpaca` in the config, export `ALPACA_API_KEY` /
  `ALPACA_SECRET_KEY`, and drive `PaperTrader.on_bar` from a data feed (the loop
  reuses the identical causal logic as the backtester, so live == backtest).
  `PaperTrader.save_checkpoint` / `load_checkpoint` persist the signal state so a
  session can resume after a restart.

See `configs/validation_spy.yaml` and `configs/paper_spy.yaml` for the full
schema.

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

Work proceeded in phases (0–10), each ending with a `phase_N_summary.md`. **All
ten phases are complete**: infrastructure, data engineering, the 38-estimator
library (+ ensembles), deviation metrics, the signal engine and backtester,
regime classifiers, the validation engine (walk-forward + bootstrap +
Monte-Carlo + multiple-testing correction), analytics and reporting, the adaptive
meta-model, paper-trading deployment, and this config-driven packaging. See the
`phase_N_summary.md` files for each phase's contract and limitations.

**Headline research finding:** on SPY, no fair-value estimator shows a
statistically significant mean-reversion edge out-of-sample once corrected for
multiple testing — the honest result the platform is built to establish rather
than obscure.
