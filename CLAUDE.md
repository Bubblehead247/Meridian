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

---

## Expanded Mandate

Meridian is the research-and-development platform for a diversified rules-based retail quant fund, operating as part of the **Fable 5 trading system**. The original mean-reversion estimator research remains the foundation. On top of it, Meridian now hosts nine strategy families, a graduation pipeline, a virtual ledger, a risk budget engine, and a monthly review process — everything needed to run a systematic multi-strategy fund from idea to live capital.

---

## Modules

| Module | Responsibility |
|---|---|
| `data/` | OHLCV ingestion, caching, universe management |
| `estimators/` | 38 fair-value estimators + registry |
| `deviations/` | Deviation metrics (z-score, ATR-norm, etc.) |
| `signals/` | Entry/exit signal logic and backtester |
| `regimes/` | Market regime classifiers; `labeler.py` owns daily 3-dim index-level labels |
| `validation/` | Walk-forward, OOS splits, bootstrap, Monte Carlo, statistical corrections |
| `analytics/` | Performance metrics (CAGR, Sharpe, Sortino, Calmar, drawdown, profit factor) |
| `portfolio/` | Position sizing, construction, ledger, allocation, correlation, risk budget, monthly review |
| `execution/` | Order execution and paper trading via alpaca-py |
| `visualization/` | Charts, heatmaps, dashboards |
| `families/` | Multi-model layer: wraps existing est+dev+signal into named strategy models per family |
| `pipeline/` | Stage-runners (backtest, walk-forward, OOS) + graduation state machine + research records |
| `scoring/` | Per-strategy scorecard: wraps analytics/metrics and adds regime splits, sleeve correlation |
| `reporting/` | Renders monthly review output to markdown under `reports/` |

---

## Strategy Families

| Family | Status | Holding Period |
|---|---|---|
| Mean reversion | Active | 2–5 days |
| Trend following | Active — scaffold | 1–3 months |
| Momentum | Active — scaffold | 1–3 weeks |
| Breakouts | Active — scaffold | 1–3 weeks |
| Pullback continuation | Active — scaffold | 1–3 weeks |
| Sector rotation | Active — scaffold | 1–3 months |
| Long-term ETF | Active — scaffold | 6–12+ months |
| Event-driven | Deferred — stub only | TBD |
| Volatility | Deferred — stub only | TBD |

---

## Regime Labeling

Three dimensions computed daily, stored as a DataFrame indexed by date.

- **Trend** (SPY/IWM): `Bull` = close > 200-day MA AND ADX > 25; `Neutral` = close near 200MA AND ADX 15–25; `Bear` = close < 200MA.
- **Volatility** (^VIX close): `Low` < 15; `Normal` 15–20; `Elevated` 20–30; `Extreme` > 30.
- **Breadth** (% of S&P 500 constituents above their 200-day MA): `Expansion` > 60%; `Neutral` 40–60%; `Contraction` < 40%.

Every backtest bar, walk-forward window, and live signal is left-joined to the regime frame on date.

---

## Graduation Pipeline

| Stage | Live Capital |
|---|---|
| Research | 0% |
| Backtest | 0% |
| Walk-forward | 0% |
| OOS holdout | 0% |
| Paper | 0% |
| Pilot | 1–3% |
| Proven | 5–10% |
| Core | 10–20% |
| Elite | 20%+ |
| Retired | 0% |

Promotion requires days-in-stage threshold AND metrics within target band AND live profile matches backtest. Retirement triggers when expectancy degrades or drawdown exceeds limit.

---

## Starting Portfolio Allocation

| Sleeve | Allocation |
|---|---|
| Long-term diversified ETF | 25% |
| Momentum swing | 15% |
| Trend-following breakout | 15% |
| Mean reversion | 15% |
| Pullback continuation | 10% |
| Sector rotation | 10% |
| Cash reserve | 5% |
| Experimental research | 5% |

Total: 100%.

---

## Scorecard Fields

Required metrics per strategy: CAGR, Sharpe, Sortino, Calmar, profit factor, win rate, avg winner, avg loser, max drawdown, Ulcer Index, trade count, turnover, avg holding period, slippage sensitivity (dict: bps → Sharpe), regime-conditional performance (metrics split by each of the 3 regime dimensions), correlation to each active sleeve.

---

## Monthly Review Checklist

- P&L vs benchmark
- Drawdown per sleeve
- Correlation changes since last review
- Position-sizing adherence
- Execution quality
- Regime/permission compliance
- Capital allocation adjustments
