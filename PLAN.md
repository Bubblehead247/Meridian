# PLAN.md — Meridian Expansion to Full Quant-Fund Research Platform

Session 001 architecture plan. Opus authors this file only. Sonnet executes it
exactly. No implementation logic this session — stubs and structure only.

Reconciliation decisions (approved):
- **Reconcile/extend existing modules** — add only genuinely-new packages; map the
  prompt's duplicate paths onto existing modules (see §2). Do not multiply entities.
- **Wrap & register** — `families/` is a thin new layer whose models compose the
  EXISTING estimators/deviations/signals; nothing existing moves or is touched.

---

## 1. Current vs. target state

| Exists today | Being added |
|---|---|
| Estimator lib (38) + registry, deviations, signals/backtest | `families/` multi-model layer wrapping them |
| `regimes/` rolling single-series classifiers | `regimes/labeler.py` index-level 3-dim regime labeler |
| `validation/` (walk-forward, OOS splits, bootstrap, MC, correction) | `pipeline/` stage-runners + graduation + research state |
| `analytics/metrics.py` (CAGR/Sharpe/Sortino/Calmar/DD/PF…) | `scoring/scorecard.py` thin layer adding Ulcer/turnover/holding/regime-split/correlation |
| `portfolio/` sizing + construction + validation | `portfolio/` ledger, allocation, correlation, risk_budget, monthly_review |
| `data/` loader+universe+cache, `execution/`, `visualization/` | `reporting/monthly_report.py` |
| Single (mean-reversion) research focus | 9 strategy families, graduation pipeline, virtual ledger, risk budget, monthly review |

## 2. Expanded directory tree  (proposed-path → actual; `[EXISTS]` reused, `[NEW]` scaffold)

```
meridian/
├── estimators/base.py            [EXISTS]   (= prompt core/base_estimator.py)
├── data/universe.py              [EXISTS]   (= prompt core/universe.py, data/universe.py)
├── data/loader.py                [EXISTS]   (= prompt data/ohlcv.py)
├── regimes/                      [EXISTS]
│   └── labeler.py                [NEW]      (= prompt core/regime_labeler.py)
├── families/                     [NEW]
│   ├── __init__.py               [NEW]
│   ├── base.py                   [NEW]      Model + StrategyFamily base
│   ├── registry.py               [NEW]      (family,model) registry (mirrors estimators/registry)
│   ├── mean_reversion/           [NEW]      wraps existing est+dev+signals
│   │   ├── __init__.py           [NEW]
│   │   └── models.py             [NEW]      RSI exhaustion, Bollinger, gap fill, z-score, ATR ext
│   ├── trend_following/{__init__,models}.py   [NEW]
│   ├── momentum/{__init__,models}.py          [NEW]
│   ├── breakouts/{__init__,models}.py         [NEW]
│   ├── pullback_continuation/{__init__,models}.py [NEW]
│   ├── sector_rotation/{__init__,models}.py   [NEW]
│   ├── long_term_etf/{__init__,models}.py     [NEW]
│   ├── event_driven/__init__.py  [NEW — stub only]
│   └── volatility/__init__.py    [NEW — stub only]
├── pipeline/                     [NEW]
│   ├── __init__.py               [NEW]
│   ├── research.py               [NEW]      idea/hypothesis records
│   ├── graduation.py             [NEW]      stage state machine
│   ├── backtest.py               [NEW]      stage-runner over signals/portfolio (reuses, no math)
│   ├── walk_forward.py           [NEW]      stage-runner over validation/walkforward (reuses)
│   └── oos.py                    [NEW]      stage-runner over data/splits + validation (reuses)
├── scoring/                      [NEW]
│   ├── __init__.py               [NEW]
│   └── scorecard.py              [NEW]      wraps analytics/metrics + extra fields
├── portfolio/                    [EXISTS — add modules]
│   ├── ledger.py                 [NEW]
│   ├── allocation.py             [NEW]
│   ├── correlation.py            [NEW]
│   ├── risk_budget.py            [NEW]
│   └── monthly_review.py         [NEW]
└── reporting/                    [NEW]
    ├── __init__.py               [NEW]
    └── monthly_report.py         [NEW]
```
Prompt `pipeline/walk_forward.py`/`oos.py` are stage-runners that **reuse**
`validation/`; they do not reimplement walk-forward/OOS math.

## 3. Module responsibilities (owns / does not own)

- `regimes/labeler.py` — owns daily index-level trend/vol/breadth labels; not single-series rolling classifiers (those stay in `regimes/classifiers.py`).
- `families/base.py` — owns the `Model` abstraction (compose est+dev+signal → entries/exits) and `StrategyFamily` grouping; not the estimator math.
- `families/registry.py` — owns `(family, model)` registration/lookup; not model logic.
- `families/<fam>/models.py` — owns that family's concrete models (stubs now); not portfolio/risk.
- `pipeline/research.py` — owns idea+hypothesis records (steps 1–2); not scoring.
- `pipeline/graduation.py` — owns stage state + promotion/retirement rules (steps 6–10); not capital math (reads ledger/scorecard).
- `pipeline/backtest.py|walk_forward.py|oos.py` — own running a model through that stage and returning results; not the underlying validation math (reused).
- `scoring/scorecard.py` — owns the full per-strategy metric set incl. regime splits + sleeve correlation; not raw metric formulas (reuses `analytics/metrics`).
- `portfolio/ledger.py` — owns per-strategy virtual accounting; not allocation policy.
- `portfolio/allocation.py` — owns target sleeve weights + ledger init; not risk limits.
- `portfolio/correlation.py` — owns inter-sleeve return correlation matrix; not sizing.
- `portfolio/risk_budget.py` — owns portfolio heat + per-strategy risk contribution + suspension triggers; not order execution.
- `portfolio/monthly_review.py` — owns the month-end evaluation process; not rendering.
- `reporting/monthly_report.py` — owns rendering the monthly review to markdown; not the evaluation logic.

## 4. Regime labeling design (3 dimensions)

Computed daily, stored as a DataFrame indexed by date with 3 categorical columns; a
`RegimeLabel` dataclass holds one day's `(trend, volatility, breadth)`.

- **Trend** (SPY or IWM): `Bull` = close>200-day MA AND ADX>25; `Neutral` = close within ±X% of 200MA AND ADX 15–25; `Bear` = close<200MA. (yfinance SPY/IWM.)
- **Volatility** (`^VIX` close): `Low`<15, `Normal` 15–20, `Elevated` 20–30, `Extreme`>30.
- **Breadth** (% of S&P 500 above their 200-day MA, from `data/universe` + `data/loader`): `Expansion`>60%, `Neutral` 40–60%, `Contraction`<40%.

Attachment: every backtest bar, walk-forward window, and live signal is left-joined to
the regime frame on date → carries all three labels. Cached like OHLCV.

## 5. Permission matrix design  (family × regime → ✅active / ⛔restricted)

Gate evaluated per signal date against the day's labels. Trend dim drives most gates;
vol/breadth modifiers noted.

| Family | Bull trend | Neutral trend | Bear trend | Modifier |
|---|---|---|---|---|
| Trend following | ✅ | ⛔ | ⛔ | ⛔ if VIX Extreme |
| Breakouts | ✅ | ⛔ | ⛔ | needs Breadth Expansion/Neutral |
| Momentum | ✅ | ⛔ | ⛔ | ⛔ if VIX Extreme |
| Pullback continuation | ✅ | ⛔ | ⛔ | needs Breadth Neutral |
| Mean reversion | ⛔ | ✅ | ✅ | active in Breadth Neutral/Contraction |
| Sector rotation | ✅ | ✅ | ✅ | always (all trend regimes) |
| Long-term ETF | ✅ | ✅ | ✅ | no regime gate |
| Cash reserve | — | — | ↑ | allocation ↑ when Bear + VIX Extreme |
| Event-driven / Volatility | deferred | deferred | deferred | stub only |

## 6. Graduation pipeline design

Per-strategy `stage` ∈ {research, backtest, walk_forward, oos, paper, pilot, proven,
core, elite, retired} + `stage_entered` date, stored in the ledger record (§7).
Promotion = evidence rules checked by `pipeline/graduation.py` against the scorecard:
days-in-stage threshold (paper≥30, pilot→proven≥90, proven→core≥180) AND metrics within
target band AND live profile matches backtest. Capital per stage from the table (0% →
1–3% → 5–10% → 10–20% → 20%+). Retirement trigger: expectancy degraded or DD>limit →
stage=`retired`, capital→0. State persisted as JSON (mirror `execution/trader` checkpoint).

## 7. Virtual ledger design

`StrategyLedger` dataclass per strategy: `name, family, stage, stage_entered,
capital_alloc, open_positions[], realized_pnl, unrealized_pnl, drawdown_cur, drawdown_max,
win_rate, expectancy, turnover, correlations{sleeve→ρ}, risk_contribution`. Updated each
bar/trade from fills. Stored one JSON per strategy under a `ledger/` data dir; `ledger.py`
owns load/update/save. One brokerage account, many virtual ledgers summing to it.

## 8. Risk budget design

Portfolio heat = Σ open-position risk, where position risk = (entry−stop distance × size)/equity.
Per-strategy contribution = its share of total heat. Limits (from decisions): risk/trade
0.25–0.50%, portfolio heat 3–5%, position ≤5–10%, sector ≤20–25%. `risk_budget.py`
computes heat, allocates each strategy a heat share, and flags **suspension** when a
strategy's DD breaches 8–12% or its heat exceeds its budget. Managed at aggregate risk,
not just dollars.

## 9. Monthly review design

`monthly_review.py` triggered at month-end (manual/scheduled). Reads: all ledgers,
scorecards, the regime log. Evaluates the checklist (P&L vs benchmark, DD/sleeve,
correlation changes, sizing adherence, execution quality, regime/permission compliance) →
emits per-sleeve actions {increase/hold/reduce/suspend} + a data dict handed to
`reporting/monthly_report.py` for a markdown report under root `reports/`.

## 10. Scorecard design

`scorecard(returns, trades, regime_labels, sleeve_returns) -> dict`. Reuses
`analytics.metrics.performance_metrics` for: cagr(float), sharpe, sortino, calmar,
profit_factor, win_rate, avg_win, avg_loss, max_drawdown, trade count(int), all float
unless noted. Adds: `ulcer_index`(float), `turnover`(float), `avg_holding_period`(float
bars), `slippage_sensitivity`(dict bps→sharpe), `regime_conditional`(dict label→metrics —
returns grouped by each of the 3 regime dims, metrics recomputed per bucket),
`sleeve_correlation`(dict sleeve→ρ). Regime split = group the return/trade series by the
attached labels (§4) and recompute the metric block per bucket.

## 11. Multi-model pattern

A **family** is a package; it contains multiple **models** (`families/base.Model`
subclasses) that each `@register_model("family", "name")` (registry mirrors
`estimators/registry.py`). Each model composes existing estimator+deviation+signal into
entry/exit rules, and is independently backtested → walk-forward → OOS → scored. Families
expose `list_models()`; the pipeline drives any model through the same 10 stages. mean_reversion ships 5 model stubs (RSI exhaustion, Bollinger reversion, gap fill, z-score
reversion, ATR extension); every other active family follows the identical pattern.

## 12. Starting portfolio allocation

`allocation.py` seeds the ledger at first run from this fixed table:

| Sleeve | Alloc | | Sleeve | Alloc |
|---|---|---|---|---|
| Long-term diversified ETF | 25% | | Sector rotation | 10% |
| Momentum swing | 15% | | Cash reserve | 5% |
| Trend-following breakout | 15% | | Experimental research | 5% |
| Mean reversion | 15% | | Pullback continuation | 10% |

Sums to 100%. Each sleeve → one `StrategyLedger` with `capital_alloc = pct × equity`.

## 13. Sonnet task list

**scaffold-sonnet** (model: sonnet) — create structure/stubs per §2; stubs only, no logic.

| ID | Agent | Task | Done when |
|---|---|---|---|
| S1 | scaffold | `regimes/labeler.py` stub (docstring owns/not-owns, stdlib imports, TODO: 3 fns) | file exists, no logic |
| S2 | scaffold | `families/` pkg: `__init__`, `base.py`, `registry.py` stubs | 3 files, registry TODO mirrors estimators/registry |
| S3 | scaffold | 7 active family pkgs (mean_reversion, trend_following, momentum, breakouts, pullback_continuation, sector_rotation, long_term_etf): each `__init__.py` + `models.py` stub | 14 files; mean_reversion/models lists the 5 model stubs in TODO |
| S4 | scaffold | 2 deferred family pkgs (event_driven, volatility): `__init__.py` only w/ `# Deferred — see PLAN.md` | 2 files |
| S5 | scaffold | `pipeline/` pkg: `__init__`, `research.py`, `graduation.py`, `backtest.py`, `walk_forward.py`, `oos.py` stubs | 6 files; backtest/walk_forward/oos TODO note "reuses validation/signals, no math" |
| S6 | scaffold | `scoring/` pkg: `__init__`, `scorecard.py` stub (TODO: wrap analytics.metrics + extra fields) | 2 files |
| S7 | scaffold | `portfolio/` new modules: `ledger.py`, `allocation.py`, `correlation.py`, `risk_budget.py`, `monthly_review.py` stubs | 5 files; existing portfolio files untouched |
| S8 | scaffold | `reporting/` pkg: `__init__`, `monthly_report.py` stub | 2 files |
| S9 | scaffold | Output checklist of every file created | checklist printed; existing files unmodified |

**docs-sonnet** (model: sonnet) — three doc tasks per Phase 3 brief.

| ID | Agent | Task | Done when |
|---|---|---|---|
| D1 | docs | Create `progress.log` with SESSION 001 header + §13 task list as `[ ]`; Notes left blank | file exists, tasks copied |
| D2 | docs | Patch `CLAUDE.md`: expanded mandate + sections (modules, 9 families w/ status+holding, 3 regime dims, graduation stages+capital, 8-sleeve alloc, scorecard fields, monthly review). Preserve all existing content | sections added, nothing removed |
| D3 | docs | Create `STRATEGY_TEMPLATE.md` with all 18 spec fields | file exists, all fields present |

## 14. Acceptance criteria (this session)

- [ ] PLAN.md written (this file).
- [ ] All [NEW] dirs/files in §2 scaffolded as stubs (no logic); existing files untouched.
- [ ] event_driven & volatility are `__init__`-only deferred stubs.
- [ ] `progress.log` initialized with the §13 task list.
- [ ] `CLAUDE.md` patched, all prior content preserved.
- [ ] `STRATEGY_TEMPLATE.md` created with all fields.
- [ ] Opus session summary appended to `progress.log` Notes.
- [ ] Repo still imports / tests still pass (stubs must not break collection).

## 15. Deferred decisions

- Breadth source: full S&P-500 %>200MA needs all constituents; interim may sample the
  universe — confirm data cost before implementing (logged, not guessed).
- ADX threshold exact "near 200MA" band (±%) for Neutral trend — set at implementation.
- Ledger persistence dir name/location (`ledger/` vs under `data_cache/`) — decide when wiring `ledger.py`.
- Slippage-sensitivity bps grid for the scorecard — pick at implementation.
- Benchmark series for monthly review (SPY vs blended) — confirm with user later.
