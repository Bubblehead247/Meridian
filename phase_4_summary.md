# Phase 4 Summary — Signal Engine + Backtester

**Status:** Complete
**Date:** 2026-06-22

Phase 4 delivers `meridian/signals/`: the **constant** mean-reversion signal
rule, a single-asset backtester, and the end-to-end comparative pipeline
(`run_backtest` / `sweep`). This is the heart of the project — every estimator
and deviation metric flows through identical trade logic, so any performance
difference is attributable to the fair-value definition, nothing else.

---

## What was built

### 1. Signal engine — `meridian/signals/engine.py`
- `SignalConfig` (frozen dataclass): `entry_threshold` (>0), `exit_threshold`
  (signed, default 0 = exit at fair value), optional `stop_threshold`,
  `max_holding`, and `allow_short`. `from_config()` reads the `signal:` section.
- `generate_positions(scores, config)` → path of `{-1, 0, +1}`. Textbook mean
  reversion: long when score ≤ −entry (price far below fair value), short when
  score ≥ +entry, close when the score reverts through the exit level.
- Stateful: a position persists until exit/stop/timeout. `NaN` scores take no
  action (hold if in a position, stay flat otherwise).
- **This logic never varies** across estimators/metrics — the fairness guarantee.

### 2. Backtester — `meridian/signals/backtest.py`
- `backtest(prices, positions, cost_bps)` → `BacktestResult`.
- **No look-ahead:** positions are lagged one bar (a decision from bar *t*'s
  score earns the *t → t+1* return).
- Flat **transaction costs** in bps charged on every change in held position
  (the documented flat-bps approximation; real slippage varies).
- Builds an equity curve, net/gross return series, and a **trade ledger**
  (direction, entry/exit time & price, bars held, per-trade return).
- `BacktestResult.summary()` gives basic sanity stats (total return, a basic
  annualized Sharpe, max drawdown, n_trades, win rate, exposure). **Full
  analytics — significance, bootstrap, attribution — are Phase 6/7.**

### 3. The comparative pipeline
- `compute_scores(prices, estimator, deviation, window, bars)` — walks the
  series **causally**, updating estimator then deviation bar by bar, so every
  score uses only past+current data (the path a live system sees). `atr_norm`
  receives the OHLC bar here.
- `run_backtest(prices, estimator, deviation, signal, ...)` — the single
  controlled pipeline: price → estimator → deviation → signal → PnL. Estimator
  and deviation are named; signal logic and costs held constant.
- `sweep(prices, estimators, deviations, ...)` → `{(est, dev): BacktestResult}`
  for every pair. Ranking + multiple-testing correction come in Phase 7.

---

## Tests — `tests/test_signals.py`

18 tests (207 total in suite, all passing):
- **Signal scenarios** (deterministic): long entry/exit at fair value, short
  entry/exit, `allow_short=False` stays flat, **stop-loss closes a widening
  long**, `max_holding` forces exit, NaN holds the position, entry>0 validation.
- **Backtest accounting**: no-look-ahead (a last-bar-only position earns
  nothing), long earns the price return, short profits when price falls, costs
  reduce return on turnover, flat positions give flat equity, trade ledger
  records a round trip.
- **Pipeline**: `compute_scores` causal & aligned (warmup NaN), `run_backtest`
  produces trades on a reverting AR(1) series, `atr_norm` threads bars through,
  `sweep` covers all pairs, and the pipeline is **deterministic** (same inputs →
  identical equity).

**Live comparative sweep** (SPY 2015-2019 in-sample, zscore, entry=1.5,
1 bp cost): ranked 10 estimators by Sharpe. `ou` — the Ornstein-Uhlenbeck
reversion-level estimator, the most explicitly mean-reversion-aware — ranked #1
(Sharpe 0.55), a reassuring sanity signal that the pipeline surfaces sensible
differences. (In-sample, single asset, no significance test — illustrative only.)

---

## Bug caught and fixed during the phase

- **Stop-loss re-entry.** A stop-out on a still-extreme score immediately
  re-entered the same side next bar, making the stop meaningless. Fixed by
  blocking re-entry on the stopped side until the score reverts through the exit
  level (then the block clears).

---

## Interface contracts handed to later phases

```python
from meridian.signals import run_backtest, sweep, SignalConfig, backtest

res = run_backtest(prices, "ou", "zscore",
                   SignalConfig(entry_threshold=1.5), window=20,
                   cost_bps=1.0, bars=ohlcv_or_None)
res.equity        # pd.Series, starts at 1.0
res.returns       # net per-bar returns
res.trades        # DataFrame ledger
res.summary()     # dict of basic stats + meta(estimator, deviation, window, cost_bps)

results = sweep(prices, estimators, deviations, signal, window=20, cost_bps=1.0)
```

---

## Known limitations / notes for Phase 5+

- **Single asset, unit position.** Position sizing and portfolio construction
  are Phase 6 (`portfolio/`). Here positions are ±1/0.
- **In-sample, no significance yet.** `summary()` stats are sanity checks.
  Phase 6 (WFO/Monte-Carlo/bootstrap) and Phase 7 (analytics + multiple-testing
  correction) provide statistically honest evaluation. Do not rank on these raw
  stats.
- **Costs are flat bps on turnover.** No spread/impact/borrow modeling.
- **Regime filters not yet applied.** Phase 5 (`regimes/`) adds a downstream
  regime gate; the signal engine is intentionally regime-agnostic.
- **Same window for estimator and deviation** in the pipeline helpers (one knob),
  matching the no-per-estimator-optimization rule. `compute_scores` accepts
  pre-built instances if independent windows are ever needed.
- **Exit/threshold scale differs by metric.** Bounded metrics (`percentile`,
  `minmax` ∈ [−1,1]) need smaller thresholds than unbounded z-scores; a sweep
  must use a metric-appropriate `entry_threshold`.

---

## Next phase

**Phase 5 — Regime classifiers:** a `meridian/regimes/` module that labels the
market (e.g. trending vs. ranging, high/low volatility) and gates signals
downstream, kept separate from the estimator interface per the design.
