# Phase 5 Summary — Regime Classifiers

**Status:** Complete
**Date:** 2026-06-22

Phase 5 delivers `meridian/regimes/`: **4 regime classifiers** that label market
state, plus a downstream gate that filters signals by regime. Per the design,
regime logic is **separate from the estimator interface** — classifiers do not
implement `BaseEstimator`, the signal engine never sees them, and gating is
applied to positions after the fact.

---

## What was built

### 1. Contract + template — `meridian/regimes/base.py`
- `BaseRegime` ABC: `fit(prices, bars=None)`, `update(price, bar=None)`,
  `label() -> str`, `state()`. Each classifier declares its `labels` tuple.
- `RollingRegime` template implements the contract once (rolling window, warmup
  → `UNKNOWN`, fit/update loop, state). Each classifier supplies only `_label()`.

### 2. The 4 classifiers — `meridian/regimes/classifiers.py`

| Name | Labels | Basis |
|------|--------|-------|
| `trend` | trend / range | Kaufman efficiency ratio (net move ÷ path length) vs threshold |
| `volatility` | low / normal / high | recent return-vol vs a longer `ref_window` baseline |
| `direction` | bull / bear | price above/below its long moving average |
| `hurst` | mean_reverting / random / trending | Hurst exponent (increment persistence) |

`hurst` is the most project-relevant: H < 0.5 = mean-reverting (the regime these
strategies want), ≈ 0.5 = random walk, > 0.5 = trending/momentum.

### 3. Downstream gate — `meridian/regimes/filter.py`
- `classify(prices, regime, window, bars)` → causal per-bar label series.
- `gate_positions(positions, labels, allowed)` → forces positions flat on any
  bar whose regime is not in `allowed`. No look-ahead (backtester still lags).
- `run_gated_backtest(...)` composes the Phase 4 pipeline + a regime gate,
  **without modifying the signal engine** — the separation principle in action.

---

## Tests — `tests/test_regimes.py`

24 tests (231 total in suite, all passing):
- **Contract sweep** over all 4: real label after warmup, state intact, warmup →
  UNKNOWN, `update == fit`, window validation.
- **Behavior**: `direction` bull/bear on rising/falling series; `trend` detects
  a strong trend vs a choppy range; `volatility` flags a vol burst as `high` and
  a quiet tail as `low`; `hurst` orders mean-reverting below trending and labels
  the extremes correctly.
- **Gate**: `classify` causal & aligned, `gate_positions` flattens disallowed
  regimes, and gating **never increases exposure**.

**Live demonstration (SPY 2015-2019, ou + zscore, entry 1.5):** restricting to
the `range` regime cut exposure 0.38 → 0.25 and trades 85 → 67 — but **lowered**
return (0.23 → 0.09) and Sharpe (0.55 → 0.31). Reported honestly: the gate is a
tool, not a guaranteed improvement; whether a given regime filter helps is an
empirical question for the validation phase, on out-of-sample data, with
significance testing.

---

## Known limitations / notes for Phase 6+

- **Hurst is noisy on short windows.** The lagged-dispersion estimator has high
  variance — two different random walks read 0.55 and 0.34 in testing. It
  reliably separates strong mean-reversion from strong persistence, but should
  be treated as *indicative*; prefer longer windows (default 100) and do not
  over-trust a single borderline reading. Tests assert robust ordering, not the
  flaky middle label.
- **Hurst measures increment persistence, not price direction.** A steady price
  ramp is "trending" to `direction`/`trend` but can read as low-H to `hurst`
  (its increments are near-constant, not persistent). Use the classifiers for
  their distinct meanings.
- **Regime gate is causal but coarse.** It flattens immediately when the regime
  flips, which can chop trades. A hysteresis/confirmation window is a possible
  Phase 6+ refinement.
- **No combined/meta regime yet.** Classifiers are independent; composing them
  (e.g. "range AND low-vol") is left to experiment configs / Phase 8 ensembling.
- **Volatility baseline is rolling, not full-history** (`ref_window`), so its
  high/low calls are relative to the recent past, by design.

---

## Interface contracts handed to later phases

```python
from meridian.regimes import create, classify, gate_positions, run_gated_backtest

clf = create("hurst", window=100)
clf.fit(prices); clf.label()                 # 'mean_reverting' | 'random' | 'trending'

labels = classify(prices, "trend", window=20)            # causal label series
gated  = gate_positions(positions, labels, allowed=["range"])
res    = run_gated_backtest(prices, "ou", "zscore", "trend", ["range"],
                            SignalConfig(entry_threshold=1.5), window=20)
```

---

## Next phase

**Phase 6 — Validation engine:** walk-forward optimization, Monte-Carlo, and
bootstrap resampling to evaluate strategies out-of-sample with honest
uncertainty — the statistical backbone that turns the Phase 4/5 sanity numbers
into defensible results.
