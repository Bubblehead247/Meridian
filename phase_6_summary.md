# Phase 6 Summary — Validation Engine

**Status:** Complete
**Date:** 2026-06-22

Phase 6 delivers `meridian/validation/`: the statistical backbone that turns the
in-sample sanity numbers of Phases 4-5 into **honest, out-of-sample,
significance-tested** results. The chain is: walk-forward gives the honest curves
→ bootstrap + Monte-Carlo give the uncertainty → multiple-testing correction
gives the verdict.

---

## What was built

### 1. Walk-forward engine — `walkforward.py`
- `WalkForwardSpec` — fold schedule in bars: `mode` (anchored/rolling),
  `train_span`, `test_span`, `step`, `min_train`.
- `make_folds(n, spec)` — disjoint, forward-only test windows that tile the
  series; train windows are expanding (anchored) or fixed-length (rolling).
- `walk_forward(...)` — scores **all** (estimator, deviation) pairs every fold.
  - **Anchored optimization:** because the estimators are causal, the position
    path is fold-independent, so it is computed **once** per pair via the Phase 4
    backtester and sliced into folds — same cost as one sweep, provably no
    look-ahead. Rolling recomputes per fold (moving warm-up start).
- `WalkForwardResult` — stitches the disjoint test windows into one continuous
  OOS return/equity series per pair; `summary()` ranks by stitched OOS Sharpe;
  `oos_market_returns()` gives the buy-and-hold benchmark over the test union.
- **Selection overlay** — `result.selection()`: each fold pick the pair with the
  best *train-window* Sharpe and take its *test* returns. This is the
  "trade the recently-best estimator" meta-strategy, chosen with train data only
  (no peeking), validated the same honest way.

### 2. Block bootstrap — `bootstrap.py`
`block_bootstrap_sharpe(returns, block, n_boot, ...)` resamples returns in
**blocks** (preserving autocorrelation) to build the Sharpe sampling
distribution → confidence interval + one-sided p-value. CI excluding zero =
distinguishable from no skill.

### 3. Monte-Carlo significance — `montecarlo.py`
`monte_carlo_pvalue(market_returns, held_positions, ...)` builds a no-skill null
by **circularly rotating** the position path against the market returns: same
number of trades, exposure, and holding lengths, but the timing alignment is
destroyed. p-value = fraction of random rotations matching/beating the real
Sharpe (with the standard +1/+1 correction).

### 4. Multiple-testing correction — `correction.py`
`benjamini_hochberg` (FDR, recommended) and `bonferroni` (FWER, strict), plus
`correct(method=...)`. Required because testing 38 estimators on shared data
guarantees some look good by chance.

### 5. End-to-end verdict table — `pipeline.py`
`validate(prices, estimators, deviation, ...)` → ranked DataFrame: `oos_sharpe`,
`oos_return`, `boot_ci_low/high`, `boot_p`, `mc_pvalue`, `q_value` (corrected),
`significant`. One call, the whole chain.

---

## Tests — `tests/test_validation.py`

16 tests (247 total in suite, all passing):
- **Folds**: anchored tiling/forward-only, rolling fixed-width, mode validation.
- **No look-ahead (key test)**: anchored stitched OOS returns **exactly equal** a
  single full backtest sliced to the test windows.
- **Selection** overlay picks the per-fold train-metric argmax; summary runs;
  rolling mode runs.
- **Bootstrap**: CI brackets the point and flags a positive-drift series;
  zero-mean noise is not significant; degenerate input → NaN.
- **Monte-Carlo**: detects perfect timing (p < 0.05), random positions not
  significant, flat positions → NaN.
- **Correction**: BH rejects the expected set, q ≥ raw p; Bonferroni threshold
  and adjusted values; `strategy_net` matches a hand computation.

---

## Headline live result (the reason this phase exists)

Full `validate()` on **SPY daily, 2010-2024**, anchored WFO (3y min train,
6-month test windows), 14 estimators, zscore, entry 1.5, 1 bp cost,
800 bootstrap + 800 Monte-Carlo draws:

| estimator | oos_sharpe | mc_pvalue (raw) | q_value (BH) | significant |
|-----------|-----------:|----------------:|-------------:|:-----------:|
| hull | 0.41 | **0.037** | 0.52 | **False** |
| hp_filter | 0.26 | 0.14 | 0.76 | False |
| lsma | 0.23 | 0.18 | 0.76 | False |
| ... | ... | ... | ... | ... |
| ar1 | −0.39 | 0.91 | 0.91 | False |

**Verdict: none significant.** `hull`'s raw p-value of 0.037 looks "significant"
at a naive 0.05 — and would fool an uncorrected analysis — but BH correction
(14 tests) lifts its q-value to 0.52. Every bootstrap CI includes zero.

This is the honest, defensible conclusion the pipeline is built to reach: on a
strongly trending, survivorship-biased single index, mean reversion shows **no
statistically robust skill** once evaluated out-of-sample with multiple-testing
correction. The machinery's job is precisely to stop us mistaking the
best-of-14 luck for skill.

---

## Interface contracts handed to later phases

```python
from meridian.validation import validate, walk_forward, WalkForwardSpec

spec = WalkForwardSpec(mode="anchored", min_train=756, test_span=126, step=126)
table = validate(prices, estimators, "zscore", signal,
                 spec=spec, window=20, cost_bps=1.0,
                 n_boot=2000, n_mc=2000, method="bh", seed=0)

wf = walk_forward(prices, estimators, ["zscore"], signal, spec=spec)
wf.summary()            # per-pair stitched OOS stats
wf.selection()          # recently-best meta-strategy
wf.stitched_returns(("hull", "zscore"))
```

---

## Known limitations / notes for Phase 7

- **Single asset so far.** Validation runs on one price series; cross-sectional
  (universe-wide) aggregation and portfolio-level validation come with Phases
  6→7→8. Multi-asset would also raise the effective sample size.
- **Monte-Carlo null is rotation-based** (timing). It preserves the position
  footprint but assumes returns are roughly stationary under rotation; a
  regime-shifting series weakens that. Bootstrap (block) complements it.
- **One window/threshold** held constant (no per-estimator tuning), by design.
  The verdict reflects that fixed configuration, not each estimator's best case.
- **Survivorship bias unaddressed by statistics.** Correcting p-values does not
  fix biased data — every report must still state the yfinance limitation.
- **Selection overlay** is provided but not yet bootstrapped/MC-tested in
  `validate()`; Phase 7 reporting can fold it into the ranking.
- These stats are the inputs to Phase 7's **analytics + reporting** layer (full
  risk metrics, drawdown curves, tear sheets, ranked reports).

---

## Next phase

**Phase 7 — Analytics and reporting:** full performance metrics (risk-adjusted
returns, drawdown, tail stats), comparison heatmaps, and generated markdown/PDF
reports that present the validated rankings with their multiple-testing caveats
and the survivorship-bias disclosure.
