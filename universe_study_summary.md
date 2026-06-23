# Universe-Wide Study Summary — Cross-Sectional Portfolio

**Status:** Complete
**Date:** 2026-06-23
**Scope:** Builds out `meridian/portfolio/` (previously a stub) and runs the
comparison cross-sectionally across many symbols — the highest-power test of a
mean-reversion estimator, flagged as the top priority after Phase 10.

---

## What was built

### 1. Sizing — `portfolio/sizing.py`
Turn a date × symbol grid of signals into portfolio weights. Uniform signature
`(signals, returns, lookback) -> weights`, gross exposure ~1:
- `equal_weight` — each active name gets `sign / n_active`.
- `inverse_vol` — weight inversely to recent volatility (risk-parity-ish),
  normalized per bar.

### 2. Cross-sectional backtester — `portfolio/portfolio.py`
`backtest_portfolio(signals, prices, sizing, cost_bps)` → `PortfolioResult`.
Sizes signals into weights, **lags them one bar** (no look-ahead), applies them
to each symbol's next return, nets out **portfolio turnover costs**, and returns
a single net return series — which plugs straight into the existing analytics
and validation machinery.

### 3. Universe runner — `portfolio/universe.py`
`run_universe_backtest(prices_by_symbol, estimator, ...)`: scores each symbol on
its own full history (correct warm-up) with the **same** estimator/deviation/
signal/window (no per-symbol optimization), aligns the signal paths on the
shared trading calendar, and aggregates into one portfolio.

### 4. Universe validation — `portfolio/validation.py`
`validate_universe(...)`: the full honest pipeline at the portfolio level —
anchored walk-forward on the stitched portfolio OOS returns, block bootstrap for
the Sharpe CI, a **portfolio-aware Monte-Carlo null** (each symbol's held-weight
path independently circularly rotated against its own returns, then
re-aggregated), and Benjamini-Hochberg correction. Returns the same verdict
table as the single-asset `validate`, plus `n_symbols`.

---

## Tests — `tests/test_portfolio.py`

11 tests (317 total in suite, all passing):
- **Sizing**: `equal_weight` splits active names and zeros flat rows, gross = 1;
  `inverse_vol` normalizes and favors quieter names.
- **Backtest**: no look-ahead (last-bar-only weights earn nothing), blended
  return is exact, costs reduce return.
- **Runner**: common-index intersection, signal-grid shape/values, gross
  exposure ≤ 1.
- **The power gain (key test)**: a 12-name portfolio of an edge lifts the Sharpe
  **above even the best single name** (~`SR·√K` diversification). Documents the
  subtlety that the Sharpe *CI width* tracks the number of time periods, not
  cross-sectional breadth — power comes from a higher point estimate.
- **Validation**: full verdict table with `n_symbols`, sorted, boolean
  significance.

---

## Headline result — the reason for the study

**Synthetic (mechanism check):** 12 independent strongly-reverting series →
portfolio Sharpe **7.19** vs mean single-name **2.16** (≈ `2.16·√12`). The
diversification machinery does exactly what theory says.

**Real data — 20 large-caps, 2010-2024, anchored WFO, 1 bp cost, zscore,
entry 1.5:**

| estimator | oos_sharpe | mc_pvalue | q_value | significant |
|-----------|-----------:|----------:|--------:|:-----------:|
| ou | 0.09 | 0.25 | 0.70 | False |
| lsma | −0.02 | 0.42 | 0.70 | False |
| hull | −0.05 | 0.46 | 0.70 | False |
| kalman | −0.11 | 0.53 | 0.70 | False |
| sma | −0.18 | 0.64 | 0.70 | False |
| ema | −0.20 | 0.66 | 0.70 | False |
| ens_invvar | −0.21 | 0.70 | 0.70 | False |

**Verdict: none significant** — and most are negative.

### Interpretation (the precise lesson)

- **Diversification amplifies a real edge but cannot manufacture one.** The
  synthetic test proves the portfolio mechanism lifts Sharpe ~√K when an edge
  exists. On real large-cap equities the per-name edge is ~0, so √K × 0 ≈ 0.
- **Costs make it worse.** Rebalancing ~20 names on daily signals adds turnover
  that drags the near-zero gross edge negative.
- This is a **stronger negative result than SPY alone**: applying the maximum
  statistical power available (a 20-name diversified portfolio, honest OOS,
  costs, correction), daily mean reversion on US large-caps shows **no robust,
  exploitable edge**. The platform's job was to establish that rigorously, and
  it did.

---

## Interface contracts

```python
from meridian.portfolio import run_universe_backtest, validate_universe

prices = {sym: series for sym in universe}                 # {symbol: adj_close}
res = run_universe_backtest(prices, "ou", "zscore", signal, sizing="equal_weight",
                            window=20, cost_bps=1.0)        # -> PortfolioResult
table = validate_universe(prices, estimators, "zscore", signal,
                          spec=spec, window=20, cost_bps=1.0,
                          n_boot=1000, n_mc=500)            # -> ranked verdict table
```

---

## Known limitations / future work

- **Shared-calendar intersection.** Symbols are aligned on the intersection of
  their date indices; staggered listings shrink the window. A union calendar
  with explicit "not yet listed" masking would handle IPOs/delistings better
  (and re-introduces the survivorship discussion).
- **Survivorship bias, amplified.** The universe is today's survivors; delisted
  names are absent. The real (bias-corrected) edge is likely *worse*, not
  better — strengthening the negative conclusion.
- **Costs are flat bps.** Real cross-sectional rebalancing also pays spread and
  impact per name; the negative drag is, if anything, understated here.
- **No CLI/portfolio config yet.** `validate_universe` is library-only; a
  `meridian universe <config>` command and a multi-symbol config schema are a
  natural next addition.
- **Equal-weight / inverse-vol only.** No long/short dollar-neutralization,
  sector caps, or position limits — straightforward extensions of the sizing
  layer.

---

## Bottom line

The universe study completes the research program's core question. Across 38
estimators, six deviation metrics, regime gating, ensembling, and now a
cross-sectional 20-name portfolio with full statistical machinery, **no
definition of fair value yields a statistically significant, cost-aware,
out-of-sample mean-reversion edge on liquid US large-caps.** The contribution is
the rigorous, reproducible framework that reaches — rather than hides from —
that conclusion.
