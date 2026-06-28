# Relative-Value / Cross-Sectional Mean-Reversion Study (pass 1)

**Status:** Complete
**Date:** 2026-06-24
**Scope:** First of three research extensions. Tests **cross-sectional** mean
reversion (price relative to peers) where **absolute** MR (price vs its own
average) showed no edge. Market-residual and sector-relative transforms, and
config/CLI wiring, are deferred.

---

## What was built

### Transform — `meridian/features/relative.py`
`cross_sectional_demean(prices_by_symbol)` → `{symbol: relative Series}`:
each bar, `rel_i = log(P_i) − mean_j(log P_j)` over the names present. A name far
above the cross-sectional average is "rich" (+), far below is "cheap" (−);
reverting that relative value is market-neutral by construction (relative values
sum to zero across names each bar). Missing/non-positive prices → NaN (untradable
that bar). This is the classic short-term-reversal signal.

### Architectural change — signal series separate from P&L prices
A relative series crosses zero, so it has no meaningful `pct_change`. Added an
optional `signal_prices_by_symbol` to `run_universe_backtest` and
`validate_universe` (`meridian/portfolio/`): **signals** are computed from the
relative series while **P&L** still uses the tradable prices. Default `None`
reproduces the absolute strategy exactly (regression-tested). This is the only
new machinery — the estimators, deviations, signal engine, portfolio backtester,
bootstrap/Monte-Carlo, and BH correction are all reused unchanged.

### Tests
`tests/test_relative.py` (transform: zero-mean per bar, rich/cheap sign, NaN
handling) + `tests/test_portfolio.py` (signal/P&L separation, default-equals-
absolute regression guard, `validate_universe` with relative signals).
**345 tests pass, lint clean.**

---

## Results — does cross-sectional MR beat absolute MR?

Equal-weight long/short portfolio, anchored walk-forward, BH correction,
`entry=1.5`. Same universes/windows for absolute vs cross-sectional.

### Window 20 (comparable to the absolute baseline) — nothing significant
| universe | best absolute | best cross-sectional |
|----------|--------------|----------------------|
| Large-caps 2010-2024 | ou 0.09 (q .63) | sma 0.06 (q .60) |
| Survivorship-free 2014-2018 | sma 0.33 (q .48) | ou 0.30 (q .50) |

### Window 5 (the reversal horizon) — a hit, then killed by costs
| universe | window | result |
|----------|:------:|--------|
| **Large-caps 2010-2024** | **5** | **sma Sharpe 0.49, q=0.013 — SIGNIFICANT**; lsma q=0.017 sig |
| Large-caps 2010-2024 | 10 | sma 0.23, q=0.18 — not significant (effect fades) |
| Survivorship-free 2014-2018 | 5 | sma 0.01 — **does not replicate** |
| Survivorship-free 2014-2018 | 10 | negative — not significant |

The large-cap window-5 result is the **first significant finding** in the whole
project, and it lands exactly where theory predicts: short horizon,
cross-sectional, liquid names.

### The decisive test — transaction costs (large-caps, cross-sectional, sma, w=5)
Mean **daily turnover = 0.84** (the book churns ~84% of gross exposure per day).

| cost | OOS Sharpe |
|-----:|-----------:|
| 1 bp | **+0.49** |
| 5 bps | **−0.46** |
| 10 bps | −1.65 |
| 20 bps | −4.02 |
| 30 bps | −6.34 |

The signal is **net-negative by 5 bps** — still an optimistic cost for daily
rebalancing across 20 names (real spread + impact are higher).

---

## Conclusion (honest)

- **A real, statistically significant *gross* signal exists**: short-horizon
  (5-day) cross-sectional reversal on liquid large-caps, 2010-2024 (Sharpe 0.49,
  q = 0.013 after correction at 1 bp). This is more than the absolute strategies
  ever produced.
- **It is not an exploitable edge.** At ~0.84 daily turnover it is wiped out by
  transaction costs — net-negative by 5 bps and catastrophic beyond — the
  textbook fate of short-term reversal in liquid names.
- **It does not replicate** on the survivorship-free universe (2014-2018), and
  fades at longer windows — so even the gross signal is period/universe-specific,
  not a stable phenomenon.
- **Caveat on search:** several windows × estimators × universes were tried; the
  within-run BH correction adjusts for the estimators in each run but not for the
  configurations explored. The cost result makes this moot (no exploitable edge
  to over-claim), but the gross-signal hit should be read as *suggestive*, not
  established.

**Net:** the platform did its job — surfaced a theory-consistent candidate edge,
then rigorously demonstrated it is not tradable after costs. The headline stands:
no robust, cost-aware, out-of-sample mean-reversion edge on US equities, absolute
*or* cross-sectional.

---

## Next passes
- **Intraday-frequency MR** — DONE (`intraday_study_summary.md`): no edge at any
  horizon 1m–1h, loses even gross (bid-ask-bounce artifact).
- **Other asset classes** — DONE (`other_asset_classes_study_summary.md`): crypto
  cost is higher not lower; ETF twins too tight; **sector/calendar spreads** gave
  the project's first cost-surviving BH-significant signal (candidate for
  confirmation, not an established edge).
- **Deferred transforms** (still open) — market-residual (beta-neutral) and
  sector-relative, plus `relative:` config/CLI wiring.
