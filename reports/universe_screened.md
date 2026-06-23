# Meridian — Universe-Wide Mean-Reversion Validation

_Generated 2026-06-23_

## Setup

- **symbols**: 24 names
- **start**: 2010-01-01
- **end**: 2024-12-31
- **deviation**: zscore
- **window**: 20
- **cost_bps**: 1.0
- **wfo**: anchored
- **correction**: bh

## Verdict

**No estimator is statistically significant** after multiple-testing correction (of 7 tested). The best out-of-sample Sharpe is `lsma` at 0.353, but its corrected q-value does not clear the threshold — consistent with no robust mean-reversion edge on this data.

## Ranked results (out-of-sample)

| estimator | oos_sharpe | oos_return | boot_ci_low | boot_ci_high | mc_pvalue | n_symbols | q_value | significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lsma | 0.353 | 0.905 | -0.183 | 0.846 | 0.070 | 24 | 0.489 | False |
| hull | 0.195 | 0.246 | -0.386 | 0.655 | 0.156 | 24 | 0.545 | False |
| ou | 0.082 | -0.056 | -0.440 | 0.616 | 0.363 | 24 | 0.671 | False |
| kalman | -0.044 | -0.346 | -0.573 | 0.481 | 0.479 | 24 | 0.671 | False |
| ema | -0.078 | -0.376 | -0.593 | 0.452 | 0.527 | 24 | 0.671 | False |
| sma | -0.152 | -0.467 | -0.659 | 0.335 | 0.649 | 24 | 0.671 | False |
| ens_invvar | -0.190 | -0.556 | -0.671 | 0.285 | 0.671 | 24 | 0.671 | False |

## Limitations and disclosures

- **Survivorship bias.** Price history comes from yfinance, which omits delisted
  companies, and index/screened universes are sampled as of today. Results are
  therefore survivorship-biased and likely optimistic. This bias is *not* removed
  by any statistic in this report.
- **Multiple testing.** Many estimators were tested on shared data, so some will
  look good by chance. Significance is reported only after correction; raw
  p-values must not be read on their own.
- **Out-of-sample discipline.** Reported performance is walk-forward
  out-of-sample. In-sample figures (if any) are sanity checks, not results.
- **Transaction costs** are modeled as flat basis points on turnover; real
  slippage, spread, impact, and borrow costs will differ.
