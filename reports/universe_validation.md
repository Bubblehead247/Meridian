# Meridian — Universe-Wide Mean-Reversion Validation

_Generated 2026-06-23_

## Setup

- **symbols**: 20 names
- **start**: 2010-01-01
- **end**: 2024-12-31
- **deviation**: zscore
- **window**: 20
- **cost_bps**: 1.0
- **wfo**: anchored
- **correction**: bh

## Verdict

**No estimator is statistically significant** after multiple-testing correction (of 7 tested). The best out-of-sample Sharpe is `ou` at 0.090, but its corrected q-value does not clear the threshold — consistent with no robust mean-reversion edge on this data.

## Ranked results (out-of-sample)

| estimator | oos_sharpe | oos_return | boot_ci_low | boot_ci_high | mc_pvalue | n_symbols | q_value | significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ou | 0.090 | 0.041 | -0.402 | 0.546 | 0.248 | 20 | 0.687 | False |
| lsma | -0.017 | -0.121 | -0.505 | 0.524 | 0.419 | 20 | 0.687 | False |
| hull | -0.053 | -0.197 | -0.540 | 0.440 | 0.447 | 20 | 0.687 | False |
| kalman | -0.112 | -0.266 | -0.525 | 0.380 | 0.545 | 20 | 0.687 | False |
| sma | -0.178 | -0.323 | -0.611 | 0.307 | 0.643 | 20 | 0.687 | False |
| ema | -0.196 | -0.349 | -0.620 | 0.304 | 0.637 | 20 | 0.687 | False |
| ens_invvar | -0.205 | -0.360 | -0.652 | 0.312 | 0.687 | 20 | 0.687 | False |

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
