# Full model sweep — 2026-09-28 (pre-registered: `research/plans/full_sweep_2026_10.json`)

96 candidates, all fixed before any result:
- hold, trend, time-series momentum and dual momentum rules on three universes: Faber's 5, Meridian's 8,
  and the published 12-ETF PAA list;
- PAA itself;
- Meridian's structures (the 8-sleeve fund, TRIMMED, C1, C3);
- 60/40, SPY and an All-Weather-style mix;
- four fixed blends.

Objective: the most return while keeping the worst drawdown within −15%, **no leverage**. Total
returns, 10 bps costs. Outputs: `run_sweep.out`, `results_full_sample.csv`, `walk_forward_picks.csv`,
`summary.json`, `periods.out`.

**Gates:** both reproduced before any result was read. Faber 5.16% / −14.99% / Sharpe 0.66; the
8-sleeve baseline 7.07% / −14.96%.

**Run notes:**
- The lab refused the first attempt: the plan lacked two required fields, and nothing was computed.
- Two later runs printed results and then crashed on output formatting (a pandas attribute clash and a
  numpy bool in JSON) before the final run was logged.
- The plan never changed, the results were identical each time, and the final run was logged once.
  Two diagnostics were added after the first crashed run (when the worst drawdowns happened; the
  deflated Sharpe on returns above T-bills); neither changes a rule.

## Results

| Test | Result |
|---|---|
| In-sample best (all 96) | Dual momentum, 6-month, top 4 of the 8 ETFs (**DUALMOM6-K4-U8**): 11.0%/yr, max DD −14.6%, 2008–26 |
| Best deployable (in-sample) | 8-sleeve baseline 7.1% / −15.0%; **TREND6-EW-U8** 7.0% / −9.6% |
| Forward walk-forward (re-pick yearly, 2012–26) | Picking from all: 9.7% but **−25.7%** DD (2015–16, holding the 3-month variant). Deployable only: 4.8% / −20.6%. **Fixed 8-sleeve baseline: 8.9% / −15.0%**; C1: 7.5% / −11.3% |
| Reverse crash test (pick on 2011–26, run on 2008–10) | Pick DUALMOM6-K4-U8: +8.4% / −13.1%. Baseline +1.7% / −12.5%; C1 +1.8% / −18.7%; 60/40 +2.1% / −22.4% |
| PBO (Sharpe-based) | **0.52**: picking the best of these 96 is no better than chance |
| Deflated Sharpe on returns above T-bills | Best (DUALMOM6-K4-U8, Sharpe 0.80): 0.99 at n=96, 0.97 at n=809. The top models reliably beat T-bills; which one is best is noise |
| Universe, paired comparisons | PAA-12 beats the 8 in **4%** of pairs (median −0.8%/yr); the 8 beat Faber-5 in 96%; PAA-12 beats Faber-5 in 93% |

By period (CAGR / max DD):

| Model | 2008–26/06 | 2008–10 | 2011–15 | 2016–20 | 2021–26/06 |
|---|---:|---:|---:|---:|---:|
| **TREND6-EW-U8** (recommended) | +7.0% / −9.6% | +9.0% / −7.4% | +3.8% / −8.5% | +8.7% / −9.3% | +7.4% / −8.5% |
| DUALMOM6-K4-U8 (in-sample best) | +11.0% / −14.6% | +8.4% / −13.1% | +7.9% / −14.6% | +14.4% / −13.0% | +12.2% / −13.9% |

TREND6 is invested 64% of the time on average; DUALMOM6-K4 89%.

## Decisions (plan rules)

- **Recommendation: TREND6-EW-U8** (small build). Each of the 8 ETFs gets 12.5% and is held while it
  is above its 6-month average, otherwise that share is in T-bills. It is the simplest deployable
  candidate within 0.5%/yr of the best deployable (near-tie set: TREND6-EW-U8, TSMOM3-EW-U8,
  TSMOM12-EW-U8, BASELINE8, C3). Its reverse-crash DD is −7.4%, so no flag.
- **The sweep adds no reliable value** over the baseline: re-picking yearly broke the drawdown cap.
- **Don't expand the universe** among ETF asset classes.

## Reading (Claude)

- There is no reliable "best" model. There is a robust **family**: trend and momentum across a small
  set of asset-class ETFs, with T-bills as the safe asset. In backtests it makes about 7–11%/yr within
  about −15%. The exact variant is mostly noise (PBO 0.52), and re-optimising it yearly hurt.
- **Dual momentum** is the return leader. It is a plateau, not a lone spike: 6-month/blend lookbacks
  with K = 3–4 all land around 9–11%. It passed the unseen-crash test. But the 3-month variants drew
  down 23–27%, it needs an engine change (fixed-slot sizing), and **U8 contains hindsight choices
  (QQQ, gold)**: the published universes did worse. Treat 11% as an optimistic ceiling. It is a pod
  candidate: pre-register it, then shadow it.
- **TREND6** matches the old fund's return with about two-thirds of its drawdown, from 8 simple rules.
  Its −9.6% leaves risk budget unused under the no-leverage rule.
- **C1 is superseded**: 6.5% at −15%, below both.
- **Expanding the universe:** more ETFs didn't help. What could matter needs things Meridian doesn't
  have: single stocks need paid point-in-time index data (Norgate, Sharadar, CRSP), and long/short
  trend via managed-futures ETFs only exists since about 2019–2020.
