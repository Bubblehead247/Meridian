# Pre-registered plans (skill lab)

Each `*.json` plan is committed before it runs; `meridian/validation/skill_lab.py` refuses an
uncommitted or edited plan, seals any hold-out from development runs, allows one final run per
plan fingerprint, and appends every run to `ledger.jsonl` (append-only: do not edit).

| Plan | Result (final run) | Consequence (fixed in the plan) |
|---|---|---|
| `faber_gtaa_5etf.json` — Faber 10-month SMA on SPY EFA IEF GSG VNQ, 2007-05..2026-09 | alpha vs equal-weight hold of the same ETFs +2.07%/yr, Newey-West t 1.68 (needed 2.0); vs SPY+IEF +1.01%, t 0.76. CAGR 5.2% vs 5.7% held; max DD −15.0% vs −48.6%; Sharpe 0.66 vs 0.45 (60/40 SPY/IEF: 0.77, −31.4%) | **Positive, not significant** → shadow sleeve from 2026-10-01, judged 2027-09-30. No live change. |
| `meridian_structure_2026_10.json` — can a simple core replace the 8 sleeves? (non-inferiority, 2011–2026/06, history already seen) | **C1 non-inferior in all 3 periods and the winner:** 50% 60/40 SPY/IEF + 25% long-term ETF rotation + 25% T-bills → 9.1%/yr, Sharpe 1.13, max DD −15.2% (baseline 8.0%, 1.00, −15.0%). C3 (post-hoc holds) also non-inferior; C2 (Faber core) failed 2011–15 drawdown | Propose C1 as the core book at the owner's −15% target; switch only with the owner's OK |
| `structure_2008_check.json` — does C1 still match the 8-sleeve baseline once 2008 is in? (matched −15% drawdown, 2008-01..2026-06) | Baseline +7.1%/yr at k 1.00; **C1 +6.5% at k 0.78**; C3 +6.6%. C1 − baseline = −0.59%/yr vs −0.5% margin → **REOPEN**. Trend and momentum halved their 2008–10 drawdowns vs holding; sector rotation didn't help; long_term_etf did worse than holding | Present to the owner; C1 stays unmerged; nothing live changes |
| `full_sweep_2026_10.json` — 96 candidates, best return at −15% | Recommend TREND6-EW-U8 (7.0% / −9.6%); in-sample best DUALMOM6-K4-U8 (11.0% / −14.6%); selection adds no reliable value; no universe expansion | Trend 6 built on a branch; dual momentum becomes a pod candidate |
| `pod_dualmom_hist.json` — dual momentum, historical (contaminated) | Alpha over holding the same 8: +5.6%/yr, t 2.79 (best of 96, so not beyond luck) | Information only |
| `pod_dualmom_forward.json` — dual momentum, forward shadow | Shadow from the 2026-09-30 close; judged 2027-09-30 or at 30 trades, whichever is later. At the historical median of 17 trades/yr, 30 trades takes a median of 20 months (80th pct 24): **expect judging around mid-2028** (`pod_dualmom_trade_rate.out`) | Pending |
| `trend6_small_mid_caps.json` — add mid-cap (IJH) and small-cap value (IJS) sleeves to trend 6 | U10 +7.1% / −10.6% vs BASE +7.0% / −9.6% (+0.06%/yr, 2 of 4 periods better, deeper DD); U9-IJH +7.2%, U9-IJS +6.9% | **Keep U8**: small/mid caps don't improve the core |
