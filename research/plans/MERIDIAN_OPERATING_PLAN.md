# Meridian operating plan — run it like a hedge fund

Draft 2026-09-27. Evidence: `research/2026-09-strategy-review/`, `research/2026-09-returns-study/`,
`research/plans/`. Claims are marked: **[data]** measured here, **[source]** looked up,
**[unverified]** general knowledge not checked, **[reasoning]** mine.

## 1. Where Meridian stands

- **[data]** Honest backtest (hindsight picks removed), 2011–2026/06: 8.0%/yr, Sharpe 1.00,
  max drawdown −15%. The live picks' 14% includes about 6 points of hindsight.
- **[data]** Beta to SPY is 0.34. Alpha is +1.9%/yr, t = 1.4, which is not significant. The
  return is exposure, not skill.
- **[data]** No sleeve shows honest skill. Three single-stock sleeves (32.5%) sit mostly in cash.
  Sector rotation, momentum and trend each trail holding their own instruments by 2–5%/yr in
  every period. The long-term ETF rotation is mixed. SVXY is worse than SPY since 2018.
- **[data]** The first skill-lab test (Faber trend rule, pre-registered): alpha +2.07%/yr,
  t 1.68, not passed. It cut drawdown to a third; it did not add return.
- **[data]** Capital is not the constraint: fractional shares, no commissions, −3.9 bps average
  slippage, and no fills under $25 since the monthly resize limit went in on 9/11.

**[reasoning]** Meridian has the shape of a multi-strategy fund, with sleeves as pods and a
graduation pipeline. It is missing the discipline: it pays pods for beta, gives capital
before evidence, and has no pod-level risk limits.

## 2. How a multi-strategy fund is run, and what Meridian copies

**[unverified]** Multi-manager funds (the "pod" model) broadly work like this. A central book
holds the risk. Portfolio managers get capital for *alpha*, not market exposure, and are
often required to hedge beta out. Capital scales with a track record. Drawdown limits are
hard: a pod is cut back at a set loss and closed at a deeper one. An independent risk
function signs off. Exact thresholds vary by firm and are not checked here.

| Hedge-fund practice | Meridian today | Meridian under this plan |
|---|---|---|
| Beta is cheap and held centrally | Every sleeve carries its own beta | **Core book** delivers the fund's market exposure at a drawdown target the owner picks |
| Pods are paid for alpha | Sleeves judged on CAGR | Pods judged on **alpha over their own benchmark** (Newey-West t), reported monthly |
| Capital follows evidence | Picks chosen by 10-year backtest | A pod gets capital only after **passing a pre-registered skill-lab plan and 6+ months of shadow trading** |
| Hard drawdown limits per pod | None | Pod at **−10%** from its peak → capital halved; at **−15%** → back to shadow |
| Risk budgets | Fixed % of equity | Pods sized by **volatility budget**; alpha pods ≤ 30% of risk until live evidence builds |
| Independent validation | Same session builds and grades | **Skill lab**: committed plans, sealed hold-outs, one final run, logged trials |

The drawdown and risk numbers are my proposals **[reasoning]**, for the owner to set.

## 3. Proposed structure

1. **Core book (most of the capital).** One simple, rule-based allocation that sets the
   fund's market exposure and drawdown. It is the benchmark every pod must beat, and it is
   chosen by the pre-registered comparison in §4.
2. **Alpha pods (start at zero capital).** Each is a hypothesis with its own plan in
   `research/plans/`. Candidates now: Faber trend (shadow from 2026-10-01, judged 2027-09-30)
   and post-earnings drift on large caps (plan not written yet).
3. **Cash/T-bills.** Anything not in the core or a funded pod sits in SGOV.
4. **Retired now:** the single-stock sleeves (TRGP, SNOW, WFRD) and SVXY. They go back to
   research status, not deleted.

## 4. The pre-registered structure test (`meridian_structure_2026_10.json`)

This does not search for skill. It asks: **can a simple core replace today's eight sleeves
without losing risk-adjusted performance?** It is a non-inferiority test, fixed before running:

- **Baseline:** today's structure, honest version (fund B, `research/2026-09-returns-study`),
  with the SGOV sweep.
- **Candidates (three, all fixed now):**
  - C1: 50% 60/40 SPY/IEF, 25% long-term ETF rotation, 25% T-bills.
  - C2: 50% Faber 5-ETF rule, 25% long-term ETF rotation, 25% 60/40 SPY/IEF.
  - C3: today's weights with the timing sleeves held instead of timed and the no-edge
    sleeves in T-bills. **This one is post-hoc** — it was seen in the returns study.
- **Rule:** a candidate is **non-inferior** if, in each of 2011–15, 2016–20 and 2021–26/06,
  its Sharpe is no more than 0.05 below the baseline's and its max drawdown no more than 2
  points worse. Among non-inferior candidates, fewer sleeves wins; then higher Sharpe.
- **Honesty note:** all of 2011–2026 has been seen during today's work. This compares
  structures on known history; it is **not a skill claim**. The live record from the switch
  date onward is the real test.

## 5. What the owner decides

1. The drawdown the core should target: about −15% (today), about −20%, or about −25%.
   Section 4 matches the baseline's risk; a different target rescales the core.
2. Whether to retire the single-stock sleeves and SVXY now.
3. The pod rules in §2 (thresholds and risk share).
4. When to switch: now, or at the ~10/11 monthly review.

Nothing changes live until the owner signs off.
