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

## 5. Review against real hedge-fund practice (2026-09-27)

Sources are **[source]** and mostly secondary (trade press, industry write-ups): the funds do
not publish these rules. Links are at the end of this section.

| Practice | What real funds do | This plan (first draft) | Verdict / change |
|---|---|---|---|
| Beta and alpha kept apart | Bridgewater split beta (All Weather, risk parity) from alpha (Pure Alpha); client portfolios end up roughly 70/30 beta/alpha. Pod shops require each pod to be close to **beta-neutral** (net exposure about −20% to +20%, factor limits on top) | Core book for beta; alpha pods ≤30% of risk | **Matches.** The 70/30 split is in line with Bridgewater's |
| Pod drawdown stops | Millennium (reported): **−5% halves the pod's risk, −7.5% closes it**. Balyasny acts at −3% to −5%. Elsewhere set per PM. There is no high-water-mark cushion: past gains don't protect against new losses | −10% halve, −15% back to shadow, on pod return | **Change.** Real stops work because the pods are beta-neutral: a loss means the *skill* failed. A long-only Meridian pod would hit −10% in any market sell-off. **Measure stops on the pod's alpha P&L (return minus beta × benchmark), at −5% (halve) and −7.5% (back to shadow).** |
| Track record before capital | Incubators run on own money for 6–12 months; allocators want 12–24 months live | 6+ months shadow | **Change.** Six months is too short for monthly-rebalanced rules (a handful of decisions). **Require 12 months of shadow, or at least 30 independent trades, whichever is later.** |
| Capital follows performance | Capital is merit-based: cut on losses, raised with track record. Millennium turns over about 15–20% of PMs a year | Capital given on passing; no ramp | **Add a ramp.** A funded pod starts at ⅓ of its target risk, goes to ⅔ after 6 months live if alpha is positive and no stop was hit, then full after 12 months. Expect most pods to fail. |
| Independent risk and validation | A central risk team (at Citadel it reports to the CEO) aggregates exposures and enforces limits | Skill lab | **Matches in spirit.** Add a fund-level check to each monthly review: total beta, pod correlations, and the gross exposure limit |
| Leverage | Fund-level gross leverage about 5.7x (Citadel) and 6.7x (Millennium) in 2023, applied to low-volatility, beta-neutral alpha | None | **Keep none.** Leverage only makes sense on proven, beta-neutral alpha, and Meridian has none yet |
| Where the edge comes from | Pod shops: idiosyncratic stock-picking by dozens to hundreds of teams with heavy data and analytics. AQR and academic work: most hedge-fund return is **alternative risk premia** (value, momentum, carry, defensive, trend, volatility) across asset classes | Pods = any passing hypothesis | **Refocus.** Meridian can't copy a pod shop's stock-picking scale. The realistic pod pipeline is **diversified risk premia and event effects** (trend across asset classes, cross-asset momentum, carry, post-earnings drift), tested through the lab |
| Long/short | Premia and pods are long/short, so they are close to market-neutral | Long-only | **Open question for the owner.** Long-only premia carry beta. A beta hedge (an SPY short, or an inverse ETF) would make pod alpha measurable and stops meaningful. Needs its own plan and a check of broker support |
| Costs | Multi-managers pass through up to about 8%/yr of expenses; Balyasny's flagship made 15.2% gross and 2.8% net in 2023 | No fees | **Meridian's edge over a real fund:** market exposure at almost no cost |
| Realistic results | Multi-manager beta to the S&P about 0.03, versus 0.24 for hedge funds broadly (10 years to 3/2024). Most hedge funds trailed the S&P 500 in 2024; Pure Alpha made 11% | Honest Meridian: beta 0.34, alpha not significant | Meridian today is a **balanced fund**, not a hedge fund. It becomes hedge-fund-like only when pods produce significant alpha at low beta |

**Net amendments (proposed):** the core book stays, holding about 70% of risk. Pod stops move to the
pod's *alpha* P&L at −5% / −7.5%. Shadow is 12 months or 30 trades before any capital. Funded pods
ramp ⅓ → ⅔ → full. The pipeline focuses on diversified risk premia and event effects. Beta
hedging for pods is the owner's decision.

Sources: [Pod shop risk limits](https://hedgefundinterview.com/pod-shop-risk-limits) ·
[Millennium multi-strategy architecture](https://navnoorbawa.substack.com/p/millennium-managements-multi-strategy) ·
[How PMs get cut at multi-managers](https://youngandcalculated.substack.com/p/how-pms-actually-get-fired-at-multi) ·
[Multi-manager 101 (The Diff)](https://capitalgains.thediff.co/p/multimanagerpodhedge-fund-101) ·
[Multi-manager overview incl. Morgan Stanley beta data](https://hedgefundinterview.com/multi-manager-hedge-funds) ·
[Bridgewater (alpha/beta separation)](https://en.wikipedia.org/wiki/Bridgewater_Associates) ·
[Bridgewater, The All Weather Story](https://www.bridgewater.com/research-and-insights/the-all-weather-story) ·
[AQR, Understanding Alternative Risk Premia](https://www.aqr.com/Insights/Research/White-Papers/Understanding-Alternative-Risk-Premia) ·
[Hedgeweek on pass-through fees](https://www.hedgeweek.com/hedge-funds-charging-billions-in-no-limit-passthrough-fees/) ·
[Bloomberg on pass-through fees](https://www.bloomberg.com/graphics/2025-hedge-fund-investment-fees/) ·
[Incubator track records](https://www.investmentlawgroup.com/perspectives/launching-an-incubator-hedge-fund/) ·
[Hedge funds vs the S&P in 2024](https://www.tipranks.com/news/hedge-funds-underperformed-the-bull-market-in-2024)

## 6. Owner decisions (2026-09-27)

- **Drawdown target: −15%**, today's level. The core book is sized to it.
- **Single-stock sleeves retired** (breakouts/TRGP, mean_reversion/SNOW, pullback/WFRD):
  `"retired": true` in `live_picks.json`. From the 9/28 session they never trade, and their
  32.5% sits in SGOV. SVXY stays for now.
- **Amended pod rules accepted** (§5): stops on alpha P&L at −5% (halve) and −7.5% (back to
  shadow); 12 months or 30 trades of shadow before capital; ramp ⅓ → ⅔ → full.
- **−15% confirmed after the 2008 correction** (returns study step 4): the core is C1 scaled to
  about 78% invested (rest in T-bills), expected about 6.5%/yr through a 2008-type crisis.
- **2026-09-28: the C1 decision is REOPENED** by the pre-registered 2008 check
  (`structure_2008_check.json`). At a matched −15% drawdown over 2008–2026 the 8-sleeve
  baseline makes 7.1%/yr and C1 6.5%. Trend and momentum halved their 2008–10 drawdowns;
  sector rotation and long_term_etf did not help. This is a near-tie on one path's worst
  drawdown, but it removes the case for retiring trend and momentum. **Next candidate (corrected
  2026-09-28 after the owner asked why 60/40 would help):** not 60/40 + timing. At a −15% cap,
  60/40's 2008 drawdown (−30.8%) is what forced C1 down to 78% invested. Pre-register the honest
  8-sleeve fund with sector rotation and SVXY retired to T-bills, against the full honest fund,
  both at a matched −15% drawdown over 2008–2026. Claude's review: `research/2026-10-review/claude_review/`.
- Still open: switch timing for the core book, and whether pods hedge their beta.

## 7. Original decision list

1. The drawdown the core should target: about −15% (today), about −20%, or about −25%.
   Section 4 matches the baseline's risk; a different target rescales the core.
2. Whether to retire the single-stock sleeves and SVXY now.
3. The pod rules in §2 (thresholds and risk share).
4. When to switch: now, or at the ~10/11 monthly review.

Nothing changes live until the owner signs off.
