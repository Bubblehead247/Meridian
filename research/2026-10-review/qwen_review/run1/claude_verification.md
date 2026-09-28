# Qwen3.6-27B review — Claude's verification (2026-09-28)

Run: thinking on (temp 0.6, top_p 0.95, top_k 20), 3,724 prompt tokens, 6,770 output tokens
(18.7k chars of reasoning), 391 s. Output: `response.json` (valid JSON, 8 findings).
Every figure it cites appears in `packet.md`; no invented numbers.

| # | Qwen's finding | Verdict | Why |
|---|---|---|---|
| F1 high | Hindsight-biased sleeve selection | **Agree** | Matches §4a/4d. Its fix (out-of-sample validation) is what the skill lab already does |
| F2 high | Too complex, capital idle, switch to C1 "immediately" | **Agree on diagnosis, not on "immediately"** | Three live changes landed 9/28; switching before they've run repeats the pile-up the owner froze SeykotaBot over |
| F3 high | SVXY "violates the −15% mandate" (sleeve DD −37.7%) | **Right conclusion, wrong reasoning** | The −15% is a fund-level limit. A 5% sleeve with a −37.7% drawdown costs the fund about 1.9 points. SVXY should go because it trails SPY on return and drawdown since 2018 |
| F4 med | CORE_EXPOSURE 0.78 is in-sample; start at 0.70 | **Valid, owner's call** | 0.78 was fitted on the same history. A margin for a crisis worse than 2008 is reasonable; 0.70 means roughly −13.5% drawdown and ~5.9%/yr (my estimate, scaling the step-4 figures) |
| F5 med | Next-open fills add gap risk; use limit or after-hours orders | **Disagree** | Measured slippage is −3.9 bps, i.e. favourable. For a monthly-rebalanced ETF core, gap risk at the open is noise |
| F6 med | Paper lacks income; deploy the total-return tracker now | **Agree** (with one misquote) | The live book already holds SGOV (the retired sleeves), so the tracker is useful before C1. It misquoted C1 as "~78% bonds/dividend ETFs/T-bills" (the packet says about three-quarters; 0.78 is CORE_EXPOSURE) |
| F7 low | Mandate beta hedges for pods | **Plausible, still the owner's open question** | Matches real multi-manager practice (packet §8). It doesn't matter until a pod is funded |
| F8 low | Timing sleeves don't beat holding | **Agree** | Matches §4c |

**Useful additions (missed risks):**
- Stock and bond correlation can spike, as in 2022 when both fell. C1's backtest includes 2022 and stayed within −15.2%, but the risk is real.
- Paper fills are simulated, so real spreads and market impact are unknown until the account goes live.
- There are tax effects if Meridian ever runs in a taxable account.

**Grades it gave:** strategy C, risk B, execution B, research process A, governance A.
"Governance A" is generous: the review/plan discipline only started on 9/27.
