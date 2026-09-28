# C1 core switch — ready for review (branch `c1-core-2026-10`, not merged)

Built 2026-09-27 for the ~10/11 review. Nothing here is live until the owner merges it.

## What changes

| Piece | Change |
|---|---|
| Allocation | `SLEEVE_ALLOCATIONS` is now C1 at `CORE_EXPOSURE = 0.78`: core_equity SPY 23.4%, core_bonds IEF 15.6%, long_term_etf 19.5%, core_tbills SGOV 41.5%. Every old sleeve stays listed at 0%. The 9-sleeve table lives on as `RESEARCH_SLEEVE_ALLOCATIONS` for the research fund tools. |
| Models | New `core_equity` / `core_bonds` / `core_tbills` families, each `buy_and_hold` (always long its one ETF), with "always permitted" regime rules. |
| Picks | `live_picks.json`: the three core picks added. momentum, sector_rotation, trend_following and volatility are marked `retired`, so their holdings are sold. |
| Monthly resize limit | A change in a sleeve's allocation now makes a resize due that day (the stamp records the weight). Without it, a mid-month switch would leave long_term_etf at its old 25% until November. |
| No-short guard | A sell of a symbol the broker holds none of is skipped and retried next session. It happens when the books count a buy that hasn't filled yet; a margin account would otherwise open a short. **Also worth putting on master now.** |
| Total-return tracker | `meridian/execution/total_return.py` runs after each real session. It logs the income paper doesn't pay (dividends, coupons, T-bill yield) and `tr_equity`, the number to judge C1 by. Output: `ledger/shadow/total_return.jsonl`. |

## Switch-day preview (`c1_switch_preview.py`, read-only, simulated fills)

See `c1_switch_preview.out`. Run on 2026-09-27 from the real account. It sells every retired
holding and buys SPY/IEF/SGOV, and QQQ goes from 3.57 to 2.61 shares. It lands exactly on the
C1 weights (SGOV 41.5%, SPY 23.4%, QQQ 19.5%, IEF 15.6%). Friday's two unfilled orders (NVDA sell,
LLY buy) show up as NVDA left over and a skipped LLY sell; they clear on 9/28. **Re-run the
preview on the switch day.**

## Checks done

- 886 tests pass, run with `NTFY_TOPIC=test-dummy`. The worktree has no `.env`, and the notify
  tests read the topic from `.env` (a pre-existing test-isolation flaw).
- New tests fail on the old code: the allocation-change resize, the no-short guard, and the
  total-return income.
- `meridian run-paper --dry-run` and `meridian chart` run cleanly on the branch. The chart
  uses price-only data, so IEF and SGOV look flat there; the total-return tracker is the fair
  measure.

## Known limits

- `CORE_EXPOSURE = 0.78` was fitted on 2008–2026, the same history it was judged on. Revisit it
  at each review.
- Live rebalancing is monthly with a 5% band; the backtest held weights fixed daily. The
  difference is small.
- `meridian review` (the monthly paper-strategy review) wasn't exercised on the branch.

## To switch (owner's OK first)

1. Re-run `python research/plans/c1_switch_preview.py` from the worktree the day before.
2. Merge `c1-core-2026-10` into master (the live folder runs master).
3. The next 15:32 CT session sends the switch orders. Check the next morning:
   `python -m quantcore.watch`, `ledger/shadow/total_return.jsonl`, and account weights near C1.
