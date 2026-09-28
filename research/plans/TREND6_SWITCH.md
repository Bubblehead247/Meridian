# Trend 6 core switch — ready for review (branch `trend6-core-2026-10`, not merged)

Built 2026-09-28 for the ~10/11 review, from the full sweep's recommendation
(`research/2026-10-sweep/README.md`). Nothing here is live until the owner merges it. It replaces
the C1 build, which was retired as the tag `archive/c1-core-2026-10`.

## The model

Eight sleeves, one ETF each, 12.5% of equity apiece: SPY, QQQ, IWM, EFA, EEM, GLD, IEF, TLT.
- At each month's last session, a sleeve holds its ETF if the dividend-adjusted month-end price is
  above the mean of its last 6 month-end prices; otherwise its 12.5% goes to SGOV.
- The decision is held all month.
- Every other sleeve is retired at 0% (their holdings are sold), including long_term_etf.

## What changed on the branch (on top of the C1 groundwork)

| Piece | Change |
|---|---|
| Model | `meridian/families/core/models.py`: `MonthlySmaTrend` registered as `core_trend_<etf>/sma6_monthly`. The final bar counts as a month-end only if it is the month's last session: quantcore's trading calendar when it covers the day, otherwise the next business day starting a new month. The fallback decides one session late when a month ends on a holiday |
| Signal prices | The model asks for **dividend-adjusted** prices for its signal (`signal_price_field`). The live runner fetches them for signals only, and sizing still uses the real close. Plain closes flipped 2.8% of decisions (mostly IEF, EFA and TLT) and cost 0.25%/yr (`trend6_signal_input_check.out`) |
| Allocation | `SLEEVE_ALLOCATIONS` = 8 × 12.5%; every old sleeve listed at 0 |
| Idle cash | The 8 trend sleeves join the SGOV sweep, so a flat sleeve holds SGOV |
| Kept from the C1 build | Retired-sleeve handling, same-day resize on an allocation change, the total-return tracker, and the research 9-sleeve table for the fund tools |
| Removed | C1's buy-and-hold families, the C1 switch doc and its preview output |

## Checks

- **Fidelity gate:** the live models, backtested one per ETF at 12.5% with idle cash in T-bills,
  give **7.02% / −9.64%**, exactly the sweep's TREND6-EW-U8 (`trend6_fidelity.out`).
- **Tests:** 894 pass (with `NTFY_TOPIC=test-dummy`; the worktree has no `.env`). New tests cover the
  month-end rule, the unfinished-month rule, adjusted signal prices, and SGOV when flat. They fail
  without the key logic.
- **Real commands run cleanly:** `meridian run-paper --dry-run`, `meridian chart`, `meridian review`.
  - Today's dry run shows six ETFs held, and IEF and TLT flat (in SGOV). These are August month-end
    decisions; September's come at the 9/30 close.
- **Switch preview** (`core_switch_preview.py`: read-only from the real account, simulated fills,
  output `core_switch_preview.out`, 2026-09-28): it sells every retired holding (AAPL, AVGO, LLY,
  META, MSFT, UNH, XLE, XLF, XLV, XLK, part of QQQ) and lands exactly on 12.5% × 6 ETFs + 25% SGOV.
  **Re-run it the day before switching.**

## What to expect (honest range)

| Universe (TREND6, EW) | CAGR 2008–26/06 | Max DD | Sharpe |
|---|---:|---:|---:|
| **U8, the one built (chosen 2026-09, includes QQQ and gold)** | 7.0% | −9.6% | 0.90 |
| U12, the published PAA list | 6.3% | −9.8% | 0.85 |
| U5, Faber's published list | 5.0% | −12.3% | 0.66 |

Neighbouring settings on U8 (10- and 12-month averages) give 6.5% and 6.3%: a plateau, not a spike.
U8 was chosen with hindsight, so **plan on roughly 5–7%/yr with drawdowns around −10 to −12%**, not
7.0% flat. Paper won't show SGOV's yield or the ETF dividends; judge it by the total-return
tracker (`ledger/shadow/total_return.jsonl`).

## Known limits

- Live resizing is monthly with a 5% band; the backtest held exact weights.
- The rule uses 10 bps costs; fills are next-open market orders.
- The −9.6% drawdown leaves part of the −15% budget unused under the no-leverage rule.

## To switch (owner's OK first)

1. Re-run `python research/plans/core_switch_preview.py` from the worktree the day before.
2. Merge `trend6-core-2026-10` into master (the live folder runs master).
3. The next 15:32 CT session sends the switch orders. Check the next morning:
   `python -m quantcore.watch`, account weights near 12.5% per held ETF, and SGOV for the flat ones.
