# Meridian strategy review — 2026-09-27

Script: `backtest_review.py` (run with `PYTHONPATH=.` from the repo root), output in `backtest_review.out`.
Sources: account PA3HZJEMP4IZ (Alpaca paper), `meridian.log`, `ledger/trade_log.jsonl`, and
backtests of the current `live_picks.json` on uncached daily bars through 2026-09-25.

**Caveat up front:** every pick was chosen on 2026-06-28 by 10-year backtest return (commit
8acf51a: "10yr +1241%", "Combined fund $10k → $87k"). Everything before 2026-06-28 is in-sample.
These are best-case numbers, not forecasts.

## 1. Live record (6/25–9/25)

| Portfolio | Return |
|---|---:|
| Meridian account | −0.03% ($10,000 → $9,997) |
| Backtest of live picks, same window | +0.27% |
| SPY | +5.81% |
| QQQ | +5.38% |
| Same instruments, same weights, buy-and-hold | +9.73% |

The totals are close, but that is partly luck: live ran different weights until 9/11 (trend
following 15%, breakouts monitor-only) and had stuck exits. The real evidence that the backtest
describes the live rules is per sleeve: time in market 7/31–9/25 matches the live log (SNOW 0% in
both once long-only, WFRD ~15% in both, TRGP close). Three months is far too short to judge
anything by itself.

## 2. Why ~40% sits in cash — by design, not a bug

Share of time each sleeve is invested (backtest 2011–2026/06; live log 7/31–9/25 agrees):

| Sleeve | Weight | In market |
|---|---:|---:|
| pullback_continuation (WFRD) | 10% | 3% |
| mean_reversion (SNOW, long-only) | 15% | 13% |
| volatility (SVXY) | 5% | 25% |
| breakouts (TRGP) | 7.5% | 37% |
| trend_following (XLK) | 7.5% | 84% |
| long_term_etf, momentum, sector_rotation | 50% | 100% |
| cash_reserve | 5% | 0% |

Average ≈ 63% invested. Unlike MeansRev (one filter idling 87%), nothing is blocked: the
single-stock sleeves simply rarely signal.

## 3. Long-run backtest, 2011 – 2026-06 (in-sample)

| Portfolio | CAGR | Max DD | Sharpe |
|---|---:|---:|---:|
| Meridian live picks (10 bps, next-open fills) | +12.7% | −17.0% | 1.15 |
| Same instruments, same weights, buy-and-hold | +14.9% | −27.0% | 0.98 |
| SPY | +12.0% | −34.1% | 0.75 |
| QQQ | +18.0% | −35.6% | 0.90 |
| 60/40 SPY/IEF | +7.6% | −21.9% | 0.78 |

At 1 bps Meridian is +14.0%, at 25 bps +10.7%. Realized slippage on 204 live fills averages
−3.9 bps (mostly overnight-gap noise), so 1–10 bps is the right range.

| Period | Meridian (10 bps) | Same-instrument B&H | SPY |
|---|---:|---:|---:|
| 2011–2015 | +8.1% | +10.6% | +10.2% |
| 2016–2020 | +12.1% | +12.8% | +12.9% |
| 2021–2026/06 | +17.8% | +21.1% | +13.0% |

The B&H lead in 2021–26 is inflated by WFRD's +63%/yr run (pure hindsight), but B&H also leads in
2011–15 and 2016–20, before WFRD or SNOW existed.

**Reading:** even with hindsight-chosen picks, the timing rules earn ~2%/yr *less* than simply
holding the same instruments, in exchange for ~10 points less drawdown. The rules are a
drawdown tool, not a return source.

## 4. Findings in the code

1. **Fixed (87e1a44, tolerance 1e-6 in 5508885): exits rejected over rounding.** Changes live
   order behavior from the Monday 9/28 session; checked read-only against the account (AVGO held
   0.051938999 → the sell is trimmed to exactly that). Books keep 6 decimals, Alpaca 9. AVGO's
   exit failed 11 sessions running from 9/11 (AAPL, XLV for days in August); 45 failed orders
   in the log overall.
2. **Not fixed: the research backtest books short signals that live never trades.**
   `Model.backtest` uses raw signals; live is long-only ("shorts … go flat").
   `rsi_exhaustion` on SNOW has 197 short days vs 182 long; its backtest drops from +30.7%/yr to
   +16.9%/yr long-only. The selection pipeline (`pipeline/backtest.py`, `oos.py`,
   `walk_forward.py`, `universe.py`, `experiments/fund.py`) all call `model.backtest`, so any
   model that emits shorts was selected on returns it can't earn. **Fixed in fe9fca1** (see §6).
   Correction: only the `fund` command builds a `regime_frame`; the gauntlet and `pipeline`
   commands pass none, so the regime gate was not part of pick selection.
3. Pullback (WFRD) holds 10% of capital and is invested 3% of the time (+8.5%/yr since 2021 vs
   +63%/yr for holding WFRD).

## 5. Options (nothing changed)

- **A. Leave as is.** It is paper, and the drawdown profile is its selling point.
- **B. Put idle sleeve cash to work** (e.g. into the long-term ETF sleeve or a T-bill ETF while
  flat, like MeansRev's SGOV sweep). Needs its own backtest.
- **C. Make research long-only** where live is long-only (fix item 2), then re-run pick
  selection.
- **D. Walk-forward pick selection** (choose on data to year X, test after X) for an honest
  out-of-sample estimate. The biggest study, and the only way to know whether the picks
  have any real edge.

## 6. Option C done (2026-09-27): long-only engine, then re-selection

**Engine (fe9fca1):** `Model.backtest` and `CrossSectionalModel.backtest` hold short signals
flat unless `allow_shorts=True`; cross-sectional shorts are dropped before sizing so longs share
the sleeve, as live does. 16 of 52 models emit shorts. Cross-checked: `build_fund_returns` now
gives SNOW +18.9%/yr (same as this review's script) and the fund 14.2% (script 14.1%, cached
vs uncached bars). The 358 records in `saved_strategies/` were scored before this change and
are stale for any model that emits shorts.

**Re-selection** (`reselect_picks.py`, output `reselect_picks.out`, all rows `reselect_all.csv`).
Method fixed before looking: the 6/28 candidate symbols per family (crypto excluded; VRT, USO
skipped for corrupt cache), every single-asset model of the family, long-only, 10 bps,
next-open fills; select by 2011–2020 Sharpe (≥5 years, ≥10 entries); judge on 2021–2026/06.

| Family | Current pick (chosen 2026 — not out-of-sample) | Honest pick (2011-20 Sharpe) | Its 2021–26/06 CAGR, Sharpe | IS→OOS rank corr. |
|---|---|---|---:|---:|
| breakouts | turtle_ma_exit / TRGP: +26.6%, 1.13 | turtle_ma_exit / SHOP | +10.5%, 0.49 | +0.02 |
| mean_reversion | rsi_exhaustion / SNOW: +12.9%, 0.60 | rsi_reversion_ma_filter / ITB | +2.5%, 0.44 | +0.29 |
| pullback | rsi_pullback / WFRD: +7.9%, 0.87 | rsi_pullback_50_200 / LITE | +0.4%, 0.11 | +0.17 |
| trend_following | ma_trend_long_only / XLK: +20.0%, 1.07 | ma_trend_long_only / SHOP | +10.1%, 0.44 | +0.16 |
| **Baseline: SPY** | | | +13.5%, 0.83 | |
| QQQ | | | +16.2%, 0.78 | |

How often timing beat simply holding the same symbol in 2021–26/06, across all eligible
candidates: breakouts 36% (median +5.3% vs +13.5% holding), mean reversion 21% (+3.6% vs
+7.9%), pullback 16% (+0.8% vs +11.6%), trend following 10% (+3.8% vs +14.0%).

**Reading:**
- In-sample Sharpe barely predicts out-of-sample Sharpe (rank correlation +0.02 to +0.29).
  The selection method has little skill; the current picks look good after 2021 only because
  they were chosen knowing 2021–2026.
- Honestly-selected picks all trail SPY after 2021. Swapping to them would be worse, not better.
- **No change to `live_picks.json` recommended.** The finding is about the single-stock
  sleeves as a whole (40% of capital, mostly idle): on this evidence they don't earn their
  place over an index. That is a question for option B/D, not a pick swap.

## 7. Option B done (2026-09-27): idle single-stock sleeve cash -> SGOV

`option_b_idle_sleeves.py` (output `.out`): total returns (adj_close — the loader's `close` is
price-only, so every other backtest in this repo leaves dividends out), 10 bps, T-bills = ^IRX.
The single-stock sleeves (32.5%) were invested 39% / 5% / 1% of days 2011–2026/06.

| Variant, 2011–2026/06 | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|
| **Current (baseline)** | +13.5% | 1.25 | −16.1% |
| Idle cash → T-bills | +14.0% | 1.29 | −16.1% |
| Idle cash → SPY | +17.6% | 1.16 | −26.6% |
| Idle cash → long_term_etf | +17.6% | 1.17 | −23.3% |
| Replace the sleeves with SPY | +15.3% | 1.05 | −26.6% |
| Replace the sleeves with long_term_etf | +15.5% | 1.05 | −23.3% |

2021–2026/06 (higher rates): T-bills +18.8% vs +17.7% baseline, same drawdown. Sweeping beats
replacing because the sleeves' picks are hindsight-good; replacing is the honest-but-lower number.

**User chose T-bills.** `live_runner.IDLE_SWEEP_FAMILIES` = breakouts, mean_reversion,
pullback_continuation: a flat sleeve targets its whole sleeve in SGOV, and targets 0 SGOV the
day it goes long. Orders net per symbol like everything else; resizing a held SGOV position
follows the monthly throttle. Dry run 9/27: 32.28 SGOV (~$3,250) would be bought.
**Alpaca paper pays no dividends, and SGOV's return is its dividend — on paper this shows ~0.**
