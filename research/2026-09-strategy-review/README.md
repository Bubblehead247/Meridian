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

The backtest reproduces the account (+0.27% vs −0.03%) once it is made long-only like live —
so the backtest below can be trusted to describe what the live rules do. Three months is far too
short to judge anything by itself.

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
| Same instruments, same weights, buy-and-hold | +14.9% | −27.0% | — |
| SPY | +12.0% | −34.1% | 0.75 |
| QQQ | +18.0% | −35.6% | 0.90 |
| 60/40 SPY/IEF | +7.6% | −21.9% | 0.78 |

At 1 bps Meridian is +14.0%, at 25 bps +10.7%. Realized slippage on 204 live fills averages
−3.9 bps (mostly overnight-gap noise), so 1–10 bps is the right range.

By period (Meridian 10 bps next-open vs SPY): 2011–15 +8.1% vs +10.2%; 2016–20 +12.1% vs
+12.9%; 2021–26/06 +17.8% vs +13.0%.

**Reading:** even with hindsight-chosen picks, the timing rules earn ~2%/yr *less* than simply
holding the same instruments, in exchange for ~10 points less drawdown. The rules are a
drawdown tool, not a return source.

## 4. Findings in the code

1. **Fixed (87e1a44): exits rejected over rounding.** Books keep 6 decimals, Alpaca 9. AVGO's
   exit failed 11 sessions running from 9/11 (AAPL, XLV for days in August); 45 failed orders
   in the log overall.
2. **Not fixed: the research backtest books short signals that live never trades.**
   `Model.backtest` uses raw signals; live is long-only ("shorts … go flat").
   `rsi_exhaustion` on SNOW has 197 short days vs 182 long; its backtest drops from +30.7%/yr to
   +16.9%/yr long-only. The gauntlet/pipeline that chose the picks uses `Model.backtest`, so any
   model that emits shorts was selected on returns it can't earn.
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
