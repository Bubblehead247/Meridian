# Where can Meridian's returns improve? — 2026-09-27

Scripts in this folder, outputs in the matching `.out` files. Total returns (`adj_close`),
10 bps per side, long-only, T-bills = ^IRX. Sources: your data unless marked.

## Decided so far (2026-09-27)

- Exit rounding bug fixed (87e1a44 / 5508885). The 9/11 changes reviewed and committed.
- Research backtests are long-only by default (fe9fca1).
- Live picks unchanged: honest re-selection did worse (option C).
- Flat single-stock sleeves park their cash in SGOV (option B, 12553fb, live from 9/28).
- Walk-forward re-selection (option D) skipped. Everything merged to master and pushed.

## 1. How much of the backtest is hindsight (step1_baseline)

Fund B swaps each hindsight choice for one fixed before looking: momentum megacap15 → the same
model on SPY QQQ IWM EFA EEM GLD IEF TLT; trend XLK → SPY; TRGP/SNOW/WFRD sleeves → T-bills.
No point-in-time index membership exists beyond 2013–2018 (survivorship_study_summary.md), so an
ETF set is the honest substitute.

| Portfolio, 2011–2026/06 | CAGR | Vol | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| **B: honest baseline** | +8.0% | 8.0% | 1.00 | −15.0% |
| A: live picks as run (hindsight) | +14.0% | 10.6% | 1.29 | −16.1% |
| SPY | +14.0% | 17.1% | 0.86 | −33.7% |
| 60/40 SPY/IEF | +9.7% | 10.0% | 0.98 | −21.0% |

**About 6 of the 14%/yr is hindsight.** Honest expectation: ~8%/yr.

Decomposition (daily regression on SPY and IEF): fund B has beta 0.34 to SPY and alpha
+1.9%/yr, t = 1.4 — not significant. Per period: +0.1%, +0.9%, +2.7%. **Meridian's return is
mostly its market exposure, about a third of the stock market.**

## 2. Levers, on the honest baseline (step2_levers; pass rule fixed before running)

| Lever, 2011–2026/06 | CAGR | Sharpe | Max DD | Verdict (all 3 periods) |
|---|---:|---:|---:|---|
| **Baseline (honest fund B)** | +8.0% | 1.00 | −15.0% | — |
| L1 cash reserve + vol-sleeve idle → T-bills | +8.1% | 1.01 | −15.0% | free (tiny) |
| L1 + no-edge 32.5% → 60/40 SPY/IEF | +10.8% | 1.02 | −20.0% | risk trade |
| L1 + no-edge 32.5% → SPY 200-day trend | +10.7% | 0.95 | −20.3% | risk trade |
| L1 + no-edge 32.5% → long_term_etf | +12.3% | 0.95 | −22.7% | risk trade |
| L1 + no-edge 32.5% → SPY | +12.3% | 0.98 | −24.3% | risk trade |
| L1 + no-edge 32.5% → honest momentum | +9.5% | 0.84 | −24.9% | risk trade |
| L1 + vol target 12% (≤1.5x, borrow T-bill+1.5% — assumed rate) | +8.8% | 0.83 | −17.3% | risk trade |
| L1 + vol target 10% | +8.2% | 0.85 | −15.4% | fails |

Timing sleeves compared with holding the same instruments (CAGR difference per period):

| Sleeve | 2011–15 | 2016–20 | 2021–26/06 |
|---|---:|---:|---:|
| long_term_etf (QQQ/IEF/KMLM rotation) | −3.2% | +0.7% | +12.8% |
| sector_rotation | −2.1% | −1.6% | −2.0% |
| momentum on the honest ETF set | −2.0% | −5.3% | −2.8% |
| trend following on SPY | −5.1% | −4.3% | −3.6% |

Volatility sleeve: SVXY's daily vol fell from 61% (2012–17) to 37% after April 2018, which fits
its Feb-2018 change (from memory it went to −0.5x; unverified). After the change the vix_band
sleeve made +10.4%/yr, Sharpe 0.52, max DD −37.7%. SPY made +15.0%, 0.82, −33.7%.

## 3. Post-hoc: hold instead of time (step3_posthoc_holds)

This was added **after** seeing the table above, so it is weaker evidence.

| Portfolio | 2011–15 | 2016–20 | 2021–26/06 | 2011–26/06 CAGR / Sharpe / DD |
|---|---:|---:|---:|---|
| **Baseline (honest fund B)** | +5.5% | +8.0% | +10.3% | +8.0% / 1.00 / −15.0% |
| Sector, momentum, trend sleeves → equal-weight holds | +6.3% | +9.3% | +11.4% | +9.1% / 1.09 / −16.3% |
| … and no-edge 32.5% → 60/40 | +9.5% | +12.6% | +12.9% | +11.7% / 1.05 / −22.0% |

## Reading (my reasoning)

1. No timing layer in Meridian shows honest evidence of adding return. The one exception is
   the long-term ETF rotation, whose gain is almost all 2022 (it sat out the bond crash).
   Meridian's own earlier studies say the same about mean reversion, sector spreads, intraday
   and crypto.
2. The timing mostly buys lower drawdown. On a risk-adjusted basis it is roughly a wash; on
   return it costs 1–5%/yr per sleeve.
3. So the real lever is **how much market exposure to hold**, not better signals. Every
   return-raising option here is a trade of drawdown for return. Which point to pick is the
   user's call.
4. Paper caveat: T-bill and dividend income shows about 0 on Alpaca paper, which pays no
   dividends. Equity-exposure levers do show on paper.

## 5. Return at a fixed drawdown budget, 2008 included (step4_return_at_budget)

Every earlier table in this study starts in 2010/2011 and so **misses 2008**. Here each portfolio
is scaled with T-bills (k < 1) or margin (k > 1; borrowing at T-bill + 1.5%, an assumed rate) so
that its worst drawdown from 2008-01 to 2026-06 hits the budget. k is calibrated on the same
history, so compare rows with each other, not with a promise.

| Portfolio | At −15% | At −20% | At −25% | At −30% |
|---|---:|---:|---:|---:|
| **C1 core (baseline for the switch)** | +6.5% (k 0.78) | +8.1% (k 1.05) | +9.4% (k 1.32) | +10.6% (k 1.60) |
| Faber on SPY alone | +5.9% | +7.2% | +8.4% | +9.4% |
| QQQ | +5.6% | +7.0% | +8.4% | +9.9% |
| 60/40 SPY/IEF | +4.8% | +5.9% | +7.1% | +8.3% |

C1 unscaled had a −19.0% drawdown with 2008 included (it was −15.2% from 2011). **At a true −15%
through a 2008-type crisis, the C1 core is about 78% invested, with the rest in T-bills, and the
expected return is about 6.5%/yr.** C1 still gives the most return per unit of drawdown of the
standard portfolios tested. Each extra 5 points of drawdown adds about 1.3–1.6%/yr. Beyond
about −20%, C1 needs margin (k > 1).
