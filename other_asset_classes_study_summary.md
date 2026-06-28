# Other-Asset-Class Mean-Reversion Study (research pass 3)

**Status:** Complete — crypto (live Crypto.com REST), cointegration ETF pairs, and
sector/calendar spreads.
**Date:** 2026-06-25
**Scope:** Third and final planned research extension. Tests whether *other asset
classes* — more reversion-prone or differently-structured instruments — change the
cost verdict that killed every US-equity mean-reversion (MR) signal in passes 1–2.

---

## What was built

### Crypto loader — `meridian/data/crypto.py`
- `_download_crypto_candles` — isolated HTTP GET to Crypto.com's **public**
  candlestick endpoint (no API keys). `_download_crypto_history` paginates
  **backward** (the endpoint caps each call at ~300 candles) via `end_ts`, so
  multi-year daily history is assembled — **2002 daily BTC bars (2021→2026)** vs the
  300-bar single-call cap.
- `load_crypto` / `load_crypto_universe` — map candles to the canonical schema
  (`adj_close = close`, no crypto splits), cached through the existing `OHLCVCache`.
  Mirror the `load_intraday` shape exactly.
- `crypto_bars_per_year` — crypto trades **365 days, 24h**, so daily annualizes by
  **365** (not the 252 equity sessions); `4h → 2190`, `1h → 8760`.
- **Live-validated:** `load_crypto("BTC_USDT","1D")` OHLC matches the Crypto.com MCP
  `get_candlestick` tool candle-for-candle (only the still-forming current-day close
  differs, pulled seconds apart).

### Pairs/spread transform — `meridian/features/pairs.py`
The one genuinely new piece of *signal* machinery:
- `hedge_ratio` — **causal** rolling OLS slope β (trailing window, `shift(1)` so β at
  bar *t* uses only data through *t-1* — no lookahead, same discipline as the estimators).
- `build_pair` → `(spread, spread_price)`: the **signal** is `log P_a − β·log P_b`
  (zero-crossing, fed as the signal input like a relative series); the **synthetic
  price** is `cumprod(1 + (r_a − β·r_b))`, whose `pct_change` exactly reproduces the
  long-spread simple return so the existing price-based P&L path is reused untouched.
- `build_pairs` → `(signal_prices, spread_prices)` dicts keyed `"A/B"`, fed straight
  into `validate_universe` as `signal_prices_by_symbol=` / `prices_by_symbol=`. Each
  pair becomes **one synthetic "symbol"** in a universe of pairs, so the whole
  universe backtester/validator is reused with **zero changes**.

**Reuse, as planned:** estimators, deviation, signal engine, portfolio backtester,
walk-forward, block bootstrap, Monte-Carlo, BH correction — all unchanged. Only the
loader and the spread transform are new.

### Tests — `tests/test_crypto.py`, `tests/test_pairs.py`
Crypto loader (stubbed download → schema, cache round-trip, **backward pagination**,
`crypto_bars_per_year` values); pairs transform (β is causal/no-lookahead, recovers
true β, synthetic-price `pct_change` reconstructs the spread return, NaN handling,
end-to-end `validate_universe` on a pairs dict). **367 tests pass, lint clean** (was 354).

---

## Results

All runs: equal-weight L/S, anchored walk-forward, BH correction, `entry=1.5`.

### A. Crypto — 12 USDT majors, daily, 2021–2026 (`periods_per_year=365`)
| strategy | window | best OOS Sharpe (1 bp) | significant |
|----------|:------:|----------------------:|:-----------:|
| Absolute | 5/10/20 | −0.51 / +0.12 / +0.03 | No |
| Cross-sectional | 5/10/20 | +0.33 / +0.39 / **+0.53 (ou)** | No (q ≥ 0.35) |

Cross-sectional crypto produces *positive gross* Sharpes (like equities) but **none
clear BH correction**. The strongest cell is **cross-sectional `ou`, window 20**
(gross **0.57**, q=0.10 — not significant, turnover **0.55/bar**); its cost sweep:

| cost | OOS Sharpe |
|-----:|-----------:|
| 1 bp | +0.53 |
| 5 bps | +0.35 |
| 10 bps | +0.12 |
| 20 bps | −0.33 |
| 30 bps | −0.79 |

So the best crypto cell is **more cost-robust than I first reported** — it stays
gross-positive out to ~10 bps (the shorter, higher-turnover sma-w5 reversal cell dies
by 5 bps, turnover 0.89). **The reason crypto is not an edge is statistical
insignificance (q=0.10), not that costs instantly kill it.** Even so, **realistic retail
crypto cost is ~10–40 bps** (taker fees + wide spreads, *above* equities), which straddles
this cell's ~15 bp break-even — so net it is marginal *and* insignificant. **The
"lower-cost crypto" hypothesis is false; crypto provides no reliable edge.**

### B. Cointegration ETF pairs (twins / substitutes) — daily, 2015–2026
Only `VOO/SPY` is statistically cointegrated (Engle-Granger p≈0); the rest (GLD/IAU,
QQQ/QQQM, IWM/VTWO, XLE/VDE, DIA/SPY) are not (p > 0.13). All windows **negative gross**
(best −0.16). Twins are *too* tight — the spread is microstructure noise with nothing
to revert. Cost sweep deeply negative. **No edge.**

### C. Sector / calendar spreads — 6 pairs, daily, 2015–2026 — **the one hit**
`XLK/XLY, XLE/XLB, XLF/XLI, XLU/XLP, SPY/RSP, QQQ/SPY`.
| window | best estimator | OOS Sharpe (1 bp) | q-value | significant |
|:------:|----------------|------------------:|--------:|:-----------:|
| 10 | sma | +0.46 | 0.19 | No |
| **20** | **lsma** | **+0.65** | **0.012** | **YES** |
| 40 | kalman | +0.62 | 0.15 | No |

**This is the only signal in the whole project that survives realistic costs.**
Focused cost sweep on the significant cell (lsma w20, **turnover only 0.29/bar**, cost
charged per leg × the `(1+|β|)=1.93` two-leg multiplier):

| cost (per leg) | OOS Sharpe |
|---------------:|-----------:|
| 0 bp | +0.73 |
| 1 bp | **+0.58** |
| 2 bp | **+0.43** |
| 3 bp | +0.28 |
| 5 bp | −0.03 |
| 10 bp | −0.79 |

Liquid sector ETFs trade at **~1–2 bps spread** per leg, so this is **plausibly
net-positive (~0.4–0.6 Sharpe) at realistic cost** — unlike every prior signal, which
died by 5 bps *total*. The low turnover (0.29 vs 0.84 for short-term reversal) is why.

---

## Conclusion (honest)

- **Crypto:** positive gross cross-sectional reversal whose best cell (ou w20) is
  reasonably cost-robust (gross-positive to ~10 bps), but it is **not statistically
  significant** (q=0.10) and realistic ~10–40 bp crypto costs straddle its break-even.
  Insignificance, not cost, is the disqualifier — the asset class does **not** rescue MR.
- **ETF twins:** nothing — spreads too tight to trade.
- **Sector/calendar spreads:** in discovery, the **first and only** signal that was
  BH-significant *and* survived realistic cost (gross 0.73, ~0.43–0.58 at 1–2 bps/leg,
  low turnover). **⚠️ This candidate was later KILLED by a pre-registered confirmation
  test** (`sector_spread_confirmation_summary.md`): with pairs chosen *mechanically* on
  pre-2020 cointegration and the frozen lsma-w20 config run on 2020+, the basket is
  **negative even gross** (−0.15 before cost, −0.40 at 2 bps/leg), CI spans zero, MC
  q=0.63. The discovery number was a product of hand-picked pairs and an
  in-sample-overlapping window — it does **not** hold out of sample.

**But it is a candidate, not an established edge:**
- It is **one cell** found among ~30+ configurations (2 pair classes × 3 windows × 5
  estimators × abs/cross-sectional). Within-run BH corrects the 5 estimators per run,
  **not** the windows/classes searched — a study-wide correction would inflate q=0.012.
- **No single pair is individually significant** (all isolated mc_p > 0.17); the result
  is a *diversification* effect across 6 weak spreads. Legitimate, but it is not one
  robust pair — it needs the basket.
- Few walk-forward folds (~2 OOS windows), 6 hand-picked pairs, and the synthetic-spread
  cost is approximated by a flat `(1+|β|)` leg multiplier (no per-leg spread/impact model).
- The "significant" flag reflects non-random *gross* timing (the MC null is on gross
  per-symbol returns); the **net** Sharpe is what the cost sweep reports, and it is the
  honest number.

**Net:** the platform did its job a third time — surfaced a theory-consistent candidate
(sector-spread relative value) that, unlike daily reversal, intraday bounce, crypto, or
ETF twins, *looked* like it cleared the cost hurdle — then subjected it to the
**dedicated, pre-registered out-of-sample confirmation** it called for, which **KILLED
it** (`sector_spread_confirmation_summary.md`). No declared edge survives.

---

## Project headline — updated

Across passes, the project tested MR on: US equities absolute (no edge), cross-sectional
daily (significant gross, dies at 5 bps), intraday 1m–1h (loses even gross), crypto
(gross, cost-robust to ~10 bps, but insignificant), ETF twins (nothing), and **sector
spreads (significant gross in discovery — but KILLED by pre-registered OOS confirmation,
negative even gross).** The honest, reproducible machinery remains the deliverable; after
confirmation, **no mean-reversion edge survives** — the last open thread is closed.

---

## Reproducing
Run from the repo root as modules (absolute `meridian.*` imports need root on the path):
- `python -m meridian.experiments.study_other_assets` — all three classes (crypto +
  both pair sets) end to end.
- `python -m meridian.experiments.study_sector_focus` — the focused cost sweep +
  per-pair attribution on the significant sector-spread cell.
- Crypto pull: `load_crypto_universe(syms, "1D", start="2021-01-01")` (public REST, no keys).

## Known limitations (this pass)
- Crypto.com public endpoint caps 300 candles/call; history is assembled by backward
  pagination (handled) — daily is the practical granularity.
- Synthetic-spread turnover counts one leg; cost charged at `cost_bps × (1+|β|)` to
  approximate two-leg trading — no per-leg spread/impact model.
- Crypto/ETF universes are hand-selected (liquid majors / economically-motivated pairs);
  the sector-spread hit is one config among many searched — read as suggestive.
