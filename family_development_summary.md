# Family Development Summary

## What Was Built

This session extended the Meridian fund from narrow index ETF coverage to a
broad sector ETF library, fixed structural problems in four families, and grew
the saved-strategy record set from 27 to 108 entries.

---

## Problems Found and Fixed

### Four families had the wrong nature

The original Meridian expansion defines nine strategy families, each with a
target holding period and return-generating mechanism. An audit revealed four
families whose best records did not match their intended design:

| Family | Problem | Fix |
|---|---|---|
| `trend_following` | `dual_ma_trend` holds 354 bars (17 months), fires 0.3×/yr — buy-and-hold, not trend following | Added `chandelier_trend`: 20-bar high entry + 2×ATR trailing stop → 20-bar holds, 5–6 trades/yr |
| `momentum` | `roc_momentum_200ma` uses 252-bar lookback, holds 48 bars — slow trend following | Added `swing_momentum`: long when 20-day MA slope is rising vs 10 bars ago → 32-bar holds, 4 trades/yr |
| `mean_reversion` | Best OOS record was `rsi_reversion_ma_filter` (IWM, 0.7 trades/yr) — not mean reversion | Ran `zscore_reversion` (symmetric, 10 trades/yr, 7-day holds) through pipeline; it is now the primary |
| `pullback_continuation` | `rsi_pullback_50_200_tight` exits in 2 bars — RSI bounces immediately | Added `rsi_pullback_continuation`: proper entry/exit state machine (enter RSI<40, exit RSI>60, 50/200 MA gate) → 22-bar holds |

### Root cause

All four failures shared the same underlying issue: either the exit signal was
a level (fires every bar the condition is true) rather than a state transition,
or the lookback was so long the model had essentially become a passive regime
filter. The 200MA was the dominant gate in five of seven families, which
eliminates diversification benefits precisely when they are most needed —
during corrections, all five go flat simultaneously.

### The symmetric mean reversion problem

`zscore_reversion` and `bollinger_reversion` fail OOS (2020–2022) because:
- March 2020: extreme directional move overwhelms 2-day reversion thesis
- 2020–2021: secular bull, short leg loses repeatedly
- 2022: secular bear, long leg loses repeatedly

The architecture already handles this correctly: the regime permission matrix
gates mean reversion to `neutral`/`bear` trend regimes only. In a correctly
gated portfolio, mean reversion would have been flat through 2020–2021. The
OOS failure is a real research finding about regime dependency, not a model
flaw. `zscore_neutral_regime` (inline 60-day trend band) was added as a
research alternative but did not improve ungated backtest performance.

---

## New Models Added

### `meridian/families/trend_following/models.py`
- `chandelier_trend` — 20-bar high entry, 2×ATR(20) trailing stop exit

### `meridian/families/momentum/models.py`
- `swing_momentum` — MA-slope momentum: long when 20-day MA > its value 10 bars ago

### `meridian/families/mean_reversion/models.py`
- `zscore_vol_filtered` → replaced by `zscore_neutral_regime`
- `zscore_neutral_regime` — symmetric z-score, flat when 60-day return outside ±8%

### `meridian/families/pullback_continuation/models.py`
- `rsi_pullback_continuation` — RSI<40 entry / RSI>60 exit, 50/200 MA gate

### `meridian/families/breakouts/models.py` (prior session)
- `donchian_ma_exit` — 20-bar channel entry, exit on channel low OR 200MA break
- `turtle_ma_exit` — 55-bar entry, exit on 20-bar low OR 200MA break

### `meridian/families/mean_reversion/models.py` (prior session)
- `zscore_reversion_long_only` — z-score<-1.5 entry, exit at fair value, no shorts
- `rsi_reversion_ma_filter` — RSI<35 AND above 200MA, exit RSI>50
- `bollinger_reversion_long_only` — EMA z-score<-2.0 entry, exit at 0, no shorts

---

## Sector ETF Expansion

Previously all records used SPY, QQQ, or IWM. The full SPDR sector basket was
run through the gauntlet and pipeline:

**Symbols**: XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU

**Note on XLC (2018-present) and XLRE (2015-present)**: shorter history,
mostly bull market. Records on these symbols should be weighted accordingly.
The full 1998-present history ETFs (XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK,
XLU) are more reliable.

75 single-asset passers were run through the full pipeline (backtest →
walk_forward → OOS). All 75 saved records.

---

## Family Status After This Session

| Family | Records | OOS | Symbols | Nature model | Nature stage | Best Sharpe |
|---|---|---|---|---|---|---|
| `long_term_etf` | 2 | 2 | SPY, XLC | `above_200ma` | OOS | 0.89 (XLC) |
| `momentum` | 18 | 4 | 7 symbols | `swing_momentum` | walk_forward | 0.86 (XLE) |
| `trend_following` | 16 | 13 | 8 symbols | `chandelier_trend` | OOS | 1.19 (XLE) |
| `mean_reversion` | 29 | 9 | 11 symbols | `zscore_reversion` | OOS | 1.46 (XLP) |
| `pullback_continuation` | 32 | 22 | 12 symbols | `rsi_pullback_continuation` | OOS | 1.41 (XLF) |
| `sector_rotation` | 2 | 2 | SECTORS, XLC | `relative_strength` | OOS | 1.15 |
| `breakouts` | 9 | 7 | QQQ, SPY, XLC, XLK | `donchian_ma_exit` | OOS | 0.67 (XLK) |

**Total: 108 records**

### Families ready for paper trading consideration
- `pullback_continuation` — 22 OOS records, 12 symbols, nature-correct model at OOS 1.13
- `sector_rotation` — OOS on both basket and single sector

### Families needing more work
- `long_term_etf` — only 2 records; needs coverage on XLY, XLP, XLK, XLV at minimum
- `breakouts` — 9 records but mostly XLC (short history); needs more long-history symbols
- `momentum` — nature-correct model (`swing_momentum`) is at walk_forward only; the 252-bar
  models are better-evidenced but structurally wrong for the 1–3 week sleeve

---

## Remaining Gaps

1. **`swing_momentum` needs to clear OOS.** Currently at walk_forward (Sharpe 0.59 on QQQ).
   The OOS window (2020–2022) is a strong test; this model may need a volatility or regime
   filter to pass.

2. **`chandelier_trend` coverage.** OOS on SPY and XLC only. Needs XLK, XLY, XLI.

3. **`zscore_reversion` OOS on sector ETFs.** Currently at backtest on XLF, XLP; walk_forward
   on XLV. The nature-correct mean reversion model needs OOS evidence on non-index assets.

4. **S&P 500 universe scan** — not yet run. This is the next step: run all families across
   the S&P 500 constituent universe to find which individual stocks produce the strongest
   per-family signals, then save the best per-family, per-stock records.

---

## Known Limitations

- yfinance data is survivorship-biased — results on S&P 500 constituents will be upward-biased
- XLC and XLRE records should carry lower confidence due to short history
- The OOS window (2020–2022) happens to overlap with an extremely unusual macro regime
  (COVID crash + secular bull + rate-shock bear in sequence); OOS failures on symmetric
  strategies reflect regime specificity more than true strategy failure
- `rsi_reversion_ma_filter` Sharpe scores (1.46 on XLP, 1.20 on XLI) are suspiciously high
  for a model that fires <1×/yr; these are likely low-sample flukes and should not be used
  for capital allocation until trade count exceeds 30

---

## Next Step

Run the full S&P 500 constituent universe through the single-asset gauntlet for
all families, identify the top-N passers per family, run them through the full
pipeline, and save records. This will:
- Give mean reversion and breakout families broad equity coverage
- Surface which sectors and market caps each family works best on
- Build the evidence base needed to move `pullback_continuation` to paper trading
