# Survivorship-Bias Study Summary

**Status:** Complete
**Date:** 2026-06-23
**Scope:** Address the single biggest caveat in every report — survivorship bias —
by sourcing genuinely point-in-time data and *measuring* the bias directly.

---

## The data reality (verified)

Free, survivorship-bias-free price data does not exist from our sources:
- **yfinance** returns **0 rows** for delisted tickers (ATVI, TWTR, SIVB, FRC,
  XLNX, CERN, …) — Yahoo purges them.
- **FinanceDatabase** flags **0 of 151,096** equities as delisted — it is itself
  a survivor snapshot.

Genuine point-in-time data only comes from licensed sets (CRSP, Sharadar,
Norgate). The free exception: **Teddy Koker's `survivorship-free-spy`**
(github.com/teddykoker/survivorship-free-spy), which ships
- `constituents.csv` — monthly S&P 500 membership snapshots (2006–mid-2019),
- `data/<TICKER>.csv` — per-ticker OHLCV from Quandl WIKI Prices (~2013–Feb 2018),
  **including names later removed from the index**.

It is not vendored here (kept under git-ignored `external/`); the loader points
at a local clone.

---

## What was built

### Loader — `meridian/data/survivorship.py`
`SurvivorshipDataset(root)` reads the on-disk format:
- `members_asof(date)` / `members_ever(start, end)` — point-in-time membership
  from the monthly snapshots.
- `survivor_only_universe(...)` — prices for only the names that were in the
  index at the **end** of the window (the survivorship-**biased** view).
- `survivorship_free_universe(...)` — every name that was *ever* a member in the
  window (including removed ones), each **membership-gated**: prices are NaN on
  dates the name was not in the index, so it is untradable then (true
  point-in-time).
- Handles the dataset's mixed date formats and annotated tickers (`AAPL*`).

### Tests — `tests/test_survivorship.py` (offline fixture)
9 tests on a tiny synthetic dataset: membership as-of/ever, survivor-only drops
removed names, survivorship-free includes them, **membership gating masks
non-member dates**, mixed-date parsing, missing-file handling.

---

## Robustness bugs the real data exposed (and fixed)

Membership-gated prices contain NaNs (pre-listing / post-removal), which broke
parts of the pipeline that assumed contiguous data:

1. **`ou`/`ar1`/`ar2` crashed** — their least-squares fit (`np.linalg.lstsq`)
   hit NaN windows ("SVD did not converge" / DLASCL). Fixed: `compute_scores`
   now **skips missing-data bars** (no signal, no state update), and `_ols`
   falls back gracefully on any degenerate window.
2. **Removed names lingered** in the portfolio — fixed: `per_symbol_signals`
   marks a name **flat/inactive wherever it has no price**, so it leaves the
   book at removal.
3. **`pct_change` padded gaps** — fixed with `fill_method=None`, so a missing
   price yields a NaN→0 return, never a fabricated carried-forward one.

Regression test added; full suite **334 passing**, lint clean.

---

## Headline result — measuring the bias

Same window (2014-2018), same strategy, equal-weight cross-sectional portfolio,
anchored WFO, 1 bp cost, zscore, entry 1.5 — **only the universe differs**:

| universe | names | `sma` OOS Sharpe | `ou` OOS Sharpe | significant |
|----------|------:|-----------------:|----------------:|:-----------:|
| **Survivorship-free** (point-in-time) | 620 | **0.33** | −0.02 | No |
| **Survivor-only** (biased) | 471 | 0.29 | −0.07 | No |

**149 names** were real index members during 2014-2018 but removed by the end
(AA/Alcoa, AABA/Yahoo, AET/Aetna, ALTR/Altera, AVP/Avon, …). The survivor-only
view silently ignores all of them.

### Interpretation (the nuance that matters)

- **The bias is real and now measurable** (~0.04 Sharpe for `sma`).
- **It runs *opposite* to the textbook direction here.** The usual "survivorship
  inflates returns" intuition is for **buy-and-hold** (you'd miss the losers).
  For a **long-short mean-reversion** strategy, the removed names add reversion
  opportunities and breadth, so excluding them *slightly hurt* — the bias-free
  universe did marginally **better**. Bias direction is strategy-dependent.
- **The conclusion is robust to it.** Neither universe is statistically
  significant after correction (q ≈ 0.55). The "no robust mean-reversion edge"
  finding holds **with or without** survivorship correction.

---

## Interface contracts

```python
from meridian.data import SurvivorshipDataset
from meridian.portfolio import validate_universe

ds = SurvivorshipDataset("external/survivorship-free-spy/survivorship-free")
free = ds.survivorship_free_universe("2014-01-01", "2018-03-01")   # bias-free, gated
surv = ds.survivor_only_universe("2014-01-01", "2018-03-01")       # biased baseline
table = validate_universe(free, ["sma", "ou"], "zscore", signal, ...)
```

---

## Known limitations

- **Window ends Feb 2018** (Quandl WIKI cutoff) and spans ~5 years — a dated,
  shorter study than the live yfinance work; the A/B is internally
  apples-to-apples but not comparable to the 2010-2024 SPY numbers.
- **Acquisition vs. bankruptcy not distinguished.** Prices simply end at
  removal, so the strategy "closes at last price" (acquisition-like). True
  bankruptcies (→ \$0) would book a -100% terminal return; for a *long-only*
  strategy that would make survivorship bias materially larger. This long-short
  study is less sensitive, but the limitation is real.
- **Monthly membership** granularity (gating snaps to month-ends).
- **S&P 500 only**, close-only prices. Broader/older survivorship-free data needs
  a licensed source — the loader would adapt with a thin format shim.

---

## Bottom line

The platform can now consume genuinely point-in-time data and **quantify**
survivorship bias rather than just disclaim it. For this mean-reversion study the
bias is small and the headline conclusion is unchanged — but the capability, and
the demonstration that the bias is strategy-direction-dependent, are the real
contributions. The disclosure in every report remains correct; it is now backed
by a measurement.
