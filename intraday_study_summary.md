# Intraday Mean-Reversion Study (research pass 2)

**Status:** Complete (hourly); minute-scale path built, gated on Alpaca keys
**Date:** 2026-06-24
**Scope:** Test MR at intraday frequency (where reversal is documented strongest)
on whether any edge clears the cost hurdle that killed the daily cross-sectional
signal in pass 1.

---

## What was built

### Intraday loaders — `meridian/data/intraday.py`
- `load_intraday` / `load_intraday_universe`: yfinance **period-based** intraday
  fetch (the daily `load_ohlcv` uses `period="max"`, which yfinance rejects for
  intraday), normalized via `normalize_ohlcv` (DatetimeIndex keeps time-of-day),
  cached per interval.
- `AlpacaDataLoader`: lazy `alpaca-py` historical client (1-min bars over years,
  free IEX tier), creds from env — built and unit-tested (cred-guard), runs when
  the user adds keys.
- `bars_per_year(interval)`: frequency-correct annualization factor
  (1h ≈ 1764, 5m ≈ 19656, 1m ≈ 98280) fed to `periods_per_year`.

### Intraday-only session handling — `backtest_portfolio(..., flatten_overnight=True)`
Derives the session date from the DatetimeIndex and zeroes weights on each
session's last bar; because positions are lagged, the next session opens flat —
the overnight gap return is never booked, nothing is carried overnight, and the
daily flatten is charged as real turnover. Threaded through `run_universe_backtest`
and `validate_universe`. Default False ⇒ daily behavior unchanged.

**11 tests** (`tests/test_intraday.py` + `tests/test_portfolio.py`): annualization,
offline loader, Alpaca cred-guard, flatten-at-close mechanics + regression guard,
end-to-end intraday `validate_universe`. **354 total pass, lint clean.**

---

## Data constraint (decisive)

Free intraday options are thin: yfinance gives **1h up to ~730 days** (the only
walk-forward-viable free granularity); 5m/15m only ~60 days, 1m only ~7 days.
FMP (`chart` MCP + `financetoolkit`) is **paywalled**. Alpaca free tier has 1-min
bars but needs keys. **So the live study could only run at the hourly horizon.**

---

## Results — yfinance 1h, 20 large-caps, ~5000 bars, flatten-at-close

`periods_per_year=1764`, equal-weight L/S, anchored WFO, BH correction, entry 1.5.

| strategy | window (bars) | best OOS Sharpe | significant |
|----------|:-------------:|----------------:|:-----------:|
| Absolute | 7 / 14 / 28 | −1.49 / −0.51 / −0.19 | No |
| Cross-sectional | 7 / 14 / 28 | −2.30 / −1.37 / −0.87 | No |

**Cost sweep** (cross-sectional, sma, window 7; turnover 0.70/bar):
−2.53 @ 1 bp → −10.3 @ 5 bp → −19.8 @ 10 bp → −36.3 @ 20 bp.

Everything is **negative even at 1 bp** — the *gross* signal itself loses, the
opposite of the daily cross-sectional case (+0.49 gross at 1 bp in pass 1).

---

## Interpretation (theory-consistent)

- **Hourly is the wrong horizon for reversal.** Documented intraday mean
  reversion is a **1–5 minute microstructure effect** (bid-ask bounce, liquidity
  provision). At the **1-hour** scale, prices *continue* within the session
  (intraday momentum), so a reversion strategy loses **gross** — exactly what the
  negative Sharpes show. The effect we wanted to test lives *below* the
  granularity free data can reach.
- **Annualization note:** Sharpes look large (±2-3) because a small per-bar mean
  is scaled by √1764; the *sign* is the result, and it is negative.
- **The honest verdict at the testable horizon:** no intraday MR edge, absolute
  or cross-sectional, at hourly frequency on liquid large-caps.

---

## What would actually test the hypothesis

The 1–5 minute test needs **minute bars** — `AlpacaDataLoader` is built and
unit-tested; it runs as soon as free `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` are
set (alpaca.markets, no funding). Even then, minute reversal in liquid names is
heavily arbitraged and the bid-ask-bounce "edge" is mostly uncapturable after
realistic costs — so the prior is still skeptical, consistent with the whole
project's findings.

---

## Running it on minute data later

```python
from meridian.data import load_intraday_universe, bars_per_year
from meridian.portfolio import validate_universe
from meridian.features import cross_sectional_demean

px = load_intraday_universe(syms, "1m", source="alpaca")      # needs Alpaca keys
rel = cross_sectional_demean(px)
validate_universe(px, ["sma", "lsma"], "zscore", signal,
                  signal_prices_by_symbol=rel, flatten_overnight=True,
                  periods_per_year=bars_per_year("1m"), ...)
```

---

## Bottom line

The platform now handles intraday data and intraday-only sessions, and the
machinery is frequency-correct. At the only free testable horizon (hourly),
intraday MR shows **no edge — negative even gross** — consistent with reversal
being a sub-hourly microstructure phenomenon. The minute-scale test is built and
awaits free Alpaca keys. The project's headline stands: no robust, cost-aware MR
edge found — daily, cross-sectional, or intraday.

## Next pass (planned)
- Other asset classes (spreads, ETF pairs, crypto — lower-cost / more
  reversion-prone instruments may change the verdict).
