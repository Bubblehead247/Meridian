# Intraday Mean-Reversion Study (research pass 2)

**Status:** Complete — hourly (yfinance) **and minute-scale (Alpaca IEX, live)**
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
FMP (`chart` MCP + `financetoolkit`) is **paywalled**. **Alpaca free tier (IEX)
provides 1-min bars over months/years with free keys** — used for the
minute-scale study below.

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

## Results — Alpaca 1-minute (IEX), 20 large-caps, ~45k RTH bars, flatten-at-close

The minute-scale test that hourly couldn't reach. `periods_per_year=98280`,
regular-trading-hours bars only, otherwise identical setup. (The
`AlpacaDataLoader` was fixed and **live-validated** here: free tier needs
`feed=IEX`, and Alpaca has no adjusted close so `adj_close = close`.)

| strategy | window (min) | best OOS Sharpe | significant |
|----------|:------------:|----------------:|:-----------:|
| Absolute | 5 / 15 / 30 | −73 / −44 / −30 | only w30 (sig. **losing**, q=.013) |
| Cross-sectional | 5 / 15 / 30 | −54 / −37 / −27 | No |

**Cost sweep** (cross-sectional, sma, window 5; turnover 0.80/bar):
−54 @ 1 bp → −104 @ 2 bp → −216 @ 5 bp → −299 @ 10 bp.

Everything **loses, even gross**, and gets worse with cost. The one "significant"
cell (absolute w30) is significant because the strategy **reliably loses** — the
Monte-Carlo rotation test detects non-random *bad* timing, not an edge.

## Interpretation (theory-consistent, now confirmed at minute scale)

- **The "1-minute reversal" is largely a bid-ask-bounce artifact.** It shows up
  in academic *close-to-close* statistics but is **not capturable with realistic
  execution**: the backtester fills one bar after the signal (`held =
  positions.shift(1)`), by which point the bounce has already reverted — so a
  reversion strategy is effectively *momentum at the execution horizon* and
  loses. This is the cleanest refutation of "maybe MR works intraday."
- **Hourly** loses for the complementary reason: at the 1-hour scale prices
  *continue* within the session (intraday momentum), so reversion loses gross.
- **Annualization note:** Sharpes look enormous (±30-300) because a small per-bar
  mean is scaled by √98280; the *sign* is the result, and it is firmly negative.
- **Caveat:** the free Alpaca feed is **IEX** (~2-3% of volume), thinner/noisier
  than the full SIP tape (paid). The full tape would be cleaner, but the
  direction is unambiguous and consistent across every horizon tested.

## Verdict

No intraday MR edge at any tested horizon (1m / 5m-implied / 15m / 30m / 1h),
absolute or cross-sectional — losing even gross at minute scale. Consistent with
the whole project: the apparent reversal lives in microstructure statistics, not
in a realistically-tradable, cost-aware strategy.

---

## Reproducing the minute-scale run

```python
from meridian.data import load_intraday_universe, bars_per_year
from meridian.portfolio import validate_universe
from meridian.features import cross_sectional_demean
# set ALPACA_API_KEY / ALPACA_SECRET_KEY (free tier, IEX feed)
px = load_intraday_universe(syms, "1m", lookback="180d", source="alpaca")
px = {k: v.between_time("13:30", "20:00") for k, v in px.items()}   # regular hours (UTC)
rel = cross_sectional_demean(px)
validate_universe(px, ["sma", "lsma"], "zscore", signal,
                  signal_prices_by_symbol=rel, flatten_overnight=True,
                  periods_per_year=bars_per_year("1m"), ...)
```

---

## Bottom line

The platform now handles intraday data and intraday-only sessions with
frequency-correct annualization, validated live against both yfinance (hourly)
and Alpaca (minute, IEX). At **every** tested horizon — 1-minute through hourly —
intraday MR shows **no edge, losing even gross**: the apparent reversal is a
bid-ask-bounce artifact that realistic one-bar-lagged execution cannot capture.
The project's headline stands and is now established across frequencies: no
robust, cost-aware mean-reversion edge — daily, cross-sectional, or intraday.

## Next pass (planned)
- Other asset classes (spreads, ETF pairs, crypto — lower-cost / more
  reversion-prone instruments may change the verdict).
