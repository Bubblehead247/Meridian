# Phase 2 Summary — Estimator Library

**Status:** Complete
**Date:** 2026-06-22

Phase 2 delivers **38 fair-value estimators** (target was 37+), all implementing
the `BaseEstimator` contract through one shared template base, all registered by
name, and all covered by a contract-enforcing test sweep plus targeted and
property-based tests.

---

## Design: one template, many estimators

The core simplification is `meridian/estimators/_rolling.py` —
`RollingEstimator(BaseEstimator)`. It implements the **entire contract once**:
the fit/update loop, `residual`, `predict_scale`, `zscore`, and `state`. Each
concrete estimator supplies only `_compute_mean()` (and optional recursive
state). This is why 38 estimators are a few hundred lines, not thousands.

Key mechanics:
- A bounded price buffer is kept; each bar computes the fair value and records
  the residual `price - fair_value`.
- `predict_scale()` = std of recent residuals; `zscore` = residual / scale.
  Scale is `NaN` until 2+ residuals exist (warmup), so early z-scores are `NaN`
  rather than misleading.
- `update(price)` runs the **identical** per-bar step as `fit`, so an
  incremental update equals having included the price in `fit` — verified for
  all 38 by `test_update_equivalent_to_fit`.
- Windowed estimators read `self._last()`; recursive ones read the previous
  fair value `self._mean` and latest price `self._buf[-1]`.

## Registry

`meridian/estimators/registry.py`: `@register("name")` collects classes;
`create(name, **kw)`, `get_estimator`, `list_estimators`, `all_estimators`.
Importing `meridian.estimators` imports every family module, populating the
registry. Estimators are now selectable from config by name (e.g. `ema`).

## The 38 estimators (by family)

- **Moving averages (17)** — `sma, ema, wma, trima, dema, tema, zlema, kama,
  t3, alma, frama, vidya, mcginley, sine_wma, geometric, harmonic, hull`
- **Robust location (5)** — `median, trimmed_mean, winsorized_mean, huber,
  midrange`
- **Regression endpoints (6)** — `lsma, quadratic_reg, cubic_reg, theil_sen,
  savgol, lowess`
- **Signal-processing filters (6)** — `kalman, hp_filter, super_smoother,
  gaussian, butterworth, fourier`
- **Stochastic / time-series models (4)** — `ar1, ar2, ou, exp_reg`

`ou` is distinctive: it reports the Ornstein-Uhlenbeck **long-run reversion
level** `c/(1-phi)`, not a one-step smooth — the most explicitly
mean-reversion-aware estimator.

## Tests — `tests/test_estimators.py`

124 estimator tests (155 total in the suite, all passing):
- **Contract sweep** (parametrized over all 38): `predict_mean` finite and
  within a sane band; `residual == price - mean`; `zscore == residual/scale`
  (or NaN); `state` carries estimator/window/n/mean.
- **update == fit** for all 38.
- **window < 2 rejected** for all 38.
- **Exact math**: SMA mean, EMA recurrence (alpha=0.4), LSMA exact on a line,
  median ignores a 1000x spike, midrange = (high+low)/2.
- **hypothesis properties**: residual identity for random estimator/price;
  z-score sign matches residual sign; z-score mean ≈ 0 over white noise.

**Live check:** all of `sma/ema/kalman/lsma/ou/t3/hull` produced fair values
within ~1.5% of the last SPY adj-close (513.52) with sensible z-scores.

---

## Bugs caught and fixed during the phase

- **Gaussian filter instability** — the 2-pole Ehlers form needs `alpha**2` on
  the input to keep DC gain at 1; without it the output drifted (245 vs prices
  ~95). Fixed and now within the contract band.
- **Huber divide-by-zero** — reweighting now masks far points instead of
  dividing the whole array, removing the RuntimeWarning.

---

## Interface contracts handed to later phases

```python
from meridian.estimators import create, list_estimators, all_estimators

est = create("ema", window=20)
est.fit(prices)            # prices: a pd.Series (use adj_close from Phase 1)
est.update(new_price)
est.predict_mean()         # current fair value
est.residual(price)        # price - fair value
est.zscore(price)          # standardized deviation (NaN during warmup)
est.state()                # dict: estimator, window, n, mean, scale, + extras
```

Phase 3 (deviation metrics) consumes `residual`/`zscore`/`predict_scale`;
Phase 4 (signals) holds signal logic constant and varies only the estimator —
the registry makes that sweep trivial.

---

## Known limitations / notes for Phase 3

- **Scale = residual std.** Every estimator currently standardizes by the std of
  its own recent residuals. Phase 3 will add alternative deviation
  normalizations (ATR-normalized, MAD-based, etc.) as a separate module, kept
  independent of the estimator per the design.
- **Warmup yields NaN z-scores** until 2+ residuals exist; signal logic must
  treat NaN as "no signal."
- **Window default is 20** and is the single knob; per the no-estimator-specific
  -optimization rule, the same window/signal settings apply across estimators in
  any comparison. Per-estimator tuning is explicitly out of scope.
- **Endpoint smoothers** (savgol, lowess, hp_filter, fourier) use only the
  trailing window, so they remain causal and online-consistent — but they are
  smoothers, not forecasts.
- **No volume/OHLC estimators.** The contract fits on a price `Series`;
  volume-weighted or range-based estimators are deferred (would need a richer
  input than the contract allows).
- **PyTorch estimators still deferred** (neural/autoencoder) per project plan.

---

## Next phase

**Phase 3 — Deviation metrics:** a `meridian/deviations/` module of alternative
ways to measure and normalize how far price sits from fair value (z-score,
ATR-normalized, MAD/robust-z, percentile rank), kept independent of the
estimators so any metric pairs with any estimator.
