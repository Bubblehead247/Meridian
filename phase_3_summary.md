# Phase 3 Summary — Deviation Metrics

**Status:** Complete
**Date:** 2026-06-22

Phase 3 delivers `meridian/deviations/`: **6 deviation metrics** that normalize
an estimator's residual (`price - fair value`) into a comparable deviation
score. They are fully **decoupled from estimators** — a metric consumes only a
residual stream (plus OHLC bars for the ATR metric), so any metric pairs with
any of the 38 estimators.

---

## Design: same pattern as Phase 2

`meridian/deviations/base.py`:
- `BaseDeviation` — the ABC contract: `fit(residuals, bars=None)`,
  `update(residual, bar=None)`, `value(residual) -> float`, `state()`.
- `RollingDeviation` — template implementing the contract once (rolling
  residual window, warmup-to-NaN, fit/update loop, state). Each metric supplies
  only `_value(r)`; ATR-style metrics also override `_post_update(bar)`.

`meridian/deviations/registry.py` mirrors the estimator registry:
`@register`, `create`, `get_deviation`, `list_deviations`, `all_deviations`.
Metrics are selectable from config by name.

## The 6 metrics — `meridian/deviations/metrics.py`

| Name | Formula / idea | Robust? | Needs OHLC? |
|------|----------------|---------|-------------|
| `zscore` | `(r - mean) / std` of residual window | no | no |
| `mad_z` | `r / (1.4826 * MAD)` — robust scale | yes | no |
| `modified_z` | `(r - median) / (1.4826 * MAD)` — Iglewicz-Hoaglin, median-centered | yes | no |
| `percentile` | empirical rank of `r` in window, mapped to [-1, 1] | yes | no |
| `minmax` | `r` placed in recent [min, max] range → [-1, 1] | no | no |
| `atr_norm` | `r / ATR` — distance in units of typical price range | n/a | **yes** |

- The `1.4826` constant makes MAD a consistent estimator of std for normal data,
  so robust and classic z-scores share a scale.
- `atr_norm` tracks true range from bars (`high/low/close`); without bars its ATR
  is undefined and `value` returns NaN.
- Warmup: every metric returns NaN until 2+ residuals exist — signal logic
  (Phase 4) must treat NaN as "no signal."

## Tests — `tests/test_deviations.py`

34 tests (189 total in suite, all passing):
- **Contract sweep** over all 6: finite value after fit, state carries
  deviation/window/n, warmup → NaN, `update == fit` equivalence, window
  validation.
- **Exact math**: zscore on a known set, percentile bounds/center, minmax
  endpoints (-1/0/+1), `mad_z` beats `zscore`'s outlier sensitivity, `atr_norm`
  = residual/ATR, atr without bars → NaN.
- **Integration**: metrics pair correctly with residuals from `sma/ema/kalman/ou`
  (the decoupling guarantee).
- **hypothesis**: `percentile`/`minmax` always within [-1, 1]; `zscore` sign
  matches the centered residual.

**Live check (full pipeline):** SPY adj-close 513.52, EMA(20) fair value 509.80,
residual +3.72 → all 6 metrics positive and sensibly scaled (`zscore` +0.53,
`atr_norm` +0.87, `percentile` +0.20, ...).

---

## Interface contracts handed to later phases

```python
from meridian.deviations import create, list_deviations

dev = create("zscore", window=20)
dev.fit(residuals, bars=ohlcv_frame_or_None)   # residuals: pd.Series of price - fair value
dev.update(residual, bar=row_or_None)          # bar: dict-like high/low/close (atr_norm only)
dev.value(residual)                            # normalized deviation (NaN during warmup)
dev.state()                                    # dict: deviation, window, n, + extras
```

Phase 4 (signal engine) will, per bar: `est.update(price)` →
`r = est.residual(price)` → `dev.update(r, bar)` → `score = dev.value(r)` →
threshold the score into entry/exit. Signal logic is held constant; estimator
and deviation metric both vary by name from config.

---

## Known limitations / notes for Phase 4

- **Residual mean is metric-dependent.** `zscore`/`modified_z` re-center
  (mean/median) so biased-residual estimators (one-step forecasts, smoothers)
  are handled; `mad_z`/`atr_norm` do not re-center (they assume residuals are
  ~0-centered). Pair accordingly.
- **ATR needs bars threaded through.** The signal engine must pass the current
  OHLC bar to `update`/`fit` when `atr_norm` is selected; residual-only metrics
  ignore it. A uniform per-bar call signature (`update(residual, bar)`) already
  supports this.
- **Single window knob** per metric, matching the no-per-estimator-optimization
  rule. The same window applies across a comparison sweep.
- **Bounded vs unbounded scores.** `percentile`/`minmax` are bounded to [-1, 1];
  z-scores are unbounded. Signal thresholds in Phase 4 must be chosen per
  metric-family (a "2.0" threshold means nothing for a bounded metric).

---

## Next phase

**Phase 4 — Signal engine + backtester:** constant entry/exit logic driven by a
deviation score (enter on extreme deviation, exit on reversion to fair value),
plus a vectorized/event backtester. Sweeps every (estimator × deviation) pair
through identical signal logic — the heart of the comparative framework.
