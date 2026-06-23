# Phase 8 Summary — Adaptive Meta-Model and Ensemble

**Status:** Complete
**Date:** 2026-06-22

Phase 8 adds **ensemble estimators** that combine several fair-value estimates
into one. The design payoff of the `BaseEstimator` contract lands here: an
ensemble *is* an estimator, so it flows through deviation metrics, the signal
engine, the validation engine, and reporting with **zero special-casing**.

---

## What was built

### `EnsembleEstimator` — `meridian/estimators/ensemble.py`
A `RollingEstimator` subclass holding member estimators (by registry name,
shared window). Each bar it updates every member, reads their fair values, and
combines them. Residual/scale/z-score/state come from the same template as every
other estimator, so the contract is satisfied for free.

**Combiners:**

| combine | type | idea |
|---------|------|------|
| `mean` | static | average of member fair values |
| `median` | static | median — outlier-resistant blend |
| `trimmed` | static | drop top/bottom 20%, average the rest |
| `weighted` | static | fixed member weights |
| `inverse_variance` | adaptive | weight ∝ 1/recent squared residual (min-variance blend) |
| `skill` | adaptive | weight ∝ recent reversion reward `-residual × next move` |

The `skill` combiner is the adaptive meta-model: it rewards members whose
deviations actually preceded reversion, and **defaults to equal weights when no
member shows positive skill** (an honest fallback, common on noisy data).

### Plumbing
- `make_ensemble(members, combine)` — build an instance (passes straight into
  `run_backtest`/`compute_scores`, which already accept estimator instances).
- `register_ensemble(name, members, combine)` — register a named ensemble that
  takes only `window`, so `create(name)` and the full `validate()` pipeline work
  by name like any single estimator.
- Four named ensembles registered on a diverse core
  (`sma, ema, kalman, lsma, median, ou`): `ens_mean`, `ens_median`,
  `ens_invvar`, `ens_skill`. The library now exposes **42 estimators**.

---

## Tests — `tests/test_ensemble.py`

12 dedicated tests (285 total in suite, all passing); the 4 named ensembles are
*also* swept by the Phase 2 contract tests automatically.
- Registration, unknown-combine / empty-members / window validation.
- **Combiner correctness**: `mean` equals the average of members' fair values;
  `median` equals their median (exact).
- **Adaptive weights** valid (non-negative, sum to 1) for both adaptive blends;
  `inverse_variance` reliably differentiates; `skill` *can* differentiate across
  data (and honestly falls back to equal weights otherwise).
- **Integration**: an ensemble instance runs through `run_backtest`; `update`
  equals `fit` for the adaptive `skill` ensemble (determinism); a registered
  ensemble runs end-to-end through `validate()`.

---

## Headline live result (the honest finding)

Validation on **SPY 2010-2024**, ensembles vs. the best singles
(anchored WFO, zscore, entry 1.5, BH correction):

| estimator | oos_sharpe | q_value | significant |
|-----------|-----------:|--------:|:-----------:|
| hull | 0.41 | 0.30 | False |
| lsma | 0.23 | 0.77 | False |
| **ens_invvar** | **0.21** | 0.77 | False |
| ou | 0.10 | 0.77 | False |
| ens_median | 0.02 | 0.77 | False |
| ens_mean | −0.23 | 0.85 | False |
| **ens_skill** | **−0.32** | 0.85 | False |

**The ensembles do not beat the best single estimator.** `ens_invvar` is
competitive (3rd of 9), but `hull` and `lsma` still lead, and the adaptive
`ens_skill` is the *worst* performer here. Nothing is significant after
correction. This is reported straight: the meta-model is mechanically sound and
fully integrated, but it adds **no robust out-of-sample edge** on this data — the
framework holds the ensemble to the same standard as everything else, and it
does not pass either.

---

## Interface contracts handed to later phases

```python
from meridian.estimators import make_ensemble, register_ensemble, create

ens = make_ensemble(["sma", "ema", "ou"], combine="inverse_variance", window=20)
# ... use exactly like any estimator: run_backtest(prices, ens, "zscore", ...)

register_ensemble("my_ens", ["lsma", "kalman", "ou"], combine="skill")
create("my_ens", window=20)          # now usable by name in validate(), sweep(), etc.
```

---

## Known limitations / notes for Phase 9+

- **No out-of-sample edge demonstrated.** On SPY the ensembles do not beat the
  best single estimator; the adaptive `skill` blend underperforms. A multi-asset
  universe (more independent data) may change this, but no claim is made.
- **`skill` is single-bar and noisy.** Its reward uses the next bar's move, so it
  often lacks the statistical power to differentiate and falls back to equal
  weights. A multi-bar / holding-horizon reward is a possible refinement.
- **Members share one window** and are equal-cost to compute; an ensemble of K
  members is K× the per-bar work of a single estimator.
- **Walk-forward selection (Phase 6) vs. ensembling**: these are two different
  meta-strategies (pick-one vs. blend-all); both are available and both validate
  through the same engine. Neither is privileged.
- **PyTorch neural/autoencoder estimators remain deferred** — the ensemble layer
  is classical (linear blends + adaptive weights), not learned.

---

## Next phase

**Phase 9 — Paper trading deployment:** wire the validated pipeline to the
`execution/` module (alpaca-py paper trading) to run a chosen estimator/ensemble
live on paper, with the same causal, no-look-ahead update loop used in backtests.
