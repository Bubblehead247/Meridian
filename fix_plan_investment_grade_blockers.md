# Fix Plan — The Four Ranked Investment-Grade Blockers

Source: `post_implementation_verification_audit.md` §10. Scope is intentionally narrow — each item closes a specific, verified gap. No new estimators, features, or ML. Ordered by severity, matching the audit.

---

## 1. Universe survivorship-bias guard is UI-only

**Problem.** The guard lives in `cli.py::_symbol_list()` (line 145) — it only runs for the four CLI commands that call it. `meridian/data/universe.py::get_universe()` (line 87) has zero gating, and `experiments/runner.py::resolve_symbols()` (line 54, the path every config-driven `meridian validate/backtest/universe/paper <config.yaml>` command uses) never touches `_symbol_list()` at all. Calling either directly from Python, or running any config-driven command, produces a silently biased backtest.

**Fix.**
- Move the actual check into `get_universe()` itself: add an `accept_survivorship_bias: bool = False` parameter. When `name` resolves to an index universe (`key in INDEX_UNIVERSES`) and the flag is `False`, raise a new `SurvivorshipBiasError(ValueError)` with the same message `cli.py` currently prints.
- Change `cli.py::_symbol_list()` to a thin wrapper: catch `SurvivorshipBiasError` and translate it into the existing CLI error message/exit path, instead of doing the check itself. This removes duplication rather than adding it.
- Add `accept_survivorship_bias` to `resolve_symbols()` in `runner.py`, sourced from `cfg["data"].get("accept_survivorship_bias", False)`, threaded into its `get_universe()` call. Document the new config key in the relevant `configs/*.yaml` comments.
- `interactive.py::_resolve()` already prints a warning (per the P0 work) — change it to actually pass the flag through instead of only warning after the fact.

**Files:** `meridian/data/universe.py`, `meridian/cli.py`, `meridian/experiments/runner.py`, `meridian/interactive.py`.

**Tests:** direct `get_universe("SP500")` raises without the flag; raising doesn't happen for ETF universes (SPY/QQQ/IWM aren't index-membership questions); `resolve_symbols()` with a bare `data.universe: SP500` config (no flag) raises the same error; existing CLI tests continue to pass with the wrapper.

**Effort:** small — one new exception class, one new parameter threaded through 4 call sites.

---

## 2. Graduation-criteria version lock has a silent, zero-log bypass

**Problem.** `graduation.py::advance()` (line 344) trusts whatever `GraduationCriteria` object the caller passes. It only compares `criteria.version` (a string) against `ledger.criteria_version` — it never re-derives the criteria's actual threshold values from `configs/graduation_criteria.yaml`. A caller can construct `GraduationCriteria(version="1.0.0", min_sharpe=-999, ...)` and `advance()` accepts it as if it came from the real config, with no mismatch raised and no log entry written.

**Fix.**
- Add a `criteria_fingerprint(criteria: GraduationCriteria) -> str` helper (a stable hash — e.g. `hashlib.sha256` — over the criteria's actual field values, not just its version label).
- At `advance()`'s existing version-bind point, also bind `ledger.criteria_fingerprint = criteria_fingerprint(criteria)` alongside `ledger.criteria_version`.
- On every subsequent call, re-load the canonical criteria via `load_criteria_config()` for the bound version, compute its fingerprint, and compare it against the *passed-in* `criteria`'s fingerprint. If they differ — regardless of what version label the caller claims — raise `CriteriaVersionMismatch` (or a new `CriteriaTamperedError` for this specific case, since it's a different failure mode: the version label lied about the underlying values).
- This closes the gap because the check no longer trusts the caller's own version string as ground truth; it trusts only what's actually on disk under that version.

**Files:** `meridian/pipeline/graduation.py`, `meridian/portfolio/ledger.py` (new `criteria_fingerprint` field alongside the existing `criteria_version`).

**Tests:** the exact adversarial case from the audit — a spoofed object with a correct version label and altered thresholds — must now raise; a genuine call using `load_criteria_config("1.0.0")` unmodified must continue to pass; an old ledger with no stored fingerprint (pre-migration) should bind on first call rather than raise, matching the existing version-bind behavior.

**Effort:** small-medium — one hashing helper, one new ledger field, one comparison added to an existing code path. No pipeline call sites change.

---

## 3. Short-reliant strategies graduate on numbers unachievable live

**Problem.** `execution/live_runner.py::run_paper_session()` flattens every short signal to zero (confirmed at the code level in the prior audit). `graduation.py` has no awareness of this at all — a strategy whose backtested Sharpe depends on its short leg can pass every stage and receive live capital allocation, then simply never take half its trades once live.

**Fix — recommend option (a) below; it requires no execution-engine work and directly measures the real gap.**
- (a) **Add a long-only shadow re-evaluation.** In `pipeline/graduation.py`, add `requires_long_only_reeval(ledger) -> bool`: `True` when the strategy's signal config has `allow_short=True` (or equivalent — check `signals/` for the actual flag name and generalize from there). When true, `evaluate()`/`advance()` must additionally receive a scorecard computed with shorts zeroed out (same backtest, `allow_short=False`), and the promotion decision must use the *worse* of the two scorecards on every metric, not just the full-short one. Surface which scorecard drove the decision in the ledger/run log for auditability.
- Do **not** pursue (b) (making `live_runner.py` support live shorting) as part of this fix — that's a real execution-engine change with its own review needs (margin/borrow modeling, corporate-action risk on shorts) and isn't required to close this specific defect; it's a legitimate separate project if the fund wants proper short exposure.
- Do **not** pursue (c) (hard-block short-capable strategies at `paper`) as the default — it would silently discard real strategies rather than measuring the actual long-only-achievable edge, which is more informative and less wasteful.

**Files:** `meridian/pipeline/graduation.py`, `meridian/scoring/` (wherever the scorecard is built — needs a `allow_short=False` re-run path), `meridian/portfolio/ledger.py` (record which scorecard variant graduated the strategy).

**Tests:** a synthetic strategy whose edge lives entirely in its short leg must fail long-only re-evaluation and therefore not graduate; a long-only strategy is unaffected (no shadow re-eval triggered, no behavior change); a strategy profitable on both legs graduates using the worse (long-only) numbers, and the ledger records that.

**Effort:** medium — touches the scoring/graduation boundary, needs a real signal-flag audit to correctly detect "this strategy uses shorts" across all 9 families, not just mean reversion.

---

## 4. P2 tooling (cost-stress, capacity, CPCV/PBO) isn't enforced at graduation

**Problem.** `analytics/capacity.py` and `validation/cpcv.py` are built and unit-tested but have zero callers in `pipeline/`, `experiments/`, or `cli.py` (confirmed by grep in the audit). They don't affect any promotion decision.

**Fix.**
- Add two new fields to `GraduationCriteria` (sourced from `configs/graduation_criteria.yaml`, versioned like the rest): `max_capacity_participation: float` (e.g. reject if `participation_rate()` at the strategy's actual position size exceeds a threshold — this is a real liquidity floor, not a nice-to-have) and `require_cost_stress_stable: bool` (gate on `cost_stress_conclusion_stable()` from `capacity.py` — the strategy's pass/fail conclusion must not flip across the existing cost grid).
- Add `pbo_max: float | None` (optional; only enforced when the scorecard includes a PBO measurement, since PBO requires multiple competing trials to be meaningful — most single-strategy evaluations won't have one, so this should default to "not gated" rather than blocking on an absent value).
- Wire these into `graduation.py::passes_metric_bar()` (or wherever the existing Sharpe/expectancy/drawdown checks live) as additional required conditions, following the exact same pattern as the existing threshold checks — no new code shape, just more fields checked the same way.
- Capacity checks need real ADV data at evaluation time — confirm `capacity.py::average_daily_volume()` has a data source wired in at the graduation call site (it currently only has one in its own test fixtures).

**Files:** `meridian/pipeline/graduation.py`, `configs/graduation_criteria.yaml`, `meridian/scoring/` (scorecard needs to actually include capacity/cost-stress/PBO numbers, not just Sharpe et al.).

**Tests:** a strategy whose position size exceeds the capacity threshold must fail promotion even with a strong Sharpe; a strategy whose cost-stress conclusion flips (profitable at 0bps, unprofitable at 25bps) must fail; a strategy with no PBO data must not be blocked by the (unset) PBO gate; existing strategies with no capacity data available should fail loudly (missing data), not silently pass.

**Effort:** medium-large — this is the one requiring the scorecard itself to grow new fields (capacity/cost-stress/PBO numbers aren't computed as part of the standard scoring flow today), not just a graduation-side check.

---

## Suggested order

1 and 2 first — both are narrow, self-contained, and close the two most severe "looks safe but isn't" gaps with the least code surface.
3 next — moderate scope, directly addresses live capital being deployed on unachievable numbers.
4 last — the largest scope, since it requires extending the scorecard itself before graduation can gate on it.

No item here requires touching the estimator library, signal logic, or adding new statistical methods — all four are wiring/enforcement fixes to controls that already exist in some form.
