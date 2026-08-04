# Research-Integrity Phase 1 Summary

**Status:** Complete
**Date:** 2026-08-04

Note on numbering: this is Phase 1 of the *adversarial research audit's*
three-phase roadmap (Research Integrity → Research Capability → Portfolio
Intelligence), not the original project's Phase 0–10 sequence (see
`phase_1_summary.md`, "Data Engineering," a different, already-complete phase).

The audit found a statistically sound validation core (walk-forward, block
bootstrap, BH/Bonferroni) undermined by two foundational defects plus four
supporting gaps. Per the audit's own rule — no Phase 2/3 work until Phase 1 is
reliable — this phase implements exactly the six fixes it prioritized, nothing
else.

---

## What was fixed

### 1. Same-bar-close fills → next-open fills (the P0 defect)

`signals/backtest.py`'s `backtest()` filled a position at the exact close price
its signal was computed from — a return no live order could achieve. It now
accepts an optional `bars` frame; when it carries an `open` column, fills use
the *next* bar's open (the earliest price actually reachable after seeing the
close that produced the signal). Without `bars`, it falls back to the old
close-approximation, now visibly tagged via `meta["fill_realism"]` instead of
being silently the only behavior.

Three call sites needed the fix, not one — `families/base.py::Model.backtest`
accepted `bars` but dropped it before calling `backtest()`, and (found only
during live verification, not by the test suite) `validation/walkforward.py`'s
`_full_net_held` had the identical gap: it threaded `bars` into
`compute_scores` for indicators but dropped it before the backtest call. That
second gap meant the *actual `meridian validate` estimator-comparison path* —
the one the audit's own headline finding depends on — was still running
same-close fills even after the first fix landed. Both are now fixed.

**Measured effect** (SPY, 2010–2024, the 8 estimators in
`configs/validation_spy.yaml`, anchored walk-forward): `hull`'s out-of-sample
Sharpe drops from 0.413 to 0.259 once fills move off the inflated same-close
convention. No estimator was significant after BH correction under either
convention — the headline "no significant OOS alpha" conclusion holds, and now
rests on a less inflated number, exactly the direction the audit predicted a
correct fix should move it.

### 2. Survivorship-bias disclosure

Universe-wide validation reports now carry a dynamic `survivorship_biased` flag
in report metadata (`experiments/runner.py::is_survivorship_biased`), rendered
as a bold banner at the top of the report (`analytics/report.py`) — not just
buried in a docstring. True for the default Wikipedia-constituent path and for
the survivorship-free dataset's own `variant: survivor` baseline (which is
itself the biased comparison arm); False only for `source: survivorship` with
the default `variant: free`. This is disclosure, not a fix — no dataset
available today can honestly cover the full 2010–2019 research window
point-in-time.

### 3. OOS run-counting guard

New `pipeline/oos_guard.py` (`OOSGuard`), following the same JSON-per-record
pattern as `portfolio/ledger.py` and `pipeline/records.py`. `run_oos_stage`
and `run_pipeline` accept an optional `guard`/`symbol`; each call increments a
persisted counter for `(family, model, symbol)` and stamps
`detail["oos_run_count"]` on the result. `pipeline/records.py::StrategyRecord`
carries the count forward so any saved strategy whose "final" OOS pass followed
prior attempts against the same holdout is visible wherever records are
reviewed. Non-blocking by design — it flags, it doesn't refuse.

### 4. Append-only experiment run log

New `experiments/run_log.py`. Every `meridian validate` / `meridian universe`
run now appends a `RunRecord` (run ID, timestamp, config path + SHA-256 of its
raw text, best-effort git SHA, run kind, meta, report path, result summary) to
a JSON-lines file next to that run's report — never overwrites. `meridian list
runs` prints the log. Verified two consecutive runs against the same config
produced two log entries, not one overwritten.

*Correctness fix found in verification, not by the test suite*: unquoted YAML
dates (e.g. `start: 2010-01-01`) parse to `datetime.date`, which isn't
JSON-serializable — `append_run` crashed on the very first real CLI run against
`configs/validation_spy.yaml`. Fixed with `default=str` and covered by a
regression test.

### 5. Sleeve correlation wired into production

`portfolio/correlation.py`'s Pearson correlation matrix was correct but never
received real return data — every production caller of `run_monthly_review`
omitted `sleeve_returns`. `experiments/fund.py::run_fund`/`run_cs_fund` now
capture each sleeve's model backtest return series and pass it through.

*Second-order bug found in verification*: once real data flowed in,
`portfolio/correlation.py::average_correlation` still returned `nan` — a plain
`.sum()` over the correlation matrix means a single zero-variance (flat,
no-trade) sleeve's `NaN` correlation poisons the average for every other,
genuinely-correlated pair. Fixed with a NaN-aware `np.nanmean` over the
off-diagonal entries. `reporting/review_runner.py`'s paper-stage path remains
unwired (documented limitation, not silently dropped) — it aggregates from
saved scorecards, not return series, and wiring it means re-running saved
models against re-fetched prices, a heavier change than Phase 1's scope.

### 6. Graduation family/sleeve key mismatch

`pipeline/graduation.py::criteria_for_family` now resolves through
`FAMILY_TO_SLEEVE` before the `FAMILY_CRITERIA` lookup. `breakouts` and
`volatility` — which fund off the `trend_following` and
`experimental_research` sleeves, not sleeves of their own name — were silently
falling back to the generic 45%-drawdown default instead of their real
sleeve-weighted portfolio-drawdown-contribution bar. Both now get the correct
criteria.

---

## Verification

- **673 tests pass** (up from the pre-Phase-1 baseline; net new tests cover
  fill-realism tagging, the next-open math on a synthetic gap, the OOS guard's
  counting and per-symbol isolation, the survivorship banner, run-log
  append-only behavior and its date-serialization edge case, sleeve
  correlation population, and the graduation key-resolution fix).
- `meridian validate configs/validation_spy.yaml` run twice against cached SPY
  data (2010–2024): confirmed the report is regenerated each time,
  `reports/run_log.jsonl` accumulated to 2 entries (not overwritten), and
  `meridian list runs` reads both back correctly.
- Direct before/after comparison of the estimator validation table (same
  config, same data, `bars=None` vs `bars=<OHLC>`) confirmed the fill-price fix
  changes measured Sharpe materially, as shown above.
- `criteria_for_family("breakouts")` / `criteria_for_family("volatility")`
  spot-checked to return the `trend_following` / `experimental_research`
  sleeve-weighted criteria, not the generic default.
- No pollution of the real repo's `reports/`/`ledger/` directories from
  verification runs — `reports/run_log.jsonl` added to `.gitignore` alongside
  the other generated report types.

## Known limitations carried forward (explicitly, not silently)

- The cross-sectional/universe-wide backtest path
  (`pipeline/orchestrator.py::run_cross_sectional_pipeline`) still has no OHLC
  bars threaded through for a whole basket, so it stays on the flagged
  close-approx fallback. Threading multi-symbol OHLC through is Phase 2 scope.
- `reporting/review_runner.py::build_paper_review` (the paper-stage monthly
  review) still shows empty sleeve correlation — it already caveats "realized
  P&L is zero" in its output; wiring it needs re-running saved models against
  re-fetched prices.
- Survivorship bias in the default universe path is now always disclosed, not
  removed — no dataset available today covers the full 2010–2019 in-sample
  window point-in-time.

## Beyond Phase 1 (Phase 2/3, per the audit's roadmap — not started)

Phase 2 (research capability): PBO/Deflated Sharpe Ratio for best-of-N
selection risk across estimators/families; correct the multiple-testing
denominator for estimator collinearity (most of the 42 moving-average-family
estimators are cascaded/adaptive variants of one idea, not independent
hypotheses); intrabar stop/gap realism and a basic limit-order type; systematic
parameter-sensitivity sweeps.

Phase 3 (portfolio intelligence): marginal/correlation-adjusted risk
contribution in `risk_budget.py`; independent corporate-action verification;
data cache TTL; consolidate the two divergent Sharpe implementations
(`validation/stats.py` vs `analytics/metrics.py`).
