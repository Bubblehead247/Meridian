# Meridian Investment-Grade Research Integrity — Gap Analysis

**Date:** 2026-08-04 (original audit) — **updated 2026-08-05** with implementation status.
**Method:** Priority-0 independent code inspection (six parallel focused audits: execution/costs/capacity, statistical methodology, data/universe/survivorship, portfolio/MCTR, graduation/lineage, test coverage). Every claim below is anchored to a file:line the auditor actually read. Claims from the prior audit round that could not be re-verified are marked "not verified."

---

## 0. Status as of 2026-08-05

All P0, P1, P2, and P3 items from §6's implementation plan are **built and tested** (824+ tests passing across the whole suite). Section 6 below is kept as originally written, with a status line added under each item — treat §6 as the historical plan, this section as the current answer to "is it done."

**Done:**
- P0-A (same-bar-close fix), P0-B (universe look-ahead block) — closed, plus 3 more instances of the *same* class of bug found beyond the one originally flagged (gauntlet + interactive paths)
- P1-A (DSR/effective-test reconciliation + cumulative trial counting), P1-B (OOS guard wired in), P1-C (versioned graduation criteria), P1-D (all 6 metamorphic invariants), P1-E (survivorship data widened from 2013–2018 to 2010–2018, doc-fix + actual data extension both done)
- P2-A (cost-stress + capacity/participation analysis), P2-B (MCTR covariance safeguards), P2-C (CPCV/PBO), P2-D (ledger integrity: atomic writes, append-only history, optimistic locking)
- P3 (short-mechanics disclosure, Harvey-Liu-Zhu external t-stat benchmark); meta-labeling deliberately still not started, per the brief's own instruction

**Still open** (not silently dropped — explicitly deferred or out of scope for what was asked):
- Monthly-review report doesn't yet surface `check_stale` (the utility exists and is tested, just isn't wired into the report template)
- The master audit's full cross-run *research lineage* system (parent experiments, hypotheses, promotion decisions, holdout-touch flags per experiment) — P1-A built a narrower cumulative-trial-count piece of this, not the complete system originally scoped
- A data-quality bug found *during* the P1-E data extension: the vendored point-in-time constituent file (`constituents.csv`) incorrectly omits AAPL and GE from several 2010 snapshots (verified — both have been S&P 500 members since well before 2010). Documented in `data/survivorship.py`, not corrected — no second source was cross-checked to fix it safely.
- Nasdaq-100 and Russell 1000 remain fully survivorship-biased; all data work so far is S&P-500-specific
- `regimes/labeler.py`'s breadth calculation still pulls today's index constituents at any historical date — a related but separate look-ahead leak, never addressed
- 2018-04 through 2019-12 remains outside the survivorship-free dataset's coverage — a hard limit of the underlying WIKI/PRICES data (frozen since April 2018), not closable for free
- Nothing from this work is committed to git yet — all changes are local, uncommitted, on a repo with no remote configured

---

## 1. Brutally honest headline

**Meridian is not investment-grade, and the reason is more serious than "some gaps remain."** The single most important claim from the prior audit round — "next-bar execution was fixed, including the cross-sectional engine" — is **false as currently deployed**. The CLI's cross-sectional pipeline (`meridian/pipeline/orchestrator.py:132`, reached from `meridian/cli.py:214-224`) still fills at the same-bar close used to generate the signal, because it never passes `bars_by_symbol` through to the model. A parallel, independently-maintained cross-sectional pipeline (`meridian/pipeline/universe.py::run_universe_pipeline`) *does* have the fix. Both are live and reachable; nothing prevents a researcher from using the broken one and believing the bug is closed, because a differently-named module elsewhere was patched.

This is exactly the failure mode the user's brief warned about: the same defect existed in three places before and was fixed in fewer places than it existed. It happened again, one commit later, in a new pair of files.

Beyond that, the platform has a second live look-ahead leak (universe construction uses today's index constituents for any historical date, unconditionally by default) and a graduation pipeline whose criteria are ordinary mutable Python constants with no lock once a strategy enters research. Those two facts alone mean: **if Meridian reports a profitable strategy today, there are at minimum three independently sufficient mechanisms by which that conclusion could be an artifact — look-ahead fills, look-ahead universe membership, and unbounded/re-runnable graduation criteria — before statistics are even considered.**

The work done in the last four commits (fill-timing fix, DSR, effective-test correction, MCTR, corporate-action check, OOS run counter) is real, competently implemented in the places it was applied, and should be preserved. But it was applied to specific files, not to the invariant. Nothing in the codebase enforces "no path may fill at signal-bar close" or "no path may pull today's index list for a historical backtest" — so each fix is a patch, not a guarantee, and the audit found a second violation of the same invariant it was supposed to close.

---

## 2. Verified strengths (preserve these)

| Item | Evidence |
|---|---|
| Single-asset next-bar fill logic | `signals/backtest.py:99-119` — correct `open.shift(-1)` fill with an honest `close_approx` fallback flag when open isn't supplied |
| DSR formula | `validation/deflated_sharpe.py:36-131` — matches Bailey & López de Prado; raw (not excess) kurtosis verified by regression test; reproduces the paper's own worked example (3.255 vs. published ≈3.26) in `tests/test_deflated_sharpe.py:75-81` |
| Effective-test correction, within-run | `validation/effective_tests.py:28-51` — correctly built from real per-trial OOS return correlation, correctly feeds Bonferroni/BH (`correction.py:19-82`) |
| Append-only run log | `experiments/run_log.py` — genuinely append-only (`"a"` mode, L95), includes `config_sha256` to detect post-hoc config edits and `git_sha` |
| Point-in-time survivorship-free gating logic | `data/survivorship.py:139-165` — correctly NaNs a ticker's prices outside its actual index-membership window; the *logic* is sound even though its *coverage* is too narrow (§3) |
| Corporate-action same-source detector | `data/corporate_actions.py` — correctly flags 2:1 and 1-for-3 splits, has adversarial tests, and its docstring correctly refuses to overclaim independent verification |
| Regression tests for two of three known bugs | `tests/test_signals.py` (fill timing) and `tests/test_pipeline_stages.py` (graduation lookup-key) are precise, targeted regressions |
| Hypothesis-based property tests on estimator math | `tests/test_estimators.py:138-158`, `tests/test_deviations.py:155-173` — real, not just named; residual/z-score identities hold across all 42 estimators |
| Cache TTL design | `data/cache.py:26-63` — opt-in, default off, correctly scoped to "when to refetch" not "what's stored" |

---

## 3. Confirmed defects (found by this audit, not the prior one)

### 3.1 — CRITICAL — Same-bar-close look-ahead is still live in the production CLI path
`meridian/pipeline/orchestrator.py::run_cross_sectional_pipeline` (line 132) calls `model.backtest(prices_by_symbol, cost_bps=cost_bps)` — its function signature (lines 99–112) doesn't even accept a `bars_by_symbol` argument. `meridian/cli.py:214-224` (`_run_cross_sectional`) builds its universe from `close` only (`cli.py:219`) and drives this exact function. **Every cross-sectional backtest, walk-forward, or OOS run invoked via the CLI today uses same-bar-close fills.** A second, better implementation (`pipeline/universe.py::run_universe_pipeline`, used by `experiments/fund.py:142` and `interactive.py:182`) does thread `bars_by_symbol` through correctly. Two parallel orchestrators exist, one fixed and one not, and nothing marks either as deprecated.

### 3.2 — CRITICAL — Universe construction leaks future index membership
`meridian/data/universe.py:79-116` `get_universe()` scrapes **today's** Wikipedia constituent list for S&P 500 / Nasdaq-100 / Russell 1000, with no date parameter — a 2015 backtest and a 2026 backtest get the identical list. `runner.py:205-218::is_survivorship_biased()` correctly detects and reports this, but nothing *blocks* a backtest from running against the biased universe. Any experiment config with `data.source` unset or `yfinance` and a named universe (e.g. `configs/example_experiment.yaml`) is both survivorship-biased and look-ahead-biased with no code-level guard.

### 3.3 — HIGH — Survivorship-free dataset covers 44% of the stated in-sample window
`external/survivorship-free-spy/survivorship-free/generate.py:148` hardcodes a slice from 2013-02-28 to 2018-02-28. `data/splits.py:21` defines `DEFAULT_IN_SAMPLE = ("2010-01-01", "2019-12-31")`. Any "in-sample, survivorship-controlled" result is actually validated on less than half the claimed window, missing 2010–2013 and 2018–2019 entirely.

### 3.4 — HIGH — Graduation criteria are unlocked, mutable, per-call constants
`meridian/pipeline/graduation.py:23,31,43-48` — `GraduationCriteria` is a plain dataclass with defaults, passed as a fully overridable optional argument at every call site (`orchestrator.py:44,60-63`). There is no versioned/immutable criteria object bound to a strategy once it enters the pipeline, no maximum time-in-stage for any non-live stage (research/backtest/walk_forward/oos/paper have no dwell cap — a strategy can sit in "research" indefinitely being re-tried), and nothing stops re-running the same strategy against loosened criteria without resetting its history. Git history is the only change trail, and it is not distinguishable from ordinary code commits.

### 3.5 — HIGH — DSR's trial count and the effective-test correction disagree on what "a trial" is, and neither is cumulative
`validation/pipeline.py:114-120` and `portfolio/validation.py:144-150` set `n_trials = len(estimators)` for DSR — the estimator count in *that one call*, unrelated to `m_eff` (the correlation-adjusted count) computed two lines away in the same pipeline for Bonferroni/BH. The two corrections use two different, mutually inconsistent definitions of "number of tests," and **neither persists across calls** — running `validate()` five times with different windows/universes and reporting the best result faces no additional penalty. This is precisely the "research-selection bias operates at run level, not program level" gap the brief called the central unresolved problem, and it independently affects both statistical mechanisms Meridian relies on.

### 3.6 — MEDIUM-HIGH — OOS run-guard was added but has never executed and is wired to nothing
`meridian/pipeline/oos_guard.py` (added in the most recent commit, `d2d5c91`) is entirely opt-in (`orchestrator.py:49` defaults `oos_guard=None`) and **no call site anywhere in the codebase — not the CLI, not `experiments/runner.py`, not `gauntlet.py`, not `sweep.py`, not `fund.py` — ever instantiates it.** `ledger/oos_runs/` does not exist on disk. `gauntlet.py`, the module that produced the 358 files under `saved_strategies/`, never calls `run_oos_stage` at all. The mechanism the prior audit described as closing the "repeated OOS testing" gap has not run once in production and cannot retroactively detect whether the holdout was already inspected multiple times before it existed.

### 3.7 — MEDIUM-HIGH — No covariance conditioning or PSD safeguards in MCTR
`portfolio/risk_budget.py:143-144` computes `.cov()` on real per-sleeve returns with no shrinkage, no window, no `min_periods`, and no handling of pairwise-complete-observations producing a non-PSD matrix under differential missingness. No condition-number check, no eigenvalue floor, anywhere. The decomposition summing to 1.0 (noted by the prior audit as reassuring) is a property of Euler decomposition arithmetic, not evidence the covariance matrix is well-conditioned or statistically reliable.

### 3.8 — MEDIUM — Ledger is overwrite-on-save, not append-only or auditable
`portfolio/ledger.py:127-154` `LedgerStore.save()` fully overwrites one JSON file per strategy with no version history, checksum, or optimistic locking against a stale in-memory object. This contradicts the "auditable per-sleeve virtual ledger" framing and is a silent-drift risk distinct from the append-only run log (§2), which *is* sound.

### 3.9 — MEDIUM — No capacity, liquidity, or market-impact modeling exists anywhere
Confirmed by direct grep across `meridian/`: zero hits for ADV, participation rate, capacity, or impact as implementations (`analytics/report.py:29` is a disclosure sentence, not code). Costs are a single flat-bps scalar per run (`signals/backtest.py:115`, `portfolio/portfolio.py:93`), with no built-in sweep across cost levels. For a platform that now allocates virtual capital by sleeve, this means no candidate strategy has ever had its capacity or cost-sensitivity actually measured.

### 3.10 — LOW-MEDIUM — Shorts are fully frictionless
No borrow cost, dividend-owed, hard-to-borrow, or locate-failure logic anywhere (`signals/engine.py` positions are symmetric `{-1,0,+1}`). If short strategies are ever graduated toward live capital, this needs to be either modeled or the limitation stated in every report that includes a short sleeve.

### 3.11 — LOW-MEDIUM — Zero-variance-sleeve NaN fix exists but has no direct unit test, and 6 of 6 requested metamorphic invariants are entirely absent
`portfolio/correlation.py:39-56` does fix the NaN-propagation bug with `np.nanmean`, but only fund-integration tests (`test_cli_fund.py`) indirectly exercise it — no `tests/test_correlation.py` case constructs an actual degenerate 3+-sleeve matrix. Separately, none of the six invariants the brief specifically requested exist in the suite: price-scaling invariance, zero-signal→zero-P&L, cost monotonicity, no-lookahead/future-mutation, truncation invariance, split-invariance. Given that §3.1 is a second occurrence of a bug the test suite should have caught the first time, this is not a hypothetical risk.

---

## 4. Unverified claims (from the prior audit round, not independently confirmed by this pass)

- Whether `execution/broker.py`'s live-order path has any short-margin/short-availability logic — not fully read.
- Whether a live-trading cache-bypass path exists outside `meridian/data/` — plausible but not located in this pass.
- Whether any currently-scheduled production run actually exercises the survivorship-biased default universe end-to-end, versus always being routed through an explicit survivorship-free config — not traced through `cli.py`/`interactive.py` guard logic.
- Bootstrap block length (`bootstrap.py`, `block=20` fixed default) — whether it is ever tuned per-instrument, or always left at default in practice.

---

## 5. Risk ranking (severity-ordered, consolidating §3)

| # | Risk | Area | Severity |
|---|---|---|---|
| 1 | Same-bar-close fills live in CLI cross-sectional path | Execution/temporal | **P0 — blocks trustworthy research** |
| 2 | Universe construction uses today's index membership for any historical date | Data/temporal | **P0** |
| 3 | Graduation criteria unlocked/mutable per call, no dwell cap | Research process | **P1** |
| 4 | DSR trial count vs. effective-test count inconsistent; neither cumulative across runs | Statistics | **P1** |
| 5 | OOS run-guard unwired, never executed | Research process | **P1** |
| 6 | Survivorship-free data covers 44% of stated in-sample window | Data | **P1** |
| 7 | No covariance conditioning/PSD checks in MCTR | Portfolio | **P2** |
| 8 | No capacity/liquidity/impact modeling; no cost-stress sweep | Economic | **P2** |
| 9 | Ledger overwrite-on-save, no audit trail | Portfolio | **P2** |
| 10 | Six requested metamorphic/property invariants entirely absent | Testing | **P1** (enabling control for everything above) |
| 11 | Shorts fully frictionless | Economic | **P3** unless shorts are near graduation |
| 12 | CPCV/PBO absent | Statistics | **P2** (per brief's reassessment: case is now stronger given multi-strategy scope) |

---

## 6. Prioritized implementation plan

Ordered per the brief's Priority 0–10 sequence, adapted to what this audit actually found.

### P0-A — Fix the live look-ahead in the cross-sectional CLI path
- **Problem:** §3.1.
- **Why it matters:** Every number the CLI has ever produced for cross-sectional strategies (which is most of the multi-family platform, not just the original 42-estimator SPY experiment) is inflated by same-bar information.
- **Proposed solution:** Delete or explicitly deprecate `pipeline/orchestrator.py::run_cross_sectional_pipeline` in favor of the already-correct `pipeline/universe.py::run_universe_pipeline`; if both must exist, make `orchestrator.py` require `bars_by_symbol` (no silent close-only fallback) and raise if omitted, mirroring `close_approx` flagging in `signals/backtest.py`. Update `cli.py:214-224` to fetch and pass open prices.
- **Files:** `meridian/pipeline/orchestrator.py`, `meridian/cli.py`, callers of `run_cross_sectional_pipeline`.
- **Tests required:** Direct unit test asserting `run_cross_sectional_pipeline` raises or produces `close_approx`-flagged output when open prices are absent, mirroring `test_signals.py`'s existing pattern; re-run any published cross-sectional results and report the delta (same discipline as the original 0.413→0.259 correction).
- **Research implications:** Any previously-reported cross-sectional Sharpe/DSR result must be treated as provisional until re-run through the corrected path.
- **New failure modes:** None expected; this restores an already-proven pattern.
- **Priority:** P0.
- **Status: DONE.** Fixed `orchestrator.py` and `cli.py` as scoped, then found the *same* look-ahead pattern independently present in three more places (`cli.py`'s gauntlet command, and `interactive.py`'s basket-mode and gauntlet paths) that weren't part of the original finding — all four now fetch and thread `bars_by_symbol` through, and `fill_realism` is surfaced on every stage's output so a remaining close-only fallback is loud, not silent.

### P0-B — Block look-ahead universe construction by default
- **Problem:** §3.2.
- **Proposed solution:** Do not remove `get_universe()`'s live-fetch capability (useful for paper/live trading), but make any backtest/walk-forward/OOS entry point require an explicit `as_of` date and either (a) route to `survivorship_free_universe()` when the date falls in its covered window, or (b) hard-fail with a clear message when it doesn't, rather than silently falling back to today's list. Wire `is_survivorship_biased()` as a blocking check, not just an advisory flag, on any `kind in {backtest, walk_forward, oos}` run.
- **Files:** `meridian/data/universe.py`, `meridian/data/runner.py`, pipeline stage entry points.
- **Tests:** A backtest invoked with a historical `as_of` date and the default yfinance universe source must fail loudly, not silently degrade.
- **Priority:** P0.
- **Status: DONE**, via a targeted variant of the proposed fix. `cli.py`'s four `--universe`-accepting commands now hard-block on an index universe (SP500/NASDAQ100/RUSSELL1000) unless `--accept-survivorship-bias` is explicitly passed; the interactive REPL prints an unmissable warning instead (a hard block doesn't fit a live prompt). ETF universes (SPY/QQQ/IWM) are unaffected since they're single fixed tickers, not membership lists.

### P1-A — Reconcile DSR and effective-test trial definitions; make trial counting cumulative
- **Problem:** §3.5.
- **Proposed solution:** Define one canonical "trial" concept shared by both mechanisms. At minimum, pass `m_eff` (not raw `len(estimators)`) into DSR's `n_trials`, and extend `experiments/run_log.py`'s append-only log with the fields needed to compute a *cumulative* trial count across all runs in a research family (family/estimator/parameter/universe tuple), not just the current call. This directly extends the already-sound append-only infrastructure rather than replacing it.
- **Files:** `meridian/validation/pipeline.py`, `meridian/portfolio/validation.py`, `meridian/validation/effective_tests.py`, `meridian/experiments/run_log.py`.
- **Tests:** Numerical test that DSR's exp-max-Sharpe bar rises monotonically as cumulative trial count (read from the run log) increases across repeated calls.
- **Priority:** P1.
- **Status: DONE.** `dsr_pvalue_eff` (m_eff-based) added alongside the existing raw-m `dsr_pvalue` in both `validate()` and `validate_universe()`; verified numerically that correlated trials (m_eff < raw m) always raise `dsr_pvalue_eff` relative to `dsr_pvalue`. `experiments.run_log.cumulative_trial_count()` reads distinct (estimator, deviation, window) trials across all logged runs against the same scope and is surfaced in the report — this is the narrower cumulative-*counting* piece, not the full research-lineage system (see §0).

### P1-B — Wire the OOS run-guard into every OOS-touching pipeline, or remove it
- **Problem:** §3.6. A mechanism that exists but isn't called is worse than no mechanism, because it creates false confidence.
- **Proposed solution:** Call `OOSGuard.record_run()` from every path that touches the holdout, including `gauntlet.py` (which currently skips the OOS stage entirely — audit whether that's intentional; if `gauntlet.py` never touches holdout data, say so explicitly in its output rather than leaving it ambiguous). Make repeated-holdout-touch a hard warning surfaced in reports, not just a silently incremented counter.
- **Files:** `meridian/pipeline/oos_guard.py`, `meridian/pipeline/gauntlet.py`, `meridian/experiments/runner.py`, `meridian/cli.py`.
- **Priority:** P1.
- **Status: DONE.** Fixed a real binding bug first (`OOSGuard`'s default path was captured at class-definition time, so it could never be monkeypatched or effectively reconfigured), then made the guard on-by-default across all three pipeline entry points with real symbol/basket identity threaded from actual callers. `gauntlet.py`'s docstring now states plainly it never touches the OOS holdout, since that turned out to be true.

### P1-C — Version-lock graduation criteria
- **Problem:** §3.4.
- **Proposed solution:** Move `GraduationCriteria` defaults into a versioned config file (e.g., `configs/graduation_criteria.yaml` with an explicit version field), bind a strategy to the criteria version active when it entered `research`, and require an explicit, logged override reason (written to the append-only run log) to use a different version for an in-flight strategy. Add a maximum dwell time for non-live stages (research/backtest/walk_forward/oos/paper) so a strategy cannot be retried indefinitely without that also being visible in the log.
- **Files:** `meridian/pipeline/graduation.py`, new `configs/graduation_criteria.yaml`, `meridian/experiments/run_log.py`.
- **Priority:** P1.
- **Status: DONE.** `configs/graduation_criteria.yaml` carries an explicit version; `StrategyLedger` binds to the version active when it enters the pipeline; `graduation.advance()` raises `CriteriaVersionMismatch` on a version change mid-flight unless the caller passes `allow_criteria_override=True` with a reason, which gets logged to the append-only run log. `check_stale()` added for pre-live dwell-time limits (non-blocking) — not yet surfaced in the monthly review report (see §0).

### P1-D — Build the six metamorphic/property invariants
- **Problem:** §3.11. This is the control that would have caught §3.1 the first time.
- **Proposed solution:** Implement, using the existing `hypothesis` dependency and the same pattern already proven in `test_estimators.py`/`test_deviations.py`: price-scaling invariance, zero-signal→zero-P&L-except-costs, cost monotonicity, no-lookahead (mutate data after t, assert nothing before t changes — run this against **both** cross-sectional orchestrators to catch exactly the P0-A class of bug), truncation invariance, split-invariance. Add a direct degenerate-covariance unit test to `tests/test_correlation.py`.
- **Files:** new `tests/test_metamorphic_*.py` files; extend `tests/test_correlation.py`.
- **Priority:** P1 (do this alongside or immediately after P0-A/P0-B, since it's the regression guard for both).
- **Status: DONE.** All six invariants built in `tests/test_metamorphic.py` (13 tests): no-lookahead/future-mutation (3 variants, across both engines), truncation invariance, zero-signal→zero-P&L, cost monotonicity, price-scaling invariance, split invariance (with a negative-control case proving the failure mode is real and detectable), and duplicate-assets (verified against the exact closed-form weight-dilution formula, not just "doesn't crash").

### P1-E — Extend or clearly re-scope the survivorship-free dataset
- **Problem:** §3.3.
- **Proposed solution:** Do not silently use the 2013–2018 dataset as if it validates the full 2010–2019 in-sample window. Either (a) source additional survivorship-free coverage for 2010–2013 and 2018–2019 (research Norgate Data / Sharadar per the brief, but only after defining exactly which fields are needed — index membership dates, delisting dates, ticker-change mapping; do not buy a vendor because it advertises "survivorship-free"), or (b) if extension isn't feasible soon, change every report that claims "in-sample validated" to explicitly state the actual covered sub-window.
- **Files:** `meridian/data/survivorship.py`, `external/survivorship-free-spy/`, report templates.
- **Priority:** P1 (documentation fix is immediate/cheap; data extension is a larger, separately scoped project).
- **Status: DONE, both halves.** Doc-fix: `coverage_warning()` flags any requested window exceeding actual coverage, surfaced in the report banner. Data extension: `external/survivorship-free-spy/survivorship-free/generate.py` rewritten to pull the full available Nasdaq Data Link `WIKI/PRICES` range (a free account, not a paid vendor) instead of a hardcoded 2013–2018 slice, and to stop depending on iShares' holdings scraper (confirmed permanently broken — iShares stopped publishing historical holdings around 2020). Coverage widened from 2013-01 to 2018-02 (44% of the 2010–2019 target window) to **2010-01-04 to 2018-03-27 (82%)**. The remaining 2018-04–2019-12 gap is WIKI/PRICES' own hard limit, not fixable for free. A real data-quality bug was found in the vendored point-in-time constituent file during this work (see §0) and documented, not silently patched.

### P2-A — Cost-stress sweep and capacity/participation-rate analysis
- **Problem:** §3.9.
- **Proposed solution:** Start with the cost-stress sweep (0/5/10/25/50/100 bps) — cheap, reuses existing flat-bps parameter. Then implement participation-rate = order size / ADV, requiring ADV data (check whether yfinance volume is sufficient or another source is needed), and report capacity-adjusted Sharpe/return/drawdown at several participation levels before any strategy is described as economically investable.
- **Files:** `meridian/experiments/sweep.py` (extend), new `meridian/analytics/capacity.py`.
- **Priority:** P2.
- **Status: DONE**, deliberately more conservatively than the original proposal. `analytics/capacity.py` (new) has `cost_stress_sweep` at the brief's exact grid (0/5/10/25/50/100bps) for both engines. For capacity, chose *not* to synthesize a capacity-adjusted Sharpe from an uncalibrated impact-cost formula — real impact is closer to a square-root-of-participation law than linear, and neither has calibration data available. Instead `participation_rate`/`max_capacity`/`capacity_stress_sweep` report real ADV-based participation directly (bound by the single worst trading day, not the average).

### P2-B — Covariance conditioning safeguards in MCTR
- **Problem:** §3.7.
- **Proposed solution:** Add a condition-number check and a minimum-history/`min_periods` gate before `.cov()` in `risk_budget.py`; consider Ledoit-Wolf shrinkage if the condition number is high, but validate the shrinkage numerically against a reference before trusting it (per Rule 5 in the brief).
- **Files:** `meridian/portfolio/risk_budget.py`.
- **Priority:** P2.
- **Status: DONE.** Joint (not pairwise) `dropna()` before `.cov()` guards against a non-PSD matrix under differential sleeve missingness; `min_periods` floor (default 60) returns the safe degenerate result below it; ridge regularization applied when condition number exceeds a threshold, explicitly labeled as a numerical-stability fix, not a Ledoit-Wolf-style shrinkage estimator. New `marginal_risk_contributions_diagnostics()` exposes n_periods/condition_number/regularized.

### P2-C — CPCV/PBO
- **Problem:** §3.12; brief's own reassessment says the case is now stronger given the platform's broadened scope.
- **Proposed solution:** Implement as a complement to, not replacement for, existing walk-forward/bootstrap/DSR, sitting between research/development and the locked validation + untouched final holdout. Source and verify the reference implementation before coding (per Rule 5) — the DSR incident already showed what happens when a formula is implemented from memory.
- **Priority:** P2.
- **Status: DONE**, with an honesty caveat carried forward rather than hidden. `validation/cpcv.py` implements purged/embargoed combinatorial splits and the logit-rank PBO formula, wired into `validate(cpcv=True)` as opt-in. The formula's *numerical* output was **not** checked against the original paper's worked example (no network access available for that specific verification during implementation) — this is stated directly in the module's docstring rather than presented as confirmed. What *was* verified numerically: the property the estimator must have by construction — PBO≈0.5 under pure-noise trials (rank symmetry), PBO low when one trial persistently dominates.

### P2-D — Ledger integrity
- **Problem:** §3.8.
- **Proposed solution:** Convert `LedgerStore.save()` to append-versioned writes (e.g., write-then-rename with a version suffix, or append events and materialize current state), matching the pattern already proven sound in `run_log.py`.
- **Priority:** P2.
- **Status: DONE.** `LedgerStore.save()` now writes atomically (temp file + `os.replace`), appends every save to `history.jsonl` (mirrors `run_log.py`'s pattern), and does optimistic locking via a `version` counter — a save from a stale in-memory copy raises `StaleLedgerWrite` instead of silently clobbering a concurrent writer, with `force=True` for deliberate overwrites.

### P3 — Short mechanics, external t-stat benchmark, meta-labeling
- Per brief: document short-frictionless limitation now; evaluate Harvey/Liu/Zhu t>3 as an external skepticism check once P0/P1 items land; do not touch meta-labeling/triple-barrier until the above foundation is closed.
- **Status: DONE** (short mechanics + Harvey-Liu-Zhu), meta-labeling still untouched as instructed. The short-mechanics disclosure ended up more specific than originally planned: `execution/live_runner.py::run_paper_session` (confirmed as the single universal live/paper path) silently flattens every short signal to no position — so a short-capable strategy's backtested P&L includes trades that would never be taken live at all, not just trades taken at an optimistic cost. `validation/external_benchmarks.py` adds `t_stat_classical`/`significant_hlz` (the classical `SR·√n` t-statistic against a 3.0 hurdle) to both `validate()` and `validate_universe()`, framed explicitly as an independent cross-check, not a fourth correction stacked on top of the others.

---

## 7. Evidence an independent reviewer should check to verify these claims

1. Re-run a cross-sectional backtest via the CLI (`meridian/cli.py` `_run_cross_sectional`) before and after any fix to §3.1, and confirm the reported Sharpe changes in the same direction/magnitude class as the original 0.413→0.259 correction did for the single-asset engine.
2. Instantiate `get_universe()` with two different historical `as_of` dates and confirm whether the returned constituent list actually differs (it currently won't — confirming §3.2).
3. Diff `external/survivorship-free-spy/survivorship-free/data/*.csv` date range against `meridian/data/splits.py::DEFAULT_IN_SAMPLE` directly.
4. Grep `meridian/` for any call site instantiating `OOSGuard` outside `tests/` — confirm it's still zero (§3.6), or confirm it's been wired in if this plan is executed.
5. Read `validation/pipeline.py:114-120` and `validation/effective_tests.py` side by side and confirm whether `n_trials` for DSR still differs from `m_eff` used for BH/Bonferroni.
6. Run `pytest -k metamorphic` (or equivalent) once §P1-D lands and confirm all six invariants are present and passing, then intentionally reintroduce the §3.1 bug in a branch and confirm the no-lookahead test fails.
7. Check `configs/graduation_criteria.yaml` (once created) has a version field, and confirm `GraduationCriteria` can no longer be constructed inline at arbitrary call sites without an explicit override flag written to the run log.

---

## 8. Evidence for the 2026-08-05 completions specifically

1. Run `pytest -q` from the repo root — 824+ tests should pass, including `tests/test_metamorphic.py` (13), `tests/test_cpcv.py` (9), `tests/test_capacity.py` (14), `tests/test_external_benchmarks.py` (9), `tests/test_ledger.py`'s new versioning/history tests, and `tests/test_risk_budget.py`'s new covariance-safeguard tests.
2. Try `meridian pipeline momentum/relative_strength --universe SP500` from the CLI — it should refuse with a survivorship-bias error unless `--accept-survivorship-bias` is also passed (§P0-B).
3. Check `meridian/data/survivorship.py::KNOWN_PRICE_COVERAGE` — should read `("2010-01-04", "2018-03-27")`, not the original `("2013-01-28", "2018-02-28")`. Cross-check against the actual min/max dates in `external/survivorship-free-spy/survivorship-free/data/*.csv` directly (that directory is gitignored — regenerate via `generate.py` with a free `NASDAQ_DATA_LINK_API_KEY` if it isn't present locally).
4. Confirm the AAPL/GE constituent-file gap independently: `SurvivorshipDataset(...).members_asof("2010-06-01")` should be missing `"AAPL"` and `"GE"` — both should be flagged as wrong (Apple joined the S&P 500 in 1982), not treated as evidence those names weren't investable then.
5. Grep for `execution/live_runner.py::run_paper_session`'s short-handling (`sig <= 0` → flattened to 0) and confirm it's still the single path both `cli.py`'s `run-paper` command and `reporting/review_runner.py` route through — this is the basis for the short-mechanics disclosure in `analytics/report.py`'s `_DISCLOSURES` block.
6. Try `validate(..., cpcv=True)` and confirm a `pbo` column appears; read `validation/cpcv.py`'s module docstring for the explicit unverified-against-primary-source caveat rather than assuming the PBO number is independently confirmed.
