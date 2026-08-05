# Meridian — Post-Implementation Research Integrity Verification Audit

**Date:** 2026-08-05
**Auditor stance:** independent verification, not the developer. Six parallel agents with no stake in the prior implementation work re-verified every claim against the actual repository — reading source, tracing call graphs, running mutation tests (deliberately breaking code to confirm tests catch it), running adversarial "try to fool the system" exercises, and independently computing numbers rather than trusting formulas by inspection. Findings below are agent-verified, not self-reported.

**Note on scope:** this working tree has no git remote and nothing from this session is committed (confirmed: `git remote -v` empty, ~46 files modified/untracked). All findings are against the *current uncommitted working tree*, not any commit. This fact is itself an audit finding — see §9.

**Security note:** during this audit, one verification agent detected and reported a prompt-injection attempt in its own tool output — a fake system-reminder tag claiming an unrelated code mutation it hadn't made was pre-existing and instructing it not to revert it or disclose it. The agent refused, reverted the file, verified restoration via `git diff` and a fresh test run, and disclosed the attempt. Independently re-verified after all six agents finished: the working tree is clean (`git status` matches the session's actual intentional changes) and the full suite passes 825/825.

---

## Executive verdict

# SUBSTANTIALLY HARDENED BUT NOT INVESTMENT-GRADE

The P0 look-ahead fixes are real and mutation-tested (deliberately broken and confirmed caught). Several P1/P2 controls are genuinely solid (ledger atomic writes and optimistic locking, CPCV purge/embargo leakage-free, DSR/HLZ formulas correct). But this audit found **multiple ways to defeat the newest research-integrity controls entirely within the documented public API**, the highest-severity being a **complete, silent bypass of the graduation-criteria version lock** (§2/§3) and a **UI-only universe-bias guard with a fully unguarded API underneath it** (§3). Both mean the specific defect they were built to close can still occur, just requires slightly more deliberate action than before. This is real, verified progress — not investment-grade.

---

## 1. Verified closed defects

| Defect | Fix | Evidence | Adversarial test | Confidence |
|---|---|---|---|---|
| Same-bar-close fill in CLI cross-sectional path (`orchestrator.py`) | `bars_by_symbol` now threaded through `cli.py` → `orchestrator.py` → `CrossSectionalModel.backtest` | Read directly; matches gap-analysis's own description of the prior broken state | **Mutation test**: reverted the fix, `fill_realism` reported `close_approx`; 2 of 42 tests (`test_signals.py`) caught it | High |
| Same fix, CLI gauntlet cross-sectional path | `bars_by_symbol` threaded through `_cmd_gauntlet` → `gauntlet_universe` | Read directly | Same underlying engine, same mutation-test coverage | High |
| Same fix, `interactive.py` basket-mode and gauntlet paths | `bars_by_symbol` threaded through both | Read directly | Same | High |
| Core fill engines (`signals/backtest.py`, `portfolio/portfolio.py`) use next-bar-open when available | Confirmed correct pre-existing logic, `fill_realism` flag set correctly | Read directly, line-level | Mutation test (see above) | High |
| Short/long/reversal fills get identical next-open discipline | Direction-agnostic fill math (`held * ret`) | Adversarial synthetic test: long→short reversal across two separate gaps, both legs correctly used their respective next-open price | Constructed and ran a targeted adversarial script | High |
| Ledger atomic writes | `tempfile.mkstemp` + `os.replace`, temp file unlinked on exception | Read directly | Confirmed no half-written file possible on any exception path | High |
| Ledger optimistic locking | `version` counter, `StaleLedgerWrite` on conflict | **Ran live**: two loaded copies, second save raised `StaleLedgerWrite`, on-disk file held the first (correct) save, not a corrupted mix | Direct execution against real `LedgerStore` | High |
| `history.jsonl` append-only in code (not filesystem-enforced) | Only ever opened `"a"` | Grepped every reference in the repo | N/A — code-convention only, not OS-enforced | High (as a code convention; not a hard guarantee) |
| CPCV purge/embargo leakage | No train-index falls inside purge/embargo windows | Adversarial params (n=733, n_groups=7, purge=4, embargo=6): 0 violations across 21 splits | Ran directly | High |
| PBO rank-symmetry property (asymptotic) | PBO≈0.5 under pure noise at realistic trial counts (12+) | Independently re-ran at 60 seeds: mean 0.500, std 0.214 at 12 trials — matches exactly | Ran directly, distinct seeds from the original test suite | High, **at 12+ trials only** — see §4 for the finite-sample caveat found at low trial counts |
| Harvey-Liu-Zhu t-stat formula | `mean/std*sqrt(n)`, no annualization bug | Bit-identical to manual calculation | Ran directly | High |
| DSR is not cherry-pickable via the report/CLI headline verdict | `## Verdict` and CLI summary read only `significant` (raw-m BH), never DSR in either form | Read `report.py`/`cli.py` directly; grepped `pipeline/` for DSR/PBO — zero matches, nothing wired into promotion | N/A — structural read | High for the report/CLI path; **residual risk** for anyone consuming the raw DataFrame directly (see §2) |
| `_gate_to_membership` (survivorship data) handles add→remove→re-add correctly | Loop over all snapshot intervals, not just one per ticker | Synthetic test: ticker added 2010, removed 2011, re-added 2012, removed 2013 — correctly gated all four transitions | Ran directly | High |
| Survivorship-free dataset actual coverage matches claim | 2010-01-04 to 2018-03-27 | Independently recomputed min/max across `data/*.csv` | Direct computation, excluded the metadata file that caused a false `1970-01-01` epoch date on first attempt | High |
| `market_hours`/trading-day gating in live scheduling | Present per a prior fix round | Confirmed present at `cli.py:500-521` | Read directly | Moderate (not re-tested live) |

---

## 2. Partially closed / mitigated defects

| Defect | What's fixed | What remains | Verdict |
|---|---|---|---|
| `fill_realism` visibility on every stage | Set correctly in `pipeline/backtest.py`, `pipeline/universe.py`, `pipeline/orchestrator.py` | **`pipeline/walk_forward.py` and `pipeline/oos.py` (single-asset) never surface `fill_realism` in `StageResult.detail`**, even though the underlying fill math is correct there. A silent-degrade-to-close-approx on these two stages produces no visible flag. | PARTIALLY CLOSED |
| Graduation-criteria version lock | Correctly blocks a version-label mismatch (`CriteriaVersionMismatch`), logs deliberate overrides | **Full silent bypass**: `advance()` trusts whatever `GraduationCriteria` object the caller passes and never re-derives it from `configs/graduation_criteria.yaml`. Passing a fabricated object with the *real* version string but a self-chosen threshold produces zero mismatch error, zero override requirement, **zero log entry** — worse than a logged override, this leaves no trace at all. Verified live: promoted a strategy this way with no `run_log.jsonl` entry. | **STILL OPEN in practice — the mechanism's core guarantee (label match implies threshold match) does not hold** |
| OOS run-guard | Genuinely increments on straightforward repeated calls, non-blocking by design (documented) | Trivially defeated by: deleting the counter file (reset confirmed live), renaming the model (fresh counter confirmed live), a case-sensitive symbol string (`"AAPL"` vs `"aapl"` don't merge, confirmed live), or a crafted key collision via `"__"` in family/model names (confirmed: two different triples produced the identical filename). Never blocks a pass regardless of count. | MITIGATED — real bookkeeping against casual re-runs, no resistance to deliberate evasion |
| Override-reason logging on graduation criteria changes | Every override IS logged to `run_log.jsonl` with a reason | The reason is **not content-checked** — `"retry"`, `"x"`, `"."` all satisfy it. Chained 10 sequential self-authored overrides, each loosening the bar, with no cap. | MITIGATED |
| Two parallel backtest engines (single-asset vs. cross-sectional) | Both hand-patched with the same fill-timing fix, each docstring cross-references the other | No shared helper, no test asserting the two produce identical output on equivalent input — the next divergent fix could still land in only one. | PARTIALLY CLOSED (architectural risk unchanged even though the specific bug is fixed in both) |
| CLI universe-bias guard | Correctly blocks the four `--universe`-gated CLI commands | **`get_universe()` itself has zero gating** — callable directly from Python with a historical date and zero warning. `experiments/runner.py::resolve_symbols()` (the config-driven `meridian validate/backtest/universe/paper <config.yaml>` path) is an entirely separate code path that never touches the guard at all. | **PARTIALLY CLOSED — architecturally, this is closer to STILL OPEN**: the guard exists at the UI layer only, exactly the failure mode the original brief warned against |
| Cumulative trial counting | Genuinely tracks `(estimator, deviation, window, scope)` and is disclosed in reports | Sample period, cost model, execution model, signal threshold, sizing scheme, and regime filter are **not** part of the tracked identity — a materially different research trial along any of those axes registers as zero new trials. Also: a tuple-vs-list `scope` mismatch silently undercounts to zero (dormant, not currently exploited by production call sites, but untested and unguarded). | PARTIALLY CLOSED |
| Effective number of tests (`m_eff`) | Correct at the boundaries (near-1 for identical series, near-m for independent ones), degenerate/zero-variance handled gracefully | Uses pairwise-complete correlation — trials with different NaN/availability patterns get correlated over non-comparable windows. Measured real sampling noise (std ≈0.18 at m=3, thin overlap) that isn't disclosed anywhere. | PARTIALLY CLOSED |
| Short-mechanics disclosure | The frictionless-shorts and live-flattening facts are both accurately disclosed in `analytics/report.py` | **The disclosure doesn't change what `graduation.py` does.** A short-reliant strategy can fully graduate and receive live capital allocation while `run_paper_session` never takes the short leg at all — confirmed end to end (backtest → WF/OOS → `passes_metric_bar` → `advance` → `portfolio/allocation.py` → monthly review report), with zero awareness of shorting anywhere in the graduation code. | **Originally classified P3 — re-classified P1 by this audit. Not "disclosed only," this is a research-to-live consistency failure with live capital consequences.** |
| Bootstrap block length | Functions correctly, handles degenerate cases | `block=20` is asserted, not derived. Measured a ~9-point p-value swing (0.529→0.621) from block-length choice alone under realistic AR(1) autocorrelation. | STILL OPEN (unjustified default, measurably consequential) |
| Metamorphic no-lookahead tests | Correctly catch a *severe* lookahead bug (full-series future peek) — confirmed via live mutation on both engines | **Do NOT catch a small (≤3-bar) lookahead regression.** The buffer built into these tests to tolerate the intentional 1-bar next-open fill lag also absorbs a genuine 3-bar future peek injected directly into `signals/backtest.py`'s return calculation — none of the 4 no-lookahead tests failed. This is exactly the size of bug a real code change is likely to introduce. | PARTIALLY CLOSED — real protection against gross lookahead, blind to small regressions |
| Metamorphic cost-monotonicity tests | Correctly designed logic (higher cost → lower or equal return) | **Seed-fragile.** A sign-inverted cost formula (`held.diff()` instead of `held.diff().abs()`, letting cost go negative) was injected into both engines; both tests passed anyway because each uses one fixed seed that happens not to expose the bug. Re-running the same broken formula across 14 seeds showed the monotonicity property actually violated in 6 of them (~43%). | STILL OPEN — the test exists but has a material chance of missing exactly the bug it's named for |

---

## 3. Remaining critical defects (ranked)

**P0**
- **Universe-bias guard is UI-only.** `get_universe()`, `resolve_symbols()` (config-driven CLI path), and any direct Python usage of `run_pipeline`/`run_universe_pipeline` bypass the guard entirely. This is the exact failure mode ("a CLI-only warning is insufficient if the underlying research API can silently execute a biased backtest") the original audit brief predicted, confirmed present.
- **Graduation-criteria version lock has a zero-log full bypass.** A fabricated `GraduationCriteria` object with a spoofed version label defeats the entire mechanism with no error and no audit trail. This is worse than the original "unlocked mutable criteria" defect in one respect: it *looks* protected (a version field exists, a mismatch check exists) while providing none of the claimed guarantee once a caller controls the criteria object, which every documented pipeline entry point (`run_pipeline`, `run_cross_sectional_pipeline`, `run_universe_pipeline`) allows via its `criteria=` kwarg.

**P1**
- **Short-mechanics is a graduation-to-live consistency failure, not a disclosure gap.** A short-reliant strategy's backtested Sharpe (which justifies capital allocation) is structurally unachievable live, and nothing in `graduation.py` detects or blocks this.
- **`regimes/labeler.py`'s breadth calculation uses today's S&P 500 constituents for historical dates**, and this is not cosmetic — it feeds `attach_regimes()` → permission gating (`families/permissions.py`) → whether a sleeve is allowed to trade on a given historical date in `meridian fund` runs. Confirmed live via a plain `meridian fund --symbol SPY` run with no `--universe` flag at all.
- **Economic-viability tooling (cost-stress sweep, capacity/participation, CPCV/PBO) is built but not wired into graduation.** A strategy can still graduate purely on Sharpe/expectancy/drawdown with zero connection to any of the P2 work. The tools existing is not the same as them being enforced.
- **The OOS holdout cannot currently be asserted pristine.** The run-guard was added roughly six weeks after `pipeline/oos.py` itself existed and was reachable; no record exists of how many times the fixed 2020–2022 holdout was touched during that window.
- **`git_sha` provenance is silently misleading on a dirty tree.** `run_log.py::_git_sha()` does a plain `git rev-parse HEAD` with no dirty-tree check. On this session's ~1,450-line uncommitted diff (spanning exactly the modules being audited — graduation, ledger, run_log, validation, risk_budget), any logged run right now stamps a commit hash that does not contain the code that produced the result, with no warning anywhere in the run log, CLI, or report.
- **New, previously undisclosed defect (found by this audit): partial per-symbol open-price coverage in the cross-sectional engine.** If one symbol in a basket has an `open` column that exists structurally but is entirely NaN for that specific symbol (a realistic real-vendor data gap, distinct from a delisting/membership gap), it silently contributes zero P&L on every bar for the whole run while `fill_realism` still reports `next_open` for the basket as a whole. No flag distinguishes this from a genuinely flat/zero-signal name.

**P2**
- Cost-stress and capacity tooling not enforced at graduation (see above — also listed here since it's independently a P2-scoped gap even setting aside the enforcement question).
- Bootstrap block length unjustified and measurably consequential.
- `m_eff`'s pairwise-complete correlation introduces undisclosed sampling noise under ragged trial overlap.
- Cumulative trial counting's tracked-dimension gap (sample period/cost model/execution model/threshold not tracked).
- `force=True` on ledger saves is fully ungated — any caller can silently clobber a concurrent write (traceable after the fact via a version/PnL discontinuity in `history.jsonl`, but not flagged as a forced overwrite specifically).

**P3**
- `cumulative_trial_count`'s tuple-vs-list `scope` equality landmine (dormant, not currently exploited, but untyped and untested against non-list scopes).
- OOS guard's collision-prone key construction (`"__"`-joined, unescaped).
- `override_reason` has no content/count gate (any non-empty string, unlimited overrides).
- Sub-symbol architectural risk between the two backtest engines (no shared cross-check test).

---

## 4. New defects introduced by the hardening work

This section is mandatory per the audit brief — these were not present before this round, or were newly discovered by this round's scrutiny:

1. **Graduation-criteria spoofing bypass** (§2/§3) — the version-lock mechanism itself is new this round; its bypass is a direct consequence of trusting caller-supplied criteria objects instead of re-deriving from the config file.
2. **Partial per-symbol open-coverage silent-zero defect** in `backtest_portfolio()` — pre-existing code, newly discovered by this audit's adversarial testing, not introduced by this round, but not previously known either.
3. **PBO finite-sample bias at low trial counts** (~5 trials: mean PBO 0.399 vs. the 0.5 null, roughly 3.3 standard errors low) — not a bug in the new CPCV code, but an unquantified property of it that the existing test suite's 12-trial-only validation doesn't cover. Worth adding to the module's own disclosure.
4. **`git_sha`'s misleading-on-dirty-tree behavior** was always present in `run_log.py`, but its consequences are now much larger, since this round added the exact modules (graduation, ledger, cpcv, capacity, external_benchmarks) whose provenance now can't be trusted from the log alone.
5. **`cumulative_trial_count`'s scope type-fragility** is new-round code with a latent, untested bug class.
6. **The metamorphic test suite has two confirmed blind spots**: it does not catch a small (≤3-bar) lookahead regression (the tolerance buffer sized for the intentional 1-bar fill lag also hides this), and its cost-monotonicity tests are seed-fragile — a sign-inverted cost formula that violates monotonicity in 6 of 14 tried seeds passes cleanly under the seed each test happens to use. Both were confirmed via live mutation testing, restored, `git diff` verified clean.
7. `tests/test_metamorphic.py`'s own docstring undercounts its coverage ("six invariants" when the file actually implements seven distinct properties) — cosmetic, but worth fixing since it's the kind of small inaccuracy that erodes trust in a document whose entire purpose is precision.

No false-positive warnings, incorrect cost accounting, or incorrect capacity assumptions were found. No previously-passing behavior was found broken (no true "REGRESSED" verdicts were assigned by any agent to a previously-verified-working piece of prior-round code) — the short-mechanics severity re-classification (P3→P1) is a *re-assessment*, not a regression, since the underlying frictionless-shorts behavior was never fixed, only disclosed.

---

## 5. Statistical methodology verification

| Method | Formula verified? | Assumptions verified? | Numerical reference verified? | Implementation verified? | Verdict |
|---|---|---|---|---|---|
| DSR | Yes (matches docstring; not re-checked against the primary paper this round — that was done in a prior round) | Yes (n<4→NaN, degenerate→NaN checked) | Not re-verified this round | Yes | VERIFIED CLOSED (dual-column cherry-pick risk closed at report/CLI level; open for raw DataFrame consumers) |
| Effective tests | Yes | Yes, including boundary + degenerate + ragged-overlap cases, numerically | N/A (not paper-pinned) | Yes | VERIFIED CLOSED at boundaries; STILL OPEN for undisclosed ragged-overlap noise |
| Bootstrap | N/A (standard method) | **Not derived** — block=20 asserted only | N/A | Yes | STILL OPEN — unjustified, measurably consequential |
| CPCV | Documented adaptation (not literal reproduction of the ML-features version) | Purge/embargo leakage-tested, 0 violations | N/A (structural) | Yes | VERIFIED CLOSED |
| PBO | NOT verified against primary paper (no network access, honestly disclosed both times) | Rank-symmetry verified at realistic trial counts; finite-sample bias found at low counts | NOT VERIFIED | Yes at the property level | NOT VERIFIED (numerical value) / MITIGATED (property-level) |
| MCTR | Euler decomposition arithmetic confirmed (sums to ~1, not proof of statistical validity — stated explicitly, not overclaimed) | Covariance safeguards present (joint dropna, min_periods, ridge) | N/A | Yes | VERIFIED CLOSED for numerical stability; the underlying covariance's *statistical* reliability remains inherently limited by sample-history length, correctly not oversold |
| HLZ benchmark | Verified — exact match | Yes | N/A (elementary) | Yes | VERIFIED CLOSED |

---

## 6. Research-to-live consistency matrix

| Feature | Backtest | WF/OOS | Paper | Live | Consistent? |
|---|---|---|---|---|---|
| Shorting | Fully simulated, frictionless | Same | Flattened to zero | Flattened to zero | **STILL OPEN** — backtest/live divergence with graduation blind to it |
| Cross-sectional sizing | Signed weights (equal_weight/inverse_vol across all signals incl. shorts) | Same | Long-only equal split among longs, a different scheme entirely | Same as paper | **STILL OPEN** |
| Position sizing (single-asset) | Raw ±1 positions | Same | Fractional shares, rounded | Same as paper | MITIGATED (paper=live identical; backtest/live delta immaterial) |
| Minimum order notional | Not modeled | Not modeled | `$1` minimum suppresses tiny deltas | Same | DISCLOSED ONLY (code comment, not in `_DISCLOSURES`) |
| Rebalance band | Not modeled | Not modeled | 5% band suppresses small adjustments | Same | NOT VERIFIED as disclosed anywhere |
| Fill price/timing | next-open when available | Same (both engines, mutation-tested) | Last close at order time, then real broker fill | Same | PARTIALLY CLOSED — correct as designed, but never reconciled against each other in any report |
| Costs | Flat bps | Same | Flat bps (simulated broker) | Real spread/slippage baked into Alpaca fills | DISCLOSED (in `_DISCLOSURES`) |
| Regime/permission gating | Downstream only, not in signal engine | Can be regime-split for reporting | **No regime reference anywhere in `run_paper_session`** | Same | **STILL OPEN** — `execution/trader.py::PaperTrader` has a working regime gate that is never actually invoked live |
| Corporate actions | `close`/`adj_close` both tracked, detector exists | Same | Raw `close`, no forced adjustment | Same | MITIGATED for the graduation-relevant path |
| Market hours | N/A | N/A | Trading-day gate confirmed present | Same | VERIFIED CLOSED |
| Missing-data handling | Fails closed (NaN→0, no fabrication) | Same | Silently drops unfetchable symbols, flagged as a skip decision | Same | VERIFIED CLOSED |

---

## 7. Data integrity

- **Point-in-time membership logic**: sound (verified via the add/remove/re-add adversarial test).
- **Survivorship coverage**: accurately re-verified at 2010-01-04 to 2018-03-27.
- **The AAPL/GE disclosure is itself inaccurate.** Independently re-checked against `constituents.csv`: **AAPL is never missing** from any snapshot 2006–2019 — the claim was simply wrong. **GE is missing far more extensively than described**: every snapshot from 2006-09-29 through 2014-02-28 (not "several 2010 snapshots"), masking GE out of the point-in-time universe for the entire 2010–2014 span. The underlying data-quality bug is real and appropriately left un-patched (correctly, per the "don't fix without a second source" discipline), but the *disclosure describing it* needs correcting.
- **Survivorship-free prices can be silently combined with survivor-biased/current metadata** (e.g., today's sector/market-cap classification from `EquityScreener`) — nothing in the codebase prevents or warns against this composition; it's simply two uncoupled public APIs.
- **Universe metadata leak beyond the CLI**: confirmed via `regimes/labeler.py` (§3) — affects real fund-run results, not just a display concern.
- **ADV point-in-time-ness**: not independently re-verified by this audit round in detail (agent scope did not report explicit findings on this beyond confirming `capacity.py`'s formulas are correct in isolation) — flagged as an area for a future pass.

---

## 8. Research-process integrity

- **Cumulative multiple testing**: real but narrower than "cumulative" implies — tracks estimator/deviation/window/scope only, not sample period, cost model, execution model, threshold, or sizing. A materially new research trial along any untracked dimension registers as zero.
- **Research lineage**: confirmed genuinely incomplete, and distinct from trial counting — no parent-experiment link, no hypothesis field, no cross-reference between `run_log.py`'s `RunRecord` and `pipeline/records.py`'s `StrategyRecord`, and **failed strategies leave no structured lineage record at all** (`record_from_pipeline` returns `None` on failure).
- **OOS protection**: real bookkeeping, trivially evadable by a determined researcher (delete file / rename model / case-sensitive symbol / key collision), never blocking regardless.
- **Graduation criteria**: version-lock mechanism exists but has a complete silent bypass via criteria-object spoofing; override mechanism has no content or count gate; `configs/graduation_criteria.yaml` can be edited directly and cited as policy with no independent review requirement (by design, not a bug, but worth naming plainly).
- **Strategy identity/history laundering**: confirmed live — a failed strategy cloned under a new ledger name has zero trace connecting it to its prior failure; there is no strategy-logic fingerprint anywhere in the schema.
- **Portfolio-selection multiplicity**: not separately audited this round (out of the six agents' scope) — remains an open question from the original brief.

**Adversarial "try to fool Meridian" results** (staying within the documented API):
- Repeated `validate()` calls with different estimator subsets: no tracking at all unless routed through the report wrapper, which only *discloses*, never *corrects* the correction. **STILL OPEN.**
- Repeated OOS evaluation for a second favorable pass: confirmed possible, non-blocking. **STILL OPEN.**
- Fresh ledger under a new name for a failed strategy: confirmed possible. **STILL OPEN.**
- Low-effort override reason pushing a marginal strategy through: confirmed possible. **MITIGATED** (logged, but content-free).
- Criteria-object spoofing: confirmed possible, **with zero log trace** — the single most severe finding in this category. **STILL OPEN.**
- Cherry-picking among DSR/eff/HLZ significance columns: closed at the report/CLI layer, open for direct DataFrame consumers. **DISCLOSED ONLY / STILL OPEN depending on entry point.**

---

## 9. Software integrity

- **Test count**: 825 collected (not 824 as claimed — trivial discrepancy), **819/825 passing when independently re-measured mid-audit**, traced to a confirmed concurrent mutation-testing collision between two audit agents on the same file, both auto-reverted. **Re-verified clean and green (825/825) after all agents finished** — see the note at the top of this document.
- **Mutation testing performed for real** (not merely described) on: fill-timing logic (2 engines), at least 4 of 7 metamorphic invariants (per agent instruction — see that agent's detailed sub-report, not reproduced in full here), OOS guard evasion routes, graduation spoofing, ledger conflict handling. In each case where a mutation SHOULD have been caught, it was, by either purpose-built adversarial scripts or the existing suite (`test_signals.py` specifically was confirmed to catch the fill-timing regression; `test_metamorphic.py` was confirmed to NOT catch it, correctly, since that's testing a different invariant class).
- **A genuine, novel bug was found**: `cumulative_trial_count`'s `scope` equality is raw Python `!=` against a JSON-round-tripped structure — a tuple-typed scope silently undercounts to zero. Dormant (both production call sites currently use lists), untested against non-list input, no type contract enforced.
- **Reproducibility**: confirmed broken in a way not previously flagged as sharply as it should be. No git remote, ~46 files of uncommitted changes, and `run_log.py::_git_sha()` has no dirty-tree check — it will silently stamp a commit hash on any run right now that doesn't contain the code that produced the result. This directly undercuts the project's own "reproducible from config alone" principle, and the failure mode is silent, not even a printed warning.
- **Duplicate implementations**: Sharpe ratio is genuinely consolidated (one source of truth, verified). The two backtest engines remain architecturally separate with no shared cross-check test, though both currently produce correct, independently-fixed fill logic.
- **Property-test depth**: moderate, not rigorous — `hypothesis`-based tests on estimators/deviations use reasonable but not adversarial ranges (no cross-scale stress, though defensibly excludes negative/NaN prices).

---

## 10. Investment-grade blockers

Only the issues that genuinely prevent investment-grade status — not padded:

1. **The universe-bias guard is UI-only**, with the underlying API (`get_universe()`, `resolve_symbols()`, direct pipeline calls) fully unguarded. Any research conducted outside the four specific CLI commands is not protected at all.
2. **The graduation-criteria version lock has a complete, silent, zero-log bypass.** As implemented, it provides the *appearance* of a control without the guarantee, which is arguably worse than no control at all for an auditor relying on its presence.
3. **Short-reliant strategies can graduate and receive live capital allocation on backtested numbers that are structurally unachievable live**, with zero mechanism connecting graduation to execution feasibility.
4. **The P2 economic-viability tooling (cost-stress, capacity, CPCV/PBO) is not enforced anywhere in the graduation decision.** Its existence does not change what determines whether a strategy gets capital.
5. **Reproducibility is not real right now**: no committed history, no remote, and a provenance field (`git_sha`) that will actively mislead rather than merely be absent.
6. **The OOS holdout's pristine status cannot be asserted** for any research predating the guard's addition six weeks into active development.

---

## 11. Required next actions

The minimum set to close the blockers — no new alpha features, no meta-labeling, no new estimators:

1. Move the universe-bias check into `get_universe()` itself (or a wrapper every pipeline entry point is forced through), not just the four CLI argparse handlers. Fix `experiments/runner.py::resolve_symbols()` to route through the same check.
2. Make `graduation.advance()` re-derive criteria from `configs/graduation_criteria.yaml` by version label rather than trusting a caller-supplied `GraduationCriteria` object's field values directly — the version string must be a lookup key into a canonical source, not just a label compared for equality.
3. Either (a) make `graduation.py` detect and gate on short-reliance (e.g., require a long-only re-backtest to determine the real promotable bar for any strategy with `allow_short=True` in its signal path), or (b) make `live_runner.py` actually support live shorting (Alpaca does), or (c) block short-capable strategies from graduating past `paper`. Pick one; a disclosure paragraph does not resolve this.
4. Wire `analytics/capacity.py::cost_stress_sweep`/`participation_rate` and `validation/cpcv.py::probability_of_backtest_overfitting` into `graduation.py::passes_metric_bar`/`check_promotion` as required gates, not optional disclosures.
5. Add a dirty-tree check to `run_log.py::_git_sha()` — at minimum, flag `git_dirty=True` in every `RunRecord` when `git status --porcelain` is non-empty, so provenance is honestly "unknown" instead of silently wrong.
6. Fix `regimes/labeler.py::build_regime_frame()` to accept and honor a point-in-time `as_of`/date parameter instead of always calling the dateless `get_universe()`.
7. Correct the AAPL/GE disclosure text in `data/survivorship.py` to accurately describe the GE gap's real scope (2006–2014, not "2010"), and remove the AAPL claim, which is false.
8. Fix the partial per-symbol open-coverage silent-zero defect in `backtest_portfolio()` — per-symbol fill-realism should be computed per-symbol, not once for the whole basket.
9. Add a `fill_realism` field to `walk_forward.py`/`oos.py`'s single-asset `StageResult.detail`.
10. Decide, deliberately, whether the current OOS holdout window should be treated as contaminated given the six-week guard-free gap, and either accept that risk explicitly or define a new locked holdout going forward.
11. Commit this work. A research result sitting only in an uncommitted working tree with no remote is not a durable result — this is table stakes for everything else in this list to mean anything over time.

---

## 32. Final count: how many credible ways could "Meridian found a profitable strategy" still be wrong, because of Meridian itself?

Counting only mechanisms confirmed live by this audit, not theoretical ones:

1. The universe could be look-ahead-biased via a direct API call or config-driven CLI run that never touches the CLI guard.
2. The strategy's regime permissions could have been computed using today's index breadth on historical dates via `labeler.py`.
3. The strategy could have graduated under a spoofed criteria object with zero record of it.
4. The strategy could rely on shorts that will never execute live, with backtested numbers no live capital could ever reproduce.
5. The strategy could have been evaluated against the OOS holdout more than once, undetected (guard evasion or pre-guard history).
6. The strategy's DSR/effective-test/PBO significance could reflect a cumulative-trial undercounting along an untracked dimension (sample period, cost model, threshold).
7. The strategy's cost-stress/capacity/PBO results, even if computed, might never have actually gated its promotion.
8. The strategy's MCTR/risk-budget numbers could rest on a covariance estimate with undisclosed ragged-overlap noise.
9. A basket-level cross-sectional result could include a symbol silently contributing zero real P&L due to per-symbol missing open data.
10. The exact code that produced the number might not correspond to what `git_sha` claims, with no warning.

**Ten.** Not all are equally severe — several are narrow or require deliberate researcher action — but numbers 1, 3, and 4 require no adversarial intent at all, just normal usage of a documented, non-obscure API path. That is too many for "investment-grade," and is why the executive verdict above is SUBSTANTIALLY HARDENED BUT NOT INVESTMENT-GRADE rather than anything higher.
