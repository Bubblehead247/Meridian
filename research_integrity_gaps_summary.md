# Research-Integrity Audit — Remaining Gaps Closed

**Status:** Complete
**Date:** 2026-08-04

This closes out every item Phases 1–3 explicitly deferred as a "known
limitation carried forward." See `research_integrity_phase1_summary.md`,
`_phase2_summary.md`, `_phase3_summary.md` for the phases this builds on.

Investigation split the deferred items into three that were real, closable
code gaps; one that turned out to already be a non-issue on inspection; and
two that must stay disclosed limitations rather than be faked shut.

---

## Closed

### 1. Cross-sectional fill-price realism

The Phase 1 P0 fix (same-bar-close fills) never extended past the
single-asset engine — the cross-sectional/universe-wide path
(`portfolio/portfolio.py::backtest_portfolio`) still filled at the exact
close the signal was computed from. Fixed the same way, five files deep:

- `portfolio/portfolio.py::backtest_portfolio` — new `open_prices` param,
  next-open fill when given, `close_approx` fallback (tagged via
  `meta["fill_realism"]`) otherwise — identical logic to the single-asset fix.
- `portfolio/universe.py::run_universe_backtest` — already accepted
  `bars_by_symbol` (only used for `atr_norm` before); now also builds
  `open_prices` from it. `portfolio/validation.py::validate_universe` needed
  **no change** — it already forwarded `bars_by_symbol` here, so it inherited
  the fix automatically.
- `families/base.py::CrossSectionalModel.backtest` — gained `bars_by_symbol`
  (mirrors `Model.backtest` already having `bars`).
- `pipeline/universe.py` — all three stage runners plus
  `run_universe_pipeline` gained a `bars_by_symbol` passthrough; added a
  bars-aware sibling to `_slice_universe` for the OOS stage.
- `experiments/fund.py::run_cs_fund` — gained `bars_by_symbol`, forwarded
  to `run_universe_pipeline` and the sleeve-return capture call.

**Live-verified**: ran `validate_universe` on a 3-symbol basket (cached
XLK/XLF/XLE, 2015–2024) with and without `bars_by_symbol` — Sharpe values
differ for every estimator once next-open fills are used (e.g. `sma`:
-0.562 → -0.365), the same qualitative effect Phase 1 found for the
single-asset engine.

### 2. DSR + effective-tests correction for the universe-wide validator

`portfolio/validation.py::validate_universe` already loops per estimator
building a stitched OOS return series — the exact input Phase 2's
`deflated_sharpe.py`/`effective_tests.py` need. Added `dsr_pvalue`,
`m_eff`, `q_value_eff`, `significant_eff` the same non-destructive way
Phase 2 wired the single-asset `validate()` — raw `q_value`/`significant`
unchanged, stay the headline. The CLI's universe-validation report path
(`experiments/runner.py::run_universe_validation_report`) needed **no
changes at all** — it already builds `bars_by_symbol` from the full OHLCV
frames it loads and passes it straight to `validate_universe`, and
`analytics/report.py`'s `## Robustness` section already renders whatever
columns are present, so both fixes reached the actual CLI report for free.

**Deliberately not extended:** the parameter-sensitivity diagnostic. Adding
it would roughly quadruple the runtime of the already-heaviest CLI command
for a robustness question the single-asset sensitivity check already answers
at the estimator/window level — a stated decision, not a miss.

**Live-verified**: same 3-symbol run — `m_eff` came back ~2.1–2.14 (well
below the raw `m=3`), correctly reflecting real correlation between
`sma`/`ema`/`hull`; `dsr_pvalue` values were populated and sensible.

### 3. Sleeve returns wired into the paper-stage review

`reporting/review_runner.py::build_paper_review` had `paper_records`
(scorecards, no return series) but never built a `sleeve_returns` dict for
`run_monthly_review` — so its correlation matrix and (once Phase 3 landed)
MCTR were always empty, for both the Phase 1 and Phase 3 gaps at once, since
both read the same argument. New `_sleeve_returns_from_records`: per family,
picks the best-Sharpe *single-asset* record, re-backtests its model on its
saved symbol (cached prices — reconstructing a historical view for
reporting, not a live decision), collects the return series. Cross-sectional
records are skipped (need a basket, not one symbol). Wired into
`build_paper_review`'s existing `run_monthly_review` call.

**Live-verified against real data**: ran `build_paper_review()` against the
actual `saved_strategies/` directory (358 real records) — `avg_correlation`
came back populated (0.112, not NaN) and `marginal_risk_contribution` values
across the six active sleeves sum to ~100%, both previously empty on this path.

---

## Investigated, already closed — no code change

**Cache TTL "not adopted by live/paper callers"** (a Phase 3 known
limitation): checked `execution/live_runner.py::_fetch_prices` — it already
calls `load_ohlcv(sym, start, use_cache=False)`, bypassing the cache
entirely, specifically because "the live path must always see today's bar."
TTL adoption there would be a no-op; the staleness problem TTL exists to
solve was already solved a different (stricter) way. Corrected the record
here rather than writing pointless code to "close" a gap that wasn't open.

---

## Staying as disclosed limitations — not attempted

These require resources not available in this environment. Attempting a
partial fix and calling it closed would be exactly the overclaiming this
audit exists to prevent:

- **Survivorship-free universe coverage** (2013–2018 only, per
  `data/survivorship.py`) — no dataset covering the full 2010–2019 in-sample
  window exists; closing this needs acquiring a new external data source.
- **Corporate-action "independent" verification** — Phase 3's same-source
  anomaly detector is what's achievable without a second data provider;
  genuine independence needs one, not available here.
- **DSR's `n_trials` using the raw, not `m_eff`-adjusted, estimator count**
  — a deliberate Phase 2 design choice ("kept intentionally separate so each
  correction is independently auditable"), re-confirmed here as intentional,
  not an oversight to fix.

---

## Verification

- **739 tests pass** (up from Phase 3's 728; net new: portfolio-level
  next-open-fill tests mirroring the single-asset ones, a
  `validate_universe` collinearity test using three near-identical
  moving-average estimators, an end-to-end `run_cs_fund`
  `bars_by_symbol` test, and 5 new tests for
  `_sleeve_returns_from_records` covering the happy path, best-of-N
  selection, cross-sectional skip, and error-swallowing).
- Both fill-realism and DSR/`m_eff` changes verified against real cached
  market data (XLK/XLF/XLE, 2015–2024), not just synthetic fixtures.
- The paper-review sleeve-return wiring verified against the actual
  `saved_strategies/` directory (358 real records), not a mock.
- No default-behavior change anywhere: every new parameter is optional and
  additive; callers that don't pass `bars_by_symbol` get byte-identical
  `close_approx` results to before this work.
- No repo pollution from any verification run.

---

## Where this leaves the audit

Every item the three-phase roadmap either committed to or explicitly
deferred has now been addressed — fixed where fixable, and left honestly
disclosed where it isn't. The audit's original central question was whether
Meridian could be trusted to say "this doesn't work" — the answer was yes,
with conditions, contingent on the same-bar-fill fix and the survivorship
disclosure. Both are now in place across every code path that runs a
backtest, single-asset or cross-sectional, live-fund or paper-review. There
is no further scope from the original audit or its three-phase roadmap left
to close.
