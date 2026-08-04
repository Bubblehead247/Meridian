# Research-Integrity Phase 3 Summary

**Status:** Complete
**Date:** 2026-08-04

Note on numbering: this is Phase 3 of the *adversarial research audit's*
three-phase roadmap (Research Integrity → Research Capability → Portfolio
Intelligence) — the final stage. See `research_integrity_phase1_summary.md`
and `research_integrity_phase2_summary.md` for the first two.

Phase 3 is about the portfolio layer: whether sleeves are evaluated by their
real contribution to the fund, not just standalone performance, plus three
smaller carried-forward items. All four reused machinery already built in
Phases 1–2.

---

## What was built

### 1. Marginal contribution to risk — MCTR (`portfolio/risk_budget.py`)

The existing `risk_contributions()` only measures each sleeve's share of
*stop-distance* heat — it can't tell you how much of the portfolio's actual
volatility a sleeve explains, which depends on its correlation with every
other sleeve, not just its own position sizing. `marginal_risk_contributions`
adds that: the standard risk-parity/component-VaR decomposition
(`CCTR_i = w_i·(Σw)_i / σ_p`, which sums exactly to `σ_p` by Euler's theorem
for degree-1-homogeneous functions). Formula verified against a standard
reference before implementation, not from memory.

Reuses Phase 1's `sleeve_returns` plumbing entirely — no new data collection.
Wired into `MonthlyReview`/the markdown report as a new `MCTR` column
alongside (not replacing) the existing heat-based `Risk Contrib`.

**Live-verified on real data** (SPY, 2015–2024, `run_fund()`): contributions
sum to exactly 1.0 across sleeves, and the run surfaced a genuinely
interesting result the heat-only metric couldn't show — `mean_reversion` has
a **negative** MCTR (-5.9%), meaning in this run it's acting as a portfolio
hedge (reduces total volatility) rather than a risk source, despite carrying
its own standalone variance. That's exactly the kind of correlation-aware
finding this item was meant to surface.

### 2. Corporate-action anomaly detection (`data/corporate_actions.py`)

Scoped honestly, not overclaimed: this is **not** independent verification
against a second data provider — none is integrated in this environment, and
adding one is out of scope. What it does: flags day-over-day jumps in the
`close`/`adj_close` ratio (which already encodes every adjustment yfinance
applied) that are too large to be a plausible dividend, tagging them
`likely_split` (matches a clean 2:1/3:1/etc. factor) or `unclean_adjustment`
(doesn't — worth a human look). Non-blocking, doesn't touch the data-loading
path by default; shipped as a tested, importable utility per the plan's own
"favor the smaller addition" guidance rather than adding a new CLI surface.

### 3. Data cache TTL (`data/cache.py`, `data/loader.py`)

`OHLCVCache` gains an opt-in `max_age_days` (instance default and per-call
override); `has_fresh()` checks file mtime. A stale file is treated as a
cache miss, triggering the existing re-download path — no new download logic.
Default (`max_age_days=None`) is byte-for-byte the old behavior, preserving
`cache.py`'s stated invariant ("clearing the cache must never change
results, only re-trigger downloads") — the TTL only automates *when* that
re-trigger happens, and only when a caller opts in.

### 4. Consolidated the two Sharpe implementations

`validation/stats.py::sharpe` and `analytics/metrics.py`'s inline Sharpe were
two hand-written copies of the same formula — confirmed mathematically
identical at `analytics`'s default `risk_free=0.0` before touching either.
Extracted a single `analytics.metrics.sharpe_ratio()`; `performance_metrics()`
now calls it internally, and `validation/stats.py::sharpe()` is a thin
wrapper around it. Verified all three (`sharpe_ratio`, `performance_metrics()
["sharpe"]`, `validation.stats.sharpe`) give bit-identical results on the same
input — one source of truth, no more drift risk.

---

## Verification

- **728 tests pass** (up from Phase 2's 709; net new: 6 MCTR tests including
  a hand-computed two-sleeve case verifying the Euler identity by hand and a
  four-sleeve sum-to-one check, 8 corporate-action anomaly tests covering
  clean/reverse splits, unclean jumps, normal dividends, and a
  never-modifies-input check, 5 cache-TTL tests including per-call override
  and the loader's forced-refetch path, 1 explicit three-way Sharpe
  consolidation regression test).
- `run_fund()` re-run on real cached SPY data (2015–2024): `MCTR` values
  populate and sum to exactly 1.0; the rendered monthly report shows the new
  `MCTR` column correctly, including the negative-contribution case.
- Sharpe consolidation confirmed with a direct three-way equality check on
  live data, not just the unit test.
- No default-behavior changes anywhere: cache TTL is opt-in
  (`max_age_days=None` preserves prior behavior exactly), corporate-action
  detection isn't wired into any load path, MCTR is additive to the existing
  `risk_contribution` field.
- No repo pollution from any verification run.

## Known limitations carried forward

- MCTR only populates when `sleeve_returns` is supplied — the same gap noted
  in Phase 1: `reporting/review_runner.py`'s paper-stage path still doesn't
  wire real return series in, so it won't show either correlation or MCTR
  data until that's addressed.
- The corporate-action anomaly detector is a same-source consistency check,
  not genuine independent verification — flagged explicitly rather than
  claimed as more than it is. A real fix would need a second data provider
  integrated, which is a larger, separate piece of work.
- Cache TTL is opt-in per-call; no caller in the live/paper path
  (`execution/live_runner.py`) has been switched to use it yet — the
  mechanism exists and is tested, but adopting it in the live session flow
  is a follow-up, not assumed done here.

---

## All three phases: where this leaves Meridian

Phase 1 fixed what could invalidate results outright. Phase 2 gave the
platform tools to tell a real edge from a lucky one. Phase 3 made the
portfolio layer evaluate sleeves by their actual contribution, not just
standalone numbers, plus closed three smaller gaps. Per the original audit's
own rule, nothing here was started before the phase before it was reliable —
each phase's live-data verification confirmed the previous phase's fixes were
still holding (e.g. Phase 3's `run_fund()` check exercises the same
same-close-fill fix from Phase 1 and the sleeve-correlation wiring it
depends on). Every formula/technique cited across all three phases that
wasn't basic arithmetic was checked against a real source before being
written into a plan, per the standing project rule adopted mid-Phase-2 — two
real errors were caught that way before they shipped (Phase 2's DSR formula
and effective-tests methodology).
