# Research-Integrity Phase 2 Summary

**Status:** Complete
**Date:** 2026-08-04

Note on numbering: this is Phase 2 of the *adversarial research audit's*
three-phase roadmap (Research Integrity → Research Capability → Portfolio
Intelligence), not the original project's Phase 0–10 sequence. See
`research_integrity_phase1_summary.md` for Phase 1.

Phase 2 gives Meridian better tools to tell a real edge from a lucky one, now
that Phase 1 removed the defects that could invalidate results outright. Four
items, all wired into the existing `validate()`/report pipeline rather than
new parallel infrastructure.

---

## What was built

### 1. Deflated Sharpe Ratio (`validation/deflated_sharpe.py`)

Answers: given N estimators were tested, is the best one's Sharpe still good
after accounting for how many chances it had to look good by luck? Formulas
were verified against real sources before implementation, not written from
memory — an earlier draft used a different, incorrect closed form for the
expected-maximum-Sharpe term; two independent sources (a secondary explainer
and the literal source of a commonly-cited reference implementation) agreed on
a different formula, which is what shipped. The expected-max-Sharpe function
was numerically checked against Bailey & López de Prado's own published
example (N=1000, V[SR]=1 → ≈3.26; this implementation gives 3.255). Skips PBO
(Probabilistic Backtest Overfitting) — its combinatorial train/test splitting
is redundant given Meridian already has real walk-forward OOS, block
bootstrap, and rotation-null Monte Carlo.

Wired into `validate()`: adds a `dsr_pvalue` column per estimator (reads
opposite a normal p-value — higher means more significant).

### 2. Effective-test correction for estimator collinearity (`validation/effective_tests.py`)

Most of the 42 estimators are correlated variants of a few underlying ideas
(17 of them are moving-average cascades), so correcting BH/Bonferroni as if
all were independent trials is needlessly conservative. Derives an effective
trial count directly from the estimators' actual OOS return correlation:
`n_eff = mean_corr + (1 - mean_corr) * m`. This formula was substituted in
after verification found it's the same one the DSR reference implementation
already uses to solve this exact problem — simpler than the originally-planned
Cheverud/Nyholt eigenvalue formula (borrowed from genomics), and now both
corrections share one methodology instead of two unrelated ones.

Wired into `validate()` and `validation/correction.py` (new `m_eff` param on
`bonferroni`/`benjamini_hochberg`/`correct`): adds `m_eff`, `q_value_eff`,
`significant_eff` columns. `q_value`/`significant` (raw-m, backward
compatible) still gate the report's headline `## Verdict` section — a result
that only clears the bar under the effective correction is visibly flagged in
a new `## Robustness` section, never silently promoted to the headline.

### 3. Intrabar stop/gap diagnostics (`signals/stop_diagnostics.py`)

Scoped down from the audit's literal suggestion (deliberately, per the plan):
no intrabar execution mechanism, no limit-order type — the signal engine is
daily-close-driven by design, and nothing today issues limit orders even
conceptually, so building one would be premature abstraction with no current
caller. What shipped instead: `flag_intrabar_stop_breaches`, a pure
post-hoc diagnostic that flags, per trade, whether any bar's high/low between
entry and the recorded exit already reached the eventual exit price earlier —
i.e. how many bars late the close-driven signal caught a move a live intraday
order could have caught sooner. Opt-in via `backtest(..., stop_diagnostics=True)`;
verified additive-only (identical `returns`/`equity` with the flag on or off).

### 4. Parameter-sensitivity diagnostic (`validation/sensitivity.py`)

`experiments/sweep.py` already does OOS-gated parameter *tuning* for the
`families/` strategy layer, but the core 42-estimator comparison — which
correctly holds `window=20` fixed per CLAUDE.md's "no estimator-specific
optimization" principle — had no check on whether that fixed choice is a
knife-edge one. `parameter_sensitivity` reruns the same walk-forward +
block-bootstrap Sharpe at neighboring windows (0.8x/1.0x/1.25x by default) and
reports a coefficient of variation and sign-stability flag — strictly
read-only, never selects a different window for the headline ranking (that
would reopen exactly the door the design principle closes).

Wired into `validate()` behind `sensitivity: bool = False` (config:
`validation.sensitivity: true`) — off by default since it roughly triples
per-estimator runtime.

---

## Verification

- **709 tests pass** (up from Phase 1's 673; net new: 11 DSR tests including
  the numeric reference-value check, 5 effective-tests-formula boundary
  tests, 2 `m_eff`-wiring tests on `correction.py`, 7 stop-diagnostics tests
  including a P&L-invariance check, 5 sensitivity tests, 5 new
  report-rendering tests for the `## Robustness` section).
- `meridian validate` re-run against cached SPY data (2010–2024, 8 estimators)
  with `validation.sensitivity: true`: report correctly shows `m_eff=3.838`
  (well below the raw m=8, reflecting real collinearity among the tested
  estimators), `dsr_pvalue` values all below the ~0.95 significance
  convention, and flags 6 of 8 estimators as sign-unstable across neighboring
  windows — a genuine, previously invisible robustness finding reinforcing
  the "no reliable edge" conclusion rather than contradicting it.
- Headline `## Verdict` unchanged by any Phase 2 addition in that live run —
  still "No estimator is statistically significant," with the new findings
  surfaced only in the additive `## Robustness` section.
- `flag_intrabar_stop_breaches` spot-checked against the same synthetic-gap
  fixture used in Phase 1's fill-realism test; behaves as expected (no
  breach flagged when the trade's span between entry and exit is too short
  to contain an earlier bar).
- No repo pollution: all new files are either explicitly new source under
  `meridian/`/`tests/`, or already-gitignored report output.

## Known limitations carried forward

- All four additions wire into the single-symbol `validation/pipeline.py::validate()`
  path only. `portfolio/validation.py::validate_universe` (the cross-sectional,
  universe-wide path) does not get DSR/`m_eff`/sensitivity — consistent with
  Phase 1's cross-sectional caveat, left for a future pass rather than
  silently assumed complete.
- The DSR's `n_trials` uses the raw estimator count, not `m_eff` — kept
  intentionally separate per the plan, so each correction is independently
  auditable; a future pass could examine whether a collinearity-adjusted
  trial count changes the DSR verdict meaningfully.

## Beyond Phase 2 (Phase 3 — Portfolio Intelligence, per the audit's roadmap, not started)

Marginal/correlation-adjusted risk contribution in `portfolio/risk_budget.py`;
independent corporate-action verification; data cache TTL; consolidating the
two divergent Sharpe implementations (`validation/stats.py` vs
`analytics/metrics.py`).
