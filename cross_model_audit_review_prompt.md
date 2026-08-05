# Independent Review Request: Meridian Research-Integrity Audit + Proposed Next Steps

You are being asked to act as an **independent, skeptical second reviewer**. Another AI assistant (Claude) conducted a research-integrity audit of a quant research platform called Meridian, implemented fixes across four rounds of work, and then proposed a set of next steps grounded in cited sources. You did not participate in any of that work. Your job is to find what it got wrong, missed, overstated, or under-justified — not to summarize or praise it.

**Ground rule: do not be sycophantic.** If the audit and fixes look sound, say so specifically and explain why you checked and weren't just deferring. If something is wrong, vague, unverifiable, or overclaimed, say so plainly. Where you are not certain, say you are not certain rather than guessing confidently.

---

## 1. What Meridian is

Meridian is a quantitative research platform originally built to test whether any of 42 different "fair value" estimators (moving averages, Kalman filters, Ornstein-Uhlenbeck processes, robust statistics, etc.) produce a robust, exploitable mean-reversion trading signal on equities. The core research pipeline is: price data → estimator → deviation/z-score → constant entry/exit signal logic (held identical across all 42 estimators, by design, so only the estimator varies) → walk-forward out-of-sample backtest → statistical significance testing with multiple-testing correction.

The original finding (established before this audit) was: **no estimator shows a statistically significant out-of-sample edge on SPY after multiple-testing correction.**

The platform later expanded into a multi-strategy fund simulator: 8 active strategy "families" (mean reversion, trend following, momentum, breakouts, pullback continuation, sector rotation, long-term ETF, volatility; a 9th, event-driven, is a stub), a graduation pipeline (research → backtest → walk-forward → OOS → paper → pilot → proven → core → elite, with capital increasing at each stage), a virtual per-sleeve ledger, a portfolio-level risk budget, and a monthly review process.

## 2. The original audit's mandate

A prior audit pass (not shown to you in full — summarized here) evaluated Meridian against 25 sections covering: data integrity (look-ahead/survivorship/selection bias), backtest realism, statistical rigor (multiple testing, overfitting, data snooping), out-of-sample discipline, robustness testing (parameter sensitivity, regime analysis), portfolio construction (genuine diversification vs. labels), the strategy graduation pipeline's objectivity, research-to-live consistency, reproducibility, experiment tracking, performance metrics correctness, capacity/liquidity, and adversarial "can this system recognize a false discovery" tests. It explicitly instructed: *"Your job is not to make Meridian produce better-looking strategies. Your job is to make Meridian harder to fool."*

That audit produced a prioritized findings list and a three-phase roadmap:
- **Phase 1 — Research Integrity**: fixes required before any research result can be trusted.
- **Phase 2 — Research Capability**: better tools to distinguish real edge from luck.
- **Phase 3 — Portfolio Intelligence**: sleeve-level risk/allocation improvements.

## 3. What was actually implemented (four commits, summarized — full diffs available on request)

**Phase 1 (Research Integrity):**
- Fixed a same-bar-close fill bug in the backtester: positions were filled at the exact close price the signal was computed from (zero latency, unrealistic) instead of the next bar's open. Found and fixed in *three separate places* in the codebase (the second and third instances of the same bug were found only via live verification against real market data, not by the test suite).
- Measured effect on real SPY data (2010–2024, 8 estimators): the best estimator's out-of-sample Sharpe dropped from 0.413 to 0.259 once the inflated fills were corrected. The "no significant edge" conclusion held under both conditions.
- Added dynamic survivorship-bias disclosure to generated reports (previously only in static documentation).
- Added a run-counter for the out-of-sample holdout (to detect/flag repeated re-testing against the same "final" data).
- Added an append-only experiment log (previously each validation run overwrote the last).
- Wired real per-sleeve return data into the portfolio's correlation calculation (previously always empty/NaN); in doing so, found and fixed a second bug where a single zero-variance sleeve's NaN correlation was silently poisoning the average via un-guarded NaN propagation.
- Fixed a bug where two of nine strategy families were being graded against the wrong (more lenient) risk/drawdown criteria due to a lookup-key mismatch.

**Phase 2 (Research Capability):**
- Added Deflated Sharpe Ratio (Bailey & López de Prado). The implementer's first-draft formula (from memory) was wrong; caught by cross-checking two independent sources (a secondary explainer and literal reference-implementation source code) and numerically validated against the paper's own published example (expected max Sharpe ≈ 3.26 for N=1000 trials, reproduced as 3.255).
- Added a collinearity-adjusted effective-number-of-tests correction for the 42-estimator multiple-testing problem (most of the 42 are correlated variants of a few underlying ideas, e.g. 17 are different moving-average formulations). Originally planned using a genomics-derived formula (Cheverud/Nyholt eigenvalue method); switched mid-implementation to a simpler formula from the same López de Prado lineage after finding it in the same reference implementation being checked for the DSR formula.
- Added an opt-in intrabar stop-diagnostic (flags, without changing any executed P&L, whether a stop-loss would have been breached earlier intraday than the daily-close-driven signal caught it).
- Added an opt-in parameter-sensitivity check (reruns the walk-forward+bootstrap at neighboring window sizes to check if the fixed window=20 choice is a knife-edge result) — explicitly designed to never change which window is used for the actual ranking, to preserve the "no per-estimator optimization" comparison discipline.
- Explicitly decided *not* to implement full Probability of Backtest Overfitting (PBO/CSCV), reasoning that walk-forward + block bootstrap + rotation-null Monte Carlo already provide adequate OOS discipline and PBO's combinatorial cost wasn't justified. (**This is one of the decisions we want you to evaluate — see Section 5.**)

**Phase 3 (Portfolio Intelligence):**
- Added marginal contribution to risk (MCTR) — the standard Euler/risk-parity decomposition of portfolio volatility into per-sleeve components (verified against a standard reference before implementation). On a real 2015–2024 SPY-basket run, contributions summed to exactly 1.0 and surfaced that the mean-reversion sleeve currently has *negative* MCTR — i.e., it's hedging the rest of the portfolio rather than adding risk, something the platform's pre-existing heat-based risk metric couldn't show.
- Added corporate-action anomaly detection: flags large jumps in a stock's close/adjusted-close ratio (which encodes every split/dividend adjustment the data vendor applied) that don't match a clean split factor. Explicitly documented as a **same-source consistency check, not independent verification** — no second data provider is integrated, and the implementer was explicit that claiming otherwise would be overclaiming.
- Added an opt-in cache time-to-live for market data (stale cached files can be treated as a miss and re-fetched); default behavior unchanged.
- Consolidated two independently-hand-written copies of the Sharpe ratio formula (found to be mathematically identical at default parameters) into one shared implementation.

**Gap-closing pass (closing everything the three phases had deferred):**
- Extended the Phase 1 fill-timing fix to the platform's cross-sectional/multi-symbol backtest engine (five files) — this had been deferred in Phase 1 as "the single-asset case only."
- Extended Phase 2's DSR/effective-test correction to the cross-sectional validator.
- Wired real per-sleeve return data into the paper-trading-stage monthly review (previously only wired for the live-fund path), closing both a Phase 1 and Phase 3 gap at once.
- Investigated a fourth deferred item (cache TTL "not adopted by live callers") and found it was **already a non-issue** — the live data-fetch path already bypasses the cache entirely by design, for a stronger reason than TTL would provide. No code change was made; the implementer corrected the record rather than writing unnecessary code.
- Left two items as **explicitly disclosed, unfixed limitations** rather than attempting a fix: (a) the survivorship-free dataset only covers 2013–2018, short of the platform's stated 2010–2019 research window, and no free/available dataset covers the full window; (b) the corporate-action detector cannot become genuinely independent verification without integrating a second data vendor, which wasn't done.

Throughout, every cited external statistical/quant technique (DSR, effective-test correction, MCTR/Euler decomposition) was checked against real external sources before being written into an implementation plan — this discipline was adopted *mid-project*, after a first attempt used a memorized (and wrong) formula for the DSR's expected-max-Sharpe term.

## 4. The reviewing implementer's own self-critique (for you to independently evaluate, not just accept)

The implementer flagged these as weaknesses in its own work:
1. This was a single-reviewer process throughout — the same AI found the issues, wrote the fixes, wrote the tests, and verified the results, with no independent check on the fixes themselves (only some formulas were externally verified).
2. **Capacity and liquidity analysis** — a section of the original audit's mandate — was never addressed in any phase, and wasn't even tracked on the "known limitations" list; it was simply dropped.
3. Transaction costs remain a flat basis-point assumption everywhere; only fill *timing* was fixed, not fill *size* realism (market impact).
4. Multiple-testing correction operates per-run; there's no accounting for cumulative look-elsewhere effect across the platform's entire research history, even though the infrastructure to compute that (an append-only run log) now exists.
5. Live-data verification was done on small samples (a handful of symbols/estimators at a time), not the full estimator registry or universe.

## 5. Proposed next steps (for you to independently evaluate)

The implementer proposed these next steps, each with a cited source:

1. **Adopt point-in-time, survivorship-free data** (e.g., Norgate Data, or Sharadar Core US Fundamentals via Nasdaq Data Link) to finally close the survivorship-bias gap that's been disclosed-but-unfixed since Phase 1.
2. **Implement Combinatorial Purged Cross-Validation (CPCV)** as a fuller treatment than the DSR-only approach chosen in Phase 2, generating a distribution of OOS Sharpe outcomes across purged/embargoed path combinations rather than one path, and directly measuring PBO — from the same Bailey/López de Prado body of work already partially adopted.
3. **Apply a stricter external significance benchmark**, citing Harvey, Liu & Zhu (2016), who argue that given how many return-predicting factors have already been published across the field, a t-statistic hurdle of 3.0 (not the conventional 2.0) is warranted — proposed as a benchmark alongside Meridian's internal multiple-testing correction, not a replacement for it.
4. **Replace flat-bps transaction costs with an Almgren-Chriss-style market-impact model** (permanent + temporary impact, square-root law).
5. **Build the capacity/liquidity analysis that was never done** — position sizing constrained by average-daily-volume participation rate.
6. **Adopt meta-labeling / triple-barrier exits** (López de Prado) as a structural upgrade to the platform's current fixed-threshold entry/exit signal logic.
7. **Add property-based tests** (the `hypothesis` library, already a declared dependency) targeting the backtest engine's fill/cost/return invariants specifically, to catch the class of bug this entire audit was built around automatically on future changes, rather than relying on manual live-verification catching it.
8. **Build a cross-run multiple-testing ledger** using the existing append-only experiment log to account for cumulative look-elsewhere effect across the project's full history, not just within a single validation run.

---

## 6. What we want from you

Please review critically and answer explicitly:

1. **Audit soundness**: Does the *reasoning* behind each Phase 1–3 fix hold up? Are there fixes described above that sound plausible but might not actually address the stated problem, or might have introduced a new problem?
2. **Verification adequacy**: Is "live-verified against real cached SPY/QQQ/basket data, plus a full test suite" sufficient evidence for the claims being made, or is this still asserting more confidence than the evidence supports?
3. **The PBO/CPCV deferral**: Was deferring full PBO/CSCV in favor of DSR + existing walk-forward/bootstrap/Monte-Carlo a defensible call, or should CPCV have been done in Phase 2 rather than deferred to "next steps"? Argue it either way with reasoning, not just agreement.
4. **Missed findings**: Given the description of Meridian's architecture in Section 1 and the fixes in Section 3, what would you look for that doesn't appear to have been checked? Be specific about *why* each thing you name matters for this specific platform, not generic best-practice list-making.
5. **Next-steps critique**: For each of the 8 proposed next steps in Section 5 — is the cited source actually saying what it's being used to justify? Is the prioritization defensible (what should actually come first)? Is anything proposed likely to be lower-value than it sounds, given what's already been built?
6. **The self-critique itself**: Is the implementer's own list of weaknesses (Section 4) complete, or self-serving in what it chose to highlight vs. omit?

Where you can, cite your own sources for any claims you make about correct methodology. If you're not sure whether a described fix is correct without seeing the actual code, say that explicitly rather than assuming it's fine.
