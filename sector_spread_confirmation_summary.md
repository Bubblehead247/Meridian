# Sector-Spread Confirmation Pass — Pre-registered

**Status:** Complete — **VERDICT: KILL.**
**Date:** 2026-06-25
**Scope:** Confirm-or-kill the one cost-surviving candidate from research pass 3 (a
diversified sector/calendar ETF-spread basket, lsma window 20) under a frozen,
pre-registered, out-of-sample protocol. Historical only (no live/paper wiring).

---

## Pre-registration (fixed BEFORE any OOS result was seen)

| Item | Value |
|------|-------|
| Strategy (frozen from discovery, **not** re-tuned) | `lsma`, `zscore`, window **20**, `entry=1.5`, equal-weight L/S |
| Selection window (in-sample) | dates **≤ 2019-12-31** — selection sees only this |
| Confirmation window (OOS) | dates **≥ 2020-01-01** |
| Candidate ETFs (fixed list) | sector SPDRs `XLB XLE XLF XLI XLK XLP XLU XLV XLY` + index `SPY RSP QQQ DIA IWM` |
| Pair rule (mechanical) | **all** unordered pairs, kept if Engle-Granger cointegration **p < 0.05 on the in-sample slice only** |
| Realistic cost | **2 bps per leg**, charged on *both* legs via true leg-level turnover |
| **PASS criterion (CONFIRM iff both)** | (1) bootstrap 95% Sharpe **CI low > 0**, AND (2) Monte-Carlo timing **Bonferroni q < 0.05** |

This attacks the four reasons the discovery was only a candidate: hand-picked pairs →
**mechanical rule**; one cell among many → **single frozen config (m=1 test)**; optimistic
flat `(1+|β|)` cost → **honest per-leg cost**; gross-only significance → **net-of-cost CI
gating**.

---

## What was built

- `meridian/features/pairs.py` — two pure additions:
  - `screen_cointegrated_pairs(prices, candidates, max_pvalue)` — forms all candidate
    pairs, Engle-Granger screen on the **passed (in-sample) slice**, returns the
    cointegrated set (lookahead-free by construction).
  - `leg_cost(held_weights, beta_by_pair, half_spread_bps)` — the **honest** per-leg
    cost: each pair's synthetic weight `w` ⇒ legs `w` and `−β·w`, cost
    `Σ(|Δw_A|+|Δw_B|)·bps/1e4`. Replaces the flat multiplier.
- `meridian/experiments/confirm_sector_spreads.py` — the pre-registered runner (frozen
  constants in the header; selects in-sample, builds the basket, nets per-leg cost,
  bootstraps + Monte-Carlo on OOS, prints CONFIRM/KILL + disclosures).
- Tests in `tests/test_pairs.py` (screen keeps cointegrated / drops independent, respects
  `max_pvalue`, uses only the passed slice; `leg_cost` reproduces two-leg turnover, β=1
  doubles, zero turnover is free). **372 tests pass, lint clean.**

---

## Results

### Mechanical in-sample selection (≤2019) → **19 pairs**
The rule selected a *different, larger* basket than the 6 hand-picked discovery pairs —
notably **XLU (utilities) dominates** (most-cointegrated leg: XLU/DIA p=0.0009, XLI/XLU,
XLB/XLU, XLU/SPY…). Cointegration was screened purely in-sample.

### Confirmation — OOS 2020+ (1627 bars), lsma w20, 2 bps/leg
| metric | value |
|--------|------:|
| net OOS Sharpe | **−0.398** |
| bootstrap 95% CI | **[−1.14, +0.25]** |
| net OOS total return | −0.244 |
| MC timing p-value | 0.634 |
| Bonferroni q (m=1) | 0.634 |

- **Criterion 1 (CI low > 0):** ✗ (−1.14)
- **Criterion 2 (q < 0.05):** ✗ (0.634)
- **→ VERDICT: KILL.**

### Why it fails — it is negative even *gross*
Break-even sweep (net OOS Sharpe vs half-spread): **0 bp → −0.15**, 1 bp → −0.28,
2 bp → −0.40, 5 bp → −0.77. The mechanically-selected basket loses **before any cost** on
true OOS data — costs only deepen it. Window sensitivity (non-gating): w10 **+0.10**,
w20 **−0.40**, w40 −0.23 — the discovery's "best" window 20 is the *worst* OOS, the
classic signature of an in-sample-overfit cell. Per-pair OOS Sharpes are scattered around
zero (best XLU/IWM +0.61, worst XLE/XLP −0.53) — no robust individual pair; the few
positives are noise across 19 names.

---

## Conclusion (honest, definitive)

**The sector-spread candidate does not survive a pre-registered out-of-sample test.** When
the pairs are chosen *mechanically* on pre-2020 cointegration and the *frozen* lsma-w20
config is run on 2020+, the basket is **negative even gross** (Sharpe −0.15 before cost,
−0.40 at realistic 2 bps/leg), the bootstrap CI spans zero, and the Monte-Carlo timing
test is insignificant (q=0.63). The discovery's promising number was a product of
**hand-picked pairs and an in-sample-overlapping window** — exactly the weaknesses this
pass was built to expose. In-sample cointegration did **not** translate into tradable,
cost-aware reversion out of sample.

This is the platform working as intended a fourth time: a theory-consistent candidate was
surfaced, then **killed** by a rigorous, honest test rather than published as an edge.

---

## Project headline — now complete and clean

Across every pass — US equities absolute, cross-sectional daily, intraday 1m–1h, crypto,
ETF twins, and sector spreads (discovery **and** this pre-registered confirmation) —
**there is no robust, cost-aware, out-of-sample mean-reversion edge.** The lone candidate
that survived discovery costs fails confirmation, negative even gross. The deliverable was
always the honest, reproducible machinery, and it has now retired its last open thread.

---

## Reproducing
`python -m meridian.experiments.confirm_sector_spreads` — prints the in-sample-selected
basket, the OOS verdict, the break-even sweep, per-pair contributions, and window
sensitivity.

## Known limitations
- 2020–2026 is **not perfectly untouched** (the discovery examined 2015–2026 sector
  spreads). This pass is far more rigorous but the **gold standard remains forward paper
  data** — the only definitive follow-on, deliberately out of scope here.
- `leg_cost` models a flat per-leg half-spread; real spread/impact varies by name and
  regime. (The result is negative even at 0 bps, so cost modeling is not the deciding
  factor here.)
