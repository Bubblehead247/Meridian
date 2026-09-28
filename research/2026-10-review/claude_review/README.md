# Claude's monthly review of Meridian (2026-09-28), same prompt as Qwen

`response.json` answers exactly the prompt Qwen got (`../qwen_review/packet.md` plus the JSON
schema), using packet facts only. **Bias:** Claude wrote the packet and made or proposed most of
the decisions it reviews, so this review is less independent than Qwen's. The findings aim
deliberately at those decisions.

## Grades

| Area | Qwen3.6-27B | Claude |
|---|---|---|
| Strategy | C | C |
| Risk | B | C |
| Execution | B | C |
| Research process | A | B |
| Governance | A | C |

## Where the two reviews differ

| Topic | Qwen | Claude |
|---|---|---|
| Switch to C1 | "Immediately" | **Not until the structure test includes 2008** (F1) — every structure comparison ran 2011–2026 |
| The −15% mandate | Agrees with it; start the core at 0.70 | Agrees with the target, but **nothing enforces it live**; it's a backtest property fitted to one path (F2) |
| Pod rules | Agree ("institutional standard") | **Three gaps:** no room in the risk budget (F3), 12 months of shadow can't validate an edge (F4), alpha P&L needs a pre-registered benchmark per pod |
| Faber | A shadow pod on the way to funding | **A drawdown tool, not alpha**: failed its test, Sharpe below 60/40 (F6) |
| long_term_etf in C1 | Not questioned | **Keeps hindsight**: instruments picked 2026-06-28, the gain is almost all 2022 (F7) |
| Non-inferiority test | Not questioned | **Margins inside sampling noise** (F8) |
| Execution | Next-open gap risk; use limit orders | Immaterial for SPY/IEF/SGOV; the −3.9 bps figure is gap noise (F11). **The real execution problem is monitoring** (F5) |
| Research process | A | B: the headline decision rests on a window without 2008 and on noise-level margins |

**Correction to my earlier verification of Qwen (`../qwen_review/run1/claude_verification.md`):**
I rejected Qwen's gap-risk finding by citing the −3.9 bps slippage. That figure compares next-open
fills with the prior close, so overnight gaps dominate it; it is evidence of neither low cost nor
harm. My conclusion stands (execution cost is immaterial for highly liquid ETFs), but the reason
I gave was wrong.

## Beyond the prompt: F1 checked with data (pre-registered `research/plans/structure_2008_check.json`)

Plan committed (4a9068c) before any 2008-inclusive baseline number was computed; run once through
the skill lab. Output: `research/plans/run_structure_2008_check.out`.

| Matched −15% worst drawdown, 2008-01..2026-06 | Exposure k | CAGR |
|---|---:|---:|
| **Baseline (8 sleeves, honest)** | 1.00 | +7.1% |
| C1 core | 0.78 | +6.5% |
| C3 (post-hoc holds) | 0.80 | +6.6% |
| 60/40 SPY/IEF | 0.47 | +4.8% |
| SPY | 0.26 | +4.2% |

**Plan rule → REOPEN**: C1 − baseline = −0.59%/yr against a −0.5% margin. Read with F8 in mind:
this is a near-tie that turns on one path's worst drawdown, not proof the baseline is better. But
the reason for it is clear:

| 2008-01..2010-12 (CAGR / max DD) | Timed (idle in T-bills) | Held |
|---|---:|---:|
| trend_following (SPY) | +5.3% / −21.4% | −2.8% / −51.9% |
| momentum (honest ETFs) | +6.0% / −18.2% | +5.7% / −34.8% |
| sector_rotation | −4.1% / −50.6% | −0.4% / −50.9% |
| long_term_etf (QQQ/IEF; no KMLM yet) | +0.3% / −28.5% | +6.1% / −24.3% |

- Trend and momentum earned their keep in 2008, cutting drawdown by more than half. That is
  exactly the crash protection the 2011–2026 tests couldn't see.
- Sector rotation didn't help even in 2008.
- long_term_etf, which C1 keeps, did **worse** than holding in 2008, adding to F7.
- The baseline's worst drawdown (−15.0%) came in 2016–20, not 2008. Timing protected in the slow
  2008 bear market but not in the fast 2020 crash.

**What this reopens (owner's call; nothing live changes, C1 stays unmerged):** retiring trend and
momentum at the switch removes the crash protection that let the baseline meet −15% without
scaling down.

**Next candidate (corrected 2026-09-28):** I first suggested 60/40 + trend + momentum. The
owner asked why, given that 60/40 scores lower, and he was right: at a −15% cap what counts is
return per unit of worst drawdown (baseline 0.47, C1 0.42, 60/40 0.28, SPY 0.21). 60/40's 2008
drawdown is what forced C1 down to 78%. The candidate with the fewest new choices is the honest
8-sleeve fund with sector rotation and SVXY retired to T-bills, pre-registered against the full
honest fund at a matched −15% drawdown.
