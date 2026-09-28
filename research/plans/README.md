# Pre-registered plans (skill lab)

Each `*.json` plan is committed before it runs; `meridian/validation/skill_lab.py` refuses an
uncommitted or edited plan, seals any hold-out from development runs, allows one final run per
plan fingerprint, and appends every run to `ledger.jsonl` (append-only: do not edit).

| Plan | Result (final run) | Consequence (fixed in the plan) |
|---|---|---|
| `faber_gtaa_5etf.json` — Faber 10-month SMA on SPY EFA IEF GSG VNQ, 2007-05..2026-09 | alpha vs equal-weight hold of the same ETFs +2.07%/yr, Newey-West t 1.68 (needed 2.0); vs SPY+IEF +1.01%, t 0.76. CAGR 5.2% vs 5.7% held; max DD −15.0% vs −48.6%; Sharpe 0.66 vs 0.45 (60/40 SPY/IEF: 0.77, −31.4%) | **Positive, not significant** → shadow sleeve from 2026-10-01, judged 2027-09-30. No live change. |
