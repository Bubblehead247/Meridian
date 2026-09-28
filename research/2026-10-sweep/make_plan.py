"""Write research/plans/full_sweep_2026_10.json: every candidate, fixed before any sweep result.

    python research/2026-10-sweep/make_plan.py
"""

import json
from pathlib import Path

UNIVERSES = {
    "U5": ["SPY", "EFA", "IEF", "GSG", "VNQ"],                        # Faber 2006/2007
    "U8": ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "IEF", "TLT"],   # Meridian's honest ETF set (2026-09)
    "U12": ["SPY", "QQQ", "IWM", "VGK", "EWJ", "EEM", "VNQ", "DBC",   # Keller & Keuning PAA (2016),
            "GLD", "HYG", "LQD", "TLT"],                               # as listed by Allocate Smartly
}

configs = []


def add(cid, family, tag, **params):
    configs.append({"id": cid, "family": family, "tag": tag, **params})


for u in UNIVERSES:
    for wt in ("EW", "IV"):
        add(f"HOLD-{wt}-{u}", "hold", "live-ready" if wt == "EW" else "engine-change", universe=u, weighting=wt)
    for L in (6, 10, 12):
        for wt in ("EW", "IV"):
            add(f"TREND{L}-{wt}-{u}", "trend", "small-build" if wt == "EW" else "engine-change",
                universe=u, sma_months=L, weighting=wt)
    for L in (3, 6, 12, "blend"):
        for wt in ("EW", "IV"):
            add(f"TSMOM{L}-{wt}-{u}", "tsmom", "small-build" if wt == "EW" else "engine-change",
                universe=u, lookback=L, weighting=wt)
    for L in (3, 6, 12, "blend"):
        for K in (2, 3, 4):
            add(f"DUALMOM{L}-K{K}-{u}", "dualmom", "engine-change", universe=u, lookback=L, top_k=K)
add("PAA-U12", "paa", "engine-change", universe="U12", protection="IEF", top_k=6, a=2)

for cid in ("BASELINE8", "TRIMMED", "C1", "C3"):
    add(cid, "meridian", "live-ready")
add("SIXTY40", "static", "live-ready")
add("SPY", "static", "live-ready")
add("ALLWEATHER", "static", "live-ready")

add("B1-6040+TREND10", "blend", "small-build", parts={"SIXTY40": 0.5, "TREND10-EW-U12": 0.5})
add("B2-6040+DUALMOM12K3", "blend", "engine-change", parts={"SIXTY40": 0.5, "DUALMOM12-K3-U12": 0.5})
add("B3-6040+PAA", "blend", "engine-change", parts={"SIXTY40": 0.5, "PAA-U12": 0.5})
add("B4-TREND10+DUALMOM12K3", "blend", "engine-change", parts={"TREND10-EW-U12": 0.5, "DUALMOM12-K3-U12": 0.5})

plan = {
    "id": "full-sweep-2026-10",
    "written": "2026-09-28, before any sweep result. Seen already: C1, the honest baseline, 60/40, SPY, Faber and the per-sleeve results of 2026-09-27/28 (the fixed and Faber rows below are not new information).",
    "hypothesis": "Find the model with the most return at the owner's -15% worst drawdown (no leverage), measure how much of any winner is selection luck, and decide whether a wider ETF universe helps.",
    "universes": UNIVERSES,
    "universe_note": "U12 is the published PAA list (Keller & Keuning 2016; tickers as listed by Allocate Smartly). HYG starts 2007-04 and counts as not held until it has enough history. Single stocks are out of scope: honest tests need paid point-in-time index membership. Managed-futures ETFs (long/short trend) start 2019-2020, too late to include 2008.",
    "families": {
        "hold": "fixed weights every month: EW = 1/N, IV = inverse 63-day volatility (normalised over the universe)",
        "trend": "each asset keeps its base weight if its month-end price > the mean of its last L month-end prices, else that weight goes to T-bills (Faber rule when L=10, EW)",
        "tsmom": "each asset keeps its base weight if its L-month total return beats T-bills over the same months (blend = mean of 1, 3, 6 and 12 months), else T-bills",
        "dualmom": "rank by L-month return; hold the top K at 1/K each if it beats T-bills over L months, else that slot is T-bills",
        "paa": "PAA: MOM = price / mean(last 13 month-end prices) - 1; n = assets with MOM > 0; IEF weight = clip((12 - n) / 6, 0, 1); the rest split equally over the top 6 by MOM",
        "meridian": "BASELINE8 = honest 8-sleeve fund (as in structure_2008_check); TRIMMED = BASELINE8 with sector_rotation and SVXY in T-bills; C1 and C3 as in structure_2008_check. Idle share of timed sleeves in T-bills",
        "static": "SIXTY40 = 60% SPY / 40% IEF; SPY; ALLWEATHER = 30% SPY / 40% TLT / 15% IEF / 7.5% GLD / 7.5% GSG (the widely published retail approximation, not Bridgewater's fund)",
        "blend": "fixed 50/50 mixes of two candidates' daily returns",
    },
    "tags": {
        "live-ready": "runs on today's engine as fixed-weight sleeves",
        "small-build": "needs a monthly-signal model per asset sleeve (small)",
        "engine-change": "needs dynamic weights, fixed-slot sizing or a variable protection share (engine change)",
    },
    "data": "Total returns (adj_close), 10 bps per side on weight changes, month-end signals traded at that close, T-bills = ^IRX/252 lagged a day, long-only, idle cash in T-bills",
    "test_start": "2008-01-01",
    "test_end": "2026-06-26",
    "holdout_start": None,
    "benchmark": "BASELINE8 (honest 8-sleeve fund) for the selection-value rule; SPY + IEF for the alpha the lab logs",
    "pass_rule": {"type": "sweep", "t_alpha_min": None, "rules": "see decision_rules"},
    "objective": "CAGR after scaling toward T-bills (k <= 1, no leverage) so the worst drawdown is no deeper than -15% over the window",
    "gates": [
        "TREND10-EW-U5 must reproduce the Faber run (2007-05-01..2026-09-25): CAGR 5.2%, max DD -15.0%, Sharpe 0.66 (rounded)",
        "BASELINE8 must reproduce structure_2008_check (2008-01..2026-06): CAGR 7.1%, max DD -15.0%",
        "If either fails, stop: no results are read",
    ],
    "tests": {
        "full_sample": "objective over 2008-01..2026-06 for every candidate (in-sample; shown, not trusted)",
        "forward_walk_forward": "each Jan 1 from 2012 to 2026: pick the best objective on 2007-07..end of the prior year (k fitted there), hold it for that year; stitch the years. Run over all candidates and over deployable ones (live-ready + small-build). BASELINE8, C1 and SIXTY40 go through the same yearly k-fitting for comparison",
        "reverse_crash": "pick the best objective on 2011-01..2026-06 (k fitted there), then run it on 2008-01..2010-12: the only unseen-crash test this data allows. Same for deployable-only, BASELINE8, C1, SIXTY40",
        "pbo": "Meridian's CPCV PBO on Sharpe (10 groups, 5 test groups, 21-bar purge and embargo) over 2007-07..2026-06: a general overfitting gauge for selecting by Sharpe, NOT a measure of the drawdown-capped selection",
        "deflated_sharpe": "for the Sharpe-best candidate, with n = 96 (this sweep) and n = 809 (cumulative trials in this line of research: 707 re-selection + 1 Faber + 3 structure + 2 check + 96)",
        "universe": "paired full-sample objective, same family and parameters: U12 vs U8, U12 vs U5, U8 vs U5",
    },
    "decision_rules": {
        "recommendation": "Among live-ready and small-build candidates, recommend the simplest whose full-sample objective is within 0.5%/yr of the best deployable one; tie-break by how often the deployable forward walk-forward picked it in 2022-2026. Simplicity order: static holds < per-asset trend/TSMOM < Meridian multi-sleeve structures < blends; fewer assets first. Flag it if its reverse-crash drawdown (k fitted on 2011-2026) is deeper than -20%. Also report the overall best (any tag) and its build cost, and say plainly when several candidates are a near-tie.",
        "selection_value": "The sweep 'adds value' only if the forward walk-forward track beats BASELINE8's walk-forward track by more than 0.5%/yr with a drawdown no deeper, AND the reverse-crash pick's 2008-10 drawdown is no deeper than BASELINE8's. Otherwise say the sweep found no reliable improvement over the baseline.",
        "expand_universe": "Recommend expanding to U12 only if U12 beats U8 in at least 2/3 of paired full-sample comparisons AND a U12 candidate is the reverse-crash pick or the majority of forward walk-forward picks. Otherwise: no expansion needed among ETF asset classes.",
    },
    "configs": configs,
    "consequences": {
        "any": "A proposal to the owner only. C1 stays unmerged; nothing live changes from this sweep.",
    },
}

out = Path("research/plans/full_sweep_2026_10.json")
out.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
print(f"{len(configs)} configs -> {out}")
