"""Run the pre-registered sweep research/plans/full_sweep_2026_10.json once.

Monthly families (hold / trend / tsmom / dualmom / PAA) are computed here from
month-end signals traded at that close; Meridian's fixed structures are rebuilt
exactly as in research/plans/run_structure_2008_check.py. Two reproduction gates
run first; if either fails the script stops before printing any result.

    PYTHONPATH=. python research/2026-10-sweep/run_sweep.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path[:0] = ["research/2026-09-returns-study", "research/plans"]
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

EVAL_START = "2007-05-01"
common.START = "2006-01-01"
common.index = lambda: common.tr("SPY").loc[EVAL_START:].index
from common import hold, index, sleeve, tbill, tr  # noqa: E402

from meridian.data import load_ohlcv  # noqa: E402
from meridian.validation import skill_lab as lab  # noqa: E402
from meridian.validation.cpcv import probability_of_backtest_overfitting  # noqa: E402
from meridian.validation.deflated_sharpe import deflated_sharpe_ratio  # noqa: E402

PLAN = "research/plans/full_sweep_2026_10.json"
OUT = Path("research/2026-10-sweep")
COST = 10 / 1e4
BUDGET = -0.15
A, B = "2008-01-01", "2026-06-26"
TRAIN0 = "2007-07-01"


# ---------------------------------------------------------------- statistics
def cagr(r):
    r = r.fillna(0.0)
    return float((1 + r).prod() ** (252 / len(r)) - 1) if len(r) else np.nan


def maxdd(r):
    eq = np.cumprod(1 + r.fillna(0.0).to_numpy())
    return float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else np.nan


def sharpe(r):
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan


def fit_k(r, t, budget=BUDGET):
    """Largest exposure k <= 1 (rest in T-bills) whose worst drawdown is no deeper than budget."""
    if maxdd(r) >= budget:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if maxdd(mid * r + (1 - mid) * t) < budget:
            hi = mid
        else:
            lo = mid
    return lo


def objective(r, t):
    k = fit_k(r, t)
    return cagr(k * r + (1 - k) * t), k


# ---------------------------------------------------------------- data
def main() -> int:
    plan = lab.load_plan(PLAN)
    cfgs = plan.data["configs"]
    U = plan.data["universes"]
    tickers = sorted({s for u in U.values() for s in u} | {"IEF", "SPY", "TLT", "GLD", "GSG"})

    spy_full = tr("SPY")
    FULL = spy_full.index
    P = pd.DataFrame({s: tr(s).reindex(FULL).ffill(limit=5) for s in tickers})
    irx = load_ohlcv("^IRX", "2006-01-01", use_cache=False)["close"]
    rf_full = (irx.reindex(FULL).ffill() / 100 / 252).shift(1).fillna(0.0)
    daily_ret = P.pct_change()
    ME = pd.DatetimeIndex(pd.Series(FULL, index=FULL).groupby([FULL.year, FULL.month]).max().values)
    M = P.loc[ME]
    valid = M.notna()
    TBm = (1 + rf_full).cumprod().loc[ME]

    def run_targets(T):
        cols = list(T.columns)
        w = T.reindex(FULL).ffill().shift(1).fillna(0.0)
        cash = 1.0 - w.sum(axis=1)
        turn = T.fillna(0.0).diff().abs().sum(axis=1)
        tc = turn.reindex(FULL).fillna(0.0).shift(1).fillna(0.0) * COST
        r = (w * daily_ret[cols].fillna(0.0)).sum(axis=1) + cash * rf_full - tc
        return r.loc[EVAL_START:]

    def base_w(univ, wt):
        v = valid[univ]
        if wt == "EW":
            return v.astype(float) / len(univ)
        vol = daily_ret[univ].rolling(63).std().loc[ME]
        inv = (1.0 / vol).where(v & vol.notna() & (vol > 0))
        return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)

    def momentum(univ, L):
        if L == "blend":
            r = sum(M[univ] / M[univ].shift(l) - 1 for l in (1, 3, 6, 12)) / 4
            h = sum(TBm / TBm.shift(l) - 1 for l in (1, 3, 6, 12)) / 4
        else:
            r = M[univ] / M[univ].shift(L) - 1
            h = TBm / TBm.shift(L) - 1
        return r, h

    def family_returns(c):
        univ = U[c["universe"]]
        fam = c["family"]
        if fam == "hold":
            return run_targets(base_w(univ, c["weighting"]))
        if fam == "trend":
            L = c["sma_months"]
            sig = (M[univ] > M[univ].rolling(L).mean()) & valid[univ]
            return run_targets(base_w(univ, c["weighting"]) * sig)
        if fam == "tsmom":
            r, h = momentum(univ, c["lookback"])
            sig = r.gt(h, axis=0) & valid[univ]
            return run_targets(base_w(univ, c["weighting"]) * sig)
        if fam == "dualmom":
            r, h = momentum(univ, c["lookback"])
            K = c["top_k"]
            top = r.rank(axis=1, ascending=False, method="first") <= K
            sel = top & r.gt(h, axis=0)
            return run_targets(sel.astype(float) / K)
        if fam == "paa":
            mom = M[univ] / M[univ].rolling(13).mean() - 1
            n = (mom > 0).sum(axis=1)
            bf = ((len(univ) - n) / 6.0).clip(0.0, 1.0)
            top = mom.rank(axis=1, ascending=False, method="first") <= 6
            T = top.astype(float).mul((1 - bf) / 6.0, axis=0)
            T["IEF"] = bf
            return run_targets(T)
        raise ValueError(fam)

    R: dict[str, pd.Series] = {}
    for c in cfgs:
        if c["family"] in ("hold", "trend", "tsmom", "dualmom", "paa"):
            R[c["id"]] = family_returns(c)

    # Meridian structures, exactly as structure_2008_check.
    tb = tbill()
    E = index()
    parts = {}
    baseline_spec = [
        (0.25, "long_term_etf", "ma_bond_rotation_cs_b126", "QQQ_IEF_KMLM"),
        (0.15, "momentum", "dual_momentum_long_only", "SPY_QQQ_IWM_EFA_EEM_GLD_IEF_TLT"),
        (0.10, "sector_rotation", "relative_strength_b05", "SECTORS"),
        (0.075, "trend_following", "ma_trend_long_only", "SPY"),
        (0.05, "volatility", "vix_band", "SVXY_^VIX"),
    ]
    base = 0.375 * tb
    for w, fam, model, sym in baseline_spec:
        r, e = sleeve(fam, model, sym)
        parts[fam] = r + (1 - e) * tb
        base = base + w * parts[fam]
    spy, ief = hold("SPY"), hold("IEF")
    sixty = 0.6 * spy + 0.4 * ief

    def ew(syms):
        return pd.concat([hold(s).where(tr(s).reindex(E).notna()) for s in syms], axis=1).mean(axis=1).fillna(0)

    swap = {"sector_rotation": ew("XLK XLV XLF XLE XLI XLY XLP XLU XLRE XLB XLC".split()),
            "momentum": ew("SPY QQQ IWM EFA EEM GLD IEF TLT".split()), "trend_following": spy}
    wmap = {fam: w for w, fam, *_ in baseline_spec}
    R["BASELINE8"] = base
    R["TRIMMED"] = base - 0.10 * parts["sector_rotation"] - 0.05 * parts["volatility"] + 0.15 * tb
    R["C1"] = 0.5 * sixty + 0.25 * parts["long_term_etf"] + 0.25 * tb
    R["C3"] = base + sum(wmap[f] * (swap[f] - parts[f]) for f in swap)
    R["SIXTY40"] = sixty
    R["SPY"] = spy
    R["ALLWEATHER"] = (0.30 * spy + 0.40 * hold("TLT") + 0.15 * ief + 0.075 * hold("GLD") + 0.075 * hold("GSG"))
    for c in cfgs:
        if c["family"] == "blend":
            R[c["id"]] = sum(wt * R[k] for k, wt in c["parts"].items())
    R = {k: v.reindex(E).fillna(0.0) for k, v in R.items()}
    ids = [c["id"] for c in cfgs]
    missing = [i for i in ids if i not in R]
    if missing:
        print("MISSING CONFIGS:", missing)
        return 1

    # ---------------------------------------------------------------- gates
    g1 = R["TREND10-EW-U5"].loc["2007-05-01":"2026-09-25"]
    g2 = R["BASELINE8"].loc[A:B]
    gates = [("Faber TREND10-EW-U5", cagr(g1), maxdd(g1), sharpe(g1), 0.052, -0.150, 0.66),
             ("BASELINE8", cagr(g2), maxdd(g2), None, 0.071, -0.150, None)]
    ok = True
    for name, c_, d_, s_, ec, ed, es in gates:
        good = abs(c_ - ec) <= 0.0015 and abs(d_ - ed) <= 0.003 and (es is None or abs(s_ - es) <= 0.03)
        ok &= good
        print(f"GATE {name}: CAGR {c_:+.2%} (want {ec:+.1%}), DD {d_:+.2%} (want {ed:+.1%})"
              + (f", Sharpe {s_:.2f} (want {es:.2f})" if es else "") + (" PASS" if good else " FAIL"))
    if not ok:
        print("A gate failed: stopping before any result is read.")
        return 1

    tag = {c["id"]: c["tag"] for c in cfgs}
    fam = {c["id"]: c["family"] for c in cfgs}
    uni = {c["id"]: c.get("universe") for c in cfgs}
    deployable = [i for i in ids if tag[i] in ("live-ready", "small-build")]

    # ---------------------------------------------------------------- full sample
    rows = []
    for i in ids:
        r = R[i].loc[A:B]
        o, k = objective(r, tb.loc[A:B])
        rows.append({"id": i, "family": fam[i], "universe": uni[i], "tag": tag[i], "obj": o, "k": k,
                     "cagr": cagr(r), "maxdd": maxdd(r), "sharpe": sharpe(r)})
    full = pd.DataFrame(rows).set_index("id").sort_values("obj", ascending=False)
    full.to_csv(OUT / "results_full_sample.csv", float_format="%.5f")

    def show(df, title):
        print(f"\n### {title}\n| Candidate | Tag | At -15% | Exposure | CAGR | Max DD | Sharpe |\n|---|---|---:|---:|---:|---:|---:|")
        for i, x in df.iterrows():
            print(f"| {i} | {x['tag']} | {x['obj']:+.1%} | {x['k']:.2f} | {x['cagr']:+.1%} | {x['maxdd']:+.1%} | {x['sharpe']:.2f} |")

    show(full.head(15), "Full sample 2008-01..2026-06, top 15 by return at -15% (IN-SAMPLE)")
    fixed = ["BASELINE8", "TRIMMED", "C1", "C3", "SIXTY40", "ALLWEATHER", "SPY", "PAA-U12", "TREND10-EW-U5"]
    show(full.loc[fixed], "Reference rows (same window)")
    print(f"\nBest deployable (live-ready/small-build): {full.loc[deployable]["obj"].idxmax()} "
          f"{full.loc[deployable]["obj"].max():+.2%}; best overall: {full["obj"].idxmax()} {full["obj"].max():+.2%}")

    # ---------------------------------------------------------------- forward walk-forward
    wf_rows, stitched = [], {"WF-all": [], "WF-deployable": [], "BASELINE8": [], "C1": [], "SIXTY40": []}
    for Y in range(2012, 2027):
        ta, tb_ = TRAIN0, f"{Y - 1}-12-31"
        xa, xb = f"{Y}-01-01", min(f"{Y}-12-31", B)
        t_tr, t_te = tb.loc[ta:tb_], tb.loc[xa:xb]
        objs = {i: objective(R[i].loc[ta:tb_], t_tr) for i in ids}
        p_all = max(ids, key=lambda i: objs[i][0])
        p_dep = max(deployable, key=lambda i: objs[i][0])
        for key, pick in (("WF-all", p_all), ("WF-deployable", p_dep), ("BASELINE8", "BASELINE8"),
                          ("C1", "C1"), ("SIXTY40", "SIXTY40")):
            k = objs[pick][1]
            stitched[key].append(k * R[pick].loc[xa:xb] + (1 - k) * t_te)
        wf_rows.append({"year": Y, "pick_all": p_all, "k_all": objs[p_all][1],
                        "pick_deployable": p_dep, "k_dep": objs[p_dep][1]})
    wf = pd.DataFrame(wf_rows)
    wf.to_csv(OUT / "walk_forward_picks.csv", index=False, float_format="%.3f")
    print("\n### Forward walk-forward picks (trained on 2007-07..prior year)\n| Year | Pick (all) | k | Pick (deployable) | k |\n|---|---|---:|---|---:|")
    for _, x in wf.iterrows():
        print(f"| {x.year} | {x.pick_all} | {x.k_all:.2f} | {x.pick_deployable} | {x.k_dep:.2f} |")
    print("\n### Forward walk-forward, stitched out-of-sample 2012-01..2026-06\n| Track | CAGR | Max DD | Sharpe |\n|---|---:|---:|---:|")
    wf_stats = {}
    for key, pieces in stitched.items():
        s = pd.concat(pieces)
        wf_stats[key] = (cagr(s), maxdd(s), sharpe(s))
        lab_ = f"**{key}**" if key == "BASELINE8" else key
        print(f"| {lab_} | {cagr(s):+.1%} | {maxdd(s):+.1%} | {sharpe(s):.2f} |")
    # Diagnostic (added after the first, crashed run; changes no rule): where each track's worst drawdown was.
    for key, pieces in stitched.items():
        s = pd.concat(pieces)
        eq = (1 + s).cumprod()
        dd = eq / eq.cummax() - 1
        trough = dd.idxmin()
        peak = eq.loc[:trough].idxmax()
        held = wf.set_index("year").loc[trough.year, "pick_all" if key == "WF-all" else "pick_deployable"] \
            if key.startswith("WF") else key
        print(f"  worst drawdown {key}: {dd.min():+.1%} from {peak.date()} to {trough.date()} (holding {held})")

    # ---------------------------------------------------------------- reverse crash test
    ta, tb_ = "2011-01-01", B
    t_tr, t_te = tb.loc[ta:tb_], tb.loc[A:"2010-12-31"]
    objs = {i: objective(R[i].loc[ta:tb_], t_tr) for i in ids}
    r_all = max(ids, key=lambda i: objs[i][0])
    r_dep = max(deployable, key=lambda i: objs[i][0])
    print("\n### Reverse crash test: picked on 2011-01..2026-06 (k fitted there), run on 2008-01..2010-12")
    print("| Candidate | Why | k | CAGR 2008-10 | Max DD 2008-10 |\n|---|---|---:|---:|---:|")
    rev = {}
    for why, i in (("pick (all)", r_all), ("pick (deployable)", r_dep), ("reference", "BASELINE8"),
                   ("reference", "C1"), ("reference", "SIXTY40"), ("reference", "SPY")):
        k = objs[i][1]
        s = k * R[i].loc[A:"2010-12-31"] + (1 - k) * t_te
        rev[i] = (cagr(s), maxdd(s), k)
        lab_ = f"**{i}**" if i == "BASELINE8" else i
        print(f"| {lab_} | {why} | {k:.2f} | {cagr(s):+.1%} | {maxdd(s):+.1%} |")

    # ---------------------------------------------------------------- PBO and deflated Sharpe
    pbo = probability_of_backtest_overfitting({i: R[i].loc[TRAIN0:B] for i in ids}, n_groups=10, n_test_groups=5,
                                              purge_bars=21, embargo_bars=21)
    s_full = {i: R[i].loc[A:B] for i in ids}
    per_sr = [float(s_full[i].mean() / s_full[i].std()) for i in ids]
    best_sr = max(ids, key=lambda i: sharpe(s_full[i]))
    dsr96 = deflated_sharpe_ratio(s_full[best_sr].to_numpy(), trial_sharpes=per_sr, n_trials=96)
    dsr809 = deflated_sharpe_ratio(s_full[best_sr].to_numpy(), trial_sharpes=per_sr, n_trials=809)
    print(f"\nPBO (Sharpe-based, {pbo['n_valid_splits']} of {pbo['n_splits']} splits): {pbo['pbo']:.2f}  "
          "(0.5 = selection no better than chance; general gauge, not the drawdown-capped objective)")
    print(f"Sharpe-best candidate {best_sr} (Sharpe {sharpe(s_full[best_sr]):.2f}): deflated-Sharpe probability "
          f"{dsr96['dsr_pvalue']:.2f} with n=96, {dsr809['dsr_pvalue']:.2f} with n=809 (>= 0.95 = significant)")
    # Diagnostic (added after the first, crashed run; changes no rule): the same test on returns in
    # excess of T-bills. Raw returns mostly test "the portfolio went up", which any long-biased fund passes.
    t_full = tb.loc[A:B]
    per_sr_x = [float((s_full[i] - t_full).mean() / (s_full[i] - t_full).std()) for i in ids]
    best_x = max(ids, key=lambda i: sharpe(s_full[i] - t_full))
    dx96 = deflated_sharpe_ratio((s_full[best_x] - t_full).to_numpy(), trial_sharpes=per_sr_x, n_trials=96)
    dx809 = deflated_sharpe_ratio((s_full[best_x] - t_full).to_numpy(), trial_sharpes=per_sr_x, n_trials=809)
    print(f"Excess-of-T-bills: best {best_x} (Sharpe {sharpe(s_full[best_x] - t_full):.2f}): deflated-Sharpe "
          f"probability {dx96['dsr_pvalue']:.2f} with n=96, {dx809['dsr_pvalue']:.2f} with n=809")

    # ---------------------------------------------------------------- universe
    print("\n### Universe: same family and parameters, full-sample return at -15%\n| Comparison | Pairs | First wins | Median difference |\n|---|---:|---:|---:|")
    uni_share = {}
    for a_, b_ in (("U12", "U8"), ("U12", "U5"), ("U8", "U5")):
        diffs = []
        for i in ids:
            if uni[i] == a_ and fam[i] in ("hold", "trend", "tsmom", "dualmom"):
                j = i[: -len(a_)] + b_
                if j in full.index:
                    diffs.append(full.loc[i, "obj"] - full.loc[j, "obj"])
        d = np.array(diffs)
        uni_share[(a_, b_)] = (d > 0).mean()
        print(f"| {a_} vs {b_} | {len(d)} | {(d > 0).mean():.0%} | {np.median(d):+.2%} |")
    fam_best = full.groupby("family")["obj"].max().sort_values(ascending=False)
    print("\nBest per family (full sample, at -15%): " + ", ".join(f"{f} {v:+.1%}" for f, v in fam_best.items()))

    # ---------------------------------------------------------------- decision rules
    simp_class = {"hold": 0, "static": 0, "trend": 1, "tsmom": 1, "meridian": 2, "blend": 3}
    n_assets = {i: (len(U[uni[i]]) if uni[i] else {"SIXTY40": 2, "SPY": 1, "ALLWEATHER": 5, "C1": 3,
                                                        "TRIMMED": 6, "BASELINE8": 8, "C3": 8}.get(i, 14)) for i in ids}
    recent = Counter(wf[wf.year >= 2022].pick_deployable)
    best_dep = full.loc[deployable]["obj"].max()
    near = [i for i in deployable if full.loc[i, "obj"] >= best_dep - 0.005]
    near.sort(key=lambda i: (simp_class[fam[i]], n_assets[i], -recent.get(i, 0)))
    rec = near[0]
    rec_k = objective(R[rec].loc["2011-01-01":B], tb.loc["2011-01-01":B])[1]
    rec_crash = maxdd(rec_k * R[rec].loc[A:"2010-12-31"] + (1 - rec_k) * t_te)
    wf_all, wf_b = wf_stats["WF-all"], wf_stats["BASELINE8"]
    adds_value = (wf_all[0] - wf_b[0] > 0.005 and wf_all[1] >= wf_b[1] and rev[r_all][1] >= rev["BASELINE8"][1])
    wf_u12 = sum(1 for p in wf.pick_all if uni.get(p) == "U12")
    expand = uni_share[("U12", "U8")] >= 2 / 3 and (uni.get(r_all) == "U12" or wf_u12 > len(wf) / 2)
    print("\n### Decisions (plan rules)")
    print(f"Near-tie set (deployable within 0.5%/yr of {best_dep:+.2%}): {', '.join(near)}")
    print(f"RECOMMENDATION: {rec} (tag {tag[rec]}, {full.loc[rec, 'obj']:+.2%}/yr at -15% in-sample); "
          f"reverse-crash DD with k fitted on 2011-26 = {rec_crash:+.1%}"
          + ("  FLAG: deeper than -20%" if rec_crash < -0.20 else ""))
    print(f"Overall best (any tag): {full["obj"].idxmax()} ({tag[full["obj"].idxmax()]}) {full["obj"].max():+.2%}")
    print(f"SELECTION ADDS VALUE: {'YES' if adds_value else 'NO'} (WF-all {wf_all[0]:+.2%}/{wf_all[1]:+.1%} vs "
          f"BASELINE8 {wf_b[0]:+.2%}/{wf_b[1]:+.1%}; reverse pick DD {rev[r_all][1]:+.1%} vs BASELINE8 {rev['BASELINE8'][1]:+.1%})")
    print(f"EXPAND UNIVERSE TO U12: {'YES' if expand else 'NO'} (U12 beats U8 in {uni_share[('U12', 'U8')]:.0%} of pairs; "
          f"reverse pick {r_all} [{uni.get(r_all)}]; U12 forward picks {wf_u12}/{len(wf)})")

    summary = {"recommendation": rec, "near_tie": near, "best_overall": str(full["obj"].idxmax()),
               "adds_value": bool(adds_value), "expand_u12": bool(expand), "pbo": float(pbo["pbo"]),
               "dsr96": float(dsr96["dsr_pvalue"]), "dsr809": float(dsr809["dsr_pvalue"]), "reverse_pick": r_all,
               "reverse_pick_deployable": r_dep,
               "wf": {k: [float(x) for x in v] for k, v in wf_stats.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, separators=(",", ":")), encoding="utf-8")

    # Logged last, so a crash above can be fixed and rerun without spending the one final run.
    lab.run(plan, {"SPY": tr("SPY")}, lambda p, cfg: R[cfg["id"]],
            lambda p: {"SPY": spy, "IEF": ief}, tb, final=True)
    print("final run logged in research/plans/ledger.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
