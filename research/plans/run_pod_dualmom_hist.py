"""Run research/plans/pod_dualmom_hist.json once (informational; contaminated by selection).

Uses the SHADOW's own target function (meridian/execution/shadow_dualmom.py), so this also
gates the shadow: it must reproduce the sweep's DUALMOM6-K4-U8 (2008-01..2026-06: 11.0%, -14.6%).

    PYTHONPATH=. python research/plans/run_pod_dualmom_hist.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

from meridian.data import load_ohlcv  # noqa: E402
from meridian.execution import shadow_dualmom as sd  # noqa: E402
from meridian.validation import skill_lab as lab  # noqa: E402

PLAN = "research/plans/pod_dualmom_hist.json"
A, B = "2008-01-01", "2026-06-26"


def main() -> int:
    plan = lab.load_plan(PLAN)
    spy = load_ohlcv("SPY", "2006-01-01", use_cache=False)
    idx = spy.index
    px = pd.DataFrame({s: load_ohlcv(s, "2006-01-01", use_cache=False)["adj_close"] for s in sd.ETFS}) \
        .reindex(idx).ffill(limit=5)
    rf = (load_ohlcv("^IRX", "2006-01-01", use_cache=False)["close"].reindex(idx).ffill() / 100 / 252) \
        .shift(1).fillna(0.0)
    ends = sd.month_ends(idx)
    T = sd.targets(px.loc[ends], (1 + rf).cumprod().loc[ends])
    w = T.reindex(idx).ffill().shift(1).fillna(0.0)
    tc = T.diff().abs().sum(axis=1).reindex(idx).fillna(0.0).shift(1).fillna(0.0) * 10 / 1e4
    ret = px.pct_change()
    pod = (w * ret.fillna(0.0)).sum(axis=1) + (1 - w.sum(axis=1)) * rf - tc
    bench = ret.fillna(0.0).mean(axis=1)

    r = pod.loc[A:B]
    eq = (1 + r).cumprod()
    cagr, dd = eq.iloc[-1] ** (252 / len(r)) - 1, (eq / eq.cummax() - 1).min()
    ok = abs(cagr - 0.1098) <= 0.0015 and abs(dd - (-0.146)) <= 0.003
    print(f"GATE shadow target function over history: CAGR {cagr:+.2%}, max DD {dd:+.2%} "
          f"(sweep +10.98%, -14.6%) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        return 1

    rep = lab.run(plan, {"SPY": spy["adj_close"]}, lambda p, cfg: pod,
                  lambda p: {"EW-hold-U8": bench}, rf, final=True)
    al = rep["best"]["alpha"]
    b = bench.loc[A:B]
    beq = (1 + b).cumprod()
    print(f"window {rep['window'][0]}..{rep['window'][1]} ({rep['years']:.1f} years); "
          f"detectable IR at t=2: {rep['mde_information_ratio']:.2f}")
    print(f"alpha over EW hold of the same 8: {al['alpha']:+.2%}/yr, NW t {al['t_alpha']:+.2f}, "
          f"beta {al['betas']['EW-hold-U8']:.2f}, IR {al['information_ratio']:+.2f}")
    print(f"pod {cagr:+.1%} / {dd:+.1%}; EW hold of the same 8: "
          f"{beq.iloc[-1] ** (252 / len(b)) - 1:+.1%} / {(beq / beq.cummax() - 1).min():+.1%}")
    print("CONTAMINATED: parameters were the best of 96 on this data; not a funding gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
