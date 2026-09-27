"""Step 4 (POST-HOC — added after step 2's L4 showed sector_rotation, honest
momentum and SPY-trend each lose 2-5%/yr to holding their own instruments in
every sub-period). Replace those timing sleeves with an equal-weight hold of
the same instruments; keep long_term_etf timed (its L4 was mixed). Same
weights, L1 sweeps, 10 bps, total returns.

    PYTHONPATH=. python research/2026-09-returns-study/step3_posthoc_holds.py
"""
import sys
sys.path.insert(0, "research/2026-09-returns-study")
from common import SUB, hold, index, stats, table, tr  # noqa: E402
import pandas as pd  # noqa: E402
from meridian.portfolio.live_picks import live_pick_weight, load_live_picks  # noqa: E402


def ew(syms):
    return pd.concat([hold(s).where(tr(s).reindex(index()).notna()) for s in syms], axis=1).mean(axis=1).fillna(0)


df = pd.read_csv("research/2026-09-returns-study/step1_returns.csv", index_col=0, parse_dates=True)
picks = load_live_picks(); w = {f: live_pick_weight(f, set(picks)) for f in picks}
tb = df["tbill"]
base = df["fund_b"]
l1 = base + 0.05 * tb  # cash reserve sweep (vol-idle sweep adds ~0; see step 2)
swap = {"sector_rotation": ew("XLK XLV XLF XLE XLI XLY XLP XLU XLRE XLB XLC".split()),
        "momentum": ew("SPY QQQ IWM EFA EEM GLD IEF TLT".split()),
        "trend_following": hold("SPY")}
held = l1 + sum(w[f] * (swap[f] - df[f"b_{f}"]) for f in swap)
no_edge = w["breakouts"] + w["mean_reversion"] + w["pullback_continuation"]
spy, ief = hold("SPY"), hold("IEF")
rows = {"Baseline (honest fund B)": base,
        "Timing sleeves -> holds": held,
        "Timing sleeves -> holds, no-edge 32.5% -> 60/40": held - no_edge * tb + no_edge * (0.6 * spy + 0.4 * ief),
        "60/40 SPY/IEF": 0.6 * spy + 0.4 * ief}
print(table(rows, baseline="Baseline (honest fund B)"))
