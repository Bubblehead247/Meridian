"""Focused follow-up on the one significant cell: sector spreads, lsma, window 20."""
from __future__ import annotations

import warnings

import numpy as np

from meridian.data.loader import load_universe
from meridian.features import build_pairs, hedge_ratio
from meridian.portfolio import run_universe_backtest, validate_universe
from meridian.signals import SignalConfig
from meridian.validation import WalkForwardSpec

warnings.filterwarnings("ignore")

SIG = SignalConfig(entry_threshold=1.5)
PAIRS = [("XLK", "XLY"), ("XLE", "XLB"), ("XLF", "XLI"), ("XLU", "XLP"),
         ("SPY", "RSP"), ("QQQ", "SPY")]
legs = sorted({s for p in PAIRS for s in p})
bars = load_universe(legs, start="2015-01-01")
px = {s: b["adj_close"] for s, b in bars.items()}
sig, prc = build_pairs(px, PAIRS, lookback=60)
spec = WalkForwardSpec(mode="anchored", min_train=750, test_span=375, step=375)

betas = [hedge_ratio(np.log(px[a]), np.log(px[b]), lookback=60).mean() for a, b in PAIRS]
leg_mult = 1.0 + float(np.nanmean(np.abs(betas)))
print(f"leg multiplier (1+|beta|) = {leg_mult:.2f}")

# turnover for lsma w20
pr = run_universe_backtest(prc, "lsma", "zscore", SIG, window=20, signal_prices_by_symbol=sig)
to = pr.weights.fillna(0.0).diff().abs().sum(axis=1).mean()
print(f"lsma w20 mean turnover/bar: {to:.3f}")

print("\nlsma w20 cost sweep (per-leg bps, x leg_mult applied):")
for c in [0, 1, 2, 3, 5, 10]:
    df = validate_universe(prc, ["lsma"], "zscore", SIG, spec=spec, window=20,
                           cost_bps=c * leg_mult, n_boot=600, n_mc=600, block=20,
                           periods_per_year=252, signal_prices_by_symbol=sig)
    r = df.iloc[0]
    print(f"  {c:2d} bp/leg -> Sharpe {r['oos_sharpe']:+.3f}  "
          f"q={r['q_value']:.3f}  sig={r['significant']}")

# which pair drives it? run each pair alone at 1bp gross
print("\nper-pair lsma w20 OOS Sharpe (1 bp, isolated):")
for key in prc:
    one_p = {key: prc[key]}
    one_s = {key: sig[key]}
    try:
        df = validate_universe(one_p, ["lsma"], "zscore", SIG, spec=spec, window=20,
                               cost_bps=1.0, n_boot=400, n_mc=400, block=20,
                               periods_per_year=252, signal_prices_by_symbol=one_s)
        print(f"  {key:10s} Sharpe {df.iloc[0]['oos_sharpe']:+.3f}  "
              f"mc_p={df.iloc[0]['mc_pvalue']:.3f}")
    except Exception as e:
        print(f"  {key:10s} ERR {e}")
