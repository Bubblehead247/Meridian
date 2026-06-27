"""Pre-registered confirmation of the sector-spread mean-reversion candidate.

Research pass 3 found the project's only cost-surviving, BH-significant signal: a
diversified basket of sector/calendar ETF spreads (lsma, window 20). It was flagged a
*candidate, not an edge* — one cell among many, hand-picked pairs, few folds, optimistic
cost. This runner CONFIRMS or KILLS it under a frozen, pre-registered protocol.

================================ PRE-REGISTRATION =============================
Everything below is fixed BEFORE looking at any out-of-sample result.

  Strategy (frozen from discovery, NOT re-tuned):
      estimator = lsma,  deviation = zscore,  window = 20,
      signal    = SignalConfig(entry_threshold=1.5),  equal-weight long/short.

  Windows:
      Selection (in-sample) : dates <= 2019-12-31  -- selection sees ONLY this.
      Confirmation (OOS)    : dates >= 2020-01-01  -- the held-out test.

  Candidate ETFs (fixed list, full pre-2020 history):
      sector SPDRs  XLB XLE XLF XLI XLK XLP XLU XLV XLY
      index ETFs    SPY RSP QQQ DIA IWM

  Pair rule (mechanical, no hand-picking):
      form ALL unordered candidate pairs, keep those passing an Engle-Granger
      cointegration screen (p < 0.05) computed on the IN-SAMPLE slice only.

  Realistic cost: 2 bps per leg, charged on BOTH legs via true leg-level turnover.

  PASS CRITERION (CONFIRM iff both hold on the net-of-cost OOS series):
      (1) bootstrap 95% Sharpe CI lower bound > 0, AND
      (2) Monte-Carlo timing test clears study-wide Bonferroni significance (q < 0.05).
      The verdict runs the SINGLE frozen config -> m = 1 test (Bonferroni is a no-op,
      stated for discipline); extra windows below are non-gating sensitivity only.
==============================================================================
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from meridian.data.loader import load_universe
from meridian.features import build_pairs, hedge_ratio, leg_cost, screen_cointegrated_pairs
from meridian.portfolio import run_universe_backtest
from meridian.portfolio.validation import _portfolio_mc_pvalue
from meridian.signals import SignalConfig
from meridian.validation.bootstrap import block_bootstrap_sharpe
from meridian.validation.correction import correct
from meridian.validation.stats import sharpe, total_return

warnings.filterwarnings("ignore")

# ---- frozen pre-registration constants --------------------------------------
ESTIMATOR = "lsma"
DEVIATION = "zscore"
WINDOW = 20
SIG = SignalConfig(entry_threshold=1.5)
LOOKBACK = 60
IS_END = "2019-12-31"
OOS_START = "2020-01-01"
HALF_SPREAD_BPS = 2.0
PPY = 252
CANDIDATES = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
              "SPY", "RSP", "QQQ", "DIA", "IWM"]


def _betas(px: dict[str, pd.Series], pairs: list[tuple[str, str]]) -> dict[str, pd.Series]:
    """Lagged hedge ratio per pair key 'A/B' (for the honest leg-level cost)."""
    return {f"{a}/{b}": hedge_ratio(np.log(px[a]), np.log(px[b]), lookback=LOOKBACK)
            for a, b in pairs}


def _net_oos(prc, sig, betas, *, half_spread_bps=HALF_SPREAD_BPS, window=WINDOW):
    """Gross basket at zero cost, net out per-leg cost, slice to the OOS window.

    Returns (net_oos, weights_oos, returns_by_symbol_oos) — the held weights and
    per-symbol gross returns are needed for the portfolio Monte-Carlo.
    """
    pr = run_universe_backtest(prc, ESTIMATOR, DEVIATION, SIG, window=window,
                               cost_bps=0.0, signal_prices_by_symbol=sig)
    cost = leg_cost(pr.weights.fillna(0.0), betas, half_spread_bps)
    net = pr.returns - cost
    oos = net.index >= pd.Timestamp(OOS_START)
    return net[oos], pr.weights[oos], pr.returns_by_symbol[oos]


def main():
    print(__doc__)

    # 1. load candidates -------------------------------------------------------
    bars = load_universe(CANDIDATES, start="2005-01-01")
    px = {s: b["adj_close"] for s, b in bars.items()}
    print(f"loaded {len(px)}/{len(CANDIDATES)} candidate ETFs")

    # 2. SELECT on the in-sample slice only ------------------------------------
    px_is = {s: v[v.index <= pd.Timestamp(IS_END)] for s, v in px.items()}
    pairs = screen_cointegrated_pairs(px_is, CANDIDATES, max_pvalue=0.05)
    print(f"\nin-sample cointegration screen (<= {IS_END}) selected {len(pairs)} pairs:")
    for a, b in pairs:
        j = pd.concat([np.log(px_is[a]), np.log(px_is[b])], axis=1).dropna()
        from statsmodels.tsa.stattools import coint
        p = coint(j.iloc[:, 0], j.iloc[:, 1])[1]
        beta = hedge_ratio(np.log(px_is[a]), np.log(px_is[b]), lookback=LOOKBACK).mean()
        print(f"  {a}/{b:5s}  IS coint p={p:.4f}  mean beta={beta:.2f}")
    if not pairs:
        print("\nNo cointegrated pairs selected in-sample -> nothing to confirm. KILL.")
        return

    # 3. BUILD the basket over full history ------------------------------------
    sig, prc = build_pairs(px, pairs, lookback=LOOKBACK)
    betas = _betas(px, pairs)

    # 4. CONFIRM on OOS at the pre-registered 2 bps/leg ------------------------
    net, w_oos, ret_oos = _net_oos(prc, sig, betas)
    boot = block_bootstrap_sharpe(net, n_boot=2000, block=20, periods_per_year=PPY, seed=0)
    mc_p = _portfolio_mc_pvalue(w_oos, ret_oos, n=2000, periods_per_year=PPY, seed=0)
    q = float(correct([mc_p], method="bonferroni")["adjusted"][0])  # m = 1

    print("\n" + "=" * 70 + "\nCONFIRMATION (OOS 2020+, lsma w20, 2 bps/leg)\n" + "=" * 70)
    print(f"  OOS bars            : {boot['n']}")
    print(f"  net OOS Sharpe      : {boot['point']:+.3f}")
    print(f"  bootstrap 95% CI    : [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]")
    print(f"  net OOS total return: {total_return(net):+.3f}")
    print(f"  MC p-value (timing) : {mc_p:.4f}   Bonferroni q (m=1): {q:.4f}")

    ci_ok = boot["ci_low"] > 0
    mc_ok = q < 0.05
    verdict = "CONFIRM" if (ci_ok and mc_ok) else "KILL"
    print(f"\n  criterion 1  CI low > 0      : {ci_ok}  ({boot['ci_low']:+.3f})")
    print(f"  criterion 2  Bonferroni q<.05: {mc_ok}  (q={q:.4f})")
    print(f"\n  >>> VERDICT: {verdict} <<<")

    # 5. disclosure: break-even cost + per-pair contribution + sensitivity -----
    print("\n--- break-even cost (net OOS Sharpe vs half-spread) ---")
    for hs in [0, 1, 2, 3, 5, 8, 10]:
        n_hs, _, _ = _net_oos(prc, sig, betas, half_spread_bps=hs)
        print(f"  {hs:2d} bp/leg -> net OOS Sharpe {sharpe(n_hs, PPY):+.3f}")

    print("\n--- per-pair isolated net OOS Sharpe (2 bps/leg) — single-pair fragility ---")
    for key in prc:
        one_p, one_s = {key: prc[key]}, {key: sig[key]}
        n1, _, _ = _net_oos(one_p, one_s, {key: betas[key]})
        print(f"  {key:10s} net OOS Sharpe {sharpe(n1, PPY):+.3f}")

    print("\n--- non-gating window sensitivity (net OOS Sharpe, 2 bps/leg) ---")
    for win in (10, 20, 40):
        nw, _, _ = _net_oos(prc, sig, betas, window=win)
        print(f"  window {win:2d} -> net OOS Sharpe {sharpe(nw, PPY):+.3f}")


if __name__ == "__main__":
    main()
