"""Research pass 3 — other-asset-class mean-reversion study.

Runs three asset classes through the existing validation/cost stack:
  A. Crypto (24/7) — absolute + cross-sectional, daily bars.
  B. Cointegration ETF pairs — synthetic-spread instruments.
  C. Sector / calendar spreads — same machinery, different legs.

Reproducible from data: estimators/deviation/signal/portfolio/WFO/bootstrap/MC/
BH are all reused; only the loaders and the spread transform are new.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from meridian.data import crypto_bars_per_year, load_crypto_universe
from meridian.data.loader import load_universe
from meridian.features import build_pairs, cross_sectional_demean
from meridian.portfolio import run_universe_backtest, validate_universe
from meridian.signals import SignalConfig
from meridian.validation import WalkForwardSpec

warnings.filterwarnings("ignore")

SIG = SignalConfig(entry_threshold=1.5)
ESTIMATORS = ["sma", "ema", "ou", "lsma", "kalman"]


def _fmt(df: pd.DataFrame) -> str:
    cols = ["estimator", "oos_sharpe", "oos_return", "mc_pvalue",
            "q_value", "significant", "n_symbols"]
    d = df[cols].copy()
    for c in ["oos_sharpe", "oos_return", "mc_pvalue", "q_value"]:
        d[c] = d[c].round(3)
    return d.to_string(index=False)


def _turnover(prices, signal_prices=None, window=10, est="sma", **kw):
    pr = run_universe_backtest(prices, est, "zscore", SIG, window=window,
                               signal_prices_by_symbol=signal_prices, **kw)
    w = pr.weights.fillna(0.0)
    return float(w.diff().abs().sum(axis=1).mean())


def _cost_sweep(prices, signal_prices, window, ppy, costs, est="sma", leg_mult=1.0, **kw):
    spec = WalkForwardSpec(mode="anchored", min_train=kw.pop("min_train", 500),
                           test_span=kw.pop("test_span", 250), step=kw.pop("step", 250))
    out = {}
    for c in costs:
        df = validate_universe(prices, [est], "zscore", SIG, spec=spec, window=window,
                               cost_bps=c * leg_mult, n_boot=400, n_mc=400, block=20,
                               periods_per_year=ppy, signal_prices_by_symbol=signal_prices, **kw)
        out[c] = float(df.iloc[0]["oos_sharpe"])
    return out


# ====================================================================== A. CRYPTO
def crypto_study():
    print("\n" + "=" * 70 + "\nA. CRYPTO (24/7, daily)\n" + "=" * 70)
    syms = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "ADA_USDT", "DOGE_USDT",
            "LTC_USDT", "BCH_USDT", "LINK_USDT", "AVAX_USDT", "DOT_USDT", "ATOM_USDT"]
    px = load_crypto_universe(syms, "1D", start="2021-01-01")
    px = {k: v for k, v in px.items() if v.dropna().shape[0] > 600}
    n = min(len(v) for v in px.values()), max(len(v) for v in px.values())
    print(f"loaded {len(px)} instruments, bars per name: {n[0]}-{n[1]}")
    ppy = crypto_bars_per_year("1D")
    spec = WalkForwardSpec(mode="anchored", min_train=500, test_span=250, step=250)
    rel = cross_sectional_demean(px)

    for window in (5, 10, 20):
        print(f"\n--- window {window} ---")
        ab = validate_universe(px, ESTIMATORS, "zscore", SIG, spec=spec, window=window,
                               cost_bps=1.0, n_boot=400, n_mc=400, block=20, periods_per_year=ppy)
        print("ABSOLUTE (1 bp):\n" + _fmt(ab))
        cs = validate_universe(px, ESTIMATORS, "zscore", SIG, spec=spec, window=window,
                               cost_bps=1.0, n_boot=400, n_mc=400, block=20, periods_per_year=ppy,
                               signal_prices_by_symbol=rel)
        print("CROSS-SECTIONAL (1 bp):\n" + _fmt(cs))

    # cost sweep on the strongest gross cell: cross-sectional ou w20 (lowest
    # turnover / highest gross Sharpe of the crypto cells — the best shot at costs)
    print("\n--- cost sweep: cross-sectional ou w20 (best gross cell) ---")
    to = _turnover(px, rel, window=20, est="ou")
    print(f"mean daily turnover: {to:.3f}")
    sweep = _cost_sweep(px, rel, 20, ppy, [1, 5, 10, 20, 30, 50], est="ou")
    print("cost_bps -> OOS Sharpe:", {k: round(v, 2) for k, v in sweep.items()})


# ====================================================================== B/C. PAIRS
def pairs_study(title, pairs, lookback=60):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    legs = sorted({s for p in pairs for s in p})
    bars = load_universe(legs, start="2015-01-01")
    px = {s: b["adj_close"] for s, b in bars.items()}
    sig, prc = build_pairs(px, pairs, lookback=lookback)
    print(f"built {len(prc)} pairs from {len(px)} legs")

    # cointegration pre-check (Engle-Granger)
    from statsmodels.tsa.stattools import coint
    print("\ncointegration (Engle-Granger p-value, lower=more cointegrated):")
    betas = {}
    for a, b in pairs:
        if a not in px or b not in px:
            continue
        j = pd.concat([np.log(px[a]), np.log(px[b])], axis=1).dropna()
        p = coint(j.iloc[:, 0], j.iloc[:, 1])[1]
        from meridian.features import hedge_ratio
        bmean = hedge_ratio(np.log(px[a]), np.log(px[b]), lookback=lookback).mean()
        betas[f"{a}/{b}"] = bmean
        print(f"  {a}/{b:8s} coint p={p:.4f}  mean beta={bmean:.2f}")

    spec = WalkForwardSpec(mode="anchored", min_train=750, test_span=375, step=375)
    for window in (10, 20, 40):
        df = validate_universe(prc, ESTIMATORS, "zscore", SIG, spec=spec, window=window,
                               cost_bps=1.0, n_boot=400, n_mc=400, block=20,
                               periods_per_year=252, signal_prices_by_symbol=sig)
        print(f"\n--- window {window} (1 bp) ---\n" + _fmt(df))

    # leg-adjusted cost sweep on best window's sma
    avg_beta = np.nanmean(list(betas.values()))
    leg_mult = 1.0 + abs(avg_beta)
    print(f"\n--- cost sweep: sma w20, leg multiplier (1+|beta|)={leg_mult:.2f} ---")
    to = _turnover(prc, sig, window=20)
    print(f"mean daily turnover (synthetic): {to:.3f}")
    sweep = _cost_sweep(prc, sig, 20, 252, [1, 2, 5, 10, 20], leg_mult=leg_mult,
                        min_train=750, test_span=375, step=375)
    print("cost_bps(per leg) -> OOS Sharpe:", {k: round(v, 2) for k, v in sweep.items()})


if __name__ == "__main__":
    crypto_study()
    pairs_study(
        "B. COINTEGRATION ETF PAIRS (twins / substitutes)",
        [("IVV", "SPY"), ("VOO", "SPY"), ("GLD", "IAU"), ("QQQ", "QQQM"),
         ("IWM", "VTWO"), ("XLE", "VDE"), ("DIA", "SPY")],
    )
    pairs_study(
        "C. SECTOR / CALENDAR SPREADS",
        [("XLK", "XLY"), ("XLE", "XLB"), ("XLF", "XLI"), ("XLU", "XLP"),
         ("SPY", "RSP"), ("QQQ", "SPY")],
    )
