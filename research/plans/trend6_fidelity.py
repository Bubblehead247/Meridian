"""Gate: the live trend 6 models, backtested one per ETF at 12.5% with idle cash in
T-bills, must reproduce the sweep's TREND6-EW-U8 (2008-01..2026-06: 7.0%/yr, max DD -9.6%).

    PYTHONPATH=. python research/plans/trend6_fidelity.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

from meridian.data import load_ohlcv  # noqa: E402
from meridian.families import create_model  # noqa: E402
from meridian.families.core.models import TREND6_ETFS  # noqa: E402

A, B = "2008-01-01", "2026-06-26"
spy = load_ohlcv("SPY", "2006-01-01", use_cache=False)
idx = spy.index
rf = (load_ohlcv("^IRX", "2006-01-01", use_cache=False)["close"].reindex(idx).ffill() / 100 / 252).shift(1).fillna(0)
fund = pd.Series(0.0, index=idx)
for etf in TREND6_ETFS:
    px = load_ohlcv(etf, "2006-01-01", use_cache=False)["adj_close"].reindex(idx).ffill(limit=5)
    bt = create_model(f"core_trend_{etf.lower()}", "sma6_monthly").backtest(px, cost_bps=10.0)
    r = bt.returns.reindex(idx).fillna(0.0)
    held = bt.positions.reindex(idx).fillna(0.0).abs().clip(upper=1)
    fund += 0.125 * (r + (1 - held) * rf)
r = fund.loc[A:B]
eq = (1 + r).cumprod()
cagr, dd = eq.iloc[-1] ** (252 / len(r)) - 1, (eq / eq.cummax() - 1).min()
ok = abs(cagr - 0.0702) <= 0.0015 and abs(dd - (-0.0964)) <= 0.003
print(f"trend 6 via live models: CAGR {cagr:+.2%}, max DD {dd:+.2%} (sweep: +7.02%, -9.64%) -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
