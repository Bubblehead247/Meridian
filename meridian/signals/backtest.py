"""Backtester and the end-to-end comparative pipeline.

`backtest` turns a position path into a net-of-cost equity curve and trade list.
`run_backtest` is the shared, controlled pipeline that the whole project is built
to compare: price -> estimator -> deviation metric -> constant signal -> PnL. By
holding everything constant except the estimator and metric (both selected by
name), a sweep isolates the effect of the fair-value definition.

No look-ahead: the position decided from bar t's score is applied to the
t -> t+1 return (positions are lagged one bar). Costs are charged when the held
position changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from meridian.deviations import create as make_deviation
from meridian.estimators import create as make_estimator
from meridian.signals.engine import SignalConfig, generate_positions

# Trading days per year — for annualizing the basic sanity stats below.
PERIODS_PER_YEAR = 252


@dataclass
class BacktestResult:
    """Outcome of a single backtest.

    Heavy performance analytics (full risk metrics, attribution) belong to
    Phase 7; `summary()` here provides only the basic stats needed to sanity-
    check a run.
    """

    equity: pd.Series           # cumulative net-of-cost equity, starts at 1.0
    returns: pd.Series          # per-bar net strategy returns
    gross_returns: pd.Series    # per-bar returns before costs
    positions: pd.Series        # position actually held each bar (already lagged)
    trades: pd.DataFrame        # one row per closed/au open trade
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        net = self.returns
        n = len(net)
        total_return = float(self.equity.iloc[-1] - 1.0) if n else 0.0
        vol = float(net.std(ddof=0))
        sharpe = (
            float(net.mean() / vol * np.sqrt(PERIODS_PER_YEAR)) if vol > 0 else float("nan")
        )
        running_max = self.equity.cummax()
        max_dd = float((self.equity / running_max - 1.0).min()) if n else 0.0
        exposure = float((self.positions != 0).mean()) if n else 0.0
        wins = (self.trades["return"] > 0).sum() if len(self.trades) else 0
        win_rate = float(wins / len(self.trades)) if len(self.trades) else float("nan")
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "n_trades": int(len(self.trades)),
            "win_rate": win_rate,
            "exposure": exposure,
            **self.meta,
        }


def backtest(
    prices: pd.Series,
    positions: pd.Series,
    cost_bps: float = 1.0,
) -> BacktestResult:
    """Run a single-asset backtest of a position path against prices.

    Args:
        prices: Price series (use adjusted close).
        positions: Desired position at each bar's close, {-1, 0, +1}.
        cost_bps: Flat transaction cost in basis points of traded notional,
            charged on every change in held position. (Real slippage varies;
            this is the documented flat-bps approximation.)

    Returns:
        A populated :class:`BacktestResult`.
    """
    prices = prices.astype(float)
    ret = prices.pct_change().fillna(0.0)

    # Lag positions one bar: the position decided at t earns the t->t+1 return.
    held = positions.shift(1).fillna(0).astype(float)

    gross = held * ret
    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost

    equity = (1.0 + net).cumprod()
    trades = _extract_trades(prices, held, net)

    return BacktestResult(
        equity=equity,
        returns=net,
        gross_returns=gross,
        positions=held,
        trades=trades,
        meta={"cost_bps": cost_bps},
    )


def _extract_trades(prices: pd.Series, held: pd.Series, net: pd.Series) -> pd.DataFrame:
    """Build a trade ledger from a held-position path and net returns."""
    cols = ["direction", "entry_time", "exit_time", "bars_held",
            "entry_price", "exit_price", "return"]
    rows = []
    idx = held.index
    cur = 0
    start = None

    def close(side: int, s: int, e: int):
        seg = net.iloc[s:e]
        tr = float((1.0 + seg).prod() - 1.0)
        rows.append(
            {
                "direction": int(side),
                "entry_time": idx[s],
                "exit_time": idx[e] if e < len(idx) else idx[-1],
                "bars_held": int(e - s),
                "entry_price": float(prices.iloc[s]),
                "exit_price": float(prices.iloc[min(e, len(idx) - 1)]),
                "return": tr,
            }
        )

    vals = held.to_numpy()
    for i, p in enumerate(vals):
        p = int(p)
        if cur == 0 and p != 0:
            cur, start = p, i
        elif cur != 0 and p != cur:
            close(cur, start, i)
            cur, start = (p, i) if p != 0 else (0, None)
    if cur != 0 and start is not None:
        close(cur, start, len(vals))

    return pd.DataFrame(rows, columns=cols)


def compute_scores(
    prices: pd.Series,
    estimator,
    deviation,
    *,
    window: int = 20,
    bars: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute a causal deviation-score series for ``prices``.

    Walks the series one bar at a time, updating the estimator and deviation
    metric so every score uses only information available up to that bar — the
    same path a live system would see.

    Args:
        prices: Price series.
        estimator: Estimator name or instance.
        deviation: Deviation-metric name or instance.
        window: Lookback applied to both when given by name.
        bars: Optional OHLC frame aligned to ``prices`` (needed by ``atr_norm``).

    Returns:
        Score series aligned to ``prices`` (NaN during warmup).
    """
    est = make_estimator(estimator, window=window) if isinstance(estimator, str) else estimator
    dev = make_deviation(deviation, window=window) if isinstance(deviation, str) else deviation

    scores = np.full(len(prices), np.nan)
    for i, price in enumerate(prices.to_numpy(dtype=float)):
        if not np.isfinite(price):
            continue  # missing-data bar (e.g. before listing): no signal, no state update
        est.update(price)
        r = est.residual(price)
        bar = bars.iloc[i] if bars is not None else None
        dev.update(r, bar=bar)
        scores[i] = dev.value(r)
    return pd.Series(scores, index=prices.index, name="score")


def run_backtest(
    prices: pd.Series,
    estimator: str,
    deviation: str,
    signal: SignalConfig | None = None,
    *,
    window: int = 20,
    cost_bps: float = 1.0,
    bars: pd.DataFrame | None = None,
) -> BacktestResult:
    """End-to-end: price -> estimator -> deviation -> signal -> backtest.

    The single controlled pipeline of the project. Estimator and deviation are
    named; signal logic and costs are held constant.
    """
    scores = compute_scores(prices, estimator, deviation, window=window, bars=bars)
    positions = generate_positions(scores, signal)
    result = backtest(prices, positions, cost_bps=cost_bps)
    result.meta.update({"estimator": estimator, "deviation": deviation, "window": window})
    return result


def sweep(
    prices: pd.Series,
    estimators: list[str],
    deviations: list[str],
    signal: SignalConfig | None = None,
    *,
    window: int = 20,
    cost_bps: float = 1.0,
    bars: pd.DataFrame | None = None,
) -> dict[tuple[str, str], BacktestResult]:
    """Run the pipeline for every (estimator, deviation) pair.

    The comparative core: identical signal logic and costs across all pairs.
    Ranking and multiple-testing correction come in Phase 7.
    """
    results: dict[tuple[str, str], BacktestResult] = {}
    for e in estimators:
        for d in deviations:
            results[(e, d)] = run_backtest(
                prices, e, d, signal, window=window, cost_bps=cost_bps, bars=bars
            )
    return results
