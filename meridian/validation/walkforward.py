"""Walk-forward evaluation engine.

Walking forward means: select/measure on a train window, score on the
*immediately following* test window, then slide forward and repeat. Gluing the
disjoint, forward-only test windows together yields one continuous
out-of-sample (OOS) track record the model never trained on.

Two modes:
- **anchored** (default): train window starts at bar 0 and grows. Because the
  estimators are causal, the position path is identical in every fold, so it is
  computed **once** per (estimator, deviation) pair and then sliced into folds —
  cheap and exact.
- **rolling**: a fixed-length train window slides, so the warm-up start moves and
  the position path is recomputed per fold.

All 38 estimators are scored every fold (the comparison is the whole point); a
selection overlay on top picks the train-window winner each fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from meridian.signals import SignalConfig, backtest, compute_scores, generate_positions
from meridian.signals.backtest import PERIODS_PER_YEAR
from meridian.validation.stats import sharpe, total_return

Key = tuple[str, str]  # (estimator, deviation)


@dataclass(frozen=True)
class WalkForwardSpec:
    """Fold schedule (all spans in bars).

    Args:
        mode: "anchored" (expanding train) or "rolling" (fixed train).
        train_span: Train length in bars (rolling mode only).
        test_span: Test (OOS) length in bars per fold.
        step: Bars to advance between folds (= test_span gives tiling windows).
        min_train: Bars required before the first test window opens.
    """

    mode: str = "anchored"
    train_span: int = 756
    test_span: int = 126
    step: int = 126
    min_train: int = 756

    def __post_init__(self):
        if self.mode not in ("anchored", "rolling"):
            raise ValueError("mode must be 'anchored' or 'rolling'")
        for f in ("train_span", "test_span", "step", "min_train"):
            if getattr(self, f) < 1:
                raise ValueError(f"{f} must be >= 1")


@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int  # == test_start
    test_start: int
    test_end: int


def make_folds(n: int, spec: WalkForwardSpec) -> list[Fold]:
    """Build the fold schedule for a series of length ``n``."""
    folds: list[Fold] = []
    test_start = spec.min_train
    while test_start < n:
        test_end = min(test_start + spec.test_span, n)
        if test_end - test_start < 1:
            break
        train_start = 0 if spec.mode == "anchored" else max(0, test_start - spec.train_span)
        if test_start - train_start < 1:
            break
        folds.append(Fold(train_start, test_start, test_start, test_end))
        if test_end >= n:
            break
        test_start += spec.step
    return folds


@dataclass
class WalkForwardResult:
    """Per-fold and stitched out-of-sample results for every pair."""

    prices: pd.Series
    keys: list[Key]
    folds: list[Fold]
    # per fold: key -> {"test_returns": Series, "test_held": Series, "train_metric": float}
    fold_data: list[dict] = field(default_factory=list)
    periods_per_year: int = PERIODS_PER_YEAR

    # --- stitched OOS series ---------------------------------------------

    def stitched_returns(self, key: Key) -> pd.Series:
        """Concatenated OOS net returns across all folds for one pair."""
        parts = [fd[key]["test_returns"] for fd in self.fold_data]
        return pd.concat(parts) if parts else pd.Series(dtype=float)

    def stitched_positions(self, key: Key) -> pd.Series:
        """Concatenated OOS held positions across all folds for one pair."""
        parts = [fd[key]["test_held"] for fd in self.fold_data]
        return pd.concat(parts) if parts else pd.Series(dtype=float)

    def oos_market_returns(self) -> pd.Series:
        """Market (buy-and-hold) returns over the union of test windows."""
        ret = self.prices.pct_change().fillna(0.0)
        parts = [ret.iloc[f.test_start : f.test_end] for f in self.folds]
        return pd.concat(parts) if parts else pd.Series(dtype=float)

    # --- summaries --------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """One row per pair: stitched OOS Sharpe, return, fold win-rate, etc."""
        rows = []
        for key in self.keys:
            rets = self.stitched_returns(key)
            fold_rets = [total_return(fd[key]["test_returns"]) for fd in self.fold_data]
            rows.append(
                {
                    "estimator": key[0],
                    "deviation": key[1],
                    "oos_sharpe": sharpe(rets, self.periods_per_year),
                    "oos_return": total_return(rets),
                    "n_folds": len(self.folds),
                    "fold_win_rate": (
                        float(np.mean(np.array(fold_rets) > 0)) if fold_rets else float("nan")
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False).reset_index(drop=True)

    def selection(self, by: str = "train_metric") -> SelectionResult:
        """Walk-forward selection overlay.

        Each fold, pick the key with the best train-window metric and take *its*
        test-window returns. Stitching those gives the OOS curve of a
        "trade the recently-best estimator" meta-strategy — chosen using only
        train data, so no look-ahead.
        """
        chosen: list[Key] = []
        parts: list[pd.Series] = []
        for fd in self.fold_data:
            best = max(self.keys, key=lambda k: _nan_to_neg(fd[k][by]))
            chosen.append(best)
            parts.append(fd[best]["test_returns"])
        returns = pd.concat(parts) if parts else pd.Series(dtype=float)
        return SelectionResult(
            returns=returns, chosen=chosen, periods_per_year=self.periods_per_year
        )


@dataclass
class SelectionResult:
    returns: pd.Series
    chosen: list[Key]
    periods_per_year: int = PERIODS_PER_YEAR

    def summary(self) -> dict:
        return {
            "oos_sharpe": sharpe(self.returns, self.periods_per_year),
            "oos_return": total_return(self.returns),
            "n_switches": int(
                sum(a != b for a, b in zip(self.chosen, self.chosen[1:], strict=False))
            ),
        }


def _nan_to_neg(x: float) -> float:
    return -np.inf if (x is None or np.isnan(x)) else x


def walk_forward(
    prices: pd.Series,
    estimators: list[str],
    deviations: list[str],
    signal: SignalConfig | None = None,
    *,
    spec: WalkForwardSpec | None = None,
    window: int = 20,
    cost_bps: float = 1.0,
    bars: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Run walk-forward evaluation for every (estimator, deviation) pair."""
    spec = spec or WalkForwardSpec()
    n = len(prices)
    folds = make_folds(n, spec)
    keys: list[Key] = [(e, d) for e in estimators for d in deviations]

    # Anchored: causal positions are fold-independent, so compute the full
    # backtest once per pair and slice it. (net, held) are reused across folds.
    full: dict[Key, tuple[pd.Series, pd.Series]] = {}
    if spec.mode == "anchored":
        for key in keys:
            full[key] = _full_net_held(prices, key, signal, window, cost_bps, bars)

    fold_data: list[dict] = []
    for f in folds:
        test_idx = prices.index[f.test_start : f.test_end]
        train_idx = prices.index[f.train_start : f.train_end]
        per_key: dict = {}
        for key in keys:
            if spec.mode == "anchored":
                net, held = full[key]
            else:
                span = slice(f.train_start, f.test_end)
                net, held = _full_net_held(
                    prices.iloc[span], key, signal, window, cost_bps,
                    None if bars is None else bars.iloc[span],
                )
            per_key[key] = {
                "test_returns": net.reindex(test_idx),
                "test_held": held.reindex(test_idx),
                "train_metric": sharpe(net.reindex(train_idx)),
            }
        fold_data.append(per_key)

    return WalkForwardResult(prices, keys, folds, fold_data)


def _full_net_held(
    prices: pd.Series,
    key: Key,
    signal: SignalConfig | None,
    window: int,
    cost_bps: float,
    bars: pd.DataFrame | None,
) -> tuple[pd.Series, pd.Series]:
    """Causal pipeline over a price span -> (net returns, held positions)."""
    est, dev = key
    scores = compute_scores(prices, est, dev, window=window, bars=bars)
    positions = generate_positions(scores, signal)
    res = backtest(prices, positions, cost_bps=cost_bps, bars=bars)
    return res.returns, res.positions
