"""Base abstractions for strategy models and strategy families.

This module owns the Model abstraction (composing estimator + deviation + signal
into entry/exit rules) and the StrategyFamily grouping; it does NOT own the
estimator math (that stays in estimators/).

A ``Model`` turns a price series into a position path and can backtest itself by reusing
the shared ``signals.backtest`` pipeline — so every registered model inherits the whole
validation/scoring/graduation stack for free. ``EstimatorModel`` is the common concrete
base: it composes a named estimator + deviation + signal (the existing pipeline) and a
subclass just declares which ones. ``StrategyFamily`` groups a family's registered models.
``register_model`` is re-exported from the registry for concrete models to apply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from meridian.families.registry import (  # noqa: F401 (re-exported for concrete models)
    create_model,
    list_models,
    register_model,
)
from meridian.signals import SignalConfig, backtest, compute_scores, generate_positions
from meridian.signals.backtest import BacktestResult


class Model(ABC):
    """A named strategy model: prices -> position path, with a self-backtest."""

    family: str | None = None
    name: str | None = None
    cross_sectional: bool = False   # single-asset; see CrossSectionalModel for the multi-asset kind

    @abstractmethod
    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        """Causal position path ({-1, 0, +1}) for ``prices``."""

    def backtest(
        self,
        prices: pd.Series,
        *,
        cost_bps: float = 1.0,
        bars: pd.DataFrame | None = None,
        regime_frame: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Backtest this model via the shared pipeline (reuses ``signals.backtest``).

        When ``regime_frame`` is supplied and the model has a declared ``family``,
        positions are gated to zero on bars the family's permission rule forbids.
        """
        positions = self.signals(prices, bars=bars)
        if regime_frame is not None and self.family is not None:
            from meridian.families.permissions import gate_positions
            positions = gate_positions(positions, self.family, regime_frame)
        result = backtest(prices, positions, cost_bps=cost_bps, bars=bars)
        result.meta.update({"family": self.family, "model": self.name})
        return result


class EstimatorModel(Model):
    """Concrete base composing a named estimator + deviation + signal (the existing pipeline).

    A concrete model declares ``estimator``/``deviation``/``window`` (and optionally a
    default ``signal``); ``window`` and ``signal`` may be overridden per instance.
    """

    estimator: str = "sma"
    deviation: str = "zscore"
    window: int = 20
    signal: SignalConfig | None = None

    def __init__(self, *, window: int | None = None, signal: SignalConfig | None = None):
        if window is not None:
            self.window = window
        if signal is not None:
            self.signal = signal

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        scores = compute_scores(
            prices, self.estimator, self.deviation, window=self.window, bars=bars
        )
        return generate_positions(scores, self.signal)


class MovingAverageTrendModel(Model):
    """Directional trend: long above the moving average, short below it.

    The complement of mean reversion — it trades *with* the deviation from fair value.
    Causal (the MA at bar t uses prices through t; the backtester lags one bar).
    """

    window = 50

    def __init__(self, *, window: int | None = None):
        if window is not None:
            self.window = window

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        ma = prices.rolling(self.window).mean()
        return np.sign(prices - ma).where(ma.notna(), 0.0).fillna(0).astype(int)


class MomentumModel(Model):
    """Cross-sectional-free momentum: long after a positive ``window``-bar return, else short."""

    window = 60

    def __init__(self, *, window: int | None = None):
        if window is not None:
            self.window = window

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        roc = prices.pct_change(self.window)
        return np.sign(roc).where(roc.notna(), 0.0).fillna(0).astype(int)


class BreakoutModel(Model):
    """Donchian breakout: long on a new N-bar high, short on a new low, held until reversed."""

    window = 20

    def __init__(self, *, window: int | None = None):
        if window is not None:
            self.window = window

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        hi = prices.rolling(self.window).max()
        lo = prices.rolling(self.window).min()
        raw = pd.Series(np.nan, index=prices.index)
        raw[prices >= hi] = 1.0
        raw[prices <= lo] = -1.0
        return raw.ffill().fillna(0).astype(int)


class PullbackModel(Model):
    """Long-only buy-the-dip: an uptrend (fast MA > slow MA) with price dipped below the fast MA."""

    fast = 20
    slow = 50

    def __init__(self, *, fast: int | None = None, slow: int | None = None):
        if fast is not None:
            self.fast = fast
        if slow is not None:
            self.slow = slow

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        fast = prices.rolling(self.fast).mean()
        slow = prices.rolling(self.slow).mean()
        long = (fast > slow) & (prices < fast) & slow.notna()
        return long.astype(int)


class LongTermETFModel(Model):
    """Long-term trend filter: hold the ETF while above its long MA, else sit in cash."""

    window = 200

    def __init__(self, *, window: int | None = None):
        if window is not None:
            self.window = window

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        ma = prices.rolling(self.window).mean()
        return ((prices >= ma) & ma.notna()).astype(int)


class CrossSectionalModel:
    """A multi-asset model: ranks a ``{symbol: price}`` universe into long/short signals.

    Distinct from ``Model`` — it operates on a whole universe each bar (relative strength,
    dual momentum, sector rotation), which a single-series model cannot express. Its
    ``backtest`` reuses ``portfolio.backtest_portfolio`` (the cross-sectional engine), so it
    inherits the same P&L/turnover/cost accounting. Subclasses tune ``lookback`` / ``quantile``
    / ``long_only`` or override ``signals``.
    """

    family: str | None = None
    name: str | None = None
    cross_sectional: bool = True

    lookback: int = 60
    quantile: float = 0.3
    long_only: bool = False
    sizing: str = "equal_weight"
    buffer_pct: float = 0.0   # hysteresis buffer; 0 = no buffer (original behaviour)

    # Market-regime overlay: when trend_filter=True the filter symbol is stripped
    # from the tradeable universe and used only to gate signals (go to cash when
    # filter symbol is below its moving average).
    trend_filter: bool = False
    trend_filter_symbol: str = "SPY"
    trend_filter_window: int = 200

    def signals(self, prices_by_symbol: dict[str, pd.Series]) -> pd.DataFrame:
        """Cross-sectional {-1,0,+1} signal frame (date × symbol)."""
        from meridian.features.cross_sectional import (
            momentum_scores,
            rank_signals,
            rank_signals_buffered,
        )

        scores = momentum_scores(prices_by_symbol, self.lookback)
        if self.buffer_pct > 0.0:
            return rank_signals_buffered(
                scores,
                quantile=self.quantile,
                long_only=self.long_only,
                buffer_pct=self.buffer_pct,
            )
        return rank_signals(scores, quantile=self.quantile, long_only=self.long_only)

    def backtest(self, prices_by_symbol: dict[str, pd.Series], *, cost_bps: float = 1.0):
        """Backtest the ranked portfolio via the shared cross-sectional engine."""
        from meridian.portfolio import backtest_portfolio

        # Separate filter symbol from the tradeable universe
        tradeable = prices_by_symbol
        filter_prices = None
        if self.trend_filter and self.trend_filter_symbol in prices_by_symbol:
            filter_prices = prices_by_symbol[self.trend_filter_symbol]
            tradeable = {k: v for k, v in prices_by_symbol.items()
                         if k != self.trend_filter_symbol}

        signals = self.signals(tradeable)

        # Apply trend filter: zero all signals when filter symbol is below its MA
        if filter_prices is not None:
            fp = pd.Series(filter_prices).reindex(signals.index).ffill()
            ma = fp.rolling(self.trend_filter_window).mean()
            signals.loc[(fp < ma) | ma.isna()] = 0

        prices = pd.DataFrame(
            {s: pd.Series(p) for s, p in tradeable.items()}
        ).reindex(signals.index)
        result = backtest_portfolio(
            signals, prices, sizing=self.sizing, cost_bps=cost_bps, lookback=self.lookback
        )
        result.meta.update({"family": self.family, "model": self.name})
        return result


class StrategyFamily:
    """Grouping of one family's registered models (a thin view over the registry)."""

    def __init__(self, name: str):
        self.name = name.lower()

    def permitted(self, label) -> bool:
        """Whether this family may trade under a regime ``label`` (the §5 permission matrix)."""
        from meridian.families.permissions import is_permitted

        return is_permitted(self.name, label)

    def gate(self, positions: pd.Series, regime_frame: pd.DataFrame) -> pd.Series:
        """Flatten ``positions`` on bars this family is not permitted to trade."""
        from meridian.families.permissions import gate_positions

        return gate_positions(positions, self.name, regime_frame)

    def model_names(self) -> list[str]:
        """The registered model names in this family."""
        return list_models(self.name)

    def create(self, model_name: str, **kwargs) -> Model:
        """Instantiate one of this family's models by name."""
        return create_model(self.name, model_name, **kwargs)

    def create_all(self, **kwargs) -> dict[str, Model]:
        """Instantiate every model in this family, keyed by model name."""
        return {n: self.create(n, **kwargs) for n in self.model_names()}
