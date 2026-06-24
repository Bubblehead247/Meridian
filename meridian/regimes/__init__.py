"""Market regime classifiers, applied downstream of estimators.

Regime logic is kept separate from the estimator interface. Classifiers label
market state; `gate_positions` / `run_gated_backtest` apply those labels as a
filter on signals.

    from meridian.regimes import create, classify, run_gated_backtest
"""

# Importing the classifiers module registers all regimes as a side effect.
from meridian.regimes import classifiers  # noqa: F401
from meridian.regimes.base import UNKNOWN, BaseRegime, RollingRegime
from meridian.regimes.filter import classify, gate_positions, run_gated_backtest
from meridian.regimes.registry import (
    all_regimes,
    create,
    get_regime,
    list_regimes,
    register,
)

__all__ = [
    "UNKNOWN",
    "BaseRegime",
    "RollingRegime",
    "register",
    "create",
    "get_regime",
    "list_regimes",
    "all_regimes",
    "classify",
    "gate_positions",
    "run_gated_backtest",
]
