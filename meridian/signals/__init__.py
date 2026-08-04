"""Entry/exit signal logic, held constant across estimators.

The signal rule and backtester here never vary across estimators or deviation
metrics — only the estimator and metric do. That is the basis of the fair
comparison at the centre of the project.

    from meridian.signals import run_backtest, sweep, SignalConfig
"""

from meridian.signals.backtest import (
    BacktestResult,
    backtest,
    compute_scores,
    run_backtest,
    sweep,
)
from meridian.signals.engine import SignalConfig, SignalState, generate_positions
from meridian.signals.stop_diagnostics import flag_intrabar_stop_breaches

__all__ = [
    "SignalConfig",
    "SignalState",
    "generate_positions",
    "BacktestResult",
    "backtest",
    "compute_scores",
    "run_backtest",
    "sweep",
    "flag_intrabar_stop_breaches",
]
