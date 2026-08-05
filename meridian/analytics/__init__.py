"""Performance metrics and report generation.

    from meridian.analytics import performance_metrics, build_validation_report
"""

from meridian.analytics.capacity import (
    average_daily_volume,
    capacity_stress_sweep,
    cost_stress_conclusion_stable,
    cost_stress_sweep,
    cost_stress_sweep_cross_sectional,
    implied_shares_traded,
    max_capacity,
    participation_rate,
)
from meridian.analytics.metrics import (
    drawdown_series,
    equity_curve,
    max_drawdown,
    max_drawdown_duration,
    performance_metrics,
    sharpe_ratio,
)
from meridian.analytics.report import build_validation_report, write_report

__all__ = [
    "performance_metrics",
    "equity_curve",
    "drawdown_series",
    "max_drawdown",
    "max_drawdown_duration",
    "sharpe_ratio",
    "build_validation_report",
    "write_report",
    "cost_stress_sweep",
    "cost_stress_sweep_cross_sectional",
    "cost_stress_conclusion_stable",
    "average_daily_volume",
    "implied_shares_traded",
    "participation_rate",
    "max_capacity",
    "capacity_stress_sweep",
]
