"""Performance metrics and report generation.

    from meridian.analytics import performance_metrics, build_validation_report
"""

from meridian.analytics.metrics import (
    drawdown_series,
    equity_curve,
    max_drawdown,
    max_drawdown_duration,
    performance_metrics,
)
from meridian.analytics.report import build_validation_report, write_report

__all__ = [
    "performance_metrics",
    "equity_curve",
    "drawdown_series",
    "max_drawdown",
    "max_drawdown_duration",
    "build_validation_report",
    "write_report",
]
