"""Deviation metrics (z-score, ATR-normalized, etc.).

Importing this package registers every deviation metric. Look them up by name:

    from meridian.deviations import create, list_deviations
    dev = create("zscore", window=20)
"""

from meridian.deviations.base import BaseDeviation, RollingDeviation

# Importing the metrics module registers all metrics as a side effect.
from meridian.deviations import metrics  # noqa: F401
from meridian.deviations.registry import (
    all_deviations,
    create,
    get_deviation,
    list_deviations,
    register,
)

__all__ = [
    "BaseDeviation",
    "RollingDeviation",
    "register",
    "create",
    "get_deviation",
    "list_deviations",
    "all_deviations",
]
