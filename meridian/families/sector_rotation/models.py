"""Concrete sector-rotation models — single-asset proxy and cross-sectional.

This module owns sector rotation family concrete models; it does NOT own
portfolio construction or risk management (those stay in portfolio/).

Importing this module registers its models under the ``sector_rotation`` family.
"""

from __future__ import annotations

from meridian.families.base import (
    CrossSectionalModel,
    MovingAverageTrendModel,
    register_model,
)


@register_model("sector_rotation", "trend_rotation")
class TrendRotationModel(MovingAverageTrendModel):
    """Single-asset proxy: hold a sector while it trends above its 100-bar average."""

    window = 100


@register_model("sector_rotation", "relative_strength")
class RelativeSectorStrengthModel(CrossSectionalModel):
    """Cross-sectional, long-only: rotate into the strongest sectors by trailing return."""

    lookback = 120
    quantile = 0.34       # roughly the top third of sectors
    long_only = True
