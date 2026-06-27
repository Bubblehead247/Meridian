"""Concrete long-term ETF models built on the long-term archetype.

This module owns long-term ETF family concrete models; it does NOT own
portfolio construction or risk management (those stay in portfolio/).

Importing this module registers its models under the ``long_term_etf`` family.
"""

from __future__ import annotations

from meridian.families.base import LongTermETFModel, register_model


@register_model("long_term_etf", "above_200ma")
class Above200MAModel(LongTermETFModel):
    """Hold the ETF while it is above its 200-day average, otherwise sit in cash (flat)."""

    window = 200


# TODO: StaticAllocationEtfModel / RiskParityEtfModel — multi-ETF weighting (needs a basket driver)
