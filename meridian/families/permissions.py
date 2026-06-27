"""Regime-based trading-permission matrix for strategy families (PLAN.md §5).

This module owns the per-family rule mapping a 3-dimension RegimeLabel to allowed/
restricted and the gate that flattens a family's positions when restricted; it does NOT
own regime labeling (regimes/labeler.py) or position sizing (portfolio/).

Each family is permitted to trade only in the regimes its rule allows. Rules read the
day's ``(trend, volatility, breadth)`` labels; because every comparison is against a
concrete label, a warm-up/``unknown`` dimension makes a gated rule fall through to
restricted (conservative — no trading until the regime is known). "No-gate" families
(long-term ETF, sector rotation) are always permitted.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from meridian.regimes.labeler import RegimeLabel, attach_regimes

#: A permission rule: given the day's RegimeLabel, may this family trade?
PermissionRule = Callable[[RegimeLabel], bool]


# --- per-family rules (the §5 matrix) ----------------------------------------

def _trend_following(label: RegimeLabel) -> bool:
    return label.trend == "bull" and label.volatility != "extreme"


def _momentum(label: RegimeLabel) -> bool:
    return label.trend == "bull" and label.volatility != "extreme"


def _breakouts(label: RegimeLabel) -> bool:
    return label.trend == "bull" and label.breadth in ("expansion", "neutral")


def _pullback_continuation(label: RegimeLabel) -> bool:
    return label.trend == "bull" and label.breadth in ("expansion", "neutral")


def _mean_reversion(label: RegimeLabel) -> bool:
    return label.trend in ("neutral", "bear") and label.breadth in ("neutral", "contraction")


def _always(label: RegimeLabel) -> bool:        # sector rotation, long-term ETF, cash
    return True


def _deferred(label: RegimeLabel) -> bool:      # event-driven, volatility — stub families
    return False


#: family/sleeve name -> permission rule. Names match the families/ package dirs
#: (plus ``cash_reserve``, a portfolio sleeve, for matrix completeness).
PERMISSIONS: dict[str, PermissionRule] = {
    "trend_following": _trend_following,
    "momentum": _momentum,
    "breakouts": _breakouts,
    "pullback_continuation": _pullback_continuation,
    "mean_reversion": _mean_reversion,
    "sector_rotation": _always,
    "long_term_etf": _always,
    "cash_reserve": _always,
    "event_driven": _deferred,
    "volatility": _deferred,
}


def _rule(family: str) -> PermissionRule:
    try:
        return PERMISSIONS[family]
    except KeyError:
        raise KeyError(
            f"Unknown family {family!r}. Known: {', '.join(sorted(PERMISSIONS))}"
        ) from None


# --- public API --------------------------------------------------------------

def is_permitted(family: str, label: RegimeLabel) -> bool:
    """Whether ``family`` may trade under the day's regime ``label``."""
    return bool(_rule(family)(label))


def permission_mask(family: str, regime_frame: pd.DataFrame) -> pd.Series:
    """Boolean Series over a regime frame: True on bars where ``family`` may trade.

    ``regime_frame`` has columns ``trend``/``volatility``/``breadth`` (from
    `regimes.labeler.regime_frame`).
    """
    rule = _rule(family)
    allowed = [
        rule(RegimeLabel(t, v, b))
        for t, v, b in zip(
            regime_frame["trend"], regime_frame["volatility"], regime_frame["breadth"],
            strict=True,
        )
    ]
    return pd.Series(allowed, index=regime_frame.index, dtype=bool)


def gate_positions(
    positions: pd.Series, family: str, regime_frame: pd.DataFrame
) -> pd.Series:
    """Flatten a family's positions on every bar where its regime forbids trading.

    ``regime_frame`` is aligned to ``positions`` (reindexed via `attach_regimes`, so a
    date with no regime row is ``unknown`` → restricted for gated families). The
    backtester applies the usual one-bar lag downstream, so no look-ahead is introduced.
    """
    aligned = attach_regimes(positions.index, regime_frame)
    mask = permission_mask(family, aligned)
    return positions.where(mask.to_numpy(), other=0).astype(int)


def cash_reserve_boost(label: RegimeLabel) -> bool:
    """Whether the cash-reserve sleeve should increase its allocation this regime.

    Per §5, cash steps up when the market is Bear trend *and* VIX Extreme. This is an
    allocation signal (consumed by portfolio/allocation), not a position gate.
    """
    return label.trend == "bear" and label.volatility == "extreme"
