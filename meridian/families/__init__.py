"""Strategy families package — multi-model layer wrapping estimators, deviations, and signals."""

from meridian.families.base import (
    BreakoutModel,
    CrossSectionalModel,
    EstimatorModel,
    LongTermETFModel,
    Model,
    MomentumModel,
    MovingAverageTrendModel,
    PullbackModel,
    StrategyFamily,
)

# Side-effect imports: loading each active family's models registers them in the registry
# (mirrors estimators/__init__), so the registry is populated on package import.
from meridian.families.breakouts import models as _breakouts_models  # noqa: F401
from meridian.families.long_term_etf import models as _long_term_etf_models  # noqa: F401
from meridian.families.mean_reversion import models as _mean_reversion_models  # noqa: F401
from meridian.families.momentum import models as _momentum_models  # noqa: F401
from meridian.families.permissions import (
    PERMISSIONS,
    cash_reserve_boost,
    gate_positions,
    is_permitted,
    permission_mask,
)
from meridian.families.pullback_continuation import models as _pullback_models  # noqa: F401
from meridian.families.registry import (
    all_models,
    create_model,
    get_model,
    list_families,
    list_models,
    register_model,
)
from meridian.families.sector_rotation import models as _sector_rotation_models  # noqa: F401
from meridian.families.trend_following import models as _trend_following_models  # noqa: F401

__all__ = [
    "PERMISSIONS",
    "is_permitted",
    "permission_mask",
    "gate_positions",
    "cash_reserve_boost",
    "Model",
    "EstimatorModel",
    "CrossSectionalModel",
    "MovingAverageTrendModel",
    "MomentumModel",
    "BreakoutModel",
    "PullbackModel",
    "LongTermETFModel",
    "StrategyFamily",
    "register_model",
    "get_model",
    "create_model",
    "list_models",
    "list_families",
    "all_models",
]
