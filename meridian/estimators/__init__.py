"""Fair-value estimators (all implement the BaseEstimator contract).

Importing this package registers every estimator. Use the registry to look them
up by name:

    from meridian.estimators import create, list_estimators
    est = create("ema", window=20)
"""

# Importing each family module registers its estimators as a side effect.
from meridian.estimators import (  # noqa: F401  (side-effect imports)
    filters,
    moving_average,
    regression,
    robust,
    stochastic,
)
from meridian.estimators.base import BaseEstimator

# Ensembles import the registry (populated above) and register named ensembles.
from meridian.estimators.ensemble import (  # noqa: F401
    EnsembleEstimator,
    make_ensemble,
    register_ensemble,
)
from meridian.estimators.registry import (
    all_estimators,
    create,
    get_estimator,
    list_estimators,
    register,
)

__all__ = [
    "BaseEstimator",
    "register",
    "create",
    "get_estimator",
    "list_estimators",
    "all_estimators",
    "EnsembleEstimator",
    "make_ensemble",
    "register_ensemble",
]
