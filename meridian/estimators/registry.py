"""Estimator registry.

All estimators register here under a short name so experiments can be driven
from config (``estimator: ema``) and so a single test can sweep the whole
library. Registration happens as a side effect of importing each family module;
``meridian.estimators.__init__`` imports them all.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from meridian.estimators.base import BaseEstimator

_REGISTRY: dict[str, type[BaseEstimator]] = {}

T = TypeVar("T", bound=type[BaseEstimator])


def register(name: str) -> Callable[[T], T]:
    """Class decorator registering an estimator under ``name``."""

    def deco(cls: T) -> T:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Estimator name already registered: {name!r}")
        cls.name = key
        _REGISTRY[key] = cls
        return cls

    return deco


def get_estimator(name: str) -> type[BaseEstimator]:
    """Return the estimator class registered under ``name``."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown estimator {name!r}. Known: {', '.join(list_estimators())}")
    return _REGISTRY[key]


def create(name: str, **kwargs) -> BaseEstimator:
    """Instantiate a registered estimator by name."""
    return get_estimator(name)(**kwargs)


def list_estimators() -> list[str]:
    """Sorted list of all registered estimator names."""
    return sorted(_REGISTRY)


def all_estimators() -> dict[str, type[BaseEstimator]]:
    """A copy of the full {name: class} registry."""
    return dict(_REGISTRY)
