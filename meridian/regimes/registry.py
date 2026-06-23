"""Regime-classifier registry (mirrors the estimator/deviation registries)."""

from __future__ import annotations

from typing import Callable, TypeVar

from meridian.regimes.base import BaseRegime

_REGISTRY: dict[str, type[BaseRegime]] = {}

T = TypeVar("T", bound=type[BaseRegime])


def register(name: str) -> Callable[[T], T]:
    """Class decorator registering a regime classifier under ``name``."""

    def deco(cls: T) -> T:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Regime name already registered: {name!r}")
        cls.name = key
        _REGISTRY[key] = cls
        return cls

    return deco


def get_regime(name: str) -> type[BaseRegime]:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown regime {name!r}. Known: {', '.join(list_regimes())}")
    return _REGISTRY[key]


def create(name: str, **kwargs) -> BaseRegime:
    """Instantiate a registered regime classifier by name."""
    return get_regime(name)(**kwargs)


def list_regimes() -> list[str]:
    return sorted(_REGISTRY)


def all_regimes() -> dict[str, type[BaseRegime]]:
    return dict(_REGISTRY)
