"""Deviation-metric registry (mirrors the estimator registry)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from meridian.deviations.base import BaseDeviation

_REGISTRY: dict[str, type[BaseDeviation]] = {}

T = TypeVar("T", bound=type[BaseDeviation])


def register(name: str) -> Callable[[T], T]:
    """Class decorator registering a deviation metric under ``name``."""

    def deco(cls: T) -> T:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Deviation name already registered: {name!r}")
        cls.name = key
        _REGISTRY[key] = cls
        return cls

    return deco


def get_deviation(name: str) -> type[BaseDeviation]:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown deviation {name!r}. Known: {', '.join(list_deviations())}")
    return _REGISTRY[key]


def create(name: str, **kwargs) -> BaseDeviation:
    """Instantiate a registered deviation metric by name."""
    return get_deviation(name)(**kwargs)


def list_deviations() -> list[str]:
    return sorted(_REGISTRY)


def all_deviations() -> dict[str, type[BaseDeviation]]:
    return dict(_REGISTRY)
