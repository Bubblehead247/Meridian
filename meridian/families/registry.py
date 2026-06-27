"""(family, model) registration and lookup for strategy families.

This module owns (family, model) registration/lookup; it does NOT own model
logic (that lives in each family's models.py).

Mirrors ``estimators/registry``: a concrete model registers under ``(family, name)`` via
the ``register_model`` decorator (applied as a side effect of importing the family's
``models`` module), so the pipeline can drive any model by name and a family can list its
own models.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

_REGISTRY: dict[tuple[str, str], type] = {}

T = TypeVar("T", bound=type)


def register_model(family: str, name: str) -> Callable[[T], T]:
    """Class decorator registering a Model under ``(family, name)``."""

    def deco(cls: T) -> T:
        key = (family.lower(), name.lower())
        if key in _REGISTRY:
            raise ValueError(f"Model already registered: {family}/{name}")
        cls.family = key[0]
        cls.name = key[1]
        _REGISTRY[key] = cls
        return cls

    return deco


def get_model(family: str, name: str) -> type:
    """Return the model class registered under ``(family, name)``."""
    key = (family.lower(), name.lower())
    if key not in _REGISTRY:
        known = ", ".join(f"{f}/{n}" for f, n in sorted(_REGISTRY))
        raise KeyError(f"Unknown model {family}/{name}. Known: {known}")
    return _REGISTRY[key]


def create_model(family: str, name: str, **kwargs):
    """Instantiate a registered model by ``(family, name)``."""
    return get_model(family, name)(**kwargs)


def list_models(family: str | None = None) -> list[str]:
    """Model names in ``family`` (or every ``family/name`` when family is None)."""
    if family is None:
        return sorted(f"{f}/{n}" for f, n in _REGISTRY)
    fam = family.lower()
    return sorted(n for f, n in _REGISTRY if f == fam)


def list_families() -> list[str]:
    """Families that have at least one registered model."""
    return sorted({f for f, _ in _REGISTRY})


def all_models() -> dict[tuple[str, str], type]:
    """A copy of the full ``{(family, name): class}`` registry."""
    return dict(_REGISTRY)
