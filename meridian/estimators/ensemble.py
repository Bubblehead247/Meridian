"""Ensemble / adaptive meta-model estimators.

An ensemble combines several estimators' fair-value estimates into one. Because
it implements the same `BaseEstimator` contract, an ensemble plugs into the
deviation metrics, signal engine, validation engine, and reports with no special
handling — the whole point of the shared interface.

Combiners:

- **mean / median / trimmed** — static, robust blends of the members' fair
  values. ``median`` of many estimators is a strong, outlier-resistant baseline.
- **weighted** — static, fixed member weights.
- **inverse_variance** — adaptive: weight each member by the inverse of its
  recent squared residual (a minimum-variance style blend; tighter trackers get
  more weight).
- **skill** — adaptive: weight each member by how well its recent deviations
  *predicted reversion* (price above a member's fair value followed by a fall,
  and vice-versa). This directly rewards mean-reversion predictive power — the
  project's core question, answered online.

Members are given by registry name and share one window; the ensemble creates
fresh member instances on every `fit`, so replay/online semantics stay clean.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from meridian.estimators._rolling import RollingEstimator
from meridian.estimators.registry import create as _create_estimator
from meridian.estimators.registry import register

_EPS = 1e-12
_STATIC = {"mean", "median", "trimmed", "weighted"}
_ADAPTIVE = {"inverse_variance", "skill"}


class EnsembleEstimator(RollingEstimator):
    """Combine member estimators' fair values into one adaptive estimate.

    Args:
        members: List of registered estimator names (share ``window``).
        window: Lookback for members and for the residual scale.
        combine: One of mean/median/trimmed/weighted/inverse_variance/skill.
        weights: Fixed weights for ``combine="weighted"`` (else equal).
        skill_window: Smoothing length for the adaptive ``skill`` reward
            (defaults to ``window``).
    """

    def __init__(
        self,
        members: list[str],
        window: int = 20,
        combine: str = "median",
        weights: list[float] | None = None,
        skill_window: int | None = None,
    ):
        if combine not in _STATIC | _ADAPTIVE:
            raise ValueError(f"unknown combine {combine!r}")
        if not members:
            raise ValueError("ensemble needs at least one member")
        self._member_names = list(members)
        self.combine = combine
        self._fixed_weights = None if weights is None else np.asarray(weights, dtype=float)
        self.skill_window = int(skill_window or window)
        super().__init__(window)

    # --- state ------------------------------------------------------------

    def _init_state(self) -> None:
        self.members = [_create_estimator(n, window=self.window) for n in self._member_names]
        k = len(self.members)
        self._res_sq = [deque(maxlen=self.window) for _ in range(k)]
        self._skill = np.zeros(k)
        self._skill_alpha = 2.0 / (self.skill_window + 1.0)
        self._prev_means: np.ndarray | None = None
        self._prev_price: float = float("nan")
        self._weights = np.full(k, 1.0 / k)

    # --- core -------------------------------------------------------------

    def _compute_mean(self) -> float:
        price = self._buf[-1]
        for m in self.members:
            m.update(price)
        means = np.array([m.predict_mean() for m in self.members], dtype=float)

        # Per-member current residual feeds the adaptive blends.
        resid = price - means
        for i, r in enumerate(resid):
            if math.isfinite(r):
                self._res_sq[i].append(r * r)

        # Reversion-skill reward: did last bar's deviation predict this move?
        # Uses the signed residual *magnitude* times the realized move, so a
        # member with large deviations that genuinely revert (price above its
        # fair value, then a fall) earns a large positive reward, while a fast
        # tracker with tiny residuals earns little — exactly the behavior a
        # mean-reversion meta-model wants.
        if self._prev_means is not None and math.isfinite(self._prev_price):
            realized = price - self._prev_price
            prev_resid = self._prev_price - self._prev_means
            reward = np.nan_to_num(-prev_resid * realized)  # +ve when reversion occurred
            self._skill = (1 - self._skill_alpha) * self._skill + self._skill_alpha * reward

        self._prev_means = means
        self._prev_price = price

        self._weights = self._compute_weights(means)
        return float(np.dot(self._weights, np.nan_to_num(means)))

    def _compute_weights(self, means: np.ndarray) -> np.ndarray:
        k = len(self.members)
        if self.combine == "mean":
            return np.full(k, 1.0 / k)
        if self.combine == "median":
            return _median_weights(means)
        if self.combine == "trimmed":
            return _trimmed_weights(means)
        if self.combine == "weighted":
            w = self._fixed_weights if self._fixed_weights is not None else np.ones(k)
            return _normalize(w)
        if self.combine == "inverse_variance":
            var = np.array([np.mean(rs) if rs else np.inf for rs in self._res_sq])
            return _normalize(1.0 / (var + _EPS))
        # skill
        w = np.clip(self._skill, 0.0, None)
        return _normalize(w) if w.sum() > 0 else np.full(k, 1.0 / k)

    def _extra_state(self) -> dict:
        return {
            "combine": self.combine,
            "members": list(self._member_names),
            "weights": self._weights.tolist(),
        }


# --- weight helpers -------------------------------------------------------

def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    total = w.sum()
    return w / total if total > 0 else np.full(len(w), 1.0 / len(w))


def _median_weights(means: np.ndarray) -> np.ndarray:
    """Weights that select the median element (averaging two for even counts)."""
    k = len(means)
    order = np.argsort(means)
    w = np.zeros(k)
    if k % 2 == 1:
        w[order[k // 2]] = 1.0
    else:
        w[order[k // 2 - 1]] = 0.5
        w[order[k // 2]] = 0.5
    return w


def _trimmed_weights(means: np.ndarray) -> np.ndarray:
    """Equal weights over the central values (drops top/bottom 20%)."""
    k = len(means)
    if k < 5:
        return np.full(k, 1.0 / k)
    cut = int(0.2 * k)
    order = np.argsort(means)
    keep = order[cut: k - cut]
    w = np.zeros(k)
    w[keep] = 1.0 / len(keep)
    return w


# --- convenience + registration ------------------------------------------

def make_ensemble(members: list[str], combine: str = "median", **kwargs) -> EnsembleEstimator:
    """Construct an EnsembleEstimator (window defaults to 20)."""
    return EnsembleEstimator(members=members, combine=combine, **kwargs)


def register_ensemble(name: str, members: list[str], combine: str = "median", **kwargs):
    """Register a named ensemble usable everywhere a single estimator name is.

    The registered class takes only ``window`` (like any estimator), binding the
    members and combiner, so ``create(name, window=...)`` and the whole
    validation/reporting pipeline work unchanged.
    """

    @register(name)
    class _Registered(EnsembleEstimator):  # noqa: N801
        def __init__(self, window: int = 20):
            super().__init__(members=members, window=window, combine=combine, **kwargs)

    _Registered.__name__ = f"Ensemble_{name}"
    _Registered.__qualname__ = _Registered.__name__
    return _Registered


# A diverse core of fast/slow, robust, filter, and reversion-aware estimators.
_CORE = ["sma", "ema", "kalman", "lsma", "median", "ou"]

register_ensemble("ens_mean", _CORE, "mean")
register_ensemble("ens_median", _CORE, "median")
register_ensemble("ens_invvar", _CORE, "inverse_variance")
register_ensemble("ens_skill", _CORE, "skill")
