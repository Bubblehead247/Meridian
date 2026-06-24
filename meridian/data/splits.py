"""Train / out-of-sample / walk-forward date splitting.

Out-of-sample discipline is a core principle. The project fixes three windows:

- **In-sample** (2010-2019): where estimators and signals may be developed.
- **Out-of-sample** (2020-2022): held out for honest evaluation.
- **Walk-forward** (2023-present): rolling, forward-only validation.

This module is the single place those boundaries are defined and applied, so no
experiment can accidentally train on its own test data. The default boundaries
match CLAUDE.md; a config may override them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Default split boundaries from CLAUDE.md.
DEFAULT_IN_SAMPLE = ("2010-01-01", "2019-12-31")
DEFAULT_OUT_OF_SAMPLE = ("2020-01-01", "2022-12-31")
DEFAULT_WALK_FORWARD_START = "2023-01-01"


@dataclass(frozen=True)
class SplitSpec:
    """Boundaries for the three evaluation windows."""

    in_sample: tuple[str, str] = DEFAULT_IN_SAMPLE
    out_of_sample: tuple[str, str] = DEFAULT_OUT_OF_SAMPLE
    walk_forward_start: str = DEFAULT_WALK_FORWARD_START

    @classmethod
    def from_config(cls, data_cfg: dict) -> SplitSpec:
        """Build a SplitSpec from an experiment config's ``data`` section.

        Accepts the keys ``in_sample`` (``[start, end]``), ``out_of_sample``
        (``[start, end]``), and ``walk_forward_start``. Missing keys fall back
        to the project defaults.
        """
        def pair(key: str, default: tuple[str, str]) -> tuple[str, str]:
            v = data_cfg.get(key, default)
            return (str(v[0]), str(v[1]))

        return cls(
            in_sample=pair("in_sample", DEFAULT_IN_SAMPLE),
            out_of_sample=pair("out_of_sample", DEFAULT_OUT_OF_SAMPLE),
            walk_forward_start=str(data_cfg.get("walk_forward_start", DEFAULT_WALK_FORWARD_START)),
        )

    def validate(self) -> None:
        """Check the windows are ordered and non-overlapping.

        Raises:
            ValueError: If in-sample is not strictly before out-of-sample, or
                out-of-sample not strictly before the walk-forward start.
        """
        is_start, is_end = map(pd.Timestamp, self.in_sample)
        oos_start, oos_end = map(pd.Timestamp, self.out_of_sample)
        wf_start = pd.Timestamp(self.walk_forward_start)

        if not is_start <= is_end:
            raise ValueError("in_sample start must be <= end")
        if not oos_start <= oos_end:
            raise ValueError("out_of_sample start must be <= end")
        if not is_end < oos_start:
            raise ValueError("in_sample must end before out_of_sample begins")
        if not oos_end < wf_start:
            raise ValueError("out_of_sample must end before walk_forward begins")


def _between(obj, start: str | None, end: str | None):
    """Slice a Series/DataFrame by its DatetimeIndex, inclusive."""
    mask = pd.Series(True, index=obj.index)
    if start is not None:
        mask &= obj.index >= pd.Timestamp(start)
    if end is not None:
        mask &= obj.index <= pd.Timestamp(end)
    return obj[mask.to_numpy()]


def split(obj, spec: SplitSpec | None = None) -> dict[str, pd.Series | pd.DataFrame]:
    """Split a time-indexed Series or DataFrame into the three windows.

    Args:
        obj: Any object with a DatetimeIndex (price Series, OHLCV frame, ...).
        spec: Boundaries to use; project defaults if None.

    Returns:
        ``{"in_sample": ..., "out_of_sample": ..., "walk_forward": ...}`` with
        each value the slice of ``obj`` falling in that window.
    """
    spec = spec or SplitSpec()
    spec.validate()
    return {
        "in_sample": _between(obj, *spec.in_sample),
        "out_of_sample": _between(obj, *spec.out_of_sample),
        "walk_forward": _between(obj, spec.walk_forward_start, None),
    }
