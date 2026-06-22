"""YAML config loading.

Reproducibility is a core principle: every experiment must be fully
reproducible from its config file alone. This module is the single place that
reads YAML configs, so all entry points load configs the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` file.

    Returns:
        The parsed config as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or does not parse to a mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {path}")

    return data
