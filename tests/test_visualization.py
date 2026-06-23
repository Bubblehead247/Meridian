"""Phase 7 tests: charts render and save without a display (Agg backend)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from meridian.visualization import drawdown_plot, equity_curves, save_fig, sharpe_heatmap


def _returns(seed=0, n=200) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.01, n))


def test_sharpe_heatmap_returns_figure():
    table = pd.DataFrame(
        {
            "estimator": ["sma", "sma", "ema", "ema"],
            "deviation": ["zscore", "mad_z", "zscore", "mad_z"],
            "oos_sharpe": [0.4, 0.2, -0.1, 0.3],
        }
    )
    fig = sharpe_heatmap(table)
    assert isinstance(fig, Figure)


def test_equity_curves_returns_figure():
    fig = equity_curves({"sma": _returns(1), "ema": _returns(2)})
    assert isinstance(fig, Figure)


def test_drawdown_plot_returns_figure():
    fig = drawdown_plot(_returns(3), name="sma")
    assert isinstance(fig, Figure)


def test_save_fig_writes_png(tmp_path):
    fig = equity_curves({"sma": _returns(4)})
    p = save_fig(fig, tmp_path / "charts" / "equity.png")
    assert p.exists()
    assert p.stat().st_size > 0
