"""Charts for comparing and presenting strategy results.

Matplotlib figures for the three views a comparative study needs: a Sharpe
heatmap across the estimator × deviation grid, overlaid out-of-sample equity
curves, and a drawdown plot. Functions return `Figure` objects so callers can
save or embed them; `save_fig` writes one to disk.

The Agg (headless) backend is selected so charts render on a server / in CI
without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; safe for servers and tests

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from meridian.analytics.metrics import drawdown_series, equity_curve  # noqa: E402


def sharpe_heatmap(
    table: pd.DataFrame,
    *,
    index: str = "estimator",
    columns: str = "deviation",
    value: str = "oos_sharpe",
    title: str = "Out-of-sample Sharpe by estimator × deviation",
):
    """Heatmap of a metric across two categorical axes.

    Args:
        table: Long-form DataFrame (e.g. a `WalkForwardResult.summary()` or a
            stacked `validate()` over multiple deviations).
        index/columns: Column names for the heatmap rows/columns.
        value: Metric column to color by.

    Returns:
        A matplotlib Figure.
    """
    pivot = table.pivot_table(index=index, columns=columns, values=value)
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * pivot.shape[1], 1 + 0.35 * pivot.shape[0]))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-np.nanmax(np.abs(data)),
                   vmax=np.nanmax(np.abs(data)))
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=value)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def equity_curves(returns_by_name: dict[str, pd.Series], *, title: str = "Out-of-sample equity"):
    """Overlay cumulative equity curves for several strategies."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, rets in returns_by_name.items():
        eq = equity_curve(rets)
        ax.plot(np.arange(len(eq)), eq.to_numpy(), label=name, linewidth=1.2)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("bar")
    ax.set_ylabel("equity (start = 1.0)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def drawdown_plot(returns, *, name: str = "strategy", title: str = "Drawdown"):
    """Filled underwater (drawdown) plot for one return series."""
    dd = drawdown_series(returns)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(dd))
    ax.fill_between(x, dd.to_numpy(), 0.0, color="firebrick", alpha=0.5)
    ax.set_xlabel("bar")
    ax.set_ylabel("drawdown")
    ax.set_title(f"{title} — {name}")
    fig.tight_layout()
    return fig


def save_fig(fig, path: str | Path, dpi: int = 120) -> Path:
    """Save a figure to disk (creates parent dirs) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
