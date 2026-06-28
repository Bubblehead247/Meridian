"""Fund equity chart rendered in the terminal using Unicode box-drawing characters.

Builds a weighted equity curve from per-sleeve return series, renders it as a
Unicode line chart in the terminal, and prints 1/3/5/10/20 year return labels.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from meridian.analytics.metrics import equity_curve

_LOOKBACKS = [("20yr", 5040), ("10yr", 2520), ("5yr", 1260), ("3yr", 756), ("1yr", 252)]
_CHART_HEIGHT = 12   # rows (excluding axes)


def _weighted_returns(returns_by_sleeve: dict[str, tuple[pd.Series, float]]) -> pd.Series:
    combined = None
    for returns, weight in returns_by_sleeve.values():
        contribution = returns * weight
        combined = contribution if combined is None else combined.add(contribution, fill_value=0.0)
    return combined if combined is not None else pd.Series(dtype=float)


def _period_return(eq: np.ndarray, lookback_bars: int) -> float | None:
    if len(eq) <= lookback_bars:
        return None
    start = eq[-(lookback_bars + 1)]
    return float(eq[-1] / start - 1.0) if start > 0 else None


def _fmt_dollar(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}k"
    return f"${v:.0f}"


def _render_chart(eq_vals: np.ndarray, dates, chart_w: int, height: int) -> list[str]:
    """Return chart_w × height grid of characters (top row = highest value)."""
    # Downsample to chart_w columns
    idx = np.linspace(0, len(eq_vals) - 1, chart_w).astype(int)
    sampled = eq_vals[idx]

    lo, hi = sampled.min(), sampled.max()
    if hi == lo:
        row_idx = np.full(chart_w, height // 2, dtype=int)
    else:
        row_idx = ((sampled - lo) / (hi - lo) * (height - 1)).round().astype(int)

    # Build blank grid
    grid = [[" "] * chart_w for _ in range(height)]

    # Draw curve — use slope characters between adjacent columns
    _SLOPES = {-1: "╲", 0: "─", 1: "╱"}
    for x, y in enumerate(row_idx):
        r = height - 1 - y
        if x == 0:
            grid[r][x] = "─"
        else:
            dy = row_idx[x] - row_idx[x - 1]
            ch = _SLOPES.get(np.clip(dy, -1, 1), "│")
            grid[r][x] = ch
            # Fill vertical gap when jump > 1
            if abs(dy) > 1:
                step = 1 if dy > 0 else -1
                for mid in range(row_idx[x - 1] + step, row_idx[x], step):
                    grid[height - 1 - mid][x] = "│"

    return ["".join(row) for row in grid]


def show_fund_equity_chart(
    returns_by_sleeve: dict[str, tuple[pd.Series, float]],
    *,
    symbol: str = "",
    equity: float = 100_000.0,
) -> None:
    """Render the fund equity chart in the terminal.

    Args:
        returns_by_sleeve: {family: (daily_returns, allocation_weight)}
        symbol: Ticker label shown in the title.
        equity: Starting account equity — scales the y-axis.
    """
    try:
        from rich.console import Console
        from rich.text import Text
        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    weighted = _weighted_returns(returns_by_sleeve)
    if weighted.empty:
        return

    eq_series = equity_curve(weighted) * 10_000.0
    eq_vals = eq_series.to_numpy()
    dates = eq_series.index

    term_w = shutil.get_terminal_size().columns
    y_label_w = 8          # width of y-axis label column
    chart_w = max(40, term_w - y_label_w - 2)

    rows = _render_chart(eq_vals, dates, chart_w, _CHART_HEIGHT)

    lo, hi = eq_vals.min(), eq_vals.max()
    y_ticks = np.linspace(hi, lo, _CHART_HEIGHT)

    title = f"Fund Equity — {symbol}  ($10k → {_fmt_dollar(eq_vals[-1])})"

    # --- print ---
    print()
    if use_rich:
        console.print(f"[bold white]{title}[/bold white]")
    else:
        print(title)
    print()

    for i, (row_str, tick) in enumerate(zip(rows, y_ticks)):
        label = f"{_fmt_dollar(tick):>7} "
        if use_rich:
            line = Text(label, style="dim")
            is_baseline = i == _CHART_HEIGHT - 1
            line.append("┤", style="dim")
            line.append(row_str, style="bold cyan" if not is_baseline else "dim")
            console.print(line, end="\n")
        else:
            print(f"{label}┤{row_str}")

    # X-axis
    x_axis = " " * y_label_w + "└" + "─" * chart_w
    print(x_axis)

    # X-axis year labels — sample ~6 evenly spaced dates
    if hasattr(dates, "year"):
        n_labels = 6
        label_idx = np.linspace(0, len(dates) - 1, n_labels).astype(int)
        label_positions = [int(i * chart_w / (len(dates) - 1)) for i in label_idx]
        year_labels = [str(dates[i].year) for i in label_idx]
        x_label_row = [" "] * chart_w
        for pos, yr in zip(label_positions, year_labels):
            for j, ch in enumerate(yr):
                if pos + j < chart_w:
                    x_label_row[pos + j] = ch
        print(" " * (y_label_w + 1) + "".join(x_label_row))

    # Period returns
    print()
    parts = []
    for label, bars in _LOOKBACKS:
        ret = _period_return(eq_vals, bars)
        if ret is not None:
            sign = "+" if ret >= 0 else ""
            parts.append(f"{label} {sign}{ret:.1%}")
    if parts:
        returns_line = "    ".join(parts)
        if use_rich:
            console.print(f"  [dim]{returns_line}[/dim]")
        else:
            print(f"  {returns_line}")
    print()
