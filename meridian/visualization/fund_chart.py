"""Fund equity chart rendered in the terminal using Unicode box-drawing characters.

Builds a weighted equity curve from per-sleeve return series, renders it as a
Unicode line chart in the terminal, and prints period return labels.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from meridian.analytics.metrics import equity_curve
from meridian.portfolio.live_picks import live_pick_weight, load_live_picks

_LOOKBACKS = [("10yr", 2520), ("5yr", 1260), ("3yr", 756), ("1yr", 252), ("6mo", 126)]
_CHART_HEIGHT = 10
_MINI_HEIGHT  = 6

#: Box-drawing chars aren't in cp1252, so plain print() (no `rich`) crashes on the
#: default Windows console codepage. rich's own Console handles encoding itself,
#: so this translation is only needed on the non-rich fallback path.
_ASCII_SAFE = str.maketrans({"┤": "|", "└": "+", "─": "-", "╲": "\\", "╱": "/", "│": "|"})


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


def _render_chart(eq_vals: np.ndarray, chart_w: int, height: int) -> list[str]:
    idx     = np.linspace(0, len(eq_vals) - 1, chart_w).astype(int)
    sampled = eq_vals[idx]
    lo, hi  = sampled.min(), sampled.max()
    if hi == lo:
        row_idx = np.full(chart_w, height // 2, dtype=int)
    else:
        row_idx = ((sampled - lo) / (hi - lo) * (height - 1)).round().astype(int)

    grid    = [[" "] * chart_w for _ in range(height)]
    _SLOPES = {-1: "╲", 0: "─", 1: "╱"}
    for x, y in enumerate(row_idx):
        r  = height - 1 - y
        if x == 0:
            grid[r][x] = "─"
        else:
            dy = row_idx[x] - row_idx[x - 1]
            ch = _SLOPES.get(int(np.clip(dy, -1, 1)), "│")
            grid[r][x] = ch
            if abs(dy) > 1:
                step = 1 if dy > 0 else -1
                for mid in range(row_idx[x - 1] + step, row_idx[x], step):
                    grid[height - 1 - mid][x] = "│"

    return ["".join(row) for row in grid]


def _print_chart(
    eq_series: pd.Series,
    *,
    title: str,
    equity: float = 10_000.0,
    height: int = _CHART_HEIGHT,
) -> None:
    try:
        from rich.console import Console
        from rich.text import Text
        console  = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    eq_vals  = (eq_series * equity).to_numpy()
    dates    = eq_series.index
    term_w   = shutil.get_terminal_size().columns
    y_lbl_w  = 8
    chart_w  = max(40, term_w - y_lbl_w - 2)

    rows   = _render_chart(eq_vals, chart_w, height)
    lo, hi = eq_vals.min(), eq_vals.max()
    ticks  = np.linspace(hi, lo, height)

    print()
    if use_rich:
        console.print(f"[bold white]{title}[/bold white]")
    else:
        print(title)
    print()

    for i, (row_str, tick) in enumerate(zip(rows, ticks, strict=False)):
        label = f"{_fmt_dollar(tick):>7} "
        if use_rich:
            line = Text(label, style="dim")
            line.append("┤", style="dim")
            line.append(row_str, style="bold cyan" if i < height - 1 else "dim")
            console.print(line, end="\n")
        else:
            print(f"{label}|{row_str.translate(_ASCII_SAFE)}")

    x_axis = " " * y_lbl_w + "+" + "-" * chart_w
    print(x_axis)

    if hasattr(dates, "year"):
        n_labels      = 6
        label_idx     = np.linspace(0, len(dates) - 1, n_labels).astype(int)
        label_pos     = [int(i * chart_w / max(len(dates) - 1, 1)) for i in label_idx]
        year_labels   = [str(dates[i].year) for i in label_idx]
        x_label_row   = [" "] * chart_w
        for pos, yr in zip(label_pos, year_labels, strict=False):
            for j, ch in enumerate(yr):
                if pos + j < chart_w:
                    x_label_row[pos + j] = ch
        print(" " * (y_lbl_w + 1) + "".join(x_label_row))

    parts = []
    for label, bars in _LOOKBACKS:
        ret = _period_return(eq_vals, bars)
        if ret is not None:
            sign = "+" if ret >= 0 else ""
            parts.append(f"{label} {sign}{ret:.1%}")
    if parts:
        line = "    ".join(parts)
        if use_rich:
            console.print(f"  [dim]{line}[/dim]")
        else:
            print(f"  {line}")
    print()


def build_fund_returns(
    *,
    price_start: str = "2010-01-01",
) -> dict[str, tuple[pd.Series, float]]:
    """Return {family: (daily_returns, sleeve_weight)} for the live pick of each family.

    Reads live_picks.json if present; otherwise falls back to top-OOS-Sharpe paper record.
    """
    from meridian.data import load_ohlcv
    from meridian.execution.live_runner import _expand_symbol
    from meridian.families import create_model
    from meridian.pipeline.records import load_records

    picks_cfg = load_live_picks()

    # Fall back: top-Sharpe paper record per family
    paper = [r for r in load_records() if r.stage_passed == "paper"]
    fallback: dict = {}
    for r in paper:
        sharpe = (r.scorecard or {}).get("sharpe") or 0
        prev   = fallback.get(r.family)
        if prev is None or sharpe > (prev.scorecard or {}).get("sharpe", 0):
            fallback[r.family] = r

    all_families = set(picks_cfg) | set(fallback)
    out: dict[str, tuple[pd.Series, float]] = {}

    for family in all_families:
        if family in picks_cfg:
            model_name = picks_cfg[family]["model"]
            symbol     = picks_cfg[family]["symbol"]
        elif family in fallback:
            rec        = fallback[family]
            model_name = rec.model
            symbol     = rec.symbol
        else:
            continue

        try:
            model = create_model(family, model_name)
        except KeyError:
            print(f"  Warning: model not found — {family}/{model_name}")
            continue

        cross = getattr(model, "cross_sectional", False)
        try:
            if cross:
                symbols     = _expand_symbol(symbol)
                prices_dict = {}
                for sym in symbols:
                    try:
                        prices_dict[sym] = load_ohlcv(sym, price_start)["close"]
                    except Exception:
                        pass
                if not prices_dict:
                    continue
                bt = model.backtest(prices_dict)
            else:
                prices_series = load_ohlcv(symbol, price_start)["close"]
                bt            = model.backtest(prices_series)
        except Exception as exc:
            print(f"  Warning: backtest failed for {family}/{model_name}/{symbol}: {exc}")
            continue

        sleeve_weight   = live_pick_weight(family, all_families)
        out[family]     = (bt.returns, sleeve_weight)

    return out


def show_fund_equity_chart(
    returns_by_sleeve: dict[str, tuple[pd.Series, float]],
    *,
    symbol: str = "",
    equity: float = 100_000.0,
) -> None:
    """Render the combined fund equity chart."""
    weighted = _weighted_returns(returns_by_sleeve)
    if weighted.empty:
        return
    eq = equity_curve(weighted)
    final = float(eq.iloc[-1]) * equity
    title = f"Fund Equity — {symbol}  ($10k → {_fmt_dollar(final)})"
    _print_chart(eq, title=title, equity=equity, height=_CHART_HEIGHT)


def show_all_family_charts(
    returns_by_family: dict[str, tuple[pd.Series, float]],
    *,
    equity: float = 10_000.0,
) -> None:
    """Render one mini-chart per family, then the combined fund chart."""
    picks_cfg = load_live_picks()

    for family in sorted(returns_by_family):
        returns, weight = returns_by_family[family]
        eq    = equity_curve(returns)
        final = float(eq.iloc[-1]) * equity
        pick  = picks_cfg.get(family, {})
        label = f"{pick.get('model','?')}/{pick.get('symbol','?')}"
        title = (
            f"{family.upper()}  [{label}]  "
            f"{weight:.0%} sleeve  ($10k → {_fmt_dollar(final)})"
        )
        _print_chart(eq, title=title, equity=equity, height=_MINI_HEIGHT)

    # Combined
    weighted = _weighted_returns(returns_by_family)
    if weighted.empty:
        return
    eq    = equity_curve(weighted)
    final = float(eq.iloc[-1]) * equity
    _print_chart(
        eq,
        title=f"COMBINED FUND  ($10k → {_fmt_dollar(final)})",
        equity=equity,
        height=_CHART_HEIGHT,
    )
