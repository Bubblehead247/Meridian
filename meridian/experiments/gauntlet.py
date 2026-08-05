"""Strategy gauntlet — run every model on a symbol/basket and rank them.

A "which strategy works best here" scoreboard: each registered model is run through the
``backtest`` stage only (full-history, in-sample) and ranked by its scorecard. This module
owns the ranking/aggregation only; it reuses ``pipeline.run_backtest_stage`` /
``run_universe_backtest_stage`` (and the scorecard inside them) for all the math and does no
network I/O itself (callers pass loaded prices), so it is fully offline-testable.

IMPORTANT: despite the name, this is a full-history ranking, not an out-of-sample one — it
never calls ``run_walk_forward_stage`` or ``run_oos_stage``, so it never touches the fixed
OOS holdout and its numbers should not be read as OOS-style evidence. A model ranked highly
here still has to clear walk-forward/OOS on its own before it means anything; use
``run_pipeline`` / ``run_cross_sectional_pipeline`` / ``run_universe_pipeline`` for that.
"""

from __future__ import annotations

import pandas as pd

from meridian.families import create_model, list_models
from meridian.pipeline import run_backtest_stage, run_universe_backtest_stage
from meridian.pipeline.graduation import criteria_for_family

_COLUMNS = [
    "family", "model", "passed", "sharpe", "cagr", "total_return",
    "max_drawdown", "n_trades", "trades_per_year", "fill_realism",
]

_PERIODS_PER_YEAR = 252


def _trades_per_year(card: dict) -> float | None:
    n_trades = card.get("n_trades")
    n_periods = card.get("n_periods")
    if n_trades is None or n_periods is None or n_periods == 0:
        return None
    return n_trades / (n_periods / _PERIODS_PER_YEAR)


def _row(family: str, name: str, result) -> dict:
    card = result.scorecard
    return {
        "family": family,
        "model": name,
        "passed": result.passed,
        "sharpe": card.get("sharpe"),
        "cagr": card.get("cagr"),
        "total_return": card.get("total_return"),
        "max_drawdown": card.get("max_drawdown"),
        "n_trades": card.get("n_trades"),
        "trades_per_year": _trades_per_year(card),
        "fill_realism": result.detail.get("fill_realism", "close_approx"),
    }


def _ranked(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_COLUMNS)
    return df.sort_values("sharpe", ascending=False, na_position="last").reset_index(drop=True)


def gauntlet_single(
    prices: pd.Series,
    *,
    bars: pd.DataFrame | None = None,
    cost_bps: float = 1.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Rank every single-asset model on one symbol's ``prices`` (best Sharpe first)."""
    rows = []
    for family, name in _single_asset_models():
        model = create_model(family, name)
        res = run_backtest_stage(
            model, prices, bars=bars, cost_bps=cost_bps, periods_per_year=periods_per_year,
            criteria=criteria_for_family(family),
        )
        rows.append(_row(family, name, res))
    return _ranked(rows)


def gauntlet_universe(
    prices_by_symbol: dict[str, pd.Series],
    *,
    cost_bps: float = 1.0,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Rank every cross-sectional model on a basket (best Sharpe first).

    ``bars_by_symbol`` (optional OHLC per symbol) enables next-open fills — without it,
    every ranked model silently falls back to same-close fills (see
    ``run_universe_backtest_stage`` / ``CrossSectionalModel.backtest``).
    """
    rows = []
    for family, name in _cross_sectional_models():
        model = create_model(family, name)
        res = run_universe_backtest_stage(
            model, prices_by_symbol, cost_bps=cost_bps, bars_by_symbol=bars_by_symbol,
            periods_per_year=periods_per_year, criteria=criteria_for_family(family),
        )
        rows.append(_row(family, name, res))
    return _ranked(rows)


def _single_asset_models() -> list[tuple[str, str]]:
    return [_split(m) for m in list_models()
            if not getattr(create_model(*_split(m)), "cross_sectional", False)]


def _cross_sectional_models() -> list[tuple[str, str]]:
    return [_split(m) for m in list_models()
            if getattr(create_model(*_split(m)), "cross_sectional", False)]


def _split(qualified: str) -> tuple[str, str]:
    family, _, name = qualified.partition("/")
    return family, name


def _fmt_num(series: pd.Series, fmt: str) -> pd.Series:
    return series.map(lambda v: "n/a" if v is None or v != v else fmt.format(v))


def format_gauntlet(df: pd.DataFrame) -> str:
    """Pretty, fixed-width table of a gauntlet result for printing."""
    if df.empty:
        return "(no models ran)"
    shown = pd.DataFrame(
        {
            "family": df["family"],
            "model": df["model"],
            "passed": df["passed"],
            "sharpe": _fmt_num(df["sharpe"], "{:.2f}"),
            "cagr": _fmt_num(df["cagr"], "{:.1%}"),
            "total_return": _fmt_num(df["total_return"], "{:.1%}"),
            "max_drawdown": _fmt_num(df["max_drawdown"], "{:.1%}"),
            "n_trades": df["n_trades"].map(lambda v: "" if v is None or v != v else int(v)),
            "trades/yr": _fmt_num(df["trades_per_year"], "{:.1f}"),
            "fill_realism": df["fill_realism"],
        }
    )
    return shown.to_string(index=False)
