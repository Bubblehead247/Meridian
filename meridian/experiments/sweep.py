"""Parameter sweep — try a strategy's settings and keep only what survives out-of-sample.

Tuning a strategy's parameters until it "passes" on the same history is overfitting: try
enough settings and one looks good by luck, then fails on new data. So this sweep scores
every parameter combination on **walk-forward out-of-sample** data (not just the full
history it was fit on) and reports how many settings were tried — a config only counts as a
real pass when it clears the metric bar OUT-OF-SAMPLE.

This module owns building the parameter grids and aggregating; it reuses
``pipeline.run_backtest_stage`` (in-sample) and ``run_walk_forward_stage`` (OOS) for the
math and does no network I/O (callers pass loaded prices), so it is offline-testable.
"""

from __future__ import annotations

import itertools

import pandas as pd

from meridian.families import create_model
from meridian.pipeline import run_backtest_stage, run_walk_forward_stage
from meridian.validation import WalkForwardSpec

#: Parameter grids per ``family/model`` (the primary tunable knob(s), kept small so the
#: number of settings tried — and the multiple-testing burden — stays modest).
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "mean_reversion/zscore_reversion": {"window": [10, 15, 20, 30, 40]},
    "mean_reversion/bollinger_reversion": {"window": [10, 15, 20, 30, 40]},
    "mean_reversion/atr_extension": {"window": [10, 15, 20, 30, 40]},
    "mean_reversion/rsi_exhaustion": {"window": [7, 10, 14, 21]},
    "mean_reversion/gap_fill": {"threshold": [0.005, 0.01, 0.02, 0.03]},
    "trend_following/ma_trend": {"window": [20, 50, 100, 150, 200]},
    "trend_following/adx_trend": {"window": [20, 50, 100, 150, 200]},
    "trend_following/channel_breakout": {"window": [10, 20, 40, 55]},
    "momentum/roc_momentum": {"window": [20, 40, 60, 120, 200]},
    "breakouts/donchian_breakout": {"window": [10, 20, 40, 55]},
    "breakouts/volume_confirmed_breakout": {"window": [10, 20, 40, 55]},
    "breakouts/nr7_breakout": {"window": [4, 5, 7, 10]},
    "pullback_continuation/ma_pullback": {"fast": [10, 20], "slow": [50, 100]},
    "pullback_continuation/rsi_pullback": {"fast": [10, 20], "slow": [50, 100]},
    "sector_rotation/trend_rotation": {"window": [50, 100, 150, 200]},
    "long_term_etf/above_200ma": {"window": [100, 150, 200, 250]},
}

_FALLBACK_WINDOW = [10, 20, 50, 100, 200]


def grid_for(family: str, name: str) -> dict[str, list]:
    """The parameter grid for a model — explicit if defined, else a window sweep if it has one."""
    key = f"{family}/{name}"
    if key in PARAM_GRIDS:
        return PARAM_GRIDS[key]
    model = create_model(family, name)
    if hasattr(model, "window"):
        return {"window": _FALLBACK_WINDOW}
    return {}


def _combos(grid: dict[str, list]) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*grid.values())]


def sweep_single(
    prices: pd.Series,
    family: str,
    name: str,
    *,
    bars: pd.DataFrame | None = None,
    cost_bps: float = 1.0,
    spec: WalkForwardSpec | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Sweep a single-asset model's parameters; score each in-sample AND walk-forward OOS.

    Returns a table (one row per setting) sorted by **out-of-sample** Sharpe, with the
    parameter columns plus ``is_sharpe``/``is_passed`` (full history) and
    ``oos_sharpe``/``oos_passed``/``oos_folds`` (walk-forward). A setting is only a real pass
    when ``oos_passed`` is True.
    """
    model = create_model(family, name)
    if getattr(model, "cross_sectional", False):
        raise ValueError(
            f"{family}/{name} is cross-sectional; sweep it on a basket (not yet wired here)"
        )
    spec = spec or WalkForwardSpec()
    grid = grid_for(family, name)
    rows = []
    for combo in _combos(grid):
        m = create_model(family, name)
        for param, value in combo.items():
            setattr(m, param, value)
        is_res = run_backtest_stage(
            m, prices, bars=bars, cost_bps=cost_bps, periods_per_year=periods_per_year
        )
        oos_res = run_walk_forward_stage(
            m, prices, bars=bars, cost_bps=cost_bps, spec=spec, periods_per_year=periods_per_year
        )
        rows.append(
            {
                **combo,
                "is_sharpe": is_res.scorecard.get("sharpe"),
                "is_passed": is_res.passed,
                "oos_sharpe": oos_res.scorecard.get("sharpe"),
                "oos_passed": oos_res.passed,
                "oos_folds": oos_res.detail.get("n_folds", 0),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("oos_sharpe", ascending=False, na_position="last").reset_index(drop=True)


def _param_cols(df: pd.DataFrame) -> list[str]:
    meta = {"is_sharpe", "is_passed", "oos_sharpe", "oos_passed", "oos_folds"}
    return [c for c in df.columns if c not in meta]


def sweep_and_confirm(
    prices: pd.Series,
    family: str,
    name: str,
    *,
    bars: pd.DataFrame | None = None,
    cost_bps: float = 1.0,
    spec: WalkForwardSpec | None = None,
    periods_per_year: int = 252,
) -> dict:
    """Sweep, then put the winning setting through the FULL pipeline (fixed OOS holdout).

    The sweep ranks settings on walk-forward folds; this then takes the best **OOS-passing**
    setting (or, if none passed, the top candidate) and runs it through
    ``pipeline.run_pipeline`` — backtest → walk-forward → the *fixed* 2010-2019/2020-2022
    holdout — which is the independent judge. ``confirmed`` is True only if that holdout
    stage passes. Returns the sweep table, the chosen params, the per-stage pipeline results,
    and the verdict.
    """
    from meridian.pipeline import run_pipeline
    from meridian.portfolio import StrategyLedger

    df = sweep_single(
        prices, family, name, bars=bars, cost_bps=cost_bps, spec=spec,
        periods_per_year=periods_per_year,
    )
    passing = df[df["oos_passed"]]
    winner = passing.iloc[0] if len(passing) else df.iloc[0]
    params = {c: winner[c] for c in _param_cols(df)}

    model = create_model(family, name)
    for param, value in params.items():
        setattr(model, param, value)
    ledger = StrategyLedger(name=f"{name}_tuned", family=family, stage="research")
    pipeline = run_pipeline(
        model, prices, ledger=ledger, bars=bars, cost_bps=cost_bps,
        periods_per_year=periods_per_year, stop_on_fail=False,
    )
    oos_stage = pipeline.get("oos")
    return {
        "sweep": df,
        "params": params,
        "winner_oos_passed": bool(winner["oos_passed"]),
        "pipeline": pipeline,
        "confirmed": bool(oos_stage is not None and oos_stage.passed),
    }


def format_confirm(result: dict, family: str, name: str) -> str:
    """Render a sweep-and-confirm result: the sweep table, the holdout run, and the verdict."""
    lines = [format_sweep(result["sweep"], family, name), "",
             "=== Auto-confirmation through the full pipeline (fixed OOS holdout) ==="]
    params = ", ".join(f"{k}={v}" for k, v in result["params"].items()) or "(default)"
    note = ("best OOS-passing setting" if result["winner_oos_passed"]
            else "top candidate (NONE passed the sweep OOS — confirming it anyway)")
    lines.append(f"Confirming {note}: {params}")
    for stage, res in result["pipeline"].items():
        sharpe = res.scorecard.get("sharpe")
        sh = f"{sharpe:.2f}" if isinstance(sharpe, float) and sharpe == sharpe else "n/a"
        extra = ""
        if stage == "oos" and res.detail.get("oos_sharpe") is not None:
            d = res.detail
            extra = f"  (holdout Sharpe {d['oos_sharpe']:.2f}, degraded={d.get('degraded')})"
        lines.append(f"  {stage:12s} passed={res.passed!s:5s} sharpe={sh}{extra}")
    verdict = ("CONFIRMED ✓ — holds up on the fixed out-of-sample holdout"
               if result["confirmed"]
               else "REJECTED ✗ — fails the fixed out-of-sample holdout (a tuning artifact)")
    lines.append(f"\nVERDICT: {verdict}")
    return "\n".join(lines)


def _fmt(series: pd.Series, fmt: str) -> pd.Series:
    return series.map(lambda v: "n/a" if v is None or v != v else fmt.format(v))


def format_sweep(df: pd.DataFrame, family: str, name: str) -> str:
    """Render a sweep result with an explicit multiple-testing caveat and the OOS verdict."""
    if df.empty:
        return f"{family}/{name}: no tunable parameters."
    param_cols = [c for c in df.columns
                  if c not in ("is_sharpe", "is_passed", "oos_sharpe", "oos_passed", "oos_folds")]
    shown = df[param_cols].copy()
    shown["is_sharpe"] = _fmt(df["is_sharpe"], "{:.2f}")
    shown["oos_sharpe"] = _fmt(df["oos_sharpe"], "{:.2f}")
    shown["oos_passed"] = df["oos_passed"]

    n = len(df)
    n_oos_pass = int(df["oos_passed"].sum())
    best = df.iloc[0]
    lines = [
        f"Sweep — {family}/{name}: {n} settings tried "
        f"(ranked by out-of-sample Sharpe; in-sample shown for reference)",
        shown.to_string(index=False),
        "",
        f"Out-of-sample passes: {n_oos_pass} of {n}.",
    ]
    if n_oos_pass:
        params = ", ".join(f"{c}={best[c]}" for c in param_cols)
        lines.append(f"Best OOS config: {params}  (oos_sharpe={best['oos_sharpe']:.2f}).")
    else:
        lines.append("No setting clears the bar out-of-sample — in-sample wins here are likely "
                     "curve-fitting, not a real edge.")
    lines.append(f"⚠ Multiple-testing: trying {n} settings inflates the chance one looks good "
                 "by luck. Treat any single OOS pass as a candidate to confirm, not proof.")
    return "\n".join(lines)
