"""Skill lab: research that can't quietly fool itself.

Why this exists (research/2026-09-returns-study): picks chosen on full history
looked like 14%/yr, the honest version made 8%, and in-sample rank predicted
almost nothing out-of-sample. The statistics were already in this package; what
was missing was discipline. The lab enforces it in code:

  1. A plan is a JSON file, committed to git before it runs. Its sha256
     fingerprint is checked on every run; a dirty or uncommitted plan is refused.
  2. Every run is appended to a ledger, so the number of configurations ever
     tried is known — and fed to the deflated Sharpe.
  3. If the plan has a hold-out, development runs see prices only up to the day
     before it, however long a strategy's lookback. A final run opens the
     hold-out, and only once per fingerprint.
  4. The report says, before the result, how big an edge the test can detect
     (a fail on an underpowered test is not evidence of no edge).
  5. Skill = alpha over the plan's named benchmark, above T-bills, with
     Newey-West standard errors (daily returns from monthly rebalancing are
     autocorrelated; ``classical_t_stat`` assumes they are not).

The lab cannot stop a researcher from having *seen* data; only the future is
truly unseen. Plans should say what forward (shadow) test follows a result.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

LEDGER = Path("research") / "plans" / "ledger.jsonl"
REQUIRED = ("id", "hypothesis", "benchmark", "test_start", "test_end", "pass_rule",
            "configs", "consequences")


class PlanError(RuntimeError):
    """The plan is missing, changed, uncommitted, or its hold-out is spent."""


@dataclass(frozen=True)
class Plan:
    path: Path
    data: dict
    fingerprint: str

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def holdout_start(self) -> pd.Timestamp | None:
        h = self.data.get("holdout_start")
        return pd.Timestamp(h) if h else None


def fingerprint(data: dict) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_plan(path: str | Path, *, require_committed: bool = True) -> Plan:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise PlanError(f"plan {path} lacks {missing}")
    if require_committed:
        _require_committed(path)
    return Plan(path, data, fingerprint(data))


def _require_committed(path: Path) -> None:
    """The commit is the timestamp proof that the plan preceded the run."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=path.parent)
    rel = path.name
    if git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise PlanError(f"plan {path} is not committed — commit it before running")
    if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise PlanError(f"plan {path} has uncommitted changes — a changed plan is a new plan")


def mde_information_ratio(years: float, t: float = 2.0) -> float:
    """Smallest annual information ratio a test of ``years`` can detect at ``t``.

    t(alpha) ≈ IR · sqrt(years), so IR_min = t / sqrt(years).
    """
    return t / np.sqrt(years) if years > 0 else float("inf")


def alpha_test(strategy: pd.Series, benchmarks: dict[str, pd.Series], rf: pd.Series,
               *, lags: int = 21, periods_per_year: int = 252) -> dict:
    """OLS of strategy excess returns on benchmark excess returns, Newey-West t."""
    import statsmodels.api as sm

    cols = {"y": strategy - rf, **{k: v - rf for k, v in benchmarks.items()}}
    df = pd.concat(cols, axis=1).dropna()
    X = sm.add_constant(df[list(benchmarks)])
    fit = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    resid_sd = float(np.std(fit.resid, ddof=len(benchmarks) + 1))
    alpha = float(fit.params["const"]) * periods_per_year
    te = resid_sd * np.sqrt(periods_per_year)
    return {
        "alpha": alpha,
        "t_alpha": float(fit.tvalues["const"]),
        "betas": {k: float(fit.params[k]) for k in benchmarks},
        "r2": float(fit.rsquared),
        "tracking_error": te,
        "information_ratio": alpha / te if te > 0 else float("nan"),
        "n_days": int(len(df)),
    }


def truncate_before_holdout(prices: dict[str, pd.Series], plan: Plan) -> dict[str, pd.Series]:
    """Development runs never receive a price on or after the hold-out start."""
    h = plan.holdout_start
    if h is None:
        return prices
    return {k: v.loc[: h - pd.Timedelta(days=1)] for k, v in prices.items()}


def _ledger_rows(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]


def trials_so_far(plan: Plan, ledger: Path = LEDGER) -> int:
    """Configurations tried under this plan id across every logged run."""
    return sum(r["n_configs"] for r in _ledger_rows(ledger) if r["plan_id"] == plan.id)


def _log(plan: Plan, kind: str, n_configs: int, ledger: Path, extra: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    row = {"plan_id": plan.id, "fingerprint": plan.fingerprint, "kind": kind,
           "n_configs": n_configs, "at": datetime.now(timezone.utc).isoformat(), **extra}
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run(plan: Plan, prices: dict[str, pd.Series],
        strategy: Callable[[dict[str, pd.Series], dict], pd.Series],
        benchmarks: Callable[[dict[str, pd.Series]], dict[str, pd.Series]],
        rf: pd.Series, *, final: bool, ledger: Path = LEDGER) -> dict:
    """Run every registered config; log it; return the skill report.

    ``strategy(prices, config) -> daily net returns``. ``benchmarks(prices)``
    returns the plan's named benchmark return series. On a development run
    both see only pre-hold-out prices.
    """
    if final:
        done = [r for r in _ledger_rows(ledger)
                if r["fingerprint"] == plan.fingerprint and r["kind"] == "final"]
        if done:
            raise PlanError(f"final run for {plan.id} already happened at {done[0]['at']}; "
                            "the hold-out is spent — a new question needs a new plan")
        seen = prices
    else:
        seen = truncate_before_holdout(prices, plan)

    start = pd.Timestamp(plan.data["test_start"])
    end = pd.Timestamp(plan.data["test_end"])
    if not final and plan.holdout_start is not None:
        end = min(end, plan.holdout_start - pd.Timedelta(days=1))

    configs = plan.data["configs"]
    bench = {k: v.loc[start:end] for k, v in benchmarks(seen).items()}
    rf_w = rf.loc[start:end]
    results = []
    for cfg in configs:
        r = strategy(seen, cfg).loc[start:end]
        results.append({"config": cfg, "returns": r,
                        "alpha": alpha_test(r, bench, rf_w)})

    from meridian.validation.deflated_sharpe import deflated_sharpe_ratio

    per_period_sr = [float(x["returns"].mean() / x["returns"].std()) for x in results]
    n_trials = max(trials_so_far(plan, ledger) + len(configs), len(configs))
    best = max(results, key=lambda x: x["alpha"]["t_alpha"])
    dsr = deflated_sharpe_ratio(best["returns"].to_numpy(), trial_sharpes=per_period_sr,
                                n_trials=n_trials) if len(configs) > 1 else None
    years = len(best["returns"]) / 252
    # A plan with t_alpha_min = null (e.g. a non-inferiority test) makes no alpha
    # claim: the lab still logs and reports, and the caller applies the plan's rule.
    t_raw = plan.data["pass_rule"].get("t_alpha_min", 2.0)
    t_needed = float(t_raw) if t_raw is not None else 2.0
    report = {
        "plan": plan.id, "fingerprint": plan.fingerprint, "final": final,
        "window": [str(start.date()), str(end.date())], "years": years,
        "mde_information_ratio": mde_information_ratio(years, t_needed),
        "n_trials_total": n_trials, "results": results, "best": best,
        "deflated_sharpe": dsr,
        "passed": None if t_raw is None else (
            best["alpha"]["alpha"] > 0 and best["alpha"]["t_alpha"] >= t_needed
            and (dsr is None or dsr["dsr_pvalue"] >= 0.95)),
    }
    _log(plan, "final" if final else "dev", len(configs), ledger,
         {"passed": report["passed"], "t_alpha": best["alpha"]["t_alpha"],
          "alpha": best["alpha"]["alpha"]})
    return report
