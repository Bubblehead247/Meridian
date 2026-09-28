"""The skill lab's guarantees: plans are frozen, hold-outs are sealed, finals happen once."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from meridian.validation import skill_lab as lab


def _plan_data(**over):
    d = {"id": "t", "hypothesis": "h", "benchmark": "b", "test_start": "2015-01-01",
         "test_end": "2020-12-31", "pass_rule": {"t_alpha_min": 2.0},
         "configs": [{"k": 1}], "consequences": {"pass": "x"}}
    d.update(over)
    return d


def _git_repo(tmp_path, data):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _commit(tmp_path):
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "plan"], cwd=tmp_path, check=True)


def test_an_uncommitted_plan_is_refused(tmp_path):
    p = _git_repo(tmp_path, _plan_data())
    with pytest.raises(lab.PlanError, match="not committed"):
        lab.load_plan(p)


def test_a_plan_edited_after_commit_is_refused(tmp_path):
    p = _git_repo(tmp_path, _plan_data())
    _commit(tmp_path)
    lab.load_plan(p)
    p.write_text(json.dumps(_plan_data(configs=[{"k": 2}])), encoding="utf-8")
    with pytest.raises(lab.PlanError, match="uncommitted changes"):
        lab.load_plan(p)


def test_fingerprint_changes_with_any_field():
    assert lab.fingerprint(_plan_data()) != lab.fingerprint(_plan_data(test_end="2021-01-01"))


def _prices():
    idx = pd.bdate_range("2014-01-01", "2020-12-31")
    rng = np.random.default_rng(1)
    return {"A": pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))), index=idx)}


def test_development_runs_never_see_the_holdout(tmp_path):
    plan = lab.Plan(tmp_path / "p.json", _plan_data(holdout_start="2019-01-01"), "fp")
    seen_last = []

    def strategy(prices, cfg):
        seen_last.append(max(s.index.max() for s in prices.values()))
        return prices["A"].pct_change().fillna(0)

    def bench(prices):
        return {"A": prices["A"].pct_change().fillna(0) * 0.5}

    rf = pd.Series(0.0, index=_prices()["A"].index)
    lab.run(plan, _prices(), strategy, bench, rf, final=False, ledger=tmp_path / "l.jsonl")
    assert seen_last[0] < pd.Timestamp("2019-01-01")


def test_the_final_run_happens_once(tmp_path):
    plan = lab.Plan(tmp_path / "p.json", _plan_data(holdout_start="2019-01-01"), "fp")
    rf = pd.Series(0.0, index=_prices()["A"].index)
    strat = lambda p, c: p["A"].pct_change().fillna(0)  # noqa: E731
    bench = lambda p: {"A": p["A"].pct_change().fillna(0) * 0.5}  # noqa: E731
    ledger = tmp_path / "l.jsonl"
    lab.run(plan, _prices(), strat, bench, rf, final=True, ledger=ledger)
    with pytest.raises(lab.PlanError, match="hold-out is spent"):
        lab.run(plan, _prices(), strat, bench, rf, final=True, ledger=ledger)


def test_every_run_counts_toward_the_trial_total(tmp_path):
    plan = lab.Plan(tmp_path / "p.json", _plan_data(configs=[{"k": 1}, {"k": 2}]), "fp")
    rf = pd.Series(0.0, index=_prices()["A"].index)
    strat = lambda p, c: p["A"].pct_change().fillna(0) * c["k"]  # noqa: E731
    bench = lambda p: {"A": p["A"].pct_change().fillna(0) * 0.5}  # noqa: E731
    ledger = tmp_path / "l.jsonl"
    lab.run(plan, _prices(), strat, bench, rf, final=False, ledger=ledger)
    r = lab.run(plan, _prices(), strat, bench, rf, final=False, ledger=ledger)
    assert r["n_trials_total"] == 4


def test_alpha_test_recovers_a_known_alpha():
    idx = pd.bdate_range("2010-01-01", periods=2520)
    rng = np.random.default_rng(2)
    m = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    y = 0.5 * m + 0.0002 + pd.Series(rng.normal(0, 0.0005, len(idx)), index=idx)
    out = lab.alpha_test(y, {"M": m}, pd.Series(0.0, index=idx))
    assert out["betas"]["M"] == pytest.approx(0.5, abs=0.02)
    assert out["alpha"] == pytest.approx(0.0002 * 252, abs=0.01)
    assert out["t_alpha"] > 2


def test_minimum_detectable_ir():
    assert lab.mde_information_ratio(16) == pytest.approx(0.5)
