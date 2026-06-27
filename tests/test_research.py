"""Tests for research-stage hypothesis records (PLAN.md §6 steps 1-2)."""

from __future__ import annotations

import pytest

from meridian.pipeline import Hypothesis, ResearchLog, add_hypothesis, to_research_ledger


def _hyp(hid="rsi_exhaustion", family="mean_reversion", model="rsi", rationale="oversold reverts"):
    return Hypothesis(id=hid, family=family, model_name=model, rationale=rationale)


def test_is_documented_requires_all_fields():
    assert _hyp().is_documented()
    assert not Hypothesis(id="x", family="f", model_name="m", rationale="   ").is_documented()
    assert not Hypothesis(id="", family="f", model_name="m", rationale="r").is_documented()


def test_add_hypothesis_appends_and_rejects_duplicates():
    log = ResearchLog()
    add_hypothesis(log, _hyp("a"))
    add_hypothesis(log, _hyp("b", family="momentum"))
    assert len(log.entries) == 2
    with pytest.raises(ValueError, match="already logged"):
        add_hypothesis(log, _hyp("a"))


def test_by_family_and_get():
    log = ResearchLog()
    add_hypothesis(log, _hyp("a", family="mean_reversion"))
    add_hypothesis(log, _hyp("b", family="momentum"))
    assert [h.id for h in log.by_family("mean_reversion")] == ["a"]
    assert log.get("b").family == "momentum"
    assert log.get("missing") is None


def test_round_trip_serialization():
    log = ResearchLog()
    add_hypothesis(log, _hyp("a"))
    add_hypothesis(log, _hyp("b", family="breakouts"))
    clone = ResearchLog.from_dict(log.to_dict())
    assert clone == log


def test_to_research_ledger_creates_research_stage_ledger():
    led = to_research_ledger(_hyp("rsi_exhaustion", family="mean_reversion"))
    assert led.stage == "research"
    assert led.name == "rsi_exhaustion" and led.family == "mean_reversion"
    assert led.capital_alloc == 0.0


def test_to_research_ledger_rejects_undocumented():
    with pytest.raises(ValueError, match="not fully documented"):
        to_research_ledger(Hypothesis(id="x", family="f", model_name="m", rationale=""))
