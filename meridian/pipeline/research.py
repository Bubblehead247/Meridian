"""Idea and hypothesis records for the early research stage of the graduation pipeline.

This module owns idea and hypothesis records (steps 1-2 of graduation); it
does NOT own scoring or metric computation (those stay in scoring/scorecard.py).

A strategy enters the pipeline at the ``research`` stage once its idea and testable
hypothesis are written down. A ``Hypothesis`` captures that; a ``ResearchLog`` collects
them; ``to_research_ledger`` mints the research-stage ``StrategyLedger`` (0% capital) that
the graduation state machine then advances.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

from meridian.portfolio.ledger import StrategyLedger


def _today() -> str:
    return date.today().isoformat()


@dataclass
class Hypothesis:
    """A documented idea + testable hypothesis for one model (research-stage record)."""

    id: str
    family: str
    model_name: str
    rationale: str                       # the idea / testable hypothesis
    created_date: str = field(default_factory=_today)

    def is_documented(self) -> bool:
        """Research-stage entry criterion: a non-empty rationale and identity."""
        return bool(self.id and self.family and self.model_name and self.rationale.strip())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Hypothesis:
        return cls(**d)


@dataclass
class ResearchLog:
    """An ordered collection of research hypotheses."""

    entries: list[Hypothesis] = field(default_factory=list)

    def by_family(self, family: str) -> list[Hypothesis]:
        return [h for h in self.entries if h.family == family]

    def get(self, hypothesis_id: str) -> Hypothesis | None:
        return next((h for h in self.entries if h.id == hypothesis_id), None)

    def to_dict(self) -> dict:
        return {"entries": [h.to_dict() for h in self.entries]}

    @classmethod
    def from_dict(cls, d: dict) -> ResearchLog:
        return cls(entries=[Hypothesis.from_dict(e) for e in d.get("entries", [])])


def add_hypothesis(log: ResearchLog, hypothesis: Hypothesis) -> ResearchLog:
    """Append a hypothesis (duplicate id rejected); returns the same log for chaining."""
    if log.get(hypothesis.id) is not None:
        raise ValueError(f"hypothesis id already logged: {hypothesis.id!r}")
    log.entries.append(hypothesis)
    return log


def to_research_ledger(hypothesis: Hypothesis) -> StrategyLedger:
    """Mint a research-stage (0% capital) ledger from a documented hypothesis."""
    if not hypothesis.is_documented():
        raise ValueError(f"hypothesis {hypothesis.id!r} is not fully documented")
    return StrategyLedger(
        name=hypothesis.id, family=hypothesis.family, stage="research",
        stage_entered=hypothesis.created_date,
    )
