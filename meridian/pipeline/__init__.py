"""Pipeline package — stage-runners that drive strategy models through research to production."""

from meridian.pipeline.backtest import StageResult, run_backtest_stage
from meridian.pipeline.graduation import (
    FAMILY_CRITERIA,
    MIN_DAYS_IN_STAGE,
    STAGE_CAPITAL,
    GraduationCriteria,
    advance,
    check_promotion,
    check_retirement,
    criteria_for_family,
    evaluate,
    next_stage,
    passes_metric_bar,
)
from meridian.pipeline.oos import compare_oos_to_is, run_oos_stage
from meridian.pipeline.oos_guard import OOSGuard, OOSRunRecord
from meridian.pipeline.orchestrator import run_cross_sectional_pipeline, run_pipeline
from meridian.pipeline.records import (
    DEFAULT_RECORDS_DIR,
    StrategyRecord,
    load_records,
    record_from_pipeline,
    save_record,
)
from meridian.pipeline.research import (
    Hypothesis,
    ResearchLog,
    add_hypothesis,
    to_research_ledger,
)
from meridian.pipeline.universe import (
    run_universe_backtest_stage,
    run_universe_oos_stage,
    run_universe_pipeline,
    run_universe_walk_forward_stage,
)
from meridian.pipeline.walk_forward import run_walk_forward_stage

#: Stage name -> runner, for the dispatcher below.
STAGE_RUNNERS = {
    "backtest": run_backtest_stage,
    "walk_forward": run_walk_forward_stage,
    "oos": run_oos_stage,
}


def run_stage(stage: str, model, prices, **kwargs) -> StageResult:
    """Dispatch to the named stage runner (``backtest`` / ``walk_forward`` / ``oos``)."""
    if stage not in STAGE_RUNNERS:
        raise KeyError(f"unknown stage {stage!r}. Known: {', '.join(STAGE_RUNNERS)}")
    return STAGE_RUNNERS[stage](model, prices, **kwargs)


__all__ = [
    "StrategyRecord",
    "DEFAULT_RECORDS_DIR",
    "save_record",
    "load_records",
    "record_from_pipeline",
    "GraduationCriteria",
    "FAMILY_CRITERIA",
    "criteria_for_family",
    "STAGE_CAPITAL",
    "MIN_DAYS_IN_STAGE",
    "next_stage",
    "check_promotion",
    "check_retirement",
    "evaluate",
    "advance",
    "passes_metric_bar",
    "Hypothesis",
    "ResearchLog",
    "add_hypothesis",
    "to_research_ledger",
    "StageResult",
    "run_backtest_stage",
    "run_walk_forward_stage",
    "run_oos_stage",
    "compare_oos_to_is",
    "OOSGuard",
    "OOSRunRecord",
    "run_stage",
    "STAGE_RUNNERS",
    "run_pipeline",
    "run_cross_sectional_pipeline",
    "run_universe_backtest_stage",
    "run_universe_walk_forward_stage",
    "run_universe_oos_stage",
    "run_universe_pipeline",
]
