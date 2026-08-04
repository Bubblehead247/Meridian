"""Validation engine: walk-forward, bootstrap, Monte-Carlo, multiple-testing.

Turns the in-sample sanity numbers of Phases 4-5 into honest, out-of-sample,
significance-tested results.

    from meridian.validation import validate, walk_forward, WalkForwardSpec
"""

from meridian.validation.bootstrap import block_bootstrap_sharpe
from meridian.validation.correction import benjamini_hochberg, bonferroni, correct
from meridian.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio_stdev,
)
from meridian.validation.effective_tests import effective_num_tests
from meridian.validation.montecarlo import monte_carlo_pvalue
from meridian.validation.pipeline import validate
from meridian.validation.sensitivity import parameter_sensitivity
from meridian.validation.stats import sharpe, strategy_net, total_return
from meridian.validation.walkforward import (
    Fold,
    SelectionResult,
    WalkForwardResult,
    WalkForwardSpec,
    make_folds,
    walk_forward,
)

__all__ = [
    "validate",
    "walk_forward",
    "WalkForwardSpec",
    "WalkForwardResult",
    "SelectionResult",
    "Fold",
    "make_folds",
    "block_bootstrap_sharpe",
    "monte_carlo_pvalue",
    "benjamini_hochberg",
    "bonferroni",
    "correct",
    "sharpe",
    "total_return",
    "strategy_net",
    "sharpe_ratio_stdev",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "effective_num_tests",
    "parameter_sensitivity",
]
