"""Phase 7 tests: performance metrics and report generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.analytics import (
    build_validation_report,
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    performance_metrics,
    sharpe_ratio,
    write_report,
)

# --- metrics: exact / known values ---------------------------------------

def test_drawdown_and_max_drawdown():
    # equity: 1.0 -> 1.1 -> 0.99 -> ... peak 1.1, trough 0.99 -> dd = 0.99/1.1 - 1
    rets = pd.Series([0.10, -0.10, 0.0])
    dd = drawdown_series(rets)
    assert dd.iloc[0] == pytest.approx(0.0)
    assert max_drawdown(rets) == pytest.approx(0.99 / 1.10 - 1.0)


def test_max_drawdown_duration_counts_underwater_bars():
    # down, down, recover above peak
    rets = pd.Series([0.0, -0.05, -0.05, 0.5])
    assert max_drawdown_duration(rets) == 2


def test_metrics_basic_fields_and_hit_rate():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 500))
    m = performance_metrics(rets)
    for key in ["total_return", "cagr", "ann_volatility", "sharpe", "sortino",
                "max_drawdown", "calmar", "hit_rate", "skew", "kurtosis",
                "var_95", "cvar_95", "tail_ratio"]:
        assert key in m
    assert 0.0 <= m["hit_rate"] <= 1.0
    assert m["max_drawdown"] <= 0.0
    assert m["var_95"] <= 0.0  # 5th percentile of returns


def test_profit_factor_all_wins_is_inf():
    m = performance_metrics(pd.Series([0.01, 0.02, 0.03]))
    assert m["profit_factor"] == float("inf")
    assert m["hit_rate"] == 1.0


def test_sharpe_ratio_matches_performance_metrics_and_validation_stats():
    """Consolidation check: analytics.sharpe_ratio, performance_metrics()'s
    'sharpe' field, and validation.stats.sharpe must all agree exactly (same
    formula, one source of truth) — this was two hand-written copies before."""
    from meridian.validation.stats import sharpe as validation_sharpe

    rng = np.random.default_rng(9)
    r = rng.normal(0.0005, 0.01, 500)
    a = sharpe_ratio(r)
    b = performance_metrics(r)["sharpe"]
    c = validation_sharpe(r)
    assert a == b == c


def test_metrics_empty_is_safe():
    m = performance_metrics(pd.Series([], dtype=float))
    assert m["n_periods"] == 0


def test_metrics_with_trades():
    rets = pd.Series([0.01, -0.01, 0.02, 0.0])
    trades = pd.DataFrame({"return": [0.05, -0.02, 0.03]})
    m = performance_metrics(rets, trades=trades)
    assert m["n_trades"] == 3
    assert m["trade_win_rate"] == pytest.approx(2 / 3)
    assert m["trade_expectancy"] == pytest.approx((0.05 - 0.02 + 0.03) / 3)


# --- report ---------------------------------------------------------------

def _fake_table(significant=False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estimator": ["hull", "ema", "ou"],
            "oos_sharpe": [0.41, 0.10, -0.05],
            "oos_return": [0.6, 0.1, -0.1],
            "mc_pvalue": [0.037, 0.4, 0.7],
            "q_value": [0.02 if significant else 0.52, 0.6, 0.7],
            "significant": [significant, False, False],
        }
    )


def test_report_includes_mandatory_disclosures():
    md = build_validation_report(_fake_table(), meta={"symbols": "SPY", "deviation": "zscore"})
    low = md.lower()
    assert "survivorship" in low
    assert "multiple testing" in low
    assert "out-of-sample" in low
    assert "transaction cost" in low


def test_report_no_significant_verdict():
    md = build_validation_report(_fake_table(significant=False))
    assert "No estimator is statistically significant" in md
    assert "hull" in md  # names the best performer


def test_report_significant_verdict_lists_winners():
    md = build_validation_report(_fake_table(significant=True))
    assert "statistically significant" in md
    assert "`hull`" in md


def test_write_report_creates_file(tmp_path):
    md = build_validation_report(_fake_table())
    p = write_report(tmp_path / "sub" / "report.md", md)
    assert p.exists()
    assert "Meridian" in p.read_text(encoding="utf-8")


# --- survivorship-bias banner ----------------------------------------------

def test_no_banner_when_flag_absent_from_meta():
    md = build_validation_report(_fake_table(), meta={"symbols": "SPY"})
    assert "Survivorship-biased run" not in md
    assert "Point-in-time universe" not in md


def test_banner_flags_biased_run():
    md = build_validation_report(_fake_table(), meta={"survivorship_biased": True})
    assert "Survivorship-biased run" in md


def test_banner_flags_point_in_time_run():
    md = build_validation_report(_fake_table(), meta={"survivorship_biased": False})
    assert "Point-in-time universe" in md


# --- robustness section (Phase 2: m_eff, DSR, sensitivity) -----------------

def _phase2_table(flip_significance=False, unstable=False) -> pd.DataFrame:
    df = _fake_table(significant=False)
    df["m_eff"] = 1.8
    df["dsr_pvalue"] = [0.6, 0.4, 0.2]
    df["significant_eff"] = [flip_significance, False, False]
    df["sharpe_sign_stable"] = [not unstable, True, True]
    return df


def test_no_robustness_section_when_columns_absent():
    md = build_validation_report(_fake_table())
    assert "## Robustness" not in md


def test_robustness_section_present_with_no_flips():
    md = build_validation_report(_phase2_table())
    assert "## Robustness" in md
    assert "m_eff=1.800" in md
    assert "No estimator's significance flips" in md
    assert "every estimator's OOS Sharpe sign is" in md


def test_robustness_section_flags_significance_flip():
    md = build_validation_report(_phase2_table(flip_significance=True))
    assert "only clear the bar under the effective-m correction" in md
    assert "`hull`" in md  # the flipped estimator is named


def test_robustness_section_flags_unstable_sensitivity():
    md = build_validation_report(_phase2_table(unstable=True))
    assert "flip sign across neighboring windows" in md


def test_robustness_section_shows_dsr_for_best_performer():
    md = build_validation_report(_phase2_table())
    assert "Deflated Sharpe Ratio" in md
    assert "dsr_pvalue=0.600" in md
