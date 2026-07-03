"""Tests for look-through sector exposure (portfolio/sectors.py)."""

from __future__ import annotations

from datetime import date

from meridian.execution.live_runner import StrategyDecision
from meridian.portfolio.sectors import (
    SECTOR_WEIGHTS,
    flag_concentration,
    lookthrough_exposures,
    position_weights_from_decisions,
)
from meridian.reporting.review_runner import _render_sector_exposure


def _dec(family, symbol_shares: dict[str, float], weight=0.15, skipped=False):
    return StrategyDecision(
        family=family, model="m", symbol="_".join(symbol_shares), as_of=date.today(),
        signals={s: (1 if sh else 0) for s, sh in symbol_shares.items()},
        target_shares=symbol_shares, orders=[], weight=weight, skipped=skipped,
    )


# --- SECTOR_WEIGHTS sanity -------------------------------------------------

def test_all_sector_weight_rows_sum_to_one():
    for sym, sectors in SECTOR_WEIGHTS.items():
        assert abs(sum(sectors.values()) - 1.0) < 0.01, f"{sym} weights sum != 1"


# --- position_weights_from_decisions ---------------------------------------

def test_position_weights_split_equally_across_longs():
    d = _dec("momentum", {"AAPL": 5.0, "MSFT": 3.0, "GOOGL": 0.0}, weight=0.15)
    w = position_weights_from_decisions([d])
    assert abs(w["AAPL"] - 0.075) < 1e-9
    assert abs(w["MSFT"] - 0.075) < 1e-9
    assert "GOOGL" not in w


def test_position_weights_accumulate_across_sleeves():
    d1 = _dec("trend_following", {"XLK": 8.0}, weight=0.15)
    d2 = _dec("sector_rotation", {"XLK": 4.0, "XLE": 4.0}, weight=0.10)
    w = position_weights_from_decisions([d1, d2])
    assert abs(w["XLK"] - 0.20) < 1e-9   # 0.15 + 0.05
    assert abs(w["XLE"] - 0.05) < 1e-9


def test_position_weights_ignore_skipped_and_monitor_only():
    skipped = _dec("volatility", {"SVXY": 1.0}, weight=0.05, skipped=True)
    monitor = _dec("breakouts", {"TRGP": 2.0}, weight=0.0)
    assert position_weights_from_decisions([skipped, monitor]) == {}


# --- lookthrough_exposures --------------------------------------------------

def test_lookthrough_stacks_etf_and_stock():
    # XLK (pure tech) + QQQ (half tech) should stack in the technology bucket
    exp = lookthrough_exposures({"XLK": 0.15, "QQQ": 0.25})
    assert abs(exp["technology"] - (0.15 + 0.25 * 0.50)) < 1e-9


def test_lookthrough_unknown_symbol_is_visible():
    exp = lookthrough_exposures({"ZZZTEST": 0.10})
    assert abs(exp["unknown"] - 0.10) < 1e-9


# --- flag_concentration -----------------------------------------------------

def test_flag_concentration_orders_worst_first():
    exp = {"technology": 0.40, "energy": 0.35, "healthcare": 0.10}
    assert flag_concentration(exp, limit=0.30) == ["technology", "energy"]


def test_flag_concentration_empty_when_diversified():
    exp = {"technology": 0.20, "energy": 0.15}
    assert flag_concentration(exp, limit=0.30) == []


# --- markdown section -------------------------------------------------------

def test_sector_report_lists_sectors_and_flags():
    decisions = [
        _dec("trend_following", {"XLK": 8.0}, weight=0.15),
        _dec("long_term_etf", {"QQQ": 3.5}, weight=0.25),
        _dec("momentum", {"LLY": 0.4, "UNH": 1.2}, weight=0.15),
    ]
    md = _render_sector_exposure(decisions)
    assert "## Sector exposure (look-through)" in md
    assert "technology" in md
    assert "healthcare" in md
    # tech = 0.15 + 0.25*0.50 = 27.5% — under the 30% limit, no flag on it
    assert "27.5%" in md


def test_sector_report_flags_over_limit():
    decisions = [
        _dec("trend_following", {"XLK": 8.0}, weight=0.15),
        _dec("sector_rotation", {"XLK": 9.0}, weight=0.10),
        _dec("long_term_etf", {"QQQ": 3.5}, weight=0.25),
    ]
    # tech = 0.15 + 0.10 + 0.125 = 37.5% > 30%
    md = _render_sector_exposure(decisions)
    assert "over limit" in md


def test_sector_report_no_positions():
    md = _render_sector_exposure([_dec("momentum", {"AAPL": 0.0})])
    assert "No open positions." in md
