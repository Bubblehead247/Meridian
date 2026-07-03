"""Look-through sector exposure for the live paper book.

Maps each tradeable symbol to sector weights. Sector ETFs and single
stocks map 1.0 to one sector; broad ETFs (QQQ, SPY) carry an approximate
look-through so stacked exposure (e.g. QQQ + XLK holding the same
mega-caps) is visible instead of hiding behind a "broad market" label.

The weights are static approximations of fund composition, refreshed by
hand — good enough to flag concentration in the monthly review, not for
precise attribution. Symbols not in the map land in "unknown" so a gap
is visible rather than silently dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meridian.execution.live_runner import StrategyDecision

#: Flag any single sector above this share of account equity.
DEFAULT_SECTOR_LIMIT = 0.30

#: symbol → {sector: weight}. Weights within a symbol sum to ~1.0.
SECTOR_WEIGHTS: dict[str, dict[str, float]] = {
    # --- broad ETFs (approximate look-through) ---
    "QQQ": {
        "technology": 0.50, "communication": 0.16, "consumer_discretionary": 0.13,
        "healthcare": 0.06, "consumer_staples": 0.06, "industrials": 0.05,
        "other": 0.04,
    },
    "SPY": {
        "technology": 0.32, "financials": 0.13, "healthcare": 0.11,
        "consumer_discretionary": 0.10, "communication": 0.09, "industrials": 0.08,
        "consumer_staples": 0.06, "energy": 0.04, "other": 0.07,
    },
    # --- SPDR sector ETFs ---
    "XLC": {"communication": 1.0},
    "XLY": {"consumer_discretionary": 1.0},
    "XLP": {"consumer_staples": 1.0},
    "XLE": {"energy": 1.0},
    "XLF": {"financials": 1.0},
    "XLV": {"healthcare": 1.0},
    "XLI": {"industrials": 1.0},
    "XLB": {"materials": 1.0},
    "XLRE": {"real_estate": 1.0},
    "XLK": {"technology": 1.0},
    "XLU": {"utilities": 1.0},
    # --- non-equity sleeves ---
    "IEF": {"fixed_income": 1.0},
    "TLT": {"fixed_income": 1.0},
    "GLD": {"commodities": 1.0},
    "KMLM": {"managed_futures": 1.0},
    "PDBC": {"commodities": 1.0},
    "SVXY": {"volatility": 1.0},
    # --- momentum basket + other live-pick stocks ---
    "AAPL": {"technology": 1.0},
    "MSFT": {"technology": 1.0},
    "NVDA": {"technology": 1.0},
    "AVGO": {"technology": 1.0},
    "AMZN": {"consumer_discretionary": 1.0},
    "HD": {"consumer_discretionary": 1.0},
    "GOOGL": {"communication": 1.0},
    "META": {"communication": 1.0},
    "JPM": {"financials": 1.0},
    "MA": {"financials": 1.0},
    "V": {"financials": 1.0},
    "XOM": {"energy": 1.0},
    "LLY": {"healthcare": 1.0},
    "UNH": {"healthcare": 1.0},
    "PG": {"consumer_staples": 1.0},
    "SNOW": {"technology": 1.0},
    "TRGP": {"energy": 1.0},
    "WFRD": {"energy": 1.0},
}


def position_weights_from_decisions(decisions: list["StrategyDecision"]) -> dict[str, float]:
    """Account weight per held symbol, summed across sleeves.

    Sizing in the live runner splits a sleeve's weight equally across its long
    signals, so each held symbol carries ``sleeve_weight / n_long``. Symbols
    held by more than one sleeve accumulate.
    """
    out: dict[str, float] = {}
    for d in decisions:
        if d.skipped or d.weight <= 0:
            continue
        held = [s for s, sh in d.target_shares.items() if sh != 0]
        if not held:
            continue
        per_symbol = d.weight / len(held)
        for sym in held:
            out[sym] = out.get(sym, 0.0) + per_symbol
    return out


def lookthrough_exposures(position_weights: dict[str, float]) -> dict[str, float]:
    """Sector → share of account equity, looking through ETFs.

    Unmapped symbols land in ``"unknown"`` so gaps in SECTOR_WEIGHTS stay visible.
    """
    out: dict[str, float] = {}
    for sym, weight in position_weights.items():
        sectors = SECTOR_WEIGHTS.get(sym, {"unknown": 1.0})
        for sector, frac in sectors.items():
            out[sector] = out.get(sector, 0.0) + weight * frac
    return out


def flag_concentration(
    exposures: dict[str, float], limit: float = DEFAULT_SECTOR_LIMIT
) -> list[str]:
    """Sectors whose exposure exceeds ``limit``, worst first."""
    over = [s for s, w in exposures.items() if w > limit]
    return sorted(over, key=lambda s: exposures[s], reverse=True)
