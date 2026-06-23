"""Config-driven experiment runner.

Turns a YAML config into a run: validation, backtest, or a paper-trading
dry-run. This is the "reproducible from config alone" principle made concrete —
every knob (data, estimators, deviation, signal, regime, walk-forward, broker)
comes from the config, nothing is hard-coded here.

Data loading (network) is kept separate from the pure run functions so the
latter can be tested offline with an injected price series.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from meridian.analytics import build_validation_report, write_report
from meridian.config import load_config
from meridian.estimators import list_estimators
from meridian.execution.broker import AlpacaBroker, SimulatedBroker
from meridian.execution.trader import PaperTrader
from meridian.signals import SignalConfig
from meridian.validation import WalkForwardSpec, validate


# --- config -> objects ----------------------------------------------------

def build_signal(cfg: dict) -> SignalConfig:
    return SignalConfig.from_config(cfg.get("signal", {}))


def build_wfo(cfg: dict) -> WalkForwardSpec:
    v = cfg.get("validation", {})
    return WalkForwardSpec(
        mode=v.get("mode", "anchored"),
        train_span=int(v.get("train_span", 756)),
        test_span=int(v.get("test_span", 126)),
        step=int(v.get("step", 126)),
        min_train=int(v.get("min_train", 756)),
    )


def resolve_estimators(cfg: dict) -> list[str]:
    """Estimator list from config: an explicit list, or 'all' for the registry."""
    spec = cfg.get("estimators", ["sma"])
    if spec == "all":
        return list_estimators()
    if isinstance(spec, str):
        return [spec]
    return list(spec)


def build_broker(cfg: dict):
    """Construct the broker named in the config's ``broker`` section."""
    b = cfg.get("broker", {"type": "simulated"})
    kind = b.get("type", "simulated")
    if kind == "simulated":
        return SimulatedBroker(cash=float(b.get("cash", 100_000)), cost_bps=float(b.get("cost_bps", 1.0)))
    if kind == "alpaca":
        return AlpacaBroker(api_key=b.get("api_key"), secret_key=b.get("secret_key"),
                            paper=bool(b.get("paper", True)))
    raise ValueError(f"unknown broker type {kind!r}")


# --- data -----------------------------------------------------------------

def load_prices(cfg: dict, cache=None) -> tuple[pd.Series, pd.DataFrame]:
    """Load OHLCV for the config's first symbol (network). Returns (prices, bars)."""
    from meridian.data import load_ohlcv

    data = cfg.get("data", {})
    symbols = data.get("symbols") or [data.get("symbol", "SPY")]
    symbol = symbols[0]
    bars = load_ohlcv(
        symbol, start=data.get("start"), end=data.get("end"),
        interval=data.get("interval", "1d"), cache=cache,
    )
    field = data.get("field", "adj_close")
    return bars[field], bars


# --- pure run functions (no network) --------------------------------------

def run_validation(cfg: dict, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the validation pipeline described by ``cfg`` on ``prices``."""
    v = cfg.get("validation", {})
    return validate(
        prices,
        resolve_estimators(cfg),
        cfg.get("deviation", "zscore"),
        build_signal(cfg),
        spec=build_wfo(cfg),
        window=int(cfg.get("window", 20)),
        cost_bps=float(cfg.get("cost_bps", 1.0)),
        bars=bars,
        n_boot=int(v.get("n_boot", 2000)),
        n_mc=int(v.get("n_mc", 2000)),
        method=v.get("correction", "bh"),
        seed=int(cfg.get("experiment", {}).get("seed", 0)),
    )


def run_validation_report(
    cfg: dict, prices: pd.Series, bars: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, str]:
    """Run validation and write a markdown report; return (table, report_path)."""
    table = run_validation(cfg, prices, bars)
    data = cfg.get("data", {})
    meta = {
        "symbols": (data.get("symbols") or [data.get("symbol", "?")]),
        "start": data.get("start"),
        "end": data.get("end"),
        "deviation": cfg.get("deviation", "zscore"),
        "window": cfg.get("window", 20),
        "cost_bps": cfg.get("cost_bps", 1.0),
        "wfo": build_wfo(cfg).mode,
        "correction": cfg.get("validation", {}).get("correction", "bh"),
    }
    md = build_validation_report(table, meta)
    out = cfg.get("report", {}).get("path", "reports/validation.md")
    write_report(out, md)
    return table, out


def run_paper_dry_run(
    cfg: dict, prices: pd.Series, bars: pd.DataFrame | None = None, warm_frac: float = 0.5
) -> tuple[pd.DataFrame, dict]:
    """Paper-trade a dry-run: warm on the first part of history, replay the rest.

    Returns the decision log and an account summary. Live trading would instead
    call `trader.on_bar` per new bar from a data feed (Phase 10 limitation).
    """
    split = int(len(prices) * warm_frac)
    warm, live = prices.iloc[:split], prices.iloc[split:]
    live_bars = None if bars is None else bars.iloc[split:]

    broker = build_broker(cfg)
    trader = PaperTrader(
        symbol=(cfg.get("data", {}).get("symbols") or ["SPY"])[0],
        broker=broker,
        estimator=cfg.get("estimator", "ou"),
        deviation=cfg.get("deviation", "zscore"),
        signal=build_signal(cfg),
        window=int(cfg.get("window", 20)),
        position_size=float(cfg.get("position_size", 1.0)),
        regime=cfg.get("regime"),
        allowed_regimes=tuple(cfg.get("allowed_regimes", ())),
    )
    trader.warm_up(warm, bars=None if bars is None else bars.iloc[:split])
    log = trader.replay(live, bars=live_bars)

    summary = {
        "bars": len(live),
        "orders": int((log["order_qty"] != 0).sum()),
        "final_position": broker.get_position(trader.symbol),
        "equity": broker.equity() if hasattr(broker, "equity") else None,
    }
    return log, summary


def load_universe_prices(cfg: dict, cache=None) -> tuple[dict, dict]:
    """Load OHLCV for every symbol in the config (network).

    Returns ``(prices_by_symbol, bars_by_symbol)`` for the universe-wide study.
    """
    from meridian.data import load_universe

    data = cfg.get("data", {})
    symbols = data.get("symbols") or [data.get("symbol", "SPY")]
    frames = load_universe(
        symbols, start=data.get("start"), end=data.get("end"),
        interval=data.get("interval", "1d"), cache=cache,
    )
    field = data.get("field", "adj_close")
    prices = {s: f[field] for s, f in frames.items()}
    return prices, dict(frames)


def run_universe_validation(
    cfg: dict, prices_by_symbol: dict, bars_by_symbol: dict | None = None
) -> pd.DataFrame:
    """Run the cross-sectional universe validation described by ``cfg``."""
    from meridian.portfolio import validate_universe

    v = cfg.get("validation", {})
    return validate_universe(
        prices_by_symbol,
        resolve_estimators(cfg),
        cfg.get("deviation", "zscore"),
        build_signal(cfg),
        sizing=cfg.get("sizing", "equal_weight"),
        spec=build_wfo(cfg),
        window=int(cfg.get("window", 20)),
        cost_bps=float(cfg.get("cost_bps", 1.0)),
        bars_by_symbol=bars_by_symbol,
        n_boot=int(v.get("n_boot", 2000)),
        n_mc=int(v.get("n_mc", 2000)),
        block=int(v.get("block", 20)),
        method=v.get("correction", "bh"),
        seed=int(cfg.get("experiment", {}).get("seed", 0)),
    )


def run_universe_validation_report(
    cfg: dict, prices_by_symbol: dict, bars_by_symbol: dict | None = None
) -> tuple[pd.DataFrame, str]:
    """Run universe validation and write a markdown report; return (table, path)."""
    table = run_universe_validation(cfg, prices_by_symbol, bars_by_symbol)
    data = cfg.get("data", {})
    meta = {
        "symbols": f"{len(prices_by_symbol)} names",
        "start": data.get("start"),
        "end": data.get("end"),
        "deviation": cfg.get("deviation", "zscore"),
        "sizing": cfg.get("sizing", "equal_weight"),
        "window": cfg.get("window", 20),
        "cost_bps": cfg.get("cost_bps", 1.0),
        "wfo": build_wfo(cfg).mode,
        "correction": cfg.get("validation", {}).get("correction", "bh"),
    }
    md = build_validation_report(
        table, meta, title="Meridian — Universe-Wide Mean-Reversion Validation"
    )
    out = cfg.get("report", {}).get("path", "reports/universe_validation.md")
    write_report(out, md)
    return table, out


def load_experiment(path: str | Path) -> dict:
    """Load a YAML experiment config (thin wrapper over `config.load_config`)."""
    return load_config(path)
