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
from meridian.experiments.run_log import append_run, new_run_record
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


def resolve_symbols(cfg: dict, screener=None) -> list[str]:
    """Resolve the trading symbols from a config.

    Priority: ``data.screen`` (build via FinanceDatabase screening) > ``data.universe``
    (a named universe, e.g. ``SP500``/``QQQ`` — routes through ``get_universe`` so index
    universes get the same survivorship-bias gate the CLI's ``--universe`` flag uses,
    since this is the entry point the config-driven ``meridian validate/backtest/universe/
    paper <config.yaml>`` commands go through instead of the CLI's own ``--universe``
    handling) > the explicit ``data.symbols`` (or single ``data.symbol``). The optional
    ``limit`` caps the screened list. ``screener`` may be injected for testing.
    """
    data = cfg.get("data", {})
    screen = data.get("screen")
    universe_name = data.get("universe")
    if universe_name:
        from meridian.data.universe import SurvivorshipBiasError, get_universe

        try:
            return list(get_universe(
                universe_name,
                accept_survivorship_bias=bool(data.get("accept_survivorship_bias", False)),
            ).symbols)
        except SurvivorshipBiasError as exc:
            raise SurvivorshipBiasError(
                f"{exc} Set data.accept_survivorship_bias: true in the config to proceed anyway."
            ) from None
    if screen:
        if screener is None:
            from meridian.data import EquityScreener

            screener = EquityScreener()
        filters = dict(screen)
        limit = filters.pop("limit", None)
        opts = {
            k: filters.pop(k)
            for k in ("only_primary_listing", "yfinance_safe")
            if k in filters
        }
        universe = screener.build_universe("screened", **opts, **filters)
        symbols = list(universe.symbols)
        if limit is not None:
            symbols = symbols[: int(limit)]
        if not symbols:
            raise ValueError("config data.screen matched no symbols")
        return symbols
    return data.get("symbols") or [data.get("symbol", "SPY")]


def build_broker(cfg: dict):
    """Construct the broker named in the config's ``broker`` section."""
    b = cfg.get("broker", {"type": "simulated"})
    kind = b.get("type", "simulated")
    if kind == "simulated":
        return SimulatedBroker(
            cash=float(b.get("cash", 100_000)), cost_bps=float(b.get("cost_bps", 1.0))
        )
    if kind == "alpaca":
        return AlpacaBroker(api_key=b.get("api_key"), secret_key=b.get("secret_key"),
                            paper=bool(b.get("paper", True)))
    raise ValueError(f"unknown broker type {kind!r}")


# --- data -----------------------------------------------------------------

def load_prices(cfg: dict, cache=None) -> tuple[pd.Series, pd.DataFrame]:
    """Load OHLCV for the config's first symbol (network). Returns (prices, bars)."""
    from meridian.data import load_ohlcv

    data = cfg.get("data", {})
    symbol = resolve_symbols(cfg)[0]
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
        sensitivity=bool(v.get("sensitivity", False)),
    )


def run_validation_report(
    cfg: dict, prices: pd.Series, bars: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, str]:
    """Run validation and write a markdown report; return (table, report_path)."""
    from meridian.experiments.run_log import cumulative_trial_count

    table = run_validation(cfg, prices, bars)
    data = cfg.get("data", {})
    symbols = data.get("symbols") or [data.get("symbol", "?")]
    estimators = resolve_estimators(cfg)
    meta = {
        "symbols": symbols,
        "start": data.get("start"),
        "end": data.get("end"),
        "deviation": cfg.get("deviation", "zscore"),
        "window": cfg.get("window", 20),
        "cost_bps": cfg.get("cost_bps", 1.0),
        "wfo": build_wfo(cfg).mode,
        "correction": cfg.get("validation", {}).get("correction", "bh"),
        "estimators": estimators,
        "scope": sorted(symbols),
    }
    out = cfg.get("report", {}).get("path", "reports/validation.md")
    log_path = Path(out).parent / "run_log.jsonl"
    sig = table[table["significant"]]["estimator"].tolist() if "significant" in table.columns else []
    append_run(new_run_record(
        config_path=cfg.get("_source_path", "?"),
        raw_config_text=cfg.get("_source_text"),
        kind="validation",
        meta=meta,
        report_path=out,
        summary={"n_tested": len(table), "n_significant": len(sig), "significant": sig},
    ), path=log_path)
    # Read back the log *after* appending, so this run's own trials are included in
    # the cumulative count the report discloses (P1-A) — this run's estimator list is
    # not a fresh, unpenalized N=len(estimators) look at the data if the same symbols
    # were already tested in a prior logged run.
    meta["cumulative_trial_count"] = cumulative_trial_count(log_path, scope=meta["scope"])
    md = build_validation_report(table, meta)
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


def is_survivorship_biased(cfg: dict) -> bool:
    """Whether the universe ``load_universe_prices`` would resolve is survivorship-biased.

    True for the default yfinance/Wikipedia path (always — see
    ``data/universe.py``'s docstring). True even for ``data.source ==
    "survivorship"`` when ``variant == "survivor"``, since that variant is
    itself the biased end-of-period baseline used for bias-quantification, not
    a bias-free universe. Only ``source: survivorship`` with the default
    ``variant: free`` is genuinely point-in-time.
    """
    data = cfg.get("data", {})
    if data.get("source") != "survivorship":
        return True
    return data.get("variant", "free") == "survivor"


def load_universe_prices(cfg: dict, cache=None) -> tuple[dict, dict]:
    """Load prices for every symbol in the config for the universe-wide study.

    Returns ``(prices_by_symbol, bars_by_symbol)``. ``data.source`` selects the
    provider: ``yfinance`` (default) or ``survivorship`` (a local point-in-time
    survivorship-bias-free dataset).
    """
    data = cfg.get("data", {})
    if data.get("source") == "survivorship":
        return _load_survivorship_universe(data)

    from meridian.data import load_universe

    symbols = resolve_symbols(cfg)
    frames = load_universe(
        symbols, start=data.get("start"), end=data.get("end"),
        interval=data.get("interval", "1d"), cache=cache,
    )
    field = data.get("field", "adj_close")
    prices = {s: f[field] for s, f in frames.items()}
    return prices, dict(frames)


def _load_survivorship_universe(data: dict) -> tuple[dict, dict]:
    """Build a universe from the local survivorship-bias-free dataset.

    Config keys under ``data``: ``root`` (dataset path), ``variant``
    (``free`` = point-in-time/bias-free, default, or ``survivor`` = the biased
    end-of-period baseline), ``start``/``end``, ``field`` (default ``close``).
    Prices only — no OHLC bars (the dataset is close-oriented), so ``atr_norm``
    is unavailable here.
    """
    from meridian.data import SurvivorshipDataset

    if "root" not in data:
        raise ValueError("survivorship source requires data.root (dataset path)")
    ds = SurvivorshipDataset(data["root"])
    variant = data.get("variant", "free")
    start, end, field = data.get("start"), data.get("end"), data.get("field", "close")
    if variant == "free":
        prices = ds.survivorship_free_universe(start, end, field=field)
    elif variant == "survivor":
        prices = ds.survivor_only_universe(start, end, field=field)
    else:
        raise ValueError(f"unknown survivorship variant {variant!r} (use 'free' or 'survivor')")
    return prices, {}


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
    from meridian.experiments.run_log import cumulative_trial_count

    table = run_universe_validation(cfg, prices_by_symbol, bars_by_symbol)
    data = cfg.get("data", {})
    estimators = resolve_estimators(cfg)
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
        "survivorship_biased": is_survivorship_biased(cfg),
        "estimators": estimators,
        "scope": sorted(prices_by_symbol),
    }
    if data.get("source") == "survivorship":
        from meridian.data.survivorship import coverage_warning

        warning = coverage_warning(data.get("start"), data.get("end"))
        if warning is not None:
            meta["survivorship_coverage_warning"] = warning
    out = cfg.get("report", {}).get("path", "reports/universe_validation.md")
    log_path = Path(out).parent / "run_log.jsonl"
    sig = table[table["significant"]]["estimator"].tolist() if "significant" in table.columns else []
    append_run(new_run_record(
        config_path=cfg.get("_source_path", "?"),
        raw_config_text=cfg.get("_source_text"),
        kind="universe_validation",
        meta=meta,
        report_path=out,
        summary={"n_tested": len(table), "n_significant": len(sig), "significant": sig},
    ), path=log_path)
    meta["cumulative_trial_count"] = cumulative_trial_count(log_path, scope=meta["scope"])
    md = build_validation_report(
        table, meta, title="Meridian — Universe-Wide Mean-Reversion Validation"
    )
    write_report(out, md)
    return table, out


def load_experiment(path: str | Path) -> dict:
    """Load a YAML experiment config (thin wrapper over `config.load_config`).

    Stashes the source path and raw text under ``_source_path``/``_source_text``
    so downstream report/run-log writers can record exactly what was loaded,
    without changing every function's signature to thread a path through.
    """
    cfg = load_config(path)
    p = Path(path)
    cfg["_source_path"] = str(p)
    try:
        cfg["_source_text"] = p.read_text(encoding="utf-8")
    except OSError:
        cfg["_source_text"] = None
    return cfg
