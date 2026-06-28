"""Command-line interface for Meridian.

Run with **no command** for a guided interactive menu (the easy way to test symbols
against strategies)::

    meridian                       # or: python -m meridian

Or drive it directly::

    meridian gauntlet --symbol SPY              # rank every strategy on a symbol
    meridian gauntlet --symbols XLK XLF XLE     # rank cross-sectional models on a basket
    meridian pipeline mean_reversion/zscore_reversion --symbols SPY QQQ
    meridian fund --universe QQQ --equity 100000
    meridian list models
    meridian validate configs/validation_spy.yaml

Also runnable as ``python -m meridian ...``. Kept dependency-free (stdlib argparse)
per the project's simplicity rule.
"""

from __future__ import annotations

import argparse
import sys

from meridian.deviations import list_deviations
from meridian.estimators import list_estimators
from meridian.experiments import runner
from meridian.regimes import list_regimes


def _cmd_validate(args) -> int:
    cfg = runner.load_experiment(args.config)
    prices, bars = runner.load_prices(cfg)
    table, path = runner.run_validation_report(cfg, prices, bars)
    sig = table[table["significant"]]["estimator"].tolist()
    print(table.to_string(index=False))
    print(f"\nSignificant after correction: {sig or 'NONE'}")
    print(f"Report written to: {path}")
    return 0


def _cmd_backtest(args) -> int:
    cfg = runner.load_experiment(args.config)
    prices, bars = runner.load_prices(cfg)
    table = runner.run_validation(cfg, prices, bars)
    print(table.to_string(index=False))
    return 0


def _cmd_universe(args) -> int:
    cfg = runner.load_experiment(args.config)
    prices, bars = runner.load_universe_prices(cfg)
    table, path = runner.run_universe_validation_report(cfg, prices, bars)
    sig = table[table["significant"]]["estimator"].tolist()
    print(table.to_string(index=False))
    print(f"\nUniverse: {len(prices)} symbols")
    print(f"Significant after correction: {sig or 'NONE'}")
    print(f"Report written to: {path}")
    return 0


def _cmd_paper(args) -> int:
    cfg = runner.load_experiment(args.config)
    prices, bars = runner.load_prices(cfg)
    log, summary = runner.run_paper_dry_run(cfg, prices, bars)
    print(log.tail(10).to_string(index=False))
    print("\nSession summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def _cmd_list(args) -> int:
    if args.kind == "saved":
        from meridian.pipeline import load_records
        records = load_records()
        if not records:
            print("No saved strategies found (run 'meridian pipeline <model> --symbol <SYM>' first).")
            return 0
        print(f"{'#':>3}  {'family/model':<32}  {'symbol':<6}  {'stage':<12}  {'date':<10}  {'sharpe':>6}  {'cagr':>7}  {'mdd':>7}")
        print("-" * 98)
        for i, rec in enumerate(records, 1):
            qualified = f"{rec.family}/{rec.model}"
            sharpe = rec.scorecard.get("sharpe")
            cagr   = rec.scorecard.get("cagr")
            mdd    = rec.scorecard.get("max_drawdown")
            sh_s  = f"{sharpe:6.2f}" if isinstance(sharpe, float) and sharpe == sharpe else "   n/a"
            ca_s  = f"{cagr:6.1%}"   if isinstance(cagr,   float) and cagr   == cagr   else "    n/a"
            mdd_s = f"{mdd:6.1%}"    if isinstance(mdd,    float) and mdd    == mdd    else "    n/a"
            print(f"{i:>3}  {qualified:<32}  {rec.symbol:<6}  {rec.stage_passed:<12}  {rec.saved_at:<10}  {sh_s}  {ca_s}  {mdd_s}")
        return 0

    def _models():
        from meridian.families import list_models

        return list_models()

    catalogs = {
        "estimators": list_estimators,
        "deviations": list_deviations,
        "regimes": list_regimes,
        "models": _models,
    }
    names = catalogs[args.kind]()
    print(f"{len(names)} {args.kind}:")
    for n in names:
        print(f"  {n}")
    return 0


def _symbol_list(args, *, default_to_symbol: bool) -> list[str]:
    """Resolve which symbols to run: --universe NAME, or --symbols ..., or the single --symbol.

    ``--universe`` expands a named set (e.g. ``SP500``, ``QQQ``) via ``get_universe``.
    When nothing is given, single-asset commands fall back to ``--symbol`` (default SPY);
    cross-sectional commands return an empty list (a basket must be named explicitly).
    """
    if getattr(args, "universe", None):
        from meridian.data.universe import get_universe

        return list(get_universe(args.universe).symbols)
    if getattr(args, "symbols", None):
        return list(args.symbols)
    return [args.symbol] if default_to_symbol else []


def _cmd_pipeline(args) -> int:
    from meridian.data.loader import load_ohlcv
    from meridian.families import create_model

    family, _, name = args.model.partition("/")
    if not name:
        print("model must be given as '<family>/<model>'", file=sys.stderr)
        return 2
    try:
        model = create_model(family, name)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if getattr(model, "cross_sectional", False):
        symbols = _symbol_list(args, default_to_symbol=False)
        if len(symbols) < 2:
            print(f"{args.model} is cross-sectional; pass --symbols A B ... "
                  f"or --universe NAME (>= 2 symbols)", file=sys.stderr)
            return 2
        return _run_cross_sectional(model, args, load_ohlcv, symbols)

    rc = 0
    for symbol in _symbol_list(args, default_to_symbol=True):
        rc |= _run_single_asset(model, family, name, args, load_ohlcv, symbol)
    return rc


def _run_single_asset(model, family, name, args, load_ohlcv, symbol) -> int:
    from meridian.pipeline import record_from_pipeline, run_pipeline, save_record
    from meridian.portfolio import StrategyLedger

    frame = load_ohlcv(symbol, args.start)
    prices = frame["close"]
    ledger = StrategyLedger(name=name, family=family, stage="research")
    results = run_pipeline(
        model, prices, ledger=ledger, bars=frame,
        cost_bps=args.cost_bps, stop_on_fail=not args.all_stages,
    )
    print(f"{args.model} on {symbol}  ({len(prices)} bars)")
    for stage, res in results.items():
        sharpe = res.scorecard.get("sharpe")
        sharpe_s = f"{sharpe:6.2f}" if isinstance(sharpe, float) and sharpe == sharpe else "   n/a"
        print(f"  {stage:12s} passed={str(res.passed):5s} sharpe={sharpe_s}"
              f"  -> {res.detail.get('ledger_action')}")
    print(f"  final graduation stage: {ledger.stage}")
    rec = record_from_pipeline(family, name, symbol, results, ledger)
    if rec is not None:
        path = save_record(rec)
        print(f"  saved -> {path}")
    return 0


def _run_cross_sectional(model, args, load_ohlcv, symbols) -> int:
    from meridian.pipeline import record_from_pipeline, run_cross_sectional_pipeline, save_record
    from meridian.portfolio import StrategyLedger

    family, name = model.family, model.name
    universe = {s: load_ohlcv(s, args.start)["close"] for s in symbols}
    ledger = StrategyLedger(name=name, family=family, stage="research")
    results = run_cross_sectional_pipeline(
        model, universe, ledger=ledger, cost_bps=args.cost_bps,
        stop_on_fail=not args.all_stages,
    )
    symbol_str = "_".join(symbols)
    print(f"{family}/{name} on {symbol_str}  (cross-sectional, {len(universe)} symbols, {len(next(iter(universe.values())))} bars)")
    for stage, res in results.items():
        sharpe = res.scorecard.get("sharpe")
        sharpe_s = f"{sharpe:6.2f}" if isinstance(sharpe, float) and sharpe == sharpe else "   n/a"
        n_trades = res.scorecard.get("n_trades", res.detail.get("n_trades", 0))
        turn = res.scorecard.get("turnover")
        turn_s = f"{turn:.2f}x/yr" if isinstance(turn, float) and turn == turn else "n/a"
        print(f"  {stage:12s} passed={str(res.passed):5s} sharpe={sharpe_s}"
              f"  n_trades={n_trades}  turnover={turn_s}"
              f"  -> {res.detail.get('ledger_action')}")
    print(f"  final graduation stage: {ledger.stage}")
    rec = record_from_pipeline(family, name, symbol_str, results, ledger)
    if rec is not None:
        path = save_record(rec)
        print(f"  saved -> {path}")
    return 0


def _cmd_fund(args) -> int:
    """Run the whole fund lifecycle for each symbol and write a report per symbol."""
    from pathlib import Path

    from meridian.data.loader import load_ohlcv
    from meridian.experiments.fund import _first_single_asset, run_fund
    from meridian.regimes.labeler import build_regime_frame
    from meridian.reporting import render_monthly_report

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regime_frame = None
    try:
        regime_frame = build_regime_frame(args.start)
    except Exception as exc:
        print(f"  (regime frame unavailable — running without permission gates: {exc})")

    for symbol in _symbol_list(args, default_to_symbol=True):
        frame = load_ohlcv(symbol, args.start)
        prices = frame["close"]
        ledgers, review = run_fund(
            prices, frame, equity=args.equity, cost_bps=args.cost_bps, regime_frame=regime_frame
        )

        print(f"\nMeridian fund run — {symbol}, equity {args.equity:,.0f}, {len(prices)} bars")
        for led in ledgers:
            model_name = _first_single_asset(led.family)
            if model_name is None:                      # cash_reserve / experimental_research
                print(f"  {led.name:22s} (no strategy model — sleeve held)")
            else:
                print(f"  {led.name:22s} {model_name:22s} -> stage={led.stage}")

        path = out_dir / f"monthly_review_{symbol}_{review.as_of}.md"
        path.write_text(render_monthly_report(review), encoding="utf-8")
        print("  sleeve actions: " + ", ".join(f"{s.sleeve}={s.action}" for s in review.sleeves))
        print(f"  report: {path}")
    return 0


def _cmd_gauntlet(args) -> int:
    from meridian.data.loader import load_ohlcv
    from meridian.experiments.gauntlet import (
        format_gauntlet,
        gauntlet_single,
        gauntlet_universe,
    )

    symbols = _symbol_list(args, default_to_symbol=True)
    if len(symbols) > 1:
        universe = {s: load_ohlcv(s, args.start)["close"] for s in symbols}
        print(f"Gauntlet — cross-sectional models on {len(universe)} names")
        df = gauntlet_universe(universe, cost_bps=args.cost_bps)
    else:
        frame = load_ohlcv(symbols[0], args.start)
        print(f"Gauntlet — single-asset models on {symbols[0]} ({len(frame)} bars)")
        df = gauntlet_single(frame["close"], bars=frame, cost_bps=args.cost_bps)
    print(format_gauntlet(df))
    return 0


def _cmd_sweep(args) -> int:
    from meridian.data.loader import load_ohlcv
    from meridian.experiments.sweep import (
        format_confirm,
        format_sweep,
        sweep_and_confirm,
        sweep_single,
    )

    family, _, name = args.model.partition("/")
    if not name:
        print("model must be given as '<family>/<model>'", file=sys.stderr)
        return 2
    symbol = _symbol_list(args, default_to_symbol=True)[0]   # single-asset sweep on one symbol
    frame = load_ohlcv(symbol, args.start)
    print(f"Tuning {args.model} on {symbol} ({len(frame)} bars)\n")
    if args.confirm:
        result = sweep_and_confirm(frame["close"], family, name, bars=frame, cost_bps=args.cost_bps)
        print(format_confirm(result, family, name))
    else:
        df = sweep_single(frame["close"], family, name, bars=frame, cost_bps=args.cost_bps)
        print(format_sweep(df, family, name))
    return 0


def _cmd_run_paper(args) -> int:
    """Compute today's signals for every paper-stage strategy and (optionally) send orders."""
    from meridian.execution.broker import AlpacaBroker, SimulatedBroker
    from meridian.execution.live_runner import print_session_report, run_paper_session

    if args.dry_run:
        broker = SimulatedBroker(cash=args.equity)
    else:
        try:
            broker = AlpacaBroker(paper=not args.live)
        except (ImportError, ValueError) as exc:
            print(f"Broker init failed: {exc}", file=sys.stderr)
            print("Use --dry-run to preview signals without a broker connection.", file=sys.stderr)
            return 1

    decisions = run_paper_session(
        broker,
        account_equity=args.equity,
        price_start=args.start or "2023-01-01",
        dry_run=args.dry_run,
    )
    print_session_report(decisions)

    active = sum(1 for d in decisions if not d.skipped)
    orders = sum(len(d.orders) for d in decisions)
    print(f"Done: {active} strategies processed, {orders} orders {'previewed' if args.dry_run else 'sent'}.")
    return 0


def _cmd_menu(args) -> int:
    from meridian.interactive import run_menu

    return run_menu()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meridian",
        description="Quant research platform — run with no command for an interactive menu",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    for name, fn, help_ in [
        ("validate", _cmd_validate, "Walk-forward validate estimators and write a report"),
        ("backtest", _cmd_backtest, "Run validation and print the ranked table (no report)"),
        ("universe", _cmd_universe, "Cross-sectional universe-wide validation across many symbols"),
        ("paper", _cmd_paper, "Paper-trade dry-run: warm on history, replay the rest"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", help="Path to a YAML experiment config")
        p.set_defaults(func=fn)

    pl = sub.add_parser("list", help="List available estimators/deviations/regimes/models")
    pl.add_argument("kind", choices=["estimators", "deviations", "regimes", "models", "saved"])
    pl.set_defaults(func=_cmd_list)

    pp = sub.add_parser(
        "pipeline", help="Drive a family model through the backtest/walk-forward/OOS stages"
    )
    pp.add_argument("model", help="Model as '<family>/<model>' (see `meridian list models`)")
    pp.add_argument("--symbol", default="SPY", help="Single symbol (single-asset model)")
    pp.add_argument("--symbols", nargs="*", help="Several symbols (a basket / multiple runs)")
    pp.add_argument("--universe", default=None,
                    help="Named universe (e.g. SP500, NASDAQ100, QQQ) instead of listing symbols")
    pp.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    pp.add_argument("--cost-bps", type=float, default=1.0, dest="cost_bps")
    pp.add_argument("--all-stages", action="store_true",
                    help="Run every stage even after one fails")
    pp.set_defaults(func=_cmd_pipeline)

    fp = sub.add_parser(
        "fund", help="Run the whole fund lifecycle and write a month-end report"
    )
    fp.add_argument("--symbol", default="SPY", help="Symbol to run the sleeve models on")
    fp.add_argument("--symbols", nargs="*", help="Run the fund on each of several symbols")
    fp.add_argument("--universe", default=None,
                    help="Run the fund on each symbol of a named universe (e.g. QQQ, SP500)")
    fp.add_argument("--equity", type=float, default=100_000.0, help="Total account equity")
    fp.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    fp.add_argument("--cost-bps", type=float, default=1.0, dest="cost_bps")
    fp.add_argument("--report-dir", default="reports", dest="report_dir")
    fp.set_defaults(func=_cmd_fund)

    gp = sub.add_parser("gauntlet", help="Rank every strategy on a symbol/basket (scoreboard)")
    gp.add_argument("--symbol", default="SPY", help="Single symbol (single-asset gauntlet)")
    gp.add_argument("--symbols", nargs="*", help="A basket (cross-sectional gauntlet)")
    gp.add_argument("--universe", default=None, help="Named universe (e.g. SP500)")
    gp.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    gp.add_argument("--cost-bps", type=float, default=1.0, dest="cost_bps")
    gp.set_defaults(func=_cmd_gauntlet)

    sp = sub.add_parser(
        "sweep", help="Tune a strategy's parameters, scored out-of-sample (honest)"
    )
    sp.add_argument("model", help="Model as '<family>/<model>'")
    sp.add_argument("--symbol", default="SPY", help="Symbol to tune on")
    sp.add_argument("--symbols", nargs="*", help="(first is used for the single-asset sweep)")
    sp.add_argument("--universe", default=None, help="(first member is used)")
    sp.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    sp.add_argument("--cost-bps", type=float, default=1.0, dest="cost_bps")
    sp.add_argument("--confirm", action="store_true",
                    help="Auto-confirm the winning setting through the full fixed-OOS holdout")
    sp.set_defaults(func=_cmd_sweep)

    rp = sub.add_parser(
        "run-paper", help="Compute today's signals for all paper-stage strategies and send orders"
    )
    rp.add_argument("--equity", type=float, default=100_000.0,
                    help="Total account equity for position sizing")
    rp.add_argument("--start", default=None,
                    help="Earliest date to fetch for indicator warmup (default: 2023-01-01)")
    rp.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Compute signals and sizes but do not send orders to the broker")
    rp.add_argument("--live", action="store_true",
                    help="Use the live Alpaca endpoint instead of paper (default: paper)")
    rp.set_defaults(func=_cmd_run_paper)

    mp = sub.add_parser("menu", help="Launch the interactive menu (same as no command)")
    mp.set_defaults(func=_cmd_menu)

    return parser


def main(argv: list[str] | None = None) -> int:
    from urllib.error import URLError

    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:                              # no subcommand -> interactive menu
        from meridian.interactive import run_menu

        return run_menu()
    try:
        return func(args)
    except URLError as exc:                       # network/data fetch failed (e.g. 403, offline)
        print(f"data fetch failed ({exc}). Check your connection, or pass explicit "
              f"--symbols instead of --universe.", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:         # unknown universe/model/etc.
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
