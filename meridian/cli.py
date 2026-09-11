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
from pathlib import Path

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
    if args.kind == "runs":
        from meridian.experiments.run_log import read_runs

        runs = read_runs()
        if not runs:
            print("No logged runs found (run 'meridian validate <config>' first).")
            return 0
        print(
            f"{'#':>3}  {'timestamp':<20}  {'kind':<18}  {'config':<28}  "
            f"{'sig/tested':<10}  {'git':<8}  report"
        )
        print("-" * 110)
        for i, r in enumerate(runs, 1):
            n_sig = r.summary.get("n_significant", "?")
            n_tested = r.summary.get("n_tested", "?")
            git = (r.git_sha or "")[:8] or "n/a"
            print(
                f"{i:>3}  {r.timestamp[:19]:<20}  {r.kind:<18}  "
                f"{Path(r.config_path).name:<28}  {f'{n_sig}/{n_tested}':<10}  "
                f"{git:<8}  {r.report_path}"
            )
        return 0

    if args.kind == "saved":
        from meridian.pipeline import load_records
        records = load_records()
        if not records:
            print(
                "No saved strategies found "
                "(run 'meridian pipeline <model> --symbol <SYM>' first)."
            )
            return 0
        print(
            f"{'#':>3}  {'family/model':<32}  {'symbol':<6}  {'stage':<12}  "
            f"{'date':<10}  {'sharpe':>6}  {'cagr':>7}  {'mdd':>7}"
        )
        print("-" * 98)
        for i, rec in enumerate(records, 1):
            qualified = f"{rec.family}/{rec.model}"
            sharpe = rec.scorecard.get("sharpe")
            cagr   = rec.scorecard.get("cagr")
            mdd    = rec.scorecard.get("max_drawdown")
            sh_s  = f"{sharpe:6.2f}" if isinstance(sharpe, float) and sharpe == sharpe else "   n/a"
            ca_s  = f"{cagr:6.1%}" if isinstance(cagr, float) and cagr == cagr else "    n/a"
            mdd_s = f"{mdd:6.1%}" if isinstance(mdd, float) and mdd == mdd else "    n/a"
            print(
                f"{i:>3}  {qualified:<32}  {rec.symbol:<6}  {rec.stage_passed:<12}  "
                f"{rec.saved_at:<10}  {sh_s}  {ca_s}  {mdd_s}"
            )
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

    Index universes (SP500/NASDAQ100/RUSSELL1000) are resolved from *today's*
    constituent list — there is no historical membership data behind them — so using
    one for a backtest/pipeline/gauntlet/sweep run is both survivorship-biased and
    look-ahead-biased (a 2015 backtest gets 2026's index). ``get_universe`` itself
    enforces this (``SurvivorshipBiasError``, a ``ValueError``); this wrapper just
    translates ``--accept-survivorship-bias`` into the API's own gate rather than
    duplicating the check.
    """
    if getattr(args, "universe", None):
        from meridian.data.universe import SurvivorshipBiasError, get_universe

        try:
            return list(get_universe(
                args.universe,
                accept_survivorship_bias=getattr(args, "accept_survivorship_bias", False),
            ).symbols)
        except SurvivorshipBiasError as exc:
            raise SurvivorshipBiasError(
                f"{exc} Pass --accept-survivorship-bias to proceed anyway."
            ) from None
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
        model, prices, ledger=ledger, bars=frame, symbol=symbol,
        cost_bps=args.cost_bps, stop_on_fail=not args.all_stages,
    )
    print(f"{args.model} on {symbol}  ({len(prices)} bars)")
    for stage, res in results.items():
        sharpe = res.scorecard.get("sharpe")
        sharpe_s = f"{sharpe:6.2f}" if isinstance(sharpe, float) and sharpe == sharpe else "   n/a"
        print(f"  {stage:12s} passed={str(res.passed):5s} sharpe={sharpe_s}"
              f"  -> {res.detail.get('ledger_action')}")
        run_count = res.detail.get("oos_run_count")
        if run_count and run_count > 1:
            print(f"  WARNING: this is OOS evaluation #{run_count} against this holdout "
                  f"for {family}/{name} on {symbol} — a prior pass may not be a first look")
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
    frames = {s: load_ohlcv(s, args.start) for s in symbols}
    universe = {s: f["close"] for s, f in frames.items()}
    bars_by_symbol = {s: f for s, f in frames.items() if "open" in f.columns}
    ledger = StrategyLedger(name=name, family=family, stage="research")
    symbol_str = "_".join(symbols)
    results = run_cross_sectional_pipeline(
        model, universe, ledger=ledger, cost_bps=args.cost_bps,
        bars_by_symbol=bars_by_symbol or None, stop_on_fail=not args.all_stages,
        basket=symbol_str,
    )
    n_bars = len(next(iter(universe.values())))
    print(
        f"{family}/{name} on {symbol_str}  "
        f"(cross-sectional, {len(universe)} symbols, {n_bars} bars)"
    )
    for stage, res in results.items():
        sharpe = res.scorecard.get("sharpe")
        sharpe_s = f"{sharpe:6.2f}" if isinstance(sharpe, float) and sharpe == sharpe else "   n/a"
        n_trades = res.scorecard.get("n_trades", res.detail.get("n_trades", 0))
        turn = res.scorecard.get("turnover")
        turn_s = f"{turn:.2f}x/yr" if isinstance(turn, float) and turn == turn else "n/a"
        print(f"  {stage:12s} passed={str(res.passed):5s} sharpe={sharpe_s}"
              f"  n_trades={n_trades}  turnover={turn_s}"
              f"  -> {res.detail.get('ledger_action')}")
        if res.detail.get("fill_realism") == "close_approx":
            print("  WARNING: fills used same-bar close (no open data) — Sharpe is inflated")
        run_count = res.detail.get("oos_run_count")
        if run_count and run_count > 1:
            print(f"  WARNING: this is OOS evaluation #{run_count} against this holdout "
                  f"for {family}/{name} on {symbol_str} — a prior pass may not be a first look")
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
        frames = {s: load_ohlcv(s, args.start) for s in symbols}
        universe = {s: f["close"] for s, f in frames.items()}
        bars_by_symbol = {s: f for s, f in frames.items() if "open" in f.columns}
        print(f"Gauntlet — cross-sectional models on {len(universe)} names")
        df = gauntlet_universe(universe, cost_bps=args.cost_bps, bars_by_symbol=bars_by_symbol or None)
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


def _cmd_review(args) -> int:
    """Build and write the monthly review from all paper-stage strategies."""
    from pathlib import Path

    from meridian.reporting.review_runner import build_paper_review, render_full_review

    print("Building monthly review...")
    review, inventory_md = build_paper_review(
        equity=args.equity,
        price_start=args.start or "2023-01-01",
        lookback_days=args.lookback,
    )

    full_md = render_full_review(review, inventory_md)

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"monthly_review_{review.as_of}.md"
    path.write_text(full_md, encoding="utf-8")

    print(f"Report: {path}")
    print(f"Benchmark (SPY {args.lookback}d): {review.benchmark_return:+.2%}")
    if review.regime:
        print(
            f"Regime: trend={review.regime[0]}  vol={review.regime[1]}  "
            f"breadth={review.regime[2]}"
        )
    print(f"Portfolio heat: {review.portfolio_heat:.2%}"
          + (" !! breached" if review.heat_breached else ""))
    print()
    print(f"{'Sleeve':<25} {'Action':<10} {'Return':>8} {'vs SPY':>8} {'DD':>7} {'OK':>4}")
    print("-" * 68)
    for s in review.sleeves:
        ok = "Y" if s.permitted else "N"
        print(f"  {s.sleeve:<23} {s.action:<10} "
              f"{s.return_pct:>7.1%} {s.excess_vs_benchmark:>+8.1%} "
              f"{s.drawdown_cur:>6.1%} {ok:>4}")
    return 0


#: Lock name held for the duration of a live paper session. Guards every route
#: into a session at once — the scheduler, the tray's "Run now", and a manual
#: `meridian run-paper` — since any two of them running together sends every
#: order twice.
SESSION_LOCK = "Meridian-session"

#: Used only when the account value cannot be read (dry runs, or a broker that
#: does not report equity). A live session sizes off the real account instead.
DEFAULT_EQUITY = 10_000.0


def _cmd_run_paper(args) -> int:
    """Compute today's signals for every paper-stage strategy and (optionally) send orders."""
    # A dry run is a local preview: no calendar gate, no lock, no alerting.
    if not args.dry_run:
        return _as_scheduled_job("session", lambda: _run_paper(args))
    return _run_paper(args)


#: Where the session report goes. The same file the old tray loop wrote.
SESSION_LOG = Path(__file__).resolve().parents[1] / "meridian.log"


def _log_session_to_file(job: str) -> None:
    """Send this run's output to meridian.log as well as stdout.

    Only the tray scheduler loop ever configured this, so when the jobs moved to
    Task Scheduler the log simply stopped: `meridian.log` ends at 2026-07-29
    15:31 and every session since has left no readable trace. The shared job log
    records that a job ran, but not the session report — which strategies fired,
    what the targets were, which orders went out.

    That detail is not a nicety. Reconstructing which sleeve owned each of the 17
    fills missing from the trade ledger (O4) was only possible *because* this file
    existed for June and July. Losing it makes the same recovery impossible next
    time.
    """
    import logging

    root = logging.getLogger()
    if any(getattr(h, "_meridian_session", False) for h in root.handlers):
        return  # already attached (a second job in the same process)

    handler = logging.FileHandler(SESSION_LOG, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s"))
    handler._meridian_session = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("meridian.session").info(f"--- {job} starting ---")

    # The session report is printed, not logged, so tee stdout into the file too.
    _tee_stdout(handler.stream)


class _Tee:
    """Write to two streams at once. Used to keep printed output in the log."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    # The secondary is a logging handler's stream, and logging closes its
    # handlers at interpreter shutdown — before the final stdout flush. Writing
    # to the log is best-effort for exactly that window: losing the last line of
    # a report is a fine trade for never crashing a session on the way out.
    def write(self, text: str) -> int:
        try:
            self._secondary.write(text)
            self._secondary.flush()
        except (ValueError, OSError):
            pass
        return self._primary.write(text)

    def flush(self) -> None:
        self._primary.flush()
        try:
            self._secondary.flush()
        except (ValueError, OSError):
            pass

    def __getattr__(self, name):
        return getattr(self._primary, name)


def _tee_stdout(stream) -> None:
    if getattr(sys.stdout, "_is_meridian_tee", False):
        return
    tee = _Tee(sys.stdout, stream)
    tee._is_meridian_tee = True  # type: ignore[attr-defined]
    sys.stdout = tee


def _as_scheduled_job(name: str, work) -> int:
    """Run ``work`` through the shared scheduled-job harness.

    Gives a task-scheduler run the same guards every other bot's jobs get: one at
    a time, a trading-day gate that fails closed, a staleness check, and an alert
    on failure. Meridian previously had no calendar gate on the CLI at all — the
    check lived only in the tray's loop — so a task firing on a holiday would have
    run a full session.
    """
    from quantcore.bot_schedule import for_bot
    from quantcore.jobs import JobSpec, run_job

    _log_session_to_file(name)

    spec = next((j for j in for_bot("meridian") if j.name == name), None)
    if spec is None:  # pragma: no cover - defensive
        return work()
    return run_job(
        JobSpec(bot=spec.bot, name=spec.name, scheduled=spec.at,
                max_lateness=spec.max_lateness,
                needs_trading_day=spec.needs_trading_day),
        work,
    )


def _run_paper(args) -> int:
    from quantcore.single_instance import SingleInstance

    from meridian.execution.broker import AlpacaBroker, SimulatedBroker
    from meridian.execution.live_runner import print_session_report, run_paper_session

    equity = args.equity if args.equity is not None else DEFAULT_EQUITY

    if args.dry_run:
        broker = SimulatedBroker(cash=equity)
    else:
        try:
            broker = AlpacaBroker(paper=not args.live)
        except (ImportError, ValueError) as exc:
            print(f"Broker init failed: {exc}", file=sys.stderr)
            print("Use --dry-run to preview signals without a broker connection.", file=sys.stderr)
            return 1

        # Size off what the account actually holds. The old default sized every
        # sleeve against a flat $10,000 while the account held ~$9,730, and the
        # gap widened with every day of P&L. An explicit --equity still wins, so
        # a deliberate notional run is still possible.
        if args.equity is None:
            live_equity, _prior = broker.get_account_equity()
            if live_equity:
                equity = live_equity
                print(f"Sizing off live account equity: ${equity:,.2f}")
            else:
                print(f"Broker reported no equity; sizing off ${equity:,.2f}.",
                      file=sys.stderr)

    # A dry run sends nothing, so it needs no lock and must not block a real one.
    with SingleInstance(SESSION_LOCK) if not args.dry_run else _NullLock() as lock:
        if not lock.acquired:
            print(
                "Another Meridian session is already running — refusing to place "
                "a second set of orders.",
                file=sys.stderr,
            )
            return 1

        decisions = run_paper_session(
            broker,
            account_equity=equity,
            price_start=args.start or "2023-01-01",
            dry_run=args.dry_run,
        )

    print_session_report(decisions)

    active = sum(1 for d in decisions if not d.skipped)
    orders = sum(len(d.orders) for d in decisions)
    verb = "previewed" if args.dry_run else "sent"
    print(f"Done: {active} strategies processed, {orders} orders {verb}.")
    return 0


class _NullLock:
    """Stands in for the session lock when nothing is being sent (dry runs)."""

    acquired = True

    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _cmd_reconcile(args) -> int:
    """Price yesterday's pending orders against actual Alpaca fills."""
    return _as_scheduled_job("reconcile", lambda: _run_reconcile(args))


def _run_reconcile(args) -> int:
    from meridian.execution.broker import AlpacaBroker
    from meridian.execution.reconcile import load_pending_orders, reconcile_pending_orders

    pending = load_pending_orders()
    if not pending:
        print("No pending orders to reconcile.")
        return 0

    try:
        broker = AlpacaBroker(paper=not args.live)
    except (ImportError, ValueError) as exc:
        print(f"Broker init failed: {exc}", file=sys.stderr)
        return 1

    records = reconcile_pending_orders(broker)
    for r in records:
        print(f"  {r['side'].upper()} {abs(r['qty']):.2f} {r['symbol']} "
              f"filled @ ${r['price']:.2f} (placed {r['submitted']})")
    remaining = len(load_pending_orders())
    print(f"Done: {len(records)} fills recorded, {remaining} orders still pending.")
    return 0


def _cmd_chart(args) -> int:
    """Render per-family equity charts then the combined fund curve."""
    from meridian.visualization.fund_chart import build_fund_returns, show_all_family_charts

    print("Building fund charts — running backtests for live picks...")
    data = build_fund_returns(price_start=args.start or "2010-01-01")
    if not data:
        print("No data available.", file=sys.stderr)
        return 1
    show_all_family_charts(data, equity=args.equity)
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
    pl.add_argument(
        "kind", choices=["estimators", "deviations", "regimes", "models", "saved", "runs"]
    )
    pl.set_defaults(func=_cmd_list)

    pp = sub.add_parser(
        "pipeline", help="Drive a family model through the backtest/walk-forward/OOS stages"
    )
    pp.add_argument("model", help="Model as '<family>/<model>' (see `meridian list models`)")
    pp.add_argument("--symbol", default="SPY", help="Single symbol (single-asset model)")
    pp.add_argument("--symbols", nargs="*", help="Several symbols (a basket / multiple runs)")
    pp.add_argument("--universe", default=None,
                    help="Named universe (e.g. SP500, NASDAQ100, QQQ) instead of listing symbols")
    pp.add_argument("--accept-survivorship-bias", action="store_true",
                    dest="accept_survivorship_bias",
                    help="Required to use an index universe (SP500/NASDAQ100/RUSSELL1000): "
                         "it reflects today's constituents, not the historical membership")
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
    fp.add_argument("--accept-survivorship-bias", action="store_true",
                    dest="accept_survivorship_bias",
                    help="Required to use an index universe (SP500/NASDAQ100/RUSSELL1000): "
                         "it reflects today's constituents, not the historical membership")
    fp.add_argument("--equity", type=float, default=100_000.0, help="Total account equity")
    fp.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    fp.add_argument("--cost-bps", type=float, default=1.0, dest="cost_bps")
    fp.add_argument("--report-dir", default="reports", dest="report_dir")
    fp.set_defaults(func=_cmd_fund)

    gp = sub.add_parser("gauntlet", help="Rank every strategy on a symbol/basket (scoreboard)")
    gp.add_argument("--symbol", default="SPY", help="Single symbol (single-asset gauntlet)")
    gp.add_argument("--symbols", nargs="*", help="A basket (cross-sectional gauntlet)")
    gp.add_argument("--universe", default=None, help="Named universe (e.g. SP500)")
    gp.add_argument("--accept-survivorship-bias", action="store_true",
                    dest="accept_survivorship_bias",
                    help="Required to use an index universe (SP500/NASDAQ100/RUSSELL1000): "
                         "it reflects today's constituents, not the historical membership")
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
    sp.add_argument("--accept-survivorship-bias", action="store_true",
                    dest="accept_survivorship_bias",
                    help="Required to use an index universe (SP500/NASDAQ100/RUSSELL1000): "
                         "it reflects today's constituents, not the historical membership")
    sp.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    sp.add_argument("--cost-bps", type=float, default=1.0, dest="cost_bps")
    sp.add_argument("--confirm", action="store_true",
                    help="Auto-confirm the winning setting through the full fixed-OOS holdout")
    sp.set_defaults(func=_cmd_sweep)

    rvp = sub.add_parser(
        "review", help="Build and write the monthly review report for all paper-stage strategies"
    )
    rvp.add_argument("--equity", type=float, default=10_000.0,
                     help="Total account equity (default: 10000)")
    rvp.add_argument("--start", default=None,
                     help="Earliest date to fetch for indicator warmup (default: 2023-01-01)")
    rvp.add_argument("--lookback", type=int, default=30,
                     help="Benchmark return lookback in calendar days (default: 30)")
    rvp.add_argument("--report-dir", default="reports", dest="report_dir",
                     help="Output directory for the markdown report (default: reports/)")
    rvp.set_defaults(func=_cmd_review)

    rp = sub.add_parser(
        "run-paper", help="Compute today's signals for all paper-stage strategies and send orders"
    )
    rp.add_argument("--equity", type=float, default=None,
                    help="Total account equity for position sizing "
                         "(default: read the live account; falls back to 10000)")
    rp.add_argument("--start", default=None,
                    help="Earliest date to fetch for indicator warmup (default: 2023-01-01)")
    rp.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Compute signals and sizes but do not send orders to the broker")
    rp.add_argument("--live", action="store_true",
                    help="Use the live Alpaca endpoint instead of paper (default: paper)")
    rp.set_defaults(func=_cmd_run_paper)

    rcp = sub.add_parser(
        "reconcile", help="Fetch actual fill prices for pending orders and log/notify them"
    )
    rcp.add_argument("--live", action="store_true",
                     help="Use the live Alpaca endpoint instead of paper (default: paper)")
    rcp.set_defaults(func=_cmd_reconcile)

    cp = sub.add_parser(
        "chart", help="Render the fund equity curve for the live picks in the terminal"
    )
    cp.add_argument("--equity", type=float, default=10_000.0,
                    help="Starting account equity for the y-axis scale (default: 10000)")
    cp.add_argument("--start", default=None,
                    help="Earliest date to fetch price history (default: 2010-01-01)")
    cp.set_defaults(func=_cmd_chart)

    mp = sub.add_parser("menu", help="Launch the interactive menu (same as no command)")
    mp.set_defaults(func=_cmd_menu)

    return parser


def main(argv: list[str] | None = None) -> int:
    from urllib.error import URLError

    try:
        from dotenv import load_dotenv
        # override=True makes this project's .env authoritative. Without it a
        # variable already set in the OS environment silently shadows the
        # project's own value — this machine has a user-level NTFY_TOPIC from an
        # unrelated project, which sent Meridian's alerts to the wrong topic.
        load_dotenv(override=True)
    except ImportError:
        pass  # python-dotenv optional; env vars can still be set manually

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
