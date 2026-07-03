"""Interactive text menu — the low-typing way to test symbols against strategies.

Run ``meridian`` with no arguments (or ``meridian menu``) to launch this. It loops: pick a
symbol / basket / universe from a numbered list (or type your own), then pick an action —
one strategy, the full gauntlet (rank all strategies), or the fund lifecycle. ``input_fn`` /
``print_fn`` are injected so the loop is testable without a real terminal. Stdlib only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from meridian.data import loader

if TYPE_CHECKING:
    import pandas as pd
    from rich.console import Console
    from rich.text import Text

# Silver metallic gradient: dark steel → bright highlight → dark steel
_SILVER = [
    "#4a4a4a", "#666666", "#868686", "#a6a6a6",
    "#c0c0c0", "#d8d8d8", "#eeeeee", "#ffffff",
    "#eeeeee", "#d8d8d8", "#c0c0c0", "#a6a6a6",
    "#868686", "#666666", "#4a4a4a",
]


def _silver_text(line: str) -> Text:
    """Apply the silver metallic gradient across one line of text."""
    from rich.text import Text
    t = Text()
    n = len(line)
    for i, ch in enumerate(line):
        pct = i / max(n - 1, 1)
        colour = _SILVER[round(pct * (len(_SILVER) - 1))]
        t.append(ch, style=f"bold {colour}")
    return t


_ART_LINES = [
    "███╗   ███╗ ███████╗ ██████╗  ██╗ ██████╗  ██╗  █████╗  ███╗   ██╗",
    "████╗ ████║ ██╔════╝ ██╔══██╗ ██║ ██╔══██╗ ██║ ██╔══██╗ ████╗  ██║",
    "██╔████╔██║ █████╗   ██████╔╝ ██║ ██║  ██║ ██║ ███████║ ██╔██╗ ██║",
    "██║╚██╔╝██║ ██╔══╝   ██╔══██╗ ██║ ██║  ██║ ██║ ██╔══██║ ██║╚██╗██║",
    "██║ ╚═╝ ██║ ███████╗ ██║  ██║ ██║ ██████╔╝ ██║ ██║  ██║ ██║ ╚████║",
    "╚═╝     ╚═╝ ╚══════╝ ╚═╝  ╚═╝ ╚═╝ ╚═════╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═══╝",
]


def _print_banner(console: Console) -> None:
    """3-D drop-shadow banner: shadow layer (near-black, +1 row +2 cols), then silver on top."""
    from rich.text import Text

    n = len(_ART_LINES)
    w = console.width
    art_w = max(len(line) for line in _ART_LINES)
    left = max(0, (w - art_w) // 2)
    shadow_x = left + 2  # shadow sits 2 cols to the right
    border = _silver_text("═" * w)
    is_tty = getattr(console.file, "isatty", lambda: False)()

    console.print(border)

    if is_tty:
        f = console.file
        # Pass 1 — shadow layer occupies rows 1..n (offset 1 row down, 2 cols right).
        # Row 0 is blank so the foreground top row has no shadow behind it.
        f.write("\n")
        for line in _ART_LINES:
            f.write(f"\033[38;2;30;30;30m{' ' * shadow_x}{line}\033[0m\n")
        # Cursor is now at row n+1. Jump back to row 0.
        f.write(f"\033[{n + 1}A\r")
        f.flush()
        # Pass 2 — silver foreground layer at rows 0..n-1.
        for line in _ART_LINES:
            console.print(_silver_text(" " * left + line))
        # Advance past the dangling shadow row at the bottom.
        f.write("\n")
        f.flush()
    else:
        for line in _ART_LINES:
            console.print(_silver_text(" " * left + line))

    subtitle = Text("institutional-grade quant research")
    subtitle.stylize("italic #a0a0a0")
    console.print(subtitle, justify="center")
    console.print(border)

#: (entry, description) presets shown at the symbol prompt.
PRESETS: list[tuple[str, str]] = [
    ("SPY", "S&P 500 ETF"),
    ("QQQ", "Nasdaq-100 ETF"),
    ("IWM", "Russell 2000 ETF"),
    ("XLK XLF XLE XLY XLV XLI", "Sector-ETF basket (6 names)"),
    ("NASDAQ100", "Nasdaq-100 constituents (universe)"),
]


def _resolve(entry: str) -> list[str]:
    """Turn a typed entry into a symbol list (a known universe name expands; else tickers)."""
    tokens = entry.split()
    if len(tokens) == 1:
        from meridian.data.universe import KNOWN_UNIVERSES, get_universe

        if tokens[0].upper() in KNOWN_UNIVERSES:
            return list(get_universe(tokens[0].upper()).symbols)
    return [t.upper() for t in tokens]


def _prompt_symbols(input_fn, print_fn) -> list[str] | None:
    print_fn("\nChoose what to test:")
    for i, (entry, desc) in enumerate(PRESETS, 1):
        print_fn(f"  {i}) {desc:34s} [{entry}]")
    print_fn("  or type tickers (e.g. AAPL MSFT) or a universe (e.g. SP500)")
    print_fn("  q) quit")
    raw = input_fn("> ").strip()
    if raw.lower() in ("q", "quit", ""):
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(PRESETS):
        raw = PRESETS[int(raw) - 1][0]
    syms = _resolve(raw)
    return syms or None


def _prompt_action(input_fn, print_fn) -> str:
    print_fn("\nAction:  1) single strategy   2) gauntlet (rank all)   3) fund lifecycle")
    print_fn("         4) tune a strategy (parameter sweep)   5) saved strategies")
    print_fn("         b) back   q) quit")
    while True:
        raw = input_fn("> ").strip().lower()
        mapping = {"1": "single", "2": "gauntlet", "3": "fund", "4": "sweep",
                   "5": "saved", "b": "back", "q": "quit"}
        if raw in mapping:
            return mapping[raw]
        print_fn("  (please enter 1, 2, 3, 4, 5, b, or q)")


def _pick_model(input_fn, print_fn, *, cross_sectional: bool) -> str | None:
    from meridian.families import create_model, list_models

    names = [m for m in list_models()
             if getattr(create_model(*m.split("/")), "cross_sectional", False) == cross_sectional]
    kind = "cross-sectional" if cross_sectional else "single-asset"
    print_fn(f"\n{kind} strategies:")
    for i, m in enumerate(names, 1):
        print_fn(f"  {i:2d}) {m}")
    raw = input_fn("pick #> ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(names):
        return names[int(raw) - 1]
    print_fn("  (invalid choice)")
    return None


def _load_one(symbol: str):
    frame = loader.load_ohlcv(symbol)
    return frame["close"], frame


def _do_single(symbols: list[str], input_fn, print_fn) -> None:
    from meridian.pipeline import (
        record_from_pipeline,
        run_pipeline,
        run_universe_pipeline,
        save_record,
    )
    from meridian.portfolio import StrategyLedger

    basket = len(symbols) > 1
    qualified = _pick_model(input_fn, print_fn, cross_sectional=basket)
    if qualified is None:
        return
    family, name = qualified.split("/")
    from meridian.families import create_model
    from meridian.pipeline.graduation import criteria_for_family

    model = create_model(family, name)
    ledger = StrategyLedger(name=name, family=family, stage="research")
    criteria = criteria_for_family(family)
    if basket:
        universe = {s: _load_one(s)[0] for s in symbols}
        results = run_universe_pipeline(model, universe, ledger=ledger, criteria=criteria)
        label = " ".join(symbols[:3]) + ("…" if len(symbols) > 3 else "")
        print_fn(f"\n{qualified} on [{label}] ({len(universe)} names)")
        basket_sym = "+".join(symbols[:3]) + (f"+{len(symbols)-3}more" if len(symbols) > 3 else "")
        rec = record_from_pipeline(family, name, basket_sym, results, ledger)
        if rec is not None:
            save_record(rec)
    else:
        prices, frame = _load_one(symbols[0])
        results = run_pipeline(model, prices, ledger=ledger, bars=frame, criteria=criteria)
        print_fn(f"\n{qualified} on {symbols[0]}  ({len(prices)} bars)")
        rec = record_from_pipeline(family, name, symbols[0], results, ledger)
        if rec is not None:
            save_record(rec)
    for stage, res in results.items():
        sharpe = res.scorecard.get("sharpe")
        sh = f"{sharpe:.2f}" if isinstance(sharpe, float) and sharpe == sharpe else "n/a"
        print_fn(f"  {stage:12s} passed={res.passed!s:5s} sharpe={sh:>6s}")
    print_fn(f"  final graduation stage: {ledger.stage}")


def _do_gauntlet(symbols: list[str], print_fn) -> None:
    from meridian.experiments.gauntlet import (
        format_gauntlet,
        gauntlet_single,
        gauntlet_universe,
    )

    if len(symbols) > 1:
        universe = {s: _load_one(s)[0] for s in symbols}
        print_fn(f"\nGauntlet — cross-sectional models on {len(universe)} names:")
        df = gauntlet_universe(universe)
    else:
        prices, frame = _load_one(symbols[0])
        print_fn(f"\nGauntlet — single-asset models on {symbols[0]} ({len(prices)} bars):")
        df = gauntlet_single(prices, bars=frame)
    print_fn(format_gauntlet(df))


_NEXT_STEP: dict[str, str] = {
    "backtest": (
        "Run walk-forward validation next (action 4 → tune, or re-run fund lifecycle). "
        "Rolling windows will show whether performance holds up across different market periods."
    ),
    "walk_forward": (
        "Run the OOS holdout test next. This tests the strategy on 2020-2022 data it has "
        "never seen — the hardest gate before paper trading."
    ),
    "oos": (
        "Ready for paper trading. Connect the strategy to Alpaca paper mode and let it run "
        "on live prices for at least 30 days with no real capital at risk."
    ),
    "paper": (
        "Now in paper trading. Let it run on live prices for 30+ days, then re-run the fund "
        "lifecycle to check whether it qualifies for Pilot (1–3% of real capital)."
    ),
    "pilot": (
        "In Pilot. After 90 days of live results within target metrics, promote to Proven "
        "and increase allocation to 5–10% of equity."
    ),
    "proven": "In Proven. After 180 days promote to Core (10–20% of equity).",
    "core": "In Core. After 365 days of strong results promote to Elite (20%+ of equity).",
    "elite": "Elite — maximum allocation. Monitor monthly and retire if expectancy degrades.",
}


def _print_next_steps(review, print_fn) -> None:
    """Print next-step guidance for sleeves labelled increase or that just promoted."""
    actionable = [s for s in review.sleeves if s.action in ("increase",)]
    if not actionable:
        return
    print_fn("\n--- Next steps ---")
    for sleeve in actionable:
        next_stage = sleeve.graduation[1] if sleeve.graduation[1] else sleeve.stage
        guidance = _NEXT_STEP.get(
            next_stage, _NEXT_STEP.get(sleeve.stage, "Review scorecard and re-run.")
        )
        print_fn(f"\n  {sleeve.sleeve} ({sleeve.family})  →  {sleeve.action.upper()}")
        print_fn(f"  Current stage : {sleeve.stage}")
        print_fn(f"  Next stage    : {next_stage}")
        print_fn(f"  What to do    : {guidance}")
        if sleeve.rationale:
            print_fn(f"  Why           : {'; '.join(sleeve.rationale)}")


def _show_fund_chart(ledgers, prices: pd.Series, bars, symbol: str, equity: float) -> None:
    """Collect per-sleeve returns and open the fund equity chart in the browser."""
    from meridian.experiments.fund import _first_single_asset
    from meridian.families import create_model
    from meridian.portfolio.allocation import SLEEVE_ALLOCATIONS
    from meridian.visualization.fund_chart import show_fund_equity_chart

    returns_by_sleeve: dict[str, tuple[pd.Series, float]] = {}
    for led in ledgers:
        model_name = _first_single_asset(led.family)
        if model_name is None:
            continue
        weight = SLEEVE_ALLOCATIONS.get(led.family, 0.0)
        if weight == 0.0:
            continue
        model = create_model(led.family, model_name)
        result = model.backtest(prices, bars=bars)
        returns_by_sleeve[led.family] = (result.returns, weight)

    if returns_by_sleeve:
        show_fund_equity_chart(returns_by_sleeve, symbol=symbol, equity=equity)


def _do_fund(symbols: list[str], input_fn, print_fn) -> None:
    from meridian.experiments.fund import run_cs_fund, run_fund
    from meridian.regimes.labeler import build_regime_frame

    raw = input_fn("equity [100000]> ").strip()
    equity = float(raw) if raw else 100_000.0

    regime_frame = None
    try:
        regime_frame = build_regime_frame()
    except Exception:
        pass  # network unavailable — run without regime gating

    if len(symbols) > 1:
        # Cross-sectional fund: run all CS models on the whole basket at once.
        prices_by_symbol = {s: _load_one(s)[0] for s in symbols}
        _ledgers, review = run_cs_fund(prices_by_symbol, equity=equity)
        label = " ".join(symbols[:3]) + ("…" if len(symbols) > 3 else "")
        print_fn(f"\nFund [{label}] ({len(symbols)} symbols): "
                 + ", ".join(f"{s.sleeve}={s.action}" for s in review.sleeves))
    else:
        prices, frame = _load_one(symbols[0])
        _ledgers, review = run_fund(
            prices, frame, symbol=symbols[0], equity=equity, regime_frame=regime_frame
        )
        print_fn(f"\nFund — {symbols[0]}: "
                 + ", ".join(f"{s.sleeve}={s.action}" for s in review.sleeves))
        _show_fund_chart(_ledgers, prices, frame, symbols[0], equity)

    _print_next_steps(review, print_fn)


def _do_sweep(symbols: list[str], input_fn, print_fn) -> None:
    from meridian.experiments.sweep import format_confirm, sweep_and_confirm

    qualified = _pick_model(input_fn, print_fn, cross_sectional=False)
    if qualified is None:
        return
    family, name = qualified.split("/")
    symbol = symbols[0]                                   # single-asset sweep on one symbol
    prices, frame = _load_one(symbol)
    print_fn(f"\nTuning {qualified} on {symbol} ({len(prices)} bars), then confirming...")
    result = sweep_and_confirm(prices, family, name, bars=frame)
    print_fn(format_confirm(result, family, name))


def _fmt(val, fmt=".2f", pct=False) -> str:
    """Format a scorecard number; return 'n/a' for None or NaN."""
    if val is None or val != val:
        return "n/a"
    return f"{val:{fmt}}" if not pct else f"{val:.1%}"


_ALL_FAMILIES = [
    "long_term_etf", "momentum", "trend_following", "mean_reversion",
    "pullback_continuation", "sector_rotation", "event_driven", "volatility",
    "experimental_research",
]

_STAGE_RANK = {
    "research": 0, "backtest": 1, "walk_forward": 2, "oos": 3,
    "paper": 4, "pilot": 5, "proven": 6, "core": 7, "elite": 8,
}

_STAGE_BAR = {
    "research":     "·",
    "backtest":     "▒",
    "walk_forward": "▓",
    "oos":          "█",
    "paper":        "█",
    "pilot":        "█",
    "proven":       "█",
    "core":         "█",
    "elite":        "█",
}


def _family_chart(records: list, print_fn) -> None:
    """Print a family × stage progress chart above the saved strategies list."""
    best: dict[str, tuple] = {}
    for r in records:
        rank = _STAGE_RANK.get(r.stage_passed, 0)
        sharpe = r.scorecard.get("sharpe") or 0.0
        prev = best.get(r.family)
        if prev is None or (rank, sharpe) > (prev[0], prev[1]):
            best[r.family] = (rank, sharpe, r.stage_passed, r.model, r.symbol)

    stages = ["backtest", "walk_forward", "oos", "paper", "pilot", "proven", "core", "elite"]
    header = (
        f"  {'family':<22}  {'stage':<13}  {'best model':<30}  "
        f"{'sym':<5}  {'sharpe':>6}  progress"
    )
    print_fn("\nFamily status:")
    print_fn("  " + "─" * (len(header) - 2))
    print_fn(header)
    print_fn("  " + "─" * (len(header) - 2))
    for fam in _ALL_FAMILIES:
        if fam in best:
            rank, sharpe, stage, model, sym = best[fam]
            bar = "".join(_STAGE_BAR.get(s, "·") if _STAGE_RANK[s] <= rank else "·" for s in stages)
            sh  = f"{sharpe:.2f}"
        else:
            stage, model, sym, bar, sh = "—", "—", "—", "·" * len(stages), "—"
        print_fn(f"  {fam:<22}  {stage:<13}  {model:<30}  {sym:<5}  {sh:>6}  {bar}")
    print_fn("  " + "─" * (len(header) - 2))
    print_fn("  progress key: · none  ▒ backtest  ▓ walk-fwd  █ oos/live")
    print_fn("")


def _do_saved(input_fn, print_fn) -> None:
    from meridian.pipeline import load_records

    records = load_records()
    if not records:
        print_fn("  (no saved strategies yet — run a single strategy that passes to save one)")
        return

    _family_chart(records, print_fn)

    header = (
        f"  {'#':>3}  {'family/model':<32}  {'sym':<6}  {'stage':<12}  "
        f"{'date':<10}  {'sharpe':>6}  {'cagr':>7}  {'mdd':>7}"
    )
    print_fn(f"All records ({len(records)} total):")
    print_fn(header)
    print_fn("  " + "-" * (len(header) - 2))
    for i, rec in enumerate(records, 1):
        qn = f"{rec.family}/{rec.model}"
        sc = rec.scorecard
        row = (
            f"  {i:>3}  {qn:<32}  {rec.symbol:<6}  {rec.stage_passed:<12}  {rec.saved_at:<10}"
            f"  {_fmt(sc.get('sharpe')):>6}  {_fmt(sc.get('cagr'), pct=True):>7}"
            f"  {_fmt(sc.get('max_drawdown'), pct=True):>7}"
        )
        print_fn(row)

    raw = input_fn("\npick # to view full scorecard (or Enter to go back)> ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(records)):
        return

    rec = records[int(raw) - 1]
    sc = rec.scorecard
    print_fn(f"\n{'=' * 56}")
    print_fn(f"  {rec.family}/{rec.model}  on  {rec.symbol}")
    print_fn(
        f"  stage: {rec.stage_passed}   saved: {rec.saved_at}   "
        f"graduation: {rec.ledger.get('stage', '?')}"
    )
    print_fn(f"{'=' * 56}")

    sections = {
        "Return": ["total_return", "cagr", "ann_volatility"],
        "Risk-adjusted": ["sharpe", "sortino", "calmar", "ulcer_index"],
        "Drawdown": ["max_drawdown", "max_drawdown_duration"],
        "Trade stats": [
            "n_trades", "trades_per_year", "trade_win_rate", "trade_avg_win", "trade_avg_loss",
            "trade_expectancy", "trade_profit_factor", "avg_holding_period",
        ],
        "Exposure": ["hit_rate", "profit_factor", "turnover"],
        "Tail": ["skew", "kurtosis", "var_95", "cvar_95", "best", "worst", "tail_ratio"],
    }
    pct_keys = {"total_return", "cagr", "ann_volatility", "max_drawdown",
                "trade_win_rate", "trade_avg_win", "trade_avg_loss",
                "trade_expectancy", "hit_rate", "var_95", "cvar_95", "best", "worst"}
    for section, keys in sections.items():
        rows = [(k, sc[k]) for k in keys if k in sc]
        if not rows:
            continue
        print_fn(f"\n  {section}:")
        for k, v in rows:
            if isinstance(v, float):
                fmtd = f"{v:.1%}" if k in pct_keys else f"{v:.3f}"
            else:
                fmtd = str(v)
            print_fn(f"    {k:<26} {fmtd}")

    slip = sc.get("slippage_sensitivity")
    if slip:
        print_fn("\n  Slippage sensitivity (Sharpe at extra cost):")
        for bps, sh in sorted(slip.items()):
            print_fn(f"    +{bps:>3} bps  ->  sharpe {sh:.2f}")


def run_menu(input_fn=input, print_fn=print) -> int:
    """Run the interactive menu loop until the user quits. Returns an exit code."""
    from rich.console import Console
    _print_banner(Console())
    while True:
        symbols = _prompt_symbols(input_fn, print_fn)
        if symbols is None:
            print_fn("Goodbye.")
            return 0
        while True:
            action = _prompt_action(input_fn, print_fn)
            if action == "back":
                break
            if action == "quit":
                print_fn("Goodbye.")
                return 0
            try:
                if action == "single":
                    _do_single(symbols, input_fn, print_fn)
                elif action == "gauntlet":
                    _do_gauntlet(symbols, print_fn)
                elif action == "fund":
                    _do_fund(symbols, input_fn, print_fn)
                elif action == "sweep":
                    _do_sweep(symbols, input_fn, print_fn)
                elif action == "saved":
                    _do_saved(input_fn, print_fn)
            except Exception as exc:  # keep the menu alive on bad data / network errors
                print_fn(f"  ! run failed: {exc}")
