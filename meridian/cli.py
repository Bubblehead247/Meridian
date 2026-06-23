"""Command-line interface for Meridian.

A thin argparse front end over the config-driven runner, so a research run or a
paper dry-run is one command:

    meridian validate configs/validation_spy.yaml
    meridian backtest configs/validation_spy.yaml
    meridian paper    configs/paper_spy.yaml
    meridian list estimators

Kept dependency-free (stdlib argparse) per the project's simplicity rule.
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
    catalogs = {
        "estimators": list_estimators,
        "deviations": list_deviations,
        "regimes": list_regimes,
    }
    names = catalogs[args.kind]()
    print(f"{len(names)} {args.kind}:")
    for n in names:
        print(f"  {n}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meridian", description="Mean-reversion research platform")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, help_ in [
        ("validate", _cmd_validate, "Walk-forward validate estimators from a config and write a report"),
        ("backtest", _cmd_backtest, "Run validation and print the ranked table (no report)"),
        ("paper", _cmd_paper, "Paper-trade dry-run: warm on history, replay the rest"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", help="Path to a YAML experiment config")
        p.set_defaults(func=fn)

    pl = sub.add_parser("list", help="List available estimators/deviations/regimes")
    pl.add_argument("kind", choices=["estimators", "deviations", "regimes"])
    pl.set_defaults(func=_cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
