"""Command-line interface for ah-premium-lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ah_premium_lab.config import LabConfig, load_config, resolve_project_root
from ah_premium_lab.core import prepare_spread_frame
from ah_premium_lab.data import load_market_data
from ah_premium_lab.report import generate_cost_sensitivity_report
from ah_premium_lab.stats import run_pair_diagnostics


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "hello":
        print(f"hello cli: {args.message}")
        return

    config = load_config(args.config)
    prepared = _prepare_pair_frame(config, args.pair)

    if args.command == "stats":
        diagnostics = run_pair_diagnostics(prepared, config.stats)
        print(json.dumps(diagnostics.to_dict(), indent=2))
        return

    if args.command == "cost-report":
        output = args.output
        if output is None:
            output_dir = Path(config.report.output_dir)
            output = output_dir / f"cost_sensitivity_{args.pair}.html"
        report_path = generate_cost_sensitivity_report(
            pair_id=args.pair,
            frame=prepared,
            config=config,
            output_path=output,
        )
        print(f"Report generated: {report_path}")
        return

    raise ValueError(f"Unknown command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    default_config = resolve_project_root(Path(__file__)) / "config" / "default.yaml"

    parser = argparse.ArgumentParser(description="AH Premium Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hello_parser = subparsers.add_parser("hello", help="Run hello CLI smoke command")
    hello_parser.add_argument("--message", default="ah-premium-lab", help="Hello message body")

    stats_parser = subparsers.add_parser("stats", help="Run statistical diagnostics for one pair")
    stats_parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to YAML/TOML config file.",
    )
    stats_parser.add_argument("--pair", required=True, help="Pair ID, e.g., 600519-06881")

    report_parser = subparsers.add_parser(
        "cost-report",
        help="Run cost sensitivity scan and generate HTML report",
    )
    report_parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to YAML/TOML config file.",
    )
    report_parser.add_argument("--pair", required=True, help="Pair ID, e.g., 600519-06881")
    report_parser.add_argument("--output", type=Path, help="Output HTML path")

    return parser


def _prepare_pair_frame(config: LabConfig, pair_id: str) -> pd.DataFrame:
    """Load and prepare spread data for a specific pair."""

    raw = load_market_data(config, pair_id=pair_id)
    prepared = prepare_spread_frame(
        raw,
        method=config.core.premium_method,
        zscore_window=config.core.zscore_window,
    )
    return prepared


if __name__ == "__main__":
    main()
