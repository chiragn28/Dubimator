"""Command line: python -m monitoring drift [--months 3] [--log-file PATH]."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from ingestion.config import DbSettings
from monitoring.drift import report


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m monitoring")
    commands = parser.add_subparsers(dest="command", required=True)
    drift = commands.add_parser("drift", help="write data/monitoring/drift_<date>.json and .html")
    drift.add_argument("--months", type=_positive, default=3, help="current window (months)")
    drift.add_argument("--log-file", type=Path, default=None, help="API JSON log to summarise")
    return parser


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def _drift(args) -> int:
    result = report(DbSettings.from_env(), months=args.months, log_file=args.log_file)
    payload = result.payload
    for name, window in payload["windows"].items():
        print(f"{name:<10} {window['start']} .. {window['end']}  {window['rows']:,} rows")
    print(f"{'feature':<16}{'kind':<13}{'psi':>8}{'ks':>8}")
    for row in payload["features"]:
        flag = "  FLAG" if row["flagged"] else ""
        print(
            f"{row['feature']:<16}{row['kind']:<13}{_fmt(row['psi']):>8}{_fmt(row['ks']):>8}{flag}"
        )
    print(f"forecast 3m: {payload['forecast']['status']}")
    if payload["search"] is not None:
        print(f"search: {payload['search']}")
    print(f"flagged: {', '.join(payload['flagged']) or 'none'}")
    print(f"wrote {result.json_path}")
    print(f"wrote {result.html_path}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()  # before DbSettings.from_env() and before MLflow reads MLFLOW_TRACKING_URI
    return {"drift": _drift}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
