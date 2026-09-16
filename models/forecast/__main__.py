"""Command line: python -m models.forecast build|infra-check|train|evaluate|predict."""

import argparse
import dataclasses
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from ingestion.config import DbSettings
from models.forecast.config import DATA_DIR, ForecastConfig
from models.forecast.features import CORE_FEATURES, add_core_features, feature_coverage
from models.forecast.infra import fetch_reference, load_projects, project_lines, validate_projects
from models.forecast.rows import excluded_summary, load_rows
from models.forecast.targets import build_targets

DEFAULTS = ForecastConfig()  # parser defaults, fixed at import (tests replace ForecastConfig)


def _settings() -> DbSettings:
    return DbSettings.from_env()


def write_json(name: str, payload: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


class Stages:
    """Wall-clock seconds per stage, merged into DATA_DIR/stage_timings.json by command."""

    def __init__(self, command: str):
        self.command = command
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = round(time.perf_counter() - started, 2)

    def save(self) -> None:
        path = DATA_DIR / "stage_timings.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        write_json("stage_timings.json", {**existing, self.command: self.seconds})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m models.forecast",
        description="Build, check, train, evaluate and query the price forecasts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="rows, targets and features plus a quality report")
    build.add_argument("--sample", type=int, help="screen a seeded random sample of N rows")
    build.add_argument("--seed", type=int, default=DEFAULTS.seed)
    commands.add_parser("infra-check", help="validate and list the infrastructure table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv()  # before any settings are read: DbSettings.from_env() needs POSTGRES_PORT
    handlers = {"build": _build, "infra-check": _infra_check}
    return handlers[args.command](args)


def _print(lines) -> None:
    for line in lines:
        print(line)


def _build(args: argparse.Namespace) -> int:
    config = dataclasses.replace(ForecastConfig(), sample_rows=args.sample, seed=args.seed)
    stages = Stages("build")
    try:
        with stages.stage("load_rows"):
            rows, quality, data_end, _, _ = load_rows(_settings(), config)
        _print(quality.lines())
        if quality.drop_share > config.max_drop_share:
            print(
                f"Build stopped: {quality.drop_share:.1%} of rows were dropped, above the "
                f"{config.max_drop_share:.0%} limit",
                file=sys.stderr,
            )
            return 1
        with stages.stage("targets"):
            frame, report = build_targets(rows, data_end)
        with stages.stage("features"):
            frame = add_core_features(frame)
        coverage = feature_coverage(frame, CORE_FEATURES)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print(report.lines())
    print("Feature coverage (non-null share):")
    _print(f"  {name:<28}{share:>7.1%}" for name, share in coverage.items())
    write_json(
        "quality.json",
        {
            "data_end": data_end.isoformat(),
            "sample_rows": config.sample_rows,
            "seed": config.seed,
            "quality": quality.to_dict(),
            "excluded": excluded_summary(quality.excluded),
            "targets": report.to_dict(),
            "feature_coverage": coverage,
        },
    )
    stages.save()
    return 0


def _infra_check(args: argparse.Namespace) -> int:
    try:
        projects = load_projects()
        area_names, data_end = fetch_reference(_settings())
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Infrastructure check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print(project_lines(projects, area_names))
    late = projects.filter(pl.col("announced_date") > data_end).height
    print(
        f"{projects.height} projects; {late} announced after the data end ({data_end}) "
        "and so affect no row"
    )
    problems = validate_projects(projects, set(area_names))
    if problems:
        print("Infrastructure table problems:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("Infrastructure table OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
