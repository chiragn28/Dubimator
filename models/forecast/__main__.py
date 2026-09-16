"""Command line: python -m models.forecast build|infra-check|train|evaluate|predict."""

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from ingestion.config import DbSettings
from models.forecast.baselines import baseline_growth
from models.forecast.config import DATA_DIR, FEATURES, HORIZON_SPECS, HORIZONS, ForecastConfig
from models.forecast.evaluate import MODEL, format_table, segment_table
from models.forecast.features import build_dataset, feature_coverage
from models.forecast.folds import usable_rows
from models.forecast.infra import fetch_reference, load_projects, project_lines, validate_projects
from models.forecast.model import latest_gates, load_champion
from models.forecast.predict import Forecaster
from models.forecast.rows import excluded_summary, load_rows
from models.forecast.train import run_training
from models.price.predictor import PriceInputError

DEFAULTS = ForecastConfig()  # parser defaults, fixed at import (tests replace ForecastConfig)

KINDS = ("apartment", "hotel_apartment", "townhouse", "villa")


class GuardError(RuntimeError):
    """A stop-point guardrail refused the data."""


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
    train = commands.add_parser("train", help="tune, evaluate, gate and register each horizon")
    train.add_argument("--trials", type=int, default=DEFAULTS.n_trials)
    train.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    train.add_argument("--no-register", action="store_true")
    commands.add_parser("evaluate", help="re-score the registered champions on their test periods")
    predict = commands.add_parser("predict", help="forecast one home as JSON")
    area = predict.add_mutually_exclusive_group(required=True)
    area.add_argument("--area")
    area.add_argument("--area-id", type=int)
    predict.add_argument("--kind", choices=KINDS, required=True)
    predict.add_argument("--status", choices=("ready", "off_plan"), required=True)
    predict.add_argument("--size", type=float, required=True, help="size in m²")
    predict.add_argument("--size-basis", choices=("built_up", "plot"), default="built_up")
    predict.add_argument("--bedrooms", type=int)
    predict.add_argument("--building")
    predict.add_argument("--project")
    predict.add_argument("--penthouse", action="store_true")
    predict.add_argument("--parking", choices=("yes", "no"))
    predict.add_argument("--property-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    load_dotenv()  # before any settings are read: DbSettings.from_env() needs POSTGRES_PORT
    handlers = {
        "build": _build,
        "infra-check": _infra_check,
        "train": _train,
        "evaluate": _evaluate,
        "predict": _predict,
    }
    return handlers[args.command](args)


def _print(lines) -> None:
    for line in lines:
        print(line)


def _dataset(config: ForecastConfig, stages: Stages):
    with stages.stage("load_rows"):
        rows, quality, data_end, areas, _ = load_rows(_settings(), config)
    _print(quality.lines())
    if quality.drop_share > config.max_drop_share:
        raise GuardError(
            f"{quality.drop_share:.1%} of rows were dropped, above the "
            f"{config.max_drop_share:.0%} limit"
        )
    projects = load_projects()
    problems = validate_projects(projects, set(areas["area_id"].to_list()))
    if problems:
        raise GuardError(
            "the infrastructure table has problems:\n" + "\n".join(f"  {p}" for p in problems)
        )
    with stages.stage("dataset"):
        frame, report = build_dataset(rows, data_end, projects)
    return frame, report, quality, projects, data_end


def _build(args: argparse.Namespace) -> int:
    config = dataclasses.replace(ForecastConfig(), sample_rows=args.sample, seed=args.seed)
    stages = Stages("build")
    try:
        frame, report, quality, _, data_end = _dataset(config, stages)
    except GuardError as exc:
        print(f"Build stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    coverage = feature_coverage(frame, FEATURES)
    _print(report.lines())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    quality.excluded.write_csv(DATA_DIR / "excluded.csv")
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


def _train(args: argparse.Namespace) -> int:
    config = dataclasses.replace(ForecastConfig(), n_trials=args.trials)
    stages = Stages("train")
    try:
        frame, report, quality, projects, data_end = _dataset(config, stages)
        with stages.stage("training"):
            summary = run_training(
                frame, report, quality, projects, data_end, config,
                device=args.device, register=not args.no_register,
            )  # fmt: skip
    except GuardError as exc:
        print(f"Training stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Device {summary.device}; MLflow run {summary.run_id}")
    for name, result in summary.results.items():
        roles = Counter(fold.role for fold in result.folds)
        counts = ", ".join(f"{roles[role]} {role}" for role in ("score", "tune", "gap", "test"))
        print(
            f"{name}: {result.status} ({len(result.folds)} folds: {counts}; {result.seconds:,.0f}s)"
        )
        _print(f"  {reason}" for reason in result.reasons)
        if result.table is not None:
            _print(format_table(name, result.table))
            coverage = ", ".join(f"{k} {v:.1%}" for k, v in result.coverage.items())
            print(f"  80% range coverage on test: {coverage}")
            means = result.fold_scores.group_by("model").agg(pl.col("mape").mean()).sort("model")
            print("  mean fold MAPE: " + ", ".join(f"{m} {v:.2%}" for m, v in means.iter_rows()))
    for name, version in summary.versions.items():
        print(f"Registered {config.model_prefix}-{name} version {version} as @champion")
    stages.save()
    if not any(result.status == "passed" for result in summary.results.values()):
        print("No horizon passed its gate; nothing was registered.", file=sys.stderr)
        return 2
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    config = ForecastConfig()
    stages = Stages("evaluate")
    try:
        champions = {name: load_champion(f"{config.model_prefix}-{name}") for name in HORIZONS}
        gates = latest_gates(config.experiment)
        frame = None
        if any(model is not None for model, _ in champions.values()):
            frame = _dataset(config, stages)[0]
        for name, (model, version) in champions.items():
            if model is None:
                reason = gates[name]["reason"] or f"no registered {name} model"
                print(f"{name}: not deployed ({reason})")
                continue
            horizon = HORIZON_SPECS[name]
            start = date.fromisoformat(model.metadata["test_cutoff"])
            end = date.fromisoformat(model.metadata["test_end"])
            test = usable_rows(frame, horizon).filter(
                pl.col("instance_date").is_between(start, end, closed="left")
            )
            predictions = {MODEL: model.predict_growth(test), **baseline_growth(test, horizon)}
            actual = test[f"growth_{name}"].to_numpy()
            table = segment_table(test, predictions, actual, model.metadata["top_areas"])
            print(f"{name} champion v{version}: test period {start} to {end}, {test.height:,} rows")
            _print(format_table(name, table))
    except GuardError as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    stages.save()
    return 0


def _predict(args: argparse.Namespace) -> int:
    request = {
        "property_id": args.property_id,
        "area": args.area,
        "area_id": args.area_id,
        "project": args.project,
        "building": args.building,
        "property_kind": args.kind,
        "status": args.status,
        "size_sqm": args.size,
        "size_basis": args.size_basis,
        "bedrooms": args.bedrooms,
        "is_penthouse": args.penthouse,
        "has_parking": None if args.parking is None else args.parking == "yes",
    }
    try:
        forecaster = Forecaster.from_registry(_settings(), ForecastConfig())
        result = forecaster.forecast({k: v for k, v in request.items() if v is not None})
    except PriceInputError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Forecast failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
