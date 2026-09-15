"""Command line: python -m models.price train|predict."""

import argparse
import dataclasses
import sys

from dotenv import load_dotenv

from ingestion.config import DbSettings
from models.price.config import TrainConfig
from models.price.predictor import PriceInputError, PricePredictor
from models.price.train import TrainingSummary, run_training

KINDS = ("apartment", "hotel_apartment", "townhouse", "villa")
SETS = ("val", "test_clean", "test_honest")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m models.price", description="Train or query the home price model."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train, evaluate and register the model")
    train.add_argument("--trials", type=int, default=TrainConfig.n_trials)
    train.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    train.add_argument("--no-register", action="store_true")
    predict = commands.add_parser("predict", help="estimate one home with the champion model")
    area = predict.add_mutually_exclusive_group(required=True)
    area.add_argument("--area")
    area.add_argument("--area-id", type=int)
    predict.add_argument("--project")
    predict.add_argument("--building")
    predict.add_argument("--kind", choices=KINDS, required=True)
    predict.add_argument("--status", choices=("ready", "off_plan"), required=True)
    predict.add_argument("--size", type=float, required=True, help="size in m²")
    predict.add_argument("--size-basis", choices=("built_up", "plot"), default="built_up")
    predict.add_argument("--bedrooms", type=int)
    predict.add_argument("--penthouse", action="store_true")
    predict.add_argument("--parking", choices=("yes", "no"))
    predict.add_argument("--asking", type=float, help="asking price in AED")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv()  # before any settings are read: without it DbSettings defaults to port 5432
    if args.command == "train":
        return _train(args)
    return _predict(args)


def _print_summary(summary: TrainingSummary) -> None:
    print(
        f"Device {summary.device}; rows " + ", ".join(f"{k}={v:,}" for k, v in summary.rows.items())
    )
    print(f"{'model':<20}{'set':<13}{'MdAPE':>8}{'PPE10':>8}{'PPE20':>8}{'RMSE(ln)':>10}")
    for model, metrics in summary.metrics.items():
        for set_name in SETS:
            if f"{set_name}.all.mdape" not in metrics:
                continue
            values = [
                metrics[f"{set_name}.all.{name}"]
                for name in ("mdape", "ppe10", "ppe20", "rmse_log")
            ]
            print(
                f"{model:<20}{set_name:<13}{values[0]:>8.2%}{values[1]:>8.1%}"
                f"{values[2]:>8.1%}{values[3]:>10.4f}"
            )
    for name, seconds in summary.seconds.items():
        print(f"  {name}: {seconds:,.1f}s")
    if summary.registered_version:
        print(
            f"Registered {TrainConfig.model_name} version {summary.registered_version} as @champion"
        )


def _train(args: argparse.Namespace) -> int:
    config = dataclasses.replace(TrainConfig(), n_trials=args.trials)
    try:
        summary = run_training(
            DbSettings.from_env(), config, device=args.device, register=not args.no_register
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print_summary(summary)
    if not summary.gate_passed:
        print(
            "Acceptance gate failed: the champion did not beat the comps baseline's test MdAPE "
            "by 10%. Nothing was registered.",
            file=sys.stderr,
        )
        return 2
    return 0


def _load_champion() -> PricePredictor:
    import mlflow

    model = mlflow.pyfunc.load_model(f"models:/{TrainConfig.model_name}@champion")
    return model.unwrap_python_model().predictor


def _predict(args: argparse.Namespace) -> int:
    request = {
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
        "asking_price_aed": args.asking,
    }
    try:
        estimate = _load_champion().predict_one({k: v for k, v in request.items() if v is not None})
    except PriceInputError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1
    print(estimate.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
