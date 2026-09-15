"""MLflow logging and model registration for the price model."""

import math
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from models.price.pyfunc import CODE_PATHS, PricePyfunc, pip_requirements

CHAMPION_ALIAS = "champion"


def configure(
    experiment: str, tracking_uri: str | None = None, artifact_location: str | None = None
) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if artifact_location and mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_location)
    mlflow.set_experiment(experiment)


def log_metrics(metrics: dict[str, float]) -> None:
    items = [(key, float(value)) for key, value in metrics.items() if math.isfinite(float(value))]
    for start in range(0, len(items), 500):
        mlflow.log_metrics(dict(items[start : start + 500]))


def log_price_model(model_dir: Path) -> str:
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=PricePyfunc(),
        artifacts={"model_dir": str(model_dir)},
        code_paths=CODE_PATHS,
        pip_requirements=pip_requirements(),
    )
    return info.model_uri


def register_champion(model_uri: str, name: str) -> str:
    version = mlflow.register_model(model_uri, name)
    MlflowClient().set_registered_model_alias(name, CHAMPION_ALIAS, version.version)
    return str(version.version)
