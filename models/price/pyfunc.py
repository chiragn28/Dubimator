"""MLflow pyfunc wrapper so serving loads one artifact: models:/zestimator-price@champion."""

import importlib.metadata
import math
from pathlib import Path

import mlflow
import pandas as pd

from models.price.predictor import PricePredictor, PriceRequest

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = [str(REPO_ROOT / "models"), str(REPO_ROOT / "ingestion")]
RUNTIME_PACKAGES = ("mlflow", "xgboost", "polars", "pandas", "pyarrow", "pydantic", "numpy")


def pip_requirements() -> list[str]:
    return [f"{name}=={importlib.metadata.version(name)}" for name in RUNTIME_PACKAGES]


def artifact_dir(context, key: str) -> Path:
    """The local path of a logged pyfunc artifact, on any OS.

    A model logged on Windows records its artifact path with backslashes
    (`artifacts\\model_dir`); loaded on Linux (the API container) that is one literal
    file name that doesn't exist, so the backslashes are turned into separators.
    """
    raw = context.artifacts[key]
    path = Path(raw)
    if path.exists():
        return path
    return Path(raw.replace("\\", "/"))


def _present(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if not (value is None or (isinstance(value, float) and math.isnan(value)))
    }


class PricePyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor = PricePredictor.from_dir(artifact_dir(context, "model_dir"))

    def predict(self, context, model_input, params=None):
        records = model_input.to_dict(orient="records")
        estimates = [
            self.predictor.predict_one(PriceRequest.parse(_present(record))).model_dump(mode="json")
            for record in records
        ]
        return pd.DataFrame(estimates)
