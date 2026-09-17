"""The Phase 4 pair model as a registered MLflow pyfunc, so the API can score new listings.

Logged without `code_paths`: it is only loaded in-process from this repository.
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from listings.detect import PRICE_FEATURE
from models.price.pyfunc import artifact_dir

LOGGER = logging.getLogger(__name__)
PAIR_MODEL_NAME = "zestimator-duplicate-pair"
PIPELINE_FILE = "pipeline.pkl"
META_FILE = "pair_model.json"


@dataclass
class PairModel:
    pipeline: object
    threshold: float
    features: tuple[str, ...]
    detect_run_id: int | None
    version: str | None = None

    def _matrix(self, frame: pl.DataFrame) -> np.ndarray:
        return frame.select(self.features).to_numpy()

    def scores(self, frame: pl.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self._matrix(frame))[:, 1]

    def price_blind_scores(self, frame: pl.DataFrame) -> np.ndarray:
        return self.scores(frame.with_columns(pl.lit(0.0).alias(PRICE_FEATURE)))

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / PIPELINE_FILE).write_bytes(pickle.dumps(self.pipeline))
        meta = {
            "threshold": self.threshold,
            "features": list(self.features),
            "detect_run_id": self.detect_run_id,
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), "utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path) -> "PairModel":
        directory = Path(directory)
        meta = json.loads((directory / META_FILE).read_text("utf-8"))
        pipeline = pickle.loads((directory / PIPELINE_FILE).read_bytes())  # our own artifact
        return cls(
            pipeline, float(meta["threshold"]), tuple(meta["features"]), meta["detect_run_id"]
        )


class PairModelPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = PairModel.load(artifact_dir(context, "model_dir"))

    def predict(self, context, model_input, params=None):
        return self.model.scores(pl.from_pandas(model_input))


def register_pair_model(
    result, detect_run_id, directory, tracking_uri=None, artifact_location=None
) -> str:
    from listings.config import DetectConfig
    from models.price.pyfunc import pip_requirements
    from models.price.registry import configure, register_champion

    config = DetectConfig()
    configure(config.experiment, tracking_uri, artifact_location)
    model = PairModel(
        result.model, float(result.threshold), tuple(result.feature_names), detect_run_id
    )
    with mlflow.start_run(run_name="pair-model"):
        mlflow.log_params({"threshold": model.threshold, "detect_run_id": detect_run_id})
        info = mlflow.pyfunc.log_model(
            artifact_path="pair_model",
            python_model=PairModelPyfunc(),
            artifacts={"model_dir": str(model.save(directory))},
            pip_requirements=pip_requirements(),
        )
    return register_champion(info.model_uri, PAIR_MODEL_NAME)


def load_pair_model(uri: str = f"models:/{PAIR_MODEL_NAME}@champion") -> PairModel | None:
    from listings.fraud import resolve_price_model_version  # resolves any models:/ URI

    version = resolve_price_model_version(uri)
    if version is None:
        return None
    name = uri.removeprefix("models:/").split("@", 1)[0].split("/", 1)[0]
    try:
        model = mlflow.pyfunc.load_model(f"models:/{name}/{version}").unwrap_python_model().model
    except Exception as exc:  # noqa: BLE001 — a missing champion is an expected state
        LOGGER.warning("pair model %s v%s could not be loaded (%s)", name, version, exc)
        return None
    model.version = version
    return model
