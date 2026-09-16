"""A trained horizon model (booster, categories, intervals) and its MLflow pyfunc.

Logged without `code_paths`: the model is only loaded in-process from this repository, which
must be importable (a bundled copy of `models/` would shadow the live one on sys.path).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import xgboost as xgb

from models.forecast.config import HORIZONS
from models.forecast.features import to_matrix
from models.forecast.intervals import half_width
from models.price.boosting import make_dmatrix
from models.price.pyfunc import pip_requirements
from models.price.registry import CHAMPION_ALIAS

LOGGER = logging.getLogger(__name__)
BOOSTER_FILE = "booster.ubj"
META_FILE = "forecast_model.json"


@dataclass
class ForecastModel:
    horizon: str
    booster: xgb.Booster
    rounds: int
    categories: dict[str, list[str]]
    intervals: dict[str, float]
    area_rows: dict[int, int]
    metadata: dict = field(default_factory=dict)

    def _dmatrix(self, frame: pl.DataFrame) -> xgb.DMatrix:
        return make_dmatrix(to_matrix(frame, self.categories))

    def predict_growth(self, frame: pl.DataFrame) -> np.ndarray:
        return self.booster.predict(self._dmatrix(frame), iteration_range=(0, self.rounds))

    def contributions(self, frame: pl.DataFrame) -> np.ndarray:
        return self.booster.predict(
            self._dmatrix(frame), pred_contribs=True, iteration_range=(0, self.rounds)
        )

    def half_width(self, segment: str) -> float:
        return half_width(self.intervals, segment)

    def importance(self) -> pl.DataFrame:
        gains = self.booster.get_score(importance_type="gain")
        return pl.DataFrame(
            {"feature": list(gains), "gain": [float(value) for value in gains.values()]},
            schema={"feature": pl.Utf8, "gain": pl.Float64},
        ).sort("gain", descending=True)

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / BOOSTER_FILE)
        meta = {
            "horizon": self.horizon,
            "rounds": self.rounds,
            "categories": self.categories,
            "intervals": self.intervals,
            "area_rows": {str(key): value for key, value in self.area_rows.items()},
            "metadata": self.metadata,
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2, default=str), "utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path) -> "ForecastModel":
        directory = Path(directory)
        booster = xgb.Booster()
        booster.load_model(directory / BOOSTER_FILE)
        booster.set_param({"device": "cpu"})
        meta = json.loads((directory / META_FILE).read_text("utf-8"))
        return cls(
            horizon=meta["horizon"],
            booster=booster,
            rounds=int(meta["rounds"]),
            categories=meta["categories"],
            intervals=meta["intervals"],
            area_rows={int(key): value for key, value in meta["area_rows"].items()},
            metadata=meta["metadata"],
        )


class ForecastPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = ForecastModel.load(Path(context.artifacts["model_dir"]))

    def predict(self, context, model_input, params=None):
        return self.model.predict_growth(pl.from_pandas(model_input))


def log_forecast_model(model: ForecastModel, directory: Path, artifact_path: str) -> str:
    info = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=ForecastPyfunc(),
        artifacts={"model_dir": str(model.save(directory))},
        pip_requirements=pip_requirements(),
    )
    return info.model_uri


def load_champion(name: str) -> tuple[ForecastModel | None, str | None]:
    """Resolve the champion alias to a version first, then load exactly that version."""
    from listings.fraud import resolve_price_model_version  # resolves any models:/ URI

    version = resolve_price_model_version(f"models:/{name}@{CHAMPION_ALIAS}")
    if version is None:
        return None, None
    try:
        pyfunc = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    except Exception as exc:  # noqa: BLE001 — no champion is an expected state
        LOGGER.warning("forecast model %s v%s could not be loaded (%s)", name, version, exc)
        return None, None
    return pyfunc.unwrap_python_model().model, version


def latest_gates(experiment: str) -> dict[str, dict[str, str]]:
    """Gate status and reason per horizon from the most recent training run's tags."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment],
            order_by=["attributes.start_time DESC"],
            max_results=1,
            output_format="list",
        )
    except Exception as exc:  # noqa: BLE001 — no experiment yet is an expected state
        LOGGER.warning("no forecast runs found in %s (%s)", experiment, exc)
        runs = []
    tags = runs[0].data.tags if runs else {}
    return {
        name: {
            "status": tags.get(f"gate.{name}.status", "unknown"),
            "reason": tags.get(f"gate.{name}.reason", ""),
        }
        for name in HORIZONS
    }
