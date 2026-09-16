"""Two GBDT rankers behind one interface, and the MLflow pyfunc that serves either one."""

import importlib.metadata
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = [str(REPO_ROOT / name) for name in ("search", "listings", "ingestion", "models")]
RUNTIME_PACKAGES = ("mlflow", "xgboost", "lightgbm", "polars", "pandas", "pyarrow", "numpy")
META_FILE = "ranker.json"


def _matrix(frame: pl.DataFrame, features) -> np.ndarray:
    return frame.select(list(features)).to_numpy().astype(np.float64)


def _group_sizes(frame: pl.DataFrame) -> np.ndarray:
    return frame.group_by("query_id", maintain_order=True).len()["len"].to_numpy().astype(np.int64)


class Ranker:
    kind = "base"
    model_file = ""

    def __init__(self, booster, features, params: dict, best_iteration: int):
        self.booster = booster
        self.features = tuple(features)
        self.params = dict(params)
        self.best_iteration = int(best_iteration)

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def importance(self) -> pl.DataFrame:
        raise NotImplementedError

    def _save_model(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def _load_model(cls, path: Path):
        raise NotImplementedError

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._save_model(directory / self.model_file)
        meta = {
            "kind": self.kind,
            "features": list(self.features),
            "params": self.params,
            "best_iteration": self.best_iteration,
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return directory


class XGBRanker(Ranker):
    kind = "xgboost"
    model_file = "model.json"

    @classmethod
    def fit(cls, train, tune, features, params, config) -> "XGBRanker":
        import xgboost as xgb

        def dmatrix(frame):
            return xgb.DMatrix(
                _matrix(frame, features),
                label=frame["grade"].to_numpy(),
                qid=frame["query_id"].to_numpy(),
                feature_names=list(features),
            )

        booster_params = {
            "objective": "rank:ndcg",
            "eval_metric": f"ndcg@{config.ndcg_k}",
            "lambdarank_pair_method": "topk",
            "tree_method": "hist",
            "device": config.device,
            "seed": config.seed,
            **params,
        }
        booster = xgb.train(
            booster_params,
            dmatrix(train),
            num_boost_round=config.max_rounds,
            evals=[(dmatrix(tune), "tune")],
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=False,
        )
        booster.set_param({"device": "cpu"})
        return cls(booster, features, params, booster.best_iteration)

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        import xgboost as xgb

        matrix = xgb.DMatrix(_matrix(frame, self.features), feature_names=list(self.features))
        return self.booster.predict(matrix, iteration_range=(0, self.best_iteration + 1))

    def importance(self) -> pl.DataFrame:
        gains = self.booster.get_score(importance_type="gain")
        return pl.DataFrame(
            {"feature": list(self.features), "gain": [gains.get(f, 0.0) for f in self.features]}
        )

    def _save_model(self, path: Path) -> None:
        self.booster.save_model(str(path))

    @classmethod
    def _load_model(cls, path: Path):
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        booster.set_param({"device": "cpu"})
        return booster


class LGBMRanker(Ranker):
    kind = "lightgbm"
    model_file = "model.txt"

    @classmethod
    def fit(cls, train, tune, features, params, config) -> "LGBMRanker":
        import lightgbm as lgb

        def dataset(frame, reference=None):
            return lgb.Dataset(
                _matrix(frame, features),
                label=frame["grade"].to_numpy(),
                group=_group_sizes(frame),
                feature_name=list(features),
                reference=reference,
                free_raw_data=False,
            )

        train_set = dataset(train)
        booster_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [config.ndcg_k],
            "verbosity": -1,
            "seed": config.seed,
            "deterministic": True,
            "force_row_wise": True,
            **params,
        }
        booster = lgb.train(
            booster_params,
            train_set,
            num_boost_round=config.max_rounds,
            valid_sets=[dataset(tune, reference=train_set)],
            callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
        )
        return cls(booster, features, params, booster.best_iteration)

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        iterations = self.best_iteration if self.best_iteration > 0 else None
        return self.booster.predict(_matrix(frame, self.features), num_iteration=iterations)

    def importance(self) -> pl.DataFrame:
        gains = self.booster.feature_importance(importance_type="gain")
        return pl.DataFrame({"feature": list(self.features), "gain": [float(g) for g in gains]})

    def _save_model(self, path: Path) -> None:
        self.booster.save_model(str(path))

    @classmethod
    def _load_model(cls, path: Path):
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))


RANKER_CLASSES: dict[str, type[Ranker]] = {"xgboost": XGBRanker, "lightgbm": LGBMRanker}


def load_ranker(directory: Path) -> Ranker:
    directory = Path(directory)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    cls = RANKER_CLASSES[meta["kind"]]
    booster = cls._load_model(directory / cls.model_file)
    return cls(booster, meta["features"], meta["params"], meta["best_iteration"])


class RankerPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.ranker = load_ranker(Path(context.artifacts["ranker_dir"]))

    def predict(self, context, model_input, params=None):
        return self.ranker.score(pl.from_pandas(model_input))


def _pip_requirements() -> list[str]:
    return [f"{name}=={importlib.metadata.version(name)}" for name in RUNTIME_PACKAGES]


def log_ranker(ranker: Ranker, directory: Path) -> str:
    """Log the ranker as a pyfunc under the active run; returns its model URI."""
    info = mlflow.pyfunc.log_model(
        artifact_path="ranker",
        python_model=RankerPyfunc(),
        artifacts={"ranker_dir": str(ranker.save(directory))},
        code_paths=CODE_PATHS,
        pip_requirements=_pip_requirements(),
    )
    return info.model_uri


def load_champion(uri: str) -> Ranker | None:
    tracking_uri = mlflow.get_tracking_uri()
    try:
        return mlflow.pyfunc.load_model(uri).unwrap_python_model().ranker
    except Exception as exc:  # noqa: BLE001 — no champion is an expected state; callers fall back
        LOGGER.warning(
            "search ranker %s unavailable via MLflow tracking URI %s (%s)", uri, tracking_uri, exc
        )
        return None
