"""Tune both rankers on the tune split, pick the winner, and gate its registration."""

import math
from dataclasses import dataclass
from pathlib import Path

import optuna
import polars as pl

from models.price.registry import register_champion
from search.config import FEATURES, TRUST_FEATURES, SearchConfig
from search.metrics import per_query_metrics, summarize
from search.ranker import RANKER_CLASSES, Ranker, log_ranker


def _xgboost_space(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
    }


def _lightgbm_space(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


SPACES = {"xgboost": _xgboost_space, "lightgbm": _lightgbm_space}


@dataclass(frozen=True)
class TrainingResult:
    rankers: dict[str, Ranker]
    tune_ndcg: dict[str, float]
    best_params: dict[str, dict]
    winner: str


def mean_ndcg(ranker: Ranker, frame: pl.DataFrame, k: int) -> float:
    scored = frame.with_columns(pl.Series("score", ranker.score(frame)))
    return summarize(per_query_metrics(scored, "score", k))["ndcg_at_10"]


def tune_ranker(kind: str, train, tune, features, config: SearchConfig, n_trials: int):
    cls, space = RANKER_CLASSES[kind], SPACES[kind]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        score = mean_ndcg(cls.fit(train, tune, features, params, config), tune, config.ndcg_k)
        return score if math.isfinite(score) else 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(objective, n_trials=n_trials)
    best_params = dict(study.best_params)
    if kind == "lightgbm":
        best_params["bagging_freq"] = 1
    ranker = cls.fit(train, tune, features, best_params, config)
    return ranker, mean_ndcg(ranker, tune, config.ndcg_k), best_params


def _splits(table: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = table.filter(pl.col("split") == "train").sort("query_id", "fused_pos")
    tune = table.filter(pl.col("split") == "tune").sort("query_id", "fused_pos")
    if train.height == 0 or tune.height == 0:
        raise ValueError("the feature table needs judged queries in both train and tune splits")
    return train, tune


def train_rankers(table: pl.DataFrame, config: SearchConfig, n_trials: int) -> TrainingResult:
    train, tune = _splits(table)
    rankers, tune_ndcg, best_params = {}, {}, {}
    for kind in RANKER_CLASSES:  # xgboost first: it wins ties
        rankers[kind], tune_ndcg[kind], best_params[kind] = tune_ranker(
            kind, train, tune, FEATURES, config, n_trials
        )
    winner = max(RANKER_CLASSES, key=lambda kind: tune_ndcg[kind])
    return TrainingResult(rankers, tune_ndcg, best_params, winner)


def fit_ablation(table: pl.DataFrame, result: TrainingResult, config: SearchConfig) -> Ranker:
    train, tune = _splits(table)
    features = tuple(name for name in FEATURES if name not in TRUST_FEATURES)
    cls = RANKER_CLASSES[result.winner]
    return cls.fit(train, tune, features, result.best_params[result.winner], config)


def gate_passes(ndcg: float, ci_low: float, baseline_ndcg: float) -> bool:
    values = (ndcg, ci_low, baseline_ndcg)
    if not all(math.isfinite(value) for value in values):
        return False
    return ndcg > baseline_ndcg and ci_low > baseline_ndcg


def register_ranker(ranker: Ranker, config: SearchConfig, directory: Path) -> str:
    """Log under the active MLflow run and move the champion alias to the new version."""
    return register_champion(log_ranker(ranker, Path(directory)), config.ranker_name)
