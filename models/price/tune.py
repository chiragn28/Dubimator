"""Optuna search over XGBoost hyperparameters (weighted val RMSE of the relative target)."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import optuna
import xgboost as xgb

from models.price.boosting import fit_xgb
from models.price.config import TrainConfig

TrialCallback = Callable[[int, dict, float, int, float], None]


@dataclass(frozen=True)
class TuneResult:
    best_params: dict
    best_iteration: int
    best_value: float
    n_trials: int


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 64.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def tune_xgboost(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    device: str,
    config: TrainConfig,
    on_trial: TrialCallback | None = None,
) -> TuneResult:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        started = time.perf_counter()
        booster = fit_xgb(params, dtrain, dval, device, config)
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        value = float(booster.best_score)
        if on_trial is not None:
            on_trial(
                trial.number,
                params,
                value,
                int(booster.best_iteration),
                time.perf_counter() - started,
            )
        return value

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(objective, n_trials=config.n_trials)
    best = study.best_trial
    return TuneResult(
        best_params=dict(best.params),
        best_iteration=int(best.user_attrs["best_iteration"]),
        best_value=float(best.value),
        n_trials=len(study.trials),
    )
