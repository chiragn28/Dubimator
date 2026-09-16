"""Per-horizon training: walk-forward tuning, conformal calibration, final fit and the gate.

Spec: "Models", "Intervals" and "Evaluation and gate" in
docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md.
"""

import dataclasses
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from models.forecast.baselines import baseline_growth
from models.forecast.config import HORIZON_SPECS, ForecastConfig, Horizon
from models.forecast.evaluate import (
    MODEL,
    ape,
    bootstrap_mape_upper,
    gate,
    segment_table,
    top_areas,
)
from models.forecast.features import fit_categories, to_matrix
from models.forecast.folds import Fold, fold_split, insufficiency, plan_folds
from models.forecast.infra import INFRA_COLUMNS
from models.forecast.intervals import coverage_by_segment, fit_intervals, interval_segments
from models.forecast.model import ForecastModel, log_forecast_model
from models.forecast.rows import DataQuality
from models.forecast.targets import TargetReport
from models.price.boosting import make_dmatrix, resolve_device
from models.price.registry import CHAMPION_ALIAS, configure, log_metrics, register_champion

FOLD_SCORE_SCHEMA = {
    "fold": pl.Int64, "role": pl.Utf8, "model": pl.Utf8, "rows": pl.Int64, "mape": pl.Float64,
}  # fmt: skip


def base_params(config: ForecastConfig) -> dict:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
    }


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 64.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def fit_booster(train, val, params, categories, device, config, label, num_rounds=None):
    full = {**base_params(config), **params, "device": device, "seed": config.seed}
    dtrain = make_dmatrix(to_matrix(train, categories), label=train[label].to_numpy())
    if val is None:
        return xgb.train(full, dtrain, num_boost_round=num_rounds), num_rounds
    dval = make_dmatrix(to_matrix(val, categories), label=val[label].to_numpy())
    booster = xgb.train(
        full,
        dtrain,
        num_boost_round=config.n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )
    return booster, int(booster.best_iteration) + 1


def predict_rounds(booster, frame, categories, rounds) -> np.ndarray:
    return booster.predict(make_dmatrix(to_matrix(frame, categories)), iteration_range=(0, rounds))


def _run_fold(frame, horizon, fold, params, categories, device, config):
    """(validation rows, predictions, rounds) or None when the fold has no validation rows."""
    label = f"growth_{horizon.name}"
    train, val = fold_split(frame, horizon, fold)
    if val.height == 0 or train.height == 0:
        return None
    booster, rounds = fit_booster(train, val, params, categories, device, config, label)
    return val, predict_rounds(booster, val, categories, rounds), rounds


def tune(frame, horizon, folds, categories, device, config) -> dict:
    label = f"growth_{horizon.name}"
    tuning = [fold for fold in folds if fold.role == "tune"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    base = base_params(config)
    study.enqueue_trial({name: base[name] for name in (
        "max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree",
        "reg_lambda",
    )})  # fmt: skip

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores = []
        for fold in tuning:
            outcome = _run_fold(frame, horizon, fold, params, categories, device, config)
            if outcome is not None:
                val, predicted, _ = outcome
                scores.append(float(np.mean(ape(predicted, val[label].to_numpy()))))
        return float(np.mean(scores)) if scores else float("inf")

    study.optimize(objective, n_trials=config.n_trials)
    return dict(study.best_params)


@dataclass
class HorizonResult:
    horizon: str
    status: str  # "passed", "failed" or "insufficient_data"
    reasons: tuple[str, ...]
    folds: list[Fold]
    table: pl.DataFrame | None = None
    fold_scores: pl.DataFrame | None = None
    coverage: dict[str, float] = dataclasses.field(default_factory=dict)
    upper: float | None = None
    model: ForecastModel | None = None
    seconds: float = 0.0

    def metrics(self) -> dict[str, float]:
        name = self.horizon
        out = {
            f"gate.{name}.passed": float(self.status == "passed"),
            f"{name}.folds": len(self.folds),
        }
        if self.table is not None:
            for row in self.table.iter_rows(named=True):
                prefix = f"{name}.test.{row['segment']}.{row['model']}"
                out[f"{prefix}.mape"] = row["mape"]
                out[f"{prefix}.median_ape"] = row["median_ape"]
                out[f"{prefix}.rows"] = row["rows"]
        if self.fold_scores is not None and self.fold_scores.height:
            means = self.fold_scores.group_by("model").agg(pl.col("mape").mean())
            for model, value in means.iter_rows():
                out[f"{name}.folds.{model}.mape_mean"] = value
        for segment, share in self.coverage.items():
            out[f"{name}.coverage.{segment}"] = share
        if self.upper is not None:
            out[f"{name}.test.all.model.mape_upper95"] = self.upper
        out[f"{name}.seconds"] = self.seconds
        return out


def run_horizon(
    frame: pl.DataFrame, horizon: Horizon, device: str, config: ForecastConfig, data_end: date
) -> HorizonResult:
    started = time.perf_counter()
    label = f"growth_{horizon.name}"
    folds = plan_folds(frame, horizon, config)
    train, test = fold_split(frame, horizon, folds[-1]) if folds else (frame.head(0), frame.head(0))
    reason = insufficiency(folds, test.height, config)
    if reason is not None:
        return HorizonResult(
            horizon.name, "insufficient_data", (f"{horizon.name}: {reason}",), folds,
            seconds=time.perf_counter() - started,
        )  # fmt: skip
    categories = fit_categories(train, config)
    params = tune(frame, horizon, folds, categories, device, config)

    scores, calibration_segments, calibration_errors, rounds = [], [], [], []
    for fold in folds[:-1]:
        outcome = _run_fold(frame, horizon, fold, params, categories, device, config)
        if outcome is None:
            continue
        val, predicted, used = outcome
        actual = val[label].to_numpy()
        for name, values in {MODEL: predicted, **baseline_growth(val, horizon)}.items():
            scores.append(
                {"fold": fold.index, "role": fold.role, "model": name, "rows": val.height,
                 "mape": float(np.mean(ape(values, actual)))}
            )  # fmt: skip
        if fold.role == "tune":
            calibration_segments += interval_segments(val).to_list()
            calibration_errors += np.abs(predicted - actual).tolist()
            rounds.append(used)
    intervals = fit_intervals(calibration_segments, calibration_errors, config)
    final_rounds = max(1, round(float(np.mean(rounds)))) if rounds else config.n_estimators

    booster, _ = fit_booster(train, None, params, categories, device, config, label, final_rounds)
    actual = test[label].to_numpy()
    predicted = predict_rounds(booster, test, categories, final_rounds)
    top = top_areas(train, config.top_areas)
    table = segment_table(test, {MODEL: predicted, **baseline_growth(test, horizon)}, actual, top)
    upper = bootstrap_mape_upper(ape(predicted, actual), config.n_bootstrap, config.seed)
    verdict = gate(table, horizon, upper)
    coverage = coverage_by_segment(interval_segments(test), predicted, actual, intervals)
    test_fold = folds[-1]
    model = ForecastModel(
        horizon=horizon.name,
        booster=booster,
        rounds=final_rounds,
        categories=categories,
        intervals=intervals,
        area_rows=dict(train.group_by("area_id").len().iter_rows()),
        metadata={
            "params": params,
            "data_end": data_end.isoformat(),
            "test_cutoff": test_fold.cutoff.isoformat(),
            "test_end": test_fold.end.isoformat(),
            "top_areas": top,
            "train_rows": train.height,
            "test_rows": test.height,
            "gate": dict(verdict.checks),
        },
    )
    return HorizonResult(
        horizon=horizon.name,
        status="passed" if verdict.passed else "failed",
        reasons=verdict.reasons,
        folds=folds,
        table=table,
        fold_scores=pl.DataFrame(scores, schema=FOLD_SCORE_SCHEMA),
        coverage=coverage,
        upper=upper,
        model=model,
        seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class TrainingSummary:
    device: str
    run_id: str
    results: dict[str, HorizonResult]
    versions: dict[str, str]
    seconds: float


def _log_horizon(result: HorizonResult) -> None:
    name = result.horizon
    log_metrics(result.metrics())
    mlflow.set_tags(
        {
            f"gate.{name}.status": result.status,
            f"gate.{name}.reason": result.reasons[0] if result.reasons else "passed",
        }
    )
    mlflow.log_dict({"folds": [fold.to_dict() for fold in result.folds]}, f"{name}/folds.json")
    if result.table is not None:
        mlflow.log_text(result.table.write_csv(), f"{name}/test_segments.csv")
        mlflow.log_text(result.fold_scores.write_csv(), f"{name}/fold_scores.csv")
    if result.model is not None:
        mlflow.log_text(result.model.importance().write_csv(), f"{name}/importance.csv")
        mlflow.log_dict(result.model.intervals, f"{name}/intervals.json")


def remove_champion(name: str) -> bool:
    """Drop the champion alias of a registered model; False when there was none to drop."""
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        client.delete_registered_model_alias(name, CHAMPION_ALIAS)
    except MlflowException:
        return False
    return True


def run_training(
    frame: pl.DataFrame,
    report: TargetReport,
    quality: DataQuality,
    projects: pl.DataFrame,
    data_end: date,
    config: ForecastConfig,
    *,
    device: str = "auto",
    register: bool = True,
    horizons: tuple[Horizon, ...] | None = None,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> TrainingSummary:
    started = time.perf_counter()
    device = resolve_device(device)
    horizons = horizons or tuple(HORIZON_SPECS.values())
    configure(config.experiment, tracking_uri, artifact_location)
    results, versions = {}, {}
    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as scratch:
        mlflow.set_tag("gate_run", "true")  # latest_gates reads only finished runs with this tag
        params = {key: str(value) for key, value in dataclasses.asdict(config).items()}
        mlflow.log_params({**params, "resolved_device": device, "data_end": data_end.isoformat()})
        mlflow.log_dict(
            {"quality": quality.to_dict(), "targets": report.to_dict()}, "data_quality.json"
        )
        mlflow.log_text(projects.select(INFRA_COLUMNS).write_csv(), "infrastructure_projects.csv")
        for horizon in horizons:
            result = run_horizon(frame, horizon, device, config, data_end)
            results[horizon.name] = result
            _log_horizon(result)
            name = f"{config.model_prefix}-{horizon.name}"
            if register and result.status == "passed":
                uri = log_forecast_model(
                    result.model, Path(scratch) / horizon.name, f"model_{horizon.name}"
                )
                versions[horizon.name] = register_champion(uri, name)
            elif register:
                removed = remove_champion(name)
                mlflow.set_tag(f"gate.{horizon.name}.champion_removed", str(removed).lower())
        mlflow.set_tag("registered", ",".join(sorted(versions)) or "none")
    return TrainingSummary(
        device, run.info.run_id, results, versions, time.perf_counter() - started
    )
