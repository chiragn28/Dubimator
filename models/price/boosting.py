"""XGBoost helpers shared by training and serving (no Optuna imports here)."""

import shutil
import subprocess

import numpy as np
import xgboost as xgb

from models.price.config import TrainConfig

BASE_PARAMS = {"objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse"}
DEVICES = ("auto", "cuda", "cpu")


def cuda_available() -> bool:
    if not xgb.build_info().get("USE_CUDA", False):
        return False
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        probe = subprocess.run([smi, "-L"], capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def resolve_device(requested: str) -> str:
    if requested not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {requested!r}")
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    if requested == "cuda" and not cuda_available():
        raise RuntimeError("device 'cuda' requested but XGBoost has no usable CUDA GPU")
    return requested


def make_dmatrix(features, label=None, weight=None) -> xgb.DMatrix:
    return xgb.DMatrix(features, label=label, weight=weight, enable_categorical=True)


def fit_xgb(
    params: dict,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix | None,
    device: str,
    config: TrainConfig,
    num_rounds: int | None = None,
) -> xgb.Booster:
    full = {**BASE_PARAMS, **params, "device": device, "seed": config.seed}
    if dval is None:
        if num_rounds is None:
            raise ValueError("num_rounds is required when there is no validation set")
        return xgb.train(full, dtrain, num_boost_round=num_rounds)
    return xgb.train(
        full,
        dtrain,
        num_boost_round=config.max_rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )


def _best_iteration(booster) -> int | None:
    try:
        return int(booster.best_iteration)
    except AttributeError:
        return None


def predict_xgb(booster, features) -> np.ndarray:
    dmatrix = make_dmatrix(features)
    best = _best_iteration(booster)
    if best is None:
        return booster.predict(dmatrix)
    return booster.predict(dmatrix, iteration_range=(0, best + 1))
