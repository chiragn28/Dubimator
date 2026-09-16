import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from models.price import boosting
from models.price.boosting import (
    cuda_available,
    fit_xgb,
    make_dmatrix,
    predict_xgb,
    resolve_device,
)
from models.price.config import TrainConfig

CONFIG = dataclasses.replace(TrainConfig(), max_rounds=500, early_stopping_rounds=10)
PARAMS = {"max_depth": 3, "learning_rate": 0.3}


def toy_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = 2.0 * x["a"].to_numpy() - x["b"].to_numpy() + rng.normal(scale=0.1, size=n)
    return x, y


def test_resolve_device_validates_and_falls_back(monkeypatch):
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cuda", "cpu")
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("tpu")
    monkeypatch.setattr(boosting, "cuda_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="no usable CUDA GPU"):
        resolve_device("cuda")


def test_early_stopping_fit_and_prediction():
    x, y = toy_data()
    dtrain = make_dmatrix(x[:300], y[:300])
    dval = make_dmatrix(x[300:], y[300:])
    booster = fit_xgb(PARAMS, dtrain, dval, "cpu", CONFIG)
    assert booster.best_iteration < CONFIG.max_rounds - 1
    predicted = predict_xgb(booster, x[300:])
    assert predicted.shape == (100,)
    assert np.sqrt(np.mean((predicted - y[300:]) ** 2)) < 0.5


def test_fixed_round_fit_needs_num_rounds():
    x, y = toy_data()
    with pytest.raises(ValueError, match="num_rounds"):
        fit_xgb(PARAMS, make_dmatrix(x, y), None, "cpu", CONFIG)
    booster = fit_xgb(PARAMS, make_dmatrix(x, y), None, "cpu", CONFIG, num_rounds=7)
    assert booster.num_boosted_rounds() == 7
    assert predict_xgb(booster, x).shape == (400,)  # no best_iteration: uses all trees


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_available(), reason="no CUDA-capable XGBoost + GPU on this machine")
def test_gpu_training_smoke():
    x, y = toy_data()
    booster = fit_xgb(PARAMS, make_dmatrix(x, y), None, "cuda", CONFIG, num_rounds=5)
    device = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    assert device.startswith("cuda")
    assert np.isfinite(predict_xgb(booster, x)).all()
