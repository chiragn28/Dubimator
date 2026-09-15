import dataclasses
import math

import numpy as np
import pandas as pd

from models.price.boosting import make_dmatrix
from models.price.config import TrainConfig
from models.price.tune import tune_xgboost

CONFIG = dataclasses.replace(TrainConfig(), n_trials=2, max_rounds=200, early_stopping_rounds=10)
RANGES = {
    "max_depth": (4, 12),
    "learning_rate": (0.02, 0.3),
    "min_child_weight": (1.0, 64.0),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_lambda": (1e-3, 10.0),
    "reg_alpha": (1e-3, 10.0),
}


def test_tune_runs_trials_and_reports_the_best():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    y = x["a"].to_numpy() + rng.normal(scale=0.1, size=500)
    calls = []
    result = tune_xgboost(
        make_dmatrix(x[:400], y[:400]),
        make_dmatrix(x[400:], y[400:]),
        "cpu",
        CONFIG,
        on_trial=lambda number, params, value, best_iteration, seconds: calls.append(
            (number, value, best_iteration)
        ),
    )
    assert result.n_trials == 2
    assert [number for number, _, _ in calls] == [0, 1]
    assert result.best_value == min(value for _, value, _ in calls)
    assert math.isfinite(result.best_value) and result.best_iteration >= 0
    assert set(result.best_params) == set(RANGES)
    for name, (low, high) in RANGES.items():
        assert low <= result.best_params[name] <= high, name
