import dataclasses
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from forecast_fixtures import prepared_history

from models.forecast.baselines import BASELINES, baseline_growth
from models.forecast.config import HORIZON_SPECS, ForecastConfig
from models.forecast.folds import (
    Fold,
    add_months,
    fold_split,
    insufficiency,
    month_start,
    plan_folds,
)
from models.forecast.targets import build_targets

THREE_M = HORIZON_SPECS["3m"]


def growth_frame(days_and_growth):
    return pl.DataFrame(
        {
            "instance_date": [day for day, _ in days_and_growth],
            "growth_3m": [value for _, value in days_and_growth],
            "growth_1y": [None] * len(days_and_growth),
            "growth_3y": [None] * len(days_and_growth),
        },
        schema={
            "instance_date": pl.Date,
            "growth_3m": pl.Float64,
            "growth_1y": pl.Float64,
            "growth_3y": pl.Float64,
        },
    )


def test_month_helpers():
    assert month_start(date(2017, 4, 27)) == date(2017, 4, 1)
    assert add_months(date(2017, 11, 1), 3) == date(2018, 2, 1)
    assert add_months(date(2017, 5, 1), -5) == date(2016, 12, 1)
    with pytest.raises(ValueError):
        add_months(date(2017, 5, 2), 1)


def test_fold_plan_by_hand():
    # first T 2015-01-10 + 731 d + 107 d = 2017-04-27, so the first cutoff is 2017-05-01
    frame = growth_frame(
        [(date(2015, 1, 10), 0.1), (date(2018, 1, 10), 0.2), (date(2016, 1, 1), None)]
    )
    folds = plan_folds(frame, THREE_M, ForecastConfig())
    assert [fold.cutoff for fold in folds] == [
        date(2017, 5, 1),
        date(2017, 8, 1),
        date(2017, 11, 1),
    ]
    assert [fold.end for fold in folds] == [date(2017, 8, 1), date(2017, 11, 1), date(2018, 2, 1)]
    assert [fold.role for fold in folds] == ["tune", "tune", "test"]
    roles = [fold.role for fold in plan_folds(frame, THREE_M, ForecastConfig(tune_folds=1))]
    assert roles == ["score", "tune", "test"]
    assert folds[0].to_dict() == {
        "index": 0, "cutoff": "2017-05-01", "end": "2017-08-01", "role": "tune",
    }  # fmt: skip
    assert plan_folds(growth_frame([(date(2015, 1, 10), None)]), THREE_M, ForecastConfig()) == []


def test_split_respects_the_target_window_guard():
    fold = Fold(0, date(2017, 5, 1), date(2017, 8, 1), "test")
    last_ok = date(2017, 5, 1) - timedelta(days=108)  # its window ends 2017-04-30
    too_late = date(2017, 5, 1) - timedelta(days=107)  # its window ends on the cutoff
    frame = growth_frame(
        [(last_ok, 0.1), (too_late, 0.1), (date(2017, 5, 1), 0.2), (date(2017, 7, 31), 0.3),
         (date(2017, 8, 1), 0.4), (date(2017, 6, 1), None)]
    )  # fmt: skip
    train, val = fold_split(frame, THREE_M, fold)
    assert train["instance_date"].to_list() == [last_ok]
    assert val["instance_date"].to_list() == [date(2017, 5, 1), date(2017, 7, 31)]


def test_every_history_fold_is_leak_free():
    rows, data_end = prepared_history()
    frame, _ = build_targets(rows, data_end)
    for horizon in HORIZON_SPECS.values():
        folds = plan_folds(frame, horizon, ForecastConfig())
        for fold in folds:
            train, val = fold_split(frame, horizon, fold)
            ends = train["instance_date"].to_list()
            assert all(day + timedelta(days=horizon.end_days) < fold.cutoff for day in ends)
            assert val["instance_date"].is_between(fold.cutoff, fold.end, closed="left").all()
            assert val[f"growth_{horizon.name}"].null_count() == 0
        cutoffs = [fold.cutoff for fold in folds]
        assert cutoffs == sorted(cutoffs)
        assert all(
            add_months(a, horizon.step_months) == b
            for a, b in zip(cutoffs, cutoffs[1:])  # noqa: RUF007
        )
    assert len(plan_folds(frame, HORIZON_SPECS["3m"], ForecastConfig())) > 10


def test_insufficiency_reasons():
    config = ForecastConfig()
    fold = Fold(0, date(2020, 1, 1), date(2020, 4, 1), "test")
    assert insufficiency([fold], 5_000, config) == "only 1 walk-forward folds (needs 2)"
    assert insufficiency([fold, fold], 999, config) == "only 999 test rows (needs 1,000)"
    assert insufficiency([fold, fold], 1_000, config) is None
    assert insufficiency([], 0, dataclasses.replace(config, min_folds=0, min_test_rows=0)) is None


def test_baselines_fall_back_from_area_to_city_to_zero():
    frame = pl.DataFrame(
        {
            "area_mom_3m": [0.1, None, None],
            "city_mom_3m": [0.2, 0.3, None],
            "area_mom_12m": [0.5, 0.5, 0.5],
            "city_mom_12m": [0.0, 0.0, 0.0],
        },
        schema=dict.fromkeys(
            ("area_mom_3m", "city_mom_3m", "area_mom_12m", "city_mom_12m"), pl.Float64
        ),
    )
    three = baseline_growth(frame, THREE_M)
    assert list(three) == list(BASELINES)
    assert three["no_change"].tolist() == [0.0, 0.0, 0.0]
    assert three["area_trend"].tolist() == [0.1, 0.3, 0.0]
    year = baseline_growth(frame, HORIZON_SPECS["1y"])
    assert year["area_trend"].tolist() == [0.5, 0.5, 0.5]
    assert isinstance(year["area_trend"], np.ndarray)
