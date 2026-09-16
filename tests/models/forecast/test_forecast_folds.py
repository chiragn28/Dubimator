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
    # first T 2015-01-10 + 731 d + 107 d = 2017-04-27, so the earliest cutoff is 2017-05-01.
    # last T 2018-01-10: last = month_start(2018-01-10) - (3 - 1) months = 2017-11-01, so the
    # test period [2017-11-01, 2018-02-01) is a full step containing the last T. Walking back
    # by 3 months: 2017-11-01, 2017-08-01, 2017-05-01 (2017-02-01 is before the earliest).
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
    # gap: end + 107 d > test cutoff 2017-11-01.
    #   fold 1 ends 2017-11-01 -> 2018-02-16 > 2017-11-01: gap
    #   fold 0 ends 2017-08-01 -> 2017-11-16 > 2017-11-01: gap
    assert [fold.role for fold in folds] == ["gap", "gap", "test"]
    roles = [fold.role for fold in plan_folds(frame, THREE_M, ForecastConfig(tune_folds=1))]
    assert roles == ["gap", "gap", "test"]
    assert folds[0].to_dict() == {
        "index": 0, "cutoff": "2017-05-01", "end": "2017-08-01", "role": "gap",
    }  # fmt: skip
    assert insufficiency(folds, 5_000, ForecastConfig()) == (
        "no tuning fold ends before the test period's target windows"
    )
    assert plan_folds(growth_frame([(date(2015, 1, 10), None)]), THREE_M, ForecastConfig()) == []


def test_fold_roles_by_hand_with_tune_and_score_folds():
    # earliest 2017-05-01 as above; last T 2019-01-10 -> last = 2018-11-01 (the test fold).
    # Cutoffs: 2017-05, 2017-08, 2017-11, 2018-02, 2018-05, 2018-08, 2018-11 (indices 0..6).
    #   2018-08 ends 2018-11-01, + 107 d = 2019-02-16 > 2018-11-01: gap
    #   2018-05 ends 2018-08-01, + 107 d = 2018-11-16 > 2018-11-01: gap
    #   2018-02 ends 2018-05-01, + 107 d = 2018-08-16 <= 2018-11-01: not gap
    # With tune_folds=2 the latest two remaining (2017-11, 2018-02) tune; the rest score.
    frame = growth_frame([(date(2015, 1, 10), 0.1), (date(2019, 1, 10), 0.2)])
    folds = plan_folds(frame, THREE_M, ForecastConfig(tune_folds=2))
    assert [fold.cutoff for fold in folds][-1] == date(2018, 11, 1)
    assert [fold.role for fold in folds] == [
        "score", "score", "tune", "tune", "gap", "gap", "test",
    ]  # fmt: skip
    four = [fold.role for fold in plan_folds(frame, THREE_M, ForecastConfig())]
    assert four == ["tune", "tune", "tune", "tune", "gap", "gap", "test"]
    assert insufficiency(folds, 5_000, ForecastConfig()) is None


def test_the_test_fold_is_a_full_step_containing_the_last_t():
    # last T 2019-02-04 is 3 days after a month start: last = 2019-02-01 - 2 months =
    # 2018-12-01, so the test fold is [2018-12-01, 2019-03-01), a full 3 months that ends after
    # the last T. (A grid walked forward from 2017-05-01 would test [2019-02-01, 2019-05-01),
    # which holds only 3 days of rows.)
    frame = growth_frame([(date(2015, 1, 10), 0.1), (date(2019, 2, 4), 0.2)])
    folds = plan_folds(frame, THREE_M, ForecastConfig())
    test = folds[-1]
    assert test.role == "test"
    assert (test.cutoff, test.end) == (date(2018, 12, 1), date(2019, 3, 1))
    assert test.cutoff <= date(2019, 2, 4) < test.end
    assert folds[0].cutoff == date(2017, 6, 1)  # the first step back that is >= 2017-05-01


def test_one_year_folds_step_six_months_back_from_the_last_t():
    # 1y: earliest = month after 2015-01-10 + 731 + 396 d = 2018-02-15 -> 2018-03-01.
    # last T 2020-04-20: last = 2020-04-01 - 5 months = 2019-11-01; test [2019-11-01, 2020-05-01).
    # Back by 6 months: 2019-11, 2019-05, 2018-11, 2018-05; 2017-11 is before the earliest.
    #   2019-05 ends 2019-11-01 -> gap; 2018-11 ends 2019-05-01, + 396 d = 2020-05-30 -> gap;
    #   2018-05 ends 2018-11-01, + 396 d = 2019-12-02 > 2019-11-01 -> gap.
    year = HORIZON_SPECS["1y"]
    frame = growth_frame([(date(2015, 1, 10), None), (date(2020, 4, 20), None)]).with_columns(
        pl.Series("growth_1y", [0.1, 0.2], dtype=pl.Float64)
    )
    folds = plan_folds(frame, year, ForecastConfig())
    assert [fold.cutoff for fold in folds] == [
        date(2018, 5, 1), date(2018, 11, 1), date(2019, 5, 1), date(2019, 11, 1),
    ]  # fmt: skip
    assert folds[-1].end == date(2020, 5, 1)
    assert [fold.role for fold in folds] == ["gap", "gap", "gap", "test"]
    assert all(add_months(f.cutoff, 6) == f.end for f in folds)


def test_no_folds_when_the_last_cutoff_is_before_the_earliest():
    frame = growth_frame([(date(2015, 1, 10), 0.1), (date(2017, 5, 20), 0.2)])
    # earliest 2017-05-01; last = 2017-05-01 - 2 months = 2017-03-01 < earliest
    assert plan_folds(frame, THREE_M, ForecastConfig()) == []


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
        if folds:
            test = folds[-1]
            assert test.role == "test"
            for fold in folds[:-1]:
                reaches = fold.end + timedelta(days=horizon.end_days) > test.cutoff
                assert (fold.role == "gap") == reaches
                if fold.role == "tune":
                    assert fold.end + timedelta(days=horizon.end_days) <= test.cutoff
            assert sum(fold.role == "tune" for fold in folds) <= ForecastConfig().tune_folds
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
    tune = Fold(0, date(2019, 1, 1), date(2019, 4, 1), "tune")
    gap = Fold(1, date(2019, 10, 1), date(2020, 1, 1), "gap")
    assert insufficiency([fold], 5_000, config) == "only 1 walk-forward folds (needs 2)"
    assert insufficiency([gap], 5, config) == "only 1 walk-forward folds (needs 2)"
    no_tune = "no tuning fold ends before the test period's target windows"
    assert insufficiency([gap, fold], 5, config) == no_tune
    assert insufficiency([tune, fold], 999, config) == "only 999 test rows (needs 1,000)"
    assert insufficiency([tune, fold], 1_000, config) is None
    loose = dataclasses.replace(config, min_folds=0, min_test_rows=0)
    assert insufficiency([tune], 0, loose) is None


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
