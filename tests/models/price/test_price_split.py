from datetime import date

import polars as pl
import pytest

from models.price.config import TrainConfig
from models.price.split import add_bulk_groups, assign_split, grouped_folds, sample_weights


def test_split_boundaries_are_exact(raw_homes):
    days = [
        date(2014, 12, 31), date(2015, 1, 1), date(2022, 6, 30), date(2022, 7, 1),
        date(2022, 10, 31), date(2022, 11, 1), date(2023, 3, 17),
    ]  # fmt: skip
    frame = assign_split(raw_homes(*({"instance_date": d} for d in days)), TrainConfig())
    assert frame["split"].to_list() == ["seed", "train", "train", "val", "val", "test", "test"]


def test_identical_sales_share_a_bulk_group_even_with_null_project(raw_homes):
    frame = add_bulk_groups(
        raw_homes(
            {"project_name": None},
            {"project_name": None},
            {"project_name": None, "price_aed": 900_000.0},
        )
    )
    groups = frame["bulk_group"].to_list()
    assert groups[0] == groups[1] != groups[2]
    assert frame["group_size"].to_list() == [2, 2, 1]
    assert frame["bulk_weight"].to_list() == [0.5, 0.5, 1.0]


def test_no_bulk_group_spans_two_splits(raw_homes):
    days = [date(2022, 6, 30), date(2022, 7, 1), date(2022, 10, 31), date(2022, 11, 1)]
    frame = raw_homes(*({"instance_date": d} for d in days for _ in range(3)))
    frame = add_bulk_groups(assign_split(frame, TrainConfig()))
    spans = frame.group_by("bulk_group").agg(pl.col("split").n_unique().alias("n"))
    assert spans["n"].max() == 1
    assert frame["group_size"].unique().to_list() == [3]


def test_recency_weight_halves_per_half_life_and_bulk_weight_divides(raw_homes):
    fit_end = date(2022, 6, 30)
    frame = add_bulk_groups(
        raw_homes(
            {"instance_date": fit_end},
            {"instance_date": date(2020, 6, 30)},  # 730 days before fit_end
            *({"instance_date": fit_end, "price_aed": 700_000.0} for _ in range(4)),
        )
    )
    weights = sample_weights(frame, fit_end, 730.5)
    assert weights.tolist() == pytest.approx([1.0, 0.5 ** (730 / 730.5), 0.25, 0.25, 0.25, 0.25])
    assert sample_weights(frame, fit_end, 730.0)[1] == pytest.approx(0.5)


def test_grouped_folds_keep_each_bulk_group_in_one_fold(raw_homes):
    rows = [{"price_aed": 500_000.0 + 1_000.0 * (i // 2)} for i in range(20)]  # 10 groups of 2
    frame = add_bulk_groups(raw_homes(*rows))
    folds = grouped_folds(frame, 5)
    per_group = (
        pl.DataFrame({"g": frame["bulk_group"], "f": folds})
        .group_by("g")
        .agg(pl.col("f").n_unique())
    )
    assert per_group["f"].max() == 1
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]
