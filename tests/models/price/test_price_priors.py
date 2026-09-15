import math

import polars as pl
import pytest

from models.price.features import (
    LocationPriorError,
    LocationPriors,
    add_location_keys,
    oof_priors,
)
from models.price.split import grouped_folds

SCHEMA = {
    "property_type": pl.Utf8,
    "area_id": pl.Int64,
    "project_name": pl.Utf8,
    "building_name": pl.Utf8,
    "y": pl.Float64,
    "bulk_weight": pl.Float64,
    "bulk_group": pl.UInt64,
}
DEFAULTS = {"property_type": "unit", "area_id": 1, "project_name": None, "building_name": None,
            "y": 0.0, "bulk_weight": 1.0}  # fmt: skip


def make_rows(*entries):
    records = [{**DEFAULTS, "bulk_group": i, **entry} for i, entry in enumerate(entries)]
    return add_location_keys(pl.DataFrame(records, schema=SCHEMA))


MARINA = {"area_id": 1, "project_name": "Marina Gate", "building_name": "Tower 1", "y": 1.0}
FIT = make_rows(*[MARINA] * 2, *[{"area_id": 2}] * 8)
# city = 2/10 = 0.2; area 1 = (2 + 10*0.2)/12 = 1/3; project = (2 + 10/3)/12 = 4/9;
# building = (2 + 40/9)/12 = 29/54


def test_shrinkage_matches_hand_computed_values():
    priors = LocationPriors.fit(FIT, shrink_k=10.0, min_level_n=3.0)
    row = priors.transform(make_rows(MARINA)).row(0, named=True)
    assert row["prior_area"] == pytest.approx(1 / 3)
    assert row["prior_project"] == pytest.approx(4 / 9)
    assert row["prior_building"] == pytest.approx(29 / 54)
    assert row["n_area"] == pytest.approx(math.log1p(2))
    assert row["loc_level"] == 0  # every level has only 2 effective sales, below 3


def test_unseen_levels_fall_back_to_their_parent():
    priors = LocationPriors.fit(FIT, shrink_k=10.0, min_level_n=2.0)
    rows = priors.transform(
        make_rows(
            MARINA,
            {"area_id": 1, "project_name": "Marina Gate", "building_name": "Tower 9"},
            {"area_id": 1, "project_name": "Unknown Project"},
            {"area_id": 3},
        )
    )
    assert rows["loc_level"].to_list() == [3, 2, 1, 0]
    unseen_building = rows.row(1, named=True)
    assert unseen_building["prior_building"] == pytest.approx(4 / 9)
    assert unseen_building["n_building"] == 0.0
    unseen_project = rows.row(2, named=True)
    assert unseen_project["prior_project"] == pytest.approx(1 / 3)
    assert unseen_project["prior_building"] == pytest.approx(1 / 3)
    unseen_area = rows.row(3, named=True)
    assert unseen_area["prior_area"] == pytest.approx(0.2)
    assert unseen_area["prior_building"] == pytest.approx(0.2)


def test_names_are_matched_by_normalized_key():
    priors = LocationPriors.fit(FIT, shrink_k=10.0, min_level_n=2.0)
    row = priors.transform(
        make_rows({"area_id": 1, "project_name": "AL MARINA-GATE", "building_name": " tower  1 "})
    ).row(0, named=True)
    assert row["loc_level"] == 3
    assert row["prior_building"] == pytest.approx(29 / 54)


def test_sparse_area_resolves_to_city_level():
    fit = make_rows(*[{"area_id": 5}] * 2, *[{"area_id": 6}] * 3)
    rows = LocationPriors.fit(fit, 10.0, 3.0).transform(make_rows({"area_id": 5}, {"area_id": 6}))
    assert rows["loc_level"].to_list() == [0, 1]


def test_transform_keeps_row_order():
    priors = LocationPriors.fit(FIT, 10.0, 3.0)
    frame = make_rows({"area_id": 2}, MARINA, {"area_id": 2})
    out = priors.transform(frame)
    assert out["area_id"].to_list() == [2, 1, 2]
    assert out.height == 3


def test_out_of_fold_priors_hide_a_bulk_sale_from_its_own_group():
    normal = [{"building_name": "Tower B", "y": 0.0} for _ in range(10)]
    bulk = {"building_name": "Tower B", "y": 5.0, "bulk_weight": 0.5}
    frame = make_rows(*normal, bulk, bulk).with_columns(
        pl.when(pl.col("y") == 5.0).then(pl.lit(99, pl.UInt64)).otherwise(pl.col("bulk_group"))
        .alias("bulk_group")
    )  # fmt: skip
    folds = grouped_folds(frame, 2)
    assert folds[10] == folds[11]  # identical sales share a fold

    oof = oof_priors(frame, folds, shrink_k=10.0, min_level_n=3.0)
    in_sample = LocationPriors.fit(frame, 10.0, 3.0).transform(frame)
    assert oof.height == frame.height
    assert oof["prior_building"][10] < 0.1  # never saw the 5.0 prices
    assert in_sample["prior_building"][10] > 0.4  # in-sample would have leaked them


def test_missing_property_type_raises():
    priors = LocationPriors.fit(FIT, 10.0, 3.0)
    with pytest.raises(LocationPriorError, match="villa"):
        priors.transform(make_rows({"property_type": "villa"}))
