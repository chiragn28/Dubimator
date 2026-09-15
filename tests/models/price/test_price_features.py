import dataclasses
import math
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

from models.price.config import CATEGORICAL_FEATURES, FEATURES, TrainConfig
from models.price.features import (
    add_model_columns,
    build_feature_set,
    feature_frame,
    fit_bounds,
    fit_size_percentiles,
)
from models.price.split import add_bulk_groups, assign_split

CONFIG = dataclasses.replace(TrainConfig(), min_index_sales=1, oof_folds=3)
DATA_END = date(2023, 3, 27)
CLEAN, SPLIT = pl.col("is_clean"), pl.col("split")
APPLY = {
    "val": (SPLIT == "val") & CLEAN,
    "test_clean": (SPLIT == "test") & CLEAN,
    "test_honest": SPLIT == "test",
}


def eval_set(rows):
    prepared = add_bulk_groups(assign_split(rows, CONFIG))
    return build_feature_set(
        prepared, CONFIG, DATA_END, fit_filter=(SPLIT == "train") & CLEAN, apply_filters=APPLY
    )


def test_feature_frame_is_exactly_the_allowlist(synthetic_homes):
    fs = eval_set(synthetic_homes())
    features = feature_frame(fs.frames["fit"], fs.categories)
    assert list(features.columns) == list(FEATURES)
    for name in FEATURES:
        expected = "category" if name in CATEGORICAL_FEATURES else "float64"
        assert str(features[name].dtype) == expected, name


def test_feature_frame_ignores_extra_and_forbidden_columns(synthetic_homes):
    fs = eval_set(synthetic_homes())
    frame = fs.frames["val"].with_columns(pl.lit(1.0).alias("price_per_sqm_aed"))
    assert "price_per_sqm_aed" not in feature_frame(frame, fs.categories).columns


def test_unseen_category_becomes_missing(synthetic_homes):
    fs = eval_set(synthetic_homes())
    frame = fs.frames["val"].with_columns(pl.lit(999, pl.Int64).alias("area_id"))
    assert feature_frame(frame, fs.categories)["area_id"].isna().all()


def test_target_is_relative_to_the_market_index(synthetic_homes):
    fit = eval_set(synthetic_homes()).frames["fit"]
    expected = np.log(fit["price_aed"] / fit["area_sqm"]) - fit["market_index"]
    assert fit["y"].to_numpy() == pytest.approx(expected.to_numpy())


def test_features_never_depend_on_same_month_or_later_prices(synthetic_homes):
    rows = synthetic_homes()
    cutoff = CONFIG.test_start  # perturb every price from the first test month on
    perturbed = rows.with_columns(
        pl.when(pl.col("instance_date") >= cutoff)
        .then(pl.col("price_aed") * 3.0)
        .otherwise(pl.col("price_aed"))
        .alias("price_aed")
    )
    base, changed = eval_set(rows), eval_set(perturbed)
    first_month_end = date(2022, 12, 1)
    for name in ("fit", "val", "test_clean", "test_honest"):
        keep = pl.col("instance_date") < first_month_end
        before = feature_frame(base.frames[name].filter(keep), base.categories)
        after = feature_frame(changed.frames[name].filter(keep), changed.categories)
        assert len(before) > 0, name  # every set has rows dated before 2022-12
        pd.testing.assert_frame_equal(before, after)


def test_weights_align_with_fit_rows(synthetic_homes):
    fs = eval_set(synthetic_homes())
    assert fs.weights.shape == (fs.frames["fit"].height,)
    assert 0.0 < fs.weights.min() and fs.weights.max() <= 1.0


def test_add_model_columns():
    frame = add_model_columns(
        pl.DataFrame({"area_sqm": [100.0, 50.0], "has_parking": [True, None]})
    )
    assert frame["log_area_sqm"].to_list() == pytest.approx([math.log(100.0), math.log(50.0)])
    assert frame["has_parking"].to_list() == [1.0, None]


def test_bounds_and_size_percentiles():
    frame = pl.DataFrame(
        {
            "property_type": ["unit"] * 106,
            "size_basis": ["built_up"] * 106,
            "segment": ["unit_ready_built_up"] * 106,
            "area_id": [1] * 101 + [2] * 5,
            "y": [i / 100 for i in range(101)] + [0.0] * 5,
            "bulk_weight": [1.0] * 106,
            "area_sqm": [float(i + 1) for i in range(106)],
        }
    )
    by_area, by_segment = fit_bounds(frame, min_area_n=30.0)
    assert by_area["area_id"].to_list() == [1]  # area 2 has only 5 sales
    assert by_area["lo"][0] == pytest.approx(0.01 - math.log(3.0))
    assert by_area["hi"][0] == pytest.approx(0.99 + math.log(3.0))
    assert by_segment.columns == ["property_type", "size_basis", "lo", "hi"]
    sizes = fit_size_percentiles(frame)
    assert sizes.columns == ["segment", "p005", "p995"]
    assert sizes["p005"][0] == pytest.approx(1.0 + 0.005 * 105)
