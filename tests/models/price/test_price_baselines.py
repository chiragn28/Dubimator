import dataclasses

import numpy as np
import polars as pl
import pytest

from models.price.baselines import comps_b0, lightgbm_b1
from models.price.config import TrainConfig
from models.price.features import build_feature_set, feature_frame
from models.price.split import add_bulk_groups, assign_split

SCHEMA = {
    "area_id": pl.Int64,
    "segment": pl.Utf8,
    "room_kind": pl.Utf8,
    "bedrooms": pl.Float64,
    "y": pl.Float64,
    "bulk_weight": pl.Float64,
}


def rows(*entries):
    """entries: (area_id, segment, bedrooms, y, bulk_weight); bedrooms None means penthouse."""
    return pl.DataFrame(
        [
            {
                "area_id": a, "segment": s,
                "room_kind": "bedrooms" if b is not None else "penthouse",
                "bedrooms": b, "y": y, "bulk_weight": w,
            }
            for a, s, b, y, w in entries
        ],
        schema=SCHEMA,
    )  # fmt: skip


FIT = rows(
    *[(1, "A", 2.0, float(v), 1.0) for v in range(5)],  # area 1, A, 2 BR: y 0..4
    *[(1, "A", 1.0, 10.0, 1.0)] * 3,  # area 1, A, 1 BR: only 3 sales
    *[(2, "A", 2.0, -5.0, 1.0)] * 4,  # area 2, A: 4 sales
)


def test_comps_fall_back_from_room_to_area_to_segment():
    target = rows(*((a, "A", b, 0.0, 1.0) for a, b in ((1, 2.0), (1, 1.0), (1, 3.0), (9, 2.0))))
    predicted = comps_b0(FIT, target, min_n=5.0)
    # level 0 (area, segment, room): median of 0..4 = 2
    # level 1 (area 1, A): weighted median of [0,1,2,3,4,10,10,10] = 3
    # level 2 (segment A): weighted median of [-5 x4, 0..4, 10 x3] = 1
    assert predicted.tolist() == pytest.approx([2.0, 3.0, 3.0, 1.0])


def test_comps_median_respects_bulk_weights():
    fit = rows(*[(3, "B", 2.0, 0.0, 0.25)] * 4, *[(3, "B", 2.0, 1.0, 1.0)] * 2)
    predicted = comps_b0(fit, rows((3, "B", 2.0, 0.0, 1.0)), min_n=1.0)
    assert predicted.tolist() == pytest.approx([1.0])  # the unweighted median would be 0


def test_comps_raise_for_a_segment_with_no_fit_rows():
    with pytest.raises(ValueError, match="no segment-level median"):
        comps_b0(FIT, rows((1, "Z", 2.0, 0.0, 1.0)), min_n=5.0)


def test_lightgbm_baseline_trains_with_early_stopping(synthetic_homes):
    config = dataclasses.replace(
        TrainConfig(), min_index_sales=1, oof_folds=3, max_rounds=200, early_stopping_rounds=10
    )
    split, clean = pl.col("split"), pl.col("is_clean")
    prepared = add_bulk_groups(assign_split(synthetic_homes(), config))
    fs = build_feature_set(
        prepared, config, prepared["instance_date"].max(),
        fit_filter=(split == "train") & clean, apply_filters={"val": (split == "val") & clean},
    )  # fmt: skip
    fit, val = fs.frames["fit"], fs.frames["val"]
    x_fit, x_val = feature_frame(fit, fs.categories), feature_frame(val, fs.categories)
    booster = lightgbm_b1(
        x_fit, fit["y"].to_numpy(), fs.weights,
        x_val, val["y"].to_numpy(), val["bulk_weight"].to_numpy(), config,
    )  # fmt: skip
    predicted = booster.predict(x_val, num_iteration=booster.best_iteration)
    assert predicted.shape == (val.height,)
    assert np.isfinite(predicted).all()
    assert 1 <= booster.best_iteration <= 200
