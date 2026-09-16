from datetime import date

import numpy as np
import polars as pl

from listings.config import LABEL_COLUMNS, PAIR_FEATURES
from listings.features import LISTING_SQL, build_features
from listings.truth import TRUTH_SQL, label_pairs

DIM = 8


def unit(index, size=DIM):
    vector = np.zeros(size)
    vector[index] = 1.0
    return vector


ATTRIBUTES = pl.DataFrame(
    {
        "listing_id": [1, 2, 3],
        "posted_at": [date(2023, 1, 1), date(2023, 1, 8), date(2023, 2, 1)],
        "asking_price_aed": [1_000_000.0, 1_100_000.0, 2_000_000.0],
        "size_sqm": [100.0, 100.0, 200.0],
        "bedrooms": [2, 2, 4],
        "area_id": [10, 10, 20],
        "building_name": ["Tower A", "Tower A", "Tower B"],
        "project_name": ["Marina Gate", "Marina Gate", "Hills"],
        "agent_id": [7, 7, 9],
    }
)


def test_feature_columns_are_exactly_the_allowlist():
    pairs = pl.DataFrame({"listing_a": [1], "listing_b": [2]})
    features = build_features(
        pairs, ATTRIBUTES, {1: unit(0), 2: unit(0), 3: unit(1)}, {1: unit(0), 2: unit(1), 3: unit(2)},
        {1: [100, 101], 2: [100, 102], 3: [103]},
        {100: unit(0), 101: unit(1), 102: unit(1), 103: unit(2)},
    )  # fmt: skip
    assert features.columns == ["listing_a", "listing_b", *PAIR_FEATURES]
    assert not set(features.columns) & set(LABEL_COLUMNS)


def test_features_are_computed_correctly():
    pairs = pl.DataFrame({"listing_a": [1, 1], "listing_b": [2, 3]})
    features = build_features(
        pairs, ATTRIBUTES, {1: unit(0), 2: unit(0), 3: unit(1)}, {1: unit(0), 2: unit(1), 3: unit(2)},
        {1: [100, 101], 2: [100, 102], 3: [103]},
        {100: unit(0), 101: unit(1), 102: unit(1), 103: unit(2)},
    ).to_dicts()  # fmt: skip

    near, far = features[0], features[1]
    assert near["text_cosine"] == 1.0  # identical text vectors
    assert near["image_mean_cosine"] == 0.0  # orthogonal listing image vectors
    assert near["image_max_cosine"] == 1.0  # photo 100 is in both, and 101/102 match too
    assert near["shared_photo_count"] == 1  # only photo 100 is literally the same row
    assert near["abs_log_price_ratio"] == abs(np.log(1_000_000 / 1_100_000))
    assert near["abs_log_size_ratio"] == 0.0
    assert (near["same_area"], near["same_building"], near["same_project"]) == (1, 1, 1)
    assert near["bedrooms_equal"] == 1
    assert near["days_apart"] == 7
    assert near["same_agent"] == 1

    assert far["same_area"] == 0 and far["same_building"] == 0 and far["same_agent"] == 0
    assert far["shared_photo_count"] == 0
    assert far["days_apart"] == 31


def test_missing_bedrooms_are_not_counted_as_equal():
    attributes = ATTRIBUTES.with_columns(pl.Series("bedrooms", [None, None, 4], dtype=pl.Int64))
    features = build_features(
        pl.DataFrame({"listing_a": [1], "listing_b": [2]}), attributes,
        {1: unit(0), 2: unit(0)}, {1: unit(0), 2: unit(0)}, {1: [100], 2: [100]}, {100: unit(0)},
    ).to_dicts()[0]  # fmt: skip
    assert features["bedrooms_equal"] == 0


def test_a_listing_without_photos_scores_zero_image_similarity():
    features = build_features(
        pl.DataFrame({"listing_a": [1], "listing_b": [2]}), ATTRIBUTES,
        {1: unit(0), 2: unit(0)}, {1: unit(0), 2: np.zeros(DIM)}, {1: [100], 2: []}, {100: unit(0)},
    ).to_dicts()[0]  # fmt: skip
    assert features["image_max_cosine"] == 0.0
    assert features["image_mean_cosine"] == 0.0
    assert features["shared_photo_count"] == 0


def test_neither_query_mentions_a_label_column():
    for name in LABEL_COLUMNS:
        assert name not in LISTING_SQL
        assert name in TRUTH_SQL  # truth.py is the one module that may read them


def test_detection_sql_selects_no_free_duplicate_oracle():
    """Provenance and variant columns give duplicates away without any similarity work."""
    from listings import candidates, features
    from listings.config import DETECTION_FORBIDDEN_COLUMNS

    statements = [
        features.LISTING_SQL,
        features.VECTOR_SQL,
        features.PHOTO_SQL,
        candidates.TEXT_SQL,
        candidates.IMAGE_SQL,
        candidates.SHARED_PHOTO_SQL,
    ]
    for statement in statements:
        for name in DETECTION_FORBIDDEN_COLUMNS:
            assert name not in statement


def test_pair_labels_cover_clones_siblings_and_controls():
    truth = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "dup_group_id": [None, 1, 1, None, None],
            "control_group_id": [None, None, None, "bldg:10:Tower A", "bldg:10:Tower A"],
            "fraud_label": [None, None, None, None, "bait_price"],
        },
        schema={
            "listing_id": pl.Int64, "dup_group_id": pl.Int64,
            "control_group_id": pl.Utf8, "fraud_label": pl.Utf8,
        },
    )  # fmt: skip
    pairs = pl.DataFrame({"listing_a": [1, 2, 4, 1], "listing_b": [2, 3, 5, 4]})
    labelled = label_pairs(pairs, truth).to_dicts()
    assert labelled[0]["is_duplicate"] is True  # clone of
    assert labelled[1]["is_duplicate"] is True  # two clones of the same source
    assert labelled[2]["is_duplicate"] is False and labelled[2]["same_building_control"] is True
    assert labelled[3]["is_duplicate"] is False and labelled[3]["same_building_control"] is False
