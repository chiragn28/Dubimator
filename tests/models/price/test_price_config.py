from datetime import date

from models.price.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    FORBIDDEN_FEATURES,
    SIZE_BOUNDS,
    TrainConfig,
)


def test_feature_allowlist_is_exact():
    assert FEATURES == (
        "property_type", "reg_type", "size_basis", "sub_kind", "room_kind",
        "bedrooms", "log_area_sqm", "has_parking", "area_id", "market_index",
        "prior_area", "prior_project", "prior_building",
        "n_area", "n_project", "n_building", "loc_level",
    )  # fmt: skip


def test_no_forbidden_feature_is_allowed():
    assert not set(FEATURES) & set(FORBIDDEN_FEATURES)
    assert {"price_per_sqm_aed", "price_robust_z", "peer_tier", "procedure_name"} <= set(
        FORBIDDEN_FEATURES
    )


def test_categoricals_are_features():
    assert set(CATEGORICAL_FEATURES) <= set(FEATURES)


def test_size_bounds_match_spec():
    assert SIZE_BOUNDS == {
        ("unit", "built_up"): (12.0, 2_000.0),
        ("villa", "built_up"): (40.0, 3_000.0),
        ("villa", "plot"): (60.0, 20_000.0),
    }


def test_default_dates_are_ordered():
    config = TrainConfig()
    assert config.index_start == date(2014, 1, 1)
    assert config.train_start == date(2015, 1, 1)
    assert config.val_start == date(2022, 7, 1)
    assert config.test_start == date(2022, 11, 1)
    assert config.index_start < config.train_start < config.val_start < config.test_start


def test_default_tunables_match_spec():
    config = TrainConfig()
    assert (config.shrink_k, config.min_level_n, config.half_life_days) == (10.0, 3.0, 730.5)
    assert (config.n_trials, config.max_rounds, config.early_stopping_rounds) == (60, 4_000, 100)
    assert config.gate_ratio == 0.90
    assert (config.experiment, config.model_name) == ("price-model", "dubimator-price")
