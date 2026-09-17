"""Constants and tunables for the price model.

Spec: docs/superpowers/specs/2026-09-15-phase3-price-model-design.md
"""

from dataclasses import dataclass
from datetime import date

HOME_UNIT_SUB_TYPES = ("Flat", "Hotel Apartment", "Stacked Townhouses")
STAT_EXCLUSION_REASONS = ("price_outlier_low", "price_outlier_high", "suspected_sqft_entry")

# (property_type, size_basis) -> inclusive (min, max) size in m². Training scope and the
# predictor enforce the same bounds.
SIZE_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    ("unit", "built_up"): (12.0, 2_000.0),
    ("villa", "built_up"): (40.0, 3_000.0),
    ("villa", "plot"): (60.0, 20_000.0),
}

CATEGORICAL_FEATURES = (
    "property_type", "reg_type", "size_basis", "sub_kind", "room_kind", "area_id",
)  # fmt: skip
FEATURES = (
    "property_type", "reg_type", "size_basis", "sub_kind", "room_kind",
    "bedrooms", "log_area_sqm", "has_parking", "area_id", "market_index",
    "prior_area", "prior_project", "prior_building",
    "n_area", "n_project", "n_building", "loc_level",
)  # fmt: skip
FORBIDDEN_FEATURES = (
    "price_per_sqm_aed", "price_robust_z", "peer_tier", "exclusion_reason",
    "source_row", "ingest_run_id", "procedure_name", "price_aed",
    "nearest_metro", "nearest_mall", "nearest_landmark",
)  # fmt: skip


@dataclass(frozen=True)
class TrainConfig:
    index_start: date = date(2014, 1, 1)
    train_start: date = date(2015, 1, 1)
    val_start: date = date(2022, 7, 1)
    test_start: date = date(2022, 11, 1)
    min_segment_rows: int = 200
    min_index_sales: int = 30
    shrink_k: float = 10.0
    min_level_n: float = 3.0
    oof_folds: int = 5
    half_life_days: float = 730.5
    comps_min_n: float = 5.0
    bounds_min_area_n: float = 30.0
    min_conformal_rows: int = 200
    n_trials: int = 60
    max_rounds: int = 4_000
    early_stopping_rounds: int = 100
    lgbm_learning_rate: float = 0.05
    seed: int = 42
    gate_ratio: float | None = 0.90  # None disables the acceptance gate (tests only)
    experiment: str = "price-model"
    model_name: str = "dubimator-price"
