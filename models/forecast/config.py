"""Constants and tunables for price forecasting.

Spec: docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md
Months are fixed day counts (1 month = 30.4375 days, rounded) so every window is exact.
"""

from dataclasses import dataclass
from pathlib import Path

from models.price.config import FORBIDDEN_FEATURES as PRICE_FORBIDDEN

DATA_DIR = Path("data/forecast")
BASE_DAYS = 92
MIN_LEVEL_SALES = 5
TRAILING_DAYS = 365
MOMENTUM_LAGS = {"3m": 91, "12m": 365, "36m": 1096}
INFRA_TYPES = ("metro_rail", "mall", "school", "park", "mixed_use", "airport")


@dataclass(frozen=True)
class Horizon:
    name: str
    start_days: int
    end_days: int
    step_months: int
    momentum: str  # the momentum length that matches this horizon (baseline and templates)
    mape_gate: float


HORIZON_SPECS = {
    "3m": Horizon("3m", 76, 107, 3, "3m", 0.15),
    "1y": Horizon("1y", 335, 396, 6, "12m", 0.20),
    "3y": Horizon("3y", 1065, 1126, 12, "36m", 0.30),
}
HORIZONS = tuple(HORIZON_SPECS)

PLOT_VILLA = "villa_plot"  # market_kind of a villa priced on its plot area
CATEGORICAL = ("market_kind", "area_code", "project_code")
FEATURES = (
    "ln_base_ppsm", "base_level_building", "base_n",
    "area_mom_3m", "area_mom_12m", "area_mom_36m",
    "city_mom_3m", "city_mom_12m", "city_mom_36m", "area_share_12m",
    "days_since_building_sale", "building_sales_12m", "area_sales_12m",
    "off_plan", "log_area_sqm", "bedrooms", "building_age_proxy_years",
    *(f"infra_active_{kind}" for kind in INFRA_TYPES),
    "infra_months_to_next", "infra_completed_24m",
    *(f"infra_mix_{kind}" for kind in INFRA_TYPES),
    *CATEGORICAL,
)  # fmt: skip
FORBIDDEN_FEATURES = (
    *PRICE_FORBIDDEN, "ppsm", "is_clean",
    *(f"growth_{name}" for name in HORIZONS),
    *(f"target_ppsm_{name}" for name in HORIZONS),
    *(f"target_n_{name}" for name in HORIZONS),
)  # fmt: skip


@dataclass(frozen=True)
class ForecastConfig:
    sample_rows: int | None = None
    seed: int = 7
    max_drop_share: float = 0.30
    outlier_z: float = 3.0
    outlier_min_group: int = 10
    min_train_months: int = 24
    tune_folds: int = 4
    n_trials: int = 30
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    alpha: float = 0.20
    min_conformal_rows: int = 200
    min_folds: int = 2
    min_test_rows: int = 1_000
    n_bootstrap: int = 1_000
    top_areas: int = 5
    low_confidence_area_rows: int = 50
    min_project_rows: int = 20
    min_driver_contribution: float = 0.005
    device: str = "cuda"
    experiment: str = "price-forecast"
    model_prefix: str = "zestimator-forecast"
    price_model_uri: str = "models:/zestimator-price@champion"
