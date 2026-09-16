"""As-of features per sale: momentum, base context, activity, property and categoricals.

Every value uses only sales dated strictly before T. Rows with a null ppsm (serving-time
query rows) never count. Spec: "Features" in
docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md.
"""

from datetime import date

import pandas as pd
import polars as pl

from models.forecast.config import (
    BASE_DAYS,
    CATEGORICAL,
    FEATURES,
    MIN_LEVEL_SALES,
    MOMENTUM_LAGS,
    TRAILING_DAYS,
    ForecastConfig,
)
from models.forecast.infra import add_infra_features
from models.forecast.targets import TargetReport, build_targets, rolling_stats

DAYS_PER_YEAR = 365.25
OTHER_PROJECT = "(other)"
PROPERTY_FEATURES = ("off_plan", "log_area_sqm", "bedrooms")
CORE_FEATURES = tuple(name for name in FEATURES if not name.startswith("infra_"))
MOMENTUM_LEVELS = ("area", "city")


def _is_sale() -> pl.Expr:
    return pl.col("ppsm").is_not_null()


def add_momentum(frame: pl.DataFrame) -> pl.DataFrame:
    frame = rolling_stats(
        frame,
        "city_key",
        start_days=-BASE_DAYS,
        length_days=BASE_DAYS,
        closed="left",
        prefix="city_base",
    )
    for label, lag in MOMENTUM_LAGS.items():
        for level in MOMENTUM_LEVELS:
            lagged = f"_{level}_lag"
            frame = rolling_stats(
                frame,
                f"{level}_key",
                start_days=-(BASE_DAYS + lag),
                length_days=BASE_DAYS,
                closed="left",
                prefix=lagged,
            )
            enough = (pl.col(f"{level}_base_n") >= MIN_LEVEL_SALES) & (
                pl.col(f"{lagged}_n") >= MIN_LEVEL_SALES
            )
            change = (pl.col(f"{level}_base_ppsm") / pl.col(f"{lagged}_ppsm")).log()
            frame = frame.with_columns(
                pl.when(enough).then(change).alias(f"{level}_mom_{label}")
            ).drop(f"{lagged}_ppsm", f"{lagged}_n")
    return frame


def _trailing_count(frame: pl.DataFrame, key: str, name: str) -> pl.DataFrame:
    counted = rolling_stats(
        frame,
        key,
        start_days=-TRAILING_DAYS,
        length_days=TRAILING_DAYS,
        closed="left",
        prefix="_trailing",
    )
    return counted.with_columns(
        pl.when(pl.col(key).is_not_null()).then(pl.col("_trailing_n")).alias(name)
    ).drop("_trailing_ppsm", "_trailing_n")


def _days_since_previous_sale(frame: pl.DataFrame, key: str, name: str) -> pl.DataFrame:
    sales = (
        frame.filter(_is_sale() & pl.col(key).is_not_null())
        .select(key, pl.col("instance_date").alias("_previous"))
        .unique()
        .sort("_previous")
    )
    joined = frame.sort("instance_date").join_asof(
        sales,
        left_on="instance_date",
        right_on="_previous",
        by=key,
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    days = (pl.col("instance_date") - pl.col("_previous")).dt.total_days().cast(pl.Float64)
    return joined.with_columns(days.alias(name)).drop("_previous").sort("row_id")


def add_activity(frame: pl.DataFrame) -> pl.DataFrame:
    frame = _trailing_count(frame, "building_key", "building_sales_12m")
    frame = _trailing_count(frame, "area_key", "area_sales_12m")
    frame = _trailing_count(frame, "city_key", "_city_sales_12m")
    frame = frame.with_columns(
        pl.when(pl.col("_city_sales_12m") > 0)
        .then(pl.col("area_sales_12m") / pl.col("_city_sales_12m"))
        .alias("area_share_12m")
    ).drop("_city_sales_12m")
    frame = _days_since_previous_sale(frame, "building_key", "days_since_building_sale")
    return _days_since_previous_sale(frame, "area_key", "days_since_area_sale")


def add_property(frame: pl.DataFrame) -> pl.DataFrame:
    first = (
        frame.filter(_is_sale() & pl.col("building_key").is_not_null())
        .group_by("building_key")
        .agg(pl.col("instance_date").min().alias("_first_sale"))
    )
    age = (pl.col("instance_date") - pl.col("_first_sale")).dt.total_days() / DAYS_PER_YEAR
    return (
        frame.join(first, on="building_key", how="left")
        .with_columns(
            pl.when(pl.col("_first_sale") < pl.col("instance_date"))
            .then(age)
            .alias("building_age_proxy_years"),
            pl.col("base_ppsm").log().alias("ln_base_ppsm"),
            (pl.col("base_level") == "building").cast(pl.Float64).alias("base_level_building"),
            (pl.col("reg_type") == "off_plan").cast(pl.Float64).alias("off_plan"),
            pl.col("area_sqm").log().alias("log_area_sqm"),
            pl.col("area_id").cast(pl.Utf8).alias("area_code"),
            pl.col("project_name").alias("project_code"),
        )
        .drop("_first_sale")
        .sort("row_id")
    )


def add_core_features(frame: pl.DataFrame) -> pl.DataFrame:
    return add_property(add_activity(add_momentum(frame)))


def fit_categories(frame: pl.DataFrame, config: ForecastConfig) -> dict[str, list[str]]:
    projects = (
        frame.filter(pl.col("project_code").is_not_null())
        .group_by("project_code")
        .len()
        .filter(pl.col("len") >= config.min_project_rows)
    )
    return {
        "market_kind": sorted(frame["market_kind"].drop_nulls().unique().to_list()),
        "area_code": sorted(frame["area_code"].drop_nulls().unique().to_list(), key=int),
        "project_code": sorted(projects["project_code"].to_list()) + [OTHER_PROJECT],
    }


def to_matrix(
    frame: pl.DataFrame, categories: dict[str, list[str]], features=FEATURES
) -> pd.DataFrame:
    """Exactly `features`, in order, as XGBoost input (categories fixed at fit time)."""
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise KeyError(f"feature columns missing: {missing}")
    known = categories["project_code"]
    frame = frame.with_columns(
        pl.when(pl.col("project_code").is_null() | pl.col("project_code").is_in(known))
        .then(pl.col("project_code"))
        .otherwise(pl.lit(OTHER_PROJECT))
        .alias("project_code")
    )
    selected = frame.select(
        pl.col(name).cast(pl.Utf8) if name in CATEGORICAL else pl.col(name).cast(pl.Float64)
        for name in features
    ).to_pandas()
    for name in CATEGORICAL:
        if name in features:
            selected[name] = pd.Categorical(selected[name], categories=categories[name])
    return selected


def feature_coverage(frame: pl.DataFrame, features) -> dict[str, float]:
    total = max(frame.height, 1)
    return {name: 1.0 - frame[name].null_count() / total for name in features}


def build_dataset(
    rows: pl.DataFrame, data_end: date, projects: pl.DataFrame
) -> tuple[pl.DataFrame, TargetReport]:
    """Targets plus every model feature, as of each sale."""
    frame, report = build_targets(rows, data_end)
    return add_infra_features(add_core_features(frame), projects), report
