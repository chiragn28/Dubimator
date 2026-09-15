"""Time-based split, bulk-sale groups, sample weights and grouped folds."""

from datetime import date

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from models.price.config import TrainConfig

# Rows identical on all of these are one bulk sale (one pricing decision). Nulls compare equal.
BULK_KEY = (
    "instance_date", "area_id", "project_name", "building_name",
    "property_type", "property_sub_type", "price_aed",
)  # fmt: skip


def assign_split(frame: pl.DataFrame, config: TrainConfig) -> pl.DataFrame:
    day = pl.col("instance_date")
    split = (
        pl.when(day < config.train_start)
        .then(pl.lit("seed"))
        .when(day < config.val_start)
        .then(pl.lit("train"))
        .when(day < config.test_start)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
    )
    return frame.with_columns(split.alias("split"))


def add_bulk_groups(frame: pl.DataFrame) -> pl.DataFrame:
    key = list(BULK_KEY)
    return frame.with_columns(
        pl.struct(key).hash(seed=0).alias("bulk_group"),
        pl.len().over(key).cast(pl.Int64).alias("group_size"),
    ).with_columns((1.0 / pl.col("group_size")).alias("bulk_weight"))


def sample_weights(frame: pl.DataFrame, fit_end: date, half_life_days: float) -> np.ndarray:
    """Recency weight (halves every half_life_days before fit_end) times bulk weight."""
    age_days = (pl.lit(fit_end) - pl.col("instance_date")).dt.total_days()
    weight = pl.lit(0.5).pow(age_days / half_life_days) * pl.col("bulk_weight")
    return frame.select(weight.alias("w"))["w"].to_numpy()


def grouped_folds(frame: pl.DataFrame, n_folds: int) -> np.ndarray:
    groups = frame["bulk_group"].to_numpy()
    folds = np.empty(frame.height, dtype=np.int64)
    splitter = GroupKFold(n_splits=n_folds)
    for fold, (_, members) in enumerate(splitter.split(np.zeros(frame.height), groups=groups)):
        folds[members] = fold
    return folds
