"""Home sales for forecasting: validation, dedup, robust outliers, and a quality report."""

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from ingestion.config import DbSettings
from models.forecast.config import ForecastConfig
from models.price.config import TrainConfig
from models.price.data import load_homes

DROP_REASONS = (
    "not_clean", "bad_price", "bad_size", "bad_date", "missing_area",
    "duplicate_transaction_id", "repeat_sale", "outlier",
)  # fmt: skip
EXCLUDED_SCHEMA = {
    "area_id": pl.Int64,
    "sub_kind": pl.Utf8,
    "reg_type": pl.Utf8,
    "month": pl.Date,
    "reason": pl.Utf8,
}
REPEAT_KEY = ("building_name", "instance_date", "area_sqm", "price_aed")
MAD_SCALE = 1.4826


@dataclass(frozen=True)
class DataQuality:
    loaded: int
    dropped: dict[str, int]
    kept: int
    excluded: pl.DataFrame = field(repr=False)
    unscreened_groups: int = 0

    @property
    def drop_share(self) -> float:
        return sum(self.dropped.values()) / self.loaded if self.loaded else 0.0

    def lines(self) -> list[str]:
        total = sum(self.dropped.values())
        out = [f"Dropped {total:,} of {self.loaded:,} rows ({100 * self.drop_share:.1f}%)"]
        out += [f"  {reason:<26}{count:>10,}" for reason, count in self.dropped.items()]
        out.append(f"  outlier groups not screened (<10 sales): {self.unscreened_groups:,}")
        out.append(f"Kept {self.kept:,} rows")
        return out

    def to_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "dropped": dict(self.dropped),
            "kept": self.kept,
            "drop_share": self.drop_share,
            "unscreened_groups": self.unscreened_groups,
        }


def validate_rows(rows: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    dropped = {}
    checks = (
        ("not_clean", pl.col("is_clean").fill_null(False)),
        ("bad_price", pl.col("price_aed").fill_null(0.0) > 0),
        ("bad_size", pl.col("area_sqm").fill_null(0.0) > 0),
        ("bad_date", pl.col("instance_date").is_not_null()),
        ("missing_area", pl.col("area_id").is_not_null()),
    )
    for reason, keep in checks:
        before = rows.height
        rows = rows.filter(keep)
        dropped[reason] = before - rows.height
    before = rows.height
    rows = rows.unique("transaction_id", keep="first", maintain_order=True)
    dropped["duplicate_transaction_id"] = before - rows.height
    before = rows.height
    repeat = rows.filter(pl.col("building_name").is_not_null())
    single = rows.filter(pl.col("building_name").is_null())
    repeat = repeat.unique(list(REPEAT_KEY), keep="last", maintain_order=True)
    rows = pl.concat([repeat, single]).sort("instance_date", "transaction_id")
    dropped["repeat_sale"] = before - rows.height
    return rows, dropped


def screen_outliers(
    rows: pl.DataFrame, config: ForecastConfig
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Robust z of ln ppsm within area x sub_kind x month; |z| > outlier_z is excluded."""
    group = ["area_id", "sub_kind", "_month"]
    frame = rows.with_columns(
        pl.col("instance_date").dt.truncate("1mo").alias("_month"),
        pl.col("ppsm").log().alias("_ln"),
    )
    stats = frame.group_by(group).agg(
        pl.col("_ln").median().alias("_median"),
        (pl.col("_ln") - pl.col("_ln").median()).abs().median().alias("_mad"),
        pl.len().alias("_n"),
    )
    frame = frame.join(stats, on=group, how="left", maintain_order="left")
    screened = (pl.col("_n") >= config.outlier_min_group) & (pl.col("_mad") > 0)
    z = (pl.col("_ln") - pl.col("_median")) / (MAD_SCALE * pl.col("_mad"))
    is_outlier = (screened & (z.abs() > config.outlier_z)).fill_null(False)
    frame = frame.with_columns(is_outlier.alias("_outlier"))
    excluded = frame.filter(pl.col("_outlier")).select(
        "area_id",
        "sub_kind",
        "reg_type",
        pl.col("_month").alias("month"),
        pl.lit("outlier").alias("reason"),
    )
    unscreened = stats.filter((pl.col("_n") < config.outlier_min_group) | (pl.col("_mad") <= 0))
    kept = frame.filter(~pl.col("_outlier")).drop(
        "_month", "_ln", "_median", "_mad", "_n", "_outlier"
    )
    return kept, excluded.cast(EXCLUDED_SCHEMA), unscreened.height


def prepare_rows(rows: pl.DataFrame, config: ForecastConfig) -> tuple[pl.DataFrame, DataQuality]:
    loaded = rows.height
    if config.sample_rows is not None and rows.height > config.sample_rows:
        rows = rows.sample(config.sample_rows, seed=config.seed).sort(
            "instance_date", "transaction_id"
        )
        loaded = rows.height
    kept, dropped = validate_rows(rows)
    kept = kept.with_columns((pl.col("price_aed") / pl.col("area_sqm")).alias("ppsm"))
    kept, excluded, unscreened = screen_outliers(kept, config)
    dropped["outlier"] = excluded.height
    kept = kept.sort("instance_date", "transaction_id").with_row_index("row_id")
    kept = kept.with_columns(pl.col("row_id").cast(pl.Int64))
    quality = DataQuality(loaded, dropped, kept.height, excluded, unscreened)
    return kept, quality


def load_rows(
    settings: DbSettings, config: ForecastConfig
) -> tuple[pl.DataFrame, DataQuality, date, pl.DataFrame, pl.DataFrame]:
    homes = load_homes(settings, TrainConfig())
    rows, quality = prepare_rows(homes.rows, config)
    return rows, quality, rows["instance_date"].max(), homes.areas, homes.aliases


def excluded_summary(excluded: pl.DataFrame) -> list[dict]:
    """Excluded-row counts per area, kind, registration type and reason (for predict)."""
    keys = ["area_id", "sub_kind", "reg_type", "reason"]
    return excluded.group_by(keys).len(name="count").sort(keys, nulls_last=True).to_dicts()
