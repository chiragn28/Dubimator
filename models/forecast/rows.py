"""Home sales for forecasting: validation, dedup, robust outliers, and a quality report."""

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from ingestion.config import DbSettings
from models.forecast.config import PLOT_VILLA, ForecastConfig
from models.price.config import TrainConfig
from models.price.data import load_homes

DROP_REASONS = (
    "not_clean", "bad_price", "bad_size", "bad_date", "missing_area",
    "duplicate_transaction_id", "repeat_sale", "outlier",
)  # fmt: skip
EXCLUDED_SCHEMA = {
    "area_id": pl.Int64,
    "sub_kind": pl.Utf8,
    "market_kind": pl.Utf8,
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


def _positive(column: str) -> pl.Expr:
    """Finite and > 0 (null, NaN and +-inf all fail)."""
    return pl.col(column).is_finite().fill_null(False) & (pl.col(column).fill_null(0) > 0)


def _in_date_order(rows: pl.DataFrame) -> pl.DataFrame:
    return rows.sort("instance_date", "transaction_id", nulls_last=True, maintain_order=True)


def excluded_rows(rows: pl.DataFrame, reason: str) -> pl.DataFrame:
    """Rows in EXCLUDED_SCHEMA form (month = the sale's month) with the given reason."""
    kind = pl.col("market_kind") if "market_kind" in rows.columns else market_kind()
    return rows.select(
        "area_id",
        "sub_kind",
        kind.alias("market_kind"),
        "reg_type",
        pl.col("instance_date").dt.truncate("1mo").alias("month"),
        pl.lit(reason).alias("reason"),
    ).cast(EXCLUDED_SCHEMA)


def validate_rows(rows: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int], pl.DataFrame]:
    """(kept rows, drop counts, excluded repeat sales).

    Rows are put in (instance_date, transaction_id) order first, so duplicate ids keep the
    earliest sale and repeat sales keep the largest transaction_id on their date (DLD has no
    source-row number, so that is what "latest" means).
    """
    dropped = {}
    checks = (
        ("not_clean", pl.col("is_clean").fill_null(False)),
        ("bad_price", _positive("price_aed")),
        ("bad_size", _positive("area_sqm")),
        ("bad_date", pl.col("instance_date").is_not_null()),
        ("missing_area", pl.col("area_id").is_not_null()),
    )
    for reason, keep in checks:
        before = rows.height
        rows = rows.filter(keep)
        dropped[reason] = before - rows.height
    rows = _in_date_order(rows)
    before = rows.height
    rows = rows.unique("transaction_id", keep="first", maintain_order=True)
    dropped["duplicate_transaction_id"] = before - rows.height
    rows = rows.with_row_index("_order")
    repeat = rows.filter(pl.col("building_name").is_not_null())
    single = rows.filter(pl.col("building_name").is_null())
    kept_repeat = repeat.unique(list(REPEAT_KEY), keep="last", maintain_order=True)
    repeats = repeat.join(kept_repeat.select("_order"), on="_order", how="anti")
    rows = pl.concat([kept_repeat, single]).sort("_order").drop("_order")
    dropped["repeat_sale"] = repeats.height
    return rows, dropped, excluded_rows(repeats.sort("_order"), "repeat_sale")


def screen_outliers(
    rows: pl.DataFrame, config: ForecastConfig
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Robust z of ln ppsm within area x market_kind x month; |z| > outlier_z is excluded."""
    group = ["area_id", "market_kind", "_month"]
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
    excluded = excluded_rows(frame.filter(pl.col("_outlier")), "outlier")
    unscreened = stats.filter((pl.col("_n") < config.outlier_min_group) | (pl.col("_mad") <= 0))
    kept = frame.filter(~pl.col("_outlier")).drop(
        "_month", "_ln", "_median", "_mad", "_n", "_outlier"
    )
    return kept, excluded, unscreened.height


def market_kind() -> pl.Expr:
    """sub_kind, except that plot-priced villas are their own market ("villa_plot")."""
    plot = (pl.col("sub_kind") == "villa") & (pl.col("size_basis") == "plot")
    return pl.when(plot).then(pl.lit(PLOT_VILLA)).otherwise(pl.col("sub_kind"))


def prepare_rows(rows: pl.DataFrame, config: ForecastConfig) -> tuple[pl.DataFrame, DataQuality]:
    loaded = rows.height
    rows = _in_date_order(rows)  # a seeded sample is then independent of the input order
    if config.sample_rows is not None and rows.height > config.sample_rows:
        rows = _in_date_order(rows.sample(config.sample_rows, seed=config.seed))
        loaded = rows.height
    kept, dropped, repeats = validate_rows(rows)
    kept = kept.with_columns(
        (pl.col("price_aed") / pl.col("area_sqm")).alias("ppsm"),
        market_kind().alias("market_kind"),
    )
    kept, excluded, unscreened = screen_outliers(kept, config)
    dropped["outlier"] = excluded.height
    kept = kept.sort("instance_date", "transaction_id").with_row_index("row_id")
    kept = kept.with_columns(pl.col("row_id").cast(pl.Int64))
    excluded = pl.concat([repeats, excluded])
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
