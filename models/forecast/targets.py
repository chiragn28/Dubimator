"""Base price and forward growth targets per sale.

Spec: docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md ("Targets").
Every window is an exact day range relative to the sale date T (see config). Rows with a
null ppsm (serving-time query rows) never count toward any window.
"""

from dataclasses import dataclass
from datetime import date

import polars as pl

from ingestion.normalize import map_unique, match_key
from models.forecast.config import BASE_DAYS, HORIZON_SPECS, HORIZONS, MIN_LEVEL_SALES

STATUSES = ("usable", "no_base", "no_target", "window_open")
LEVELS = ("building", "area")


def add_keys(frame: pl.DataFrame) -> pl.DataFrame:
    """building_key ("{area_id}|{match_key}"), area_key ("{area_id}|{sub_kind}"), city_key."""
    building = map_unique(frame["building_name"], match_key, pl.Utf8)
    return (
        frame.with_columns(building.alias("_building"))
        .with_columns(
            pl.when(pl.col("_building").is_not_null())
            .then(pl.format("{}|{}", "area_id", "_building"))
            .alias("building_key"),
            pl.when(pl.col("sub_kind").is_not_null())
            .then(pl.format("{}|{}", "area_id", "sub_kind"))
            .alias("area_key"),
            pl.col("sub_kind").alias("city_key"),
        )
        .drop("_building")
    )


def rolling_stats(
    frame: pl.DataFrame,
    key: str,
    *,
    start_days: int,
    length_days: int,
    closed: str,
    prefix: str,
) -> pl.DataFrame:
    """Median ppsm and count of same-key sales in a day window placed relative to each row.

    The window starts at T + start_days and spans length_days (Polars offset/period).
    closed="left" gives [start, start + length); closed="both" includes both ends.
    """
    median, count = f"{prefix}_ppsm", f"{prefix}_n"
    keyed = frame.filter(pl.col(key).is_not_null()).sort(key, "instance_date", "row_id")
    stats = keyed.rolling(
        index_column="instance_date",
        period=f"{length_days}d",
        offset=f"{start_days}d",
        closed=closed,
        group_by=key,
    ).agg(
        pl.col("ppsm").median().alias(median),
        pl.col("ppsm").count().alias(count),
    )
    # Rolling keeps each group's own rows in date order, but with several distinct keys it can
    # emit whole (key, date) blocks in a different order than `keyed` (Polars partitions groups
    # for parallel execution), so positional alignment cannot be assumed. Rows sharing a key and
    # date share a window, so `stats` holds identical rows per (key, date); take one and join
    # back on (key, instance_date) rather than relying on row order.
    stats = stats.unique([key, "instance_date"], keep="first")
    aligned = (
        stats.select(key, "instance_date")
        .sort(key, "instance_date")
        .equals(keyed.select(key, "instance_date").unique().sort(key, "instance_date"))
    )
    if not aligned:
        raise RuntimeError(f"rolling output for {key} is not aligned with its input")
    keyed = keyed.join(stats, on=[key, "instance_date"], how="left")
    unkeyed = frame.filter(pl.col(key).is_null()).with_columns(
        pl.lit(None, dtype=pl.Float64).alias(median), pl.lit(0).alias(count)
    )
    return (
        pl.concat([keyed, unkeyed], how="vertical_relaxed")
        .with_columns(pl.col(count).fill_null(0).cast(pl.Int64))
        .sort("row_id")
    )


def add_base(frame: pl.DataFrame) -> pl.DataFrame:
    """Median ppsm over [T-92 d, T): the building's if it has 5+ sales, else the area's."""
    for level in LEVELS:
        frame = rolling_stats(
            frame,
            f"{level}_key",
            start_days=-BASE_DAYS,
            length_days=BASE_DAYS,
            closed="left",
            prefix=f"{level}_base",
        )
    building = pl.col("building_base_n") >= MIN_LEVEL_SALES
    area = pl.col("area_base_n") >= MIN_LEVEL_SALES

    def pick(part: str) -> pl.Expr:
        return (
            pl.when(building)
            .then(pl.col(f"building_base_{part}"))
            .when(area)
            .then(pl.col(f"area_base_{part}"))
        )

    return frame.with_columns(
        pl.when(building)
        .then(pl.lit("building"))
        .when(area)
        .then(pl.lit("area"))
        .alias("base_level"),
        pick("ppsm").alias("base_ppsm"),
        pick("n").alias("base_n"),
    )


def add_targets(frame: pl.DataFrame, data_end: date) -> pl.DataFrame:
    """growth_<h> = ln(median ppsm in the target window / base), at the base's level."""
    has_base = pl.col("base_level").is_not_null()
    at_building = pl.col("base_level") == "building"
    for name in HORIZONS:
        spec = HORIZON_SPECS[name]
        for level in LEVELS:
            frame = rolling_stats(
                frame,
                f"{level}_key",
                start_days=spec.start_days,
                length_days=spec.end_days - spec.start_days,
                closed="both",
                prefix=f"_{level}_{name}",
            )
        window_end = pl.col("instance_date").dt.offset_by(f"{spec.end_days}d")
        enough = pl.col(f"target_n_{name}") >= MIN_LEVEL_SALES
        status = (
            pl.when(~has_base)
            .then(pl.lit("no_base"))
            .when(window_end > pl.lit(data_end))
            .then(pl.lit("window_open"))
            .when(~enough)
            .then(pl.lit("no_target"))
            .otherwise(pl.lit("usable"))
        )
        frame = (
            frame.with_columns(
                pl.when(at_building)
                .then(pl.col(f"_building_{name}_ppsm"))
                .otherwise(pl.col(f"_area_{name}_ppsm"))
                .alias(f"target_ppsm_{name}"),
                pl.when(at_building)
                .then(pl.col(f"_building_{name}_n"))
                .otherwise(pl.col(f"_area_{name}_n"))
                .alias(f"target_n_{name}"),
            )
            .with_columns(status.alias(f"status_{name}"))
            .with_columns(
                pl.when(pl.col(f"status_{name}") == "usable")
                .then((pl.col(f"target_ppsm_{name}") / pl.col("base_ppsm")).log())
                .alias(f"growth_{name}")
            )
            .drop([f"_{level}_{name}_{part}" for level in LEVELS for part in ("ppsm", "n")])
        )
    return frame


@dataclass(frozen=True)
class TargetReport:
    rows: int
    horizons: dict[str, dict]

    def lines(self) -> list[str]:
        out = [f"Target rows: {self.rows:,}"]
        for name, counts in self.horizons.items():
            parts = ", ".join(f"{status} {counts[status]:,}" for status in STATUSES)
            out.append(f"  {name}: {parts}; last usable T {counts['last_t'] or 'none'}")
        return out

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "horizons": {
                name: {
                    **counts,
                    "last_t": counts["last_t"].isoformat() if counts["last_t"] else None,
                }
                for name, counts in self.horizons.items()
            },
        }


def target_report(frame: pl.DataFrame) -> TargetReport:
    real = frame.filter(pl.col("ppsm").is_not_null())
    horizons = {}
    for name in HORIZONS:
        counts: dict = dict.fromkeys(STATUSES, 0)
        counts.update(dict(real.group_by(f"status_{name}").len().iter_rows()))
        usable = real.filter(pl.col(f"status_{name}") == "usable")
        counts["last_t"] = usable["instance_date"].max() if usable.height else None
        horizons[name] = counts
    return TargetReport(real.height, horizons)


def build_targets(rows: pl.DataFrame, data_end: date) -> tuple[pl.DataFrame, TargetReport]:
    frame = add_targets(add_base(add_keys(rows)), data_end)
    return frame, target_report(frame)
