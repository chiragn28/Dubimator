"""Load in-scope home sales from Postgres and apply the training scope."""

from dataclasses import dataclass
from datetime import date

import polars as pl

from ingestion.config import DbSettings
from models.price.config import (
    HOME_UNIT_SUB_TYPES,
    SIZE_BOUNDS,
    STAT_EXCLUSION_REASONS,
    TrainConfig,
)
from models.price.features import derive_segments

RAW_SCHEMA = {
    "transaction_id": pl.Utf8,
    "instance_date": pl.Date,
    "property_type": pl.Utf8,
    "property_sub_type": pl.Utf8,
    "reg_type": pl.Utf8,
    "area_id": pl.Int64,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "rooms": pl.Utf8,
    "has_parking": pl.Boolean,
    "area_sqm": pl.Float64,
    "price_aed": pl.Float64,
    "ingest_run_id": pl.Int64,
    "is_clean": pl.Boolean,
}
_SELECTED = ", ".join(name for name in RAW_SCHEMA if name != "is_clean")

# exclusion_reason is read only to split clean sales from the honest-holdout rows.
HOMES_SQL = f"""
SELECT {_SELECTED}, exclusion_reason IS NULL AS is_clean
FROM dld.transactions
WHERE (property_type = 'villa'
       OR (property_type = 'unit' AND property_sub_type IN %(unit_sub_types)s))
  AND instance_date >= %(index_start)s
  AND (exclusion_reason IS NULL
       OR (exclusion_reason IN %(stat_reasons)s AND instance_date >= %(test_start)s))
"""


@dataclass(frozen=True)
class HomesData:
    rows: pl.DataFrame
    data_end: date
    lineage: dict[str, object]
    drop_counts: dict[str, int]
    supported_segments: tuple[str, ...]
    areas: pl.DataFrame
    aliases: pl.DataFrame


def _in_bounds() -> pl.Expr:
    inside = pl.lit(False)
    for (property_type, basis), (low, high) in SIZE_BOUNDS.items():
        inside = inside | (
            (pl.col("property_type") == property_type)
            & (pl.col("size_basis") == basis)
            & pl.col("area_sqm").is_between(low, high)
        )
    return inside.fill_null(False)


def _count_by_segment(frame: pl.DataFrame, rule: str) -> dict[str, int]:
    counts = frame.group_by("segment").len().sort("segment")
    return {f"{rule}.{segment}": count for segment, count in counts.iter_rows()}


def prepare_homes(
    raw: pl.DataFrame, config: TrainConfig
) -> tuple[pl.DataFrame, dict[str, int], tuple[str, ...]]:
    """Derive segments, drop out-of-bounds sizes and unsupported segments, count drops."""
    frame = derive_segments(raw)
    drops = _count_by_segment(frame.filter(~_in_bounds()), "size_bounds")
    kept = frame.filter(_in_bounds())
    in_train = (
        pl.col("is_clean")
        & (pl.col("instance_date") >= config.train_start)
        & (pl.col("instance_date") < config.val_start)
    )
    counts = kept.filter(in_train).group_by("segment").len()
    supported = tuple(
        sorted(segment for segment, count in counts.iter_rows() if count >= config.min_segment_rows)
    )
    in_scope = pl.col("segment").is_in(list(supported))
    drops |= _count_by_segment(kept.filter(~in_scope), "unsupported_segment")
    return kept.filter(in_scope), drops, supported


def load_homes(settings: DbSettings, config: TrainConfig) -> HomesData:
    params = {
        "unit_sub_types": HOME_UNIT_SUB_TYPES,
        "index_start": config.index_start,
        "stat_reasons": STAT_EXCLUSION_REASONS,
        "test_start": config.test_start,
    }
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(HOMES_SQL, params)
            raw = pl.DataFrame(cur.fetchall(), schema=RAW_SCHEMA, orient="row")
            run_ids = sorted(raw["ingest_run_id"].unique().to_list())
            if len(run_ids) != 1:
                raise ValueError(f"expected rows from exactly one ingestion run, found {run_ids}")
            cur.execute(
                "SELECT source_sha256 FROM dld.ingestion_runs WHERE run_id = %s", (run_ids[0],)
            )
            (source_sha256,) = cur.fetchone()
            cur.execute("SELECT area_id, name_en FROM dld.areas ORDER BY area_id")
            areas = pl.DataFrame(
                cur.fetchall(), schema={"area_id": pl.Int64, "name_en": pl.Utf8}, orient="row"
            )
            cur.execute(
                "SELECT alias_key, area_id FROM dld.area_aliases ORDER BY alias_key, area_id"
            )
            aliases = pl.DataFrame(
                cur.fetchall(), schema={"alias_key": pl.Utf8, "area_id": pl.Int64}, orient="row"
            )
    finally:
        conn.close()
    rows, drops, supported = prepare_homes(raw, config)
    return HomesData(
        rows=rows,
        data_end=rows.filter(pl.col("is_clean"))["instance_date"].max(),
        lineage={"ingest_run_id": run_ids[0], "source_sha256": source_sha256},
        drop_counts=drops,
        supported_segments=supported,
        areas=areas,
        aliases=aliases,
    )
