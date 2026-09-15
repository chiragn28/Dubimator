from pathlib import Path

import polars as pl

from ingestion.normalize import map_unique, match_key

CURATED_ALIASES_PATH = Path(__file__).parent / "reference" / "curated_area_aliases.csv"
MASTER_PROJECT_MIN_ROWS = 50
MASTER_PROJECT_MIN_SHARE = 0.8
ALIAS_SCHEMA = {"alias_key": pl.Utf8, "alias": pl.Utf8, "area_id": pl.Int64, "source": pl.Utf8}


def _keys(series: pl.Series) -> pl.Series:
    return map_unique(series, match_key, pl.Utf8)


def build_areas(classified: pl.DataFrame) -> pl.DataFrame:
    rows = classified.filter(pl.col("area_id").is_not_null())
    names = (
        rows.group_by("area_id", "area_name", "area_name_ar")
        .len()
        .sort(["area_id", "len", "area_name"], descending=[False, True, False], nulls_last=True)
        .unique(subset="area_id", keep="first", maintain_order=True)
    )
    counts = rows.group_by("area_id").agg(
        pl.col("exclusion_reason").is_null().sum().cast(pl.Int64).alias("market_sales")
    )
    areas = (
        names.join(counts, on="area_id")
        .with_columns(
            pl.coalesce("area_name", pl.format("Area {}", pl.col("area_id"))).alias("name_en")
        )
        .select("area_id", "name_en", pl.col("area_name_ar").alias("name_ar"), "market_sales")
        .sort("area_id")
    )
    return areas.with_columns(_keys(areas["name_en"]).alias("match_key")).select(
        "area_id", "name_en", "name_ar", "match_key", "market_sales"
    )


def _master_project_aliases(classified: pl.DataFrame) -> pl.DataFrame:
    counts = (
        classified.filter(pl.col("master_project").is_not_null() & pl.col("area_id").is_not_null())
        .group_by("master_project", "area_id")
        .len()
    )
    top = (
        counts.with_columns(pl.col("len").sum().over("master_project").alias("total"))
        .sort(["master_project", "len", "area_id"], descending=[False, True, False])
        .unique(subset="master_project", keep="first", maintain_order=True)
        .filter(
            (pl.col("total") >= MASTER_PROJECT_MIN_ROWS)
            & (pl.col("len") / pl.col("total") >= MASTER_PROJECT_MIN_SHARE)
        )
    )
    return top.select(
        _keys(top["master_project"]).alias("alias_key"),
        pl.col("master_project").alias("alias"),
        "area_id",
        pl.lit("master_project").alias("source"),
    )


def _curated_aliases(areas: pl.DataFrame, curated_path: Path) -> tuple[pl.DataFrame, list[str]]:
    curated = pl.read_csv(curated_path, infer_schema=False)
    records, unresolved = [], []
    for alias, target in curated.select("alias", "area_name_en").iter_rows():
        ids = areas.filter(pl.col("match_key") == match_key(target))["area_id"].to_list()
        if not ids:
            unresolved.append(alias)
        records.extend(
            {"alias_key": match_key(alias), "alias": alias, "area_id": area_id, "source": "curated"}
            for area_id in ids
        )
    return pl.DataFrame(records, schema=ALIAS_SCHEMA), unresolved


def build_area_aliases(
    classified: pl.DataFrame, areas: pl.DataFrame, curated_path: Path = CURATED_ALIASES_PATH
) -> tuple[pl.DataFrame, list[str]]:
    official = areas.select(
        pl.col("match_key").alias("alias_key"),
        pl.col("name_en").alias("alias"),
        "area_id",
        pl.lit("official").alias("source"),
    )
    curated, unresolved = _curated_aliases(areas, curated_path)
    aliases = (
        pl.concat(
            [
                frame.cast(ALIAS_SCHEMA)
                for frame in (official, _master_project_aliases(classified), curated)
            ]
        )
        .filter(pl.col("alias_key").is_not_null())
        .unique(subset=["alias_key", "area_id"], keep="first", maintain_order=True)
    )
    return aliases, unresolved
