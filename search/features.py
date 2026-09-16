"""The ranker's features for (query, candidate) rows.

Inputs are the PARSED query, the listing's own attributes, the predicted Phase 4 flags and the
Phase 3 price estimates. Nothing here reads a ground-truth label (see the leakage test).
"""

import logging
from datetime import date

import polars as pl

from ingestion.normalize import map_unique, match_key
from listings.fraud import (
    PriceInputError,
    load_fraud_attributes,
    load_price_predictor,
    price_request,
    resolve_price_model_version,
)
from search.config import FEATURES, KIND_SQL, SearchConfig
from search.lexicon import Lexicon
from search.parse import ParsedQuery, parse
from search.store import (
    ESTIMATE_SCHEMA,
    estimates_corpus_run,
    latest_corpus_run,
    read_estimates,
    read_judgments,
    read_queries,
    replace_estimates,
)

LOGGER = logging.getLogger(__name__)
NAN = float("nan")
ATTRIBUTE_SQL = f"""
SELECT l.listing_id, l.area_id, l.area_name, {KIND_SQL} AS kind, l.building_name,
       l.project_name, l.bedrooms, l.size_sqm, l.asking_price_aed, l.posted_at, l.title,
       l.description
FROM listings.listings AS l
ORDER BY l.listing_id
"""
ATTRIBUTE_SCHEMA = {
    "listing_id": pl.Int64,
    "area_id": pl.Int64,
    "area_name": pl.Utf8,
    "kind": pl.Utf8,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "bedrooms": pl.Int64,
    "size_sqm": pl.Float64,
    "asking_price_aed": pl.Float64,
    "posted_at": pl.Date,
    "title": pl.Utf8,
    "description": pl.Utf8,
}
FLAG_COLUMNS = {
    "bait_price": "flag_bait_price",
    "photo_reuse": "flag_photo_reuse",
    "inconsistent_relist": "flag_inconsistent_relist",
}
FLAGS_SQL = """
SELECT listing_id, flag FROM listings.fraud_flags
WHERE detect_run_id = (SELECT max(detect_run_id) FROM listings.detect_runs)
"""
QUERY_SCHEMA = {
    "query_id": pl.Int64,
    "q_area_ids": pl.List(pl.Int64),
    "q_building_key": pl.Utf8,
    "q_bedrooms": pl.Int64,
    "q_type": pl.Utf8,
    "q_budget_min": pl.Float64,
    "q_budget_max": pl.Float64,
    "q_min_size": pl.Float64,
    "q_amenities": pl.List(pl.Utf8),
}


def load_listing_attributes(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(ATTRIBUTE_SQL)
        frame = pl.DataFrame(cur.fetchall(), schema=ATTRIBUTE_SCHEMA, orient="row")
    return frame.with_columns(
        map_unique(frame["building_name"], match_key, pl.Utf8).alias("building_key"),
        map_unique(frame["project_name"], match_key, pl.Utf8).alias("project_key"),
    )


def load_predicted_flags(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(FLAGS_SQL)
        rows = cur.fetchall()
    frame = pl.DataFrame(rows, schema={"listing_id": pl.Int64, "flag": pl.Utf8}, orient="row")
    return (
        frame.filter(pl.col("flag").is_in(list(FLAG_COLUMNS)))
        .group_by("listing_id")
        .agg(
            (pl.col("flag") == flag).any().cast(pl.Float64).alias(column)
            for flag, column in FLAG_COLUMNS.items()
        )
        .sort("listing_id")
    )


def compute_estimates(rows: pl.DataFrame, predictor, log_every: int = 2_000) -> pl.DataFrame:
    """Phase 3 estimates for listing rows shaped like listings.fraud.load_fraud_attributes."""
    out = []
    for index, row in enumerate(rows.iter_rows(named=True), start=1):
        try:
            estimate = predictor.predict_one(price_request(row))
        except PriceInputError as exc:
            LOGGER.debug("listing %s cannot be priced: %s", row["listing_id"], exc)
            continue
        low, high = (float(value) for value in estimate.range_80)
        out.append((row["listing_id"], float(estimate.estimate_aed), low, high))
        if index % log_every == 0:
            LOGGER.info("priced %s of %s listings", index, rows.height)
    return pl.DataFrame(out, schema=ESTIMATE_SCHEMA, orient="row")


def refresh_estimates(conn, config: SearchConfig) -> dict[str, float]:
    """Recompute search.listing_estimates unless they already belong to the latest corpus."""
    corpus_run_id = latest_corpus_run(conn)
    stats = {
        "estimates_cached": 0.0,
        "value_features_skipped": 0.0,
        "estimates_written": 0.0,
        "estimates_unsupported": 0.0,
    }
    if estimates_corpus_run(conn) == corpus_run_id:
        stats["estimates_cached"] = 1.0
        return stats
    predictor = load_price_predictor(config.price_model_uri)
    if predictor is None:
        stats["value_features_skipped"] = 1.0
        return stats
    rows = load_fraud_attributes(conn)
    estimates = compute_estimates(rows, predictor)
    version = resolve_price_model_version(config.price_model_uri)
    written = replace_estimates(conn, estimates, corpus_run_id, version)
    stats["estimates_written"] = float(written)
    stats["estimates_unsupported"] = float(rows.height - estimates.height)
    return stats


def reference_date(attributes: pl.DataFrame) -> date:
    return attributes["posted_at"].max()


def query_frame(parsed: dict[int, ParsedQuery]) -> pl.DataFrame:
    rows = [
        {
            "query_id": query_id,
            "q_area_ids": list(query.area_ids),
            "q_building_key": match_key(query.building) if query.building else None,
            "q_bedrooms": query.bedrooms,
            "q_type": query.property_type,
            "q_budget_min": query.budget_min,
            "q_budget_max": query.budget_max,
            "q_min_size": query.min_size_sqm,
            "q_amenities": list(query.amenities),
        }
        for query_id, query in parsed.items()
    ]
    return pl.DataFrame(rows, schema=QUERY_SCHEMA)


def _flag(expression: pl.Expr) -> pl.Expr:
    return expression.fill_null(False).cast(pl.Float64)


def _when_stated(stated: pl.Expr, value: pl.Expr) -> pl.Expr:
    return pl.when(stated).then(value).otherwise(pl.lit(NAN))


def _amenity_hits(rows: pl.DataFrame) -> pl.DataFrame:
    exploded = (
        rows.select("_row", pl.col("description").str.to_lowercase(), "q_amenities")
        .explode("q_amenities", empty_as_null=True)
        .filter(pl.col("q_amenities").is_not_null())
    )
    hits = exploded.group_by("_row").agg(
        pl.col("description")
        .str.contains(pl.col("q_amenities").str.to_lowercase(), literal=True)
        .sum()
        .cast(pl.Float64)
        .alias("amenity_hits")
    )
    return rows.join(hits, on="_row", how="left", maintain_order="left")


def build_features(
    candidates: pl.DataFrame,
    queries: pl.DataFrame,
    attributes: pl.DataFrame,
    flags: pl.DataFrame,
    estimates: pl.DataFrame,
    reference: date,
) -> pl.DataFrame:
    attribute_columns = [
        "listing_id", "area_id", "kind", "building_key", "project_key", "bedrooms",
        "size_sqm", "asking_price_aed", "posted_at", "description",
    ]  # fmt: skip
    rows = (
        candidates.with_row_index("_row")
        .join(queries, on="query_id", how="left", maintain_order="left")
        .join(
            attributes.select(attribute_columns), on="listing_id", how="left", maintain_order="left"
        )
        .join(flags, on="listing_id", how="left", maintain_order="left")
        .join(estimates, on="listing_id", how="left", maintain_order="left")
    )
    rows = _amenity_hits(rows)
    price = pl.col("asking_price_aed")
    beds_stated = pl.col("q_bedrooms").is_not_null()
    area_stated = pl.col("q_area_ids").list.len().fill_null(0) > 0
    type_stated = pl.col("q_type").is_not_null()
    building_stated = pl.col("q_building_key").is_not_null()
    has_estimate = pl.col("estimate").is_not_null()
    same_building = (pl.col("building_key") == pl.col("q_building_key")) | (
        pl.col("project_key") == pl.col("q_building_key")
    )
    expressions = {
        "beds_diff": (pl.col("bedrooms") - pl.col("q_bedrooms")).abs().cast(pl.Float64),
        "beds_stated": beds_stated.cast(pl.Float64),
        "price_over_max": price / pl.col("q_budget_max") - 1.0,
        "price_under_min": 1.0 - price / pl.col("q_budget_min"),
        "budget_stated": _flag(
            pl.col("q_budget_min").is_not_null() | pl.col("q_budget_max").is_not_null()
        ),
        "size_ratio": pl.col("size_sqm") / pl.col("q_min_size"),
        "area_match": _when_stated(
            area_stated, _flag(pl.col("q_area_ids").list.contains(pl.col("area_id")))
        ),
        "area_stated": area_stated.cast(pl.Float64),
        "type_match": _when_stated(type_stated, _flag(pl.col("kind") == pl.col("q_type"))),
        "type_stated": type_stated.cast(pl.Float64),
        "building_match": _when_stated(building_stated, _flag(same_building)),
        "amenity_hits": pl.col("amenity_hits").fill_null(0.0),
        "amenity_asked": pl.col("q_amenities").list.len().fill_null(0).cast(pl.Float64),
        "semantic_cos": pl.col("semantic_cos"),
        "fulltext_rank": pl.col("fulltext_rank").fill_null(0.0),
        "semantic_pos": pl.col("semantic_pos").cast(pl.Float64),
        "fulltext_pos": pl.col("fulltext_pos").cast(pl.Float64),
        "rrf_score": pl.col("rrf_score"),
        "price_to_estimate": price / pl.col("estimate"),
        "within_interval": _when_stated(
            has_estimate, _flag((price >= pl.col("low")) & (price <= pl.col("high")))
        ),
        **{column: pl.col(column).fill_null(0.0) for column in FLAG_COLUMNS.values()},
        "cluster_size": pl.col("cluster_size").cast(pl.Float64),
        "days_since_posted": (pl.lit(reference) - pl.col("posted_at"))
        .dt.total_days()
        .cast(pl.Float64),
    }
    assert tuple(expressions) == FEATURES, "feature expressions out of order"
    return rows.sort("_row").select(
        "query_id",
        "listing_id",
        *(
            expression.cast(pl.Float64).fill_null(NAN).alias(name)
            for name, expression in expressions.items()
        ),
    )


def feature_table(conn, lexicon: Lexicon) -> pl.DataFrame:
    """Every stored judgment with its query's split and kind, its grade, and the features."""
    queries = read_queries(conn)
    judgments = read_judgments(conn)
    attributes = load_listing_attributes(conn)
    parsed = {
        query_id: parse(text, lexicon)
        for query_id, text in queries.select("query_id", "text").iter_rows()
    }
    table = build_features(
        judgments.drop("grade"),
        query_frame(parsed),
        attributes,
        load_predicted_flags(conn),
        read_estimates(conn),
        reference_date(attributes),
    )
    labels = judgments.select("query_id", "listing_id", "grade", "fused_pos").join(
        queries.select("query_id", "split", "kind"), on="query_id", how="left"
    )
    return (
        table.join(labels, on=["query_id", "listing_id"], how="left", maintain_order="left")
        .select("query_id", "split", "kind", "listing_id", "grade", "fused_pos", *FEATURES)
        .sort("query_id", "fused_pos")
    )
