"""Relevance grades 0-3 for (query, listing) pairs, from the query's TRUE slots.

This module reads ground truth on purpose: grading is where true slots and fraud labels meet.
Rules: docs/superpowers/plans/2026-09-16-phase5-search-ranking.md, Task 3.
"""

import polars as pl

from ingestion.normalize import match_key

GRADE_SCHEMA = {
    "listing_id": pl.Int64,
    "area_id": pl.Int64,
    "kind": pl.Utf8,
    "building_key": pl.Utf8,
    "project_key": pl.Utf8,
    "bedrooms": pl.Int64,
    "size_sqm": pl.Float64,
    "asking_price_aed": pl.Float64,
    "description": pl.Utf8,
    "fraud_label": pl.Utf8,
}


def _all(expressions: list[pl.Expr]) -> pl.Expr:
    return pl.all_horizontal(expressions) if expressions else pl.lit(True)


def _soft_slots(true_slots: dict, margin: float) -> list[tuple[pl.Expr, pl.Expr]]:
    """(exact, near) per stated soft slot; near is always a superset of exact."""
    soft = []
    bedrooms = true_slots.get("bedrooms")
    if bedrooms is not None:
        difference = (pl.col("bedrooms") - bedrooms).abs()
        soft.append((difference == 0, difference <= 1))
    low, high = true_slots.get("budget_min"), true_slots.get("budget_max")
    if low is not None or high is not None:
        price = pl.col("asking_price_aed")
        exact, near = [], []
        if low is not None:
            exact.append(price >= low)
            near.append(price >= low * (1.0 - margin))
        if high is not None:
            exact.append(price <= high)
            near.append(price <= high * (1.0 + margin))
        soft.append((_all(exact), _all(near)))
    min_size = true_slots.get("min_size_sqm")
    if min_size is not None:
        size = pl.col("size_sqm")
        soft.append((size >= min_size, size >= min_size * (1.0 - margin)))
    amenities = [amenity.lower() for amenity in true_slots.get("amenities") or []]
    if amenities:
        text = pl.col("description").str.to_lowercase()
        hits = pl.sum_horizontal(
            [text.str.contains(amenity, literal=True).cast(pl.Int64) for amenity in amenities]
        )
        allowed_missing = 1 if len(amenities) == 2 else 0
        soft.append((hits == len(amenities), hits >= len(amenities) - allowed_missing))
    return [(exact.fill_null(False), near.fill_null(False)) for exact, near in soft]


def grade_frame(true_slots: dict, listings: pl.DataFrame, near_margin: float = 0.10) -> pl.Series:
    area_ok, type_ok, hard = pl.lit(True), pl.lit(True), []
    if true_slots.get("area_id") is not None:
        area_ok = pl.col("area_id") == true_slots["area_id"]
        hard.append(area_ok)
    if true_slots.get("property_type"):
        type_ok = (pl.col("kind") == true_slots["property_type"]).fill_null(False)
        hard.append(type_ok)
    if true_slots.get("building"):
        key = match_key(true_slots["building"])
        hard.append(
            ((pl.col("building_key") == key) | (pl.col("project_key") == key)).fill_null(False)
        )
    soft = _soft_slots(true_slots, near_margin)
    hard_ok = _all(hard)
    all_near = _all([near for _, near in soft])
    near_count = (
        pl.sum_horizontal([(~exact & near).cast(pl.Int64) for exact, near in soft])
        if soft
        else pl.lit(0)
    )
    graded = (
        pl.when(hard_ok & all_near & (near_count == 0))
        .then(3)
        .when(hard_ok & all_near & (near_count == 1))
        .then(2)
        .when(area_ok & type_ok)
        .then(1)
        .otherwise(0)
    )
    capped = pl.when(pl.col("fraud_label").is_not_null()).then(pl.min_horizontal(graded, 1))
    expression = capped.otherwise(graded).cast(pl.Int64).alias("grade")
    return listings.select(expression).to_series()


def grade(true_slots: dict, listing: dict, near_margin: float = 0.10) -> int:
    row = {name: listing.get(name) for name in GRADE_SCHEMA}
    frame = pl.DataFrame([row], schema=GRADE_SCHEMA)
    return int(grade_frame(true_slots, frame, near_margin)[0])
