import dataclasses
import math
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from search_fixtures import PREDICTED_BAIT_IDS

import search.features as features_module
from models.price.predictor import PriceInputError
from search.config import FEATURES, LABEL_READERS, SEARCH_FORBIDDEN, SearchConfig
from search.features import (
    build_features,
    compute_estimates,
    feature_table,
    load_listing_attributes,
    load_predicted_flags,
    query_frame,
    reference_date,
    refresh_estimates,
)
from search.lexicon import load_lexicon
from search.parse import ParsedQuery
from search.queries import build_query_set
from search.retrieve import CANDIDATE_SCHEMA
from search.store import ESTIMATE_SCHEMA, read_estimates, replace_query_set

NAN = float("nan")


def _attributes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "listing_id": [1, 2],
            "area_id": [1, 2],
            "kind": ["flat", "villa"],
            "building_key": ["marina gate", None],
            "project_key": ["project 1", "project 2"],
            "bedrooms": [2, None],
            "size_sqm": [100.0, 300.0],
            "asking_price_aed": [1_000_000.0, 3_000_000.0],
            "posted_at": [date(2023, 1, 1), date(2023, 1, 11)],
            "description": ["Features include balcony, sea view.", "Features include shared pool."],
        }
    )


def _candidates() -> pl.DataFrame:
    rows = [
        (1, 1, 0.9, 1, 0.2, 1, 0.03, 1, 2),
        (1, 2, None, None, 0.1, 2, 0.02, 2, 1),
        (2, 2, 0.5, 1, 0.0, None, 0.01, 1, 1),
    ]
    schema = {"query_id": pl.Int64, **CANDIDATE_SCHEMA}
    return pl.DataFrame(rows, schema=schema, orient="row")


PARSED = {
    1: ParsedQuery(
        area_ids=(1,),
        building="Marina Gate",
        bedrooms=3,
        property_type="flat",
        budget_max=900_000.0,
        min_size_sqm=80.0,
        amenities=("balcony", "shared pool"),
    ),
    2: ParsedQuery(free_text="quiet"),
}
FLAGS = pl.DataFrame(
    {
        "listing_id": [2],
        "flag_bait_price": [1.0],
        "flag_photo_reuse": [0.0],
        "flag_inconsistent_relist": [0.0],
    }
)
ESTIMATES = pl.DataFrame(
    {"listing_id": [1], "estimate": [1_250_000.0], "low": [1_100_000.0], "high": [1_400_000.0]},
    schema=ESTIMATE_SCHEMA,
)


def _same(actual, expected):
    if isinstance(expected, float) and math.isnan(expected):
        return math.isnan(actual)
    return actual == pytest.approx(expected)


@pytest.fixture
def table():
    return build_features(
        _candidates(), query_frame(PARSED), _attributes(), FLAGS, ESTIMATES, date(2023, 1, 21)
    )


def test_columns_order_and_types(table):
    assert table.columns == ["query_id", "listing_id", *FEATURES]
    assert all(table.schema[name] == pl.Float64 for name in FEATURES)
    assert table.select("query_id", "listing_id").rows() == [(1, 1), (1, 2), (2, 2)]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            0,
            {
                "beds_diff": 1.0, "beds_stated": 1.0, "price_over_max": 1e6 / 9e5 - 1,
                "price_under_min": NAN, "budget_stated": 1.0, "size_ratio": 1.25,
                "area_match": 1.0, "area_stated": 1.0, "type_match": 1.0, "type_stated": 1.0,
                "building_match": 1.0, "amenity_hits": 1.0, "amenity_asked": 2.0,
                "semantic_cos": 0.9, "fulltext_rank": 0.2, "semantic_pos": 1.0,
                "fulltext_pos": 1.0, "rrf_score": 0.03, "price_to_estimate": 0.8,
                "within_interval": 0.0, "flag_bait_price": 0.0, "flag_photo_reuse": 0.0,
                "flag_inconsistent_relist": 0.0, "cluster_size": 2.0, "days_since_posted": 20.0,
            },
        ),
        (
            1,
            {
                "beds_diff": NAN, "area_match": 0.0, "type_match": 0.0, "building_match": 0.0,
                "amenity_hits": 1.0, "semantic_cos": NAN, "semantic_pos": NAN,
                "fulltext_pos": 2.0, "price_to_estimate": NAN, "within_interval": NAN,
                "flag_bait_price": 1.0, "cluster_size": 1.0, "days_since_posted": 10.0,
                "size_ratio": 3.75,
            },
        ),
        (
            2,
            {
                "beds_diff": NAN, "beds_stated": 0.0, "price_over_max": NAN,
                "budget_stated": 0.0, "size_ratio": NAN, "area_match": NAN, "area_stated": 0.0,
                "type_match": NAN, "type_stated": 0.0, "building_match": NAN,
                "amenity_hits": 0.0, "amenity_asked": 0.0, "fulltext_rank": 0.0,
                "fulltext_pos": NAN,
            },
        ),
    ],
)  # fmt: skip
def test_feature_values(table, row, expected):
    values = table.row(row, named=True)
    for name, value in expected.items():
        assert _same(values[name], value), name


def test_within_interval_is_one_inside_the_range():
    estimates = ESTIMATES.with_columns(pl.lit(900_000.0).alias("low"))
    out = build_features(
        _candidates(), query_frame(PARSED), _attributes(), FLAGS, estimates, date(2023, 1, 21)
    )
    assert out["within_interval"][0] == 1.0


def test_value_features_are_nan_without_estimates():
    empty = pl.DataFrame(schema=ESTIMATE_SCHEMA)
    out = build_features(
        _candidates(), query_frame(PARSED), _attributes(), FLAGS, empty, date(2023, 1, 21)
    )
    assert out["price_to_estimate"].is_nan().all() and out["within_interval"].is_nan().all()


def test_empty_candidates_give_an_empty_table():
    empty = pl.DataFrame(schema={"query_id": pl.Int64, **CANDIDATE_SCHEMA})
    out = build_features(
        empty, query_frame(PARSED), _attributes(), FLAGS, ESTIMATES, date(2023, 1, 21)
    )
    assert out.height == 0 and out.columns == ["query_id", "listing_id", *FEATURES]


def test_attributes_flags_and_reference_date_load_from_postgres(search_db):
    settings, listings, _ = search_db
    conn = settings.connect()
    try:
        attributes = load_listing_attributes(conn)
        flags = load_predicted_flags(conn)
    finally:
        conn.close()
    assert attributes.height == listings.height
    assert not set(attributes.columns) & set(SEARCH_FORBIDDEN)
    assert {"building_key", "project_key", "kind", "description"} <= set(attributes.columns)
    assert sorted(flags["listing_id"]) == list(PREDICTED_BAIT_IDS)
    assert (flags["flag_bait_price"] == 1.0).all() and (flags["flag_photo_reuse"] == 0.0).all()
    assert reference_date(attributes) == listings["posted_at"].max()


class StubPredictor:
    def predict_one(self, request):
        if request["area_id"] == 6:
            raise PriceInputError("area_id", "unsupported in this stub")
        price = request["size_sqm"] * 10_000.0
        return SimpleNamespace(estimate_aed=price, range_80=(price * 0.9, price * 1.1))


def test_compute_estimates_skips_unsupported_rows(search_db):
    from listings.fraud import load_fraud_attributes

    settings, listings, _ = search_db
    conn = settings.connect()
    try:
        rows = load_fraud_attributes(conn)
    finally:
        conn.close()
    estimates = compute_estimates(rows, StubPredictor(), log_every=50)
    unsupported = listings.filter(pl.col("area_id") == 6).height
    assert estimates.height == listings.height - unsupported
    first = estimates.filter(pl.col("listing_id") == 1).row(0, named=True)
    size = listings.filter(pl.col("listing_id") == 1)["size_sqm"][0]
    assert first["estimate"] == pytest.approx(size * 10_000)
    assert first["low"] < first["estimate"] < first["high"]


def test_refresh_estimates_writes_once_per_corpus(search_db, monkeypatch):
    settings, listings, _ = search_db
    calls = []
    monkeypatch.setattr(
        features_module, "load_price_predictor", lambda uri: calls.append(uri) or StubPredictor()
    )
    monkeypatch.setattr(features_module, "resolve_price_model_version", lambda uri: "9")
    conn = settings.connect()
    try:
        first = refresh_estimates(conn, SearchConfig())
        conn.commit()
        second = refresh_estimates(conn, SearchConfig())
        stored = read_estimates(conn)
    finally:
        conn.close()
    assert first["estimates_cached"] == 0.0 and first["estimates_written"] == stored.height > 0
    assert first["estimates_unsupported"] == listings.filter(pl.col("area_id") == 6).height
    assert second["estimates_cached"] == 1.0 and len(calls) == 1


def test_refresh_estimates_without_a_price_model_skips_and_writes_nothing(search_db, monkeypatch):
    settings, _, _ = search_db
    monkeypatch.setattr(features_module, "load_price_predictor", lambda uri: None)
    conn = settings.connect()
    try:
        stats = refresh_estimates(conn, SearchConfig())
        stored = read_estimates(conn)
    finally:
        conn.close()
    assert stats["value_features_skipped"] == 1.0 and stored.height == 0


def test_feature_table_covers_every_judgment(search_db, fake_embedder):
    settings, _, _ = search_db
    config = dataclasses.replace(SearchConfig(), n_queries=40, ef_search=100)
    conn = settings.connect()
    try:
        lexicon = load_lexicon(conn)
        queries, judgments = build_query_set(conn, fake_embedder, lexicon, config)
        replace_query_set(conn, queries, judgments)
        conn.commit()
        table = feature_table(conn, lexicon)
    finally:
        conn.close()
    assert table.height == judgments.height
    assert table.columns[:6] == ["query_id", "split", "kind", "listing_id", "grade", "fused_pos"]
    assert table.columns[6:] == list(FEATURES)
    assert table["grade"].null_count() == 0 and table["split"].null_count() == 0
    assert table.select("query_id", "fused_pos").is_duplicated().sum() == 0
    assert table["price_to_estimate"].is_nan().all()  # no estimates were computed here


def test_no_search_module_outside_the_label_readers_names_a_label():
    """Labels, true slots and generator flags must never reach features, retrieval or the engine."""
    package = Path(features_module.__file__).parent
    exempt = {*LABEL_READERS, "config.py"}  # config.py only declares the forbidden list
    pattern = re.compile(r"\b(" + "|".join(SEARCH_FORBIDDEN) + r")\b")
    offenders = {
        path.name: sorted(set(pattern.findall(path.read_text(encoding="utf-8"))))
        for path in package.rglob("*.py")
        if path.name not in exempt and pattern.search(path.read_text(encoding="utf-8"))
    }
    assert offenders == {}
