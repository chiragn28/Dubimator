import polars as pl
import pytest

from listings.generate import SUB_KINDS
from search.config import FEATURES, KIND_SQL, KINDS, TRUST_FEATURES, listing_kind
from search.store import (
    ESTIMATE_SCHEMA,
    JUDGMENT_SCHEMA,
    QUERY_SCHEMA,
    apply_schema,
    estimates_corpus_run,
    queries_corpus_run,
    read_estimates,
    read_judgments,
    read_queries,
    replace_estimates,
    replace_query_set,
)


def test_features_are_the_25_in_order_and_trust_is_a_subset():
    assert len(FEATURES) == 25 and len(set(FEATURES)) == 25
    assert FEATURES[0] == "beds_diff" and FEATURES[-1] == "days_since_posted"
    assert set(TRUST_FEATURES) <= set(FEATURES)


@pytest.mark.parametrize(
    ("property_type", "sub_type", "kind"),
    [
        ("villa", None, "villa"),
        ("unit", "Flat", "flat"),
        ("unit", "Hotel Apartment", "hotel_apartment"),
        ("unit", "Stacked Townhouses", "townhouse"),
        ("unit", "Office", None),
    ],
)
def test_listing_kind(property_type, sub_type, kind):
    assert listing_kind(property_type, sub_type) == kind


def test_kind_sql_names_every_sub_type_and_kind():
    for sub_type, kind in SUB_KINDS.items():
        assert f"'{sub_type}'" in KIND_SQL and f"'{kind}'" in KIND_SQL
    assert set(SUB_KINDS.values()) | {"villa"} == set(KINDS)


def _queries(corpus_run_id: int, ids=(1, 2)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": list(ids),
            "text": [f"2BR in Dubai Marina #{i}" for i in ids],
            "template_id": [0] * len(ids),
            "kind": ["specified"] * len(ids),
            "split": ["train"] * len(ids),
            "true_slots": ['{"bedrooms": 2}'] * len(ids),
            "seed_listing_id": [1] * len(ids),
            "n_grade3": [3] * len(ids),
            "corpus_run_id": [corpus_run_id] * len(ids),
        },
        schema=QUERY_SCHEMA,
    )


def _judgments(ids=(1, 2)) -> pl.DataFrame:
    rows = [
        (qid, listing, 3 - pos, 0.9, pos + 1, 0.1, None, 0.03, pos + 1, 1)
        for qid in ids
        for pos, listing in enumerate((5, 6, 7))
    ]
    return pl.DataFrame(rows, schema=JUDGMENT_SCHEMA, orient="row")


def test_schema_is_idempotent_and_fills_the_full_text_column(search_db):
    settings, listings, _ = search_db
    conn = settings.connect()
    try:
        apply_schema(conn)
        apply_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM listings.listings "
                "WHERE search_tsv @@ to_tsquery('english', 'marina')"
            )
            (hits,) = cur.fetchone()
            cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'listings'")
            indexes = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
    assert hits >= listings.filter(pl.col("area_id") == 1).height
    assert "listings_search_tsv_idx" in indexes


def test_query_set_round_trips_and_is_replaced_whole(search_db):
    settings, _, corpus_run_id = search_db
    conn = settings.connect()
    try:
        replace_query_set(conn, _queries(corpus_run_id), _judgments())
        conn.commit()
        replace_query_set(conn, _queries(corpus_run_id, ids=(7,)), _judgments(ids=(7,)))
        conn.commit()
        queries, judgments = read_queries(conn), read_judgments(conn)
        run = queries_corpus_run(conn)
    finally:
        conn.close()
    assert queries["query_id"].to_list() == [7]
    assert "true_slots" not in queries.columns
    assert judgments.height == 3 and judgments["grade"].to_list() == [3, 2, 1]
    assert judgments["fulltext_pos"].null_count() == 3
    assert run == corpus_run_id


def test_read_queries_filters_by_split(search_db):
    settings, _, corpus_run_id = search_db
    frame = _queries(corpus_run_id).with_columns(pl.Series("split", ["train", "report"]))
    conn = settings.connect()
    try:
        replace_query_set(conn, frame, _judgments())
        conn.commit()
        report = read_queries(conn, splits=("report",))
        some = read_judgments(conn, query_ids=[2])
    finally:
        conn.close()
    assert report["query_id"].to_list() == [2]
    assert set(some["query_id"].to_list()) == {2}


def test_estimates_round_trip_with_their_corpus_run(search_db):
    settings, _, corpus_run_id = search_db
    estimates = pl.DataFrame(
        {"listing_id": [1, 2], "estimate": [1e6, 2e6], "low": [9e5, 1.8e6], "high": [1.1e6, 2.2e6]},
        schema=ESTIMATE_SCHEMA,
    )
    conn = settings.connect()
    try:
        assert estimates_corpus_run(conn) is None
        written = replace_estimates(conn, estimates, corpus_run_id, "2")
        conn.commit()
        back, run = read_estimates(conn), estimates_corpus_run(conn)
    finally:
        conn.close()
    assert written == 2 and run == corpus_run_id
    assert back.sort("listing_id")["estimate"].to_list() == [1e6, 2e6]
