import polars as pl
import pytest

from listings.generate import generate_corpus
from listings.load import (
    apply_schema,
    create_vector_indexes,
    latest_corpus_run,
    load_corpus,
    vector_literal,
    write_listing_embeddings,
    write_photo_embeddings,
)

POOL = tuple(range(1, 21))


def corpus_for(sales_frame, areas_frame, config):
    return generate_corpus(sales_frame(), areas_frame, POOL, config)


def query(settings, sql, params=None):
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def test_vector_literal_formats_for_pgvector():
    assert vector_literal([0.5, -0.25, 0.0]) == "[0.5,-0.25,0.0]"


def test_schema_is_idempotent(pg_test_db):
    conn = pg_test_db.connect()
    try:
        apply_schema(conn)
        apply_schema(conn)
        conn.commit()
    finally:
        conn.close()
    tables = {
        row[0]
        for row in query(pg_test_db, "SELECT tablename FROM pg_tables WHERE schemaname='listings'")
    }
    assert {
        "listings",
        "photos",
        "listing_photos",
        "listing_embeddings",
        "duplicate_pairs",
        "fraud_flags",
        "corpus_runs",
        "detect_runs",
    } <= tables


def test_load_corpus_writes_every_table_and_links_the_run(
    pg_test_db, sales_frame, areas_frame, small_corpus_config
):
    corpus = corpus_for(sales_frame, areas_frame, small_corpus_config)
    run_id = load_corpus(pg_test_db, corpus, seed=small_corpus_config.seed, photo_dataset_sha="abc")

    assert (
        query(pg_test_db, "SELECT count(*) FROM listings.listings")[0][0] == corpus.listings.height
    )
    assert query(pg_test_db, "SELECT count(*) FROM listings.photos")[0][0] == corpus.photos.height
    assert (
        query(pg_test_db, "SELECT count(*) FROM listings.listing_photos")[0][0]
        == corpus.listing_photos.height
    )
    assert query(pg_test_db, "SELECT count(DISTINCT corpus_run_id) FROM listings.listings") == [
        (1,)
    ]
    row = query(
        pg_test_db,
        "SELECT seed, photo_dataset_sha, counts->>'listings' FROM listings.corpus_runs WHERE corpus_run_id=%s",
        (run_id,),
    )[0]
    assert row[0] == small_corpus_config.seed
    assert row[1] == "abc"
    assert int(row[2]) == corpus.listings.height
    assert latest_corpus_run_of(pg_test_db) == run_id


def latest_corpus_run_of(settings):
    conn = settings.connect()
    try:
        return latest_corpus_run(conn)
    finally:
        conn.close()


def test_second_load_replaces_the_previous_corpus(
    pg_test_db, sales_frame, areas_frame, small_corpus_config
):
    corpus = corpus_for(sales_frame, areas_frame, small_corpus_config)
    first = load_corpus(pg_test_db, corpus, seed=1, photo_dataset_sha="a")
    second = load_corpus(pg_test_db, corpus, seed=2, photo_dataset_sha="b")
    assert second > first
    assert (
        query(pg_test_db, "SELECT count(*) FROM listings.listings")[0][0] == corpus.listings.height
    )
    assert query(pg_test_db, "SELECT DISTINCT corpus_run_id FROM listings.listings") == [(second,)]


def test_embeddings_round_trip_and_knn_finds_the_nearest(
    pg_test_db, sales_frame, areas_frame, small_corpus_config
):
    corpus = corpus_for(sales_frame, areas_frame, small_corpus_config)
    load_corpus(pg_test_db, corpus, seed=1, photo_dataset_sha="a")

    photo_ids = corpus.photos["photo_id"].to_list()[:6]
    photo_vectors = pl.DataFrame(
        {
            "photo_id": photo_ids,
            "embedding": [
                vector_literal([1.0 if i == j else 0.0 for j in range(512)])
                for i in range(len(photo_ids))
            ],
        }
    )
    listing_ids = corpus.listings["listing_id"].to_list()[:4]
    listing_vectors = pl.DataFrame(
        {
            "listing_id": listing_ids,
            "text_embedding": [
                vector_literal([1.0 if i == j else 0.0 for j in range(384)])
                for i in range(len(listing_ids))
            ],
            "image_embedding": [
                vector_literal([1.0 if i == j else 0.0 for j in range(512)])
                for i in range(len(listing_ids))
            ],
        }
    )
    conn = pg_test_db.connect()
    try:
        assert write_photo_embeddings(conn, photo_vectors) == len(photo_ids)
        assert write_listing_embeddings(conn, listing_vectors) == len(listing_ids)
        create_vector_indexes(conn)
        conn.commit()
    finally:
        conn.close()

    probe = vector_literal([1.0 if j == 0 else 0.0 for j in range(512)])
    nearest = query(
        pg_test_db,
        "SELECT photo_id FROM listings.photos WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> %s::vector LIMIT 1",
        (probe,),
    )
    assert nearest[0][0] == photo_ids[0]
    indexes = {
        row[0]
        for row in query(pg_test_db, "SELECT indexname FROM pg_indexes WHERE schemaname='listings'")
    }
    assert {
        "photos_embedding_hnsw",
        "listing_text_embedding_hnsw",
        "listing_image_embedding_hnsw",
    } <= indexes


def test_duplicate_pairs_must_be_canonical(
    pg_test_db, sales_frame, areas_frame, small_corpus_config
):
    corpus = corpus_for(sales_frame, areas_frame, small_corpus_config)
    run_id = load_corpus(pg_test_db, corpus, seed=1, photo_dataset_sha="a")
    a, b = corpus.listings["listing_id"].to_list()[:2]
    conn = pg_test_db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO listings.detect_runs (corpus_run_id, threshold) VALUES (%s, 0.5) "
                "RETURNING detect_run_id",
                (run_id,),
            )
            detect_run_id = cur.fetchone()[0]
            with pytest.raises(Exception, match="duplicate_pairs_canonical"):
                cur.execute(
                    "INSERT INTO listings.duplicate_pairs "
                    "(detect_run_id, listing_a, listing_b, score, decision, signals) "
                    "VALUES (%s, %s, %s, 0.9, true, '{}'::jsonb)",
                    (detect_run_id, max(a, b), min(a, b)),
                )
    finally:
        conn.rollback()
        conn.close()


def test_the_listings_table_says_it_is_synthetic(pg_test_db):
    conn = pg_test_db.connect()
    try:
        apply_schema(conn)
        conn.commit()
    finally:
        conn.close()
    comment = query(pg_test_db, "SELECT obj_description('listings.listings'::regclass)")[0][0]
    assert "ynthetic" in comment
