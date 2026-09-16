"""Load the synthetic corpus and its vectors into Postgres schema `listings`."""

import json
from pathlib import Path

import polars as pl

from ingestion.config import DbSettings
from ingestion.load import LoadInvariantError, copy_frame
from listings.generate import Corpus

SCHEMA_SQL = Path(__file__).parent / "sql" / "schema.sql"
TRUNCATE_LOCK_TIMEOUT = "60s"
# Truncated together: a new corpus invalidates every detection result that referenced it.
CORPUS_TABLES = (
    "duplicate_pairs",
    "fraud_flags",
    "detect_runs",
    "listing_photos",
    "listing_embeddings",
    "photos",
    "listings",
)


def vector_literal(values) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def apply_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))


def start_corpus_run(conn, seed: int, photo_dataset_sha: str, counts: dict) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO listings.corpus_runs (seed, photo_dataset_sha, counts) "
            "VALUES (%s, %s, %s::jsonb) RETURNING corpus_run_id",
            (seed, photo_dataset_sha, json.dumps(counts)),
        )
        return cur.fetchone()[0]


def latest_corpus_run(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT max(corpus_run_id) FROM listings.corpus_runs")
        (run_id,) = cur.fetchone()
    if run_id is None:
        raise RuntimeError("no corpus has been loaded yet — run `python -m listings build` first")
    return run_id


def _verify_loaded(cur, table: str, expected: int) -> None:
    cur.execute(f"SELECT count(*) FROM {table}")
    (loaded,) = cur.fetchone()
    if loaded != expected:
        raise LoadInvariantError(f"{table}: loaded {loaded} rows but expected {expected}")


def replace_corpus(conn, corpus: Corpus, corpus_run_id: int) -> None:
    """Truncate the corpus tables and COPY this corpus in. Caller owns the transaction."""
    stamped = pl.lit(corpus_run_id, dtype=pl.Int64)
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL lock_timeout = '{TRUNCATE_LOCK_TIMEOUT}'")
        cur.execute(f"TRUNCATE {', '.join(f'listings.{t}' for t in CORPUS_TABLES)}")
        copy_frame(
            cur, "listings.listings", corpus.listings.with_columns(stamped.alias("corpus_run_id"))
        )
        _verify_loaded(cur, "listings.listings", corpus.listings.height)
        # embedding is left out of the COPY: it is filled in later by write_photo_embeddings
        copy_frame(
            cur, "listings.photos", corpus.photos.with_columns(stamped.alias("corpus_run_id"))
        )
        _verify_loaded(cur, "listings.photos", corpus.photos.height)
        copy_frame(cur, "listings.listing_photos", corpus.listing_photos)
        _verify_loaded(cur, "listings.listing_photos", corpus.listing_photos.height)


def load_corpus(settings: DbSettings, corpus: Corpus, seed: int, photo_dataset_sha: str) -> int:
    conn = settings.connect()
    try:
        apply_schema(conn)
        corpus_run_id = start_corpus_run(conn, seed, photo_dataset_sha, corpus.counts)
        replace_corpus(conn, corpus, corpus_run_id)
        conn.commit()
        return corpus_run_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def write_photo_embeddings(conn, frame: pl.DataFrame) -> int:
    """frame: photo_id, embedding (pgvector text form)."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE photo_vectors (photo_id bigint, embedding text) ON COMMIT DROP"
        )
        copy_frame(cur, "photo_vectors", frame.select("photo_id", "embedding"))
        cur.execute(
            "UPDATE listings.photos AS p SET embedding = v.embedding::vector "
            "FROM photo_vectors AS v WHERE p.photo_id = v.photo_id"
        )
        return cur.rowcount


def write_listing_embeddings(conn, frame: pl.DataFrame) -> int:
    """frame: listing_id, text_embedding, image_embedding (pgvector text form)."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE listing_vectors "
            "(listing_id bigint, text_embedding text, image_embedding text) ON COMMIT DROP"
        )
        copy_frame(
            cur,
            "listing_vectors",
            frame.select("listing_id", "text_embedding", "image_embedding"),
        )
        cur.execute(
            "INSERT INTO listings.listing_embeddings (listing_id, text_embedding, image_embedding) "
            "SELECT listing_id, text_embedding::vector, image_embedding::vector FROM listing_vectors "
            "ON CONFLICT (listing_id) DO UPDATE SET "
            "text_embedding = EXCLUDED.text_embedding, image_embedding = EXCLUDED.image_embedding"
        )
        return cur.rowcount


def create_vector_indexes(conn) -> None:
    """HNSW, cosine. Built after the vectors are in: much faster than incremental inserts."""
    statements = (
        (
            "CREATE INDEX IF NOT EXISTS photos_embedding_hnsw ON listings.photos "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS listing_text_embedding_hnsw ON listings.listing_embeddings "
            "USING hnsw (text_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS listing_image_embedding_hnsw ON listings.listing_embeddings "
            "USING hnsw (image_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        ),
    )
    with conn.cursor() as cur:
        for statement in statements:
            cur.execute(statement)


EMBEDDING_INPUT_SQL = {
    "photos": "SELECT photo_id, path FROM listings.photos ORDER BY photo_id",
    "listings": "SELECT listing_id, title, description FROM listings.listings ORDER BY listing_id",
    "listing_photos": (
        "SELECT listing_id, photo_id, position FROM listings.listing_photos "
        "ORDER BY listing_id, position"
    ),
}
EMBEDDING_INPUT_SCHEMA = {
    "photos": {"photo_id": pl.Int64, "path": pl.Utf8},
    "listings": {"listing_id": pl.Int64, "title": pl.Utf8, "description": pl.Utf8},
    "listing_photos": {"listing_id": pl.Int64, "photo_id": pl.Int64, "position": pl.Int64},
}


def read_corpus_for_embedding(conn) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    frames = {}
    with conn.cursor() as cur:
        for name, sql in EMBEDDING_INPUT_SQL.items():
            cur.execute(sql)
            frames[name] = pl.DataFrame(
                cur.fetchall(), schema=EMBEDDING_INPUT_SCHEMA[name], orient="row"
            )
    # An explicit 3-tuple, not tuple(frames.values()): the arity is then checkable by callers
    # and by static analysis, and it doesn't silently depend on EMBEDDING_INPUT_SQL's key order.
    return frames["photos"], frames["listings"], frames["listing_photos"]
