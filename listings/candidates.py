"""Candidate pairs from pgvector.

Comparing all 20,000 listings pairwise is 200 million comparisons. Three indexed
channels bring that down to a few tens of candidates per listing.
"""

import time
from dataclasses import dataclass

import polars as pl

from listings.config import DetectConfig

PAIR_SCHEMA = {"listing_a": pl.Int64, "listing_b": pl.Int64}

TEXT_SQL = """
SELECT least(p.listing_id, n.listing_id) AS listing_a,
       greatest(p.listing_id, n.listing_id) AS listing_b
FROM listings.listing_embeddings p
CROSS JOIN LATERAL (
    SELECT e.listing_id
    FROM listings.listing_embeddings e
    WHERE e.listing_id <> p.listing_id AND e.text_embedding IS NOT NULL
    ORDER BY e.text_embedding <=> p.text_embedding
    LIMIT %(k)s
) n
WHERE p.text_embedding IS NOT NULL
"""

IMAGE_SQL = """
SELECT least(p.listing_id, n.listing_id) AS listing_a,
       greatest(p.listing_id, n.listing_id) AS listing_b
FROM listings.listing_embeddings p
CROSS JOIN LATERAL (
    SELECT e.listing_id
    FROM listings.listing_embeddings e
    WHERE e.listing_id <> p.listing_id AND e.image_embedding IS NOT NULL
    ORDER BY e.image_embedding <=> p.image_embedding
    LIMIT %(k)s
) n
WHERE p.image_embedding IS NOT NULL
"""

# Photos used by more than max_photo_fanout listings are agency stock: pairing every
# user with every other would be quadratic and would flag legitimate reuse anyway.
SHARED_PHOTO_SQL = """
WITH usage AS (
    SELECT photo_id, count(*) AS listings
    FROM listings.listing_photos
    GROUP BY photo_id
)
SELECT a.listing_id AS listing_a, b.listing_id AS listing_b
FROM listings.listing_photos a
JOIN usage ON usage.photo_id = a.photo_id AND usage.listings <= %(fanout)s
JOIN listings.listing_photos b ON b.photo_id = a.photo_id AND b.listing_id > a.listing_id
"""


@dataclass(frozen=True)
class CandidateStats:
    text_pairs: int
    image_pairs: int
    shared_photo_pairs: int
    total: int
    seconds: float
    text_seconds: float = 0.0
    image_seconds: float = 0.0
    shared_photo_seconds: float = 0.0


def _query(conn, sql: str, params: dict) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pl.DataFrame(cur.fetchall(), schema=PAIR_SCHEMA, orient="row").unique()


def fetch_candidates(conn, config: DetectConfig) -> tuple[pl.DataFrame, CandidateStats]:
    started = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("SET LOCAL hnsw.ef_search = %s", (config.ef_search,))
    queries = {
        "text": (TEXT_SQL, {"k": config.text_top_k}),
        "image": (IMAGE_SQL, {"k": config.photo_top_k}),
        "shared_photo": (SHARED_PHOTO_SQL, {"fanout": config.max_photo_fanout}),
    }
    channels, seconds = {}, {}
    for name, (sql, params) in queries.items():
        channel_started = time.perf_counter()
        channels[name] = _query(conn, sql, params)
        seconds[name] = time.perf_counter() - channel_started
    labelled = [
        frame.with_columns(pl.lit(name).alias("source"))
        for name, frame in channels.items()
        if frame.height
    ]
    if not labelled:
        empty = pl.DataFrame(
            {"listing_a": [], "listing_b": [], "sources": []},
            schema={**PAIR_SCHEMA, "sources": pl.Utf8},
        )
        return empty, CandidateStats(
            0,
            0,
            0,
            0,
            time.perf_counter() - started,
            seconds["text"],
            seconds["image"],
            seconds["shared_photo"],
        )

    pairs = (
        pl.concat(labelled)
        .group_by(["listing_a", "listing_b"])
        .agg(pl.col("source").unique().sort().str.join(",").alias("sources"))
        .sort(["listing_a", "listing_b"])
    )
    return pairs, CandidateStats(
        text_pairs=channels["text"].height,
        image_pairs=channels["image"].height,
        shared_photo_pairs=channels["shared_photo"].height,
        total=pairs.height,
        seconds=time.perf_counter() - started,
        text_seconds=seconds["text"],
        image_seconds=seconds["image"],
        shared_photo_seconds=seconds["shared_photo"],
    )


def brute_force_pairs(conn, metric: str, top_k: int) -> pl.DataFrame:
    """Exact nearest neighbours by sequential scan — the yardstick for index recall and timing."""
    sql = {"text": TEXT_SQL, "image": IMAGE_SQL}[metric]
    with conn.cursor() as cur:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        cur.execute(sql, {"k": top_k})
        rows = cur.fetchall()
        cur.execute("SET LOCAL enable_indexscan = on")
        cur.execute("SET LOCAL enable_bitmapscan = on")
    return pl.DataFrame(rows, schema=PAIR_SCHEMA, orient="row").unique()


def _pair_set(frame: pl.DataFrame) -> set[tuple[int, int]]:
    return set(zip(frame["listing_a"].to_list(), frame["listing_b"].to_list()))


def benchmark_text_retrieval(
    conn, config: DetectConfig, candidates: pl.DataFrame | None = None
) -> dict[str, float]:
    """Index vs. exact, like for like: the HNSW text lookup against the same text query run as
    an exact, index-disabled sequential scan (same k, whole corpus). Both are timed here.

    `text_index_recall` is the share of the exact text-neighbour pairs the indexed text
    channel returned. `candidates_exact_text_recall` is the share of them present anywhere in
    `candidates` (all three channels) — the looser figure, logged under its own name.
    """
    with conn.cursor() as cur:
        cur.execute("SET LOCAL hnsw.ef_search = %s", (config.ef_search,))
    started = time.perf_counter()
    indexed = _query(conn, TEXT_SQL, {"k": config.text_top_k})
    index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    exact = brute_force_pairs(conn, "text", config.text_top_k)
    exact_seconds = time.perf_counter() - started

    exact_pairs = _pair_set(exact)

    def share(found: set[tuple[int, int]]) -> float:
        return len(exact_pairs & found) / len(exact_pairs) if exact_pairs else float("nan")

    metrics = {
        "retrieval.bench.text_index_seconds": index_seconds,
        "retrieval.bench.text_exact_seconds": exact_seconds,
        "retrieval.bench.text_index_pairs": float(indexed.height),
        "retrieval.bench.text_exact_pairs": float(exact.height),
        "retrieval.bench.text_index_recall": share(_pair_set(indexed)),
    }
    if candidates is not None:
        metrics["retrieval.bench.candidates_exact_text_recall"] = share(_pair_set(candidates))
    return metrics
