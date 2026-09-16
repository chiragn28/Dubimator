"""The twelve pair features. This module never reads a label column.

Photo-level similarity is the expensive part: each pair compares its photos with
the other listing's (4 x 4 dot products). Vectors are stacked per listing once,
so the per-pair work is a single small matrix multiply.
"""

import numpy as np
import polars as pl

from listings.config import PAIR_FEATURES

LISTING_SQL = """
SELECT listing_id, posted_at, asking_price_aed, size_sqm, bedrooms,
       area_id, building_name, project_name, agent_id, photo_set_id
FROM listings.listings
ORDER BY listing_id
"""
LISTING_ATTRIBUTE_SCHEMA = {
    "listing_id": pl.Int64,
    "posted_at": pl.Date,
    "asking_price_aed": pl.Float64,
    "size_sqm": pl.Float64,
    "bedrooms": pl.Int64,
    "area_id": pl.Int64,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "agent_id": pl.Int64,
    "photo_set_id": pl.Int64,
}
VECTOR_SQL = """
SELECT listing_id, text_embedding::text, image_embedding::text
FROM listings.listing_embeddings
"""
PHOTO_SQL = """
SELECT lp.listing_id, lp.photo_id, p.embedding::text
FROM listings.listing_photos lp
JOIN listings.photos p ON p.photo_id = lp.photo_id
WHERE p.embedding IS NOT NULL
ORDER BY lp.listing_id, lp.position
"""


def parse_vector(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    return np.array([float(value) for value in text.strip("[]").split(",")])


def load_listing_attributes(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(LISTING_SQL)
        return pl.DataFrame(cur.fetchall(), schema=LISTING_ATTRIBUTE_SCHEMA, orient="row")


def load_listing_vectors(conn) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    text: dict[int, np.ndarray] = {}
    image: dict[int, np.ndarray] = {}
    with conn.cursor() as cur:
        cur.execute(VECTOR_SQL)
        for listing_id, text_vector, image_vector in cur.fetchall():
            parsed_text, parsed_image = parse_vector(text_vector), parse_vector(image_vector)
            if parsed_text is not None:
                text[listing_id] = parsed_text
            if parsed_image is not None:
                image[listing_id] = parsed_image
    return text, image


def load_photo_sets(conn) -> tuple[dict[int, list[int]], dict[int, np.ndarray]]:
    listing_photos: dict[int, list[int]] = {}
    vectors: dict[int, np.ndarray] = {}
    with conn.cursor() as cur:
        cur.execute(PHOTO_SQL)
        for listing_id, photo_id, embedding in cur.fetchall():
            listing_photos.setdefault(listing_id, []).append(photo_id)
            if photo_id not in vectors:
                vectors[photo_id] = parse_vector(embedding)
    return listing_photos, vectors


def _dimension(vectors: dict[int, np.ndarray], fallback: int = 1) -> int:
    for vector in vectors.values():
        return int(vector.size)
    return fallback


def _stack(vectors: dict[int, np.ndarray], ids, dim: int) -> np.ndarray:
    zero = np.zeros(dim)
    return np.vstack([vectors.get(int(listing_id), zero) for listing_id in ids])


def _photo_similarity(
    listing_a,
    listing_b,
    listing_photo_ids: dict[int, list[int]],
    photo_vectors: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    matrices: dict[int, np.ndarray] = {}
    for listing_id, photo_ids in listing_photo_ids.items():
        stacked = [photo_vectors[pid] for pid in photo_ids if pid in photo_vectors]
        if stacked:
            matrices[listing_id] = np.vstack(stacked)
    sets = {listing_id: set(ids) for listing_id, ids in listing_photo_ids.items()}

    best = np.zeros(len(listing_a))
    shared = np.zeros(len(listing_a), dtype=np.int64)
    for index, (a, b) in enumerate(zip(listing_a, listing_b)):
        a, b = int(a), int(b)
        shared[index] = len(sets.get(a, set()) & sets.get(b, set()))
        left, right = matrices.get(a), matrices.get(b)
        if left is not None and right is not None:
            # Clipped at 0: a listing with no photos scores 0, so photos that actively
            # disagree must not rank below having no photos at all.
            best[index] = max(float((left @ right.T).max()), 0.0)
    return best, shared


def build_features(
    pairs: pl.DataFrame,
    attributes: pl.DataFrame,
    text_vectors: dict[int, np.ndarray],
    image_vectors: dict[int, np.ndarray],
    listing_photo_ids: dict[int, list[int]],
    photo_vectors: dict[int, np.ndarray],
) -> pl.DataFrame:
    # maintain_order="left": feature rows stay in the order the pairs were given in, so a
    # caller can line them up with its own bookkeeping and two runs give the same frame.
    joined = pairs.join(
        attributes, left_on="listing_a", right_on="listing_id", how="inner", maintain_order="left"
    ).join(
        attributes,
        left_on="listing_b",
        right_on="listing_id",
        how="inner",
        suffix="_b",
        maintain_order="left",
    )

    listing_a = joined["listing_a"].to_numpy()
    listing_b = joined["listing_b"].to_numpy()
    text_dim, image_dim = _dimension(text_vectors), _dimension(image_vectors)
    text_cosine = np.einsum(
        "ij,ij->i",
        _stack(text_vectors, listing_a, text_dim),
        _stack(text_vectors, listing_b, text_dim),
    )
    image_mean_cosine = np.einsum(
        "ij,ij->i",
        _stack(image_vectors, listing_a, image_dim),
        _stack(image_vectors, listing_b, image_dim),
    )
    image_max_cosine, shared_photo_count = _photo_similarity(
        listing_a, listing_b, listing_photo_ids, photo_vectors
    )

    def same(column: str) -> pl.Expr:
        return (pl.col(column) == pl.col(f"{column}_b")).fill_null(False).cast(pl.Int64)

    return joined.select(
        "listing_a",
        "listing_b",
        pl.Series("text_cosine", text_cosine),
        pl.Series("image_max_cosine", image_max_cosine),
        pl.Series("image_mean_cosine", image_mean_cosine),
        pl.Series("shared_photo_count", shared_photo_count).cast(pl.Int64),
        # log(a / b) rather than log(a) - log(b): for the near-equal prices of a repost the
        # difference of two logs loses precision to cancellation.
        (pl.col("asking_price_aed") / pl.col("asking_price_aed_b"))
        .log()
        .abs()
        .alias("abs_log_price_ratio"),
        (pl.col("size_sqm") / pl.col("size_sqm_b")).log().abs().alias("abs_log_size_ratio"),
        same("area_id").alias("same_area"),
        same("building_name").alias("same_building"),
        same("project_name").alias("same_project"),
        same("bedrooms").alias("bedrooms_equal"),
        (pl.col("posted_at") - pl.col("posted_at_b"))
        .dt.total_days()
        .abs()
        .cast(pl.Float64)
        .alias("days_apart"),
        same("agent_id").alias("same_agent"),
    ).select("listing_a", "listing_b", *PAIR_FEATURES)
