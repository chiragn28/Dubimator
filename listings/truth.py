"""Ground truth for the synthetic corpus.

This is the ONLY module allowed to read dup_group_id, control_group_id or
fraud_label. Detection reads listing content and vectors; labels are training
targets and evaluation answers, never inputs.
"""

import polars as pl

TRUTH_SQL = """
SELECT listing_id, dup_group_id, control_group_id, fraud_label
FROM listings.listings
ORDER BY listing_id
"""
TRUTH_SCHEMA = {
    "listing_id": pl.Int64,
    "dup_group_id": pl.Int64,
    "control_group_id": pl.Utf8,
    "fraud_label": pl.Utf8,
}


def load_truth(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(TRUTH_SQL)
        return pl.DataFrame(cur.fetchall(), schema=TRUTH_SCHEMA, orient="row")


def label_pairs(pairs: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Add is_duplicate and same_building_control to candidate pairs."""
    # maintain_order="left": callers index labels against the pairs they passed in.
    joined = pairs.join(
        truth, left_on="listing_a", right_on="listing_id", how="inner", maintain_order="left"
    ).join(
        truth,
        left_on="listing_b",
        right_on="listing_id",
        how="inner",
        suffix="_b",
        maintain_order="left",
    )
    cloned_from = (pl.col("dup_group_id") == pl.col("listing_b")) | (
        pl.col("dup_group_id_b") == pl.col("listing_a")
    )
    siblings = (
        pl.col("dup_group_id").is_not_null()
        & pl.col("dup_group_id_b").is_not_null()
        & (pl.col("dup_group_id") == pl.col("dup_group_id_b"))
    )
    is_duplicate = (cloned_from | siblings).fill_null(False)
    control = (
        pl.col("control_group_id").is_not_null()
        & (pl.col("control_group_id") == pl.col("control_group_id_b"))
        & ~is_duplicate
    ).fill_null(False)
    return joined.select(
        "listing_a",
        "listing_b",
        is_duplicate.alias("is_duplicate"),
        control.alias("same_building_control"),
    )


DUPLICATE_TRUTH_SQL = """
SELECT clone.listing_id AS clone_id,
       clone.dup_group_id AS source_id,
       (clone.description = source.description) AS same_text,
       EXISTS (
           SELECT 1
           FROM listings.listing_photos lp
           JOIN listings.photos p ON p.photo_id = lp.photo_id
           WHERE lp.listing_id = clone.listing_id AND p.variant_of IS NOT NULL
       ) AS edited_photos
FROM listings.listings clone
JOIN listings.listings source ON source.listing_id = clone.dup_group_id
WHERE clone.dup_group_id IS NOT NULL
"""


def load_duplicate_truth(conn) -> pl.DataFrame:
    """Planted clone -> source pairs, with the pattern read off the content."""
    with conn.cursor() as cur:
        cur.execute(DUPLICATE_TRUTH_SQL)
        rows = cur.fetchall()
    pattern = []
    listing_a, listing_b = [], []
    for clone_id, source_id, same_text, edited_photos in rows:
        listing_a.append(min(clone_id, source_id))
        listing_b.append(max(clone_id, source_id))
        if edited_photos:
            pattern.append("edited_photo")
        elif same_text:
            pattern.append("exact_repost")
        else:
            pattern.append("reworded")
    return pl.DataFrame(
        {"listing_a": listing_a, "listing_b": listing_b, "pattern": pattern},
        schema={"listing_a": pl.Int64, "listing_b": pl.Int64, "pattern": pl.Utf8},
    )


# A duplicate group is a planted source plus its clones (the clones' dup_group_id is the
# source's listing_id). Its listings are relist truth when the group's planted asking prices
# spread by more than `spread` — the same max/min - 1 test inconsistent_relist applies.
RELIST_TRUTH_SQL = """
WITH members AS (
    SELECT dup_group_id AS group_id, listing_id, asking_price_aed
    FROM listings.listings
    WHERE dup_group_id IS NOT NULL
    UNION
    SELECT source.listing_id AS group_id, source.listing_id, source.asking_price_aed
    FROM listings.listings source
    WHERE EXISTS (
        SELECT 1 FROM listings.listings clone WHERE clone.dup_group_id = source.listing_id
    )
), spread AS (
    SELECT group_id
    FROM members
    GROUP BY group_id
    HAVING count(*) > 1
       AND min(asking_price_aed) > 0
       AND max(asking_price_aed) / min(asking_price_aed) - 1 > %(spread)s
)
SELECT DISTINCT m.listing_id
FROM members m
JOIN spread USING (group_id)
ORDER BY m.listing_id
"""


def load_relist_truth(conn, spread: float) -> pl.DataFrame:
    """Listings in planted duplicate groups whose asking prices differ by more than `spread`."""
    with conn.cursor() as cur:
        cur.execute(RELIST_TRUTH_SQL, {"spread": spread})
        return pl.DataFrame(cur.fetchall(), schema={"listing_id": pl.Int64}, orient="row")
