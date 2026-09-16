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
