import math

import polars as pl

MARKET_SALE_PROCEDURES = frozenset(
    {"Sell", "Sell - Pre registration", "Delayed Sell", "Sale On Payment Plan"}
)
REASONS = (
    "duplicate_transaction_id",
    "mortgage",
    "gift",
    "non_market_procedure",
    "missing_date",
    "missing_price",
    "invalid_area",
    "price_below_floor",
    "suspected_sqft_entry",
    "price_outlier_low",
    "price_outlier_high",
)
PEER_TIERS = (
    ("area_id", "property_type", "reg_type", "year"),
    ("area_id", "property_type", "year"),
    ("property_type", "reg_type", "year"),
    ("property_type", "year"),
    ("property_type",),
)
MIN_PEER_GROUP = 30
Z_THRESHOLD = 3.5
PRICE_FLOOR_AED = 10_000
SQFT_PER_SQM = 10.7639
MAD_SCALE = 1.4826
_LN_SQFT_PER_SQM = math.log(SQFT_PER_SQM)


class NoPeerGroupError(RuntimeError):
    pass


def _deterministic_reason() -> pl.Expr:
    procedure = pl.col("procedure_name")
    area = pl.col("area_sqm")
    return (
        pl.when(~pl.col("transaction_id").is_first_distinct())
        .then(pl.lit("duplicate_transaction_id"))
        .when(pl.col("trans_group") == "mortgages")
        .then(pl.lit("mortgage"))
        .when(pl.col("trans_group") == "gifts")
        .then(pl.lit("gift"))
        .when(procedure.is_null() | ~procedure.is_in(list(MARKET_SALE_PROCEDURES)))
        .then(pl.lit("non_market_procedure"))
        .when(pl.col("instance_date").is_null())
        .then(pl.lit("missing_date"))
        .when(pl.col("price_aed").is_null())
        .then(pl.lit("missing_price"))
        .when(area.is_null() | (area <= 0))
        .then(pl.lit("invalid_area"))
        .when(pl.col("price_aed") < PRICE_FLOOR_AED)
        .then(pl.lit("price_below_floor"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


def _by_tier(prefix: str) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Float64)
    for tier in reversed(range(1, len(PEER_TIERS) + 1)):
        expr = pl.when(pl.col("peer_tier") == tier).then(pl.col(f"{prefix}_{tier}")).otherwise(expr)
    return expr


def _outlier_columns(candidates: pl.DataFrame) -> pl.DataFrame:
    if candidates.height == 0:
        return pl.DataFrame(
            schema={
                "source_row": pl.Int64,
                "peer_tier": pl.Int16,
                "price_robust_z": pl.Float64,
                "outlier_reason": pl.Utf8,
            }
        )
    c = candidates.select(
        "source_row",
        "property_type",
        "reg_type",
        "year",
        pl.col("area_id").fill_null(-1),
        pl.col("price_per_sqm_aed").log().alias("x"),
    )
    for tier, keys in enumerate(PEER_TIERS, start=1):
        keys = list(keys)
        c = c.with_columns(
            pl.col("x").median().over(keys).alias(f"med_{tier}"),
            pl.len().over(keys).alias(f"n_{tier}"),
        ).with_columns(
            (pl.col("x") - pl.col(f"med_{tier}")).abs().median().over(keys).alias(f"mad_{tier}")
        )

    tier_expr = pl.lit(None, dtype=pl.Int16)
    for tier in reversed(range(1, len(PEER_TIERS) + 1)):
        qualifies = (pl.col(f"n_{tier}") >= MIN_PEER_GROUP) & (pl.col(f"mad_{tier}") > 0)
        tier_expr = pl.when(qualifies).then(pl.lit(tier, dtype=pl.Int16)).otherwise(tier_expr)
    c = c.with_columns(tier_expr.alias("peer_tier"))

    missing = c["peer_tier"].null_count()
    if missing:
        raise NoPeerGroupError(
            f"{missing} rows have no peer group with >= {MIN_PEER_GROUP} rows and non-zero spread"
        )

    scale = MAD_SCALE * _by_tier("mad")
    median = _by_tier("med")
    c = c.with_columns(
        ((pl.col("x") - median) / scale).alias("price_robust_z"),
        ((pl.col("x") + _LN_SQFT_PER_SQM - median) / scale).alias("z_sqft"),
    )
    z = pl.col("price_robust_z")
    outlier_reason = (
        pl.when((z < -Z_THRESHOLD) & (pl.col("z_sqft").abs() <= Z_THRESHOLD))
        .then(pl.lit("suspected_sqft_entry"))
        .when(z < -Z_THRESHOLD)
        .then(pl.lit("price_outlier_low"))
        .when(z > Z_THRESHOLD)
        .then(pl.lit("price_outlier_high"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )
    return c.select(
        "source_row", "peer_tier", "price_robust_z", outlier_reason.alias("outlier_reason")
    )


def classify(typed: pl.DataFrame) -> pl.DataFrame:
    df = typed.sort("source_row").with_columns(_deterministic_reason().alias("exclusion_reason"))
    outliers = _outlier_columns(df.filter(pl.col("exclusion_reason").is_null()))
    return (
        df.join(outliers, on="source_row", how="left")
        .with_columns(pl.coalesce("exclusion_reason", "outlier_reason").alias("exclusion_reason"))
        .drop("outlier_reason")
        .sort("source_row")
    )
