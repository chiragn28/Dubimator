"""Three fraud flags.

bait_price uses the Phase 3 champion price model, so a listing priced far under
what the market model says is flagged. photo_reuse and inconsistent_relist are
corpus-level patterns. None of them reads a ground-truth label.
"""

import json
import logging
from dataclasses import dataclass

import polars as pl

from ingestion.load import copy_frame
from listings.config import DetectConfig

try:  # the predictor ships with Phase 3; tests stub it
    from models.price.predictor import PriceInputError
except ImportError:  # pragma: no cover

    class PriceInputError(ValueError):  # type: ignore[no-redef]
        pass


LOGGER = logging.getLogger(__name__)
FLAGS = ("bait_price", "photo_reuse", "inconsistent_relist")
FLAG_SCHEMA = {"listing_id": pl.Int64, "flag": pl.Utf8, "detail": pl.Utf8}
KIND_BY_SUB_TYPE = {
    "Flat": "apartment",
    "Hotel Apartment": "hotel_apartment",
    "Stacked Townhouses": "townhouse",
}
FRAUD_LISTING_SQL = """
SELECT listing_id, asking_price_aed, area_id, area_name, building_name, project_name,
       property_type, property_sub_type, reg_type, size_sqm, bedrooms, photo_set_id
FROM listings.listings
ORDER BY listing_id
"""
FRAUD_LISTING_SCHEMA = {
    "listing_id": pl.Int64,
    "asking_price_aed": pl.Float64,
    "area_id": pl.Int64,
    "area_name": pl.Utf8,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "property_type": pl.Utf8,
    "property_sub_type": pl.Utf8,
    "reg_type": pl.Utf8,
    "size_sqm": pl.Float64,
    "bedrooms": pl.Int64,
    "photo_set_id": pl.Int64,
}


@dataclass(frozen=True)
class FraudResult:
    flags: pl.DataFrame
    stats: dict[str, float]


def load_fraud_attributes(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(FRAUD_LISTING_SQL)
        return pl.DataFrame(cur.fetchall(), schema=FRAUD_LISTING_SCHEMA, orient="row")


def load_price_predictor(uri: str):
    """The Phase 3 champion, or None when it cannot be loaded (detection carries on)."""
    try:
        import mlflow

        return mlflow.pyfunc.load_model(uri).unwrap_python_model().predictor
    except Exception as exc:  # noqa: BLE001 — any failure here is non-fatal by design
        LOGGER.warning("price model %s unavailable (%s); skipping the bait_price flag", uri, exc)
        return None


def price_request(row: dict) -> dict:
    """A listing row as a PriceRequest payload; unknown fields are left out, never None."""
    villa = row["property_type"] == "villa"
    request = {
        "area_id": int(row["area_id"]),
        "property_kind": "villa" if villa else KIND_BY_SUB_TYPE[row["property_sub_type"]],
        "status": row["reg_type"],
        "size_sqm": float(row["size_sqm"]),
        "size_basis": "plot" if villa and row["property_sub_type"] is None else "built_up",
        "asking_price_aed": float(row["asking_price_aed"]),
    }
    if row.get("building_name"):
        request["building"] = row["building_name"]
    if row.get("project_name"):
        request["project"] = row["project_name"]
    if row.get("bedrooms") is not None:
        request["bedrooms"] = int(row["bedrooms"])
    return request


def bait_price_flags(
    attributes: pl.DataFrame, predictor, config: DetectConfig, log_every: int = 2_000
) -> tuple[pl.DataFrame, dict[str, float]]:
    rows = []
    checked = unsupported = 0
    for index, listing in enumerate(attributes.iter_rows(named=True), start=1):
        checked += 1
        try:
            estimate = predictor.predict_one(price_request(listing))
        except PriceInputError as exc:
            unsupported += 1
            LOGGER.debug("listing %s cannot be priced: %s", listing["listing_id"], exc)
            continue
        low = float(estimate.range_80[0])
        asking = float(listing["asking_price_aed"])
        if asking < low * (1.0 - config.bait_margin):
            rows.append(
                {
                    "listing_id": listing["listing_id"],
                    "flag": "bait_price",
                    "detail": json.dumps(
                        {
                            "asking_price_aed": asking,
                            "estimate_aed": float(estimate.estimate_aed),
                            "range_80_low": low,
                            "below_low_pct": round((1.0 - asking / low) * 100.0, 1),
                        }
                    ),
                }
            )
        if index % log_every == 0:
            LOGGER.info("priced %s of %s listings", index, attributes.height)
    stats = {
        "bait_price_checked": float(checked),
        "bait_price_unsupported": float(unsupported),
        "bait_price_skipped": 0.0,
        "bait_price_flagged": float(len(rows)),
    }
    return pl.DataFrame(rows, schema=FLAG_SCHEMA), stats


def photo_reuse_flags(attributes: pl.DataFrame, config: DetectConfig) -> pl.DataFrame:
    spread = (
        attributes.group_by("photo_set_id")
        .agg(pl.col("area_id").n_unique().alias("areas"), pl.len().alias("listings"))
        .filter(pl.col("areas") >= config.photo_reuse_min_areas)
    )
    flagged = attributes.join(spread, on="photo_set_id", how="inner")
    return pl.DataFrame(
        [
            {
                "listing_id": row["listing_id"],
                "flag": "photo_reuse",
                "detail": json.dumps(
                    {
                        "photo_set_id": row["photo_set_id"],
                        "areas": row["areas"],
                        "listings": row["listings"],
                    }
                ),
            }
            for row in flagged.iter_rows(named=True)
        ],
        schema=FLAG_SCHEMA,
    )


def _clusters(pairs: pl.DataFrame) -> list[list[int]]:
    """Connected components over flagged pairs (union-find)."""
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in zip(pairs["listing_a"].to_list(), pairs["listing_b"].to_list()):
        root_a, root_b = find(int(a)), find(int(b))
        if root_a != root_b:
            parent[root_b] = root_a
    groups: dict[int, list[int]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [sorted(members) for members in groups.values() if len(members) > 1]


def inconsistent_relist_flags(
    attributes: pl.DataFrame, flagged_pairs: pl.DataFrame, config: DetectConfig
) -> pl.DataFrame:
    prices = dict(zip(attributes["listing_id"].to_list(), attributes["asking_price_aed"].to_list()))
    rows = []
    for cluster in _clusters(flagged_pairs):
        values = [prices[listing_id] for listing_id in cluster if listing_id in prices]
        if len(values) < 2 or min(values) <= 0:
            continue
        spread = max(values) / min(values) - 1.0
        if spread > config.relist_price_spread:
            detail = json.dumps(
                {
                    "spread": round(spread, 4),
                    "cluster_size": len(cluster),
                    "min_price_aed": min(values),
                    "max_price_aed": max(values),
                }
            )
            rows.extend(
                {"listing_id": listing_id, "flag": "inconsistent_relist", "detail": detail}
                for listing_id in cluster
            )
    return pl.DataFrame(rows, schema=FLAG_SCHEMA)


_MISSING = object()


def run_fraud_checks(
    conn, flagged_pairs: pl.DataFrame, config: DetectConfig, predictor=_MISSING
) -> FraudResult:
    attributes = load_fraud_attributes(conn)
    if predictor is _MISSING:
        predictor = load_price_predictor(config.price_model_uri)

    frames = [photo_reuse_flags(attributes, config)]
    stats = {"listings": float(attributes.height)}
    if predictor is None:
        stats |= {"bait_price_skipped": 1.0, "bait_price_flagged": 0.0}
    else:
        bait, bait_stats = bait_price_flags(attributes, predictor, config)
        frames.append(bait)
        stats |= bait_stats
    frames.append(inconsistent_relist_flags(attributes, flagged_pairs, config))

    flags = pl.concat(frames).sort(["flag", "listing_id"])
    for name in FLAGS:
        stats.setdefault(f"{name}_flagged", float(flags.filter(pl.col("flag") == name).height))
    return FraudResult(flags, stats)


def write_fraud_flags(conn, result: FraudResult, detect_run_id: int) -> int:
    if result.flags.is_empty():
        return 0
    frame = result.flags.select(
        pl.lit(detect_run_id, dtype=pl.Int64).alias("detect_run_id"),
        "listing_id",
        "flag",
        "detail",
    )
    with conn.cursor() as cur:
        copy_frame(cur, "listings.fraud_flags", frame)
    return frame.height
