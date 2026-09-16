import random
from datetime import date, timedelta

import numpy as np
import polars as pl

from ingestion.normalize import match_key
from listings.generate import LISTING_SCHEMA
from listings.text import ListingFacts, make_description, make_title
from search.config import listing_kind

AREAS = (
    (1, "Dubai Marina"),
    (2, "Jumeirah Village Circle"),
    (3, "Business Bay"),
    (4, "Downtown Dubai"),
    (5, "Arabian Ranches"),
    (6, "Palm Jumeirah"),
)
ALIASES = (("JVC", 2), ("Marina", 1), ("Downtown", 4), ("The Palm", 6))
BUILDINGS = {
    1: ("Marina Gate", "Princess Tower"),
    2: ("Belgravia Heights",),
    3: ("Executive Towers", "Bay Square"),
    4: ("Burj Views", "Address Residence"),
    5: (None,),
    6: ("Shoreline Apartments",),
}
SUB_TYPES = ("Flat", "Flat", "Hotel Apartment", "Stacked Townhouses")
N_CLONES = 10
BAIT_IDS = tuple(range(11, 17))  # ground-truth bait listings
PREDICTED_BAIT_IDS = BAIT_IDS[:4]  # the "detector" only caught four of them


def build_search_listings(n: int = 240, seed: int = 5) -> pl.DataFrame:
    """Hand-built listings over six real Dubai area names. The last N_CLONES rows are exact
    reposts of listings 1..N_CLONES; listings BAIT_IDS are bait-priced (ground truth)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n - N_CLONES):
        area_id, area_name = AREAS[i % len(AREAS)]
        villa = area_id == 5 or (area_id == 6 and i % 3 == 0)
        property_type = "villa" if villa else "unit"
        sub_type = None if villa else SUB_TYPES[i % len(SUB_TYPES)]
        kind = listing_kind(property_type, sub_type)
        building = None if villa else BUILDINGS[area_id][i % len(BUILDINGS[area_id])]
        bedrooms = 3 + i % 3 if villa else i % 4
        size = 250.0 + 10.0 * (i % 20) if villa else 40.0 + 25.0 * bedrooms + i % 7
        price = round((size * (9_000 + 1_500 * area_id) + (i % 11) * 20_000) / 10_000) * 10_000
        listing_id = i + 1
        fraud_label = None
        if listing_id in BAIT_IDS:
            fraud_label, price = "bait_price", round(price * 0.5 / 10_000) * 10_000
        facts = ListingFacts(
            bedrooms, size, area_name, None, building, f"Project {area_id}", property_type, kind
        )
        rows.append(
            {
                "listing_id": listing_id,
                "source_transaction_id": f"s{i}",
                "agent_id": i % 20 + 1,
                "posted_at": date(2023, 1, 1) + timedelta(days=i % 180),
                "title": make_title(rng, facts),
                "description": make_description(rng, facts),
                "asking_price_aed": float(price),
                "area_id": area_id,
                "area_name": area_name,
                "building_name": building,
                "project_name": f"Project {area_id}",
                "property_type": property_type,
                "property_sub_type": sub_type,
                "reg_type": "ready",
                "size_sqm": size,
                "bedrooms": bedrooms,
                "photo_set_id": 1,
                "dup_group_id": None,
                "control_group_id": None,
                "fraud_label": fraud_label,
                "is_synthetic": True,
            }
        )
    for offset in range(N_CLONES):
        source = dict(rows[offset])
        source.update(
            listing_id=n - N_CLONES + offset + 1,
            source_transaction_id=f"c{offset}",
            posted_at=source["posted_at"] + timedelta(days=5),
            dup_group_id=source["listing_id"],
            agent_id=source["agent_id"] % 20 + 1,
        )
        rows.append(source)
    return pl.DataFrame(rows, schema=LISTING_SCHEMA)


def _area_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    areas = pl.DataFrame(
        {
            "area_id": [area_id for area_id, _ in AREAS],
            "name_en": [name for _, name in AREAS],
            "name_ar": [None] * len(AREAS),
            "match_key": [match_key(name) for _, name in AREAS],
            "market_sales": [100] * len(AREAS),
        },
        schema={
            "area_id": pl.Int64,
            "name_en": pl.Utf8,
            "name_ar": pl.Utf8,
            "match_key": pl.Utf8,
            "market_sales": pl.Int64,
        },
    )
    official = [(match_key(name), name, area_id, "official") for area_id, name in AREAS]
    curated = [(match_key(alias), alias, area_id, "curated") for alias, area_id in ALIASES]
    aliases = pl.DataFrame(
        official + curated,
        schema={"alias_key": pl.Utf8, "alias": pl.Utf8, "area_id": pl.Int64, "source": pl.Utf8},
        orient="row",
    )
    return areas, aliases


def seed_search_db(settings, listings: pl.DataFrame) -> int:
    """dld areas + aliases, the listings corpus with fake text vectors and HNSW indexes, one
    detect run (clone pairs flagged, four predicted bait flags), and the search schema."""
    from ingestion.load import apply_schema as apply_dld_schema
    from ingestion.load import copy_frame
    from listings.embed import IMAGE_DIM, FakeEmbedder
    from listings.load import apply_schema as apply_listings_schema
    from listings.load import (
        create_vector_indexes,
        start_corpus_run,
        vector_literal,
        write_listing_embeddings,
    )
    from search.store import apply_schema as apply_search_schema

    areas, aliases = _area_frames()
    conn = settings.connect()
    try:
        apply_dld_schema(conn)
        with conn.cursor() as cur:
            copy_frame(cur, "dld.areas", areas)
            copy_frame(cur, "dld.area_aliases", aliases)
        apply_listings_schema(conn)
        corpus_run_id = start_corpus_run(conn, 5, "test", {"listings": listings.height})
        with conn.cursor() as cur:
            copy_frame(
                cur,
                "listings.listings",
                listings.with_columns(pl.lit(corpus_run_id, dtype=pl.Int64).alias("corpus_run_id")),
            )
        texts = [f"{t}\n{d}" for t, d in zip(listings["title"], listings["description"])]
        vectors = FakeEmbedder().embed_texts(texts)
        image = np.zeros(IMAGE_DIM)
        image[0] = 1.0  # a zero vector has no cosine; any fixed unit vector will do
        write_listing_embeddings(
            conn,
            pl.DataFrame(
                {
                    "listing_id": listings["listing_id"],
                    "text_embedding": [vector_literal(v) for v in vectors],
                    "image_embedding": [vector_literal(image)] * listings.height,
                }
            ),
        )
        create_vector_indexes(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO listings.detect_runs (corpus_run_id, threshold) "
                "VALUES (%s, 0.5) RETURNING detect_run_id",
                (corpus_run_id,),
            )
            (detect_run_id,) = cur.fetchone()
            clones = listings.filter(pl.col("dup_group_id").is_not_null())
            for clone_id, source_id in clones.select("listing_id", "dup_group_id").iter_rows():
                cur.execute(
                    "INSERT INTO listings.duplicate_pairs "
                    "VALUES (%s, %s, %s, 0.99, true, '{}'::jsonb)",
                    (detect_run_id, min(clone_id, source_id), max(clone_id, source_id)),
                )
            for listing_id in PREDICTED_BAIT_IDS:
                cur.execute(
                    "INSERT INTO listings.fraud_flags VALUES (%s, %s, 'bait_price', '{}'::jsonb)",
                    (detect_run_id, listing_id),
                )
        apply_search_schema(conn)
        conn.commit()
    finally:
        conn.close()
    return corpus_run_id
