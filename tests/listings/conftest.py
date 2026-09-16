import io
import math
import zipfile
from datetime import date, timedelta

import polars as pl
import pytest
from PIL import Image

from listings.config import PHOTO_DATASET_DIR, PHOTO_ROOMS, CorpusConfig
from listings.generate import LISTING_SCHEMA  # noqa: F401  (schema sanity at import time)

SALES_SCHEMA = {
    "transaction_id": pl.Utf8,
    "instance_date": pl.Date,
    "property_type": pl.Utf8,
    "property_sub_type": pl.Utf8,
    "reg_type": pl.Utf8,
    "area_id": pl.Int64,
    "area_name": pl.Utf8,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "bedrooms": pl.Int64,
    "area_sqm": pl.Float64,
    "price_aed": pl.Float64,
}


def make_image(seed: int, size: tuple[int, int] = (256, 192)) -> Image.Image:
    """A deterministic image built from low-frequency waves.

    Smooth and seed-specific on purpose: cropping or resizing barely changes it (so
    edited-photo tests stay meaningful), while different seeds look genuinely different.
    """
    image = Image.new("RGB", size)
    pixels = image.load()
    across, down = 1 + seed % 3, 1 + (seed // 3) % 3
    for x in range(size[0]):
        for y in range(size[1]):
            red = int(127 + 120 * math.sin(across * math.pi * x / size[0]))
            green = int(127 + 120 * math.sin(down * math.pi * y / size[1]))
            pixels[x, y] = (red, green, (seed * 37) % 256)
    return image


def build_archive(set_ids, rooms=PHOTO_ROOMS) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for set_id in set_ids:
            for index, room in enumerate(rooms):
                photo = io.BytesIO()
                make_image(set_id * 10 + index).save(photo, format="JPEG", quality=92)
                archive.writestr(f"{PHOTO_DATASET_DIR}/{set_id}_{room}.jpg", photo.getvalue())
    return buffer.getvalue()


@pytest.fixture
def photo_archive():
    return build_archive


@pytest.fixture
def photo_pool(tmp_path):
    """An extracted 6-set pool under tmp_path, ready for ensure_pool to find."""
    from listings.photos import ensure_pool

    data = build_archive(range(1, 7))
    return ensure_pool(CorpusConfig(), data_dir=tmp_path, fetch=lambda _url: data)


def build_sales(n: int = 3_000, areas: int = 12, buildings_per_area: int = 5) -> pl.DataFrame:
    """Synthetic DLD-shaped sales: every building gets enough sales to be 'busy'."""
    rows = []
    for index in range(n):
        area_id = index % areas + 1
        building = index // areas % buildings_per_area + 1
        villa = index % 7 == 0
        rows.append(
            {
                "transaction_id": f"t{index}",
                "instance_date": date(2021, 1, 1) + timedelta(days=index % 800),
                "property_type": "villa" if villa else "unit",
                "property_sub_type": None if villa else "Flat",
                "reg_type": "off_plan" if index % 5 == 0 else "ready",
                "area_id": area_id,
                "area_name": f"Area {area_id}",
                "building_name": None if villa else f"Tower {area_id}-{building}",
                "project_name": f"Project {area_id}-{building}",
                "bedrooms": None if villa else index % 4,
                "area_sqm": 60.0 + (index % 40) * 5.0,
                "price_aed": 700_000.0 + (index % 50) * 25_000.0,
            }
        )
    return pl.DataFrame(rows, schema=SALES_SCHEMA)


@pytest.fixture
def sales_frame():
    return build_sales


@pytest.fixture
def areas_frame():
    return pl.DataFrame(
        {
            "area_id": list(range(1, 13)),
            "name_en": [f"Area {i}" for i in range(1, 13)],
            "name_ar": [f"منطقة {i}" for i in range(1, 13)],
        },
        schema={"area_id": pl.Int64, "name_en": pl.Utf8, "name_ar": pl.Utf8},
    )


@pytest.fixture
def small_corpus_config():
    """A 300-listing corpus: same shape as production, small enough for unit tests."""
    return CorpusConfig(
        n_listings=300,
        n_base=240,
        n_exact_repost=30,
        n_reworded=15,
        n_edited_photo=15,
        n_bait_price=12,
        n_price_shifted_reposts=8,
        n_from_busy_buildings=40,
        n_stock_sets=2,
        n_agents=25,
        min_building_sales=4,
        stock_min_areas=3,
    )


@pytest.fixture
def fake_embedder():
    from listings.embed import FakeEmbedder

    return FakeEmbedder()


@pytest.fixture
def loaded_corpus(pg_test_db, sales_frame, areas_frame, small_corpus_config, photo_pool, tmp_path):
    """A small corpus in Postgres with fake embeddings and HNSW indexes built."""
    from listings.embed import FakeEmbedder, embed_listings, embed_photos
    from listings.generate import generate_corpus, render_variants
    from listings.load import (
        create_vector_indexes,
        load_corpus,
        write_listing_embeddings,
        write_photo_embeddings,
    )

    config = small_corpus_config
    # The 6-set pool leaves 4 non-stock sets, so the sales must span at most 4 areas for every
    # area to get a local plain set (see generate._plain_sets_by_area).
    corpus = generate_corpus(sales_frame(areas=4), areas_frame, photo_pool.set_ids, config)
    data_dir = photo_pool.root.parent
    render_variants(corpus, photo_pool, config, data_dir / "variants")
    corpus_run_id = load_corpus(pg_test_db, corpus, seed=config.seed, photo_dataset_sha="test")

    embedder = FakeEmbedder()
    photo_frame, photo_vectors = embed_photos(corpus.photos, embedder, data_dir)
    listing_frame = embed_listings(corpus.listings, corpus.listing_photos, photo_vectors, embedder)
    conn = pg_test_db.connect()
    try:
        write_photo_embeddings(conn, photo_frame)
        write_listing_embeddings(conn, listing_frame)
        create_vector_indexes(conn)
        conn.commit()
    finally:
        conn.close()
    return pg_test_db, corpus, corpus_run_id
