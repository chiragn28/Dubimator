from pathlib import Path

import numpy as np
import polars as pl
import pytest

from listings.config import CorpusConfig
from listings.embed import (
    IMAGE_DIM,
    TEXT_DIM,
    embed_listings,
    embed_photos,
    normalize,
    resolve_device,
)
from listings.photos import make_variant

CONFIG = CorpusConfig()


def cosine(a, b):
    return float(np.dot(a, b))


def test_normalize_makes_unit_rows_and_leaves_zeros_alone():
    matrix = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(matrix[0], [0.6, 0.8])
    assert np.allclose(matrix[1], [0.0, 0.0])


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cuda", "cpu")
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("tpu")


def test_fake_image_embedding_is_deterministic_and_normalised(photo_pool, fake_embedder):
    paths = [photo_pool.path(1, "kitchen"), photo_pool.path(2, "kitchen")]
    first, second = fake_embedder.embed_images(paths), fake_embedder.embed_images(paths)
    assert first.shape == (2, IMAGE_DIM)
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_fake_image_embedding_survives_editing(photo_pool, fake_embedder, tmp_path):
    source = photo_pool.path(1, "bedroom")
    edited = make_variant(source, tmp_path / "edited.jpg", CONFIG)
    other = photo_pool.path(4, "bedroom")
    vectors = fake_embedder.embed_images([source, edited, other])
    assert cosine(vectors[0], vectors[1]) > 0.9  # the edit keeps it close
    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2])


def test_fake_text_embedding_tracks_wording(fake_embedder):
    original = (
        "Just listed: 2-bedroom apartment in Marsa Dubai. Features include balcony, gym access."
    )
    reworded = "Available now: 2-bedroom flat in Marsa Dubai. Features include gym access, balcony."
    unrelated = "Exclusive villa in Hadaeq Sheikh Mohammed Bin Rashid with maid's room and pool."
    vectors = fake_embedder.embed_texts([original, reworded, unrelated])
    assert vectors.shape == (3, TEXT_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2])
    assert np.allclose(fake_embedder.embed_texts([original])[0], vectors[0])


def test_embed_photos_returns_literals_and_vectors(photo_pool, fake_embedder, tmp_path):
    photos = pl.DataFrame(
        {
            "photo_id": [11, 12],
            "path": ["photos/1_kitchen.jpg", "photos/2_kitchen.jpg"],
        }
    )
    frame, vectors = embed_photos(photos, fake_embedder, photo_pool.root.parent)
    assert frame.columns == ["photo_id", "embedding"]
    assert frame["photo_id"].to_list() == [11, 12]
    assert frame["embedding"][0].startswith("[") and frame["embedding"][0].endswith("]")
    assert len(frame["embedding"][0].strip("[]").split(",")) == IMAGE_DIM
    assert set(vectors) == {11, 12}
    assert np.allclose(np.linalg.norm(vectors[11]), 1.0)


def test_embed_photos_reports_a_missing_file(photo_pool, fake_embedder):
    photos = pl.DataFrame({"photo_id": [1], "path": ["photos/999_kitchen.jpg"]})
    with pytest.raises(FileNotFoundError, match="999_kitchen"):
        embed_photos(photos, fake_embedder, photo_pool.root.parent)


def test_embed_listings_averages_its_photos(fake_embedder):
    listings = pl.DataFrame(
        {
            "listing_id": [1, 2],
            "title": ["2-bedroom apartment in Marsa Dubai", "Villa in Dubai Hills"],
            "description": ["Balcony and gym access.", "Private pool and garden."],
        }
    )
    listing_photos = pl.DataFrame(
        {"listing_id": [1, 1, 2, 2], "photo_id": [10, 11, 12, 13], "position": [0, 1, 0, 1]}
    )
    vectors = {
        10: np.eye(IMAGE_DIM)[0],
        11: np.eye(IMAGE_DIM)[1],
        12: np.eye(IMAGE_DIM)[2],
        13: np.eye(IMAGE_DIM)[2],
    }
    frame = embed_listings(listings, listing_photos, vectors, fake_embedder)
    assert frame.columns == ["listing_id", "text_embedding", "image_embedding"]
    first = np.array([float(x) for x in frame["image_embedding"][0].strip("[]").split(",")])
    assert np.allclose(first[:2], [0.5**0.5, 0.5**0.5])  # mean of two unit axes, renormalised
    second = np.array([float(x) for x in frame["image_embedding"][1].strip("[]").split(",")])
    assert np.allclose(second[2], 1.0)  # both photos identical -> that axis


def test_a_listing_with_no_photos_gets_a_zero_image_vector(fake_embedder):
    listings = pl.DataFrame({"listing_id": [5], "title": ["Studio"], "description": ["Small."]})
    empty = pl.DataFrame(
        {"listing_id": [], "photo_id": [], "position": []},
        schema={"listing_id": pl.Int64, "photo_id": pl.Int64, "position": pl.Int64},
    )
    frame = embed_listings(listings, empty, {}, fake_embedder)
    values = [float(x) for x in frame["image_embedding"][0].strip("[]").split(",")]
    assert values == [0.0] * IMAGE_DIM


def clip_is_cached() -> bool:
    """True only when the CLIP weights are already on disk. This test never downloads."""
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    return any(cache.glob("models--sentence-transformers--clip-ViT-B-32"))


@pytest.mark.skipif(not clip_is_cached(), reason="CLIP weights not cached; opt-in test")
def test_real_clip_embeds_photos(photo_pool):
    from listings.embed import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(device="cpu")
    vectors = embedder.embed_images([photo_pool.path(1, "kitchen"), photo_pool.path(2, "kitchen")])
    assert vectors.shape == (2, IMAGE_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
