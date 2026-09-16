"""Photo and text embeddings.

Everything downstream talks to the `Embedder` protocol, so tests inject a
deterministic double and never download a model or need a GPU.
"""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
from PIL import Image

from listings.load import vector_literal

IMAGE_DIM = 512
TEXT_DIM = 384
IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICES = ("auto", "cuda", "cpu")
_WORD = re.compile(r"[a-z0-9؀-ۿ]+")


class Embedder(Protocol):
    def embed_images(self, paths: Sequence[Path]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def resolve_device(requested: str = "auto") -> str:
    if requested not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {requested!r}")
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"


@dataclass
class SentenceTransformerEmbedder:
    """CLIP ViT-B/32 for photos, all-MiniLM-L6-v2 for text. Models load on first use."""

    device: str = "cpu"
    batch_size: int = 64
    _image_model: object | None = field(default=None, repr=False)
    _text_model: object | None = field(default=None, repr=False)

    def _model(self, name: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name, device=self.device)

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        if self._image_model is None:
            self._image_model = self._model(IMAGE_MODEL)
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        vectors = self._image_model.encode(
            images, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return normalize(vectors)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self._text_model is None:
            self._text_model = self._model(TEXT_MODEL)
        vectors = self._text_model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return normalize(vectors)


class FakeEmbedder:
    """Deterministic stand-in: a 16x16 grey thumbnail for photos, hashed words for text.

    Not a model, but a real similarity measure: an edited photo stays close to its
    source and a reworded description stays closer to its source than to an
    unrelated listing, which is what the detection tests need.
    """

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        rows = []
        for path in paths:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"photo not found: {path}")
            with Image.open(path) as image:
                thumbnail = image.convert("L").resize((16, 16), Image.Resampling.LANCZOS)
            values = np.asarray(thumbnail, dtype=np.float64).reshape(-1)
            vector = np.zeros(IMAGE_DIM, dtype=np.float64)
            vector[: values.size] = values - values.mean()
            rows.append(vector)
        return normalize(np.vstack(rows)) if rows else np.zeros((0, IMAGE_DIM))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(TEXT_DIM, dtype=np.float64)
            for word in _WORD.findall(text.lower()):
                digest = hashlib.sha256(word.encode("utf-8")).digest()
                vector[int.from_bytes(digest[:4], "big") % TEXT_DIM] += 1.0
            rows.append(vector)
        return normalize(np.vstack(rows)) if rows else np.zeros((0, TEXT_DIM))


def embed_photos(
    photos: pl.DataFrame, embedder: Embedder, data_dir: Path
) -> tuple[pl.DataFrame, dict[int, np.ndarray]]:
    """Embed every photo row (`photo_id`, `path`), paths being relative to data_dir."""
    data_dir = Path(data_dir)
    photo_ids = photos["photo_id"].to_list()
    paths = [data_dir / path for path in photos["path"].to_list()]
    vectors = embedder.embed_images(paths) if paths else np.zeros((0, IMAGE_DIM))
    frame = pl.DataFrame(
        {
            "photo_id": photo_ids,
            "embedding": [vector_literal(vector) for vector in vectors],
        },
        schema={"photo_id": pl.Int64, "embedding": pl.Utf8},
    )
    return frame, dict(zip(photo_ids, vectors))


def embed_listings(
    listings: pl.DataFrame,
    listing_photos: pl.DataFrame,
    photo_vectors: dict[int, np.ndarray],
    embedder: Embedder,
) -> pl.DataFrame:
    """Text vectors from title + description; image vectors as the mean of a listing's photos."""
    texts = [
        f"{title}\n{description}"
        for title, description in zip(
            listings["title"].to_list(), listings["description"].to_list()
        )
    ]
    text_vectors = embedder.embed_texts(texts)

    grouped: dict[int, list[np.ndarray]] = {}
    for row in listing_photos.iter_rows(named=True):
        vector = photo_vectors.get(row["photo_id"])
        if vector is not None:
            grouped.setdefault(row["listing_id"], []).append(vector)

    image_vectors = []
    for listing_id in listings["listing_id"].to_list():
        vectors = grouped.get(listing_id)
        mean = np.mean(vectors, axis=0) if vectors else np.zeros(IMAGE_DIM)
        image_vectors.append(normalize(mean))

    return pl.DataFrame(
        {
            "listing_id": listings["listing_id"].to_list(),
            "text_embedding": [vector_literal(vector) for vector in text_vectors],
            "image_embedding": [vector_literal(vector) for vector in image_vectors],
        },
        schema={"listing_id": pl.Int64, "text_embedding": pl.Utf8, "image_embedding": pl.Utf8},
    )
