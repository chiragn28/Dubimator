import io
import math
import zipfile

import pytest
from PIL import Image

from listings.config import PHOTO_DATASET_DIR, PHOTO_ROOMS


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
    from listings.config import CorpusConfig
    from listings.photos import ensure_pool

    data = build_archive(range(1, 7))
    return ensure_pool(CorpusConfig(), data_dir=tmp_path, fetch=lambda _url: data)
