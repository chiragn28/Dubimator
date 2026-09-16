"""Fetch the Houses-dataset photo pool and build edited variants.

The dataset (emanhamed/Houses-dataset, 535 properties x 4 rooms) states no licence,
so the images are never committed: they are downloaded into data/listings/ (gitignored)
and cited in the README.
"""

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from listings.config import PHOTO_DATASET_DIR, PHOTO_DATASET_URL, PHOTO_ROOMS, CorpusConfig

DATA_DIR = Path("data/listings")
PHOTO_DIR = DATA_DIR / "photos"
VARIANT_DIR = DATA_DIR / "variants"
_PHOTO_NAME = re.compile(r"^(\d+)_([a-z]+)\.jpg$")
DOWNLOAD_TIMEOUT = 120


class PhotoDatasetError(RuntimeError):
    """The photo dataset is missing, incomplete or unreadable."""


@dataclass(frozen=True)
class PhotoPool:
    root: Path
    set_ids: tuple[int, ...]
    archive_sha256: str

    def path(self, set_id: int, room: str) -> Path:
        return self.root / f"{set_id}_{room}.jpg"


def download_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    return response.content


def _extract(data: bytes, root: Path) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise PhotoDatasetError(f"photo archive is not a readable zip: {exc}") from exc
    root.mkdir(parents=True, exist_ok=True)
    wanted = f"{PHOTO_DATASET_DIR}/"
    found = 0
    for member in archive.namelist():
        if not member.startswith(wanted) or member.endswith("/"):
            continue
        name = member[len(wanted) :]
        if not _PHOTO_NAME.match(name):
            continue
        (root / name).write_bytes(archive.read(member))
        found += 1
    if found == 0:
        raise PhotoDatasetError(f"archive contains no photos under {PHOTO_DATASET_DIR!r}")


def _scan(root: Path) -> tuple[int, ...]:
    rooms: dict[int, set[str]] = {}
    for path in root.glob("*.jpg"):
        match = _PHOTO_NAME.match(path.name)
        if match:
            rooms.setdefault(int(match.group(1)), set()).add(match.group(2))
    if not rooms:
        raise PhotoDatasetError(f"no photos found in {root}")
    incomplete = sorted(set_id for set_id, found in rooms.items() if not set(PHOTO_ROOMS) <= found)
    if incomplete:
        raise PhotoDatasetError(
            f"photo sets missing one of {PHOTO_ROOMS}: {incomplete[:10]} "
            f"({len(incomplete)} of {len(rooms)})"
        )
    return tuple(sorted(rooms))


def ensure_pool(
    config: CorpusConfig,
    data_dir: Path = DATA_DIR,
    url: str = PHOTO_DATASET_URL,
    fetch=download_bytes,
) -> PhotoPool:
    """Return the photo pool, downloading and extracting it only when it isn't there yet."""
    root = Path(data_dir) / "photos"
    marker = Path(data_dir) / "photos.sha256"
    if not root.is_dir() or not any(root.glob("*.jpg")):
        data = fetch(url)
        digest = hashlib.sha256(data).hexdigest()
        _extract(data, root)
        marker.write_text(digest, encoding="utf-8")
    digest = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    return PhotoPool(root=root, set_ids=_scan(root), archive_sha256=digest)


def make_variant(source: Path, target: Path, config: CorpusConfig) -> Path:
    """One edited photo: centre-crop, resize, re-save as JPEG (all three, in that order)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        width, height = image.size
        crop_w, crop_h = int(width * config.crop_fraction), int(height * config.crop_fraction)
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        edited = image.convert("RGB").crop((left, top, left + crop_w, top + crop_h))
        edited = edited.resize(
            (int(crop_w * config.resize_fraction), int(crop_h * config.resize_fraction)),
            Image.Resampling.LANCZOS,
        )
        edited.save(target, format="JPEG", quality=config.jpeg_quality)
    return target
