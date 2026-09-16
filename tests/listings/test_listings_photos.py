import io
import zipfile

import pytest
from PIL import Image

from listings.config import PHOTO_DATASET_DIR, PHOTO_ROOMS, CorpusConfig
from listings.photos import PhotoDatasetError, ensure_pool, make_variant

CONFIG = CorpusConfig()


def test_downloads_extracts_and_reports_the_archive_hash(tmp_path, photo_archive):
    data = photo_archive(range(1, 4))
    calls = []

    def fetch(url):
        calls.append(url)
        return data

    pool = ensure_pool(CONFIG, data_dir=tmp_path, url="https://example.invalid/x.zip", fetch=fetch)
    assert pool.set_ids == (1, 2, 3)
    assert len(pool.archive_sha256) == 64
    assert pool.path(2, "kitchen").is_file()
    assert calls == ["https://example.invalid/x.zip"]

    again = ensure_pool(CONFIG, data_dir=tmp_path, fetch=fetch)
    assert calls == ["https://example.invalid/x.zip"]  # not downloaded twice
    assert again.set_ids == pool.set_ids
    assert again.archive_sha256 == pool.archive_sha256


def test_a_set_missing_a_room_fails_loudly(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for room in PHOTO_ROOMS:
            archive.writestr(f"{PHOTO_DATASET_DIR}/1_{room}.jpg", b"x")
        archive.writestr(f"{PHOTO_DATASET_DIR}/2_bathroom.jpg", b"x")  # set 2 is incomplete
    with pytest.raises(PhotoDatasetError, match="2"):
        ensure_pool(CONFIG, data_dir=tmp_path, fetch=lambda _url: buffer.getvalue())


def test_a_corrupt_archive_fails_loudly(tmp_path):
    with pytest.raises(PhotoDatasetError, match="archive"):
        ensure_pool(CONFIG, data_dir=tmp_path, fetch=lambda _url: b"not a zip")


def test_an_empty_archive_fails_loudly(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Houses-dataset-master/README.md", b"nothing here")
    with pytest.raises(PhotoDatasetError, match="no photos"):
        ensure_pool(CONFIG, data_dir=tmp_path, fetch=lambda _url: buffer.getvalue())


def test_variant_is_cropped_resized_and_recompressed(tmp_path, photo_pool):
    source = photo_pool.path(1, "bedroom")
    target = tmp_path / "variants" / "1_bedroom__edited.jpg"
    result = make_variant(source, target, CONFIG)

    assert result == target and target.is_file()
    with Image.open(source) as before, Image.open(target) as after:
        # crop then resize, each rounded down — the same order the implementation applies
        assert after.width == int(int(before.width * CONFIG.crop_fraction) * CONFIG.resize_fraction)
        assert after.height == int(
            int(before.height * CONFIG.crop_fraction) * CONFIG.resize_fraction
        )
    assert target.read_bytes() != source.read_bytes()


def test_variant_generation_is_deterministic(tmp_path, photo_pool):
    source = photo_pool.path(3, "kitchen")
    first = make_variant(source, tmp_path / "a.jpg", CONFIG).read_bytes()
    second = make_variant(source, tmp_path / "b.jpg", CONFIG).read_bytes()
    assert first == second
