# Phase 4 — Duplicate and Fraud Listing Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a labelled synthetic listings corpus over real DLD sales, embed photos and text on the GPU, retrieve candidates through pgvector, and flag duplicates with several signals in agreement plus three fraud checks — with precision, recall and control false-positive rates measured against ground truth.

**Architecture:** A new top-level package `listings/`, one module per job, following the `ingestion/` and `models/price/` patterns: photos → generate → load (Postgres schema `listings`) → embed → candidates → features → detect → fraud → evaluate, driven by `python -m listings build|embed|detect|evaluate`. Embedding goes through an `Embedder` protocol so tests inject a deterministic fake and never download a model.

**Tech Stack:** Python 3.11, Polars, psycopg2, pgvector 0.8.6 (HNSW, cosine), sentence-transformers (CLIP ViT-B/32 + all-MiniLM-L6-v2) on CUDA, Pillow, scikit-learn, MLflow 2.17.2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-phase4-duplicate-fraud-design.md`

## Global Constraints

- Corpus: 20,000 listings = 17,000 base + 1,200 exact reposts + 900 reworded + 900 edited-photo clones. 600 `bait_price` listings. 300 exact reposts also get a ±15–30% asking-price change (for `inconsistent_relist`).
- Base listings are sampled from `dld.market_sales` with `instance_date >= 2021-01-01`, homes only: `property_type = 'villa'`, or `'unit'` with `property_sub_type IN ('Flat', 'Hotel Apartment', 'Stacked Townhouses')`. At least 1,500 come from buildings with ≥ 4 sales.
- Asking price = DLD sale price × uniform(1.00, 1.08), rounded to the nearest 10,000. Bait listings: × uniform(0.40, 0.65) of the sale price.
- Photos: `emanhamed/Houses-dataset`, 535 sets × 4 rooms (`<id>_<bathroom|bedroom|frontal|kitchen>.jpg`). 40 sets are "agency stock", reused by about 25% of listings across ≥ 5 areas. Images are NEVER committed; they live under `data/listings/` (gitignored) and the README cites Ahmed & Moustafa (2016).
- Edited-photo variants: crop to 85%, resize to 70%, re-save at JPEG quality 60.
- Embeddings: CLIP ViT-B/32 → 512 dims; all-MiniLM-L6-v2 → 384 dims; both L2-normalised. A listing's image vector is the normalised mean of its photo vectors.
- Postgres schema `listings`, separate from `dld`. HNSW indexes with `vector_cosine_ops`, m = 16, ef_construction = 64; queries set `hnsw.ef_search = 100`.
- Retrieval has three channels (see ruling 5): top 20 by text embedding, top 20 by the listing's average image embedding, and listings sharing an identical photo where that photo is used by at most `max_photo_fanout` (50) listings. Pairs are canonical (`listing_a < listing_b`) and never duplicated.
- The twelve pair features, in this exact order: `text_cosine, image_max_cosine, image_mean_cosine, shared_photo_count, abs_log_price_ratio, abs_log_size_ratio, same_area, same_building, same_project, bedrooms_equal, days_apart, same_agent`.
- **Labels are never features.** `dup_group_id`, `control_group_id` and `fraud_label` are read by exactly one module, `listings/truth.py`, which turns them into training targets and evaluation ground truth. `features.py`, `candidates.py`, `fraud.py` and `detect.py`'s inference path select explicit column lists that exclude them, and a test asserts no other module's SQL mentions them. Training a supervised model on labels is intended; letting a label become an input is not.
- Split by `posted_at`: first 60% train, next 20% threshold selection, last 20% reporting. A pair belongs to the split of its later listing.
- Model: `LogisticRegression(class_weight="balanced", max_iter=1000)` inside a `StandardScaler` pipeline. Threshold = the lowest score whose validation precision ≥ 0.98. Baseline: `image_max_cosine >= 0.95`.
- Fraud: `bait_price` when the asking price is > 10% below the Phase 3 champion's 80% lower bound (skip the flag with a warning if the model can't load); `photo_reuse` when a photo set spans ≥ 5 areas; `inconsistent_relist` when a duplicate cluster's asking prices differ by > 20%.
- `load_dotenv()` runs before any settings are read. `DbSettings.from_env()` raises without `POSTGRES_PORT`; the project DB is on 5433. NEVER connect to port 5432 (a native Windows Postgres). Never `docker compose down -v`.
- Windows/LightGBM hazard: `models/price/__init__.py` preloads the system msvcp140.dll and `tests/conftest.py` imports `models.price` first. Do not reorder those imports; do not add module-level `pandas`/`pyarrow`/`mlflow` imports to `tests/models/price/conftest.py`.
- Test layout: no `__init__.py` under `tests/`; Phase 4 test files are `tests/listings/test_listings_<topic>.py`; shared helpers are fixtures in `tests/listings/conftest.py`.
- Tests never download a model and never need a GPU: they inject the fake embedder. The one opt-in real-CLIP test is skipped unless the model is already cached.
- Every MLflow-touching test uses the `temp_mlflow` fixture; only the real Task 11 run writes to the server at 127.0.0.1:5000.
- Work directly on `master`. Stage specific files only; never `git add -A`. Never commit `.env`, `data/`, `mlruns/`.
- Commit trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Run from the repo root `C:\Users\cnaya\OneDrive\Desktop\dubimator` in Git Bash, always through `uv run`. At the end of every task run `uv run ruff format .`, then `uv run ruff check . && uv run ruff format --check .` must be clean.

## Controller rulings made while planning

1. **Text templates live with generation** (`text.py` beside `generate.py`, one task) — the templates exist only to feed the generator, and a reviewer judging one judges the other.
2. **Pair features, ground truth and the decision model are one task** — the feature order, the model's coefficients and the threshold are meaningless apart, and the model is fitted on exactly those columns against exactly that truth. `truth.py` is separate from `features.py` so that the one module allowed to read labels is obvious at a glance.
3. **`photos.py` downloads the dataset as a zip via `codeload.github.com`** rather than shelling out to `git clone`: no git dependency, resumable, and the SHA-256 of the archive is recorded in `corpus_runs.photo_dataset_sha`.
4. **The corpus is generated in Polars and loaded with COPY**, mirroring Phase 2, so 20,000 listings and ~100k photo rows load in one transaction.
6. **No new fixture file.** The integration test loads the existing `tests/fixtures/price_sample.csv` (2,937 real DLD rows, Phase 3) through Phase 2's `run_pipeline`, so the corpus is generated from real sales without adding another committed fixture.
7. **`detect` writes results; `evaluate` re-scores in memory.** Only flagged pairs are stored, so `evaluate` repeats the deterministic scoring step rather than persisting hundreds of thousands of scored pairs.
5. **The image retrieval channel matches listing image vectors, not individual photos** — a deviation from the spec's "top 10 nearest photos per photo, mapped back to their listings". A stock photo set shared by ~5,000 listings would map every photo match back to thousands of listings and generate millions of pairs. Instead: (a) nearest neighbours on the listing's average image vector, and (b) an exact-shared-photo channel capped at `max_photo_fanout` listings per photo, which skips exactly those stock photos. Text similarity still catches reposts that use stock photos, and retrieval recall is measured, so the choice is checked rather than assumed. This goes into the spec's amendments in Task 11.

## File map

| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml`, `uv.lock`, `listings/config.py` | dependencies, disk check, every constant | 1 |
| `listings/photos.py` | fetch and verify the photo dataset, build edited variants | 2 |
| `listings/text.py`, `listings/generate.py` | seeded description templates; corpus, planted patterns, controls, labels | 3 |
| `listings/sql/schema.sql`, `listings/load.py` | schema, COPY load, HNSW indexes, corpus_runs | 4 |
| `listings/embed.py` | `Embedder` protocol, CLIP + MiniLM, device resolution, fake for tests | 5 |
| `listings/candidates.py` | pgvector retrieval → canonical candidate pairs | 6 |
| `listings/features.py`, `listings/truth.py`, `listings/detect.py` | twelve pair features; the one label-reading module; split, model, threshold, baseline, `duplicate_pairs` | 7 |
| `listings/fraud.py` | `bait_price`, `photo_reuse`, `inconsistent_relist` | 8 |
| `listings/evaluate.py` | metrics, PR curve, control false-positive rates, timings, MLflow | 9 |
| `listings/__main__.py`, `tests/listings/test_listings_integration.py` | CLI and the end-to-end test over real DLD fixture rows (ruling 6) | 10 |
| real run, `README.md`, spec amendments | build/embed/detect/evaluate on 20k, documented results | 11 |

Tests live in `tests/listings/`; `tests/listings/conftest.py` is created in Task 2 and extended in Tasks 3, 5 and 6.

---

## Task 1: Dependencies, disk check, config module

**Files:**
- Create: `listings/__init__.py`, `listings/config.py`, `tests/listings/test_listings_config.py`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Produces (`listings.config`):
  - `PHOTO_DATASET_URL: str`, `PHOTO_ROOMS: tuple[str, ...]`, `HOME_UNIT_SUB_TYPES: tuple[str, ...]`
  - `PAIR_FEATURES: tuple[str, ...]` (the twelve, in order)
  - `LABEL_COLUMNS: tuple[str, ...]` = `("dup_group_id", "control_group_id", "fraud_label")`
  - `CorpusConfig` and `DetectConfig`, frozen dataclasses with the fields below

- [ ] **Step 1: Check free disk space before installing**

```bash
df -h /c | tail -1
```

Expected: at least 8 GB free in the `Avail` column (the CUDA torch wheel is ~2.5 GB, models ~600 MB, photos ~176 MB, plus room for the corpus). If there is less, STOP and report BLOCKED with the output — do not install.

- [ ] **Step 2: Add dependencies**

```bash
uv add sentence-transformers pillow
uv sync
uv run python -c "import torch, sentence_transformers, PIL; print(torch.__version__, torch.cuda.is_available(), sentence_transformers.__version__, PIL.__version__)"
```

Expected: a torch version, then `True` (CUDA visible on this RTX 3060), then the two library versions.

If `torch.cuda.is_available()` prints `False`, the default PyPI wheel is CPU-only. Report DONE_WITH_CONCERNS with the torch version string — the controller decides whether to install the CUDA wheel from the PyTorch index. Do not add an index URL on your own.

- [ ] **Step 3: Write the failing config test**

`tests/listings/test_listings_config.py`:

```python
from listings.config import (
    LABEL_COLUMNS,
    PAIR_FEATURES,
    PHOTO_ROOMS,
    CorpusConfig,
    DetectConfig,
)


def test_pair_features_are_exact_and_ordered():
    assert PAIR_FEATURES == (
        "text_cosine", "image_max_cosine", "image_mean_cosine", "shared_photo_count",
        "abs_log_price_ratio", "abs_log_size_ratio", "same_area", "same_building",
        "same_project", "bedrooms_equal", "days_apart", "same_agent",
    )  # fmt: skip


def test_label_columns_are_named_so_detection_can_exclude_them():
    assert LABEL_COLUMNS == ("dup_group_id", "control_group_id", "fraud_label")
    assert not set(LABEL_COLUMNS) & set(PAIR_FEATURES)


def test_corpus_counts_add_up_to_the_requested_total():
    config = CorpusConfig()
    assert config.n_listings == 20_000
    assert config.n_base == 17_000
    assert (config.n_exact_repost, config.n_reworded, config.n_edited_photo) == (1_200, 900, 900)
    assert config.n_base + config.n_exact_repost + config.n_reworded + config.n_edited_photo == (
        config.n_listings
    )
    assert config.n_bait_price == 600
    assert config.n_price_shifted_reposts == 300
    assert config.n_price_shifted_reposts <= config.n_exact_repost


def test_corpus_photo_and_sampling_settings():
    config = CorpusConfig()
    assert PHOTO_ROOMS == ("bathroom", "bedroom", "frontal", "kitchen")
    assert config.n_stock_sets == 40
    assert config.stock_share == 0.25
    assert config.stock_min_areas == 5
    assert config.min_building_sales == 4
    assert config.n_from_busy_buildings == 1_500
    assert config.asking_factor == (1.00, 1.08)
    assert config.bait_factor == (0.40, 0.65)
    assert config.seed == 42


def test_detect_settings_match_the_spec():
    config = DetectConfig()
    assert (config.text_top_k, config.photo_top_k, config.ef_search) == (20, 20, 100)
    assert config.max_photo_fanout == 50
    assert (config.train_share, config.threshold_share) == (0.60, 0.20)
    assert config.target_precision == 0.98
    assert config.baseline_image_cosine == 0.95
    assert config.photo_reuse_min_areas == 5
    assert config.relist_price_spread == 0.20
    assert config.bait_margin == 0.10
    assert config.experiment == "listing-dedup"
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/listings/test_listings_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'listings'`.

- [ ] **Step 5: Implement the package and config**

`listings/__init__.py`:

```python
"""Synthetic listings corpus, duplicate detection and fraud flags (Phase 4)."""
```

`listings/config.py`:

```python
"""Constants and tunables for the listings corpus and duplicate detection.

Spec: docs/superpowers/specs/2026-09-16-phase4-duplicate-fraud-design.md
"""

from dataclasses import dataclass

PHOTO_DATASET_URL = "https://codeload.github.com/emanhamed/Houses-dataset/zip/refs/heads/master"
PHOTO_DATASET_DIR = "Houses-dataset-master/Houses Dataset"
PHOTO_ROOMS = ("bathroom", "bedroom", "frontal", "kitchen")
HOME_UNIT_SUB_TYPES = ("Flat", "Hotel Apartment", "Stacked Townhouses")

# Pair features, in the order the model sees them.
PAIR_FEATURES = (
    "text_cosine", "image_max_cosine", "image_mean_cosine", "shared_photo_count",
    "abs_log_price_ratio", "abs_log_size_ratio", "same_area", "same_building",
    "same_project", "bedrooms_equal", "days_apart", "same_agent",
)  # fmt: skip

# Ground truth. Only evaluate.py may read these.
LABEL_COLUMNS = ("dup_group_id", "control_group_id", "fraud_label")


@dataclass(frozen=True)
class CorpusConfig:
    n_listings: int = 20_000
    n_base: int = 17_000
    n_exact_repost: int = 1_200
    n_reworded: int = 900
    n_edited_photo: int = 900
    n_bait_price: int = 600
    n_price_shifted_reposts: int = 300
    n_stock_sets: int = 40
    stock_share: float = 0.25
    stock_min_areas: int = 5
    min_building_sales: int = 4
    n_from_busy_buildings: int = 1_500
    asking_factor: tuple[float, float] = (1.00, 1.08)
    bait_factor: tuple[float, float] = (0.40, 0.65)
    price_shift: tuple[float, float] = (0.15, 0.30)
    repost_days: tuple[int, int] = (1, 30)
    n_agents: int = 400
    posted_from: str = "2023-01-01"
    posted_to: str = "2023-06-30"
    sales_from: str = "2021-01-01"
    crop_fraction: float = 0.85
    resize_fraction: float = 0.70
    jpeg_quality: int = 60
    seed: int = 42


@dataclass(frozen=True)
class DetectConfig:
    text_top_k: int = 20
    photo_top_k: int = 20  # nearest listings by average image vector
    ef_search: int = 100
    train_share: float = 0.60
    threshold_share: float = 0.20
    target_precision: float = 0.98
    baseline_image_cosine: float = 0.95
    max_photo_fanout: int = 50
    photo_reuse_min_areas: int = 5
    relist_price_spread: float = 0.20
    bait_margin: float = 0.10
    price_model_uri: str = "models:/dubimator-price@champion"
    experiment: str = "listing-dedup"
    seed: int = 42
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/listings/test_listings_config.py -v`
Expected: PASS.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add pyproject.toml uv.lock listings/__init__.py listings/config.py tests/listings/test_listings_config.py
git commit -m "$(cat <<'EOF'
feat(listings): add embedding dependencies and Phase 4 configuration

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---
## Task 2: Photo pool — fetch, verify, edited variants

**Files:**
- Create: `listings/photos.py`, `tests/listings/conftest.py`, `tests/listings/test_listings_photos.py`

**Interfaces:**
- Consumes: `listings.config` (`PHOTO_DATASET_URL`, `PHOTO_DATASET_DIR`, `PHOTO_ROOMS`, `CorpusConfig`)
- Produces (`listings.photos`):
  - `DATA_DIR = Path("data/listings")`, `PHOTO_DIR = DATA_DIR / "photos"`, `VARIANT_DIR = DATA_DIR / "variants"`
  - `PhotoDatasetError(RuntimeError)`
  - `PhotoPool`, a frozen dataclass: `root: Path`, `set_ids: tuple[int, ...]`, `archive_sha256: str`, and `.path(set_id: int, room: str) -> Path`
  - `ensure_pool(config, data_dir=DATA_DIR, url=PHOTO_DATASET_URL, fetch=download_bytes) -> PhotoPool` — downloads and extracts only when the photos aren't already there
  - `make_variant(source: Path, target: Path, config) -> Path` — one "edited" photo: centre-crop to `crop_fraction`, resize to `resize_fraction`, save as JPEG at `jpeg_quality`
  - `download_bytes(url: str) -> bytes`
- Fixtures in `tests/listings/conftest.py`: `photo_archive` (builds a zip of a miniature dataset) and `photo_pool` (an extracted pool on disk)

A variant applies all three edits in sequence, so one edited photo per source photo: that is what a lightly re-processed repost looks like.

- [ ] **Step 1: Write the shared fixtures**

`tests/listings/conftest.py`:

```python
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
    from listings.photos import ensure_pool
    from listings.config import CorpusConfig

    data = build_archive(range(1, 7))
    return ensure_pool(CorpusConfig(), data_dir=tmp_path, fetch=lambda _url: data)
```

- [ ] **Step 2: Write the failing tests**

`tests/listings/test_listings_photos.py`:

```python
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
        assert after.height == int(int(before.height * CONFIG.crop_fraction) * CONFIG.resize_fraction)
    assert target.read_bytes() != source.read_bytes()


def test_variant_generation_is_deterministic(tmp_path, photo_pool):
    source = photo_pool.path(3, "kitchen")
    first = make_variant(source, tmp_path / "a.jpg", CONFIG).read_bytes()
    second = make_variant(source, tmp_path / "b.jpg", CONFIG).read_bytes()
    assert first == second
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/listings/test_listings_photos.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'listings.photos'`.

- [ ] **Step 4: Implement `photos.py`**

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/listings/ -v`
Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add listings/photos.py tests/listings/conftest.py tests/listings/test_listings_photos.py
git commit -m "$(cat <<'EOF'
feat(listings): fetch and verify the photo pool, build edited variants

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Listing text and corpus generation

**Files:**
- Create: `listings/text.py`, `listings/generate.py`, `tests/listings/test_listings_text.py`, `tests/listings/test_listings_generate.py`
- Modify: `tests/listings/conftest.py` (add the `sales_frame`, `areas_frame` and `small_corpus_config` fixtures)

**Interfaces:**
- Consumes: `listings.config` (`CorpusConfig`, `HOME_UNIT_SUB_TYPES`, `PHOTO_ROOMS`), `listings.photos` (`PhotoPool`, `make_variant`, `VARIANT_DIR`), `ingestion.config.DbSettings`
- Produces (`listings.text`):
  - `ListingFacts`, a frozen dataclass: `bedrooms: int | None, size_sqm: float, area_name: str, area_name_ar: str | None, building_name: str | None, project_name: str | None, property_type: str, sub_kind: str`
  - `make_title(rng: random.Random, facts: ListingFacts) -> str`
  - `make_description(rng: random.Random, facts: ListingFacts) -> str`
- Produces (`listings.generate`):
  - `LISTING_SCHEMA`, `PHOTO_SCHEMA`, `LISTING_PHOTO_SCHEMA` (polars dtypes)
  - `Corpus`, a frozen dataclass: `listings: pl.DataFrame`, `photos: pl.DataFrame`, `listing_photos: pl.DataFrame`, `counts: dict[str, int]`
  - `load_sales(settings, config) -> pl.DataFrame` and `load_areas(settings) -> pl.DataFrame`
  - `generate_corpus(sales, areas, pool_set_ids, config) -> Corpus` — pure, deterministic under `config.seed`
  - `render_variants(corpus, pool, config, variant_dir=VARIANT_DIR) -> int`
- Fixtures added to `tests/listings/conftest.py`: `sales_frame`, `areas_frame`, `small_corpus_config`

**Ground truth that is derived, not stored:** which pattern a clone is (exact
repost, reworded, edited photos) is worked out at evaluation time by comparing
a clone with its source — same text and same photos, different text, or variant
photos. The corpus therefore stores no `clone_kind` column, and the detector
cannot read one.

**Never stored:** the DLD sale price. Bait pricing must be caught by the Phase 3
model, not by comparing against a stored answer.

- [ ] **Step 1: Add the fixtures**

Append to `tests/listings/conftest.py` (merge imports into the existing block):

```python
from datetime import date

import polars as pl

from listings.config import CorpusConfig
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
```

- [ ] **Step 2: Write the failing text tests**

`tests/listings/test_listings_text.py`:

```python
import random

from listings.text import ListingFacts, make_description, make_title

FLAT = ListingFacts(
    bedrooms=2,
    size_sqm=96.5,
    area_name="Marsa Dubai",
    area_name_ar="مرسى دبي",
    building_name="Marina Gate 1",
    project_name="Marina Gate",
    property_type="unit",
    sub_kind="flat",
)
VILLA = ListingFacts(
    bedrooms=None,
    size_sqm=520.0,
    area_name="Hadaeq Sheikh Mohammed Bin Rashid",
    area_name_ar=None,
    building_name=None,
    project_name="Dubai Hills",
    property_type="villa",
    sub_kind="villa",
)


def test_text_is_deterministic_for_a_seed():
    first = make_description(random.Random(7), FLAT)
    second = make_description(random.Random(7), FLAT)
    assert first == second
    assert make_description(random.Random(8), FLAT) != first


def test_description_states_the_facts():
    text = make_description(random.Random(1), FLAT)
    assert "2" in text and "Marsa Dubai" in text
    assert "97 sqm" in text or "96" in text
    assert len(text) > 80


def test_title_mentions_kind_and_location():
    title = make_title(random.Random(2), FLAT)
    assert "Marsa Dubai" in title or "Marina Gate 1" in title
    assert "2" in title
    assert len(title) <= 120


def test_villa_without_bedrooms_or_building_still_reads_correctly():
    title, text = make_title(random.Random(3), VILLA), make_description(random.Random(3), VILLA)
    assert "None" not in title and "None" not in text
    assert "villa" in title.lower() or "villa" in text.lower()


def test_arabic_area_names_appear_in_some_descriptions():
    texts = [make_description(random.Random(seed), FLAT) for seed in range(40)]
    assert any("مرسى دبي" in text for text in texts)
    assert all(text.strip() for text in texts)
```

- [ ] **Step 3: Implement `listings/text.py`**

```python
"""Seeded listing titles and descriptions. No text is copied from any real listing."""

import random
from dataclasses import dataclass

OPENERS = (
    "Available now",
    "Just listed",
    "Exclusive listing",
    "Priced to sell",
    "New to the market",
    "Rare opportunity",
)
AMENITIES = (
    "covered parking", "shared pool", "fully fitted kitchen", "built-in wardrobes",
    "balcony", "gym access", "24/7 security", "children's play area", "maid's room",
    "sea view", "community view", "central air conditioning",
)  # fmt: skip
BLURBS = (
    "Managed by an experienced local agent.",
    "Viewings available seven days a week.",
    "Handover documents ready for a quick transfer.",
    "Tenanted until the end of the quarter; investor friendly.",
    "Vacant on transfer.",
)
CALLS = (
    "Call today to arrange a viewing.",
    "Message us for the full photo set.",
    "Contact the listing agent for floor plans.",
    "Book a viewing this week.",
)
KIND_WORDS = {
    "flat": ("apartment", "flat"),
    "hotel_apartment": ("hotel apartment", "serviced apartment"),
    "townhouse": ("townhouse", "stacked townhouse"),
    "villa": ("villa", "family villa"),
}


@dataclass(frozen=True)
class ListingFacts:
    bedrooms: int | None
    size_sqm: float
    area_name: str
    area_name_ar: str | None
    building_name: str | None
    project_name: str | None
    property_type: str
    sub_kind: str


def _bedroom_phrase(bedrooms: int | None) -> str:
    if bedrooms is None:
        return ""
    if bedrooms == 0:
        return "Studio "
    return f"{bedrooms}-bedroom "


def _where(facts: ListingFacts) -> str:
    return facts.building_name or facts.project_name or facts.area_name


def make_title(rng: random.Random, facts: ListingFacts) -> str:
    kind = rng.choice(KIND_WORDS[facts.sub_kind])
    return f"{_bedroom_phrase(facts.bedrooms)}{kind} in {_where(facts)}, {facts.area_name}".strip()


def make_description(rng: random.Random, facts: ListingFacts) -> str:
    kind = rng.choice(KIND_WORDS[facts.sub_kind])
    area = facts.area_name
    if facts.area_name_ar and rng.random() < 0.25:
        area = f"{facts.area_name} ({facts.area_name_ar})"
    sentences = [
        f"{rng.choice(OPENERS)}: {_bedroom_phrase(facts.bedrooms).lower()}{kind} in {area}.",
        f"The property offers {facts.size_sqm:.0f} sqm of space"
        + (f" in {facts.building_name}." if facts.building_name else "."),
    ]
    amenities = rng.sample(AMENITIES, rng.randint(2, 5))
    sentences.append("Features include " + ", ".join(amenities) + ".")
    if facts.project_name and rng.random() < 0.6:
        sentences.append(f"Part of the {facts.project_name} development.")
    sentences.append(rng.choice(BLURBS))
    sentences.append(rng.choice(CALLS))
    return " ".join(sentences)
```

- [ ] **Step 4: Run the text tests**

Run: `uv run pytest tests/listings/test_listings_text.py -v`
Expected: PASS (they failed at Step 2 with `ModuleNotFoundError: No module named 'listings.text'`).

- [ ] **Step 5: Write the failing generation tests**

`tests/listings/test_listings_generate.py`:

```python
import polars as pl
import pytest
from PIL import Image

from listings.generate import generate_corpus, render_variants

POOL = tuple(range(1, 21))


def build(sales_frame, areas_frame, config):
    return generate_corpus(sales_frame(), areas_frame, POOL, config)


def test_corpus_is_deterministic(sales_frame, areas_frame, small_corpus_config):
    first = build(sales_frame, areas_frame, small_corpus_config)
    second = build(sales_frame, areas_frame, small_corpus_config)
    assert first.listings.equals(second.listings)
    assert first.listing_photos.equals(second.listing_photos)
    assert first.photos.equals(second.photos)


def test_counts_match_the_config(sales_frame, areas_frame, small_corpus_config):
    config = small_corpus_config
    corpus = build(sales_frame, areas_frame, config)
    assert corpus.listings.height == config.n_listings
    assert corpus.counts["base"] == config.n_base
    assert corpus.counts["exact_repost"] == config.n_exact_repost
    assert corpus.counts["reworded"] == config.n_reworded
    assert corpus.counts["edited_photo"] == config.n_edited_photo
    assert corpus.counts["bait_price"] == config.n_bait_price
    assert corpus.listings["listing_id"].n_unique() == config.n_listings
    assert corpus.listings["is_synthetic"].all()


def test_every_listing_has_four_photos_in_order(sales_frame, areas_frame, small_corpus_config):
    corpus = build(sales_frame, areas_frame, small_corpus_config)
    per_listing = corpus.listing_photos.group_by("listing_id").agg(
        pl.col("position").sort().alias("positions")
    )
    assert per_listing.height == corpus.listings.height
    assert per_listing["positions"].to_list() == [[0, 1, 2, 3]] * per_listing.height
    known = set(corpus.photos["photo_id"].to_list())
    assert set(corpus.listing_photos["photo_id"].to_list()) <= known


def clones_with_sources(corpus):
    listings = corpus.listings
    clones = listings.filter(pl.col("dup_group_id").is_not_null())
    sources = listings.rename({c: f"src_{c}" for c in listings.columns})
    return clones.join(sources, left_on="dup_group_id", right_on="src_listing_id", how="inner")


def photo_ids(corpus, listing_id):
    rows = corpus.listing_photos.filter(pl.col("listing_id") == listing_id).sort("position")
    return rows["photo_id"].to_list()


def test_clones_point_at_their_source_and_post_later(sales_frame, areas_frame, small_corpus_config):
    config = small_corpus_config
    corpus = build(sales_frame, areas_frame, config)
    joined = clones_with_sources(corpus)
    assert joined.height == config.n_exact_repost + config.n_reworded + config.n_edited_photo
    assert (joined["posted_at"] > joined["src_posted_at"]).all()
    assert (joined["agent_id"] != joined["src_agent_id"]).all()
    assert (joined["source_transaction_id"] == joined["src_source_transaction_id"]).all()
    assert joined["dup_group_id"].is_in(corpus.listings["listing_id"]).all()


def test_the_three_clone_patterns_are_present_and_distinguishable(
    sales_frame, areas_frame, small_corpus_config
):
    config = small_corpus_config
    corpus = build(sales_frame, areas_frame, config)
    variants = dict(zip(corpus.photos["photo_id"].to_list(), corpus.photos["variant_of"].to_list()))
    exact = reworded = edited = 0
    for clone in clones_with_sources(corpus).iter_rows(named=True):
        mine = photo_ids(corpus, clone["listing_id"])
        theirs = photo_ids(corpus, clone["dup_group_id"])
        if all(variants[p] is not None for p in mine):
            assert [variants[p] for p in mine] == theirs  # variants of the source's photos
            assert clone["description"] == clone["src_description"]
            edited += 1
        elif mine == theirs and clone["description"] == clone["src_description"]:
            exact += 1
        else:
            assert mine == theirs and clone["description"] != clone["src_description"]
            reworded += 1
    assert (exact, reworded, edited) == (
        config.n_exact_repost, config.n_reworded, config.n_edited_photo,
    )  # fmt: skip


def test_some_reposts_shift_the_price_and_the_rest_keep_it(
    sales_frame, areas_frame, small_corpus_config
):
    config = small_corpus_config
    joined = clones_with_sources(build(sales_frame, areas_frame, config))
    same = joined.filter(pl.col("asking_price_aed") == pl.col("src_asking_price_aed"))
    shifted = joined.filter(pl.col("asking_price_aed") != pl.col("src_asking_price_aed"))
    assert shifted.height == config.n_price_shifted_reposts
    assert same.height == joined.height - config.n_price_shifted_reposts
    ratio = (shifted["asking_price_aed"] / shifted["src_asking_price_aed"]).to_list()
    assert all(0.70 <= value <= 1.30 for value in ratio)
    assert all(abs(value - 1.0) >= 0.10 for value in ratio)


def test_same_building_controls_are_labelled_in_groups(
    sales_frame, areas_frame, small_corpus_config
):
    config = small_corpus_config
    listings = build(sales_frame, areas_frame, config).listings
    controls = listings.filter(pl.col("control_group_id").is_not_null())
    assert controls.height >= config.n_from_busy_buildings
    groups = controls.group_by("control_group_id").agg(
        pl.col("building_name").n_unique().alias("buildings"),
        pl.col("source_transaction_id").n_unique().alias("sales"),
        pl.len().alias("members"),
    )
    assert (groups["members"] >= 2).all()
    assert (groups["buildings"] == 1).all()
    assert (groups["sales"] == groups["members"]).all()  # different units, not the same sale


def test_stock_photo_sets_span_several_areas(sales_frame, areas_frame, small_corpus_config):
    config = small_corpus_config
    corpus = build(sales_frame, areas_frame, config)
    stock_sets = set(corpus.photos.filter(pl.col("is_stock"))["set_id"].to_list())
    assert len(stock_sets) == config.n_stock_sets
    spread = (
        corpus.listings.filter(pl.col("photo_set_id").is_in(list(stock_sets)))
        .group_by("photo_set_id")
        .agg(pl.col("area_id").n_unique().alias("areas"))
    )
    assert (spread["areas"] >= config.stock_min_areas).all()


def test_bait_listings_are_cheap_and_labelled(sales_frame, areas_frame, small_corpus_config):
    config = small_corpus_config
    listings = build(sales_frame, areas_frame, config).listings
    bait = listings.filter(pl.col("fraud_label") == "bait_price")
    assert bait.height == config.n_bait_price
    assert bait["dup_group_id"].is_null().all()  # bait sits on base listings
    normal = listings.filter(pl.col("fraud_label").is_null())
    assert bait["asking_price_aed"].median() < normal["asking_price_aed"].median()


def test_asking_prices_are_rounded(sales_frame, areas_frame, small_corpus_config):
    listings = build(sales_frame, areas_frame, small_corpus_config).listings
    assert (listings["asking_price_aed"] % 10_000 == 0).all()
    assert (listings["asking_price_aed"] > 0).all()


def test_generation_fails_loudly_when_there_are_too_few_sales(
    sales_frame, areas_frame, small_corpus_config
):
    with pytest.raises(ValueError, match="not enough sales"):
        generate_corpus(sales_frame(n=100), areas_frame, POOL, small_corpus_config)


def test_render_variants_writes_one_file_per_variant_row(
    sales_frame, areas_frame, small_corpus_config, photo_pool, tmp_path
):
    config = small_corpus_config
    corpus = generate_corpus(sales_frame(), areas_frame, photo_pool.set_ids, config)
    written = render_variants(corpus, photo_pool, config, tmp_path / "variants")
    assert written == config.n_edited_photo * 4
    for path in corpus.photos.filter(pl.col("variant_kind") == "edited")["path"].to_list():
        target = tmp_path / "variants" / path.split("/")[-1]
        assert target.is_file()
        with Image.open(target) as image:
            assert image.width > 0
```

- [ ] **Step 6: Run the generation tests to verify they fail**

Run: `uv run pytest tests/listings/test_listings_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'listings.generate'`.

- [ ] **Step 7: Implement `listings/generate.py`**

Two details worth knowing before you read the code:
- Base listings are posted in `[posted_from, posted_to − max repost days]`, so a
  clone's later date always lands inside the window without clipping.
- Bait pricing multiplies the listing's own asking price (which is itself the
  sale price × 1.00–1.08), not a stored sale price. The corpus deliberately
  keeps no sale price, so nothing downstream can cheat by comparing to it.

```python
"""Generate the synthetic listings corpus from real DLD sales.

Real: area, building, project, size, bedrooms, price level, from dld.market_sales.
Invented: text, agents, posting dates, planted duplicates, fraud cases.
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from ingestion.config import DbSettings
from listings.config import HOME_UNIT_SUB_TYPES, PHOTO_ROOMS, CorpusConfig
from listings.photos import VARIANT_DIR, PhotoPool, make_variant
from listings.text import ListingFacts, make_description, make_title

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
SALES_SQL = f"""
SELECT {", ".join(SALES_SCHEMA)}
FROM dld.market_sales
WHERE instance_date >= %(sales_from)s
  AND (property_type = 'villa'
       OR (property_type = 'unit' AND property_sub_type IN %(unit_sub_types)s))
  AND price_aed > 0 AND area_sqm > 0 AND area_id IS NOT NULL
"""
AREAS_SQL = "SELECT area_id, name_en, name_ar FROM dld.areas ORDER BY area_id"
AREAS_SCHEMA = {"area_id": pl.Int64, "name_en": pl.Utf8, "name_ar": pl.Utf8}

LISTING_SCHEMA = {
    "listing_id": pl.Int64,
    "source_transaction_id": pl.Utf8,
    "agent_id": pl.Int64,
    "posted_at": pl.Date,
    "title": pl.Utf8,
    "description": pl.Utf8,
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
    "dup_group_id": pl.Int64,
    "control_group_id": pl.Utf8,
    "fraud_label": pl.Utf8,
    "is_synthetic": pl.Boolean,
}
PHOTO_SCHEMA = {
    "photo_id": pl.Int64,
    "set_id": pl.Int64,
    "room": pl.Utf8,
    "path": pl.Utf8,
    "variant_of": pl.Int64,
    "variant_kind": pl.Utf8,
    "is_stock": pl.Boolean,
}
LISTING_PHOTO_SCHEMA = {"listing_id": pl.Int64, "photo_id": pl.Int64, "position": pl.Int64}
SUB_KINDS = {"Flat": "flat", "Hotel Apartment": "hotel_apartment", "Stacked Townhouses": "townhouse"}
ROOM_POSITION = {room: index for index, room in enumerate(PHOTO_ROOMS)}


@dataclass(frozen=True)
class Corpus:
    listings: pl.DataFrame
    photos: pl.DataFrame
    listing_photos: pl.DataFrame
    counts: dict[str, int]


def load_sales(settings: DbSettings, config: CorpusConfig) -> pl.DataFrame:
    params = {"sales_from": config.sales_from, "unit_sub_types": HOME_UNIT_SUB_TYPES}
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(SALES_SQL, params)
            return pl.DataFrame(cur.fetchall(), schema=SALES_SCHEMA, orient="row")
    finally:
        conn.close()


def load_areas(settings: DbSettings) -> pl.DataFrame:
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(AREAS_SQL)
            return pl.DataFrame(cur.fetchall(), schema=AREAS_SCHEMA, orient="row")
    finally:
        conn.close()


def _sub_kind(property_type: str, sub_type: str | None) -> str:
    if property_type == "villa":
        return "villa"
    kind = SUB_KINDS.get(sub_type or "")
    if kind is None:
        raise ValueError(f"unexpected unit sub-type {sub_type!r}")
    return kind


def _sample_base_sales(sales: pl.DataFrame, config: CorpusConfig) -> pl.DataFrame:
    """Sales for the base listings: whole groups from busy buildings, then the rest."""
    key = ["area_id", "building_name"]
    per_building = 3
    with_building = sales.filter(pl.col("building_name").is_not_null())
    busy = (
        with_building.group_by(key)
        .len()
        .filter(pl.col("len") >= config.min_building_sales)
        .select(key)
    )
    ranked = (
        with_building.join(busy, on=key, how="inner")
        .with_columns(pl.col("transaction_id").hash(seed=config.seed).alias("__h"))
        .sort([*key, "__h"])
        .with_columns(pl.int_range(pl.len()).over(key).alias("__rank"))
        .filter(pl.col("__rank") < per_building)
        .drop("__h", "__rank")
    )
    sizes = ranked.group_by(key, maintain_order=True).len().with_columns(
        pl.col("len").cum_sum().alias("__cum")
    )
    wanted = sizes.filter(pl.col("__cum") - pl.col("len") < config.n_from_busy_buildings).select(key)
    busy_rows = ranked.join(wanted, on=key, how="inner").with_columns(
        pl.concat_str(
            [pl.lit("bldg:"), pl.col("area_id").cast(pl.Utf8), pl.lit(":"), pl.col("building_name")]
        ).alias("control_group_id")
    )
    remaining = config.n_base - busy_rows.height
    rest = sales.join(busy_rows.select("transaction_id"), on="transaction_id", how="anti")
    if remaining < 0:
        raise ValueError(
            f"busy-building sample is {busy_rows.height} rows, more than n_base={config.n_base}"
        )
    if rest.height < remaining:
        raise ValueError(
            f"not enough sales: need {remaining} more for n_base={config.n_base}, have {rest.height}"
        )
    others = rest.sample(n=remaining, seed=config.seed, shuffle=True).with_columns(
        pl.lit(None, pl.Utf8).alias("control_group_id")
    )
    columns = [*sales.columns, "control_group_id"]
    return pl.concat([busy_rows.select(columns), others.select(columns)]).sample(
        fraction=1.0, shuffle=True, seed=config.seed + 1
    )


def _base_photo_rows(set_ids, stock_sets: set[int]) -> pl.DataFrame:
    rows = []
    for set_id in sorted(set_ids):
        for room in PHOTO_ROOMS:
            rows.append(
                {
                    "photo_id": len(rows) + 1,
                    "set_id": set_id,
                    "room": room,
                    "path": f"photos/{set_id}_{room}.jpg",
                    "variant_of": None,
                    "variant_kind": None,
                    "is_stock": set_id in stock_sets,
                }
            )
    return pl.DataFrame(rows, schema=PHOTO_SCHEMA)


def _build_base_listings(
    base: pl.DataFrame, areas: pl.DataFrame, config: CorpusConfig, stock: list[int], plain: list[int]
) -> pl.DataFrame:
    rng = random.Random(config.seed)
    arabic = dict(zip(areas["area_id"].to_list(), areas["name_ar"].to_list()))
    start = date.fromisoformat(config.posted_from)
    span = (date.fromisoformat(config.posted_to) - start).days - config.repost_days[1]
    if span <= 0:
        raise ValueError("posting window is shorter than the repost delay")
    rows = []
    for index, sale in enumerate(base.iter_rows(named=True)):
        listing_id = index + 1
        photo_set = rng.choice(stock if rng.random() < config.stock_share else plain)
        asking = round(sale["price_aed"] * rng.uniform(*config.asking_factor) / 10_000) * 10_000
        facts = ListingFacts(
            bedrooms=sale["bedrooms"],
            size_sqm=sale["area_sqm"],
            area_name=sale["area_name"],
            area_name_ar=arabic.get(sale["area_id"]),
            building_name=sale["building_name"],
            project_name=sale["project_name"],
            property_type=sale["property_type"],
            sub_kind=_sub_kind(sale["property_type"], sale["property_sub_type"]),
        )
        text_rng = random.Random(config.seed * 7919 + listing_id)
        rows.append(
            {
                "listing_id": listing_id,
                "source_transaction_id": sale["transaction_id"],
                "agent_id": rng.randrange(1, config.n_agents + 1),
                "posted_at": start + timedelta(days=rng.randint(0, span)),
                "title": make_title(text_rng, facts),
                "description": make_description(text_rng, facts),
                "asking_price_aed": float(asking),
                "area_id": sale["area_id"],
                "area_name": sale["area_name"],
                "building_name": sale["building_name"],
                "project_name": sale["project_name"],
                "property_type": sale["property_type"],
                "property_sub_type": sale["property_sub_type"],
                "reg_type": sale["reg_type"],
                "size_sqm": sale["area_sqm"],
                "bedrooms": sale["bedrooms"],
                "photo_set_id": photo_set,
                "dup_group_id": None,
                "control_group_id": sale["control_group_id"],
                "fraud_label": None,
                "is_synthetic": True,
            }
        )
    return pl.DataFrame(rows, schema=LISTING_SCHEMA)


def _apply_bait_prices(listings: pl.DataFrame, config: CorpusConfig) -> pl.DataFrame:
    rng = random.Random(config.seed + 17)
    chosen = rng.sample(listings["listing_id"].to_list(), config.n_bait_price)
    factors = {listing_id: rng.uniform(*config.bait_factor) for listing_id in chosen}
    asking = []
    labels = []
    for row in listings.iter_rows(named=True):
        factor = factors.get(row["listing_id"])
        if factor is None:
            asking.append(row["asking_price_aed"])
            labels.append(row["fraud_label"])
        else:
            asking.append(float(round(row["asking_price_aed"] * factor / 10_000) * 10_000))
            labels.append("bait_price")
    return listings.with_columns(
        pl.Series("asking_price_aed", asking, dtype=pl.Float64),
        pl.Series("fraud_label", labels, dtype=pl.Utf8),
    )


def _base_listing_photos(listings: pl.DataFrame, photos: pl.DataFrame) -> pl.DataFrame:
    base = (
        photos.filter(pl.col("variant_of").is_null())
        .select(pl.col("set_id").alias("photo_set_id"), "photo_id", "room")
        .with_columns(
            pl.col("room")
            .replace_strict(ROOM_POSITION, return_dtype=pl.Int64)
            .alias("position")
        )
    )
    return (
        listings.select("listing_id", "photo_set_id")
        .join(base, on="photo_set_id", how="inner")
        .select("listing_id", "photo_id", "position")
        .sort(["listing_id", "position"])
        .cast(LISTING_PHOTO_SCHEMA)
    )
```

Continue `listings/generate.py` with the clone planting, validation and the two
public entry points:

```python
def _plant_clones(
    listings: pl.DataFrame,
    photos: pl.DataFrame,
    listing_photos: pl.DataFrame,
    arabic: dict[int, str | None],
    config: CorpusConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Clone base listings into exact reposts, reworded copies and edited-photo copies."""
    rng = random.Random(config.seed + 99)
    photos_by_listing: dict[int, list[int]] = {}
    for row in listing_photos.sort(["listing_id", "position"]).iter_rows(named=True):
        photos_by_listing.setdefault(row["listing_id"], []).append(row["photo_id"])
    photo_meta = {row["photo_id"]: row for row in photos.iter_rows(named=True)}
    by_id = {row["listing_id"]: row for row in listings.iter_rows(named=True)}

    kinds = (
        ["exact"] * config.n_exact_repost
        + ["reworded"] * config.n_reworded
        + ["edited"] * config.n_edited_photo
    )
    sources = rng.sample(sorted(by_id), len(kinds))
    shifted = set(rng.sample(range(config.n_exact_repost), config.n_price_shifted_reposts))
    next_listing_id = max(by_id) + 1
    next_photo_id = int(photos["photo_id"].max()) + 1

    clone_rows, clone_photo_rows, variant_rows = [], [], []
    for position, (source_id, kind) in enumerate(zip(sources, kinds)):
        source = by_id[source_id]
        listing_id = next_listing_id + position
        agent = rng.randrange(1, config.n_agents + 1)
        while agent == source["agent_id"] and config.n_agents > 1:
            agent = rng.randrange(1, config.n_agents + 1)

        asking = source["asking_price_aed"]
        if kind == "exact" and position in shifted:
            direction = rng.choice((-1.0, 1.0))
            moved = asking * (1.0 + direction * rng.uniform(*config.price_shift))
            asking = float(round(moved / 10_000) * 10_000)

        title, description = source["title"], source["description"]
        if kind == "reworded":
            facts = ListingFacts(
                bedrooms=source["bedrooms"],
                size_sqm=source["size_sqm"],
                area_name=source["area_name"],
                area_name_ar=arabic.get(source["area_id"]),
                building_name=source["building_name"],
                project_name=source["project_name"],
                property_type=source["property_type"],
                sub_kind=_sub_kind(source["property_type"], source["property_sub_type"]),
            )
            for attempt in range(5):
                text_rng = random.Random(config.seed * 7919 + listing_id + 500_000 + attempt * 7)
                title, description = make_title(text_rng, facts), make_description(text_rng, facts)
                if description != source["description"]:
                    break
            else:
                raise ValueError(f"could not reword listing {source_id} into different text")

        source_photos = photos_by_listing[source_id]
        if kind == "edited":
            photo_ids = []
            for photo_id in source_photos:
                meta = photo_meta[photo_id]
                variant_rows.append(
                    {
                        "photo_id": next_photo_id,
                        "set_id": meta["set_id"],
                        "room": meta["room"],
                        "path": f"variants/{next_photo_id}.jpg",
                        "variant_of": photo_id,
                        "variant_kind": "edited",
                        "is_stock": meta["is_stock"],
                    }
                )
                photo_ids.append(next_photo_id)
                next_photo_id += 1
        else:
            photo_ids = source_photos
        for slot, photo_id in enumerate(photo_ids):
            clone_photo_rows.append(
                {"listing_id": listing_id, "photo_id": photo_id, "position": slot}
            )

        clone_rows.append(
            {
                **source,
                "listing_id": listing_id,
                "agent_id": agent,
                "posted_at": source["posted_at"] + timedelta(days=rng.randint(*config.repost_days)),
                "title": title,
                "description": description,
                "asking_price_aed": asking,
                "dup_group_id": source_id,
                "control_group_id": None,
                "fraud_label": None,
            }
        )
    return (
        pl.DataFrame(clone_rows, schema=LISTING_SCHEMA),
        pl.DataFrame(clone_photo_rows, schema=LISTING_PHOTO_SCHEMA),
        pl.DataFrame(variant_rows, schema=PHOTO_SCHEMA),
    )


def _validate(listings: pl.DataFrame, photos: pl.DataFrame, config: CorpusConfig) -> None:
    if listings.height != config.n_listings:
        raise ValueError(f"generated {listings.height} listings, expected {config.n_listings}")
    stock_sets = photos.filter(pl.col("is_stock"))["set_id"].unique().to_list()
    spread = (
        listings.filter(pl.col("photo_set_id").is_in(stock_sets))
        .group_by("photo_set_id")
        .agg(pl.col("area_id").n_unique().alias("areas"))
    )
    thin = spread.filter(pl.col("areas") < config.stock_min_areas)
    if spread.height < len(stock_sets) or thin.height:
        raise ValueError(
            f"every stock photo set must span >= {config.stock_min_areas} areas; "
            f"unused or thin sets: {thin.to_dicts()}"
        )


def generate_corpus(
    sales: pl.DataFrame, areas: pl.DataFrame, pool_set_ids, config: CorpusConfig
) -> Corpus:
    if len(pool_set_ids) <= config.n_stock_sets:
        raise ValueError(
            f"photo pool has {len(pool_set_ids)} sets; need more than n_stock_sets="
            f"{config.n_stock_sets}"
        )
    shuffled = sorted(pool_set_ids)
    random.Random(config.seed).shuffle(shuffled)
    stock = sorted(shuffled[: config.n_stock_sets])
    plain = sorted(shuffled[config.n_stock_sets :])

    photos = _base_photo_rows(pool_set_ids, set(stock))
    base = _build_base_listings(_sample_base_sales(sales, config), areas, config, stock, plain)
    base = _apply_bait_prices(base, config)
    base_photos = _base_listing_photos(base, photos)

    arabic = dict(zip(areas["area_id"].to_list(), areas["name_ar"].to_list()))
    clones, clone_photos, variants = _plant_clones(base, photos, base_photos, arabic, config)

    listings = pl.concat([base, clones])
    all_photos = pl.concat([photos, variants])
    listing_photos = pl.concat([base_photos, clone_photos])
    _validate(listings, all_photos, config)
    counts = {
        "listings": listings.height,
        "base": base.height,
        "exact_repost": config.n_exact_repost,
        "reworded": config.n_reworded,
        "edited_photo": config.n_edited_photo,
        "bait_price": int(listings.filter(pl.col("fraud_label") == "bait_price").height),
        "price_shifted": config.n_price_shifted_reposts,
        "photos": all_photos.height,
        "variants": variants.height,
    }
    return Corpus(listings, all_photos, listing_photos, counts)


def render_variants(
    corpus: Corpus, pool: PhotoPool, config: CorpusConfig, variant_dir: Path = VARIANT_DIR
) -> int:
    """Write one edited image per variant row. Returns how many files were written."""
    variant_dir = Path(variant_dir)
    sources = {
        row["photo_id"]: (row["set_id"], row["room"]) for row in corpus.photos.iter_rows(named=True)
    }
    written = 0
    for row in corpus.photos.filter(pl.col("variant_kind") == "edited").iter_rows(named=True):
        set_id, room = sources[row["variant_of"]]
        make_variant(pool.path(set_id, room), variant_dir / Path(row["path"]).name, config)
        written += 1
    return written
```

- [ ] **Step 8: Run the generation tests to verify they pass**

Run: `uv run pytest tests/listings/ -v`
Expected: PASS.

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add listings/text.py listings/generate.py tests/listings/conftest.py tests/listings/test_listings_text.py tests/listings/test_listings_generate.py
git commit -m "$(cat <<'EOF'
feat(listings): generate the labelled synthetic corpus from real DLD sales

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Postgres schema and loader

**Files:**
- Create: `listings/sql/schema.sql`, `listings/load.py`, `tests/listings/test_listings_load.py`

**Interfaces:**
- Consumes: `ingestion.config.DbSettings`, `ingestion.load.copy_frame(cur, table, frame)` (Phase 2, reused for COPY), `listings.generate.Corpus`, the `pg_test_db` fixture
- Produces (`listings.load`):
  - `SCHEMA_SQL: Path`
  - `apply_schema(conn) -> None` (idempotent)
  - `start_corpus_run(conn, seed: int, photo_dataset_sha: str, counts: dict) -> int`
  - `replace_corpus(conn, corpus: Corpus, corpus_run_id: int) -> None` — truncates and COPYs listings, photos, listing_photos
  - `load_corpus(settings, corpus, seed, photo_dataset_sha) -> int` — one transaction, returns the corpus run id
  - `write_photo_embeddings(conn, frame) -> int` and `write_listing_embeddings(conn, frame) -> int`
  - `create_vector_indexes(conn) -> None` — HNSW, built after the vectors are in
  - `latest_corpus_run(conn) -> int`
  - `vector_literal(values) -> str` — pgvector's `[1,2,3]` text form

**Why the indexes are built after loading:** HNSW builds much faster over a
filled table than incrementally on insert, and the corpus is written once.

- [ ] **Step 1: Write the schema**

`listings/sql/schema.sql`:

```sql
CREATE SCHEMA IF NOT EXISTS listings;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS listings.corpus_runs (
    corpus_run_id serial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    seed integer NOT NULL,
    photo_dataset_sha text,
    counts jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS listings.listings (
    listing_id bigint PRIMARY KEY,
    source_transaction_id text NOT NULL,
    agent_id integer NOT NULL,
    posted_at date NOT NULL,
    title text NOT NULL,
    description text NOT NULL,
    asking_price_aed double precision NOT NULL,
    area_id integer NOT NULL,
    area_name text,
    building_name text,
    project_name text,
    property_type text NOT NULL,
    property_sub_type text,
    reg_type text NOT NULL,
    size_sqm double precision NOT NULL,
    bedrooms smallint,
    photo_set_id integer NOT NULL,
    dup_group_id bigint,
    control_group_id text,
    fraud_label text,
    is_synthetic boolean NOT NULL DEFAULT true,
    corpus_run_id integer NOT NULL REFERENCES listings.corpus_runs (corpus_run_id)
);

CREATE TABLE IF NOT EXISTS listings.photos (
    photo_id bigint PRIMARY KEY,
    set_id integer NOT NULL,
    room text NOT NULL,
    path text NOT NULL,
    variant_of bigint,
    variant_kind text,
    is_stock boolean NOT NULL,
    embedding vector(512),
    corpus_run_id integer NOT NULL REFERENCES listings.corpus_runs (corpus_run_id)
);

CREATE TABLE IF NOT EXISTS listings.listing_photos (
    listing_id bigint NOT NULL REFERENCES listings.listings (listing_id),
    photo_id bigint NOT NULL REFERENCES listings.photos (photo_id),
    position smallint NOT NULL,
    PRIMARY KEY (listing_id, position)
);

CREATE TABLE IF NOT EXISTS listings.listing_embeddings (
    listing_id bigint PRIMARY KEY REFERENCES listings.listings (listing_id),
    text_embedding vector(384),
    image_embedding vector(512)
);

CREATE TABLE IF NOT EXISTS listings.detect_runs (
    detect_run_id serial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    corpus_run_id integer NOT NULL REFERENCES listings.corpus_runs (corpus_run_id),
    threshold double precision,
    metrics jsonb
);

CREATE TABLE IF NOT EXISTS listings.duplicate_pairs (
    detect_run_id integer NOT NULL REFERENCES listings.detect_runs (detect_run_id),
    listing_a bigint NOT NULL REFERENCES listings.listings (listing_id),
    listing_b bigint NOT NULL REFERENCES listings.listings (listing_id),
    score double precision NOT NULL,
    decision boolean NOT NULL,
    signals jsonb NOT NULL,
    PRIMARY KEY (detect_run_id, listing_a, listing_b),
    CONSTRAINT duplicate_pairs_canonical CHECK (listing_a < listing_b)
);

CREATE TABLE IF NOT EXISTS listings.fraud_flags (
    detect_run_id integer NOT NULL REFERENCES listings.detect_runs (detect_run_id),
    listing_id bigint NOT NULL REFERENCES listings.listings (listing_id),
    flag text NOT NULL,
    detail jsonb NOT NULL,
    PRIMARY KEY (detect_run_id, listing_id, flag)
);

CREATE INDEX IF NOT EXISTS listings_posted_at_idx ON listings.listings (posted_at);
CREATE INDEX IF NOT EXISTS listings_area_building_idx
    ON listings.listings (area_id, building_name);
CREATE INDEX IF NOT EXISTS listing_photos_photo_idx ON listings.listing_photos (photo_id);

COMMENT ON TABLE listings.listings IS
    'Synthetic listings generated over real DLD sales (Phase 4). Text, agents, posting '
    'dates, duplicates and fraud cases are invented; area, building, size and price level '
    'come from dld.market_sales.';
COMMENT ON COLUMN listings.listings.dup_group_id IS
    'Ground truth: the listing this one was cloned from. Never a detection input.';
COMMENT ON COLUMN listings.listings.control_group_id IS
    'Ground truth: same-building group that must NOT be flagged. Never a detection input.';
COMMENT ON COLUMN listings.listings.fraud_label IS
    'Ground truth fraud case. Never a detection input.';
```

- [ ] **Step 2: Write the failing loader tests**

`tests/listings/test_listings_load.py`:

```python
import polars as pl
import pytest

from listings.generate import generate_corpus
from listings.load import (
    apply_schema,
    create_vector_indexes,
    latest_corpus_run,
    load_corpus,
    vector_literal,
    write_listing_embeddings,
    write_photo_embeddings,
)

POOL = tuple(range(1, 21))


def corpus_for(sales_frame, areas_frame, config):
    return generate_corpus(sales_frame(), areas_frame, POOL, config)


def query(settings, sql, params=None):
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def test_vector_literal_formats_for_pgvector():
    assert vector_literal([0.5, -0.25, 0.0]) == "[0.5,-0.25,0.0]"


def test_schema_is_idempotent(pg_test_db):
    conn = pg_test_db.connect()
    try:
        apply_schema(conn)
        apply_schema(conn)
        conn.commit()
    finally:
        conn.close()
    tables = {row[0] for row in query(pg_test_db, "SELECT tablename FROM pg_tables WHERE schemaname='listings'")}
    assert {"listings", "photos", "listing_photos", "listing_embeddings", "duplicate_pairs", "fraud_flags", "corpus_runs", "detect_runs"} <= tables


def test_load_corpus_writes_every_table_and_links_the_run(
    pg_test_db, sales_frame, areas_frame, small_corpus_config
):
    corpus = corpus_for(sales_frame, areas_frame, small_corpus_config)
    run_id = load_corpus(pg_test_db, corpus, seed=small_corpus_config.seed, photo_dataset_sha="abc")

    assert query(pg_test_db, "SELECT count(*) FROM listings.listings")[0][0] == corpus.listings.height
    assert query(pg_test_db, "SELECT count(*) FROM listings.photos")[0][0] == corpus.photos.height
    assert (
        query(pg_test_db, "SELECT count(*) FROM listings.listing_photos")[0][0]
        == corpus.listing_photos.height
    )
    assert query(pg_test_db, "SELECT count(DISTINCT corpus_run_id) FROM listings.listings") == [(1,)]
    row = query(
        pg_test_db,
        "SELECT seed, photo_dataset_sha, counts->>'listings' FROM listings.corpus_runs WHERE corpus_run_id=%s",
        (run_id,),
    )[0]
    assert row[0] == small_corpus_config.seed
    assert row[1] == "abc"
    assert int(row[2]) == corpus.listings.height
    assert latest_corpus_run_of(pg_test_db) == run_id


def latest_corpus_run_of(settings):
    conn = settings.connect()
    try:
        return latest_corpus_run(conn)
    finally:
        conn.close()


def test_second_load_replaces_the_previous_corpus(
    pg_test_db, sales_frame, areas_frame, small_corpus_config
):
    corpus = corpus_for(sales_frame, areas_frame, small_corpus_config)
    first = load_corpus(pg_test_db, corpus, seed=1, photo_dataset_sha="a")
    second = load_corpus(pg_test_db, corpus, seed=2, photo_dataset_sha="b")
    assert second > first
    assert query(pg_test_db, "SELECT count(*) FROM listings.listings")[0][0] == corpus.listings.height
    assert query(pg_test_db, "SELECT DISTINCT corpus_run_id FROM listings.listings") == [(second,)]


def test_embeddings_round_trip_and_knn_finds_the_nearest(
    pg_test_db, sales_frame, areas_frame, small_corpus_config
):
    corpus = corpus_for(sales_frame, areas_frame, small_corpus_config)
    load_corpus(pg_test_db, corpus, seed=1, photo_dataset_sha="a")

    photo_ids = corpus.photos["photo_id"].to_list()[:6]
    photo_vectors = pl.DataFrame(
        {
            "photo_id": photo_ids,
            "embedding": [vector_literal([1.0 if i == j else 0.0 for j in range(512)]) for i in range(len(photo_ids))],
        }
    )
    listing_ids = corpus.listings["listing_id"].to_list()[:4]
    listing_vectors = pl.DataFrame(
        {
            "listing_id": listing_ids,
            "text_embedding": [vector_literal([1.0 if i == j else 0.0 for j in range(384)]) for i in range(len(listing_ids))],
            "image_embedding": [vector_literal([1.0 if i == j else 0.0 for j in range(512)]) for i in range(len(listing_ids))],
        }
    )
    conn = pg_test_db.connect()
    try:
        assert write_photo_embeddings(conn, photo_vectors) == len(photo_ids)
        assert write_listing_embeddings(conn, listing_vectors) == len(listing_ids)
        create_vector_indexes(conn)
        conn.commit()
    finally:
        conn.close()

    probe = vector_literal([1.0 if j == 0 else 0.0 for j in range(512)])
    nearest = query(
        pg_test_db,
        "SELECT photo_id FROM listings.photos WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> %s::vector LIMIT 1",
        (probe,),
    )
    assert nearest[0][0] == photo_ids[0]
    indexes = {row[0] for row in query(pg_test_db, "SELECT indexname FROM pg_indexes WHERE schemaname='listings'")}
    assert {"photos_embedding_hnsw", "listing_text_embedding_hnsw", "listing_image_embedding_hnsw"} <= indexes


def test_duplicate_pairs_must_be_canonical(pg_test_db, sales_frame, areas_frame, small_corpus_config):
    corpus = corpus_for(sales_frame, areas_frame, small_corpus_config)
    run_id = load_corpus(pg_test_db, corpus, seed=1, photo_dataset_sha="a")
    a, b = corpus.listings["listing_id"].to_list()[:2]
    conn = pg_test_db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO listings.detect_runs (corpus_run_id, threshold) VALUES (%s, 0.5) "
                "RETURNING detect_run_id",
                (run_id,),
            )
            detect_run_id = cur.fetchone()[0]
            with pytest.raises(Exception, match="duplicate_pairs_canonical"):
                cur.execute(
                    "INSERT INTO listings.duplicate_pairs "
                    "(detect_run_id, listing_a, listing_b, score, decision, signals) "
                    "VALUES (%s, %s, %s, 0.9, true, '{}'::jsonb)",
                    (detect_run_id, max(a, b), min(a, b)),
                )
    finally:
        conn.rollback()
        conn.close()


def test_the_listings_table_says_it_is_synthetic(pg_test_db):
    conn = pg_test_db.connect()
    try:
        apply_schema(conn)
        conn.commit()
    finally:
        conn.close()
    comment = query(pg_test_db, "SELECT obj_description('listings.listings'::regclass)")[0][0]
    assert "ynthetic" in comment
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/listings/test_listings_load.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'listings.load'`.

- [ ] **Step 4: Implement `listings/load.py`**

```python
"""Load the synthetic corpus and its vectors into Postgres schema `listings`."""

import json
from pathlib import Path

import polars as pl

from ingestion.config import DbSettings
from ingestion.load import copy_frame
from listings.generate import Corpus

SCHEMA_SQL = Path(__file__).parent / "sql" / "schema.sql"
# Truncated together: a new corpus invalidates every detection result that referenced it.
CORPUS_TABLES = (
    "duplicate_pairs",
    "fraud_flags",
    "detect_runs",
    "listing_photos",
    "listing_embeddings",
    "photos",
    "listings",
)


def vector_literal(values) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def apply_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))


def start_corpus_run(conn, seed: int, photo_dataset_sha: str, counts: dict) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO listings.corpus_runs (seed, photo_dataset_sha, counts) "
            "VALUES (%s, %s, %s::jsonb) RETURNING corpus_run_id",
            (seed, photo_dataset_sha, json.dumps(counts)),
        )
        return cur.fetchone()[0]


def latest_corpus_run(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT max(corpus_run_id) FROM listings.corpus_runs")
        (run_id,) = cur.fetchone()
    if run_id is None:
        raise RuntimeError("no corpus has been loaded yet — run `python -m listings build` first")
    return run_id


def replace_corpus(conn, corpus: Corpus, corpus_run_id: int) -> None:
    """Truncate the corpus tables and COPY this corpus in. Caller owns the transaction."""
    stamped = pl.lit(corpus_run_id, dtype=pl.Int64)
    with conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout = '60s'")
        cur.execute(f"TRUNCATE {', '.join(f'listings.{t}' for t in CORPUS_TABLES)}")
        copy_frame(cur, "listings.listings", corpus.listings.with_columns(stamped.alias("corpus_run_id")))
        # embedding is left out of the COPY: it is filled in later by write_photo_embeddings
        copy_frame(cur, "listings.photos", corpus.photos.with_columns(stamped.alias("corpus_run_id")))
        copy_frame(cur, "listings.listing_photos", corpus.listing_photos)


def load_corpus(settings: DbSettings, corpus: Corpus, seed: int, photo_dataset_sha: str) -> int:
    conn = settings.connect()
    try:
        apply_schema(conn)
        corpus_run_id = start_corpus_run(conn, seed, photo_dataset_sha, corpus.counts)
        replace_corpus(conn, corpus, corpus_run_id)
        conn.commit()
        return corpus_run_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def write_photo_embeddings(conn, frame: pl.DataFrame) -> int:
    """frame: photo_id, embedding (pgvector text form)."""
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE photo_vectors (photo_id bigint, embedding text) ON COMMIT DROP")
        copy_frame(cur, "photo_vectors", frame.select("photo_id", "embedding"))
        cur.execute(
            "UPDATE listings.photos AS p SET embedding = v.embedding::vector "
            "FROM photo_vectors AS v WHERE p.photo_id = v.photo_id"
        )
        return cur.rowcount


def write_listing_embeddings(conn, frame: pl.DataFrame) -> int:
    """frame: listing_id, text_embedding, image_embedding (pgvector text form)."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE listing_vectors "
            "(listing_id bigint, text_embedding text, image_embedding text) ON COMMIT DROP"
        )
        copy_frame(
            cur,
            "listing_vectors",
            frame.select("listing_id", "text_embedding", "image_embedding"),
        )
        cur.execute(
            "INSERT INTO listings.listing_embeddings (listing_id, text_embedding, image_embedding) "
            "SELECT listing_id, text_embedding::vector, image_embedding::vector FROM listing_vectors "
            "ON CONFLICT (listing_id) DO UPDATE SET "
            "text_embedding = EXCLUDED.text_embedding, image_embedding = EXCLUDED.image_embedding"
        )
        return cur.rowcount


def create_vector_indexes(conn) -> None:
    """HNSW, cosine. Built after the vectors are in: much faster than incremental inserts."""
    statements = (
        "CREATE INDEX IF NOT EXISTS photos_embedding_hnsw ON listings.photos "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
        "CREATE INDEX IF NOT EXISTS listing_text_embedding_hnsw ON listings.listing_embeddings "
        "USING hnsw (text_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
        "CREATE INDEX IF NOT EXISTS listing_image_embedding_hnsw ON listings.listing_embeddings "
        "USING hnsw (image_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
    )
    with conn.cursor() as cur:
        for statement in statements:
            cur.execute(statement)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/listings/ -v`
Expected: PASS (the Docker stack must be up; the DB tests use the throwaway `dubimator_test` database).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add listings/sql/schema.sql listings/load.py tests/listings/test_listings_load.py
git commit -m "$(cat <<'EOF'
feat(listings): Postgres schema with pgvector storage and the corpus loader

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Embeddings (CLIP + MiniLM, GPU or CPU, with a test double)

**Files:**
- Create: `listings/embed.py`, `tests/listings/test_listings_embed.py`
- Modify: `tests/listings/conftest.py` (add the `fake_embedder` fixture)

**Interfaces:**
- Consumes: `listings.load.vector_literal`, `listings.config`
- Produces (`listings.embed`):
  - `IMAGE_DIM = 512`, `TEXT_DIM = 384`, `IMAGE_MODEL`, `TEXT_MODEL`, `DEVICES = ("auto", "cuda", "cpu")`
  - `Embedder` — a `Protocol` with `embed_images(paths: Sequence[Path]) -> np.ndarray` (n × 512) and `embed_texts(texts: Sequence[str]) -> np.ndarray` (n × 384), both L2-normalised
  - `normalize(matrix) -> np.ndarray`
  - `resolve_device(requested="auto") -> str`
  - `SentenceTransformerEmbedder(device="cpu", batch_size=64)` — the real one; models load lazily on first use
  - `FakeEmbedder()` — deterministic, no model, no network
  - `embed_photos(photos, embedder, data_dir) -> tuple[pl.DataFrame, dict[int, np.ndarray]]` — the frame is `photo_id, embedding` (pgvector text), the dict is the raw vectors
  - `embed_listings(listings, listing_photos, photo_vectors, embedder) -> pl.DataFrame` — `listing_id, text_embedding, image_embedding`
- Fixture `fake_embedder` in `tests/listings/conftest.py`

**Why the fake is perceptual, not random:** tests need an embedder where a
cropped, resized, re-saved photo stays close to its source, and a reworded
description stays closer to its source than to an unrelated listing —
otherwise the duplicate-detection tests would prove nothing. So the fake
downscales images to 16×16 grey (an average-hash style embedding) and hashes
text into word buckets. Both are real, cheap similarity measures; neither
needs a model download.

- [ ] **Step 1: Add the fixture**

Append to `tests/listings/conftest.py`:

```python
@pytest.fixture
def fake_embedder():
    from listings.embed import FakeEmbedder

    return FakeEmbedder()
```

- [ ] **Step 2: Write the failing tests**

`tests/listings/test_listings_embed.py`:

```python
import numpy as np
import polars as pl
import pytest

from listings.config import CorpusConfig
from listings.embed import (
    IMAGE_DIM,
    TEXT_DIM,
    FakeEmbedder,
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
    original = "Just listed: 2-bedroom apartment in Marsa Dubai. Features include balcony, gym access."
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/listings/test_listings_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'listings.embed'`.

- [ ] **Step 4: Implement `listings/embed.py`**

```python
"""Photo and text embeddings.

Everything downstream talks to the `Embedder` protocol, so tests inject a
deterministic double and never download a model or need a GPU.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

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
        for title, description in zip(listings["title"].to_list(), listings["description"].to_list())
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/listings/ -v`
Expected: PASS. `test_real_clip_embeds_photos` is skipped unless the CLIP model is already cached — that is expected on a fresh machine and is not a failure.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add listings/embed.py tests/listings/conftest.py tests/listings/test_listings_embed.py
git commit -m "$(cat <<'EOF'
feat(listings): CLIP and MiniLM embeddings behind a protocol with a deterministic test double

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Candidate retrieval through pgvector

**Files:**
- Create: `listings/candidates.py`, `tests/listings/test_listings_candidates.py`
- Modify: `tests/listings/conftest.py` (add the `loaded_corpus` fixture)

**Interfaces:**
- Consumes: `listings.load` (`load_corpus`, `write_photo_embeddings`, `write_listing_embeddings`, `create_vector_indexes`), `listings.embed` (`embed_photos`, `embed_listings`, `FakeEmbedder`), `listings.config.DetectConfig`, the `pg_test_db` fixture
- Produces (`listings.candidates`):
  - `CandidateStats`, a frozen dataclass: `text_pairs: int`, `image_pairs: int`, `shared_photo_pairs: int`, `total: int`, `seconds: float`
  - `fetch_candidates(conn, config) -> tuple[pl.DataFrame, CandidateStats]` — the frame has `listing_a`, `listing_b` (canonical, unique) and `sources` (comma-separated channels that proposed it)
  - `brute_force_pairs(conn, metric, top_k) -> pl.DataFrame` — exact nearest neighbours by sequential scan, for the index-correctness test and the timing comparison
- Fixture `loaded_corpus` in `tests/listings/conftest.py`: builds a small corpus, loads it into `pg_test_db`, embeds with `FakeEmbedder`, writes vectors, creates the indexes, and returns `(settings, corpus, corpus_run_id)`

Three channels, unioned:
1. **text** — `text_top_k` nearest by `text_embedding`
2. **image** — `photo_top_k` nearest by the listing's average `image_embedding`
3. **shared photo** — listings sharing a `photo_id` used by at most `max_photo_fanout` listings (skipping agency stock, which would otherwise produce millions of pairs)

- [ ] **Step 1: Add the `loaded_corpus` fixture**

Append to `tests/listings/conftest.py`:

```python
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
    corpus = generate_corpus(sales_frame(), areas_frame, photo_pool.set_ids, config)
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
```

- [ ] **Step 2: Write the failing tests**

`tests/listings/test_listings_candidates.py`:

```python
import dataclasses

import polars as pl

from listings.candidates import brute_force_pairs, fetch_candidates
from listings.config import DetectConfig

CONFIG = DetectConfig()


def candidates_for(settings, config=CONFIG):
    conn = settings.connect()
    try:
        return fetch_candidates(conn, config)
    finally:
        conn.close()


def test_pairs_are_canonical_unique_and_never_self(loaded_corpus):
    settings, corpus, _ = loaded_corpus
    pairs, stats = candidates_for(settings)

    assert pairs.height > 0
    assert (pairs["listing_a"] < pairs["listing_b"]).all()
    assert pairs.select("listing_a", "listing_b").is_duplicated().sum() == 0
    known = set(corpus.listings["listing_id"].to_list())
    assert set(pairs["listing_a"].to_list()) <= known
    assert set(pairs["listing_b"].to_list()) <= known
    assert stats.total == pairs.height
    assert stats.seconds >= 0.0
    assert {"text", "image", "shared_photo"} >= set(
        source for row in pairs["sources"].to_list() for source in row.split(",")
    )


def test_planted_duplicates_are_retrieved(loaded_corpus):
    settings, corpus, _ = loaded_corpus
    pairs, _ = candidates_for(settings)
    found = {(a, b) for a, b in zip(pairs["listing_a"].to_list(), pairs["listing_b"].to_list())}

    clones = corpus.listings.filter(pl.col("dup_group_id").is_not_null())
    truth = {
        (min(row["listing_id"], row["dup_group_id"]), max(row["listing_id"], row["dup_group_id"]))
        for row in clones.iter_rows(named=True)
    }
    recall = len(truth & found) / len(truth)
    assert recall >= 0.95, f"retrieval recall {recall:.3f} — duplicates missed here can never be caught"


def test_stock_photos_do_not_explode_the_shared_photo_channel(loaded_corpus):
    settings, corpus, _ = loaded_corpus
    tight = dataclasses.replace(CONFIG, max_photo_fanout=3)
    pairs, stats = candidates_for(settings, tight)
    _, wide_stats = candidates_for(settings, dataclasses.replace(CONFIG, max_photo_fanout=10_000))
    assert stats.shared_photo_pairs < wide_stats.shared_photo_pairs
    assert pairs.height <= wide_stats.total


def test_index_retrieval_agrees_with_an_exact_scan(loaded_corpus):
    settings, _, _ = loaded_corpus
    conn = settings.connect()
    try:
        approximate, _ = fetch_candidates(conn, CONFIG)
        exact = brute_force_pairs(conn, "text", CONFIG.text_top_k)
    finally:
        conn.close()
    found = {(a, b) for a, b in zip(approximate["listing_a"], approximate["listing_b"])}
    truth = {(a, b) for a, b in zip(exact["listing_a"], exact["listing_b"])}
    recall = len(truth & found) / len(truth)
    assert recall >= 0.95, f"HNSW recall against an exact scan was {recall:.3f}"


def test_a_corpus_without_vectors_yields_no_pairs(pg_test_db, sales_frame, areas_frame, small_corpus_config):
    from listings.generate import generate_corpus
    from listings.load import load_corpus

    corpus = generate_corpus(sales_frame(), areas_frame, tuple(range(1, 21)), small_corpus_config)
    load_corpus(pg_test_db, corpus, seed=1, photo_dataset_sha="x")  # no embeddings written
    conn = pg_test_db.connect()
    try:
        pairs, stats = fetch_candidates(conn, dataclasses.replace(CONFIG, max_photo_fanout=0))
    finally:
        conn.close()
    assert pairs.height == 0 and stats.total == 0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/listings/test_listings_candidates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'listings.candidates'`.

- [ ] **Step 4: Implement `listings/candidates.py`**

```python
"""Candidate pairs from pgvector.

Comparing all 20,000 listings pairwise is 200 million comparisons. Three indexed
channels bring that down to a few tens of candidates per listing.
"""

import time
from dataclasses import dataclass

import polars as pl

from listings.config import DetectConfig

PAIR_SCHEMA = {"listing_a": pl.Int64, "listing_b": pl.Int64}

TEXT_SQL = """
SELECT least(p.listing_id, n.listing_id) AS listing_a,
       greatest(p.listing_id, n.listing_id) AS listing_b
FROM listings.listing_embeddings p
CROSS JOIN LATERAL (
    SELECT e.listing_id
    FROM listings.listing_embeddings e
    WHERE e.listing_id <> p.listing_id AND e.text_embedding IS NOT NULL
    ORDER BY e.text_embedding <=> p.text_embedding
    LIMIT %(k)s
) n
WHERE p.text_embedding IS NOT NULL
"""

IMAGE_SQL = """
SELECT least(p.listing_id, n.listing_id) AS listing_a,
       greatest(p.listing_id, n.listing_id) AS listing_b
FROM listings.listing_embeddings p
CROSS JOIN LATERAL (
    SELECT e.listing_id
    FROM listings.listing_embeddings e
    WHERE e.listing_id <> p.listing_id AND e.image_embedding IS NOT NULL
    ORDER BY e.image_embedding <=> p.image_embedding
    LIMIT %(k)s
) n
WHERE p.image_embedding IS NOT NULL
"""

# Photos used by more than max_photo_fanout listings are agency stock: pairing every
# user with every other would be quadratic and would flag legitimate reuse anyway.
SHARED_PHOTO_SQL = """
WITH usage AS (
    SELECT photo_id, count(*) AS listings
    FROM listings.listing_photos
    GROUP BY photo_id
)
SELECT a.listing_id AS listing_a, b.listing_id AS listing_b
FROM listings.listing_photos a
JOIN usage ON usage.photo_id = a.photo_id AND usage.listings <= %(fanout)s
JOIN listings.listing_photos b ON b.photo_id = a.photo_id AND b.listing_id > a.listing_id
"""


@dataclass(frozen=True)
class CandidateStats:
    text_pairs: int
    image_pairs: int
    shared_photo_pairs: int
    total: int
    seconds: float


def _query(conn, sql: str, params: dict) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pl.DataFrame(cur.fetchall(), schema=PAIR_SCHEMA, orient="row").unique()


def fetch_candidates(conn, config: DetectConfig) -> tuple[pl.DataFrame, CandidateStats]:
    started = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("SET LOCAL hnsw.ef_search = %s", (config.ef_search,))
    channels = {
        "text": _query(conn, TEXT_SQL, {"k": config.text_top_k}),
        "image": _query(conn, IMAGE_SQL, {"k": config.photo_top_k}),
        "shared_photo": _query(conn, SHARED_PHOTO_SQL, {"fanout": config.max_photo_fanout}),
    }
    labelled = [
        frame.with_columns(pl.lit(name).alias("source"))
        for name, frame in channels.items()
        if frame.height
    ]
    if not labelled:
        empty = pl.DataFrame(
            {"listing_a": [], "listing_b": [], "sources": []},
            schema={**PAIR_SCHEMA, "sources": pl.Utf8},
        )
        return empty, CandidateStats(0, 0, 0, 0, time.perf_counter() - started)

    pairs = (
        pl.concat(labelled)
        .group_by(["listing_a", "listing_b"])
        .agg(pl.col("source").unique().sort().str.join(",").alias("sources"))
        .sort(["listing_a", "listing_b"])
    )
    return pairs, CandidateStats(
        text_pairs=channels["text"].height,
        image_pairs=channels["image"].height,
        shared_photo_pairs=channels["shared_photo"].height,
        total=pairs.height,
        seconds=time.perf_counter() - started,
    )


def brute_force_pairs(conn, metric: str, top_k: int) -> pl.DataFrame:
    """Exact nearest neighbours by sequential scan — the yardstick for index recall and timing."""
    sql = {"text": TEXT_SQL, "image": IMAGE_SQL}[metric]
    with conn.cursor() as cur:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        cur.execute(sql, {"k": top_k})
        rows = cur.fetchall()
        cur.execute("SET LOCAL enable_indexscan = on")
        cur.execute("SET LOCAL enable_bitmapscan = on")
    return pl.DataFrame(rows, schema=PAIR_SCHEMA, orient="row").unique()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/listings/ -v`
Expected: PASS.

If `test_planted_duplicates_are_retrieved` fails, do NOT lower the threshold: report the measured recall and the per-channel stats as a concern. Retrieval recall is a headline metric, and a low value is a finding for the controller, not a number to tune away.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add listings/candidates.py tests/listings/conftest.py tests/listings/test_listings_candidates.py
git commit -m "$(cat <<'EOF'
feat(listings): retrieve duplicate candidates through pgvector in three channels

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Pair features, ground truth, and the decision model

**Files:**
- Create: `listings/features.py`, `listings/truth.py`, `listings/detect.py`, `tests/listings/test_listings_features.py`, `tests/listings/test_listings_detect.py`

**Interfaces:**
- Consumes: `listings.candidates.fetch_candidates`, `listings.config` (`PAIR_FEATURES`, `LABEL_COLUMNS`, `DetectConfig`), `listings.load.latest_corpus_run`, the `loaded_corpus` fixture
- Produces (`listings.features`):
  - `LISTING_SQL: str` and `load_listing_attributes(conn) -> pl.DataFrame` (no label columns)
  - `load_listing_vectors(conn) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]` — text, image
  - `load_photo_sets(conn) -> tuple[dict[int, list[int]], dict[int, np.ndarray]]` — listing → photo ids, photo id → vector
  - `build_features(pairs, attributes, text_vectors, image_vectors, listing_photo_ids, photo_vectors) -> pl.DataFrame` — `listing_a`, `listing_b`, then exactly `PAIR_FEATURES`
- Produces (`listings.truth`) — the only module that reads label columns:
  - `TRUTH_SQL: str`, `load_truth(conn) -> pl.DataFrame` (`listing_id`, `dup_group_id`, `control_group_id`, `fraud_label`)
  - `label_pairs(pairs, truth) -> pl.DataFrame` — adds `is_duplicate` and `same_building_control` (the stock-photo control is derived in Task 9 from photo sets and areas, which truth.py does not read)
- Produces (`listings.detect`):
  - `DetectionResult`, a frozen dataclass: `pairs: pl.DataFrame`, `threshold: float`, `model`, `feature_names: tuple[str, ...]`, `stats: dict[str, float]`
  - `assign_pair_split(pairs, attributes, config) -> pl.DataFrame` — adds `split` (`train` / `threshold` / `report`) from the later listing's `posted_at`
  - `fit_pair_model(features, labels, config)` → a fitted `Pipeline`
  - `choose_threshold(scores, labels, target_precision) -> float`
  - `baseline_decisions(features, config) -> np.ndarray`
  - `run_detection(conn, config) -> DetectionResult`
  - `write_detection(conn, result, corpus_run_id) -> int` — creates the detect run and writes `duplicate_pairs`

**Two duplicates of the same source are duplicates of each other.** `label_pairs`
marks a pair as duplicate when either listing was cloned from the other, or both
were cloned from the same source.

- [ ] **Step 1: Write the failing feature tests**

`tests/listings/test_listings_features.py`:

```python
from datetime import date

import numpy as np
import polars as pl

from listings.config import LABEL_COLUMNS, PAIR_FEATURES
from listings.features import LISTING_SQL, build_features, load_listing_attributes
from listings.truth import TRUTH_SQL, label_pairs, load_truth

DIM = 8


def unit(index, size=DIM):
    vector = np.zeros(size)
    vector[index] = 1.0
    return vector


ATTRIBUTES = pl.DataFrame(
    {
        "listing_id": [1, 2, 3],
        "posted_at": [date(2023, 1, 1), date(2023, 1, 8), date(2023, 2, 1)],
        "asking_price_aed": [1_000_000.0, 1_100_000.0, 2_000_000.0],
        "size_sqm": [100.0, 100.0, 200.0],
        "bedrooms": [2, 2, 4],
        "area_id": [10, 10, 20],
        "building_name": ["Tower A", "Tower A", "Tower B"],
        "project_name": ["Marina Gate", "Marina Gate", "Hills"],
        "agent_id": [7, 7, 9],
    }
)


def test_feature_columns_are_exactly_the_allowlist():
    pairs = pl.DataFrame({"listing_a": [1], "listing_b": [2]})
    features = build_features(
        pairs, ATTRIBUTES, {1: unit(0), 2: unit(0), 3: unit(1)}, {1: unit(0), 2: unit(1), 3: unit(2)},
        {1: [100, 101], 2: [100, 102], 3: [103]},
        {100: unit(0), 101: unit(1), 102: unit(1), 103: unit(2)},
    )  # fmt: skip
    assert features.columns == ["listing_a", "listing_b", *PAIR_FEATURES]
    assert not set(features.columns) & set(LABEL_COLUMNS)


def test_features_are_computed_correctly():
    pairs = pl.DataFrame({"listing_a": [1, 1], "listing_b": [2, 3]})
    features = build_features(
        pairs, ATTRIBUTES, {1: unit(0), 2: unit(0), 3: unit(1)}, {1: unit(0), 2: unit(1), 3: unit(2)},
        {1: [100, 101], 2: [100, 102], 3: [103]},
        {100: unit(0), 101: unit(1), 102: unit(1), 103: unit(2)},
    ).to_dicts()  # fmt: skip

    near, far = features[0], features[1]
    assert near["text_cosine"] == 1.0  # identical text vectors
    assert near["image_mean_cosine"] == 0.0  # orthogonal listing image vectors
    assert near["image_max_cosine"] == 1.0  # photo 100 is in both, and 101/102 match too
    assert near["shared_photo_count"] == 1  # only photo 100 is literally the same row
    assert near["abs_log_price_ratio"] == abs(np.log(1_000_000 / 1_100_000))
    assert near["abs_log_size_ratio"] == 0.0
    assert (near["same_area"], near["same_building"], near["same_project"]) == (1, 1, 1)
    assert near["bedrooms_equal"] == 1
    assert near["days_apart"] == 7
    assert near["same_agent"] == 1

    assert far["same_area"] == 0 and far["same_building"] == 0 and far["same_agent"] == 0
    assert far["shared_photo_count"] == 0
    assert far["days_apart"] == 31


def test_missing_bedrooms_are_not_counted_as_equal():
    attributes = ATTRIBUTES.with_columns(
        pl.Series("bedrooms", [None, None, 4], dtype=pl.Int64)
    )
    features = build_features(
        pl.DataFrame({"listing_a": [1], "listing_b": [2]}), attributes,
        {1: unit(0), 2: unit(0)}, {1: unit(0), 2: unit(0)}, {1: [100], 2: [100]}, {100: unit(0)},
    ).to_dicts()[0]  # fmt: skip
    assert features["bedrooms_equal"] == 0


def test_a_listing_without_photos_scores_zero_image_similarity():
    features = build_features(
        pl.DataFrame({"listing_a": [1], "listing_b": [2]}), ATTRIBUTES,
        {1: unit(0), 2: unit(0)}, {1: unit(0), 2: np.zeros(DIM)}, {1: [100], 2: []}, {100: unit(0)},
    ).to_dicts()[0]  # fmt: skip
    assert features["image_max_cosine"] == 0.0
    assert features["image_mean_cosine"] == 0.0
    assert features["shared_photo_count"] == 0


def test_neither_query_mentions_a_label_column():
    for name in LABEL_COLUMNS:
        assert name not in LISTING_SQL
        assert name in TRUTH_SQL  # truth.py is the one module that may read them


def test_pair_labels_cover_clones_siblings_and_controls():
    truth = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "dup_group_id": [None, 1, 1, None, None],
            "control_group_id": [None, None, None, "bldg:10:Tower A", "bldg:10:Tower A"],
            "fraud_label": [None, None, None, None, "bait_price"],
        },
        schema={
            "listing_id": pl.Int64, "dup_group_id": pl.Int64,
            "control_group_id": pl.Utf8, "fraud_label": pl.Utf8,
        },
    )  # fmt: skip
    pairs = pl.DataFrame({"listing_a": [1, 2, 4, 1], "listing_b": [2, 3, 5, 4]})
    labelled = label_pairs(pairs, truth).to_dicts()
    assert labelled[0]["is_duplicate"] is True  # clone of
    assert labelled[1]["is_duplicate"] is True  # two clones of the same source
    assert labelled[2]["is_duplicate"] is False and labelled[2]["same_building_control"] is True
    assert labelled[3]["is_duplicate"] is False and labelled[3]["same_building_control"] is False
```

- [ ] **Step 2: Run the feature tests to verify they fail**

Run: `uv run pytest tests/listings/test_listings_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'listings.features'`.

- [ ] **Step 3: Implement `listings/features.py`**

```python
"""The twelve pair features. This module never reads a label column.

Photo-level similarity is the expensive part: each pair compares its photos with
the other listing's (4 x 4 dot products). Vectors are stacked per listing once,
so the per-pair work is a single small matrix multiply.
"""

import numpy as np
import polars as pl

from listings.config import PAIR_FEATURES

LISTING_SQL = """
SELECT listing_id, posted_at, asking_price_aed, size_sqm, bedrooms,
       area_id, building_name, project_name, agent_id, photo_set_id
FROM listings.listings
ORDER BY listing_id
"""
LISTING_ATTRIBUTE_SCHEMA = {
    "listing_id": pl.Int64,
    "posted_at": pl.Date,
    "asking_price_aed": pl.Float64,
    "size_sqm": pl.Float64,
    "bedrooms": pl.Int64,
    "area_id": pl.Int64,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "agent_id": pl.Int64,
    "photo_set_id": pl.Int64,
}
VECTOR_SQL = """
SELECT listing_id, text_embedding::text, image_embedding::text
FROM listings.listing_embeddings
"""
PHOTO_SQL = """
SELECT lp.listing_id, lp.photo_id, p.embedding::text
FROM listings.listing_photos lp
JOIN listings.photos p ON p.photo_id = lp.photo_id
WHERE p.embedding IS NOT NULL
ORDER BY lp.listing_id, lp.position
"""


def parse_vector(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    return np.array([float(value) for value in text.strip("[]").split(",")])


def load_listing_attributes(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(LISTING_SQL)
        return pl.DataFrame(cur.fetchall(), schema=LISTING_ATTRIBUTE_SCHEMA, orient="row")


def load_listing_vectors(conn) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    text: dict[int, np.ndarray] = {}
    image: dict[int, np.ndarray] = {}
    with conn.cursor() as cur:
        cur.execute(VECTOR_SQL)
        for listing_id, text_vector, image_vector in cur.fetchall():
            parsed_text, parsed_image = parse_vector(text_vector), parse_vector(image_vector)
            if parsed_text is not None:
                text[listing_id] = parsed_text
            if parsed_image is not None:
                image[listing_id] = parsed_image
    return text, image


def load_photo_sets(conn) -> tuple[dict[int, list[int]], dict[int, np.ndarray]]:
    listing_photos: dict[int, list[int]] = {}
    vectors: dict[int, np.ndarray] = {}
    with conn.cursor() as cur:
        cur.execute(PHOTO_SQL)
        for listing_id, photo_id, embedding in cur.fetchall():
            listing_photos.setdefault(listing_id, []).append(photo_id)
            if photo_id not in vectors:
                vectors[photo_id] = parse_vector(embedding)
    return listing_photos, vectors


def _dimension(vectors: dict[int, np.ndarray], fallback: int = 1) -> int:
    for vector in vectors.values():
        return int(vector.size)
    return fallback


def _stack(vectors: dict[int, np.ndarray], ids, dim: int) -> np.ndarray:
    zero = np.zeros(dim)
    return np.vstack([vectors.get(int(listing_id), zero) for listing_id in ids])


def _photo_similarity(
    listing_a, listing_b, listing_photo_ids: dict[int, list[int]], photo_vectors: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    matrices: dict[int, np.ndarray] = {}
    for listing_id, photo_ids in listing_photo_ids.items():
        stacked = [photo_vectors[pid] for pid in photo_ids if pid in photo_vectors]
        if stacked:
            matrices[listing_id] = np.vstack(stacked)
    sets = {listing_id: set(ids) for listing_id, ids in listing_photo_ids.items()}

    best = np.zeros(len(listing_a))
    shared = np.zeros(len(listing_a), dtype=np.int64)
    for index, (a, b) in enumerate(zip(listing_a, listing_b)):
        a, b = int(a), int(b)
        shared[index] = len(sets.get(a, set()) & sets.get(b, set()))
        left, right = matrices.get(a), matrices.get(b)
        if left is not None and right is not None:
            best[index] = float((left @ right.T).max())
    return best, shared


def build_features(
    pairs: pl.DataFrame,
    attributes: pl.DataFrame,
    text_vectors: dict[int, np.ndarray],
    image_vectors: dict[int, np.ndarray],
    listing_photo_ids: dict[int, list[int]],
    photo_vectors: dict[int, np.ndarray],
) -> pl.DataFrame:
    joined = pairs.join(
        attributes, left_on="listing_a", right_on="listing_id", how="inner"
    ).join(attributes, left_on="listing_b", right_on="listing_id", how="inner", suffix="_b")

    listing_a = joined["listing_a"].to_numpy()
    listing_b = joined["listing_b"].to_numpy()
    text_dim, image_dim = _dimension(text_vectors), _dimension(image_vectors)
    text_cosine = np.einsum(
        "ij,ij->i", _stack(text_vectors, listing_a, text_dim), _stack(text_vectors, listing_b, text_dim)
    )
    image_mean_cosine = np.einsum(
        "ij,ij->i",
        _stack(image_vectors, listing_a, image_dim),
        _stack(image_vectors, listing_b, image_dim),
    )
    image_max_cosine, shared_photo_count = _photo_similarity(
        listing_a, listing_b, listing_photo_ids, photo_vectors
    )

    def same(column: str) -> pl.Expr:
        return (pl.col(column) == pl.col(f"{column}_b")).fill_null(False).cast(pl.Int64)

    return joined.select(
        "listing_a",
        "listing_b",
        pl.Series("text_cosine", text_cosine),
        pl.Series("image_max_cosine", image_max_cosine),
        pl.Series("image_mean_cosine", image_mean_cosine),
        pl.Series("shared_photo_count", shared_photo_count).cast(pl.Int64),
        (pl.col("asking_price_aed").log() - pl.col("asking_price_aed_b").log())
        .abs()
        .alias("abs_log_price_ratio"),
        (pl.col("size_sqm").log() - pl.col("size_sqm_b").log()).abs().alias("abs_log_size_ratio"),
        same("area_id").alias("same_area"),
        same("building_name").alias("same_building"),
        same("project_name").alias("same_project"),
        same("bedrooms").alias("bedrooms_equal"),
        (pl.col("posted_at") - pl.col("posted_at_b"))
        .dt.total_days()
        .abs()
        .cast(pl.Float64)
        .alias("days_apart"),
        same("agent_id").alias("same_agent"),
    ).select("listing_a", "listing_b", *PAIR_FEATURES)
```

- [ ] **Step 4: Implement `listings/truth.py`**

```python
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
    joined = pairs.join(truth, left_on="listing_a", right_on="listing_id", how="inner").join(
        truth, left_on="listing_b", right_on="listing_id", how="inner", suffix="_b"
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
```

- [ ] **Step 5: Run the feature tests to verify they pass**

Run: `uv run pytest tests/listings/test_listings_features.py -v`
Expected: PASS.

- [ ] **Step 6: Write the failing detection tests**

`tests/listings/test_listings_detect.py`:

```python
import json
from datetime import date

import numpy as np
import polars as pl
import pytest

from listings.config import PAIR_FEATURES, DetectConfig
from listings.detect import (
    assign_pair_split,
    baseline_decisions,
    choose_threshold,
    fit_pair_model,
    run_detection,
    write_detection,
)
from listings.load import latest_corpus_run

CONFIG = DetectConfig()


def test_pairs_are_split_by_the_later_listing():
    attributes = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "posted_at": [date(2023, 1, d) for d in (1, 2, 3, 4, 5)],
        }
    )
    pairs = pl.DataFrame({"listing_a": [1, 1, 1], "listing_b": [2, 4, 5]})
    split = assign_pair_split(pairs, attributes, CONFIG)
    assert split["split"].to_list() == ["train", "threshold", "report"]
    assert split["pair_date"].to_list() == [date(2023, 1, 2), date(2023, 1, 4), date(2023, 1, 5)]


def test_threshold_is_the_deepest_cut_that_still_meets_precision():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = np.array([True, True, True, False, True])
    # top 3 are all duplicates (precision 1.0); adding the 4th drops precision to 0.75
    assert choose_threshold(scores, labels, 0.98) == pytest.approx(0.7)
    assert choose_threshold(scores, labels, 0.75) == pytest.approx(0.5)


def test_threshold_flags_nothing_when_precision_is_unreachable():
    scores = np.array([0.9, 0.8])
    labels = np.array([False, False])
    assert choose_threshold(scores, labels, 0.98) > 0.9


def test_baseline_uses_image_similarity_alone():
    features = pl.DataFrame({"image_max_cosine": [0.96, 0.94]})
    assert baseline_decisions(features, CONFIG).tolist() == [True, False]


def test_training_without_positive_pairs_fails_loudly():
    features = pl.DataFrame({name: [0.0, 1.0] for name in PAIR_FEATURES})
    with pytest.raises(ValueError, match="no duplicate pairs"):
        fit_pair_model(features, np.array([False, False]), CONFIG)


def test_detection_scores_duplicates_above_controls(loaded_corpus):
    settings, _, _ = loaded_corpus
    conn = settings.connect()
    try:
        result = run_detection(conn, CONFIG)
    finally:
        conn.close()

    assert set(PAIR_FEATURES) <= set(result.pairs.columns)
    assert {"score", "decision", "split", "is_duplicate", "baseline_decision"} <= set(result.pairs.columns)
    assert 0.0 <= result.threshold <= 1.0
    duplicates = result.pairs.filter(pl.col("is_duplicate"))
    controls = result.pairs.filter(pl.col("same_building_control"))
    assert duplicates.height > 0 and controls.height > 0
    assert duplicates["score"].mean() > controls["score"].mean()
    assert result.stats["candidate_pairs"] == result.pairs.height


def test_detection_beats_the_single_signal_baseline_on_controls(loaded_corpus):
    settings, _, _ = loaded_corpus
    conn = settings.connect()
    try:
        result = run_detection(conn, CONFIG)
    finally:
        conn.close()
    controls = result.pairs.filter(pl.col("same_building_control"))
    model_flags = controls["decision"].sum()
    baseline_flags = controls["baseline_decision"].sum()
    assert model_flags <= baseline_flags, (
        f"the model flagged {model_flags} same-building controls, the image-only baseline "
        f"{baseline_flags} — multi-signal agreement is supposed to help here"
    )


def test_write_detection_stores_flagged_pairs_with_their_signals(loaded_corpus):
    settings, _, corpus_run_id = loaded_corpus
    conn = settings.connect()
    try:
        result = run_detection(conn, CONFIG)
        detect_run_id = write_detection(conn, result, corpus_run_id)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT listing_a, listing_b, score, decision, signals FROM listings.duplicate_pairs "
                "WHERE detect_run_id = %s",
                (detect_run_id,),
            )
            rows = cur.fetchall()
            cur.execute("SELECT corpus_run_id, threshold FROM listings.detect_runs WHERE detect_run_id = %s", (detect_run_id,))
            run = cur.fetchone()
    finally:
        conn.close()

    flagged = result.pairs.filter(pl.col("decision"))
    assert len(rows) == flagged.height
    assert all(row[0] < row[1] for row in rows)
    assert all(row[3] is True for row in rows)
    if rows:
        signals = rows[0][4]
        signals = json.loads(signals) if isinstance(signals, str) else signals
        assert set(signals) == set(PAIR_FEATURES)
    assert run[0] == corpus_run_id
    assert run[1] == pytest.approx(result.threshold)
```

- [ ] **Step 7: Run the detection tests to verify they fail**

Run: `uv run pytest tests/listings/test_listings_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'listings.detect'`.

- [ ] **Step 8: Implement `listings/detect.py`**

Only flagged pairs are written to `duplicate_pairs` — the table exists so the API
and UI can explain a flag, and evaluation works from the in-memory result, which
holds every scored candidate.

```python
"""Split pairs by time, fit the pair model, pick a threshold, write decisions."""

import json
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ingestion.load import copy_frame
from listings.candidates import fetch_candidates
from listings.config import PAIR_FEATURES, DetectConfig
from listings.features import (
    build_features,
    load_listing_attributes,
    load_listing_vectors,
    load_photo_sets,
)
from listings.truth import label_pairs, load_truth


@dataclass(frozen=True)
class DetectionResult:
    pairs: pl.DataFrame
    threshold: float
    model: Pipeline
    feature_names: tuple[str, ...]
    stats: dict[str, float]


def assign_pair_split(
    pairs: pl.DataFrame, attributes: pl.DataFrame, config: DetectConfig
) -> pl.DataFrame:
    """A pair belongs to the split of its later listing: 'is this new post a repost?'"""
    posted = attributes.select("listing_id", "posted_at")
    joined = pairs.join(posted, left_on="listing_a", right_on="listing_id", how="inner").join(
        posted, left_on="listing_b", right_on="listing_id", how="inner", suffix="_b"
    )
    dates = attributes["posted_at"].sort()
    train_end = dates[min(int(dates.len() * config.train_share), dates.len() - 1)]
    threshold_end = dates[
        min(int(dates.len() * (config.train_share + config.threshold_share)), dates.len() - 1)
    ]
    later = pl.max_horizontal("posted_at", "posted_at_b")
    split = (
        pl.when(later <= train_end)
        .then(pl.lit("train"))
        .when(later <= threshold_end)
        .then(pl.lit("threshold"))
        .otherwise(pl.lit("report"))
    )
    return joined.with_columns(later.alias("pair_date"), split.alias("split")).drop(
        "posted_at", "posted_at_b"
    )


def fit_pair_model(features: pl.DataFrame, labels: np.ndarray, config: DetectConfig) -> Pipeline:
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0:
        raise ValueError("training split has no duplicate pairs to learn from")
    if (~labels).sum() == 0:
        raise ValueError("training split has no non-duplicate pairs to learn from")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=config.seed
                ),
            ),
        ]
    )
    model.fit(features.select(PAIR_FEATURES).to_numpy(), labels)
    return model


def choose_threshold(scores: np.ndarray, labels: np.ndarray, target_precision: float) -> float:
    """The lowest score at which running precision still meets the target."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if scores.size == 0:
        return 1.0
    order = np.argsort(-scores)
    ranked_scores, ranked_labels = scores[order], labels[order]
    precision = np.cumsum(ranked_labels) / np.arange(1, scores.size + 1)
    acceptable = np.flatnonzero(precision >= target_precision)
    if acceptable.size == 0:
        return float(ranked_scores[0]) + 1e-9  # nothing reaches the target: flag nothing
    return float(ranked_scores[acceptable[-1]])


def baseline_decisions(features: pl.DataFrame, config: DetectConfig) -> np.ndarray:
    """The single-signal rule the multi-signal model is measured against."""
    return (features["image_max_cosine"].to_numpy() >= config.baseline_image_cosine)


def run_detection(conn, config: DetectConfig) -> DetectionResult:
    started = time.perf_counter()
    pairs, candidate_stats = fetch_candidates(conn, config)
    attributes = load_listing_attributes(conn)
    text_vectors, image_vectors = load_listing_vectors(conn)
    listing_photo_ids, photo_vectors = load_photo_sets(conn)

    features = build_features(
        pairs.select("listing_a", "listing_b"),
        attributes,
        text_vectors,
        image_vectors,
        listing_photo_ids,
        photo_vectors,
    )
    features = assign_pair_split(features, attributes, config)
    labelled = features.join(
        label_pairs(features.select("listing_a", "listing_b"), load_truth(conn)),
        on=["listing_a", "listing_b"],
        how="left",
    ).with_columns(
        pl.col("is_duplicate").fill_null(False), pl.col("same_building_control").fill_null(False)
    )

    train = labelled.filter(pl.col("split") == "train")
    model = fit_pair_model(train, train["is_duplicate"].to_numpy(), config)
    scores = model.predict_proba(labelled.select(PAIR_FEATURES).to_numpy())[:, 1]
    labelled = labelled.with_columns(pl.Series("score", scores))

    validation = labelled.filter(pl.col("split") == "threshold")
    threshold = choose_threshold(
        validation["score"].to_numpy(), validation["is_duplicate"].to_numpy(), config.target_precision
    )
    labelled = labelled.with_columns(
        (pl.col("score") >= threshold).alias("decision"),
        pl.Series("baseline_decision", baseline_decisions(labelled, config)),
    )
    stats = {
        "candidate_pairs": float(labelled.height),
        "text_pairs": float(candidate_stats.text_pairs),
        "image_pairs": float(candidate_stats.image_pairs),
        "shared_photo_pairs": float(candidate_stats.shared_photo_pairs),
        "retrieval_seconds": candidate_stats.seconds,
        "detect_seconds": time.perf_counter() - started,
        "train_pairs": float(train.height),
        "flagged": float(labelled["decision"].sum()),
    }
    return DetectionResult(labelled, threshold, model, PAIR_FEATURES, stats)


def write_detection(conn, result: DetectionResult, corpus_run_id: int) -> int:
    """Record the run and store the flagged pairs with the signals behind each decision."""
    flagged = result.pairs.filter(pl.col("decision"))
    signals = [
        json.dumps({name: row[name] for name in PAIR_FEATURES})
        for row in flagged.iter_rows(named=True)
    ]
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO listings.detect_runs (corpus_run_id, threshold, metrics) "
            "VALUES (%s, %s, %s::jsonb) RETURNING detect_run_id",
            (corpus_run_id, result.threshold, json.dumps(result.stats)),
        )
        detect_run_id = cur.fetchone()[0]
        if flagged.height:
            frame = flagged.select(
                pl.lit(detect_run_id, dtype=pl.Int64).alias("detect_run_id"),
                "listing_a",
                "listing_b",
                "score",
                "decision",
                pl.Series("signals", signals),
            )
            copy_frame(cur, "listings.duplicate_pairs", frame)
    return detect_run_id
```

- [ ] **Step 9: Run every listings test**

Run: `uv run pytest tests/listings/ -v`
Expected: PASS.

If `test_detection_beats_the_single_signal_baseline_on_controls` fails, report the
two counts as a concern rather than adjusting the threshold or the features — a
model that flags controls more often than the crude baseline is a finding.

- [ ] **Step 10: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add listings/features.py listings/truth.py listings/detect.py tests/listings/test_listings_features.py tests/listings/test_listings_detect.py
git commit -m "$(cat <<'EOF'
feat(listings): pair features, ground truth and the multi-signal duplicate model

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Fraud flags

**Files:**
- Create: `listings/fraud.py`, `tests/listings/test_listings_fraud.py`

**Interfaces:**
- Consumes: `listings.config.DetectConfig`, `listings.detect.DetectionResult`, `ingestion.load.copy_frame`, and the Phase 3 champion (`models:/dubimator-price@champion`) through `mlflow.pyfunc`
- Produces (`listings.fraud`):
  - `FLAGS = ("bait_price", "photo_reuse", "inconsistent_relist")`
  - `FRAUD_LISTING_SQL: str`, `load_fraud_attributes(conn) -> pl.DataFrame` (no label columns)
  - `load_price_predictor(uri) -> object | None` — returns `None` (with a warning) when the model can't be loaded
  - `price_request(row) -> dict` — a listing row as a `PriceRequest` payload
  - `bait_price_flags(attributes, predictor, config) -> tuple[pl.DataFrame, dict[str, float]]`
  - `photo_reuse_flags(attributes, config) -> pl.DataFrame`
  - `inconsistent_relist_flags(attributes, flagged_pairs, config) -> pl.DataFrame`
  - `run_fraud_checks(conn, flagged_pairs, config, predictor=...) -> FraudResult` (`flags`, `stats`)
  - `write_fraud_flags(conn, result, detect_run_id) -> int`

**Detection must survive a missing price model.** If MLflow is down or nothing is
registered, `load_price_predictor` logs a warning and returns `None`; the bait
check is skipped, `stats["bait_price_skipped"] = 1.0`, and the other two flags
still run.

**Speed:** the price model is called once per listing (about 10 ms each, so
roughly 3–6 minutes for 20,000). The CLI logs progress every 2,000 listings.

- [ ] **Step 1: Write the failing tests**

`tests/listings/test_listings_fraud.py`:

```python
import json

import polars as pl
import pytest

from listings.config import DetectConfig
from listings.fraud import (
    bait_price_flags,
    inconsistent_relist_flags,
    photo_reuse_flags,
    price_request,
    run_fraud_checks,
    write_fraud_flags,
)

CONFIG = DetectConfig()

ATTRIBUTES = pl.DataFrame(
    {
        "listing_id": [1, 2, 3, 4],
        "asking_price_aed": [1_000_000.0, 400_000.0, 2_000_000.0, 2_100_000.0],
        "area_id": [10, 10, 20, 20],
        "area_name": ["Marsa Dubai"] * 2 + ["Hadaeq Sheikh Mohammed Bin Rashid"] * 2,
        "building_name": ["Marina Gate 1", "Marina Gate 1", None, None],
        "project_name": ["Marina Gate", "Marina Gate", "Dubai Hills", "Dubai Hills"],
        "property_type": ["unit", "unit", "villa", "villa"],
        "property_sub_type": ["Flat", "Flat", None, "Villa"],
        "reg_type": ["ready", "ready", "ready", "off_plan"],
        "size_sqm": [100.0, 100.0, 520.0, 300.0],
        "bedrooms": [2, 2, None, None],
        "photo_set_id": [7, 7, 8, 9],
    }
)


class StubPredictor:
    """Estimates 1,000,000 with an 80% range of 900,000-1,100,000, whatever it is asked."""

    def __init__(self, unsupported: set[int] | None = None):
        self.calls = []
        self.unsupported = unsupported or set()

    def predict_one(self, request):
        from listings.fraud import PriceInputError

        self.calls.append(request)
        if len(self.calls) in self.unsupported:
            raise PriceInputError("status", "unsupported combination")
        return type(
            "Estimate",
            (),
            {"estimate_aed": 1_000_000.0, "range_80": (900_000.0, 1_100_000.0)},
        )()


def test_price_request_maps_listing_shapes():
    rows = ATTRIBUTES.to_dicts()
    flat = price_request(rows[0])
    assert flat["property_kind"] == "apartment" and flat["size_basis"] == "built_up"
    assert flat["status"] == "ready" and flat["area_id"] == 10
    assert flat["size_sqm"] == 100.0 and flat["bedrooms"] == 2
    assert flat["building"] == "Marina Gate 1" and flat["project"] == "Marina Gate"

    plot_villa = price_request(rows[2])
    assert plot_villa["property_kind"] == "villa" and plot_villa["size_basis"] == "plot"
    assert "bedrooms" not in plot_villa  # unknown bedrooms are left out, not sent as None

    built_villa = price_request(rows[3])
    assert built_villa["size_basis"] == "built_up" and built_villa["status"] == "off_plan"


def test_bait_price_flags_only_the_cheap_listing():
    predictor = StubPredictor()
    flags, stats = bait_price_flags(ATTRIBUTES, predictor, CONFIG)
    assert flags["listing_id"].to_list() == [2]
    detail = json.loads(flags["detail"][0])
    assert detail["asking_price_aed"] == 400_000.0
    assert detail["range_80_low"] == 900_000.0
    assert stats["bait_price_checked"] == 4.0
    assert stats["bait_price_skipped"] == 0.0


def test_listings_the_price_model_cannot_price_are_skipped_not_failed():
    predictor = StubPredictor(unsupported={2})
    flags, stats = bait_price_flags(ATTRIBUTES, predictor, CONFIG)
    assert 2 not in flags["listing_id"].to_list()
    assert stats["bait_price_unsupported"] == 1.0


def test_photo_reuse_needs_several_areas():
    attributes = pl.DataFrame(
        {
            "listing_id": list(range(1, 8)),
            "photo_set_id": [5, 5, 5, 5, 5, 6, 6],
            "area_id": [1, 2, 3, 4, 5, 1, 2],
        }
    )
    flags = photo_reuse_flags(attributes, CONFIG)
    assert sorted(flags["listing_id"].to_list()) == [1, 2, 3, 4, 5]
    assert json.loads(flags["detail"][0])["areas"] == 5
    assert json.loads(flags["detail"][0])["photo_set_id"] == 5


def test_inconsistent_relist_flags_clusters_whose_prices_disagree():
    attributes = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "asking_price_aed": [1_000_000.0, 1_050_000.0, 1_400_000.0, 2_000_000.0, 2_020_000.0],
        }
    )
    flagged = pl.DataFrame({"listing_a": [1, 2, 4], "listing_b": [2, 3, 5]})
    flags = inconsistent_relist_flags(attributes, flagged, CONFIG)
    # 1-2-3 is one cluster spanning 1.0M-1.4M (40%); 4-5 spans 1% and is fine
    assert sorted(flags["listing_id"].to_list()) == [1, 2, 3]
    detail = json.loads(flags["detail"][0])
    assert detail["spread"] == pytest.approx(0.4)
    assert detail["cluster_size"] == 3


def test_a_missing_price_model_skips_bait_but_keeps_the_rest(loaded_corpus, caplog):
    settings, _, _ = loaded_corpus
    conn = settings.connect()
    try:
        result = run_fraud_checks(
            conn, pl.DataFrame({"listing_a": [], "listing_b": []}), CONFIG, predictor=None
        )
    finally:
        conn.close()
    assert result.stats["bait_price_skipped"] == 1.0
    assert "bait_price" not in result.flags["flag"].to_list()
    assert set(result.flags["flag"].to_list()) <= {"photo_reuse", "inconsistent_relist"}


def test_write_fraud_flags_stores_them_per_run(loaded_corpus):
    settings, _, corpus_run_id = loaded_corpus
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO listings.detect_runs (corpus_run_id, threshold) VALUES (%s, 0.5) "
                "RETURNING detect_run_id",
                (corpus_run_id,),
            )
            detect_run_id = cur.fetchone()[0]
        result = run_fraud_checks(
            conn, pl.DataFrame({"listing_a": [], "listing_b": []}), CONFIG, predictor=None
        )
        written = write_fraud_flags(conn, result, detect_run_id)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*), count(DISTINCT flag) FROM listings.fraud_flags WHERE detect_run_id=%s",
                (detect_run_id,),
            )
            count, flags = cur.fetchone()
    finally:
        conn.close()
    assert count == written == result.flags.height
    assert flags >= 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/listings/test_listings_fraud.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'listings.fraud'`.

- [ ] **Step 3: Implement `listings/fraud.py`**

```python
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
    prices = dict(
        zip(attributes["listing_id"].to_list(), attributes["asking_price_aed"].to_list())
    )
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/listings/ -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add listings/fraud.py tests/listings/test_listings_fraud.py
git commit -m "$(cat <<'EOF'
feat(listings): bait-price, photo-reuse and inconsistent-relist fraud flags

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Evaluation and MLflow logging

**Files:**
- Create: `listings/evaluate.py`, `tests/listings/test_listings_evaluate.py`
- Modify: `listings/truth.py` (add `load_duplicate_truth`)

**Interfaces:**
- Consumes: `listings.detect.DetectionResult`, `listings.fraud.FraudResult`, `listings.candidates.brute_force_pairs`, `listings.truth`, the `temp_mlflow` fixture from `tests/models/price/conftest.py` (see the note below)
- Produces (`listings.truth`):
  - `load_duplicate_truth(conn) -> pl.DataFrame` — every planted clone → source pair as `listing_a`, `listing_b`, `pattern` (`exact_repost` / `reworded` / `edited_photo`), derived by comparing each clone with its source
- Produces (`listings.evaluate`):
  - `pair_metrics(labels, decisions) -> dict[str, float]` — precision, recall, f1, counts
  - `stock_photo_control_pairs(pairs, attributes) -> pl.Series` — boolean mask: same photo set, different area, not duplicates
  - `evaluate_detection(result, truth_pairs, attributes, config) -> dict[str, float]`
  - `evaluate_fraud(fraud_result, truth, config) -> dict[str, float]`
  - `save_pr_curve(scores, labels, path) -> Path`
  - `log_run(metrics, params, artifacts, config, tracking_uri=None, artifact_location=None) -> str` — returns the MLflow run id

**The `temp_mlflow` fixture already exists** in `tests/models/price/conftest.py` but
pytest fixtures are per-directory. Copy that fixture into
`tests/listings/conftest.py` (same body) so listings tests never write to the real
server. Keep its `import mlflow` inside the function body.

**Recall's denominator** is the planted clone → source pairs. Two clones of one
source are also true duplicates and count as correct when flagged (they are marked
duplicates by `label_pairs`), but they are not in the denominator, since nothing
planted them directly.

- [ ] **Step 1: Write the failing tests**

`tests/listings/test_listings_evaluate.py`:

```python
import polars as pl
import pytest

from listings.config import DetectConfig
from listings.evaluate import (
    evaluate_detection,
    evaluate_fraud,
    pair_metrics,
    save_pr_curve,
    stock_photo_control_pairs,
)

CONFIG = DetectConfig()


def test_pair_metrics_are_hand_checkable():
    labels = [True, True, True, False, False]
    decisions = [True, True, False, True, False]
    metrics = pair_metrics(labels, decisions)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert (metrics["true_positives"], metrics["false_positives"]) == (2.0, 1.0)
    assert (metrics["false_negatives"], metrics["pairs"]) == (1.0, 5.0)


def test_pair_metrics_handle_nothing_flagged():
    metrics = pair_metrics([True, False], [False, False])
    assert metrics["precision"] == 0.0 and metrics["recall"] == 0.0 and metrics["f1"] == 0.0


def test_stock_photo_controls_are_same_set_different_area_and_not_duplicates():
    attributes = pl.DataFrame(
        {"listing_id": [1, 2, 3, 4], "photo_set_id": [5, 5, 5, 6], "area_id": [10, 20, 10, 30]}
    )
    pairs = pl.DataFrame(
        {
            "listing_a": [1, 1, 1],
            "listing_b": [2, 3, 4],
            "is_duplicate": [False, False, False],
        }
    )
    mask = stock_photo_control_pairs(pairs, attributes).to_list()
    assert mask == [True, False, False]  # same set + different area; same area; different set


def test_evaluate_detection_reports_every_slice():
    pairs = pl.DataFrame(
        {
            "listing_a": [1, 1, 2, 3],
            "listing_b": [2, 3, 3, 4],
            "split": ["report"] * 4,
            "score": [0.9, 0.8, 0.2, 0.1],
            "decision": [True, True, False, False],
            "baseline_decision": [True, True, True, False],
            "is_duplicate": [True, False, False, False],
            "same_building_control": [False, True, True, False],
        }
    )
    attributes = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4],
            "photo_set_id": [5, 5, 6, 6],
            "area_id": [10, 20, 30, 30],
        }
    )
    truth = pl.DataFrame(
        {"listing_a": [1, 5], "listing_b": [2, 6], "pattern": ["exact_repost", "reworded"]}
    )
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {"detect_seconds": 1.0}})(),
        truth,
        attributes,
        CONFIG,
    )
    assert metrics["report.precision"] == pytest.approx(0.5)  # 1 of 2 flagged pairs is a duplicate
    assert metrics["report.recall"] == pytest.approx(1.0)  # the one report-split duplicate is caught
    assert metrics["retrieval.recall"] == pytest.approx(0.5)  # pair (5,6) was never retrieved
    assert metrics["control.same_building.model_fp_rate"] == pytest.approx(0.5)
    assert metrics["control.same_building.baseline_fp_rate"] == pytest.approx(1.0)
    assert metrics["pattern.exact_repost.recall"] == pytest.approx(1.0)
    assert metrics["pattern.reworded.recall"] == pytest.approx(0.0)
    assert metrics["threshold"] == 0.5
    assert "report.pr_auc" in metrics


def test_evaluate_fraud_scores_bait_against_its_label():
    flags = pl.DataFrame(
        {
            "listing_id": [1, 2],
            "flag": ["bait_price", "bait_price"],
            "detail": ["{}", "{}"],
        }
    )
    truth = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4],
            "dup_group_id": [None, None, None, None],
            "control_group_id": [None, None, None, None],
            "fraud_label": ["bait_price", None, "bait_price", None],
        },
        schema={
            "listing_id": pl.Int64, "dup_group_id": pl.Int64,
            "control_group_id": pl.Utf8, "fraud_label": pl.Utf8,
        },
    )  # fmt: skip
    metrics = evaluate_fraud(
        type("F", (), {"flags": flags, "stats": {"bait_price_skipped": 0.0}})(), truth, CONFIG
    )
    assert metrics["fraud.bait_price.precision"] == pytest.approx(0.5)
    assert metrics["fraud.bait_price.recall"] == pytest.approx(0.5)
    assert metrics["fraud.bait_price.flagged"] == 2.0


def test_pr_curve_is_written(tmp_path):
    path = save_pr_curve([0.9, 0.2, 0.7], [True, False, True], tmp_path / "pr.png")
    assert path.is_file() and path.stat().st_size > 0


def test_duplicate_truth_reads_the_planted_patterns(loaded_corpus):
    from listings.truth import load_duplicate_truth

    settings, corpus, _ = loaded_corpus
    conn = settings.connect()
    try:
        truth = load_duplicate_truth(conn)
    finally:
        conn.close()
    counts = dict(truth.group_by("pattern").len().iter_rows())
    assert counts == {
        "exact_repost": corpus.counts["exact_repost"],
        "reworded": corpus.counts["reworded"],
        "edited_photo": corpus.counts["edited_photo"],
    }
    assert (truth["listing_a"] < truth["listing_b"]).all()


def test_log_run_writes_to_the_temporary_store(temp_mlflow, tmp_path):
    import mlflow

    from listings.evaluate import log_run

    artifact = tmp_path / "pr.png"
    artifact.write_bytes(b"png")
    run_id = log_run(
        {"report.precision": 0.99, "report.recall": 0.8},
        {"threshold": 0.5, "corpus_run_id": 1},
        [artifact],
        CONFIG,
        tracking_uri=temp_mlflow["tracking_uri"],
        artifact_location=temp_mlflow["artifact_location"],
    )
    run = mlflow.get_run(run_id)
    assert run.data.metrics["report.precision"] == pytest.approx(0.99)
    assert run.data.params["threshold"] == "0.5"
    assert "pr.png" in {item.path for item in mlflow.MlflowClient().list_artifacts(run_id)}
```

- [ ] **Step 2: Copy the `temp_mlflow` fixture and run the tests to verify they fail**

Copy the `temp_mlflow` fixture verbatim from `tests/models/price/conftest.py` into
`tests/listings/conftest.py` (it imports `mlflow` inside the function body — keep
that).

Run: `uv run pytest tests/listings/test_listings_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'listings.evaluate'`.

- [ ] **Step 3: Add `load_duplicate_truth` to `listings/truth.py`**

```python
DUPLICATE_TRUTH_SQL = """
SELECT clone.listing_id AS clone_id,
       clone.dup_group_id AS source_id,
       (clone.description = source.description) AS same_text,
       EXISTS (
           SELECT 1
           FROM listings.listing_photos lp
           JOIN listings.photos p ON p.photo_id = lp.photo_id
           WHERE lp.listing_id = clone.listing_id AND p.variant_of IS NOT NULL
       ) AS edited_photos
FROM listings.listings clone
JOIN listings.listings source ON source.listing_id = clone.dup_group_id
WHERE clone.dup_group_id IS NOT NULL
"""


def load_duplicate_truth(conn) -> pl.DataFrame:
    """Planted clone -> source pairs, with the pattern read off the content."""
    with conn.cursor() as cur:
        cur.execute(DUPLICATE_TRUTH_SQL)
        rows = cur.fetchall()
    pattern = []
    listing_a, listing_b = [], []
    for clone_id, source_id, same_text, edited_photos in rows:
        listing_a.append(min(clone_id, source_id))
        listing_b.append(max(clone_id, source_id))
        if edited_photos:
            pattern.append("edited_photo")
        elif same_text:
            pattern.append("exact_repost")
        else:
            pattern.append("reworded")
    return pl.DataFrame(
        {"listing_a": listing_a, "listing_b": listing_b, "pattern": pattern},
        schema={"listing_a": pl.Int64, "listing_b": pl.Int64, "pattern": pl.Utf8},
    )
```

- [ ] **Step 4: Implement `listings/evaluate.py`**

```python
"""Measure the detector against ground truth and log the run to MLflow."""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from listings.config import DetectConfig
from listings.truth import load_duplicate_truth  # noqa: F401  (re-exported for the CLI)


def pair_metrics(labels, decisions) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    decisions = np.asarray(decisions, dtype=bool)
    true_positives = float(np.sum(labels & decisions))
    false_positives = float(np.sum(~labels & decisions))
    false_negatives = float(np.sum(labels & ~decisions))
    precision = true_positives / (true_positives + false_positives) if decisions.any() else 0.0
    recall = true_positives / (true_positives + false_negatives) if labels.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "pairs": float(labels.size),
    }


def stock_photo_control_pairs(pairs: pl.DataFrame, attributes: pl.DataFrame) -> pl.Series:
    """Unrelated listings that happen to share an agency photo set: must not be flagged."""
    lookup = attributes.select("listing_id", "photo_set_id", "area_id")
    joined = pairs.join(lookup, left_on="listing_a", right_on="listing_id", how="left").join(
        lookup, left_on="listing_b", right_on="listing_id", how="left", suffix="_b"
    )
    mask = (
        (pl.col("photo_set_id") == pl.col("photo_set_id_b"))
        & (pl.col("area_id") != pl.col("area_id_b"))
        & ~pl.col("is_duplicate")
    ).fill_null(False)
    return joined.select(mask.alias("stock_photo_control"))["stock_photo_control"]


def _control_rates(pairs: pl.DataFrame, mask: pl.Series, name: str) -> dict[str, float]:
    controls = pairs.filter(mask)
    if controls.is_empty():
        return {f"control.{name}.pairs": 0.0}
    return {
        f"control.{name}.pairs": float(controls.height),
        f"control.{name}.model_fp_rate": float(controls["decision"].mean()),
        f"control.{name}.baseline_fp_rate": float(controls["baseline_decision"].mean()),
    }


def evaluate_detection(result, truth_pairs: pl.DataFrame, attributes: pl.DataFrame, config: DetectConfig) -> dict[str, float]:
    pairs = result.pairs
    report = pairs.filter(pl.col("split") == "report")
    metrics = {f"report.{key}": value for key, value in pair_metrics(report["is_duplicate"], report["decision"]).items()}
    metrics |= {
        f"report.baseline_{key}": value
        for key, value in pair_metrics(report["is_duplicate"], report["baseline_decision"]).items()
    }
    if report["is_duplicate"].any() and not report["is_duplicate"].all():
        metrics["report.pr_auc"] = float(
            average_precision_score(report["is_duplicate"].to_numpy(), report["score"].to_numpy())
        )
    else:
        metrics["report.pr_auc"] = float("nan")

    retrieved = set(zip(pairs["listing_a"].to_list(), pairs["listing_b"].to_list()))
    planted = list(zip(truth_pairs["listing_a"].to_list(), truth_pairs["listing_b"].to_list()))
    metrics["retrieval.recall"] = (
        float(sum(pair in retrieved for pair in planted) / len(planted)) if planted else 0.0
    )

    flagged = {
        (a, b)
        for a, b, decision in zip(
            pairs["listing_a"].to_list(), pairs["listing_b"].to_list(), pairs["decision"].to_list()
        )
        if decision
    }
    for pattern in ("exact_repost", "reworded", "edited_photo"):
        subset = [
            (a, b)
            for (a, b), kind in zip(planted, truth_pairs["pattern"].to_list())
            if kind == pattern
        ]
        metrics[f"pattern.{pattern}.recall"] = (
            float(sum(pair in flagged for pair in subset) / len(subset)) if subset else 0.0
        )

    metrics |= _control_rates(pairs, pairs["same_building_control"], "same_building")
    metrics |= _control_rates(pairs, stock_photo_control_pairs(pairs, attributes), "stock_photo")
    metrics["threshold"] = float(result.threshold)
    metrics |= {f"stats.{key}": float(value) for key, value in result.stats.items()}
    return metrics


def evaluate_fraud(fraud_result, truth: pl.DataFrame, config: DetectConfig) -> dict[str, float]:
    metrics = {f"stats.{key}": float(value) for key, value in fraud_result.stats.items()}
    flagged = set(
        fraud_result.flags.filter(pl.col("flag") == "bait_price")["listing_id"].to_list()
    )
    actual = set(truth.filter(pl.col("fraud_label") == "bait_price")["listing_id"].to_list())
    hits = len(flagged & actual)
    metrics["fraud.bait_price.flagged"] = float(len(flagged))
    metrics["fraud.bait_price.precision"] = float(hits / len(flagged)) if flagged else 0.0
    metrics["fraud.bait_price.recall"] = float(hits / len(actual)) if actual else 0.0
    for name in ("photo_reuse", "inconsistent_relist"):
        metrics[f"fraud.{name}.flagged"] = float(
            fraud_result.flags.filter(pl.col("flag") == name).height
        )
    return metrics


def save_pr_curve(scores, labels, path: Path) -> Path:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    plt.switch_backend("Agg")
    precision, recall, _ = precision_recall_curve(np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float))
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision)
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title("Duplicate detection: precision vs recall")
    axis.grid(alpha=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def log_run(
    metrics: dict[str, float],
    params: dict,
    artifacts,
    config: DetectConfig,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> str:
    import math

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if artifact_location and mlflow.get_experiment_by_name(config.experiment) is None:
        mlflow.create_experiment(config.experiment, artifact_location=artifact_location)
    mlflow.set_experiment(config.experiment)
    with mlflow.start_run(run_name="listing-dedup") as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items() if math.isfinite(float(value))}
        )
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return run.info.run_id
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/listings/ -v`
Expected: PASS. `git status --short` must show no `mlruns/` directory.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add listings/evaluate.py listings/truth.py tests/listings/conftest.py tests/listings/test_listings_evaluate.py
git commit -m "$(cat <<'EOF'
feat(listings): evaluate duplicates and fraud against ground truth, log to MLflow

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Command line and the end-to-end test

**Files:**
- Create: `listings/__main__.py`, `tests/listings/test_listings_integration.py`
- Modify: `listings/load.py` (add `read_corpus_for_embedding`)

**Interfaces:**
- Consumes: every module from Tasks 2–9; `ingestion.config.DbSettings`; `ingestion.pipeline.run_pipeline` (test only); the `pg_test_db`, `temp_mlflow` and `photo_pool` fixtures
- Produces (`listings.load`):
  - `read_corpus_for_embedding(conn) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]` — photos (`photo_id`, `path`), listings (`listing_id`, `title`, `description`), listing_photos (`listing_id`, `photo_id`, `position`)
- Produces (`listings.__main__`):
  - `main(argv=None) -> int` with `build`, `embed`, `detect` and `evaluate`
  - `build [--listings N] [--seed N] [--data-dir PATH]`
  - `embed [--device auto|cuda|cpu] [--data-dir PATH] [--fake]`
  - `detect` — scores candidates, writes `duplicate_pairs` and `fraud_flags`
  - `evaluate` — re-scores in memory, prints the metric table, logs to MLflow
  - exit codes: 0 success, 1 error

`--fake` on `embed` uses the deterministic embedder instead of the real models: it
is what the integration test uses, and it makes the pipeline runnable on a machine
with no model download.

- [ ] **Step 1: Write the failing integration test**

`tests/listings/test_listings_integration.py`:

```python
import dataclasses
from pathlib import Path

import mlflow
import polars as pl

from ingestion.pipeline import run_pipeline
from listings.config import CorpusConfig, DetectConfig
from listings.detect import run_detection, write_detection
from listings.embed import FakeEmbedder, embed_listings, embed_photos
from listings.evaluate import evaluate_detection, evaluate_fraud, log_run, save_pr_curve
from listings.fraud import run_fraud_checks, write_fraud_flags
from listings.generate import generate_corpus, load_areas, load_sales, render_variants
from listings.load import (
    create_vector_indexes,
    load_corpus,
    read_corpus_for_embedding,
    write_listing_embeddings,
    write_photo_embeddings,
)
from listings.truth import load_duplicate_truth, load_truth

DLD_FIXTURE = Path("tests/fixtures/price_sample.csv")
SMALL_CORPUS = CorpusConfig(
    n_listings=400,
    n_base=320,
    n_exact_repost=40,
    n_reworded=20,
    n_edited_photo=20,
    n_bait_price=20,
    n_price_shifted_reposts=10,
    n_from_busy_buildings=40,
    n_stock_sets=2,
    n_agents=30,
    stock_min_areas=2,
)
SMALL_DETECT = dataclasses.replace(DetectConfig(), text_top_k=10, photo_top_k=10)


def test_the_whole_pipeline_runs_on_real_dld_rows(pg_test_db, temp_mlflow, photo_pool, tmp_path):
    run_pipeline(DLD_FIXTURE, pg_test_db)  # real DLD rows -> dld.market_sales

    sales = load_sales(pg_test_db, SMALL_CORPUS)
    areas = load_areas(pg_test_db)
    assert sales.height > SMALL_CORPUS.n_base, "fixture should hold enough home sales"

    data_dir = photo_pool.root.parent
    corpus = generate_corpus(sales, areas, photo_pool.set_ids, SMALL_CORPUS)
    render_variants(corpus, photo_pool, SMALL_CORPUS, data_dir / "variants")
    corpus_run_id = load_corpus(pg_test_db, corpus, SMALL_CORPUS.seed, photo_pool.archive_sha256)

    conn = pg_test_db.connect()
    try:
        photos, listings, listing_photos = read_corpus_for_embedding(conn)
        embedder = FakeEmbedder()
        photo_frame, photo_vectors = embed_photos(photos, embedder, data_dir)
        listing_frame = embed_listings(listings, listing_photos, photo_vectors, embedder)
        write_photo_embeddings(conn, photo_frame)
        write_listing_embeddings(conn, listing_frame)
        create_vector_indexes(conn)
        conn.commit()

        result = run_detection(conn, SMALL_DETECT)
        detect_run_id = write_detection(conn, result, corpus_run_id)
        fraud = run_fraud_checks(conn, result.pairs.filter(pl.col("decision")), SMALL_DETECT, predictor=None)
        write_fraud_flags(conn, fraud, detect_run_id)
        conn.commit()

        truth_pairs = load_duplicate_truth(conn)
        truth = load_truth(conn)
        from listings.features import load_listing_attributes

        attributes = load_listing_attributes(conn)
    finally:
        conn.close()

    metrics = evaluate_detection(result, truth_pairs, attributes, SMALL_DETECT)
    metrics |= evaluate_fraud(fraud, truth, SMALL_DETECT)

    assert metrics["retrieval.recall"] >= 0.9, metrics["retrieval.recall"]
    assert metrics["report.recall"] > 0.0
    assert metrics["report.precision"] >= 0.9, metrics["report.precision"]
    assert metrics["pattern.exact_repost.recall"] >= 0.9
    assert metrics["control.same_building.model_fp_rate"] <= metrics.get(
        "control.same_building.baseline_fp_rate", 1.0
    )
    assert metrics["stats.bait_price_skipped"] == 1.0  # no price model in the test environment

    curve = save_pr_curve(result.pairs["score"], result.pairs["is_duplicate"], tmp_path / "pr.png")
    run_id = log_run(
        metrics,
        {"corpus_run_id": corpus_run_id, "threshold": result.threshold},
        [curve],
        SMALL_DETECT,
        tracking_uri=temp_mlflow["tracking_uri"],
        artifact_location=temp_mlflow["artifact_location"],
    )
    assert mlflow.get_run(run_id).data.metrics["retrieval.recall"] >= 0.9

    conn = pg_test_db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM listings.duplicate_pairs WHERE detect_run_id=%s", (detect_run_id,))
            flagged = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM listings.fraud_flags WHERE detect_run_id=%s", (detect_run_id,))
            flags = cur.fetchone()[0]
    finally:
        conn.close()
    assert flagged == result.pairs.filter(pl.col("decision")).height > 0
    assert flags == fraud.flags.height
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/listings/test_listings_integration.py -v`
Expected: FAIL with `ImportError: cannot import name 'read_corpus_for_embedding'`.

- [ ] **Step 3: Add `read_corpus_for_embedding` to `listings/load.py`**

```python
EMBEDDING_INPUT_SQL = {
    "photos": "SELECT photo_id, path FROM listings.photos ORDER BY photo_id",
    "listings": "SELECT listing_id, title, description FROM listings.listings ORDER BY listing_id",
    "listing_photos": (
        "SELECT listing_id, photo_id, position FROM listings.listing_photos "
        "ORDER BY listing_id, position"
    ),
}
EMBEDDING_INPUT_SCHEMA = {
    "photos": {"photo_id": pl.Int64, "path": pl.Utf8},
    "listings": {"listing_id": pl.Int64, "title": pl.Utf8, "description": pl.Utf8},
    "listing_photos": {"listing_id": pl.Int64, "photo_id": pl.Int64, "position": pl.Int64},
}


def read_corpus_for_embedding(conn) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    frames = []
    with conn.cursor() as cur:
        for name, sql in EMBEDDING_INPUT_SQL.items():
            cur.execute(sql)
            frames.append(
                pl.DataFrame(cur.fetchall(), schema=EMBEDDING_INPUT_SCHEMA[name], orient="row")
            )
    return tuple(frames)
```

- [ ] **Step 4: Implement `listings/__main__.py`**

```python
"""Command line: python -m listings build|embed|detect|evaluate."""

import argparse
import dataclasses
import logging
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from ingestion.config import DbSettings
from listings.candidates import brute_force_pairs
from listings.config import CorpusConfig, DetectConfig
from listings.detect import run_detection, write_detection
from listings.embed import FakeEmbedder, SentenceTransformerEmbedder, embed_listings, embed_photos, resolve_device
from listings.evaluate import evaluate_detection, evaluate_fraud, log_run, save_pr_curve
from listings.fraud import run_fraud_checks, write_fraud_flags
from listings.generate import generate_corpus, load_areas, load_sales, render_variants
from listings.load import (
    create_vector_indexes,
    latest_corpus_run,
    load_corpus,
    read_corpus_for_embedding,
    write_listing_embeddings,
    write_photo_embeddings,
)
from listings.photos import DATA_DIR, ensure_pool
from listings.truth import load_duplicate_truth, load_truth

LOGGER = logging.getLogger("listings")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m listings",
        description="Build the synthetic listings corpus and detect duplicates and fraud.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="generate the corpus and load it into Postgres")
    build.add_argument("--listings", type=int, default=CorpusConfig.n_listings)
    build.add_argument("--seed", type=int, default=CorpusConfig.seed)
    build.add_argument("--data-dir", type=Path, default=DATA_DIR)

    embed = commands.add_parser("embed", help="embed photos and text, then build the vector indexes")
    embed.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    embed.add_argument("--data-dir", type=Path, default=DATA_DIR)
    embed.add_argument("--fake", action="store_true", help="use the deterministic test embedder")

    commands.add_parser("detect", help="score candidates and write duplicates and fraud flags")
    commands.add_parser("evaluate", help="measure against ground truth and log to MLflow")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    load_dotenv()  # before any settings are read
    commands = {"build": _build, "embed": _embed, "detect": _detect, "evaluate": _evaluate}
    try:
        return commands[args.command](args)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _build(args) -> int:
    config = dataclasses.replace(
        CorpusConfig(),
        seed=args.seed,
        n_listings=args.listings,
        n_base=args.listings - (CorpusConfig.n_exact_repost + CorpusConfig.n_reworded + CorpusConfig.n_edited_photo),
    )
    settings = DbSettings.from_env()
    pool = ensure_pool(config, data_dir=args.data_dir)
    print(f"photo pool: {len(pool.set_ids)} sets under {pool.root}")
    corpus = generate_corpus(load_sales(settings, config), load_areas(settings), pool.set_ids, config)
    written = render_variants(corpus, pool, config, Path(args.data_dir) / "variants")
    corpus_run_id = load_corpus(settings, corpus, config.seed, pool.archive_sha256)
    print(f"corpus run {corpus_run_id}: {corpus.counts}, {written} edited photos rendered")
    return 0


def _embed(args) -> int:
    settings = DbSettings.from_env()
    device = "cpu" if args.fake else resolve_device(args.device)
    embedder = FakeEmbedder() if args.fake else SentenceTransformerEmbedder(device=device)
    conn = settings.connect()
    try:
        photos, listings, listing_photos = read_corpus_for_embedding(conn)
        print(f"embedding {photos.height} photos and {listings.height} listings on {device}")
        photo_frame, photo_vectors = embed_photos(photos, embedder, args.data_dir)
        listing_frame = embed_listings(listings, listing_photos, photo_vectors, embedder)
        write_photo_embeddings(conn, photo_frame)
        write_listing_embeddings(conn, listing_frame)
        create_vector_indexes(conn)
        conn.commit()
    finally:
        conn.close()
    print("vectors written and HNSW indexes built")
    return 0


def _detect(args) -> int:
    settings, config = DbSettings.from_env(), DetectConfig()
    conn = settings.connect()
    try:
        corpus_run_id = latest_corpus_run(conn)
        result = run_detection(conn, config)
        detect_run_id = write_detection(conn, result, corpus_run_id)
        flagged = result.pairs.filter(result.pairs["decision"])
        fraud = run_fraud_checks(conn, flagged, config)
        write_fraud_flags(conn, fraud, detect_run_id)
        conn.commit()
    finally:
        conn.close()
    print(
        f"detect run {detect_run_id}: {result.pairs.height:,} candidate pairs, "
        f"{flagged.height:,} flagged at threshold {result.threshold:.4f}, "
        f"{fraud.flags.height:,} fraud flags"
    )
    return 0


def _evaluate(args) -> int:
    settings, config = DbSettings.from_env(), DetectConfig()
    conn = settings.connect()
    try:
        corpus_run_id = latest_corpus_run(conn)
        result = run_detection(conn, config)
        flagged = result.pairs.filter(result.pairs["decision"])
        fraud = run_fraud_checks(conn, flagged, config)
        truth_pairs, truth = load_duplicate_truth(conn), load_truth(conn)
        from listings.features import load_listing_attributes

        attributes = load_listing_attributes(conn)
        exact = brute_force_pairs(conn, "text", config.text_top_k)
    finally:
        conn.close()

    metrics = evaluate_detection(result, truth_pairs, attributes, config)
    metrics |= evaluate_fraud(fraud, truth, config)
    metrics["retrieval.exact_pairs"] = float(exact.height)
    for name, value in sorted(metrics.items()):
        print(f"{name:<45}{value:>12.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        curve = save_pr_curve(result.pairs["score"], result.pairs["is_duplicate"], Path(tmp) / "pr.png")
        run_id = log_run(
            metrics,
            {
                "corpus_run_id": corpus_run_id,
                "threshold": result.threshold,
                "text_top_k": config.text_top_k,
                "photo_top_k": config.photo_top_k,
                "max_photo_fanout": config.max_photo_fanout,
                "target_precision": config.target_precision,
            },
            [curve],
            config,
        )
    print(f"logged MLflow run {run_id} in experiment {config.experiment}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4b: Add the label-leakage scan and the CLI error-path tests**

Append to `tests/listings/test_listings_integration.py`:

```python
def test_only_truth_py_reads_label_columns():
    """Every other module's SQL must select listing content, never ground truth."""
    import inspect

    from listings import candidates, detect, features, fraud, load, truth

    for module in (candidates, detect, features, fraud, load):
        source = inspect.getsource(module)
        for label in ("dup_group_id", "control_group_id", "fraud_label"):
            statements = [
                line
                for line in source.splitlines()
                if label in line and "SELECT" in source[: source.index(line)].split("
")[-1].upper()
            ]
            assert not statements, f"{module.__name__} selects {label}: {statements}"
        # a blunt second check: the label names must not appear in any SQL constant
        for name, value in vars(module).items():
            if name.endswith("SQL") and isinstance(value, str):
                for label in ("dup_group_id", "control_group_id", "fraud_label"):
                    assert label not in value, f"{module.__name__}.{name} mentions {label}"
    assert "dup_group_id" in truth.TRUTH_SQL  # truth.py is the one place they belong


def test_cli_reports_failures_and_exits_1(monkeypatch, capsys):
    from listings import __main__ as cli

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)

    def boom(_args):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(cli, "_detect", boom)
    assert cli.main(["detect"]) == 1
    assert "detect failed: RuntimeError: database unreachable" in capsys.readouterr().err


def test_cli_rejects_an_unknown_command():
    import pytest

    from listings import __main__ as cli

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["nonsense"])
    assert exit_info.value.code == 2
```

- [ ] **Step 5: Run the integration test and the whole suite**

Run: `uv run pytest tests/listings/test_listings_integration.py -v` then `uv run pytest -q`
Expected: PASS. The integration test takes a couple of minutes (it runs the Phase 2 ingestion on 2,937 rows first).

If `report.precision` or `retrieval.recall` falls short, report the numbers as a
concern — do not relax the assertions.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add listings/__main__.py listings/load.py tests/listings/test_listings_integration.py
git commit -m "$(cat <<'EOF'
feat(listings): python -m listings build|embed|detect|evaluate with an end-to-end test

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: The real run and documentation

**Files:**
- Modify: `README.md`, `docs/superpowers/specs/2026-09-16-phase4-duplicate-fraud-design.md` (amendments)

**Interfaces:**
- Consumes: the `python -m listings` CLI and the Phase 3 champion model

**Rules for this task**
- The Docker stack must be up (`docker compose ps`), `dld` must hold the full ingestion, and `models:/dubimator-price@champion` must be registered (Phase 3 v2).
- The real run writes to the real MLflow server — that is intended here.
- NEVER connect to port 5432. Never `docker compose down -v`.
- The photo download is about 176 MB; the two models about 600 MB on first use.
- Steps 1–3 take a while: `build` a few minutes (rendering 3,600 edited photos), `embed` a few minutes on the GPU, `detect` and `evaluate` a few minutes each plus 3–6 minutes for the bait-price model calls. Run each with the Bash tool's `run_in_background: true`, tee to a log under `/tmp`, and poll until it exits.
- If a step exits non-zero, diagnose and report before changing code. Report DONE_WITH_CONCERNS if `embed` reports `cpu` — the GPU is a user requirement.

- [ ] **Step 1: Build the corpus**

```bash
uv run python -m listings build 2>&1 | tee /tmp/listings-build.log
```

Expected: the photo pool reports 535 sets, then a corpus run id with counts
(`listings: 20000`, `base: 17000`, `exact_repost: 1200`, `reworded: 900`,
`edited_photo: 900`, `bait_price: 600`) and `3600 edited photos rendered`.

Sanity-check the database:

```bash
MSYS_NO_PATHCONV=1 docker exec dubimator-postgres psql -U dubimator -d dubimator -c "
SELECT count(*) AS listings,
       count(DISTINCT area_id) AS areas,
       count(*) FILTER (WHERE dup_group_id IS NOT NULL) AS clones,
       count(*) FILTER (WHERE control_group_id IS NOT NULL) AS controls,
       count(*) FILTER (WHERE fraud_label = 'bait_price') AS bait
FROM listings.listings;"
```

- [ ] **Step 2: Embed on the GPU**

```bash
uv run python -m listings embed 2>&1 | tee /tmp/listings-embed.log
```

Expected: `embedding <n> photos and 20000 listings on cuda`, then the indexes are
built. Record the photo and listing counts and the wall-clock time.

- [ ] **Step 3: Detect and evaluate**

```bash
uv run python -m listings detect 2>&1 | tee /tmp/listings-detect.log
uv run python -m listings evaluate 2>&1 | tee /tmp/listings-evaluate.log
```

Expected: `detect` prints the candidate-pair count, how many were flagged and the
threshold; `evaluate` prints the full metric table and the MLflow run id. Copy both
logs next to your report.

Then measure the index against brute force on the real corpus, so the README's
claim is a measurement rather than an assertion:

```bash
uv run python - <<'PY'
import time
from dotenv import load_dotenv
load_dotenv()
from ingestion.config import DbSettings
from listings.candidates import brute_force_pairs, fetch_candidates
from listings.config import DetectConfig

config = DetectConfig()
conn = DbSettings.from_env().connect()
try:
    start = time.perf_counter()
    pairs, stats = fetch_candidates(conn, config)
    indexed = time.perf_counter() - start
    start = time.perf_counter()
    exact = brute_force_pairs(conn, "text", config.text_top_k)
    sequential = time.perf_counter() - start
finally:
    conn.close()
overlap = len(
    set(zip(pairs["listing_a"], pairs["listing_b"])) & set(zip(exact["listing_a"], exact["listing_b"]))
)
print(f"indexed {indexed:.1f}s for {pairs.height} pairs; exact text scan {sequential:.1f}s")
print(f"exact text pairs {exact.height}, of which retrieved by the index: {overlap}")
PY
```

- [ ] **Step 4: Update the README**

Insert a `## Duplicate and fraud detection` section immediately before
`## Cost breakdown (current)`. Replace every `‹…›` with a real number from the logs;
no `‹` may remain.

```markdown
## Duplicate and fraud detection

Finds duplicate property listings and suspicious postings using photo and text
similarity, on a **synthetic listings corpus built over real DLD sales**.

**What is real and what is not.** DLD publishes transactions, not listings: no
photos, no descriptions, no agents. So Phase 4 generates 20,000 listings whose
location, building, size, bedrooms and price level come from real 2021–2023 DLD
sales, and whose text, agents, posting dates, duplicates and fraud cases are
invented and labelled. Photos come from the public
[Houses-dataset](https://github.com/emanhamed/Houses-dataset) (535 properties ×
4 rooms; Ahmed & Moustafa, *House Price Estimation from Visual and Textual
Features*, 2016). The images are downloaded, never committed. Every number below
is measured on that synthetic corpus and does not transfer to a real portal.

```bash
uv run python -m listings build      # ‹›s: corpus + ‹› edited photo variants
uv run python -m listings embed      # ‹›s on the RTX 3060 (CLIP ViT-B/32 + all-MiniLM-L6-v2)
uv run python -m listings detect     # candidates, scores, duplicate_pairs + fraud_flags
uv run python -m listings evaluate   # metrics against ground truth, logged to MLflow
```

**What gets planted** (all labelled): 1,200 exact reposts, 900 reworded copies,
900 with cropped/resized/recompressed photos, plus two kinds of listing that must
**not** be flagged — different units in the same building (often sharing developer
photos) and unrelated listings sharing an agency photo set — and 600 bait-priced
listings.

**How it decides.** Candidates come from three indexed lookups per listing (text
neighbours, image neighbours, and listings sharing an identical photo), never from
comparing all pairs. Each candidate is then scored on twelve signals — text and
photo similarity, shared photos, price and size gaps, same area/building/project,
bedrooms, days apart, same agent — by a logistic regression whose threshold is set
for at least 98% precision, because wrongly accusing a real listing is worse than
missing a repost.

**Results** on the reporting split (the most recent posting dates, never used for
fitting or threshold selection):

| | Multi-signal model | Photos-only baseline |
|---|---|---|
| Precision | ‹›% | ‹›% |
| Recall | ‹›% | ‹›% |
| PR-AUC | ‹› | — |
| False positives on same-building controls | ‹›% | ‹›% |
| False positives on stock-photo controls | ‹›% | ‹›% |

Recall by planted pattern: exact repost ‹›%, reworded ‹›%, edited photos ‹›%.
Retrieval recall (duplicates that reached the scoring stage at all): ‹›%.

**Why an index and not brute force.** Comparing every pair of 20,000 listings is
200 million comparisons (about 3.2 billion at photo level). Measured here:
‹›s for the indexed lookups versus ‹›s for a single exact text scan, and the index
returned ‹›% of what the exact scan found.

**Fraud flags.** `bait_price` asks the Phase 3 price model what the home is worth
and flags asking prices more than 10% below its 80% range: precision ‹›%, recall
‹›% against the planted cases. `photo_reuse` flags a photo set spanning 5 or more
areas (‹› listings), and `inconsistent_relist` flags duplicate clusters whose
asking prices differ by more than 20% (‹› listings). If the price model cannot be
loaded, that flag is skipped and the rest still run.

**Write-up**

*Business problem.* Duplicate and fraudulent listings waste buyers' time and
damage a portal's credibility. The cost of a wrong accusation is high, so the
system is tuned for precision and every flag records the evidence behind it in
`listings.duplicate_pairs.signals`, ready for a review queue.

*Metric optimised.* Precision first (a floor of 98% chosen on the validation
split), with recall reported at that bar, plus the false-positive rate on the two
control groups — the cases a naive photo-similarity rule gets wrong.

*What I would do differently with real listings.* Real duplicate labels do not
exist, so I would bootstrap from agent-reported duplicates and moderator actions
and treat them as noisy positives; add watermark and logo detection, which real
agency photos carry; use ANN over photo embeddings with a fanout cap per photo
once stock photos are identified; and re-check the threshold per market segment,
since a luxury villa repost and a studio repost do not carry the same cost.
```

Also update:
- the status line at the top: `> **Status:** Phase 4 (duplicate and fraud detection) complete.`
- the Mermaid diagram: add `LISTINGS[("listings schema<br/>synthetic corpus + vectors")]` fed by `PG` and by `python -m listings`
- `## Module layout`: add ``- `listings/` — synthetic listings corpus, duplicate detection, fraud flags (`python -m listings`)``
- `## Cost breakdown (current)`: add `| Photo dataset + embedding models (one-off download) | $0 — about 0.8 GB on disk |`

- [ ] **Step 5: Add the spec amendments**

Append an `## Amendments during implementation` section to
`docs/superpowers/specs/2026-09-16-phase4-duplicate-fraud-design.md` recording:
- the retrieval change (listing image vectors plus a capped shared-photo channel instead of per-photo kNN, with the reason and the measured retrieval recall);
- `truth.py` as the only module that reads labels, and that the pair model is trained on them by design;
- `reg_type` added to the corpus so the Phase 3 price model can be called;
- the integration test using `tests/fixtures/price_sample.csv` rather than a new fixture;
- `detect` writing only flagged pairs while `evaluate` re-scores in memory;
- bait pricing applied to the listing's own asking price (the corpus deliberately stores no sale price);
- the real-run headline numbers and the date;
- anything else the implementers recorded as a concern.

- [ ] **Step 6: Final verification and commit**

```bash
uv run pytest -q
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git status --short
```

Expected: all tests pass; ruff clean; `git status` shows only `README.md` and the
spec as modified — nothing under `data/`, no `mlruns/`.

```bash
git add README.md docs/superpowers/specs/2026-09-16-phase4-duplicate-fraud-design.md
git commit -m "$(cat <<'EOF'
docs: Phase 4 duplicate and fraud detection results, write-up and spec amendments

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```
