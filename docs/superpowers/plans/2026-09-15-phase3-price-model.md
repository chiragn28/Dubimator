# Phase 3 — Price Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, evaluate and register a home price model on `dld.market_sales`. It uses location priors, a market index and XGBoost on the GPU, has calibrated price ranges, and is served through a self-contained MLflow pyfunc (`models:/dubimator-price@champion`).

**Architecture:** The package `models/price/` holds small modules:
- pure Polars feature code: segments, market index, location priors
- a thin XGBoost layer (`boosting.py`) and Optuna tuning (`tune.py`)
- baselines and metrics
- a DB-free `PricePredictor` that loads a model directory
- MLflow logging and registration

`train.py` orchestrates the whole run, and `python -m models.price` is the command line.

**Tech Stack:** Python 3.11, Polars, pandas, NumPy, XGBoost (CUDA), LightGBM, Optuna, scikit-learn, pydantic v2, MLflow 2.17.2, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-phase3-price-model-design.md`

## Global Constraints

- Python is pinned to 3.11 (`.python-version`). `mlflow==2.17.2` exactly, matching the server.
- Scope is homes only:
  - `property_type = 'villa'`, or
  - `property_type = 'unit'` with `property_sub_type IN ('Flat', 'Hotel Apartment', 'Stacked Townhouses')`.
- Split dates: train 2015-01-01 ≤ d < 2022-07-01; val 2022-07-01 ≤ d < 2022-11-01; test 2022-11-01 ≤ d ≤ `DATA_END`. Sales from 2014 only seed the market index.
- Size bounds in m², inclusive:
  - unit, built-up: 12–2,000
  - villa, built-up: 40–3,000
  - villa, plot: 60–20,000
- The feature allowlist, in this exact order: `property_type, reg_type, size_basis, sub_kind, room_kind, bedrooms, log_area_sqm, has_parking, area_id, market_index, prior_area, prior_project, prior_building, n_area, n_project, n_building, loc_level`.
- Never features: `price_per_sqm_aed, price_robust_z, peer_tier, exclusion_reason, source_row, ingest_run_id, procedure_name, price_aed, nearest_metro, nearest_mall, nearest_landmark`.
- The target is `y = ln(price_aed / area_sqm) − market_index`, and the price is `exp(ŷ + market_index) × area_sqm`.
- Shrinkage `K = 10`. `loc_level` is the deepest level with `n_eff ≥ 3`. Recency half-life is 730.5 days. Bulk weight is `1 / group_size`.
- The acceptance gate: champion `test_clean.all.mdape` ≤ 0.90 × B0's `test_clean.all.mdape`. Otherwise nothing is registered and the CLI exits with code 2.
- Tests connect to Postgres at `127.0.0.1:${POSTGRES_PORT}` (5433 on this machine). NEVER touch the native Windows Postgres on host port 5432.
- `tests/conftest.py` loads `.env`, which points MLflow at the real server. Every test that touches MLflow MUST use the `temp_mlflow` fixture (Task 10), so tests never write to the real server or to `./mlruns`.
- Work directly on `master`. Stage specific files only; never `git add -A` or `git add .`. Never commit `.env`, `data/raw/*` or `mlruns/`.
- Commit message trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Run commands from the repo root `C:\Users\cnaya\OneDrive\Desktop\dubimator` in Git Bash, always through `uv run`.
- At the end of every task, run `uv run ruff format .`, then `uv run ruff check . && uv run ruff format --check .` must be clean. The plan's code is correct but not always formatter-wrapped; formatting it is expected.
- Test layout follows Phase 2:
  - no `__init__.py` anywhere under `tests/` (a `tests/models` package would shadow the real `models` package)
  - every Phase 3 test file is named `tests/models/price/test_price_<topic>.py` (pytest needs unique basenames)
  - shared helpers are **fixtures** in `tests/models/price/conftest.py`, never imported modules

## Controller rulings made while planning (the spec travels with these)

1. **Two modules the spec didn't list:**
   - `boosting.py` holds the XGBoost helpers, so the predictor, and later the API, never import Optuna.
   - `plots.py` holds the evaluation plots.
2. **`PriceEstimate.model_version` carries the production MLflow run id.** The registered version number doesn't exist until after the artifact is logged, and the registry maps a run id to its version. The spec says "the registered version", so this ruling goes into the spec's amendments in Task 12.
3. **Validation weights** for early stopping are `bulk_weight` only. Recency weights are defined relative to the end of the fitted data and don't apply to later rows.
4. **The spec's leakage test** "perturbing val/test prices leaves their features unchanged" is implemented precisely. Perturb every price from month M on; every row dated in month M or earlier must keep identical features. The market index legitimately uses earlier val and test months.

## File map

| File | Responsibility | Task |
|---|---|---|
| `.python-version`, `pyproject.toml`, `uv.lock` | Python 3.11 pin, new dependencies | 1 |
| `models/price/__init__.py`, `models/price/config.py` | constants and `TrainConfig` | 1 |
| `models/price/features.py` | `derive_segments` (T2), `MarketIndex` (T4), priors (T5), assembly (T6) | 2, 4, 5, 6 |
| `models/price/data.py` | SQL load, scope, bounds, lineage, area reference | 2 |
| `models/price/split.py` | split, bulk groups, weights, grouped folds | 3 |
| `models/price/evaluate.py` | metrics, slices, conformal, coverage | 7 |
| `models/price/baselines.py` | B0 comps, B1 LightGBM | 7 |
| `models/price/boosting.py`, `models/price/tune.py` | device selection, XGBoost fit and predict, Optuna | 8 |
| `models/price/predictor.py` | request and response models, `ModelBundle`, `PricePredictor` | 9 |
| `models/price/pyfunc.py`, `models/price/registry.py`, `models/price/plots.py` | MLflow wrapper, logging and registration, plots | 10 |
| `models/price/train.py`, `scripts/build_price_fixture.py`, `tests/fixtures/price_sample.csv` | orchestration, integration fixture | 11 |
| `models/price/__main__.py`, `README.md`, spec amendments | CLI, real training run, documentation | 12 |

Tests live in `tests/models/price/`. Shared helpers go in `tests/models/price/conftest.py`: created in Task 2, extended in Tasks 6, 9 and 10.

---

## Task 1: Python 3.11, dependencies, config module

**Files:**
- Create: `.python-version` (via `uv python pin`), `models/price/__init__.py`, `models/price/config.py`, `tests/models/price/test_price_config.py`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Produces (`models.price.config`):
  - `HOME_UNIT_SUB_TYPES: tuple[str, ...]`
  - `STAT_EXCLUSION_REASONS: tuple[str, ...]`
  - `SIZE_BOUNDS: dict[tuple[str, str], tuple[float, float]]`
  - `FEATURES: tuple[str, ...]`, `CATEGORICAL_FEATURES: tuple[str, ...]`, `FORBIDDEN_FEATURES: tuple[str, ...]`
  - `TrainConfig`, a frozen dataclass with the fields shown below

- [ ] **Step 1: Pin Python and add dependencies**

```bash
uv python pin 3.11
uv add "mlflow==2.17.2" xgboost lightgbm optuna scikit-learn pandas pyarrow matplotlib pydantic
uv sync
uv run python --version
uv run python -c "import xgboost, lightgbm, optuna, sklearn, mlflow, pydantic; print(xgboost.__version__, xgboost.build_info()['USE_CUDA'], mlflow.__version__, pydantic.VERSION)"
```

Expected:
- `Python 3.11.x`
- the XGBoost version followed by `True` (the Windows wheel includes CUDA), then `2.17.2`, then a pydantic 2.x version

If `USE_CUDA` prints `False`, report DONE_WITH_CONCERNS with the XGBoost version; the controller rules on it. Training still works on the CPU.

If `uv add` fails to resolve because a pinned package (MLflow 2.17.2's dependencies) has no release for newer Python versions, narrow `requires-python` in `pyproject.toml` to `">=3.11,<3.13"`, retry, and report the change as a concern.

- [ ] **Step 2: Confirm the existing suite still passes on 3.11**

Run: `uv run pytest -q`
Expected: all existing tests pass (the stack must be up: `docker compose up -d --wait`).

- [ ] **Step 3: Write the failing config test**

`tests/models/price/test_price_config.py` (no `__init__.py` files anywhere under `tests/`):

```python
from datetime import date

from models.price.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    FORBIDDEN_FEATURES,
    SIZE_BOUNDS,
    TrainConfig,
)


def test_feature_allowlist_is_exact():
    assert FEATURES == (
        "property_type", "reg_type", "size_basis", "sub_kind", "room_kind",
        "bedrooms", "log_area_sqm", "has_parking", "area_id", "market_index",
        "prior_area", "prior_project", "prior_building",
        "n_area", "n_project", "n_building", "loc_level",
    )  # fmt: skip


def test_no_forbidden_feature_is_allowed():
    assert not set(FEATURES) & set(FORBIDDEN_FEATURES)
    assert {"price_per_sqm_aed", "price_robust_z", "peer_tier", "procedure_name"} <= set(
        FORBIDDEN_FEATURES
    )


def test_categoricals_are_features():
    assert set(CATEGORICAL_FEATURES) <= set(FEATURES)


def test_size_bounds_match_spec():
    assert SIZE_BOUNDS == {
        ("unit", "built_up"): (12.0, 2_000.0),
        ("villa", "built_up"): (40.0, 3_000.0),
        ("villa", "plot"): (60.0, 20_000.0),
    }


def test_default_dates_are_ordered():
    config = TrainConfig()
    assert config.index_start == date(2014, 1, 1)
    assert config.train_start == date(2015, 1, 1)
    assert config.val_start == date(2022, 7, 1)
    assert config.test_start == date(2022, 11, 1)
    assert config.index_start < config.train_start < config.val_start < config.test_start


def test_default_tunables_match_spec():
    config = TrainConfig()
    assert (config.shrink_k, config.min_level_n, config.half_life_days) == (10.0, 3.0, 730.5)
    assert (config.n_trials, config.max_rounds, config.early_stopping_rounds) == (60, 4_000, 100)
    assert config.gate_ratio == 0.90
    assert (config.experiment, config.model_name) == ("price-model", "dubimator-price")
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/models/price/test_price_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.price'`.

- [ ] **Step 5: Implement the config module**

`models/price/__init__.py`:

```python
"""Home price estimation model (Phase 3)."""
```

`models/price/config.py`:

```python
"""Constants and tunables for the price model.

Spec: docs/superpowers/specs/2026-09-15-phase3-price-model-design.md
"""

from dataclasses import dataclass
from datetime import date

HOME_UNIT_SUB_TYPES = ("Flat", "Hotel Apartment", "Stacked Townhouses")
STAT_EXCLUSION_REASONS = ("price_outlier_low", "price_outlier_high", "suspected_sqft_entry")

# (property_type, size_basis) -> inclusive (min, max) size in m². Training scope and the
# predictor enforce the same bounds.
SIZE_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    ("unit", "built_up"): (12.0, 2_000.0),
    ("villa", "built_up"): (40.0, 3_000.0),
    ("villa", "plot"): (60.0, 20_000.0),
}

CATEGORICAL_FEATURES = (
    "property_type", "reg_type", "size_basis", "sub_kind", "room_kind", "area_id",
)  # fmt: skip
FEATURES = (
    "property_type", "reg_type", "size_basis", "sub_kind", "room_kind",
    "bedrooms", "log_area_sqm", "has_parking", "area_id", "market_index",
    "prior_area", "prior_project", "prior_building",
    "n_area", "n_project", "n_building", "loc_level",
)  # fmt: skip
FORBIDDEN_FEATURES = (
    "price_per_sqm_aed", "price_robust_z", "peer_tier", "exclusion_reason",
    "source_row", "ingest_run_id", "procedure_name", "price_aed",
    "nearest_metro", "nearest_mall", "nearest_landmark",
)  # fmt: skip


@dataclass(frozen=True)
class TrainConfig:
    index_start: date = date(2014, 1, 1)
    train_start: date = date(2015, 1, 1)
    val_start: date = date(2022, 7, 1)
    test_start: date = date(2022, 11, 1)
    min_segment_rows: int = 200
    min_index_sales: int = 30
    shrink_k: float = 10.0
    min_level_n: float = 3.0
    oof_folds: int = 5
    half_life_days: float = 730.5
    comps_min_n: float = 5.0
    bounds_min_area_n: float = 30.0
    min_conformal_rows: int = 200
    n_trials: int = 60
    max_rounds: int = 4_000
    early_stopping_rounds: int = 100
    lgbm_learning_rate: float = 0.05
    seed: int = 42
    gate_ratio: float | None = 0.90  # None disables the acceptance gate (tests only)
    experiment: str = "price-model"
    model_name: str = "dubimator-price"
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/models/price/test_price_config.py tests/test_module_placeholders.py -v`
Expected: PASS.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add .python-version pyproject.toml uv.lock models/price/__init__.py models/price/config.py tests/models/price/test_price_config.py
git commit -m "$(cat <<'EOF'
feat(price): pin Python 3.11, add modelling dependencies and price config

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---
## Task 2: Segment derivations and the homes loader

**Files:**
- Create: `models/price/features.py` (only `derive_segments` for now), `models/price/data.py`, `tests/models/price/conftest.py`, `tests/models/price/test_price_segments.py`, `tests/models/price/test_price_data.py`

**Interfaces:**
- Consumes:
  - `ingestion.config.DbSettings` (`.connect()`)
  - `ingestion.pipeline.run_pipeline(csv_path: Path, settings) -> RunSummary` (has `.run_id`), in the test only
  - the `pg_test_db` fixture from `tests/conftest.py`
- Produces:
  - `models.price.features.derive_segments(frame: pl.DataFrame) -> pl.DataFrame`
    - adds `size_basis`, `sub_kind`, `room_kind`, `bedrooms` (Float64) and `segment`
  - `models.price.data.RAW_SCHEMA: dict[str, pl.DataType]`
    - columns: `transaction_id, instance_date, property_type, property_sub_type, reg_type, area_id, building_name, project_name, rooms, has_parking, area_sqm, price_aed, ingest_run_id, is_clean`
  - `models.price.data.HOMES_SQL: str`
  - `models.price.data.prepare_homes(raw, config) -> tuple[pl.DataFrame, dict[str, int], tuple[str, ...]]`
    - returns rows, drop counts keyed `"{rule}.{segment}"`, and supported segments
  - `models.price.data.HomesData`, a frozen dataclass: `rows, data_end: date, lineage: dict, drop_counts: dict[str, int], supported_segments: tuple[str, ...], areas: pl.DataFrame (area_id, name_en), aliases: pl.DataFrame (alias_key, area_id)`
  - `models.price.data.load_homes(settings: DbSettings, config: TrainConfig) -> HomesData`
  - test fixture `raw_homes` in `tests/models/price/conftest.py`, which returns a builder `build(*overrides: dict) -> pl.DataFrame` with one row per override dict, each starting from `BASE_HOME`

- [ ] **Step 1: Write the shared test helper**

`tests/models/price/conftest.py`:

```python
from datetime import date

import polars as pl
import pytest

from models.price.data import RAW_SCHEMA

BASE_HOME = {
    "transaction_id": "t",
    "instance_date": date(2020, 1, 15),
    "property_type": "unit",
    "property_sub_type": "Flat",
    "reg_type": "ready",
    "area_id": 1,
    "building_name": "Tower A",
    "project_name": "Project P",
    "rooms": "1 B/R",
    "has_parking": True,
    "area_sqm": 80.0,
    "price_aed": 800_000.0,
    "ingest_run_id": 1,
    "is_clean": True,
}


def build_raw_homes(*overrides: dict) -> pl.DataFrame:
    """One raw home row per override dict, each starting from BASE_HOME."""
    records = [
        {**BASE_HOME, "transaction_id": f"t{i}", **row} for i, row in enumerate(overrides)
    ]
    return pl.DataFrame(records, schema=RAW_SCHEMA)


@pytest.fixture
def raw_homes():
    return build_raw_homes
```

- [ ] **Step 2: Write the failing tests**

`tests/models/price/test_price_segments.py`:

```python
from models.price.features import derive_segments


def test_size_basis_and_sub_kind(raw_homes):
    frame = derive_segments(
        raw_homes(
            {"property_type": "unit", "property_sub_type": "Flat"},
            {"property_type": "unit", "property_sub_type": "Hotel Apartment"},
            {"property_type": "unit", "property_sub_type": "Stacked Townhouses"},
            {"property_type": "villa", "property_sub_type": "Villa"},
            {"property_type": "villa", "property_sub_type": None},
        )
    )
    assert frame["size_basis"].to_list() == ["built_up", "built_up", "built_up", "built_up", "plot"]
    assert frame["sub_kind"].to_list() == ["flat", "hotel_apartment", "townhouse", "villa", "villa"]


def test_room_kind_and_bedrooms_for_every_rooms_value(raw_homes):
    rooms = ["Studio", "2 B/R", "Penthouse", "Single Room", None, "Office"]
    frame = derive_segments(raw_homes(*({"rooms": value} for value in rooms)))
    assert frame["room_kind"].to_list() == [
        "studio", "bedrooms", "penthouse", "single_room", "unknown", "unknown",
    ]  # fmt: skip
    assert frame["bedrooms"].to_list() == [0.0, 2.0, None, None, None, None]


def test_segment_name(raw_homes):
    frame = derive_segments(
        raw_homes(
            {"property_type": "unit", "reg_type": "ready"},
            {"property_type": "villa", "property_sub_type": None, "reg_type": "off_plan"},
        )
    )
    assert frame["segment"].to_list() == ["unit_ready_built_up", "villa_off_plan_plot"]
```

`tests/models/price/test_price_data.py`:

```python
import dataclasses
from datetime import date
from pathlib import Path

from ingestion.pipeline import run_pipeline
from models.price.config import FORBIDDEN_FEATURES, HOME_UNIT_SUB_TYPES, TrainConfig
from models.price.data import HOMES_SQL, RAW_SCHEMA, load_homes, prepare_homes

SMALL = dataclasses.replace(TrainConfig(), min_segment_rows=1)


def test_sql_never_selects_target_derived_columns():
    for name in ("price_per_sqm_aed", "price_robust_z", "peer_tier", "source_row"):
        assert name not in HOMES_SQL


def test_only_lineage_and_target_columns_overlap_the_forbidden_list():
    assert set(RAW_SCHEMA) & set(FORBIDDEN_FEATURES) == {"ingest_run_id", "price_aed"}


def test_size_bounds_drop_and_count_per_segment(raw_homes):
    raw = raw_homes(
        {"area_sqm": 11.9},
        {"area_sqm": 12.0},
        {"area_sqm": 2_000.0},
        {"area_sqm": 2_000.5},
        {"property_type": "villa", "property_sub_type": None, "area_sqm": 59.0},
        {"property_type": "villa", "property_sub_type": None, "area_sqm": 60.0},
        {"property_type": "villa", "property_sub_type": "Villa", "area_sqm": 40.0},
    )
    rows, drops, _ = prepare_homes(raw, SMALL)
    assert rows["area_sqm"].to_list() == [12.0, 2_000.0, 60.0, 40.0]
    assert drops == {"size_bounds.unit_ready_built_up": 2, "size_bounds.villa_ready_plot": 1}


def test_unsupported_segments_are_dropped_and_counted(raw_homes):
    config = dataclasses.replace(TrainConfig(), min_segment_rows=2)
    plot_villa = {
        "property_type": "villa", "property_sub_type": None, "reg_type": "off_plan",
        "area_sqm": 400.0,
    }  # fmt: skip
    rows, drops, supported = prepare_homes(raw_homes({}, {}, plot_villa), config)
    assert supported == ("unit_ready_built_up",)
    assert drops == {"unsupported_segment.villa_off_plan_plot": 1}
    assert rows.height == 2


def test_support_counts_only_clean_train_period_rows(raw_homes):
    config = dataclasses.replace(TrainConfig(), min_segment_rows=2)
    raw = raw_homes(
        {"instance_date": date(2016, 5, 1)},
        {"instance_date": date(2022, 8, 1)},  # val period: doesn't count
        {"instance_date": date(2016, 6, 1), "is_clean": False},  # not clean: doesn't count
    )
    _, drops, supported = prepare_homes(raw, config)
    assert supported == ()
    assert drops == {"unsupported_segment.unit_ready_built_up": 3}


def test_load_homes_reads_scope_lineage_and_reference_tables(pg_test_db):
    summary = run_pipeline(Path("tests/fixtures/dld_sample.csv"), pg_test_db)
    config = dataclasses.replace(SMALL, index_start=date(1990, 1, 1))
    homes = load_homes(pg_test_db, config)

    assert homes.rows.height > 0
    assert set(homes.rows["property_type"].unique()) <= {"unit", "villa"}
    units = homes.rows.filter(homes.rows["property_type"] == "unit")
    assert set(units["property_sub_type"].unique()) <= set(HOME_UNIT_SUB_TYPES)
    assert homes.lineage["ingest_run_id"] == summary.run_id
    assert len(homes.lineage["source_sha256"]) == 64
    clean = homes.rows.filter(homes.rows["is_clean"])
    assert homes.data_end == clean["instance_date"].max()
    stat = homes.rows.filter(~homes.rows["is_clean"])
    assert stat.height == 0 or stat["instance_date"].min() >= config.test_start
    assert homes.areas.columns == ["area_id", "name_en"] and homes.areas.height > 0
    assert homes.aliases.columns == ["alias_key", "area_id"] and homes.aliases.height > 0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/models/price/test_price_segments.py tests/models/price/test_price_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.price.data'`.

- [ ] **Step 4: Implement `derive_segments`**

`models/price/features.py`:

```python
"""Feature engineering for the price model: segments, market index, location priors."""

import polars as pl

SUB_KINDS = {"Flat": "flat", "Hotel Apartment": "hotel_apartment", "Stacked Townhouses": "townhouse"}
_BEDROOMS = r"^(\d) B/R$"


def derive_segments(frame: pl.DataFrame) -> pl.DataFrame:
    """Add size_basis, sub_kind, room_kind, bedrooms and segment (pure; no target use)."""
    villa = pl.col("property_type") == "villa"
    rooms = pl.col("rooms")
    bedroom_count = rooms.str.extract(_BEDROOMS, 1).cast(pl.Float64)
    size_basis = (
        pl.when(villa & pl.col("property_sub_type").is_null())
        .then(pl.lit("plot"))
        .otherwise(pl.lit("built_up"))
    )
    sub_kind = (
        pl.when(villa)
        .then(pl.lit("villa"))
        .otherwise(
            pl.col("property_sub_type").replace_strict(SUB_KINDS, default=None, return_dtype=pl.Utf8)
        )
    )
    room_kind = (
        pl.when(rooms == "Studio")
        .then(pl.lit("studio"))
        .when(bedroom_count.is_not_null())
        .then(pl.lit("bedrooms"))
        .when(rooms == "Penthouse")
        .then(pl.lit("penthouse"))
        .when(rooms == "Single Room")
        .then(pl.lit("single_room"))
        .otherwise(pl.lit("unknown"))
    )
    bedrooms = pl.when(rooms == "Studio").then(pl.lit(0.0)).otherwise(bedroom_count)
    return frame.with_columns(
        size_basis.alias("size_basis"),
        sub_kind.alias("sub_kind"),
        room_kind.alias("room_kind"),
        bedrooms.alias("bedrooms"),
    ).with_columns(
        pl.concat_str(
            [pl.col("property_type"), pl.col("reg_type"), pl.col("size_basis")], separator="_"
        ).alias("segment")
    )
```

- [ ] **Step 5: Implement the loader**

`models/price/data.py`:

```python
"""Load in-scope home sales from Postgres and apply the training scope."""

from dataclasses import dataclass
from datetime import date

import polars as pl

from ingestion.config import DbSettings
from models.price.config import (
    HOME_UNIT_SUB_TYPES,
    SIZE_BOUNDS,
    STAT_EXCLUSION_REASONS,
    TrainConfig,
)
from models.price.features import derive_segments

RAW_SCHEMA = {
    "transaction_id": pl.Utf8,
    "instance_date": pl.Date,
    "property_type": pl.Utf8,
    "property_sub_type": pl.Utf8,
    "reg_type": pl.Utf8,
    "area_id": pl.Int64,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "rooms": pl.Utf8,
    "has_parking": pl.Boolean,
    "area_sqm": pl.Float64,
    "price_aed": pl.Float64,
    "ingest_run_id": pl.Int64,
    "is_clean": pl.Boolean,
}
_SELECTED = ", ".join(name for name in RAW_SCHEMA if name != "is_clean")

# exclusion_reason is read only to split clean sales from the honest-holdout rows.
HOMES_SQL = f"""
SELECT {_SELECTED}, exclusion_reason IS NULL AS is_clean
FROM dld.transactions
WHERE (property_type = 'villa'
       OR (property_type = 'unit' AND property_sub_type IN %(unit_sub_types)s))
  AND instance_date >= %(index_start)s
  AND (exclusion_reason IS NULL
       OR (exclusion_reason IN %(stat_reasons)s AND instance_date >= %(test_start)s))
"""


@dataclass(frozen=True)
class HomesData:
    rows: pl.DataFrame
    data_end: date
    lineage: dict[str, object]
    drop_counts: dict[str, int]
    supported_segments: tuple[str, ...]
    areas: pl.DataFrame
    aliases: pl.DataFrame


def _in_bounds() -> pl.Expr:
    inside = pl.lit(False)
    for (property_type, basis), (low, high) in SIZE_BOUNDS.items():
        inside = inside | (
            (pl.col("property_type") == property_type)
            & (pl.col("size_basis") == basis)
            & pl.col("area_sqm").is_between(low, high)
        )
    return inside.fill_null(False)


def _count_by_segment(frame: pl.DataFrame, rule: str) -> dict[str, int]:
    counts = frame.group_by("segment").len().sort("segment")
    return {f"{rule}.{segment}": count for segment, count in counts.iter_rows()}


def prepare_homes(
    raw: pl.DataFrame, config: TrainConfig
) -> tuple[pl.DataFrame, dict[str, int], tuple[str, ...]]:
    """Derive segments, drop out-of-bounds sizes and unsupported segments, count drops."""
    frame = derive_segments(raw)
    drops = _count_by_segment(frame.filter(~_in_bounds()), "size_bounds")
    kept = frame.filter(_in_bounds())
    in_train = (
        pl.col("is_clean")
        & (pl.col("instance_date") >= config.train_start)
        & (pl.col("instance_date") < config.val_start)
    )
    counts = kept.filter(in_train).group_by("segment").len()
    supported = tuple(
        sorted(segment for segment, count in counts.iter_rows() if count >= config.min_segment_rows)
    )
    in_scope = pl.col("segment").is_in(list(supported))
    drops |= _count_by_segment(kept.filter(~in_scope), "unsupported_segment")
    return kept.filter(in_scope), drops, supported


def load_homes(settings: DbSettings, config: TrainConfig) -> HomesData:
    params = {
        "unit_sub_types": HOME_UNIT_SUB_TYPES,
        "index_start": config.index_start,
        "stat_reasons": STAT_EXCLUSION_REASONS,
        "test_start": config.test_start,
    }
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(HOMES_SQL, params)
            raw = pl.DataFrame(cur.fetchall(), schema=RAW_SCHEMA, orient="row")
            run_ids = sorted(raw["ingest_run_id"].unique().to_list())
            if len(run_ids) != 1:
                raise ValueError(f"expected rows from exactly one ingestion run, found {run_ids}")
            cur.execute(
                "SELECT source_sha256 FROM dld.ingestion_runs WHERE run_id = %s", (run_ids[0],)
            )
            (source_sha256,) = cur.fetchone()
            cur.execute("SELECT area_id, name_en FROM dld.areas ORDER BY area_id")
            areas = pl.DataFrame(
                cur.fetchall(), schema={"area_id": pl.Int64, "name_en": pl.Utf8}, orient="row"
            )
            cur.execute("SELECT alias_key, area_id FROM dld.area_aliases ORDER BY alias_key, area_id")
            aliases = pl.DataFrame(
                cur.fetchall(), schema={"alias_key": pl.Utf8, "area_id": pl.Int64}, orient="row"
            )
    finally:
        conn.close()
    rows, drops, supported = prepare_homes(raw, config)
    return HomesData(
        rows=rows,
        data_end=rows.filter(pl.col("is_clean"))["instance_date"].max(),
        lineage={"ingest_run_id": run_ids[0], "source_sha256": source_sha256},
        drop_counts=drops,
        supported_segments=supported,
        areas=areas,
        aliases=aliases,
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/models/price/test_price_segments.py tests/models/price/test_price_data.py -v`
Expected: PASS. The DB test needs the stack up; it creates and drops `dubimator_test`.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add models/price/features.py models/price/data.py tests/models/price/conftest.py tests/models/price/test_price_segments.py tests/models/price/test_price_data.py
git commit -m "$(cat <<'EOF'
feat(price): derive home segments and load in-scope sales with lineage

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Time split, bulk groups, weights, grouped folds

**Files:**
- Create: `models/price/split.py`, `tests/models/price/test_price_split.py`

**Interfaces:**
- Consumes: the `raw_homes` fixture (Task 2), `TrainConfig` (Task 1)
- Produces (`models.price.split`):
  - `BULK_KEY: tuple[str, ...]`
  - `assign_split(frame, config) -> pl.DataFrame`: adds `split` ∈ {`seed`, `train`, `val`, `test`}
  - `add_bulk_groups(frame) -> pl.DataFrame`: adds `bulk_group` (UInt64), `group_size` (Int64) and `bulk_weight` (Float64 = 1/group_size)
  - `sample_weights(frame, fit_end: date, half_life_days: float) -> np.ndarray`: recency × bulk weight
  - `grouped_folds(frame, n_folds: int) -> np.ndarray`: an int fold id per row; a bulk group never spans two folds

- [ ] **Step 1: Write the failing tests**

`tests/models/price/test_price_split.py`:

```python
from datetime import date

import polars as pl
import pytest

from models.price.config import TrainConfig
from models.price.split import add_bulk_groups, assign_split, grouped_folds, sample_weights


def test_split_boundaries_are_exact(raw_homes):
    days = [
        date(2014, 12, 31), date(2015, 1, 1), date(2022, 6, 30), date(2022, 7, 1),
        date(2022, 10, 31), date(2022, 11, 1), date(2023, 3, 17),
    ]  # fmt: skip
    frame = assign_split(raw_homes(*({"instance_date": d} for d in days)), TrainConfig())
    assert frame["split"].to_list() == ["seed", "train", "train", "val", "val", "test", "test"]


def test_identical_sales_share_a_bulk_group_even_with_null_project(raw_homes):
    frame = add_bulk_groups(
        raw_homes(
            {"project_name": None},
            {"project_name": None},
            {"project_name": None, "price_aed": 900_000.0},
        )
    )
    groups = frame["bulk_group"].to_list()
    assert groups[0] == groups[1] != groups[2]
    assert frame["group_size"].to_list() == [2, 2, 1]
    assert frame["bulk_weight"].to_list() == [0.5, 0.5, 1.0]


def test_no_bulk_group_spans_two_splits(raw_homes):
    days = [date(2022, 6, 30), date(2022, 7, 1), date(2022, 10, 31), date(2022, 11, 1)]
    frame = raw_homes(*({"instance_date": d} for d in days for _ in range(3)))
    frame = add_bulk_groups(assign_split(frame, TrainConfig()))
    spans = frame.group_by("bulk_group").agg(pl.col("split").n_unique().alias("n"))
    assert spans["n"].max() == 1
    assert frame["group_size"].unique().to_list() == [3]


def test_recency_weight_halves_per_half_life_and_bulk_weight_divides(raw_homes):
    fit_end = date(2022, 6, 30)
    frame = add_bulk_groups(
        raw_homes(
            {"instance_date": fit_end},
            {"instance_date": date(2020, 6, 30)},  # 730 days before fit_end
            *({"instance_date": fit_end, "price_aed": 700_000.0} for _ in range(4)),
        )
    )
    weights = sample_weights(frame, fit_end, 730.5)
    assert weights.tolist() == pytest.approx([1.0, 0.5 ** (730 / 730.5), 0.25, 0.25, 0.25, 0.25])
    assert sample_weights(frame, fit_end, 730.0)[1] == pytest.approx(0.5)


def test_grouped_folds_keep_each_bulk_group_in_one_fold(raw_homes):
    rows = [{"price_aed": 500_000.0 + 1_000.0 * (i // 2)} for i in range(20)]  # 10 groups of 2
    frame = add_bulk_groups(raw_homes(*rows))
    folds = grouped_folds(frame, 5)
    per_group = pl.DataFrame({"g": frame["bulk_group"], "f": folds}).group_by("g").agg(
        pl.col("f").n_unique()
    )
    assert per_group["f"].max() == 1
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/models/price/test_price_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.price.split'`.

- [ ] **Step 3: Implement `split.py`**

`models/price/split.py`:

```python
"""Time-based split, bulk-sale groups, sample weights and grouped folds."""

from datetime import date

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from models.price.config import TrainConfig

# Rows identical on all of these are one bulk sale (one pricing decision). Nulls compare equal.
BULK_KEY = (
    "instance_date", "area_id", "project_name", "building_name",
    "property_type", "property_sub_type", "price_aed",
)  # fmt: skip


def assign_split(frame: pl.DataFrame, config: TrainConfig) -> pl.DataFrame:
    day = pl.col("instance_date")
    split = (
        pl.when(day < config.train_start)
        .then(pl.lit("seed"))
        .when(day < config.val_start)
        .then(pl.lit("train"))
        .when(day < config.test_start)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
    )
    return frame.with_columns(split.alias("split"))


def add_bulk_groups(frame: pl.DataFrame) -> pl.DataFrame:
    key = list(BULK_KEY)
    return frame.with_columns(
        pl.struct(key).hash(seed=0).alias("bulk_group"),
        pl.len().over(key).cast(pl.Int64).alias("group_size"),
    ).with_columns((1.0 / pl.col("group_size")).alias("bulk_weight"))


def sample_weights(frame: pl.DataFrame, fit_end: date, half_life_days: float) -> np.ndarray:
    """Recency weight (halves every half_life_days before fit_end) times bulk weight."""
    age_days = (pl.lit(fit_end) - pl.col("instance_date")).dt.total_days()
    weight = pl.lit(0.5).pow(age_days / half_life_days) * pl.col("bulk_weight")
    return frame.select(weight.alias("w"))["w"].to_numpy()


def grouped_folds(frame: pl.DataFrame, n_folds: int) -> np.ndarray:
    groups = frame["bulk_group"].to_numpy()
    folds = np.empty(frame.height, dtype=np.int64)
    splitter = GroupKFold(n_splits=n_folds)
    for fold, (_, members) in enumerate(splitter.split(np.zeros(frame.height), groups=groups)):
        folds[members] = fold
    return folds
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/models/price/test_price_split.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add models/price/split.py tests/models/price/test_price_split.py
git commit -m "$(cat <<'EOF'
feat(price): time split, bulk-sale groups, recency weights, grouped folds

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Market index

**Files:**
- Modify: `models/price/features.py` (append)
- Create: `tests/models/price/test_price_market_index.py`

**Interfaces:**
- Consumes: the frame columns `segment`, `property_type`, `size_basis`, `instance_date`, `price_aed`, `area_sqm`
- Produces (`models.price.features`):
  - `MarketIndexError(RuntimeError)`
  - `INDEX_WINDOWS = (3, 6, 12)`
  - `month_number(day: date) -> int`, `month_from_number(n: int) -> date`
  - `MarketIndex`, a frozen dataclass with field `table: pl.DataFrame` (`segment` Utf8, `month` Date as the first of the month, `value` Float64, `source` Utf8)
    - `MarketIndex.fit(sales, first_month: date, last_month: date, min_sales: int) -> MarketIndex`
    - `.lookup(frame) -> pl.Series` named `market_index`, aligned to the frame's rows; raises `MarketIndexError` if any row is missing
    - `.value_at(segment: str, month: date) -> float`
  - `source` is one of `segment_3`, `segment_6`, `segment_12`, `pooled_3`, `pooled_6`, `pooled_12` or `carried`

The rule (spec, "Market index"):
- `I(segment, m)` = the median of `ln(price_aed / area_sqm)` over sales in months [m−3, m−1], i.e. strictly before m.
- If the window has fewer than `min_sales` sales, widen it to 6 and then 12 months.
- If still short, pool over `reg_type` (the pool is `property_type` + `size_basis`), using the same window sequence.
- If still short, carry forward the segment's last value.
- If there is no earlier value to carry, raise `MarketIndexError`.

- [ ] **Step 1: Write the failing tests**

`tests/models/price/test_price_market_index.py`:

```python
import math
from datetime import date

import polars as pl
import pytest

from models.price.features import MarketIndex, MarketIndexError

U = "unit_ready_built_up"
SCHEMA = {
    "segment": pl.Utf8,
    "property_type": pl.Utf8,
    "size_basis": pl.Utf8,
    "instance_date": pl.Date,
    "price_aed": pl.Float64,
    "area_sqm": pl.Float64,
}


def make_sales(entries):
    """entries: (segment, day, ln_price_per_sqm) triples."""
    records = []
    for segment, day, value in entries:
        size_basis = "plot" if segment.endswith("_plot") else "built_up"
        records.append(
            {
                "segment": segment,
                "property_type": segment.split("_", 1)[0],
                "size_basis": size_basis,
                "instance_date": day,
                "price_aed": math.exp(value) * 100.0,
                "area_sqm": 100.0,
            }
        )
    return pl.DataFrame(records, schema=SCHEMA)


def index_row(index, segment, month):
    row = index.table.filter((pl.col("segment") == segment) & (pl.col("month") == month))
    assert row.height == 1
    return row["value"][0], row["source"][0]


def test_index_is_the_median_of_the_three_prior_months():
    sales = make_sales(
        [(U, date(2020, 1, 10), 1.0), (U, date(2020, 2, 10), 2.0),
         (U, date(2020, 3, 10), 3.0), (U, date(2020, 4, 10), 100.0)]
    )  # fmt: skip
    index = MarketIndex.fit(sales, date(2020, 4, 1), date(2020, 4, 1), min_sales=3)
    value, source = index_row(index, U, date(2020, 4, 1))
    assert value == pytest.approx(2.0)
    assert source == "segment_3"


def test_index_never_sees_same_month_or_later_prices():
    entries = [(U, date(2020, m, d), 1.0 + m / 10) for m in range(1, 7) for d in (5, 20)]
    base = MarketIndex.fit(make_sales(entries), date(2020, 3, 1), date(2020, 6, 1), min_sales=1)
    perturbed_entries = [
        (s, day, v + (5.0 if day >= date(2020, 5, 1) else 0.0)) for s, day, v in entries
    ]
    perturbed = MarketIndex.fit(
        make_sales(perturbed_entries), date(2020, 3, 1), date(2020, 6, 1), min_sales=1
    )
    for month in (date(2020, 3, 1), date(2020, 4, 1), date(2020, 5, 1)):
        assert index_row(perturbed, U, month)[0] == pytest.approx(index_row(base, U, month)[0])
    assert index_row(perturbed, U, date(2020, 6, 1))[0] != pytest.approx(
        index_row(base, U, date(2020, 6, 1))[0]
    )


def test_window_widens_when_three_months_are_thin():
    entries = [(U, date(2019, 11, 1 + i), 1.0) for i in range(5)] + [(U, date(2020, 2, 1), 9.0)]
    index = MarketIndex.fit(make_sales(entries), date(2020, 4, 1), date(2020, 4, 1), min_sales=3)
    value, source = index_row(index, U, date(2020, 4, 1))
    assert (value, source) == (pytest.approx(1.0), "segment_6")


def test_pools_over_reg_type_when_the_segment_has_no_history():
    ready, off_plan = "villa_ready_built_up", "villa_off_plan_built_up"
    entries = [(ready, date(2020, m, 1), 4.0) for m in (1, 2, 3)] + [(off_plan, date(2020, 4, 2), 7.0)]
    index = MarketIndex.fit(make_sales(entries), date(2020, 4, 1), date(2020, 4, 1), min_sales=3)
    value, source = index_row(index, off_plan, date(2020, 4, 1))
    assert (value, source) == (pytest.approx(4.0), "pooled_3")


def test_carries_the_last_value_forward_when_history_runs_out():
    entries = [(U, date(2020, m, d), 2.0) for m in (1, 2, 3) for d in (1, 2, 3)]
    index = MarketIndex.fit(make_sales(entries), date(2020, 4, 1), date(2021, 6, 1), min_sales=3)
    value, source = index_row(index, U, date(2021, 6, 1))
    assert (value, source) == (pytest.approx(2.0), "carried")


def test_raises_when_there_is_no_earlier_sale_at_all():
    sales = make_sales([(U, date(2020, 4, 10), 1.0)])
    with pytest.raises(MarketIndexError, match="no earlier sales"):
        MarketIndex.fit(sales, date(2020, 4, 1), date(2020, 4, 1), min_sales=1)


def test_lookup_aligns_rows_and_rejects_months_outside_the_fit():
    entries = [(U, date(2020, m, 1), float(m)) for m in (1, 2, 3)]
    index = MarketIndex.fit(make_sales(entries), date(2020, 4, 1), date(2020, 5, 1), min_sales=1)
    rows = make_sales([(U, date(2020, 5, 20), 0.0), (U, date(2020, 4, 3), 0.0)])
    assert index.lookup(rows).to_list() == pytest.approx([2.5, 2.0])
    assert index.lookup(rows).name == "market_index"
    assert index.value_at(U, date(2020, 5, 17)) == pytest.approx(2.5)
    with pytest.raises(MarketIndexError):
        index.lookup(make_sales([(U, date(2020, 9, 1), 0.0)]))
```

The numbers in the lookup test:
- 2020-04 uses Jan–Mar, values 1, 2, 3 → median 2.0.
- 2020-05 uses Feb–Apr (Apr has no sales), values 2, 3 → median 2.5.
- `min_sales=1` keeps the 3-month window.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/models/price/test_price_market_index.py -v`
Expected: FAIL with `ImportError: cannot import name 'MarketIndex'`.

- [ ] **Step 3: Implement the market index**

In `models/price/features.py`, replace the imports at the top with:

```python
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
```

Append to `models/price/features.py`:

```python
INDEX_WINDOWS = (3, 6, 12)
INDEX_SCHEMA = {"segment": pl.Utf8, "month": pl.Date, "value": pl.Float64, "source": pl.Utf8}


class MarketIndexError(RuntimeError):
    """The market index cannot be computed or looked up for a segment/month."""


def month_number(day: date) -> int:
    return day.year * 12 + day.month - 1


def month_from_number(number: int) -> date:
    return date(number // 12, number % 12 + 1, 1)


def _window_median(
    months: np.ndarray, values: np.ndarray, target: int, width: int, min_sales: int
) -> float | None:
    """Median of values in months [target - width, target - 1]; None if too few sales."""
    low = np.searchsorted(months, target - width, side="left")
    high = np.searchsorted(months, target, side="left")
    if high - low < min_sales:
        return None
    return float(np.median(values[low:high]))


def _resolve(candidates, target: int, min_sales: int) -> tuple[float | None, str | None]:
    for label, months, values in candidates:
        for width in INDEX_WINDOWS:
            value = _window_median(months, values, target, width, min_sales)
            if value is not None:
                return value, f"{label}_{width}"
    return None, None


@dataclass(frozen=True)
class MarketIndex:
    """Median ln(price per m²) per segment over the months strictly before each month."""

    table: pl.DataFrame

    @classmethod
    def fit(
        cls, sales: pl.DataFrame, first_month: date, last_month: date, min_sales: int
    ) -> "MarketIndex":
        day = pl.col("instance_date")
        data = sales.select(
            "segment",
            pl.concat_str([pl.col("property_type"), pl.col("size_basis")], separator="_").alias(
                "pool"
            ),
            (day.dt.year().cast(pl.Int64) * 12 + day.dt.month().cast(pl.Int64) - 1).alias("m"),
            (pl.col("price_aed") / pl.col("area_sqm")).log().alias("v"),
        ).sort("m")

        def history(column: str, key: str) -> tuple[np.ndarray, np.ndarray]:
            subset = data.filter(pl.col(column) == key)
            return subset["m"].to_numpy(), subset["v"].to_numpy()

        records = []
        for segment, pool in data.select("segment", "pool").unique().sort("segment").iter_rows():
            candidates = (("segment", *history("segment", segment)), ("pooled", *history("pool", pool)))
            last = None
            for target in range(month_number(first_month), month_number(last_month) + 1):
                value, source = _resolve(candidates, target, min_sales)
                if value is None:
                    if last is None:
                        raise MarketIndexError(
                            f"no market index for {segment} in "
                            f"{month_from_number(target):%Y-%m}: no earlier sales"
                        )
                    value, source = last, "carried"
                last = value
                records.append((segment, month_from_number(target), value, source))
        return cls(pl.DataFrame(records, schema=INDEX_SCHEMA, orient="row"))

    def lookup(self, frame: pl.DataFrame) -> pl.Series:
        keyed = frame.select(
            pl.col("segment"), pl.col("instance_date").dt.truncate("1mo").alias("month")
        ).with_row_index("__row")
        joined = keyed.join(
            self.table.select("segment", "month", "value"), on=["segment", "month"], how="left"
        ).sort("__row")
        if joined["value"].null_count():
            raise MarketIndexError(
                "market index missing for some rows: segment or month outside the fitted range"
            )
        return joined["value"].alias("market_index")

    def value_at(self, segment: str, month: date) -> float:
        first = month.replace(day=1)
        match = self.table.filter((pl.col("segment") == segment) & (pl.col("month") == first))
        if match.height != 1:
            raise MarketIndexError(f"no market index for {segment} in {first:%Y-%m}")
        return float(match["value"][0])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/models/price/test_price_market_index.py tests/models/price/test_price_segments.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add models/price/features.py tests/models/price/test_price_market_index.py
git commit -m "$(cat <<'EOF'
feat(price): leak-free monthly market index with widening, pooling and carry-forward

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Location priors (hierarchical shrinkage, out-of-fold)

**Files:**
- Modify: `models/price/features.py` (append)
- Create: `tests/models/price/test_price_priors.py`

**Interfaces:**
- Consumes:
  - `ingestion.normalize.map_unique(series, fn, dtype)` and `ingestion.normalize.match_key(value)`
  - `models.price.split.grouped_folds` (Task 3), in the tests
- Produces (`models.price.features`):
  - `LEVEL_KEYS: dict[str, tuple[str, ...]]`: levels `city`, `area`, `project`, `building`
  - `PRIOR_COLUMNS = ("prior_area", "prior_project", "prior_building", "n_area", "n_project", "n_building", "loc_level")`
  - `LocationPriorError(RuntimeError)`
  - `add_location_keys(frame) -> pl.DataFrame`: adds `project_key` and `building_key`, the `match_key` of the project and building names
  - `LocationPriors`, a frozen dataclass: `stats: dict[str, pl.DataFrame]` (per level: key columns + `sum_wy` + `sum_w`), `shrink_k: float`, `min_level_n: float`
    - `LocationPriors.fit(frame, shrink_k, min_level_n) -> LocationPriors`: the frame needs the level keys, `y` and `bulk_weight`
    - `.transform(frame) -> pl.DataFrame`: the same rows in the same order, plus `PRIOR_COLUMNS`
  - `oof_priors(frame, folds: np.ndarray, shrink_k, min_level_n) -> pl.DataFrame`: the same rows in the same order, plus `PRIOR_COLUMNS`, each fold transformed by priors fitted on the other folds

The rule (spec, "Location priors"), with every sum coalesced to 0 when a key is unseen or absent:
- `city = Σwy / Σw` for the property type
- `prior_area = (Σwy_area + K·city) / (Σw_area + K)`
- `prior_project = (Σwy_project + K·prior_area) / (Σw_project + K)`
- `prior_building = (Σwy_building + K·prior_project) / (Σw_building + K)`

So an unseen or absent level equals its parent exactly. `n_* = log1p(Σw)`. `loc_level` is 3 if `Σw_building ≥ min_level_n`, else 2 if `Σw_project ≥ min_level_n`, else 1 if `Σw_area ≥ min_level_n`, else 0.

- [ ] **Step 1: Write the failing tests**

`tests/models/price/test_price_priors.py`:

```python
import math

import polars as pl
import pytest

from models.price.features import (
    LocationPriorError,
    LocationPriors,
    add_location_keys,
    oof_priors,
)
from models.price.split import grouped_folds

SCHEMA = {
    "property_type": pl.Utf8,
    "area_id": pl.Int64,
    "project_name": pl.Utf8,
    "building_name": pl.Utf8,
    "y": pl.Float64,
    "bulk_weight": pl.Float64,
    "bulk_group": pl.UInt64,
}
DEFAULTS = {"property_type": "unit", "area_id": 1, "project_name": None, "building_name": None,
            "y": 0.0, "bulk_weight": 1.0}  # fmt: skip


def make_rows(*entries):
    records = [{**DEFAULTS, "bulk_group": i, **entry} for i, entry in enumerate(entries)]
    return add_location_keys(pl.DataFrame(records, schema=SCHEMA))


MARINA = {"area_id": 1, "project_name": "Marina Gate", "building_name": "Tower 1", "y": 1.0}
FIT = make_rows(*[MARINA] * 2, *[{"area_id": 2}] * 8)
# city = 2/10 = 0.2; area 1 = (2 + 10*0.2)/12 = 1/3; project = (2 + 10/3)/12 = 4/9;
# building = (2 + 40/9)/12 = 29/54


def test_shrinkage_matches_hand_computed_values():
    priors = LocationPriors.fit(FIT, shrink_k=10.0, min_level_n=3.0)
    row = priors.transform(make_rows(MARINA)).row(0, named=True)
    assert row["prior_area"] == pytest.approx(1 / 3)
    assert row["prior_project"] == pytest.approx(4 / 9)
    assert row["prior_building"] == pytest.approx(29 / 54)
    assert row["n_area"] == pytest.approx(math.log1p(2))
    assert row["loc_level"] == 0  # every level has only 2 effective sales, below 3


def test_unseen_levels_fall_back_to_their_parent():
    priors = LocationPriors.fit(FIT, shrink_k=10.0, min_level_n=2.0)
    rows = priors.transform(
        make_rows(
            MARINA,
            {"area_id": 1, "project_name": "Marina Gate", "building_name": "Tower 9"},
            {"area_id": 1, "project_name": "Unknown Project"},
            {"area_id": 3},
        )
    )
    assert rows["loc_level"].to_list() == [3, 2, 1, 0]
    unseen_building = rows.row(1, named=True)
    assert unseen_building["prior_building"] == pytest.approx(4 / 9)
    assert unseen_building["n_building"] == 0.0
    unseen_project = rows.row(2, named=True)
    assert unseen_project["prior_project"] == pytest.approx(1 / 3)
    assert unseen_project["prior_building"] == pytest.approx(1 / 3)
    unseen_area = rows.row(3, named=True)
    assert unseen_area["prior_area"] == pytest.approx(0.2)
    assert unseen_area["prior_building"] == pytest.approx(0.2)


def test_names_are_matched_by_normalized_key():
    priors = LocationPriors.fit(FIT, shrink_k=10.0, min_level_n=2.0)
    row = priors.transform(
        make_rows({"area_id": 1, "project_name": "AL MARINA-GATE", "building_name": " tower  1 "})
    ).row(0, named=True)
    assert row["loc_level"] == 3
    assert row["prior_building"] == pytest.approx(29 / 54)


def test_sparse_area_resolves_to_city_level():
    fit = make_rows(*[{"area_id": 5}] * 2, *[{"area_id": 6}] * 3)
    rows = LocationPriors.fit(fit, 10.0, 3.0).transform(make_rows({"area_id": 5}, {"area_id": 6}))
    assert rows["loc_level"].to_list() == [0, 1]


def test_transform_keeps_row_order():
    priors = LocationPriors.fit(FIT, 10.0, 3.0)
    frame = make_rows({"area_id": 2}, MARINA, {"area_id": 2})
    out = priors.transform(frame)
    assert out["area_id"].to_list() == [2, 1, 2]
    assert out.height == 3


def test_out_of_fold_priors_hide_a_bulk_sale_from_its_own_group():
    normal = [{"building_name": "Tower B", "y": 0.0} for _ in range(10)]
    bulk = {"building_name": "Tower B", "y": 5.0, "bulk_weight": 0.5}
    frame = make_rows(*normal, bulk, bulk).with_columns(
        pl.when(pl.col("y") == 5.0).then(pl.lit(99, pl.UInt64)).otherwise(pl.col("bulk_group"))
        .alias("bulk_group")
    )  # fmt: skip
    folds = grouped_folds(frame, 2)
    assert folds[10] == folds[11]  # identical sales share a fold

    oof = oof_priors(frame, folds, shrink_k=10.0, min_level_n=3.0)
    in_sample = LocationPriors.fit(frame, 10.0, 3.0).transform(frame)
    assert oof.height == frame.height
    assert oof["prior_building"][10] < 0.1  # never saw the 5.0 prices
    assert in_sample["prior_building"][10] > 0.4  # in-sample would have leaked them


def test_missing_property_type_raises():
    priors = LocationPriors.fit(FIT, 10.0, 3.0)
    with pytest.raises(LocationPriorError, match="villa"):
        priors.transform(make_rows({"property_type": "villa"}))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/models/price/test_price_priors.py -v`
Expected: FAIL with `ImportError: cannot import name 'LocationPriorError'`.

- [ ] **Step 3: Implement the priors**

Add to the imports at the top of `models/price/features.py`:

```python
from ingestion.normalize import map_unique, match_key
```

Append to `models/price/features.py`:

```python
LEVEL_KEYS = {
    "city": ("property_type",),
    "area": ("property_type", "area_id"),
    "project": ("property_type", "area_id", "project_key"),
    "building": ("property_type", "area_id", "building_key"),
}
PRIOR_COLUMNS = (
    "prior_area", "prior_project", "prior_building",
    "n_area", "n_project", "n_building", "loc_level",
)  # fmt: skip


class LocationPriorError(RuntimeError):
    """A row's property type has no prior fit data."""


def add_location_keys(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        map_unique(frame["project_name"], match_key, pl.Utf8).alias("project_key"),
        map_unique(frame["building_name"], match_key, pl.Utf8).alias("building_key"),
    )


@dataclass(frozen=True)
class LocationPriors:
    """Bulk-weighted relative-price statistics per location level, shrunk down the chain."""

    stats: dict[str, pl.DataFrame]
    shrink_k: float
    min_level_n: float

    @classmethod
    def fit(cls, frame: pl.DataFrame, shrink_k: float, min_level_n: float) -> "LocationPriors":
        stats = {}
        for level, keys in LEVEL_KEYS.items():
            stats[level] = (
                frame.drop_nulls(list(keys))
                .group_by(list(keys))
                .agg(
                    (pl.col("y") * pl.col("bulk_weight")).sum().alias("sum_wy"),
                    pl.col("bulk_weight").sum().alias("sum_w"),
                )
                .sort(list(keys))
            )
        return cls(stats, shrink_k, min_level_n)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        out = frame.with_row_index("__p_row")
        for level, keys in LEVEL_KEYS.items():
            table = self.stats[level].rename({"sum_wy": f"__p_wy_{level}", "sum_w": f"__p_w_{level}"})
            out = out.join(table, on=list(keys), how="left")
        out = out.sort("__p_row")
        if out["__p_w_city"].null_count():
            missing = sorted(out.filter(pl.col("__p_w_city").is_null())["property_type"].unique())
            raise LocationPriorError(f"no prior fit data for property type(s) {missing}")

        def wy(level: str) -> pl.Expr:
            return pl.col(f"__p_wy_{level}").fill_null(0.0)

        def w(level: str) -> pl.Expr:
            return pl.col(f"__p_w_{level}").fill_null(0.0)

        k = self.shrink_k
        out = out.with_columns((pl.col("__p_wy_city") / pl.col("__p_w_city")).alias("__p_city"))
        out = out.with_columns(((wy("area") + k * pl.col("__p_city")) / (w("area") + k)).alias("prior_area"))
        out = out.with_columns(
            ((wy("project") + k * pl.col("prior_area")) / (w("project") + k)).alias("prior_project")
        )
        out = out.with_columns(
            ((wy("building") + k * pl.col("prior_project")) / (w("building") + k)).alias(
                "prior_building"
            )
        )
        n = self.min_level_n
        out = out.with_columns(
            w("area").log1p().alias("n_area"),
            w("project").log1p().alias("n_project"),
            w("building").log1p().alias("n_building"),
            pl.when(w("building") >= n)
            .then(3)
            .when(w("project") >= n)
            .then(2)
            .when(w("area") >= n)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("loc_level"),
        )
        return out.drop([name for name in out.columns if name.startswith("__p_")])


def oof_priors(
    frame: pl.DataFrame, folds: np.ndarray, shrink_k: float, min_level_n: float
) -> pl.DataFrame:
    """Transform each fold with priors fitted on the other folds (no row sees its own price)."""
    indexed = frame.with_columns(pl.Series("__oof_fold", folds)).with_row_index("__oof_row")
    parts = []
    for fold in np.unique(folds):
        in_fold = pl.col("__oof_fold") == fold
        priors = LocationPriors.fit(indexed.filter(~in_fold), shrink_k, min_level_n)
        parts.append(priors.transform(indexed.filter(in_fold)))
    return pl.concat(parts).sort("__oof_row").drop("__oof_row", "__oof_fold")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/models/price/ -v`
Expected: PASS (all Phase 3 tests so far).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add models/price/features.py tests/models/price/test_price_priors.py
git commit -m "$(cat <<'EOF'
feat(price): hierarchical location priors with shrinkage and grouped out-of-fold fitting

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Feature assembly, reference tables, leakage test

**Files:**
- Modify: `models/price/features.py` (append), `tests/models/price/conftest.py` (add the `synthetic_homes` fixture)
- Create: `tests/models/price/test_price_features.py`

**Interfaces:**
- Consumes:
  - `derive_segments`, `MarketIndex`, `LocationPriors`, `add_location_keys`, `oof_priors` (Tasks 2, 4, 5)
  - `grouped_folds`, `sample_weights`, `assign_split`, `add_bulk_groups` (Task 3)
  - `FEATURES`, `CATEGORICAL_FEATURES`, `TrainConfig` (Task 1)
- Produces (`models.price.features`):
  - `add_model_columns(frame) -> pl.DataFrame`: adds `log_area_sqm` and casts `has_parking` to Float64
  - `add_target(frame) -> pl.DataFrame`: adds `y = ln(price_aed / area_sqm) − market_index`
  - `fit_categories(frame) -> dict[str, list[str]]`: sorted string categories per categorical feature
  - `feature_frame(frame, categories) -> pd.DataFrame`: exactly `FEATURES`, in order. Categoricals become `pd.Categorical` with the fixed categories (unseen values become NaN); everything else is float64
  - `fit_bounds(frame, min_area_n) -> tuple[pl.DataFrame, pl.DataFrame]`
    - bounds per area: `property_type, size_basis, area_id, lo, hi`, for areas with `Σ bulk_weight ≥ min_area_n`
    - bounds per segment: `property_type, size_basis, lo, hi`
    - where `lo = p01(y) − ln 3` and `hi = p99(y) + ln 3`
  - `fit_size_percentiles(frame) -> pl.DataFrame`: `segment, p005, p995` of `area_sqm`
  - `FeatureSet`, a frozen dataclass: `frames: dict[str, pl.DataFrame]`, `weights: np.ndarray` (aligned to `frames["fit"]`), `index: MarketIndex`, `priors: LocationPriors` (fitted on all fit rows, for serving), `categories: dict[str, list[str]]`
  - `build_feature_set(rows, config, data_end, fit_filter: pl.Expr, apply_filters: dict[str, pl.Expr]) -> FeatureSet`
    - `rows` must already carry `split`, `bulk_group`, `group_size` and `bulk_weight` (from `assign_split` and `add_bulk_groups`)
    - `frames["fit"]` holds out-of-fold priors; each apply frame uses priors fitted on all fit rows
  - fixture `synthetic_homes` → a builder `build(seed=0, per_month=6) -> pl.DataFrame`: prepared rows (as `prepare_homes` would return them) from 2014-01 to 2023-03, with two segments (`unit_ready_built_up`, `villa_ready_built_up`), three areas, and stat-excluded (`is_clean=False`) rows in the test period

- [ ] **Step 1: Add the synthetic data fixture**

Append to `tests/models/price/conftest.py` (merge the imports into its import block):

```python
import math
from datetime import timedelta

import numpy as np

from models.price.features import derive_segments


def build_synthetic_homes(seed: int = 0, per_month: int = 6) -> pl.DataFrame:
    """Prepared home rows 2014-01..2023-03: prices rise 0.4%/month, area and villa premia."""
    rng = np.random.default_rng(seed)
    records = []
    month = date(2014, 1, 1)
    while month <= date(2023, 3, 1):
        months_since = (month.year - 2014) * 12 + month.month - 1
        level = math.log(9_000.0) + 0.004 * months_since
        for j in range(per_month):
            villa = j % 3 == 2
            area_id = int(rng.integers(1, 4))
            size = float(rng.uniform(250.0, 600.0) if villa else rng.uniform(40.0, 150.0))
            ln_ppsqm = level + 0.15 * area_id + (0.2 if villa else 0.0) + rng.normal(0.0, 0.1)
            records.append(
                {
                    "transaction_id": f"s{len(records)}",
                    "instance_date": month.replace(day=int(rng.integers(1, 28))),
                    "property_type": "villa" if villa else "unit",
                    "property_sub_type": "Villa" if villa else "Flat",
                    "reg_type": "ready",
                    "area_id": area_id,
                    "building_name": None if villa else f"Tower {area_id}{int(rng.integers(0, 3))}",
                    "project_name": f"Project {area_id}",
                    "rooms": f"{int(rng.integers(1, 4))} B/R",
                    "has_parking": bool(rng.integers(0, 2)),
                    "area_sqm": round(size, 1),
                    "price_aed": round(math.exp(ln_ppsqm) * size, -2),
                    "ingest_run_id": 1,
                    "is_clean": not (month >= date(2022, 11, 1) and j == 0),
                }
            )
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    return derive_segments(pl.DataFrame(records, schema=RAW_SCHEMA))


@pytest.fixture
def synthetic_homes():
    return build_synthetic_homes
```

- [ ] **Step 2: Write the failing tests**

`tests/models/price/test_price_features.py`:

```python
import dataclasses
import math
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

from models.price.config import CATEGORICAL_FEATURES, FEATURES, TrainConfig
from models.price.features import (
    add_model_columns,
    build_feature_set,
    feature_frame,
    fit_bounds,
    fit_size_percentiles,
)
from models.price.split import add_bulk_groups, assign_split

CONFIG = dataclasses.replace(TrainConfig(), min_index_sales=1, oof_folds=3)
DATA_END = date(2023, 3, 27)
CLEAN, SPLIT = pl.col("is_clean"), pl.col("split")
APPLY = {
    "val": (SPLIT == "val") & CLEAN,
    "test_clean": (SPLIT == "test") & CLEAN,
    "test_honest": SPLIT == "test",
}


def eval_set(rows):
    prepared = add_bulk_groups(assign_split(rows, CONFIG))
    return build_feature_set(
        prepared, CONFIG, DATA_END, fit_filter=(SPLIT == "train") & CLEAN, apply_filters=APPLY
    )


def test_feature_frame_is_exactly_the_allowlist(synthetic_homes):
    fs = eval_set(synthetic_homes())
    features = feature_frame(fs.frames["fit"], fs.categories)
    assert list(features.columns) == list(FEATURES)
    for name in FEATURES:
        expected = "category" if name in CATEGORICAL_FEATURES else "float64"
        assert str(features[name].dtype) == expected, name


def test_feature_frame_ignores_extra_and_forbidden_columns(synthetic_homes):
    fs = eval_set(synthetic_homes())
    frame = fs.frames["val"].with_columns(pl.lit(1.0).alias("price_per_sqm_aed"))
    assert "price_per_sqm_aed" not in feature_frame(frame, fs.categories).columns


def test_unseen_category_becomes_missing(synthetic_homes):
    fs = eval_set(synthetic_homes())
    frame = fs.frames["val"].with_columns(pl.lit(999, pl.Int64).alias("area_id"))
    assert feature_frame(frame, fs.categories)["area_id"].isna().all()


def test_target_is_relative_to_the_market_index(synthetic_homes):
    fit = eval_set(synthetic_homes()).frames["fit"]
    expected = np.log(fit["price_aed"] / fit["area_sqm"]) - fit["market_index"]
    assert fit["y"].to_numpy() == pytest.approx(expected.to_numpy())


def test_features_never_depend_on_same_month_or_later_prices(synthetic_homes):
    rows = synthetic_homes()
    cutoff = CONFIG.test_start  # perturb every price from the first test month on
    perturbed = rows.with_columns(
        pl.when(pl.col("instance_date") >= cutoff)
        .then(pl.col("price_aed") * 3.0)
        .otherwise(pl.col("price_aed"))
        .alias("price_aed")
    )
    base, changed = eval_set(rows), eval_set(perturbed)
    first_month_end = date(2022, 12, 1)
    for name in ("fit", "val", "test_clean", "test_honest"):
        keep = pl.col("instance_date") < first_month_end
        before = feature_frame(base.frames[name].filter(keep), base.categories)
        after = feature_frame(changed.frames[name].filter(keep), changed.categories)
        assert len(before) > 0, name  # every set has rows dated before 2022-12
        pd.testing.assert_frame_equal(before, after)


def test_weights_align_with_fit_rows(synthetic_homes):
    fs = eval_set(synthetic_homes())
    assert fs.weights.shape == (fs.frames["fit"].height,)
    assert 0.0 < fs.weights.min() and fs.weights.max() <= 1.0


def test_add_model_columns():
    frame = add_model_columns(
        pl.DataFrame({"area_sqm": [100.0, 50.0], "has_parking": [True, None]})
    )
    assert frame["log_area_sqm"].to_list() == pytest.approx([math.log(100.0), math.log(50.0)])
    assert frame["has_parking"].to_list() == [1.0, None]


def test_bounds_and_size_percentiles():
    frame = pl.DataFrame(
        {
            "property_type": ["unit"] * 106,
            "size_basis": ["built_up"] * 106,
            "segment": ["unit_ready_built_up"] * 106,
            "area_id": [1] * 101 + [2] * 5,
            "y": [i / 100 for i in range(101)] + [0.0] * 5,
            "bulk_weight": [1.0] * 106,
            "area_sqm": [float(i + 1) for i in range(106)],
        }
    )
    by_area, by_segment = fit_bounds(frame, min_area_n=30.0)
    assert by_area["area_id"].to_list() == [1]  # area 2 has only 5 sales
    assert by_area["lo"][0] == pytest.approx(0.01 - math.log(3.0))
    assert by_area["hi"][0] == pytest.approx(0.99 + math.log(3.0))
    assert by_segment.columns == ["property_type", "size_basis", "lo", "hi"]
    sizes = fit_size_percentiles(frame)
    assert sizes.columns == ["segment", "p005", "p995"]
    assert sizes["p005"][0] == pytest.approx(1.0 + 0.005 * 105)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/models/price/test_price_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'add_model_columns'`.

- [ ] **Step 4: Implement the assembly**

Add to the imports at the top of `models/price/features.py`, keeping ruff's import order:

```python
import math

import pandas as pd

from models.price.config import CATEGORICAL_FEATURES, FEATURES, TrainConfig
from models.price.split import grouped_folds, sample_weights
```

Append to `models/price/features.py`:

```python
LN3 = math.log(3.0)


def add_model_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("area_sqm").log().alias("log_area_sqm"),
        pl.col("has_parking").cast(pl.Float64).alias("has_parking"),
    )


def add_target(frame: pl.DataFrame) -> pl.DataFrame:
    """y = ln(price per m²) − market index: the premium over the current market level."""
    y = (pl.col("price_aed") / pl.col("area_sqm")).log() - pl.col("market_index")
    return frame.with_columns(y.alias("y"))


def fit_categories(frame: pl.DataFrame) -> dict[str, list[str]]:
    return {
        name: sorted(frame[name].drop_nulls().cast(pl.Utf8).unique().to_list())
        for name in CATEGORICAL_FEATURES
    }


def feature_frame(frame: pl.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Exactly FEATURES, in order; categories fixed at fit time (unseen values become NaN)."""
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise KeyError(f"feature columns missing: {missing}")
    selected = frame.select(
        [
            pl.col(name).cast(pl.Utf8) if name in CATEGORICAL_FEATURES else pl.col(name).cast(pl.Float64)
            for name in FEATURES
        ]
    ).to_pandas()
    for name in CATEGORICAL_FEATURES:
        selected[name] = pd.Categorical(selected[name], categories=categories[name])
    return selected


def fit_bounds(frame: pl.DataFrame, min_area_n: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Plausible range of y: [p1 − ln 3, p99 + ln 3] per area (if well sampled) and per segment."""
    stats = [
        pl.col("y").quantile(0.01, "linear").alias("p01"),
        pl.col("y").quantile(0.99, "linear").alias("p99"),
        pl.col("bulk_weight").sum().alias("n"),
    ]

    def finish(table: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
        return table.select(
            *keys, (pl.col("p01") - LN3).alias("lo"), (pl.col("p99") + LN3).alias("hi")
        ).sort(keys)

    area_keys = ["property_type", "size_basis", "area_id"]
    segment_keys = ["property_type", "size_basis"]
    by_area = frame.group_by(area_keys).agg(stats).filter(pl.col("n") >= min_area_n)
    by_segment = frame.group_by(segment_keys).agg(stats)
    return finish(by_area, area_keys), finish(by_segment, segment_keys)


def fit_size_percentiles(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by("segment")
        .agg(
            pl.col("area_sqm").quantile(0.005, "linear").alias("p005"),
            pl.col("area_sqm").quantile(0.995, "linear").alias("p995"),
        )
        .sort("segment")
    )


@dataclass(frozen=True)
class FeatureSet:
    frames: dict[str, pl.DataFrame]
    weights: np.ndarray
    index: MarketIndex
    priors: LocationPriors
    categories: dict[str, list[str]]


def build_feature_set(
    rows: pl.DataFrame,
    config: TrainConfig,
    data_end: date,
    fit_filter: pl.Expr,
    apply_filters: dict[str, pl.Expr],
) -> FeatureSet:
    """Index, target and priors for one fit.

    frames["fit"] gets out-of-fold priors; apply frames get priors fitted on all fit rows.
    """
    rows = add_location_keys(add_model_columns(rows))
    index = MarketIndex.fit(
        rows.filter(pl.col("is_clean")),
        config.train_start.replace(day=1),
        data_end.replace(day=1),
        config.min_index_sales,
    )
    rows = rows.filter(pl.col("split") != "seed")
    rows = add_target(rows.with_columns(index.lookup(rows)))
    fit_rows = rows.filter(fit_filter)
    if fit_rows.height == 0:
        raise ValueError("fit_filter selected no rows")
    folds = grouped_folds(fit_rows, config.oof_folds)
    frames = {"fit": oof_priors(fit_rows, folds, config.shrink_k, config.min_level_n)}
    priors = LocationPriors.fit(fit_rows, config.shrink_k, config.min_level_n)
    for name, expr in apply_filters.items():
        frames[name] = priors.transform(rows.filter(expr))
    weights = sample_weights(fit_rows, fit_rows["instance_date"].max(), config.half_life_days)
    return FeatureSet(frames, weights, index, priors, fit_categories(fit_rows))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/models/price/ -v`
Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add models/price/features.py tests/models/price/conftest.py tests/models/price/test_price_features.py
git commit -m "$(cat <<'EOF'
feat(price): assemble the allowlisted feature set with leak-free priors and reference tables

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Metrics, conformal intervals, baselines

**Files:**
- Create: `models/price/evaluate.py`, `models/price/baselines.py`, `tests/models/price/test_price_evaluate.py`, `tests/models/price/test_price_baselines.py`

**Interfaces:**
- Consumes: `build_feature_set`, `feature_frame` (Task 6); the `synthetic_homes` fixture; `assign_split`, `add_bulk_groups` (Task 3)
- Produces (`models.price.evaluate`):
  - `CONFORMAL_LEVELS = {"q80": 0.20, "q95": 0.05}`
  - `price_metrics(actual, predicted) -> dict[str, float]`, with keys `mdape, ppe10, ppe20, rmse_log, n`
  - `sliced_metrics(frame, set_name) -> dict[str, float]`
    - the frame has columns `actual, predicted, segment, loc_level, bulk_group`
    - keys are `{set}.{slice}.{metric}`, where the slice is `all`, `segment.<s>`, `loc_level.<n>` or `dedup`
  - `conformal_quantile(abs_errors, alpha) -> float`: the ⌈(n+1)(1−α)⌉-th smallest value, or `inf` if that rank exceeds n
  - `fit_conformal(segments, abs_errors, min_rows) -> dict[str, dict[str, float]]`
    - `{"_pooled": {...}, "<segment>": {"q80": .., "q95": ..}}`
    - a segment with fewer than `min_rows` rows copies the pooled values
    - raises `ValueError` if the pooled q95 is infinite
  - `quantiles_for(conformal, segment) -> dict[str, float]`: falls back to `_pooled`
  - `coverage(actual, low, high) -> float`: the share inside `[low, high]`, inclusive
- Produces (`models.price.baselines`):
  - `comps_b0(fit: pl.DataFrame, target: pl.DataFrame, min_n: float) -> np.ndarray`: ŷ in target (relative) space
  - `lightgbm_b1(X_fit, y_fit, w_fit, X_val, y_val, w_val, config) -> lightgbm.Booster`, trained with early stopping. Predict with `booster.predict(X, num_iteration=booster.best_iteration)`

- [ ] **Step 1: Write the failing evaluation tests**

`tests/models/price/test_price_evaluate.py`:

```python
import math

import numpy as np
import polars as pl
import pytest

from models.price.evaluate import (
    conformal_quantile,
    coverage,
    fit_conformal,
    price_metrics,
    quantiles_for,
    sliced_metrics,
)


def test_price_metrics_match_hand_computed_values():
    metrics = price_metrics([100.0, 100.0, 100.0, 100.0], [110.0, 90.0, 130.0, 100.0])
    assert metrics["mdape"] == pytest.approx(0.10)
    assert metrics["ppe10"] == pytest.approx(0.75)
    assert metrics["ppe20"] == pytest.approx(0.75)
    expected_rmse = math.sqrt(
        (math.log(1.1) ** 2 + math.log(0.9) ** 2 + math.log(1.3) ** 2 + 0.0) / 4
    )
    assert metrics["rmse_log"] == pytest.approx(expected_rmse)
    assert metrics["n"] == 4.0


def test_sliced_metrics_cover_segments_levels_and_dedup():
    frame = pl.DataFrame(
        {
            "actual": [100.0, 100.0, 200.0, 200.0],
            "predicted": [110.0, 110.0, 200.0, 260.0],
            "segment": ["a", "a", "b", "b"],
            "loc_level": [3, 3, 1, 0],
            "bulk_group": [7, 7, 8, 9],
        }
    )
    metrics = sliced_metrics(frame, "test_clean")
    assert metrics["test_clean.all.n"] == 4.0
    assert metrics["test_clean.segment.a.mdape"] == pytest.approx(0.10)
    assert metrics["test_clean.segment.b.ppe20"] == pytest.approx(0.5)
    assert metrics["test_clean.loc_level.3.n"] == 2.0
    assert metrics["test_clean.loc_level.0.mdape"] == pytest.approx(0.30)
    assert metrics["test_clean.dedup.n"] == 3.0


def test_conformal_quantile_uses_the_finite_sample_rank():
    errors = [i / 10 for i in range(1, 11)]  # 0.1 .. 1.0
    assert conformal_quantile(errors, 0.20) == pytest.approx(0.9)  # rank ceil(11*0.8) = 9
    assert conformal_quantile(errors, 0.05) == math.inf  # rank 11 > n
    assert conformal_quantile(list(range(1, 20)), 0.05) == 19  # rank ceil(20*0.95) = 19


def test_small_segments_use_the_pooled_quantiles():
    segments = ["big"] * 30 + ["small"] * 5
    errors = [0.1] * 30 + [0.9] * 5
    conformal = fit_conformal(segments, errors, min_rows=20)
    assert conformal["big"] == {"q80": pytest.approx(0.1), "q95": pytest.approx(0.1)}
    assert conformal["small"] == conformal["_pooled"]
    assert quantiles_for(conformal, "never_seen") == conformal["_pooled"]


def test_fit_conformal_refuses_too_few_rows():
    with pytest.raises(ValueError, match="too few validation rows"):
        fit_conformal(["a"] * 5, [0.1] * 5, min_rows=1)


def test_coverage_is_inclusive_at_the_edges():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    assert coverage(actual, np.array([1.0] * 4), np.array([3.0] * 4)) == pytest.approx(0.75)
```

Check on the sliced numbers: in segment b the APEs are 0 and 0.3, so `ppe20` is 0.5. Level 0 is the single row 200 → 260, so its APE is 0.30.

- [ ] **Step 2: Write the failing baseline tests**

`tests/models/price/test_price_baselines.py`:

```python
import dataclasses

import numpy as np
import polars as pl
import pytest

from models.price.baselines import comps_b0, lightgbm_b1
from models.price.config import TrainConfig
from models.price.features import build_feature_set, feature_frame
from models.price.split import add_bulk_groups, assign_split

SCHEMA = {
    "area_id": pl.Int64,
    "segment": pl.Utf8,
    "room_kind": pl.Utf8,
    "bedrooms": pl.Float64,
    "y": pl.Float64,
    "bulk_weight": pl.Float64,
}


def rows(*entries):
    """entries: (area_id, segment, bedrooms, y, bulk_weight); bedrooms None means penthouse."""
    return pl.DataFrame(
        [
            {
                "area_id": a, "segment": s,
                "room_kind": "bedrooms" if b is not None else "penthouse",
                "bedrooms": b, "y": y, "bulk_weight": w,
            }
            for a, s, b, y, w in entries
        ],
        schema=SCHEMA,
    )  # fmt: skip


FIT = rows(
    *[(1, "A", 2.0, float(v), 1.0) for v in range(5)],  # area 1, A, 2 BR: y 0..4
    *[(1, "A", 1.0, 10.0, 1.0)] * 3,  # area 1, A, 1 BR: only 3 sales
    *[(2, "A", 2.0, -5.0, 1.0)] * 4,  # area 2, A: 4 sales
)


def test_comps_fall_back_from_room_to_area_to_segment():
    target = rows(*((a, "A", b, 0.0, 1.0) for a, b in ((1, 2.0), (1, 1.0), (1, 3.0), (9, 2.0))))
    predicted = comps_b0(FIT, target, min_n=5.0)
    # level 0 (area, segment, room): median of 0..4 = 2
    # level 1 (area 1, A): weighted median of [0,1,2,3,4,10,10,10] = 3
    # level 2 (segment A): weighted median of [-5 x4, 0..4, 10 x3] = 1
    assert predicted.tolist() == pytest.approx([2.0, 3.0, 3.0, 1.0])


def test_comps_median_respects_bulk_weights():
    fit = rows(*[(3, "B", 2.0, 0.0, 0.25)] * 4, *[(3, "B", 2.0, 1.0, 1.0)] * 2)
    predicted = comps_b0(fit, rows((3, "B", 2.0, 0.0, 1.0)), min_n=1.0)
    assert predicted.tolist() == pytest.approx([1.0])  # the unweighted median would be 0


def test_comps_raise_for_a_segment_with_no_fit_rows():
    with pytest.raises(ValueError, match="no segment-level median"):
        comps_b0(FIT, rows((1, "Z", 2.0, 0.0, 1.0)), min_n=5.0)


def test_lightgbm_baseline_trains_with_early_stopping(synthetic_homes):
    config = dataclasses.replace(
        TrainConfig(), min_index_sales=1, oof_folds=3, max_rounds=200, early_stopping_rounds=10
    )
    split, clean = pl.col("split"), pl.col("is_clean")
    prepared = add_bulk_groups(assign_split(synthetic_homes(), config))
    fs = build_feature_set(
        prepared, config, prepared["instance_date"].max(),
        fit_filter=(split == "train") & clean, apply_filters={"val": (split == "val") & clean},
    )  # fmt: skip
    fit, val = fs.frames["fit"], fs.frames["val"]
    x_fit, x_val = feature_frame(fit, fs.categories), feature_frame(val, fs.categories)
    booster = lightgbm_b1(
        x_fit, fit["y"].to_numpy(), fs.weights,
        x_val, val["y"].to_numpy(), val["bulk_weight"].to_numpy(), config,
    )  # fmt: skip
    predicted = booster.predict(x_val, num_iteration=booster.best_iteration)
    assert predicted.shape == (val.height,)
    assert np.isfinite(predicted).all()
    assert 1 <= booster.best_iteration <= 200
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/models/price/test_price_evaluate.py tests/models/price/test_price_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.price.evaluate'`.

- [ ] **Step 4: Implement `evaluate.py`**

```python
"""Price metrics, per-slice breakdowns, split-conformal intervals and coverage."""

import math

import numpy as np
import polars as pl

CONFORMAL_LEVELS = {"q80": 0.20, "q95": 0.05}


def price_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ape = np.abs(predicted - actual) / actual
    log_error = np.log(predicted) - np.log(actual)
    return {
        "mdape": float(np.median(ape)),
        "ppe10": float(np.mean(ape <= 0.10)),
        "ppe20": float(np.mean(ape <= 0.20)),
        "rmse_log": float(np.sqrt(np.mean(log_error**2))),
        "n": float(actual.size),
    }


def sliced_metrics(frame: pl.DataFrame, set_name: str) -> dict[str, float]:
    """Metrics overall, per segment, per location level and per bulk group (dedup)."""
    out: dict[str, float] = {}

    def add(slice_name: str, part: pl.DataFrame) -> None:
        if part.height == 0:
            return
        values = price_metrics(part["actual"].to_numpy(), part["predicted"].to_numpy())
        for metric, value in values.items():
            out[f"{set_name}.{slice_name}.{metric}"] = value

    add("all", frame)
    for segment in sorted(frame["segment"].unique().to_list()):
        add(f"segment.{segment}", frame.filter(pl.col("segment") == segment))
    for level in sorted(frame["loc_level"].unique().to_list()):
        add(f"loc_level.{level}", frame.filter(pl.col("loc_level") == level))
    add("dedup", frame.unique("bulk_group", keep="first", maintain_order=True))
    return out


def conformal_quantile(abs_errors, alpha: float) -> float:
    errors = np.sort(np.asarray(abs_errors, dtype=float))
    rank = math.ceil((errors.size + 1) * (1 - alpha))
    if errors.size == 0 or rank > errors.size:
        return math.inf
    return float(errors[rank - 1])


def fit_conformal(segments, abs_errors, min_rows: int) -> dict[str, dict[str, float]]:
    frame = pl.DataFrame({"segment": list(segments), "e": np.asarray(abs_errors, dtype=float)})
    pooled = {
        name: conformal_quantile(frame["e"].to_numpy(), alpha)
        for name, alpha in CONFORMAL_LEVELS.items()
    }
    if not all(math.isfinite(value) for value in pooled.values()):
        raise ValueError(f"too few validation rows ({frame.height}) for a 95% conformal interval")
    out = {"_pooled": pooled}
    for segment in sorted(frame["segment"].unique().to_list()):
        errors = frame.filter(pl.col("segment") == segment)["e"].to_numpy()
        if errors.size >= min_rows:
            out[segment] = {
                name: conformal_quantile(errors, alpha) for name, alpha in CONFORMAL_LEVELS.items()
            }
        else:
            out[segment] = dict(pooled)
    return out


def quantiles_for(conformal: dict[str, dict[str, float]], segment: str) -> dict[str, float]:
    return conformal.get(segment, conformal["_pooled"])


def coverage(actual, low, high) -> float:
    actual = np.asarray(actual, dtype=float)
    return float(np.mean((actual >= low) & (actual <= high)))
```

- [ ] **Step 5: Implement `baselines.py`**

```python
"""B0: index-adjusted comps. B1: LightGBM on the same features (CPU)."""

import lightgbm as lgb
import numpy as np
import polars as pl

from models.price.config import TrainConfig

COMPS_LEVELS = (("area_id", "segment", "room_bucket"), ("area_id", "segment"), ("segment",))


def _room_bucket() -> pl.Expr:
    counted = pl.col("room_kind").is_in(["studio", "bedrooms"])
    bedrooms = pl.col("bedrooms").cast(pl.Int64).cast(pl.Utf8)
    return pl.when(counted).then(bedrooms).otherwise(pl.col("room_kind")).alias("room_bucket")


def _weighted_medians(fit: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Per group: the smallest y whose cumulative bulk weight reaches half the group's weight."""
    return (
        fit.sort("y")
        .with_columns(
            pl.col("bulk_weight").cum_sum().over(keys).alias("__cw"),
            pl.col("bulk_weight").sum().over(keys).alias("__tw"),
        )
        .filter(pl.col("__cw") >= pl.col("__tw") / 2)
        .group_by(keys)
        .agg(pl.col("y").min().alias("med"), pl.col("__tw").first().alias("n"))
    )


def comps_b0(fit: pl.DataFrame, target: pl.DataFrame, min_n: float) -> np.ndarray:
    """Weighted-median relative price of comparable sales, falling back to broader groups."""
    fit = fit.with_columns(_room_bucket())
    out = target.with_columns(_room_bucket()).with_row_index("__row")
    for i, keys in enumerate(COMPS_LEVELS):
        table = _weighted_medians(fit, list(keys)).rename({"med": f"__med{i}", "n": f"__n{i}"})
        out = out.join(table, on=list(keys), how="left")
    out = out.sort("__row")
    pick = (
        pl.when(pl.col("__n0") >= min_n)
        .then(pl.col("__med0"))
        .when(pl.col("__n1") >= min_n)
        .then(pl.col("__med1"))
        .otherwise(pl.col("__med2"))
    )
    predicted = out.select(pick.alias("y_hat"))["y_hat"]
    if predicted.null_count():
        raise ValueError("comps baseline has no segment-level median for some rows")
    return predicted.to_numpy()


def lightgbm_b1(x_fit, y_fit, w_fit, x_val, y_val, w_val, config: TrainConfig) -> lgb.Booster:
    params = {
        "objective": "regression",
        "learning_rate": config.lgbm_learning_rate,
        "seed": config.seed,
        "verbose": -1,
    }
    train = lgb.Dataset(x_fit, label=y_fit, weight=w_fit)
    valid = lgb.Dataset(x_val, label=y_val, weight=w_val, reference=train)
    return lgb.train(
        params,
        train,
        num_boost_round=config.max_rounds,
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/models/price/test_price_evaluate.py tests/models/price/test_price_baselines.py -v`
Expected: PASS.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add models/price/evaluate.py models/price/baselines.py tests/models/price/test_price_evaluate.py tests/models/price/test_price_baselines.py
git commit -m "$(cat <<'EOF'
feat(price): price metrics, conformal intervals, comps and LightGBM baselines

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: XGBoost helpers (GPU/CPU) and Optuna tuning

**Files:**
- Create: `models/price/boosting.py`, `models/price/tune.py`, `tests/models/price/test_price_boosting.py`, `tests/models/price/test_price_tune.py`

**Interfaces:**
- Consumes: `TrainConfig` (Task 1)
- Produces (`models.price.boosting`, which imports xgboost only, never Optuna):
  - `BASE_PARAMS`, `DEVICES = ("auto", "cuda", "cpu")`
  - `cuda_available() -> bool`: XGBoost built with CUDA **and** `nvidia-smi -L` succeeds
  - `resolve_device(requested: str) -> str`
    - `auto` → `cuda` or `cpu`
    - `cuda` without a usable GPU → `RuntimeError`
    - anything else → `ValueError`
  - `make_dmatrix(features, label=None, weight=None) -> xgb.DMatrix`, with `enable_categorical=True`
  - `fit_xgb(params, dtrain, dval, device, config, num_rounds=None) -> xgb.Booster`
    - with `dval`: early stopping, up to `config.max_rounds`
    - without it: exactly `num_rounds`
  - `predict_xgb(booster, features) -> np.ndarray`: uses `best_iteration` when the booster has one
- Produces (`models.price.tune`):
  - `TuneResult`, a frozen dataclass: `best_params: dict`, `best_iteration: int`, `best_value: float`, `n_trials: int`
  - `suggest_params(trial) -> dict`: the spec's search space
  - `tune_xgboost(dtrain, dval, device, config, on_trial=None) -> TuneResult`
    - `on_trial(number: int, params: dict, value: float, best_iteration: int, seconds: float)` is called after each trial

- [ ] **Step 1: Write the failing tests**

`tests/models/price/test_price_boosting.py`:

```python
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from models.price import boosting
from models.price.boosting import (
    cuda_available,
    fit_xgb,
    make_dmatrix,
    predict_xgb,
    resolve_device,
)
from models.price.config import TrainConfig

CONFIG = dataclasses.replace(TrainConfig(), max_rounds=500, early_stopping_rounds=10)
PARAMS = {"max_depth": 3, "learning_rate": 0.3}


def toy_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = 2.0 * x["a"].to_numpy() - x["b"].to_numpy() + rng.normal(scale=0.1, size=n)
    return x, y


def test_resolve_device_validates_and_falls_back(monkeypatch):
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cuda", "cpu")
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("tpu")
    monkeypatch.setattr(boosting, "cuda_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="no usable CUDA GPU"):
        resolve_device("cuda")


def test_early_stopping_fit_and_prediction():
    x, y = toy_data()
    dtrain = make_dmatrix(x[:300], y[:300])
    dval = make_dmatrix(x[300:], y[300:])
    booster = fit_xgb(PARAMS, dtrain, dval, "cpu", CONFIG)
    assert booster.best_iteration < CONFIG.max_rounds - 1
    predicted = predict_xgb(booster, x[300:])
    assert predicted.shape == (100,)
    assert np.sqrt(np.mean((predicted - y[300:]) ** 2)) < 0.5


def test_fixed_round_fit_needs_num_rounds():
    x, y = toy_data()
    with pytest.raises(ValueError, match="num_rounds"):
        fit_xgb(PARAMS, make_dmatrix(x, y), None, "cpu", CONFIG)
    booster = fit_xgb(PARAMS, make_dmatrix(x, y), None, "cpu", CONFIG, num_rounds=7)
    assert booster.num_boosted_rounds() == 7
    assert predict_xgb(booster, x).shape == (400,)  # no best_iteration: uses all trees


@pytest.mark.skipif(not cuda_available(), reason="no CUDA-capable XGBoost + GPU on this machine")
def test_gpu_training_smoke():
    x, y = toy_data()
    booster = fit_xgb(PARAMS, make_dmatrix(x, y), None, "cuda", CONFIG, num_rounds=5)
    device = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    assert device.startswith("cuda")
    assert np.isfinite(predict_xgb(booster, x)).all()
```

`tests/models/price/test_price_tune.py`:

```python
import dataclasses
import math

import numpy as np
import pandas as pd

from models.price.boosting import make_dmatrix
from models.price.config import TrainConfig
from models.price.tune import tune_xgboost

CONFIG = dataclasses.replace(TrainConfig(), n_trials=2, max_rounds=200, early_stopping_rounds=10)
RANGES = {
    "max_depth": (4, 12),
    "learning_rate": (0.02, 0.3),
    "min_child_weight": (1.0, 64.0),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_lambda": (1e-3, 10.0),
    "reg_alpha": (1e-3, 10.0),
}


def test_tune_runs_trials_and_reports_the_best():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    y = x["a"].to_numpy() + rng.normal(scale=0.1, size=500)
    calls = []
    result = tune_xgboost(
        make_dmatrix(x[:400], y[:400]),
        make_dmatrix(x[400:], y[400:]),
        "cpu",
        CONFIG,
        on_trial=lambda number, params, value, best_iteration, seconds: calls.append(
            (number, value, best_iteration)
        ),
    )
    assert result.n_trials == 2
    assert [number for number, _, _ in calls] == [0, 1]
    assert result.best_value == min(value for _, value, _ in calls)
    assert math.isfinite(result.best_value) and result.best_iteration >= 0
    assert set(result.best_params) == set(RANGES)
    for name, (low, high) in RANGES.items():
        assert low <= result.best_params[name] <= high, name
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/models/price/test_price_boosting.py tests/models/price/test_price_tune.py -v`
Expected: FAIL with `ImportError` (the modules don't exist yet).

- [ ] **Step 3: Implement `boosting.py`**

```python
"""XGBoost helpers shared by training and serving (no Optuna imports here)."""

import shutil
import subprocess

import numpy as np
import xgboost as xgb

from models.price.config import TrainConfig

BASE_PARAMS = {"objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse"}
DEVICES = ("auto", "cuda", "cpu")


def cuda_available() -> bool:
    if not xgb.build_info().get("USE_CUDA", False):
        return False
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        probe = subprocess.run([smi, "-L"], capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def resolve_device(requested: str) -> str:
    if requested not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {requested!r}")
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    if requested == "cuda" and not cuda_available():
        raise RuntimeError("device 'cuda' requested but XGBoost has no usable CUDA GPU")
    return requested


def make_dmatrix(features, label=None, weight=None) -> xgb.DMatrix:
    return xgb.DMatrix(features, label=label, weight=weight, enable_categorical=True)


def fit_xgb(
    params: dict,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix | None,
    device: str,
    config: TrainConfig,
    num_rounds: int | None = None,
) -> xgb.Booster:
    full = {**BASE_PARAMS, **params, "device": device, "seed": config.seed}
    if dval is None:
        if num_rounds is None:
            raise ValueError("num_rounds is required when there is no validation set")
        return xgb.train(full, dtrain, num_boost_round=num_rounds)
    return xgb.train(
        full,
        dtrain,
        num_boost_round=config.max_rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )


def _best_iteration(booster) -> int | None:
    try:
        return int(booster.best_iteration)
    except AttributeError:
        return None


def predict_xgb(booster, features) -> np.ndarray:
    dmatrix = make_dmatrix(features)
    best = _best_iteration(booster)
    if best is None:
        return booster.predict(dmatrix)
    return booster.predict(dmatrix, iteration_range=(0, best + 1))
```

- [ ] **Step 4: Implement `tune.py`**

```python
"""Optuna search over XGBoost hyperparameters (weighted val RMSE of the relative target)."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import optuna
import xgboost as xgb

from models.price.boosting import fit_xgb
from models.price.config import TrainConfig

TrialCallback = Callable[[int, dict, float, int, float], None]


@dataclass(frozen=True)
class TuneResult:
    best_params: dict
    best_iteration: int
    best_value: float
    n_trials: int


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 64.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def tune_xgboost(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    device: str,
    config: TrainConfig,
    on_trial: TrialCallback | None = None,
) -> TuneResult:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        started = time.perf_counter()
        booster = fit_xgb(params, dtrain, dval, device, config)
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        value = float(booster.best_score)
        if on_trial is not None:
            on_trial(
                trial.number, params, value, int(booster.best_iteration), time.perf_counter() - started
            )
        return value

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(objective, n_trials=config.n_trials)
    best = study.best_trial
    return TuneResult(
        best_params=dict(best.params),
        best_iteration=int(best.user_attrs["best_iteration"]),
        best_value=float(best.value),
        n_trials=len(study.trials),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/models/price/test_price_boosting.py tests/models/price/test_price_tune.py -v`
Expected: PASS. On this laptop `test_gpu_training_smoke` must **pass**, not skip: the RTX 3060 is present. If it skips, report DONE_WITH_CONCERNS with the output of `uv run python -c "import xgboost; print(xgboost.build_info())"` and `nvidia-smi -L`.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add models/price/boosting.py models/price/tune.py tests/models/price/test_price_boosting.py tests/models/price/test_price_tune.py
git commit -m "$(cat <<'EOF'
feat(price): XGBoost GPU/CPU helpers and Optuna hyperparameter search

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Predictor — request validation, fallback, clipping, intervals, model directory

**Files:**
- Create: `models/price/predictor.py`, `tests/models/price/test_price_predictor.py`
- Modify: `tests/models/price/conftest.py` (add the `tiny_bundle` fixture)

**Interfaces:**
- Consumes:
  - `match_key` (ingestion)
  - `SIZE_BOUNDS`, `FEATURES`, `CATEGORICAL_FEATURES` (Task 1)
  - `MarketIndex`, `INDEX_SCHEMA`, `LocationPriors`, `LEVEL_KEYS`, `add_location_keys`, `add_model_columns`, `feature_frame` (Tasks 4–6)
  - `predict_xgb` (Task 8), `quantiles_for` (Task 7)
- Produces (`models.price.predictor`):
  - `PriceInputError(ValueError)`, with attributes `.field` and `.message`; `str()` is `"{field}: {message}"`
  - `PriceRequest` (pydantic, `extra="forbid"`), with the spec's fields; `PriceRequest.parse(data: dict) -> PriceRequest` raises `PriceInputError`
  - `PriceEstimate` (pydantic): `estimate_aed, range_80, range_95, price_per_sqm_aed, location_level, confidence, market_label, asking_vs_estimate_pct, flags, as_of, model_version`
  - `ModelBundle`, a frozen dataclass: `booster, index, priors, categories, bounds_area, bounds_segment, size_percentiles, areas, aliases, conformal, supported_segments, data_end, metadata`
  - `save_bundle(bundle, path: Path) -> None` and `load_bundle(path: Path) -> ModelBundle`
    - the directory holds `booster.json`, `tables/*.parquet` and `metadata.json`
  - `PricePredictor(bundle)`
    - `PricePredictor.from_dir(path)`
    - `.predict_one(request: PriceRequest | dict) -> PriceEstimate`
    - `.model_version -> str`
  - fixture `tiny_bundle` → a builder `build(y_hat: float | None = None) -> ModelBundle`
    - `y_hat` given: a stub booster that always predicts `y_hat`
    - `None`: a real 5-round XGBoost booster (needed to save the bundle)

The tiny bundle's numbers, which the tests rely on:
- **Market index:** `ln(10,000)` for every supported segment in 2023-03.
- **Data end:** 2023-03-17.
- **Priors:** 5 unit sales in area 10 (project "Marina Gate", building "Marina Gate 1", y = 0.1) and 5 villa sales in area 20 (y = 0).
- **Conformal:** pooled q80 0.1 and q95 0.2; `unit_ready_built_up` q80 0.15 and q95 0.3.
- **Bounds:** area bounds only for (unit, built_up, area 10), lo −0.5 and hi 0.5. Segment bounds lo −1 and hi 1.
- **Size percentiles:** `unit_ready_built_up` p005 30, p995 400.
- **Aliases:**
  - "marsa dubai", "dubai marina", "jbr" → 10
  - "barsha south fourth", "jvc" → 20
  - "mushrif" → 30 and 31
- **Supported segments:** everything except `villa_off_plan_plot`.

So a y_hat=0 apartment of 100 m² estimates `exp(ln 10,000) × 100 = 1,000,000`.

- [ ] **Step 1: Add the `tiny_bundle` fixture**

Append to `tests/models/price/conftest.py` (merge the imports into its import block):

```python
import pandas as pd
import xgboost as xgb

from models.price.config import CATEGORICAL_FEATURES, FEATURES
from models.price.features import INDEX_SCHEMA, LocationPriors, MarketIndex, add_location_keys
from models.price.predictor import ModelBundle

SUPPORTED = (
    "unit_off_plan_built_up", "unit_ready_built_up",
    "villa_off_plan_built_up", "villa_ready_built_up", "villa_ready_plot",
)  # fmt: skip
TINY_CATEGORIES = {
    "property_type": ["unit", "villa"],
    "reg_type": ["off_plan", "ready"],
    "size_basis": ["built_up", "plot"],
    "sub_kind": ["flat", "hotel_apartment", "townhouse", "villa"],
    "room_kind": ["bedrooms", "penthouse", "single_room", "studio", "unknown"],
    "area_id": ["10", "20"],
}


class ConstantBooster:
    """Stands in for an XGBoost booster: always predicts the same relative price."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, dmatrix, **_):
        return np.full(dmatrix.num_row(), self.value)


def _train_tiny_booster() -> xgb.Booster:
    rng = np.random.default_rng(0)
    columns = {}
    for name in FEATURES:
        if name in CATEGORICAL_FEATURES:
            values = rng.choice(TINY_CATEGORIES[name], size=64)
            columns[name] = pd.Categorical(values, categories=TINY_CATEGORIES[name])
        else:
            columns[name] = rng.normal(size=64)
    frame = pd.DataFrame(columns)[list(FEATURES)]
    dtrain = xgb.DMatrix(frame, label=rng.normal(scale=0.05, size=64), enable_categorical=True)
    return xgb.train({"max_depth": 2, "tree_method": "hist"}, dtrain, num_boost_round=5)


def build_tiny_bundle(y_hat: float | None = None) -> ModelBundle:
    march = date(2023, 3, 1)
    index = MarketIndex(
        pl.DataFrame(
            [(segment, march, math.log(10_000.0), "segment_3") for segment in SUPPORTED],
            schema=INDEX_SCHEMA,
            orient="row",
        )
    )
    fit = add_location_keys(
        pl.DataFrame(
            {
                "property_type": ["unit"] * 5 + ["villa"] * 5,
                "area_id": [10] * 5 + [20] * 5,
                "project_name": ["Marina Gate"] * 5 + [None] * 5,
                "building_name": ["Marina Gate 1"] * 5 + [None] * 5,
                "y": [0.1] * 5 + [0.0] * 5,
                "bulk_weight": [1.0] * 10,
            }
        )
    )
    return ModelBundle(
        booster=_train_tiny_booster() if y_hat is None else ConstantBooster(y_hat),
        index=index,
        priors=LocationPriors.fit(fit, shrink_k=10.0, min_level_n=3.0),
        categories=TINY_CATEGORIES,
        bounds_area=pl.DataFrame(
            {"property_type": ["unit"], "size_basis": ["built_up"], "area_id": [10],
             "lo": [-0.5], "hi": [0.5]}
        ),  # fmt: skip
        bounds_segment=pl.DataFrame(
            {"property_type": ["unit", "villa", "villa"], "size_basis": ["built_up", "built_up", "plot"],
             "lo": [-1.0] * 3, "hi": [1.0] * 3}
        ),  # fmt: skip
        size_percentiles=pl.DataFrame(
            {"segment": ["unit_ready_built_up"], "p005": [30.0], "p995": [400.0]}
        ),
        areas=pl.DataFrame(
            {"area_id": [10, 20, 30, 31],
             "name_en": ["Marsa Dubai", "Al Barsha South Fourth", "Mushrif", "Mushrif"]}
        ),  # fmt: skip
        aliases=pl.DataFrame(
            {"alias_key": ["barsha south fourth", "dubai marina", "jbr", "jvc", "marsa dubai",
                           "mushrif", "mushrif"],
             "area_id": [20, 10, 10, 20, 10, 30, 31]}
        ),  # fmt: skip
        conformal={
            "_pooled": {"q80": 0.1, "q95": 0.2},
            "unit_ready_built_up": {"q80": 0.15, "q95": 0.3},
        },
        supported_segments=SUPPORTED,
        data_end=date(2023, 3, 17),
        metadata={"model_version": "run-abc"},
    )


@pytest.fixture
def tiny_bundle():
    return build_tiny_bundle
```

- [ ] **Step 2: Write the failing tests**

`tests/models/price/test_price_predictor.py`:

```python
import math
from datetime import date

import pytest

from models.price.predictor import (
    PriceInputError,
    PricePredictor,
    PriceRequest,
    load_bundle,
    save_bundle,
)

REQ = {"area": "Dubai Marina", "property_kind": "apartment", "status": "ready", "size_sqm": 100.0}


def predict(bundle, **changes):
    return PricePredictor(bundle).predict_one({**REQ, **changes})


def test_estimate_ranges_and_metadata(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0))
    assert estimate.estimate_aed == pytest.approx(1_000_000.0)
    assert estimate.price_per_sqm_aed == pytest.approx(10_000.0)
    assert estimate.range_80 == pytest.approx((1e6 * math.exp(-0.15), 1e6 * math.exp(0.15)))
    assert estimate.range_95 == pytest.approx((1e6 * math.exp(-0.3), 1e6 * math.exp(0.3)))
    assert (estimate.location_level, estimate.confidence) == ("area", "medium")
    assert estimate.as_of == date(2023, 3, 17)
    assert estimate.model_version == "run-abc"
    assert estimate.flags == []
    assert estimate.market_label is None and estimate.asking_vs_estimate_pct is None


def test_known_building_resolves_at_building_level(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0), project="Marina Gate", building="MARINA GATE 1")
    assert (estimate.location_level, estimate.confidence) == ("building", "high")
    assert estimate.flags == []


def test_unknown_building_falls_back_and_is_flagged(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0), project="Marina Gate", building="Marina Gate 9")
    assert (estimate.location_level, estimate.confidence) == ("project", "high")
    assert estimate.flags == ["location_fallback"]


def test_alias_resolves_and_an_area_without_sales_is_city_level(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0), area="JVC")
    assert (estimate.location_level, estimate.confidence) == ("city", "low")
    assert estimate.estimate_aed > 0


def test_area_id_can_be_given_directly(tiny_bundle):
    bundle = tiny_bundle(y_hat=0.0)
    request = {**REQ, "area": None, "area_id": 10}
    assert PricePredictor(bundle).predict_one(request).location_level == "area"
    with pytest.raises(PriceInputError) as info:
        PricePredictor(bundle).predict_one({**request, "area_id": 999})
    assert info.value.field == "area_id"


@pytest.mark.parametrize(
    ("area", "message"), [("Atlantis", "unknown area"), ("Mushrif", "ambiguous")]
)
def test_unknown_and_ambiguous_area_names_are_rejected(tiny_bundle, area, message):
    with pytest.raises(PriceInputError, match=message) as info:
        predict(tiny_bundle(y_hat=0.0), area=area)
    assert info.value.field == "area"


@pytest.mark.parametrize("changes", [{"area_id": 10}, {"area": None}])
def test_exactly_one_of_area_and_area_id(tiny_bundle, changes):
    with pytest.raises(PriceInputError, match="exactly one of area or area_id"):
        predict(tiny_bundle(y_hat=0.0), **changes)


def test_implausible_predictions_are_clipped_and_flagged(tiny_bundle):
    high = predict(tiny_bundle(y_hat=5.0))
    assert high.estimate_aed == pytest.approx(1e6 * math.exp(0.5))  # area 10 bound hi = 0.5
    assert high.flags == ["implausible_clipped"]
    low = predict(tiny_bundle(y_hat=-50.0))
    assert low.estimate_aed == pytest.approx(1e6 * math.exp(-0.5))
    assert low.estimate_aed > 0
    villa = predict(
        tiny_bundle(y_hat=5.0), area="JVC", property_kind="villa", size_basis="plot", size_sqm=500.0
    )
    assert villa.estimate_aed == pytest.approx(10_000.0 * 500.0 * math.exp(1.0))  # segment bound


def test_market_labels_at_the_interval_edges(tiny_bundle):
    bundle = tiny_bundle(y_hat=0.0)
    low, high = predict(bundle).range_80
    assert predict(bundle, asking_price_aed=high).market_label == "fair"
    assert predict(bundle, asking_price_aed=low).market_label == "fair"
    assert predict(bundle, asking_price_aed=high * 1.01).market_label == "above_market"
    below = predict(bundle, asking_price_aed=low * 0.99)
    assert below.market_label == "below_market"
    assert predict(bundle, asking_price_aed=1_100_000.0).asking_vs_estimate_pct == 10.0


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"size_sqm": 11.0}, "size_sqm"),
        ({"size_sqm": 2_500.0}, "size_sqm"),
        ({"size_sqm": -5.0}, "size_sqm"),
        ({"size_basis": "plot"}, "size_basis"),
        ({"bedrooms": 9}, "bedrooms"),
        ({"is_penthouse": True, "property_kind": "villa", "size_sqm": 300.0}, "is_penthouse"),
        ({"property_kind": "villa", "status": "off_plan", "size_basis": "plot",
          "size_sqm": 400.0}, "status"),
        ({"property_kind": "castle"}, "property_kind"),
        ({"surprise": 1}, "surprise"),
    ],
)  # fmt: skip
def test_invalid_requests_raise_price_input_error(tiny_bundle, changes, field):
    with pytest.raises(PriceInputError) as info:
        predict(tiny_bundle(y_hat=0.0), **changes)
    assert info.value.field == field


def test_missing_required_field_names_the_field():
    with pytest.raises(PriceInputError) as info:
        PriceRequest.parse({"area": "JBR", "status": "ready", "size_sqm": 80.0})
    assert info.value.field == "property_kind"
    assert str(info.value).startswith("property_kind: ")


def test_unusual_size_is_flagged_not_rejected(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0), size_sqm=25.0)
    assert estimate.flags == ["unusual_size"]
    assert estimate.estimate_aed == pytest.approx(250_000.0)


def test_penthouse_and_studio_requests_are_accepted(tiny_bundle):
    bundle = tiny_bundle(y_hat=0.0)
    assert predict(bundle, is_penthouse=True, bedrooms=3).estimate_aed > 0
    assert predict(bundle, bedrooms=0).estimate_aed > 0


def test_bundle_round_trips_through_a_directory(tiny_bundle, tmp_path):
    bundle = tiny_bundle()  # real booster
    save_bundle(bundle, tmp_path / "model")
    assert (tmp_path / "model" / "booster.json").is_file()
    loaded = load_bundle(tmp_path / "model")
    assert loaded.supported_segments == bundle.supported_segments
    assert loaded.conformal == bundle.conformal
    assert loaded.data_end == bundle.data_end
    direct = PricePredictor(bundle).predict_one(REQ)
    reloaded = PricePredictor.from_dir(tmp_path / "model").predict_one(REQ)
    assert reloaded.estimate_aed == pytest.approx(direct.estimate_aed)
    assert reloaded.model_version == "run-abc"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/models/price/test_price_predictor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.price.predictor'`. Because `conftest.py` now imports it, the whole directory errors until Step 4.

- [ ] **Step 4: Implement `predictor.py`**

```python
"""DB-free home price predictor: validation, location fallback, clipping, intervals, flags."""

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl
import xgboost as xgb
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ingestion.normalize import match_key
from models.price.boosting import predict_xgb
from models.price.config import SIZE_BOUNDS
from models.price.evaluate import quantiles_for
from models.price.features import (
    LEVEL_KEYS,
    LocationPriors,
    MarketIndex,
    add_location_keys,
    add_model_columns,
    feature_frame,
)

KIND_TO_TYPE = {
    "apartment": ("unit", "flat"),
    "hotel_apartment": ("unit", "hotel_apartment"),
    "townhouse": ("unit", "townhouse"),
    "villa": ("villa", "villa"),
}
LEVEL_NAMES = {3: "building", 2: "project", 1: "area", 0: "city"}
CONFIDENCE = {3: "high", 2: "high", 1: "medium", 0: "low"}
TABLE_NAMES = (
    "market_index", "prior_city", "prior_area", "prior_project", "prior_building",
    "bounds_area", "bounds_segment", "size_percentiles", "areas", "aliases",
)  # fmt: skip
ROW_SCHEMA = {
    "property_type": pl.Utf8,
    "reg_type": pl.Utf8,
    "size_basis": pl.Utf8,
    "sub_kind": pl.Utf8,
    "room_kind": pl.Utf8,
    "bedrooms": pl.Float64,
    "area_sqm": pl.Float64,
    "has_parking": pl.Boolean,
    "area_id": pl.Int64,
    "project_name": pl.Utf8,
    "building_name": pl.Utf8,
    "segment": pl.Utf8,
}


class PriceInputError(ValueError):
    """A request the model must not answer; the API maps it to HTTP 422."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


class PriceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: str | None = None
    area_id: int | None = None
    project: str | None = None
    building: str | None = None
    property_kind: Literal["apartment", "hotel_apartment", "townhouse", "villa"]
    status: Literal["ready", "off_plan"]
    size_sqm: float = Field(gt=0)
    size_basis: Literal["built_up", "plot"] = "built_up"
    bedrooms: int | None = Field(default=None, ge=0, le=8)
    is_penthouse: bool = False
    has_parking: bool | None = None
    asking_price_aed: float | None = Field(default=None, gt=0)

    @classmethod
    def parse(cls, data: dict) -> "PriceRequest":
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            error = exc.errors()[0]
            field = ".".join(str(part) for part in error["loc"]) or "request"
            raise PriceInputError(field, error["msg"]) from None


class PriceEstimate(BaseModel):
    estimate_aed: float
    range_80: tuple[float, float]
    range_95: tuple[float, float]
    price_per_sqm_aed: float
    location_level: Literal["building", "project", "area", "city"]
    confidence: Literal["high", "medium", "low"]
    market_label: Literal["below_market", "fair", "above_market"] | None
    asking_vs_estimate_pct: float | None
    flags: list[str]
    as_of: date
    model_version: str


@dataclass(frozen=True)
class ModelBundle:
    booster: object  # xgb.Booster; tests may pass any object with .predict(dmatrix)
    index: MarketIndex
    priors: LocationPriors
    categories: dict[str, list[str]]
    bounds_area: pl.DataFrame
    bounds_segment: pl.DataFrame
    size_percentiles: pl.DataFrame
    areas: pl.DataFrame
    aliases: pl.DataFrame
    conformal: dict[str, dict[str, float]]
    supported_segments: tuple[str, ...]
    data_end: date
    metadata: dict


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    tables = path / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    bundle.booster.save_model(str(path / "booster.json"))
    frames = {
        "market_index": bundle.index.table,
        **{f"prior_{level}": table for level, table in bundle.priors.stats.items()},
        "bounds_area": bundle.bounds_area,
        "bounds_segment": bundle.bounds_segment,
        "size_percentiles": bundle.size_percentiles,
        "areas": bundle.areas,
        "aliases": bundle.aliases,
    }
    for name, frame in frames.items():
        frame.write_parquet(tables / f"{name}.parquet")
    metadata = {
        "categories": bundle.categories,
        "conformal": bundle.conformal,
        "supported_segments": list(bundle.supported_segments),
        "data_end": bundle.data_end.isoformat(),
        "shrink_k": bundle.priors.shrink_k,
        "min_level_n": bundle.priors.min_level_n,
        "metadata": bundle.metadata,
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def load_bundle(path: Path) -> ModelBundle:
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    tables = {name: pl.read_parquet(path / "tables" / f"{name}.parquet") for name in TABLE_NAMES}
    booster = xgb.Booster()
    booster.load_model(str(path / "booster.json"))
    booster.set_param({"device": "cpu"})  # serving never assumes a GPU
    priors = LocationPriors(
        {level: tables[f"prior_{level}"] for level in LEVEL_KEYS}, meta["shrink_k"], meta["min_level_n"]
    )
    return ModelBundle(
        booster=booster,
        index=MarketIndex(tables["market_index"]),
        priors=priors,
        categories=meta["categories"],
        bounds_area=tables["bounds_area"],
        bounds_segment=tables["bounds_segment"],
        size_percentiles=tables["size_percentiles"],
        areas=tables["areas"],
        aliases=tables["aliases"],
        conformal=meta["conformal"],
        supported_segments=tuple(meta["supported_segments"]),
        data_end=date.fromisoformat(meta["data_end"]),
        metadata=meta["metadata"],
    )


def _room_fields(request: PriceRequest) -> tuple[str, float | None]:
    if request.is_penthouse:
        return "penthouse", None
    if request.bedrooms is None:
        return "unknown", None
    if request.bedrooms == 0:
        return "studio", 0.0
    return "bedrooms", float(request.bedrooms)


class PricePredictor:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self._area_ids = set(bundle.areas["area_id"].to_list())

    @classmethod
    def from_dir(cls, path: Path) -> "PricePredictor":
        return cls(load_bundle(Path(path)))

    @property
    def model_version(self) -> str:
        return str(self.bundle.metadata.get("model_version", "unregistered"))

    def _validate(self, request: PriceRequest) -> tuple[str, str, str]:
        if (request.area is None) == (request.area_id is None):
            raise PriceInputError("area", "give exactly one of area or area_id")
        property_type, sub_kind = KIND_TO_TYPE[request.property_kind]
        if request.size_basis == "plot" and property_type != "villa":
            raise PriceInputError("size_basis", "'plot' is only valid for a villa")
        if request.is_penthouse and request.property_kind not in ("apartment", "hotel_apartment"):
            raise PriceInputError("is_penthouse", "only valid for an apartment or hotel apartment")
        low, high = SIZE_BOUNDS[(property_type, request.size_basis)]
        if not low <= request.size_sqm <= high:
            raise PriceInputError(
                "size_sqm",
                f"must be between {low:g} and {high:g} m² for a {request.property_kind} "
                f"({request.size_basis})",
            )
        segment = f"{property_type}_{request.status}_{request.size_basis}"
        if segment not in self.bundle.supported_segments:
            raise PriceInputError(
                "status",
                f"{request.property_kind} + {request.status} + {request.size_basis} "
                "is not supported by this model",
            )
        return property_type, sub_kind, segment

    def _resolve_area(self, request: PriceRequest) -> int:
        if request.area_id is not None:
            if request.area_id not in self._area_ids:
                raise PriceInputError("area_id", f"unknown area_id {request.area_id}")
            return request.area_id
        key = match_key(request.area)
        aliases = self.bundle.aliases
        ids = sorted(set(aliases.filter(pl.col("alias_key") == key)["area_id"].to_list())) if key else []
        if not ids:
            raise PriceInputError("area", f"unknown area {request.area!r}")
        if len(ids) > 1:
            raise PriceInputError("area", f"area {request.area!r} is ambiguous: use area_id, one of {ids}")
        return ids[0]

    def _y_bounds(self, property_type: str, size_basis: str, area_id: int) -> tuple[float, float]:
        same_kind = (pl.col("property_type") == property_type) & (pl.col("size_basis") == size_basis)
        table = self.bundle.bounds_area.filter(same_kind & (pl.col("area_id") == area_id))
        if table.height == 0:
            table = self.bundle.bounds_segment.filter(same_kind)
        if table.height != 1:
            raise RuntimeError(f"no plausibility bounds for {property_type}/{size_basis}")
        return float(table["lo"][0]), float(table["hi"][0])

    def _unusual_size(self, segment: str, size: float) -> bool:
        row = self.bundle.size_percentiles.filter(pl.col("segment") == segment)
        if row.height == 0:
            return False
        return not float(row["p005"][0]) <= size <= float(row["p995"][0])

    def predict_one(self, request: PriceRequest | dict) -> PriceEstimate:
        if not isinstance(request, PriceRequest):
            request = PriceRequest.parse(request)
        property_type, sub_kind, segment = self._validate(request)
        area_id = self._resolve_area(request)
        index_value = self.bundle.index.value_at(segment, self.bundle.data_end)
        room_kind, bedrooms = _room_fields(request)
        row = pl.DataFrame(
            {
                "property_type": [property_type],
                "reg_type": [request.status],
                "size_basis": [request.size_basis],
                "sub_kind": [sub_kind],
                "room_kind": [room_kind],
                "bedrooms": [bedrooms],
                "area_sqm": [float(request.size_sqm)],
                "has_parking": [request.has_parking],
                "area_id": [area_id],
                "project_name": [request.project],
                "building_name": [request.building],
                "segment": [segment],
            },
            schema=ROW_SCHEMA,
        )
        row = add_location_keys(add_model_columns(row))
        row = self.bundle.priors.transform(row.with_columns(pl.lit(index_value).alias("market_index")))
        y_hat = float(predict_xgb(self.bundle.booster, feature_frame(row, self.bundle.categories))[0])

        flags = []
        low, high = self._y_bounds(property_type, request.size_basis, area_id)
        if not low <= y_hat <= high:
            y_hat = min(max(y_hat, low), high)
            flags.append("implausible_clipped")
        unseen_project = request.project is not None and row["n_project"][0] == 0.0
        unseen_building = request.building is not None and row["n_building"][0] == 0.0
        if unseen_project or unseen_building:
            flags.append("location_fallback")
        if self._unusual_size(segment, request.size_sqm):
            flags.append("unusual_size")

        size = float(request.size_sqm)
        centre = y_hat + index_value
        q = quantiles_for(self.bundle.conformal, segment)
        estimate = math.exp(centre) * size
        range_80 = (math.exp(centre - q["q80"]) * size, math.exp(centre + q["q80"]) * size)
        range_95 = (math.exp(centre - q["q95"]) * size, math.exp(centre + q["q95"]) * size)
        label, pct = None, None
        if request.asking_price_aed is not None:
            asking = request.asking_price_aed
            if asking < range_80[0]:
                label = "below_market"
            elif asking > range_80[1]:
                label = "above_market"
            else:
                label = "fair"
            pct = round((asking / estimate - 1.0) * 100.0, 1)
        level = int(row["loc_level"][0])
        return PriceEstimate(
            estimate_aed=estimate,
            range_80=range_80,
            range_95=range_95,
            price_per_sqm_aed=estimate / size,
            location_level=LEVEL_NAMES[level],
            confidence=CONFIDENCE[level],
            market_label=label,
            asking_vs_estimate_pct=pct,
            flags=flags,
            as_of=self.bundle.data_end,
            model_version=self.model_version,
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/models/price/ -v`
Expected: PASS (every Phase 3 test so far).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add models/price/predictor.py tests/models/price/conftest.py tests/models/price/test_price_predictor.py
git commit -m "$(cat <<'EOF'
feat(price): DB-free predictor with validation, location fallback, clipping and intervals

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: MLflow pyfunc, registry, evaluation plots

**Files:**
- Create: `models/price/pyfunc.py`, `models/price/registry.py`, `models/price/plots.py`, `tests/models/price/test_price_registry.py`, `tests/models/price/test_price_plots.py`
- Modify: `tests/models/price/conftest.py` (add the `temp_mlflow` fixture)

**Interfaces:**
- Consumes: `PricePredictor`, `PriceRequest`, `save_bundle` (Task 9); the `tiny_bundle` fixture
- Produces:
  - `models.price.pyfunc`:
    - `CODE_PATHS: list[str]`: absolute paths to `models/` and `ingestion/`
    - `pip_requirements() -> list[str]`: exact `==` pins
    - `PricePyfunc(mlflow.pyfunc.PythonModel)`
      - `load_context` builds `self.predictor`
      - `predict(context, model_input: pd.DataFrame)` returns one row per request, with `PriceEstimate` fields; NaN or None cells count as absent fields
  - `models.price.registry`:
    - `configure(experiment, tracking_uri=None, artifact_location=None) -> None`: creates the experiment with `artifact_location` when it's new
    - `log_metrics(metrics: dict[str, float]) -> None`: drops non-finite values and batches 500 at a time
    - `log_price_model(model_dir: Path) -> str`: returns the `runs:/…/model` URI; must be called inside an active run
    - `register_champion(model_uri, name) -> str`: registers a version, points alias `champion` at it, and returns the version number
  - `models.price.plots.save_eval_plots(frame, importance: dict[str, float], out_dir: Path) -> list[Path]`
    - writes `feature_importance.png`, `residuals_by_segment.png`, `error_by_loc_level.png` and `pred_vs_actual.png`
    - the frame has `actual, predicted, segment, loc_level`
  - fixture `temp_mlflow` → `{"tracking_uri": "sqlite:///…/mlflow.db", "artifact_location": "file:///…/artifacts"}`
    - redirects both the MLflow environment variable and MLflow's module-level tracking URI for the test, and restores them afterwards

- [ ] **Step 1: Add the `temp_mlflow` fixture**

Append to `tests/models/price/conftest.py`:

```python
@pytest.fixture
def temp_mlflow(tmp_path, monkeypatch):
    """Throwaway MLflow tracking + registry store. Never the real server, never ./mlruns."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    # mlflow.set_tracking_uri() writes this module global; monkeypatch restores it after the test.
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", uri)
    monkeypatch.setattr("mlflow.tracking.fluent._active_experiment_id", None)
    yield {"tracking_uri": uri, "artifact_location": (tmp_path / "artifacts").as_uri()}
    while mlflow.active_run():
        mlflow.end_run()
```

If either private attribute path doesn't exist in mlflow 2.17.2, find the module global that `mlflow.set_tracking_uri` writes (`python -c "import inspect, mlflow; print(inspect.getsource(mlflow.set_tracking_uri))"`) and patch that instead. Report the change as a concern.

- [ ] **Step 2: Write the failing tests**

`tests/models/price/test_price_registry.py`:

```python
import mlflow
import pandas as pd

from models.price.predictor import save_bundle
from models.price.pyfunc import pip_requirements
from models.price.registry import configure, log_metrics, log_price_model, register_champion

REQUESTS = pd.DataFrame(
    [
        {"area": "Dubai Marina", "property_kind": "apartment", "status": "ready",
         "size_sqm": 100.0, "bedrooms": 2},
        {"area": "JBR", "property_kind": "apartment", "status": "ready",
         "size_sqm": 80.0, "bedrooms": None},
    ]
)  # fmt: skip


def test_configure_points_mlflow_at_the_temporary_store(temp_mlflow):
    configure("price-test", temp_mlflow["tracking_uri"], temp_mlflow["artifact_location"])
    assert mlflow.get_tracking_uri() == temp_mlflow["tracking_uri"]
    experiment = mlflow.get_experiment_by_name("price-test")
    assert experiment.artifact_location == temp_mlflow["artifact_location"]


def test_log_register_load_and_predict(temp_mlflow, tiny_bundle, tmp_path):
    configure("price-test", temp_mlflow["tracking_uri"], temp_mlflow["artifact_location"])
    save_bundle(tiny_bundle(), tmp_path / "model_dir")
    with mlflow.start_run(run_name="xgb-production"):
        log_metrics({"test_clean.all.mdape": 0.12, "not_finite": float("nan")})
        model_uri = log_price_model(tmp_path / "model_dir")
    assert register_champion(model_uri, "dubimator-price-test") == "1"

    model = mlflow.pyfunc.load_model("models:/dubimator-price-test@champion")
    out = model.predict(REQUESTS)
    assert len(out) == 2
    assert (out["estimate_aed"] > 0).all()
    assert model.unwrap_python_model().predictor.model_version == "run-abc"

    runs = mlflow.search_runs(experiment_names=["price-test"])
    assert runs["metrics.test_clean.all.mdape"].iloc[0] == 0.12
    assert "metrics.not_finite" not in runs.columns


def test_registering_again_moves_the_champion_alias(temp_mlflow, tiny_bundle, tmp_path):
    configure("price-test", temp_mlflow["tracking_uri"], temp_mlflow["artifact_location"])
    save_bundle(tiny_bundle(), tmp_path / "model_dir")
    for expected in ("1", "2"):
        with mlflow.start_run():
            uri = log_price_model(tmp_path / "model_dir")
        assert register_champion(uri, "dubimator-price-test") == expected
    alias = mlflow.MlflowClient().get_model_version_by_alias("dubimator-price-test", "champion")
    assert alias.version == "2"


def test_pip_requirements_are_exact_pins():
    requirements = pip_requirements()
    assert any(line.startswith("xgboost==") for line in requirements)
    assert all("==" in line for line in requirements)
```

`tests/models/price/test_price_plots.py`:

```python
import polars as pl

from models.price.plots import save_eval_plots


def test_writes_the_four_evaluation_plots(tmp_path):
    frame = pl.DataFrame(
        {
            "actual": [100.0, 200.0, 300.0, 400.0] * 5,
            "predicted": [110.0, 190.0, 330.0, 380.0] * 5,
            "segment": ["unit_ready_built_up", "villa_ready_plot"] * 10,
            "loc_level": [0, 1, 2, 3] * 5,
        }
    )
    paths = save_eval_plots(frame, {"prior_building": 3.0, "log_area_sqm": 1.0}, tmp_path / "plots")
    assert sorted(path.name for path in paths) == [
        "error_by_loc_level.png", "feature_importance.png",
        "pred_vs_actual.png", "residuals_by_segment.png",
    ]  # fmt: skip
    assert all(path.stat().st_size > 0 for path in paths)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/models/price/test_price_registry.py tests/models/price/test_price_plots.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.price.pyfunc'`.

- [ ] **Step 4: Implement `pyfunc.py`**

```python
"""MLflow pyfunc wrapper so serving loads one artifact: models:/dubimator-price@champion."""

import importlib.metadata
import math
from pathlib import Path

import mlflow
import pandas as pd

from models.price.predictor import PricePredictor, PriceRequest

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = [str(REPO_ROOT / "models"), str(REPO_ROOT / "ingestion")]
RUNTIME_PACKAGES = ("mlflow", "xgboost", "polars", "pandas", "pyarrow", "pydantic", "numpy")


def pip_requirements() -> list[str]:
    return [f"{name}=={importlib.metadata.version(name)}" for name in RUNTIME_PACKAGES]


def _present(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if not (value is None or (isinstance(value, float) and math.isnan(value)))
    }


class PricePyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor = PricePredictor.from_dir(Path(context.artifacts["model_dir"]))

    def predict(self, context, model_input, params=None):
        records = model_input.to_dict(orient="records")
        estimates = [
            self.predictor.predict_one(PriceRequest.parse(_present(record))).model_dump(mode="json")
            for record in records
        ]
        return pd.DataFrame(estimates)
```

- [ ] **Step 5: Implement `registry.py`**

```python
"""MLflow logging and model registration for the price model."""

import math
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from models.price.pyfunc import CODE_PATHS, PricePyfunc, pip_requirements

CHAMPION_ALIAS = "champion"


def configure(
    experiment: str, tracking_uri: str | None = None, artifact_location: str | None = None
) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if artifact_location and mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_location)
    mlflow.set_experiment(experiment)


def log_metrics(metrics: dict[str, float]) -> None:
    items = [(key, float(value)) for key, value in metrics.items() if math.isfinite(float(value))]
    for start in range(0, len(items), 500):
        mlflow.log_metrics(dict(items[start : start + 500]))


def log_price_model(model_dir: Path) -> str:
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=PricePyfunc(),
        artifacts={"model_dir": str(model_dir)},
        code_paths=CODE_PATHS,
        pip_requirements=pip_requirements(),
    )
    return info.model_uri


def register_champion(model_uri: str, name: str) -> str:
    version = mlflow.register_model(model_uri, name)
    MlflowClient().set_registered_model_alias(name, CHAMPION_ALIAS, version.version)
    return str(version.version)
```

- [ ] **Step 6: Implement `plots.py`**

```python
"""Evaluation plots logged on the champion-eval run."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def save_eval_plots(frame: pl.DataFrame, importance: dict[str, float], out_dir: Path) -> list[Path]:
    plt.switch_backend("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)

    actual = frame["actual"].to_numpy()
    predicted = frame["predicted"].to_numpy()

    names = sorted(importance, key=importance.get)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, [importance[name] for name in names])
    ax.set_xlabel("total gain")
    ax.set_title("Feature importance")
    save(fig, "feature_importance.png")

    log_ratio = np.log(predicted / actual)
    segment_values = frame["segment"].to_numpy()
    segments = sorted(frame["segment"].unique().to_list())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([log_ratio[segment_values == s] for s in segments], showfliers=False)
    ax.set_xticks(range(1, len(segments) + 1), segments, rotation=20)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_ylabel("ln(predicted / actual)")
    ax.set_title("Residuals by segment")
    save(fig, "residuals_by_segment.png")

    ape = np.abs(predicted - actual) / actual
    levels_values = frame["loc_level"].to_numpy()
    levels = sorted(frame["loc_level"].unique().to_list())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(level) for level in levels], [float(np.median(ape[levels_values == level])) for level in levels])
    ax.set_xlabel("location level (0 city, 1 area, 2 project, 3 building)")
    ax.set_ylabel("MdAPE")
    ax.set_title("Error by location level")
    save(fig, "error_by_loc_level.png")

    x, y = np.log10(actual), np.log10(predicted)
    fig, ax = plt.subplots(figsize=(6, 6))
    cells = ax.hexbin(x, y, gridsize=60, bins="log", mincnt=1)
    limits = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(limits, limits, color="red", linewidth=0.8)
    ax.set_xlabel("log10 actual price (AED)")
    ax.set_ylabel("log10 predicted price (AED)")
    ax.set_title("Predicted vs actual")
    fig.colorbar(cells, ax=ax)
    save(fig, "pred_vs_actual.png")
    return paths
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/models/price/ -v`
Expected: PASS. Afterwards `git status --short` must show **no** new `mlruns/` directory or `*.db` file in the repo root; the tests only write under `tmp_path`.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add models/price/pyfunc.py models/price/registry.py models/price/plots.py tests/models/price/conftest.py tests/models/price/test_price_registry.py tests/models/price/test_price_plots.py
git commit -m "$(cat <<'EOF'
feat(price): MLflow pyfunc wrapper, champion registration and evaluation plots

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Training orchestration, real-data fixture, integration test

**Files:**
- Create: `models/price/train.py`, `scripts/build_price_fixture.py`, `tests/fixtures/price_sample.csv` (generated), `tests/models/price/test_price_fixture.py`, `tests/models/price/test_price_train_integration.py`

**Interfaces:**
- Consumes: everything from Tasks 1–10. From Phase 2: `ingestion.pipeline.run_pipeline` and `ingestion.schema.EXPECTED_COLUMNS`
- Produces (`models.price.train`):
  - `TrainingSummary`, a dataclass
    - `device: str`, `rows: dict[str, int]`, `drop_counts: dict[str, int]`
    - `metrics: dict[str, dict[str, float]]`, keyed by run name (`b0-comps`, `b1-lightgbm`, `xgb-champion-eval`)
    - `best_params: dict`, `seconds: dict[str, float]`, `gate_passed: bool`
    - `model_uri: str | None`, `registered_version: str | None`
  - `run_training(settings, config, device="auto", register=True, tracking_uri=None, artifact_location=None, log=print) -> TrainingSummary`

Run order and MLflow run names (spec, "Training"):
1. `b0-comps`
2. `b1-lightgbm`
3. `xgb-tune`, with nested `trial-000` …
4. `xgb-champion-eval`
5. the acceptance gate
6. `xgb-production`
7. registration

If the gate fails, the summary comes back with `gate_passed=False`, and there's no production run and no registration.

- [ ] **Step 1: Write the fixture builder and generate the fixture**

`scripts/build_price_fixture.py`:

```python
"""Regenerate tests/fixtures/price_sample.csv: ~3,000 real DLD home sales for the Phase 3
integration test, stratified by split period and property kind.

Run from the repo root: uv run python scripts/build_price_fixture.py
"""

from datetime import date
from pathlib import Path

import polars as pl

SEED = 42
SOURCE = Path("data/raw/Transactions.csv")
TARGET = Path("tests/fixtures/price_sample.csv")
MARKET = ["Sell", "Sell - Pre registration", "Delayed Sell", "Sale On Payment Plan"]
PERIODS = {  # name: (start inclusive, end exclusive, rows)
    "seed": (date(2014, 1, 1), date(2015, 1, 1), 300),
    "train": (date(2015, 1, 1), date(2022, 7, 1), 1_500),
    "val": (date(2022, 7, 1), date(2022, 11, 1), 500),
    "test": (date(2022, 11, 1), date(2023, 4, 1), 700),
}


def main() -> None:
    # Keep "null" as literal text so the fixture round-trips in the source format.
    df = pl.read_csv(SOURCE, infer_schema=False)
    day = pl.col("instance_date").str.strptime(pl.Date, "%d-%m-%Y", strict=False)
    worth = pl.col("actual_worth").cast(pl.Float64, strict=False)
    size = pl.col("procedure_area").cast(pl.Float64, strict=False)
    ptype, sub = pl.col("property_type_en"), pl.col("property_sub_type_en").fill_null("")
    market = (
        (pl.col("trans_group_en") == "Sales")
        & pl.col("procedure_name_en").is_in(MARKET)
        & (worth >= 10_000)
        & (size > 0)
    )
    kinds = {  # name: (filter, share of each period)
        "flat": ((ptype == "Unit") & (sub == "Flat"), 0.50),
        "hotel_apartment": ((ptype == "Unit") & (sub == "Hotel Apartment"), 0.10),
        "townhouse": ((ptype == "Unit") & (sub == "Stacked Townhouses"), 0.05),
        "villa_built_up": ((ptype == "Villa") & (sub == "Villa"), 0.20),
        "villa_plot": ((ptype == "Villa") & (sub == ""), 0.15),
    }
    picks = []
    for start, end, rows in PERIODS.values():
        in_period = market & (day >= start) & (day < end)
        for condition, share in kinds.values():
            subset = df.filter(in_period & condition)
            picks.append(subset.sample(n=min(round(rows * share), subset.height), seed=SEED))

    sample = pl.concat(picks).unique(subset="transaction_id", keep="first", maintain_order=True)
    if not 2_500 <= sample.height <= 3_100:
        raise SystemExit(f"fixture has {sample.height} rows, expected 2,500-3,100")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    sample.write_csv(TARGET)
    print(f"wrote {sample.height} rows to {TARGET}")


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/build_price_fixture.py`
Expected: `wrote N rows to tests/fixtures/price_sample.csv`, with N between 2,500 and 3,100. Stacked townhouses are rare, so their strata may come in short. The file is real DLD data (about 2 MB) and is committed like `dld_sample.csv`.

- [ ] **Step 2: Write the fixture test**

`tests/models/price/test_price_fixture.py`:

```python
from datetime import date
from pathlib import Path

import polars as pl

from ingestion.schema import EXPECTED_COLUMNS

FIXTURE = Path("tests/fixtures/price_sample.csv")


def test_price_fixture_is_real_dld_shaped_and_spans_every_split():
    frame = pl.read_csv(FIXTURE, infer_schema=False)
    assert set(frame.columns) == set(EXPECTED_COLUMNS)
    assert 2_500 <= frame.height <= 3_100
    days = frame["instance_date"].str.strptime(pl.Date, "%d-%m-%Y")
    assert days.min() < date(2015, 1, 1)
    assert days.max() >= date(2022, 11, 1)
    assert set(frame["property_type_en"].unique()) == {"Unit", "Villa"}
```

Run: `uv run pytest tests/models/price/test_price_fixture.py -v`
Expected: PASS.

- [ ] **Step 3: Write the failing integration test**

`tests/models/price/test_price_train_integration.py`:

```python
import dataclasses
from pathlib import Path

import mlflow
import polars as pl

from ingestion.pipeline import run_pipeline
from models.price.config import TrainConfig
from models.price.train import run_training

FIXTURE = Path("tests/fixtures/price_sample.csv")
SMALL = dataclasses.replace(
    TrainConfig(),
    min_segment_rows=10,
    min_index_sales=3,
    oof_folds=3,
    comps_min_n=2.0,
    bounds_min_area_n=5.0,
    min_conformal_rows=20,
    n_trials=2,
    max_rounds=200,
    early_stopping_rounds=20,
    gate_ratio=None,
    experiment="price-integration",
    model_name="dubimator-price-it",
)


def train(settings, temp_mlflow, config):
    run_pipeline(FIXTURE, settings)
    return run_training(
        settings,
        config,
        device="cpu",
        register=True,
        tracking_uri=temp_mlflow["tracking_uri"],
        artifact_location=temp_mlflow["artifact_location"],
        log=lambda _message: None,
    )


def test_end_to_end_training_registers_a_working_champion(pg_test_db, temp_mlflow):
    summary = train(pg_test_db, temp_mlflow, SMALL)

    assert summary.gate_passed and summary.registered_version == "1"
    assert set(summary.metrics) == {"b0-comps", "b1-lightgbm", "xgb-champion-eval"}
    for metrics in summary.metrics.values():
        assert metrics["test_clean.all.n"] > 0
        assert 0.0 < metrics["test_clean.all.mdape"] < 1.0
        assert metrics["test_honest.all.n"] >= metrics["test_clean.all.n"]
    champion = summary.metrics["xgb-champion-eval"]
    assert 0.0 <= champion["test_clean.all.coverage_80"] <= 1.0
    assert any(key.startswith("test_clean.segment.") for key in champion)

    run_names = set(mlflow.search_runs(experiment_names=["price-integration"])["tags.mlflow.runName"])
    assert {"b0-comps", "b1-lightgbm", "xgb-tune", "trial-000", "trial-001",
            "xgb-champion-eval", "xgb-production"} <= run_names  # fmt: skip

    model = mlflow.pyfunc.load_model("models:/dubimator-price-it@champion")
    predictor = model.unwrap_python_model().predictor
    areas = predictor.bundle.priors.stats["area"].filter(pl.col("property_type") == "unit")
    busiest_area = int(areas.sort("sum_w", descending=True)["area_id"][0])
    estimate = predictor.predict_one(
        {"area_id": busiest_area, "property_kind": "apartment", "status": "ready",
         "size_sqm": 90.0, "bedrooms": 2}
    )  # fmt: skip
    assert estimate.estimate_aed > 0
    assert estimate.range_80[0] < estimate.estimate_aed < estimate.range_80[1]
    assert estimate.as_of == predictor.bundle.data_end
    assert estimate.model_version == summary.model_uri.split("/")[1]


def test_failed_gate_registers_nothing(pg_test_db, temp_mlflow):
    summary = train(pg_test_db, temp_mlflow, dataclasses.replace(SMALL, gate_ratio=0.0))
    assert not summary.gate_passed
    assert summary.registered_version is None and summary.model_uri is None
    run_names = set(mlflow.search_runs(experiment_names=["price-integration"])["tags.mlflow.runName"])
    assert "xgb-champion-eval" in run_names and "xgb-production" not in run_names
```

`summary.model_uri` looks like `runs:/<run_id>/model`, so `split("/")[1]` is the production run id, which the predictor reports as `model_version` (controller ruling 2).

Run: `uv run pytest tests/models/price/test_price_train_integration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.price.train'`.

- [ ] **Step 4: Implement `train.py`**

```python
"""Phase 3 orchestration: load → features → baselines → tuning → evaluation → gate →
production refit → MLflow registration."""

import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from ingestion.config import DbSettings
from models.price.baselines import comps_b0, lightgbm_b1
from models.price.boosting import fit_xgb, make_dmatrix, predict_xgb, resolve_device
from models.price.config import TrainConfig
from models.price.data import load_homes
from models.price.evaluate import coverage, fit_conformal, quantiles_for, sliced_metrics
from models.price.features import (
    FeatureSet,
    build_feature_set,
    feature_frame,
    fit_bounds,
    fit_size_percentiles,
)
from models.price.plots import save_eval_plots
from models.price.predictor import ModelBundle, save_bundle
from models.price.registry import configure, log_metrics, log_price_model, register_champion
from models.price.split import add_bulk_groups, assign_split
from models.price.tune import tune_xgboost

EVAL_SETS = ("val", "test_clean", "test_honest")


@dataclass
class TrainingSummary:
    device: str
    rows: dict[str, int]
    drop_counts: dict[str, int]
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    gate_passed: bool = False
    model_uri: str | None = None
    registered_version: str | None = None


def _prices(frame: pl.DataFrame, y_hat) -> np.ndarray:
    return np.exp(np.asarray(y_hat) + frame["market_index"].to_numpy()) * frame["area_sqm"].to_numpy()


def _evaluate(
    fs: FeatureSet, predict: Callable[[pl.DataFrame], np.ndarray]
) -> tuple[dict[str, float], dict[str, pl.DataFrame]]:
    metrics: dict[str, float] = {}
    frames: dict[str, pl.DataFrame] = {}
    for name in EVAL_SETS:
        frame = fs.frames.get(name)
        if frame is None or frame.height == 0:
            continue
        evaluated = frame.select(
            pl.col("price_aed").alias("actual"), "segment", "loc_level", "bulk_group"
        ).with_columns(pl.Series("predicted", _prices(frame, predict(frame))))
        metrics |= sliced_metrics(evaluated, name)
        frames[name] = evaluated
    return metrics, frames


def _coverage_metrics(
    frames: dict[str, pl.DataFrame], conformal: dict[str, dict[str, float]]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("test_clean", "test_honest"):
        frame = frames.get(name)
        if frame is None:
            continue
        for label in ("q80", "q95"):
            q = np.array([quantiles_for(conformal, s)[label] for s in frame["segment"].to_list()])
            predicted = frame["predicted"].to_numpy()
            frame_q = frame.with_columns(
                pl.Series("low", predicted * np.exp(-q)), pl.Series("high", predicted * np.exp(q))
            )
            parts = [("all", frame_q)] + [
                (f"segment.{segment}", frame_q.filter(pl.col("segment") == segment))
                for segment in sorted(frame_q["segment"].unique().to_list())
            ]
            for slice_name, part in parts:
                out[f"{name}.{slice_name}.coverage_{label[1:]}"] = coverage(
                    part["actual"].to_numpy(), part["low"].to_numpy(), part["high"].to_numpy()
                )
    return out


def run_training(
    settings: DbSettings,
    config: TrainConfig,
    device: str = "auto",
    register: bool = True,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
    log: Callable[[str], None] = print,
) -> TrainingSummary:
    started = time.perf_counter()
    device = resolve_device(device)
    homes = load_homes(settings, config)
    rows = add_bulk_groups(assign_split(homes.rows, config))
    split, clean = pl.col("split"), pl.col("is_clean")
    eval_fs = build_feature_set(
        rows,
        config,
        homes.data_end,
        fit_filter=(split == "train") & clean,
        apply_filters={
            "val": (split == "val") & clean,
            "test_clean": (split == "test") & clean,
            "test_honest": split == "test",
        },
    )
    fit, val = eval_fs.frames["fit"], eval_fs.frames["val"]
    x_fit, x_val = feature_frame(fit, eval_fs.categories), feature_frame(val, eval_fs.categories)
    y_fit, y_val = fit["y"].to_numpy(), val["y"].to_numpy()
    w_val = val["bulk_weight"].to_numpy()  # ruling 3: bulk weight only for validation rows
    summary = TrainingSummary(
        device=device,
        rows={name: frame.height for name, frame in eval_fs.frames.items()},
        drop_counts=homes.drop_counts,
    )
    summary.seconds["prepare"] = time.perf_counter() - started
    log(f"Prepared rows {summary.rows} (device {device}); dropped {homes.drop_counts}")

    configure(config.experiment, tracking_uri, artifact_location)
    base_params = {
        "device": device,
        "data_end": homes.data_end.isoformat(),
        "supported_segments": ",".join(homes.supported_segments),
        **{f"lineage.{key}": value for key, value in homes.lineage.items()},
        **{f"rows.{key}": value for key, value in summary.rows.items()},
    }

    with mlflow.start_run(run_name="b0-comps"):
        mlflow.log_params(base_params)
        metrics, _ = _evaluate(eval_fs, lambda frame: comps_b0(fit, frame, config.comps_min_n))
        log_metrics(metrics)
        summary.metrics["b0-comps"] = metrics
    log(f"b0-comps test_clean MdAPE {summary.metrics['b0-comps'].get('test_clean.all.mdape')}")

    with mlflow.start_run(run_name="b1-lightgbm"):
        mlflow.log_params(base_params)
        clock = time.perf_counter()
        lgbm = lightgbm_b1(x_fit, y_fit, eval_fs.weights, x_val, y_val, w_val, config)
        summary.seconds["b1-lightgbm"] = time.perf_counter() - clock
        metrics, _ = _evaluate(
            eval_fs,
            lambda frame: lgbm.predict(
                feature_frame(frame, eval_fs.categories), num_iteration=lgbm.best_iteration
            ),
        )
        log_metrics(metrics | {"best_iteration": lgbm.best_iteration})
        summary.metrics["b1-lightgbm"] = metrics
    log(f"b1-lightgbm test_clean MdAPE {summary.metrics['b1-lightgbm'].get('test_clean.all.mdape')}")

    dtrain = make_dmatrix(x_fit, y_fit, eval_fs.weights)
    dval = make_dmatrix(x_val, y_val, w_val)
    with mlflow.start_run(run_name="xgb-tune"):
        mlflow.log_params(base_params | {"n_trials": config.n_trials})

        def on_trial(number: int, params: dict, value: float, best_iteration: int, seconds: float):
            with mlflow.start_run(run_name=f"trial-{number:03d}", nested=True):
                mlflow.log_params(params)
                log_metrics(
                    {"val_rmse_weighted": value, "best_iteration": best_iteration, "seconds": seconds}
                )
            log(f"  trial {number:3d}: val rmse {value:.5f}, {best_iteration} rounds, {seconds:.1f}s")

        clock = time.perf_counter()
        result = tune_xgboost(dtrain, dval, device, config, on_trial)
        summary.seconds["xgb-tune"] = time.perf_counter() - clock
        mlflow.log_params({f"best.{key}": value for key, value in result.best_params.items()})
        log_metrics({"best_val_rmse_weighted": result.best_value, "seconds": summary.seconds["xgb-tune"]})
    summary.best_params = result.best_params

    with mlflow.start_run(run_name="xgb-champion-eval"):
        mlflow.log_params(base_params | result.best_params)
        booster = fit_xgb(result.best_params, dtrain, dval, device, config)
        metrics, frames = _evaluate(
            eval_fs, lambda frame: predict_xgb(booster, feature_frame(frame, eval_fs.categories))
        )
        val_frame = frames["val"]
        abs_error = np.abs(np.log(val_frame["predicted"].to_numpy()) - np.log(val_frame["actual"].to_numpy()))
        conformal = fit_conformal(val_frame["segment"].to_list(), abs_error, config.min_conformal_rows)
        metrics |= _coverage_metrics(frames, conformal)
        log_metrics(metrics | {"best_iteration": float(booster.best_iteration)})
        summary.metrics["xgb-champion-eval"] = metrics
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plot_frame = frames.get("test_clean", val_frame)
            save_eval_plots(plot_frame, booster.get_score(importance_type="total_gain"), out)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
            (out / "conformal.json").write_text(json.dumps(conformal, indent=2))
            mlflow.log_artifacts(tmp)
    log(f"xgb-champion-eval test_clean MdAPE {metrics.get('test_clean.all.mdape')}")

    baseline = summary.metrics["b0-comps"].get("test_clean.all.mdape")
    champion = metrics.get("test_clean.all.mdape")
    summary.gate_passed = config.gate_ratio is None or (
        baseline is not None and champion is not None and champion <= config.gate_ratio * baseline
    )
    if not summary.gate_passed:
        log(f"Acceptance gate failed: champion {champion} vs {config.gate_ratio} x B0 {baseline}")
        summary.seconds["total"] = time.perf_counter() - started
        return summary

    clock = time.perf_counter()
    prod_fs = build_feature_set(
        rows, config, homes.data_end, fit_filter=clean & (split != "seed"), apply_filters={}
    )
    prod_fit = prod_fs.frames["fit"]
    dall = make_dmatrix(
        feature_frame(prod_fit, prod_fs.categories), prod_fit["y"].to_numpy(), prod_fs.weights
    )
    num_rounds = int(booster.best_iteration) + 1
    headline = {key: value for key, value in metrics.items() if ".all." in key}
    with mlflow.start_run(run_name="xgb-production") as run:
        mlflow.log_params(
            base_params
            | result.best_params
            | {"num_boost_round": num_rounds, "rows.production_fit": prod_fit.height}
        )
        prod_booster = fit_xgb(result.best_params, dall, None, device, config, num_rounds=num_rounds)
        bounds_area, bounds_segment = fit_bounds(prod_fit, config.bounds_min_area_n)
        bundle = ModelBundle(
            booster=prod_booster,
            index=prod_fs.index,
            priors=prod_fs.priors,
            categories=prod_fs.categories,
            bounds_area=bounds_area,
            bounds_segment=bounds_segment,
            size_percentiles=fit_size_percentiles(prod_fit),
            areas=homes.areas,
            aliases=homes.aliases,
            conformal=conformal,
            supported_segments=homes.supported_segments,
            data_end=homes.data_end,
            metadata={
                "model_version": run.info.run_id,
                "lineage": homes.lineage,
                "params": result.best_params,
                "num_boost_round": num_rounds,
                "eval_metrics": headline,
                "device": device,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model_dir"
            save_bundle(bundle, model_dir)
            summary.model_uri = log_price_model(model_dir)
        log_metrics(headline)
    summary.seconds["xgb-production"] = time.perf_counter() - clock
    if register:
        summary.registered_version = register_champion(summary.model_uri, config.model_name)
        log(f"Registered {config.model_name} v{summary.registered_version} as @champion")
    summary.seconds["total"] = time.perf_counter() - started
    return summary
```

- [ ] **Step 5: Run the integration tests to verify they pass**

Run: `uv run pytest tests/models/price/test_price_train_integration.py -v`
Expected: PASS (a few minutes on the CPU). Then run `git status --short`: there must be no `mlruns/` or `.db` file in the repo root.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: every test passes, with only the GPU smoke test allowed to skip, and only on a machine without CUDA.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add models/price/train.py scripts/build_price_fixture.py tests/fixtures/price_sample.csv tests/models/price/test_price_fixture.py tests/models/price/test_price_train_integration.py
git commit -m "$(cat <<'EOF'
feat(price): end-to-end training run with baselines, tuning, gate and champion registration

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: CLI, the real training run, documentation

**Files:**
- Create: `models/price/__main__.py`, `tests/models/price/test_price_cli.py`
- Modify: `README.md`, `docs/superpowers/specs/2026-09-15-phase3-price-model-design.md` (amendments section)

**Interfaces:**
- Consumes: `run_training`, `TrainingSummary` (Task 11); `PricePredictor`, `PriceInputError` (Task 9); the `tiny_bundle` fixture
- Produces: `models.price.__main__.main(argv: list[str] | None = None) -> int`
  - `train [--trials N] [--device auto|cuda|cpu] [--no-register]`: exit 0 on success, 1 on an error, 2 when the gate fails
  - `predict (--area NAME | --area-id ID) --kind KIND --status STATUS --size M2 [--size-basis] [--bedrooms] [--penthouse] [--parking yes|no] [--project] [--building] [--asking AED]`: exit 0, or 1 on invalid input
  - `main()` calls `load_dotenv()` before anything reads DB or MLflow settings

- [ ] **Step 1: Write the failing CLI tests**

`tests/models/price/test_price_cli.py`:

```python
import json

from models.price import __main__ as cli
from models.price.predictor import PricePredictor
from models.price.train import TrainingSummary

METRICS = {
    f"{set_name}.all.{metric}": value
    for set_name in ("val", "test_clean", "test_honest")
    for metric, value in (("mdape", 0.12), ("ppe10", 0.45), ("ppe20", 0.75), ("rmse_log", 0.2))
}


def summary(gate_passed=True):
    return TrainingSummary(
        device="cpu",
        rows={"fit": 10, "val": 3},
        drop_counts={},
        metrics={"b0-comps": METRICS, "xgb-champion-eval": METRICS},
        seconds={"total": 1.0},
        gate_passed=gate_passed,
        registered_version="3" if gate_passed else None,
    )


def test_train_loads_dotenv_before_building_db_settings(monkeypatch, capsys):
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: monkeypatch.setenv("POSTGRES_PORT", "6543"))
    seen = {}

    def fake_run(settings, config, **kwargs):
        seen.update(port=settings.port, trials=config.n_trials, kwargs=kwargs)
        return summary()

    monkeypatch.setattr(cli, "run_training", fake_run)
    assert cli.main(["train", "--trials", "3", "--device", "cpu", "--no-register"]) == 0
    assert seen == {"port": 6543, "trials": 3, "kwargs": {"device": "cpu", "register": False}}
    out = capsys.readouterr().out
    assert "xgb-champion-eval" in out and "12.00%" in out
    assert "version 3" in out


def test_train_exits_2_when_the_gate_fails(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "run_training", lambda *a, **k: summary(gate_passed=False))
    assert cli.main(["train"]) == 2
    assert "Acceptance gate failed" in capsys.readouterr().err


def test_train_exits_1_on_errors(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)

    def boom(*args, **kwargs):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(cli, "run_training", boom)
    assert cli.main(["train"]) == 1
    assert "Training failed: RuntimeError: database unreachable" in capsys.readouterr().err


def test_predict_prints_the_estimate_as_json(monkeypatch, capsys, tiny_bundle):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "_load_champion", lambda: PricePredictor(tiny_bundle(y_hat=0.0)))
    code = cli.main(
        ["predict", "--area", "Dubai Marina", "--kind", "apartment", "--status", "ready",
         "--size", "100", "--bedrooms", "2", "--parking", "yes", "--asking", "1100000"]
    )  # fmt: skip
    assert code == 0
    estimate = json.loads(capsys.readouterr().out)
    assert round(estimate["estimate_aed"]) == 1_000_000
    assert estimate["market_label"] == "fair"


def test_predict_reports_invalid_input(monkeypatch, capsys, tiny_bundle):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "_load_champion", lambda: PricePredictor(tiny_bundle(y_hat=0.0)))
    code = cli.main(["predict", "--area", "Atlantis", "--kind", "apartment", "--status", "ready",
                     "--size", "100"])  # fmt: skip
    assert code == 1
    assert "Invalid input: area:" in capsys.readouterr().err
```

The asking price of 1,100,000 is 10% above the 1,000,000 estimate, which is inside the 80% range of ×e^±0.15 (about ±16%), so the label is "fair".

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/models/price/test_price_cli.py -v`
Expected: FAIL (`models.price.__main__` doesn't exist yet).

- [ ] **Step 3: Implement the CLI**

`models/price/__main__.py`:

```python
"""Command line: python -m models.price train|predict."""

import argparse
import dataclasses
import sys

from dotenv import load_dotenv

from ingestion.config import DbSettings
from models.price.config import TrainConfig
from models.price.predictor import PriceInputError, PricePredictor
from models.price.train import TrainingSummary, run_training

KINDS = ("apartment", "hotel_apartment", "townhouse", "villa")
SETS = ("val", "test_clean", "test_honest")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m models.price", description="Train or query the home price model."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train, evaluate and register the model")
    train.add_argument("--trials", type=int, default=TrainConfig.n_trials)
    train.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    train.add_argument("--no-register", action="store_true")
    predict = commands.add_parser("predict", help="estimate one home with the champion model")
    area = predict.add_mutually_exclusive_group(required=True)
    area.add_argument("--area")
    area.add_argument("--area-id", type=int)
    predict.add_argument("--project")
    predict.add_argument("--building")
    predict.add_argument("--kind", choices=KINDS, required=True)
    predict.add_argument("--status", choices=("ready", "off_plan"), required=True)
    predict.add_argument("--size", type=float, required=True, help="size in m²")
    predict.add_argument("--size-basis", choices=("built_up", "plot"), default="built_up")
    predict.add_argument("--bedrooms", type=int)
    predict.add_argument("--penthouse", action="store_true")
    predict.add_argument("--parking", choices=("yes", "no"))
    predict.add_argument("--asking", type=float, help="asking price in AED")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv()  # before any settings are read: without it DbSettings defaults to port 5432
    if args.command == "train":
        return _train(args)
    return _predict(args)


def _print_summary(summary: TrainingSummary) -> None:
    print(f"Device {summary.device}; rows " + ", ".join(f"{k}={v:,}" for k, v in summary.rows.items()))
    print(f"{'model':<20}{'set':<13}{'MdAPE':>8}{'PPE10':>8}{'PPE20':>8}{'RMSE(ln)':>10}")
    for model, metrics in summary.metrics.items():
        for set_name in SETS:
            if f"{set_name}.all.mdape" not in metrics:
                continue
            values = [metrics[f"{set_name}.all.{name}"] for name in ("mdape", "ppe10", "ppe20", "rmse_log")]
            print(
                f"{model:<20}{set_name:<13}{values[0]:>8.2%}{values[1]:>8.1%}"
                f"{values[2]:>8.1%}{values[3]:>10.4f}"
            )
    for name, seconds in summary.seconds.items():
        print(f"  {name}: {seconds:,.1f}s")
    if summary.registered_version:
        print(f"Registered {TrainConfig.model_name} version {summary.registered_version} as @champion")


def _train(args: argparse.Namespace) -> int:
    config = dataclasses.replace(TrainConfig(), n_trials=args.trials)
    try:
        summary = run_training(
            DbSettings.from_env(), config, device=args.device, register=not args.no_register
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print_summary(summary)
    if not summary.gate_passed:
        print(
            "Acceptance gate failed: the champion did not beat the comps baseline's test MdAPE "
            "by 10%. Nothing was registered.",
            file=sys.stderr,
        )
        return 2
    return 0


def _load_champion() -> PricePredictor:
    import mlflow

    model = mlflow.pyfunc.load_model(f"models:/{TrainConfig.model_name}@champion")
    return model.unwrap_python_model().predictor


def _predict(args: argparse.Namespace) -> int:
    request = {
        "area": args.area,
        "area_id": args.area_id,
        "project": args.project,
        "building": args.building,
        "property_kind": args.kind,
        "status": args.status,
        "size_sqm": args.size,
        "size_basis": args.size_basis,
        "bedrooms": args.bedrooms,
        "is_penthouse": args.penthouse,
        "has_parking": None if args.parking is None else args.parking == "yes",
        "asking_price_aed": args.asking,
    }
    try:
        estimate = _load_champion().predict_one({k: v for k, v in request.items() if v is not None})
    except PriceInputError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1
    print(estimate.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass, then commit the CLI**

Run: `uv run pytest tests/models/price/test_price_cli.py -v` → PASS.

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add models/price/__main__.py tests/models/price/test_price_cli.py
git commit -m "$(cat <<'EOF'
feat(price): python -m models.price train|predict command line

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Run the real training on the GPU**

The Docker stack must be up (`docker compose up -d --wait`), and `dld` must hold the full ingestion (run 6: 707,655 market sales). This takes roughly 15–45 minutes, so start it in the background and wait for it to exit:

```bash
uv run python -m models.price train 2>&1 | tee /tmp/dubimator-price-train.log
```

Expected:
- the first log line reports `device cuda`
- 60 trial lines
- a metric table
- `Registered dubimator-price version N as @champion`
- exit code 0

What to do with each exit code:
- **2 (the gate failed):** STOP. Report BLOCKED with the whole log. Do not change the gate, the features or the config to get past it; the controller and the user decide.
- **1:** debug the error (systematic debugging) and report what failed and why before changing any code.
- **`device cpu` on the first line:** report DONE_WITH_CONCERNS; the GPU isn't being used.

Then check the model answers:

```bash
uv run python -m models.price predict --area "JVC" --kind apartment --status ready --size 75 --bedrooms 1 --asking 900000
uv run python -m models.price predict --area "Dubai Hills" --kind villa --status ready --size 350 --bedrooms 4
uv run python -m models.price predict --area "Dubai Marina" --kind apartment --status off_plan --size 110 --bedrooms 2
```

Each must print JSON with `estimate_aed > 0`, `as_of` equal to `2023-03-17`, and plausible AED figures. Report all three outputs.

- [ ] **Step 6: Measure CPU speed for the README comparison**

```bash
uv run python -m models.price train --trials 3 --device cpu --no-register 2>&1 | tee /tmp/dubimator-price-cpu.log
```

The CPU seconds per trial are `xgb-tune` seconds ÷ 3 from this log. The GPU figure is `xgb-tune` seconds ÷ 60 from the GPU log. The gate may pass or fail here; only the timing matters, and nothing is registered.

- [ ] **Step 7: Collect the per-slice numbers**

```bash
uv run python - <<'PY'
import mlflow
from dotenv import load_dotenv

load_dotenv()
for name in ("b0-comps", "b1-lightgbm", "xgb-champion-eval"):
    run = mlflow.search_runs(
        experiment_names=["price-model"],
        filter_string=f"tags.mlflow.runName = '{name}'",
        order_by=["start_time DESC"],
        max_results=1,
    ).iloc[0]
    for column in sorted(c for c in run.index if c.startswith("metrics.test_")):
        if name == "xgb-champion-eval" or ".all." in column:
            print(name, column.removeprefix("metrics."), round(run[column], 4))
PY
```

Save the output next to the training log (`/tmp/dubimator-price-metrics.txt`). Every number in the README comes from these two files. Round MdAPE to one decimal place of a percent.

- [ ] **Step 8: Update the README**

Make these edits in `README.md`:

1. Status line: replace `> **Status:** Phase 2 (DLD ingestion) complete. See` with `> **Status:** Phase 3 (price model) complete. See`.
2. In the Mermaid block, after the line `    ML -.->|"runs + artifacts"| MLRUNS`, add:

```
    TRAIN["python -m models.price train<br/>(XGBoost on the GPU)"]
    PG -->|"home sales"| TRAIN
    TRAIN -->|"runs + dubimator-price@champion"| ML
```

3. Insert this section immediately before `## Cost breakdown (current)`. Replace every `‹…›` with the real value from the logs; no `‹` may remain.

```markdown
## Price model

Estimates the fair market price of a Dubai home (apartment, hotel apartment,
townhouse or villa) **as of 2023-03-17**, the last date in the DLD data. It
gives an 80% and a 95% price range, and it labels an asking price as below
market, fair or above market. It uses only real DLD transactions; there is no
synthetic data.

```bash
uv run python -m models.price train      # ‹N› min on an RTX 3060 Laptop GPU (60 Optuna trials)
uv run python -m models.price predict --area "JVC" --kind apartment --status ready --size 75 --bedrooms 1 --asking 900000
```

**How it works** (design: `docs/superpowers/specs/2026-09-15-phase3-price-model-design.md`)

- **Data:** homes sold from 2015-01-01 on. Offices, shops, land and whole
  buildings are out of scope. Sizes outside plausible bounds are dropped and
  counted.
- **Split by time:** train up to 2022-06, validate on 2022-07 to 2022-10, and
  test on 2022-11 to 2023-03. The test set is touched once.
- **Target:** ln(price per m²) minus a leak-free market index (the median of
  the previous three months for that segment). Trees can't extrapolate, and
  this keeps the 2022–23 boom from being underpredicted.
- **Location:** building → project → area → city priors, each shrunk toward
  its parent. A new or sparse location falls back automatically, and the
  response says which level it used.
- **Leakage guards:** an exact feature allowlist, enforced by tests.
  Out-of-fold priors are grouped by bulk sale, so 94 identical sales can't
  reveal each other's price.
- **Model:** XGBoost on the GPU, tuned by Optuna. It is compared against an
  area-comps rule (B0) and LightGBM (B1), and it must beat B0's test MdAPE by
  10% to be registered.
- **Serving:** one MLflow pyfunc, `models:/dubimator-price@champion`.
  - input validation
  - unseen-location fallback
  - clipping of implausible predictions
  - conformal price ranges

**Results** on the test set (2022-11-01 to 2023-03-17, never used for fitting
or tuning). "Honest" adds back the ‹N› rows that Phase 2's price-based outlier
filter removed.

| Model | MdAPE | PPE10 | PPE20 | MdAPE (honest) |
|---|---|---|---|---|
| B0 comps | ‹›% | ‹›% | ‹›% | ‹›% |
| B1 LightGBM | ‹›% | ‹›% | ‹›% | ‹›% |
| **XGBoost (champion)** | **‹›%** | **‹›%** | **‹›%** | **‹›%** |

Champion by segment (clean test set):

| Segment | Test sales | MdAPE | 80% range coverage |
|---|---|---|---|
| ‹one row per segment from metrics: segment.<s>.n / .mdape / .coverage_80› | | | |

80% range coverage overall: ‹›% (target 80%). 95%: ‹›%.
Tuning speed: ‹›s per trial on the GPU vs ‹›s per trial on the CPU.

**Write-up**

*Business problem.* Buyers, sellers and listing platforms need a
defensible fair-value figure for a home, and a way to spot listings priced
far from it. DLD registrations record what homes actually sold for. The model
learns fair value from those sales and flags an asking price outside its 80%
range as below or above market.

*Metric optimised.* Training minimises the weighted squared error of the
relative log price, which is roughly relative error: a 10% miss on an
AED 800k flat counts the same as one on an AED 8M villa. Results are reported
as MdAPE, PPE10 and PPE20, the standard automated-valuation metrics. They're
measured on a later period the model never saw, on both the cleaned test set
and an honest one, because the cleaning rule itself used the price.

*What I'd do differently with production data and traffic.*
- Add unit-level attributes that DLD doesn't publish, such as floor, view and
  condition. They're the biggest missing signal, and listing data would
  supply them.
- Retrain monthly on fresh DLD exports (this data ends in March 2023) and
  watch drift. The market index keeps the price level current between
  retrains, but location premiums move too.
- Make the intervals conditional, for example Mondrian conformal by location
  level, so ranges in sparse areas widen honestly.
- Evaluate against listing-to-sale outcomes, not just registered prices.
- Geocode buildings to use distances instead of area IDs.
```

4. In `## Cost breakdown (current)`, add a table row: `| Price model training (local RTX 3060) | $0 — runs on your machine |`.
5. In `## Module layout`, replace ``- `models/` — price/fraud/ranking model code (Phase 3+)`` with ``- `models/price/` — home price model: features, training, evaluation, predictor (`python -m models.price`)``.

- [ ] **Step 9: Add the spec amendments**

Append to `docs/superpowers/specs/2026-09-15-phase3-price-model-design.md`:

```markdown
## Amendments during implementation

- **Two extra modules:** `boosting.py` holds the XGBoost helpers, so serving
  never imports Optuna, and `plots.py` holds the evaluation plots.
- **`PriceEstimate.model_version`** is the production MLflow run id, not the
  registry version number. The version doesn't exist until after the artifact
  is logged; the registry maps a run id to its version.
- **Validation weights** for early stopping are the bulk weight only. Recency
  is measured from the end of the fitted data, so it doesn't apply to later
  rows.
- **The val/test perturbation leakage test** is stated precisely: perturbing
  every price from month M on leaves the features of every row dated in month
  M or earlier unchanged. The market index legitimately uses earlier val and
  test months.
- **Test layout follows Phase 2:** there are no `__init__.py` files under
  `tests/`, Phase 3 test files are named `test_price_*.py`, and shared
  helpers are fixtures.
- **Real run (‹date›):**
  - ‹N› training rows
  - champion test MdAPE ‹›% against B0 ‹›%
  - registered as `dubimator-price` v‹N›
```

- [ ] **Step 10: Final verification and commit**

```bash
uv run pytest -q
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git status --short
```

Expected:
- every test passes
- ruff is clean
- `git status` shows only `README.md` and the spec as modified: no `mlruns/`, no `.db`, and nothing under `data/`

```bash
git add README.md docs/superpowers/specs/2026-09-15-phase3-price-model-design.md
git commit -m "$(cat <<'EOF'
docs: Phase 3 price model results, write-up and spec amendments

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```
