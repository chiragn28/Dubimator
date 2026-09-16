# Phase 6 — Multi-Horizon Price Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forecast what a Dubai home will be worth in 3 months, 1 year and 3 years, using three separate XGBoost models of market growth. Each forecast carries a conformal 80% range and a confidence label, is validated walk-forward, and has its own gate. Horizons that fail are published as `not_deployed`.

**Architecture:** A new package, `models/forecast/`.
- **Pipeline:** rows (validation, dedup, outliers) → targets (base price, forward windows) → features (momentum, context, property, sourced infrastructure) → folds → baselines → tuned XGBoost per horizon → conformal intervals → segmented evaluation and gate → MLflow registry → `Forecaster` JSON.
- **CLI:** `python -m models.forecast build|infra-check|train|evaluate|predict`.
- **Reuse:** Phase 3's `load_homes`, conformal quantile, registry helpers and price predictor.

**Tech Stack:** Python 3.11, Polars 1.44 (time-based `rolling`), pandas (for XGBoost categoricals only), XGBoost 3.2 (CUDA; `pred_contribs` for SHAP), Optuna 5, MLflow 2.17.2, psycopg2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md` (the user's request, adapted, is in `docs/superpowers/briefs/2026-09-16-price-forecasting-brief.md`).

## Global Constraints

### Time windows
Every window is expressed in **days** (1 month = 30.4375 days, rounded):

| Window | Days |
|---|---|
| Base | [T − 92 d, T) |
| 3m target | [T + 76 d, T + 107 d] |
| 1y target | [T + 335 d, T + 396 d] |
| 3y target | [T + 1065 d, T + 1126 d] |
| Momentum lags | 91 d (3m), 365 d (12m), 1096 d (36m) |
| Trailing counts | 365 d |

### Levels
- **Building key:** `"{area_id}|{match_key(building_name)}"`. It is null when `building_name` is null.
- **Area key:** `"{area_id}|{sub_kind}"`.
- **City key:** `sub_kind`.
- **Minimum sales:** a level's base or target needs **at least 5 sales** in its window.

### Target
- The base comes from the building level, falling back to the area level. Rows with neither are excluded with `no_base`.
- The target is `growth_<h> = ln(median ppsm in the target window ÷ base ppsm)` at **the base's level**.
- Rows without a target are excluded for that horizon with `no_target`. Nothing is imputed, and there is never a current-price fallback.

### Rows
- **Input:** `models.price.data.load_homes` rows with `is_clean` true.
- **Drop reasons:**
  - `bad_price` (≤ 0 or null)
  - `bad_size` (≤ 0 or null)
  - `bad_date` (null)
  - `missing_area` (null `area_id`)
  - `duplicate_transaction_id`
  - `repeat_sale` (same building, date, size and price; keep the last)
  - `outlier` (|robust z| > 3 within area × sub_kind × month; groups under 10 sales are not screened)
- **Report line:** `Dropped X of Y rows (Z%)`.
- **Guardrail:** `build` exits 1 if more than 30% of rows are dropped.

### Features
- Every feature uses only sales strictly before T, or infrastructure dates on or before T.
- `FEATURES` is the fixed allowlist in `models/forecast/config.py`. `FORBIDDEN_FEATURES` must never appear in it. It includes DLD's `nearest_metro`, `nearest_mall` and `nearest_landmark` (snapshots that would leak) and every target column.

### Infrastructure table
- **File:** `models/forecast/reference/infrastructure_projects.csv`.
- **Size:** at least 15 real projects.
- **Sources:** every row has an `https://` `source_url` to a public source, and dates are never guessed.
- **Timing:** a project counts only from its `announced_date`, and a completion only from its `actual_completion_date`.
- **`affected_area_ids`:** semicolon-separated ids that exist in `dld.areas`.

### Validation
- **Walk-forward folds per horizon:**
  - Cutoffs step by 3 months (3m horizon), 6 months (1y) and 12 months (3y).
  - The first cutoff is the first day of the month after `first_T + 24 months + end_days`.
  - A training row has `T + end_days < cutoff`. **This is the target-window guard.**
  - Validation rows fall in `[cutoff, next cutoff)`.
  - The last fold is the **test** fold.
- **Tuning:** the tuning folds are at most the last 4 before the test fold.
- **Enough data:** a horizon needs at least 2 folds and at least 1,000 test rows. Otherwise it is `insufficient_data`.

### Model
- **Starting parameters:** XGBoost `reg:squarederror`, `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`, early stopping after 50 rounds.
- **Tuning:** Optuna runs 30 trials, seeded. The objective is the mean validation MAPE over the tuning folds.
- **Final model:** refit on the test fold's training rows, with `n_estimators` = the mean best iteration from the tuning-fold refits.

### Metrics
- **Price error:** APE = `|exp(pred_growth − true_growth) − 1|`. Report MAPE and median APE.
- **Segments:**
  - `all`
  - `off_plan` / `ready`
  - `top5` / `rest` (top 5 areas by training rows)
  - `age_lt1`, `age_1to5`, `age_gt5`, `age_unknown`
  - `age_gt2`, which is used for the gate only
- **Flag:** any segment with MAPE above 25%.

### Baselines
- `no_change` (growth 0).
- `area_trend`: `area_mom_<len>` for the horizon's length, falling back to `city_mom_<len>`, then to 0.

### Gate for each horizon
A horizon passes only if all three hold:
1. MAPE ≤ 0.15 (3m), 0.20 (1y) or 0.30 (3y) on each of `ready`, `top5` and `age_gt2`.
2. The model's `all` MAPE is below both baselines' `all` MAPE.
3. The bootstrap 95% upper bound of the model's `all` MAPE (1,000 row resamples, seeded) is below the lower of the two baselines' `all` MAPE.

A passing horizon is registered as `zestimator-forecast-<h>` with alias `champion`. A failing horizon is never registered.

### Intervals and confidence
- **Intervals:** split conformal on |log error| from the tuning-fold validation predictions, with quantile level `(1 − 0.20)(1 + 1/n)` (reuse `models.price.evaluate.conformal_quantile(errors, 0.20)`).
  - The segment is `"{reg_type}_{unit|villa}"`.
  - A segment with fewer than 200 rows falls back to the pooled value.
- **Confidence labels:**
  - HIGH: `base_level == "building"`, `base_n >= 10` and the latest comparable sale at most 90 days old.
  - MEDIUM: `base_n >= 5` and the latest comparable at most 180 days old.
  - LOW: otherwise, or when the area has fewer than 50 training rows.
- **LOW message:** exactly `"Limited comparable data for this property type/area. Estimate has wide uncertainty (±X%)."`, where X is the rounded half-width of the range as a percentage of the point.

### Output
- **Shape:** the JSON shape in the spec's "Prediction" section.
- **Ranges:** never output a number without a range.
- **Key drivers:** the top 3 |SHAP| contributions above 0.005, rendered from templates and never invented.
- **Models:** registered as MLflow pyfuncs **without `code_paths`** (the repo must be importable). The experiment is `price-forecast`.

### Environment and safety
- **Before settings:** call `load_dotenv()` before any settings are read.
- **Database settings:** `DbSettings.from_env()` raises an error when no port is set. The project DB is on port 5433.
  - NEVER connect to port 5432.
  - Never run `docker compose down -v`.
- **Windows DLL hazard:** `models/forecast/__init__.py` must `import models.price` first. That module preloads the system `msvcp140.dll` before pandas/pyarrow load, which LightGBM and XGBoost need. Do not reorder those imports.
- **Tests:**
  - Never need a GPU or the network; tests force `device="cpu"`.
  - Use the MLflow fixture `temp_mlflow`, copied into `tests/models/forecast/conftest.py` (it touches the sqlite store during setup). Create experiments inside its `artifact_location`, never in `./mlruns`.
  - Run sequentially, in the foreground. Never background your own test run.
  - Layout: no `__init__.py` under `tests/`. Test files are `tests/models/forecast/test_forecast_<topic>.py`; shared helpers go in `tests/models/forecast/forecast_fixtures.py`, imported as `from forecast_fixtures import ...`.
- **Git:**
  - Work on `master`. Stage specific files only; never `git add -A`. Never commit `.env`, `data/`, `mlruns/` or `.superpowers/`. Add `data/forecast/` to `.gitignore` in Task 1.
  - Commit trailer, verbatim: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **Commands and task completion:**
  - Run from the repo root `C:\Users\cnaya\OneDrive\Desktop\zestimator` in Git Bash, always through `uv run`.
  - Finish every task with: `uv run ruff format .`, then `uv run ruff check . && uv run ruff format --check .`, then `uv run pytest -q -W error tests/models/forecast`.

### User stop points
At each of these, the implementer reports and the controller shows the user before continuing:
1. The `build --sample 10000` data-quality report (Task 4).
2. The infrastructure table (Task 5).
3. The segmented results table from the real `train` (Task 11).

## Controller rulings made while planning

1. **Months become fixed day counts** (the table above). This makes the windows exact and lets Polars `rolling` compute them. It is a spec amendment.
2. **Median of ppsm, then ln** for base and target, rather than the median of ln ppsm. The two differ only for even counts. This matches how the spec phrases it.
3. **Tuning uses at most the last 4 tuning folds.** The 3m horizon has about 20 folds; tuning on all of them would take hours. Every fold is still scored and reported with the final parameters, and the tuning-fold refits provide the conformal calibration. This is a spec amendment.
4. **The final model uses a fixed tree count.** It is refit with `n_estimators` = the mean best iteration over the tuning-fold refits, with no early stopping, because the test fold must never guide training.
5. **The 3y horizon will very likely be `insufficient_data` with the 2023 file.**
   - The guard needs T + 1126 d < cutoff.
   - With 24 months of history before the first cutoff, the first 3y cutoff is about 2019-02.
   - Its validation rows need targets reaching 2022–23, but the data ends 2023-03-17.
   - This is reported, not worked around, and the README explains that a newer DLD file fixes it.
6. **The 10,000-row sample is seeded and random.** It reports data quality only. Because building windows are sparse in a sample, most sampled rows end up `no_base`, and the report says so.
7. **Forecasts are computed at T = data_end + 1 day**, so the base window includes the last day of data. `as_of` in the JSON is `data_end`.
8. **Key drivers come from the 1y model** when it is deployed, otherwise from the 3m model, otherwise the list is empty.
9. **The CLI end-to-end test monkeypatches `load_rows` and the area-id lookup.** It uses a synthetic history, matching Phase 3's CLI tests. The real SQL path is covered by a separate database test that loads `tests/fixtures/price_sample.csv` through Phase 2's `run_pipeline`.
10. **Project names become a categorical** only when a project has at least 20 training rows. All others map to `"(other)"`. This keeps XGBoost's category count bounded.
11. **A target window that ends after the data end gets its own status, `window_open`**, alongside the spec's `no_base` and `no_target`. A partly observed window would bias the median.
12. **Categories are fitted once per horizon, on the test fold's training rows, and used for every fold.** Unused category levels are harmless; this keeps one encoding per model.
13. **`planned_completion_date` is the date stated at announcement.** Later revisions would leak hindsight into older rows.
14. **Early stopping and fold scoring share each fold's validation rows**, as the spec states. The test fold never guides training (ruling 4).
15. **A forecast's range is `estimate × exp(growth ± half-width)`.** It covers growth uncertainty; the current estimate's own range is reported separately in `current_range_80`.
16. **A missing champion is reported as `not_deployed`.** The reason is the latest run's gate reason when that run failed or had too little data; otherwise it is `"no registered <h> model"`. `as_of` other than the data end is rejected (ruling 7).

## File map

| File | Responsibility | Task |
|---|---|---|
| `models/forecast/__init__.py`, `config.py`, `rows.py`; `.gitignore`; `tests/models/forecast/{forecast_fixtures.py,conftest.py}` | constants and horizons; row validation, dedup, outliers, quality report | 1 |
| `models/forecast/targets.py` | keys, rolling helper, base, target windows | 2 |
| `models/forecast/features.py` (core) | momentum, context, property, categorical matrix | 3 |
| `models/forecast/__main__.py` (build only) | `build [--sample]` + STOP 1 real sample run | 4 |
| `models/forecast/infra.py`, `reference/infrastructure_projects.csv`, `__main__.py` (+infra-check) | table loader/validator; researched sourced table; STOP 2 | 5 |
| `models/forecast/infra.py` (+features), `features.py` (+`build_dataset`), `__main__.py` (build uses them) | as-of infrastructure features; full-dataset allowlist and as-of tests | 6 |
| `models/forecast/folds.py`, `baselines.py` | walk-forward folds with target-window guard; baselines | 7 |
| `models/forecast/intervals.py`, `evaluate.py` | conformal, confidence, LOW message; APE, segments, bootstrap, gate | 8 |
| `models/forecast/model.py`, `train.py` | ForecastModel (save/load/pyfunc/register); per-horizon run | 9 |
| `models/forecast/predict.py` | Forecaster, key drivers, exclusions, JSON | 10 |
| `models/forecast/__main__.py` (+train/evaluate/predict), integration test, real run → STOP 3 | CLI and real training | 11 |
| `README.md`, spec amendments | documentation from the real run | 12 |

---

## Task 1: Config, rows and the synthetic history

**Files:**
- Create:
  - `models/forecast/__init__.py`
  - `models/forecast/config.py`
  - `models/forecast/rows.py`
  - `tests/models/forecast/forecast_fixtures.py`
  - `tests/models/forecast/conftest.py`
  - `tests/models/forecast/test_forecast_rows.py`
- Modify: `.gitignore` (add `data/forecast/` after `data/search/`)

**Interfaces:**

*Consumes:*
- `models.price.data.RAW_SCHEMA`
- `models.price.data.load_homes(settings, TrainConfig) -> HomesData` (with `.rows`, `.data_end`, `.areas`, `.aliases`)
- `models.price.features.derive_segments`
- `models.price.config.FORBIDDEN_FEATURES`
- `ingestion.normalize.map_unique`, `match_key`

*Produces, from `models.forecast.config`:*
- Constants: `HORIZONS`, `HORIZON_SPECS`, `BASE_DAYS`, `MIN_LEVEL_SALES`, `MOMENTUM_LAGS`, `TRAILING_DAYS`, `INFRA_TYPES`, `FEATURES`, `CATEGORICAL`, `FORBIDDEN_FEATURES`, `DATA_DIR`.
- `Horizon`, a frozen dataclass with fields `name, start_days, end_days, step_months, momentum, mape_gate`.
- `ForecastConfig`, a frozen dataclass.

*Produces, from `models.forecast.rows`:*
- `DataQuality(loaded: int, dropped: dict[str, int], kept: int, excluded: pl.DataFrame, unscreened_groups: int)`, frozen, with:
  - `.drop_share`
  - `.lines() -> list[str]`
  - `.to_dict() -> dict`
- `EXCLUDED_SCHEMA`: columns `area_id, sub_kind, reg_type, month, reason`.
- `validate_rows(rows: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]`
- `screen_outliers(rows, config) -> tuple[pl.DataFrame, pl.DataFrame, int]`: returns kept rows, excluded rows (with `EXCLUDED_SCHEMA`), and the unscreened group count.
- `prepare_rows(rows, config) -> tuple[pl.DataFrame, DataQuality]`: the kept rows gain `ppsm` and `row_id`.
- `load_rows(settings, config) -> tuple[pl.DataFrame, DataQuality, date, pl.DataFrame, pl.DataFrame]`: returns rows, quality, data_end, areas and aliases.

*Produces, from `forecast_fixtures`:*
- `build_history(seed=0, areas=3, buildings_per_area=3, per_building_per_week=2, start=date(2015,1,5), end=date(2023,3,13)) -> pl.DataFrame`, prepared rows shaped like `load_homes().rows`. Area `a` grows at `0.003 + 0.002*a` per month in log price.
- `AREA_NAMES`

*Produces, from `conftest.py`:* the fixtures `history` and `temp_mlflow`.

- [ ] **Step 1: Ignore the data dir**

Append the line `data/forecast/` to `.gitignore`, directly after `data/search/`.

- [ ] **Step 2: Write the package and config**

`models/forecast/__init__.py`:

```python
"""Multi-horizon home price forecasting (Phase 6)."""

import models.price  # noqa: F401 — Windows DLL preload before pandas/pyarrow/xgboost
```

`models/forecast/config.py`:

```python
"""Constants and tunables for price forecasting.

Spec: docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md
Months are fixed day counts (1 month = 30.4375 days, rounded) so every window is exact.
"""

from dataclasses import dataclass
from pathlib import Path

from models.price.config import FORBIDDEN_FEATURES as PRICE_FORBIDDEN

DATA_DIR = Path("data/forecast")
BASE_DAYS = 92
MIN_LEVEL_SALES = 5
TRAILING_DAYS = 365
MOMENTUM_LAGS = {"3m": 91, "12m": 365, "36m": 1096}
INFRA_TYPES = ("metro_rail", "mall", "school", "park", "mixed_use", "airport")


@dataclass(frozen=True)
class Horizon:
    name: str
    start_days: int
    end_days: int
    step_months: int
    momentum: str  # the momentum length that matches this horizon (baseline and templates)
    mape_gate: float


HORIZON_SPECS = {
    "3m": Horizon("3m", 76, 107, 3, "3m", 0.15),
    "1y": Horizon("1y", 335, 396, 6, "12m", 0.20),
    "3y": Horizon("3y", 1065, 1126, 12, "36m", 0.30),
}
HORIZONS = tuple(HORIZON_SPECS)

CATEGORICAL = ("sub_kind", "area_code", "project_code")
FEATURES = (
    "ln_base_ppsm", "base_level_building", "base_n",
    "area_mom_3m", "area_mom_12m", "area_mom_36m",
    "city_mom_3m", "city_mom_12m", "city_mom_36m", "area_share_12m",
    "days_since_building_sale", "building_sales_12m", "area_sales_12m",
    "off_plan", "log_area_sqm", "bedrooms", "building_age_proxy_years",
    *(f"infra_active_{kind}" for kind in INFRA_TYPES),
    "infra_months_to_next", "infra_completed_24m",
    *(f"infra_mix_{kind}" for kind in INFRA_TYPES),
    *CATEGORICAL,
)  # fmt: skip
FORBIDDEN_FEATURES = (
    *PRICE_FORBIDDEN, "ppsm", "is_clean",
    *(f"growth_{name}" for name in HORIZONS),
    *(f"target_ppsm_{name}" for name in HORIZONS),
    *(f"target_n_{name}" for name in HORIZONS),
)  # fmt: skip


@dataclass(frozen=True)
class ForecastConfig:
    sample_rows: int | None = None
    seed: int = 7
    max_drop_share: float = 0.30
    outlier_z: float = 3.0
    outlier_min_group: int = 10
    min_train_months: int = 24
    tune_folds: int = 4
    n_trials: int = 30
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    alpha: float = 0.20
    min_conformal_rows: int = 200
    min_folds: int = 2
    min_test_rows: int = 1_000
    n_bootstrap: int = 1_000
    top_areas: int = 5
    low_confidence_area_rows: int = 50
    min_project_rows: int = 20
    min_driver_contribution: float = 0.005
    device: str = "cuda"
    experiment: str = "price-forecast"
    model_prefix: str = "zestimator-forecast"
    price_model_uri: str = "models:/zestimator-price@champion"
```

- [ ] **Step 3: Write the synthetic history and fixtures**

`tests/models/forecast/forecast_fixtures.py`:

```python
"""Synthetic prepared home rows with known growth, shaped like models.price.data.load_homes."""

import math
from datetime import date, timedelta

import numpy as np
import polars as pl

from models.price.data import RAW_SCHEMA
from models.price.features import derive_segments

AREA_NAMES = {1: "Dubai Marina", 2: "Jumeirah Village Circle", 3: "Arabian Ranches"}


def monthly_growth(area_id: int) -> float:
    return 0.003 + 0.002 * area_id


def build_history(
    seed: int = 0,
    areas: int = 3,
    buildings_per_area: int = 3,
    per_building_per_week: int = 2,
    start: date = date(2015, 1, 5),
    end: date = date(2023, 3, 13),
) -> pl.DataFrame:
    """Weekly sales per building, plus one villa sale per area per week (area-level only).

    ln ppsm = ln(9,000) + 0.1 * area + monthly_growth(area) * months + N(0, 0.03).
    Off-plan units are every third building; bedrooms cycle 0..3.
    """
    rng = np.random.default_rng(seed)
    records = []
    day = start
    while day <= end:
        months = (day - start).days / 30.4375
        for area_id in range(1, areas + 1):
            level = math.log(9_000.0) + 0.1 * area_id + monthly_growth(area_id) * months
            for building in range(buildings_per_area):
                for sale in range(per_building_per_week):
                    size = float(60 + 20 * ((building + sale) % 4))
                    ppsm = math.exp(level + rng.normal(0.0, 0.03))
                    records.append(
                        {
                            "transaction_id": f"s{len(records)}",
                            "instance_date": day + timedelta(days=int(rng.integers(0, 7))),
                            "property_type": "unit",
                            "property_sub_type": "Flat",
                            "reg_type": "off_plan" if building % 3 == 0 else "ready",
                            "area_id": area_id,
                            "building_name": f"Tower {area_id}-{building}",
                            "project_name": f"Project {area_id}",
                            "rooms": "Studio" if sale % 4 == 0 else f"{1 + sale % 3} B/R",
                            "has_parking": True,
                            "area_sqm": size,
                            "price_aed": round(ppsm * size, 0),
                            "ingest_run_id": 1,
                            "is_clean": True,
                        }
                    )
            villa_ppsm = math.exp(level + 0.2 + rng.normal(0.0, 0.03))
            records.append(
                {
                    "transaction_id": f"v{len(records)}",
                    "instance_date": day,
                    "property_type": "villa",
                    "property_sub_type": "Villa",
                    "reg_type": "ready",
                    "area_id": area_id,
                    "building_name": None,
                    "project_name": f"Project {area_id}",
                    "rooms": "4 B/R",
                    "has_parking": True,
                    "area_sqm": 300.0,
                    "price_aed": round(villa_ppsm * 300.0, 0),
                    "ingest_run_id": 1,
                    "is_clean": True,
                }
            )
        day += timedelta(days=7)
    frame = pl.DataFrame(records, schema=RAW_SCHEMA)
    return derive_segments(frame).sort("instance_date", "transaction_id")
```

`tests/models/forecast/conftest.py`:

```python
import pytest
from forecast_fixtures import build_history


@pytest.fixture(scope="session")
def history():
    return build_history()


@pytest.fixture
def temp_mlflow(tmp_path, monkeypatch):
    """Throwaway MLflow store; touched during setup so alembic's logging reset happens here."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", uri)
    monkeypatch.setattr("mlflow.tracking.fluent._active_experiment_id", None)
    mlflow.search_experiments()
    yield {"tracking_uri": uri, "artifact_location": (tmp_path / "artifacts").as_uri()}
    while mlflow.active_run():
        mlflow.end_run()
```

- [ ] **Step 4: Write the failing tests**

`tests/models/forecast/test_forecast_rows.py`:

```python
import dataclasses
from datetime import date

import polars as pl
import pytest
from forecast_fixtures import build_history

from models.forecast.config import (
    FEATURES,
    FORBIDDEN_FEATURES,
    HORIZON_SPECS,
    HORIZONS,
    ForecastConfig,
)
from models.forecast.rows import DataQuality, prepare_rows, screen_outliers, validate_rows

SMALL = build_history(areas=2, buildings_per_area=2, start=date(2020, 1, 6), end=date(2020, 6, 29))


def test_config_is_consistent():
    assert HORIZONS == ("3m", "1y", "3y")
    assert [HORIZON_SPECS[h].end_days for h in HORIZONS] == [107, 396, 1126]
    assert not set(FEATURES) & set(FORBIDDEN_FEATURES)
    assert {"nearest_metro", "growth_1y", "ppsm"} <= set(FORBIDDEN_FEATURES)
    assert len(FEATURES) == len(set(FEATURES))


def _with(frame: pl.DataFrame, index: int, **values) -> pl.DataFrame:
    rows = frame.to_dicts()
    rows[index] = {**rows[index], **values}
    return pl.DataFrame(rows, schema=frame.schema)


def test_validate_rows_drops_each_bad_value_with_its_reason():
    frame = SMALL.head(10)
    frame = _with(frame, 0, price_aed=0.0)
    frame = _with(frame, 1, area_sqm=None)
    frame = _with(frame, 2, instance_date=None)
    frame = _with(frame, 3, area_id=None)
    frame = _with(frame, 4, transaction_id=frame["transaction_id"][5])
    frame = _with(frame, 6, is_clean=False)
    kept, dropped = validate_rows(frame)
    assert dropped == {
        "not_clean": 1,
        "bad_price": 1,
        "bad_size": 1,
        "bad_date": 1,
        "missing_area": 1,
        "duplicate_transaction_id": 1,
        "repeat_sale": 0,
    }
    assert kept.height == 4


def test_validate_rows_keeps_the_last_of_a_repeat_sale():
    frame = SMALL.filter(pl.col("building_name").is_not_null()).head(3)
    first = frame.row(0, named=True)
    copy = {**first, "transaction_id": "copy"}
    frame = pl.concat([frame, pl.DataFrame([copy], schema=frame.schema)])
    kept, dropped = validate_rows(frame)
    assert dropped["repeat_sale"] == 1
    assert "copy" in kept["transaction_id"].to_list()
    assert first["transaction_id"] not in kept["transaction_id"].to_list()


def test_outliers_are_screened_within_area_kind_month():
    frame = SMALL.with_columns((pl.col("price_aed") / pl.col("area_sqm")).alias("ppsm"))
    in_march = (pl.col("instance_date").dt.month() == 3) & pl.col("building_name").is_not_null()
    victim = frame.filter(in_march).row(0, named=True)
    frame = frame.with_columns(
        pl.when(pl.col("transaction_id") == victim["transaction_id"])
        .then(pl.col("ppsm") * 5)
        .otherwise(pl.col("ppsm"))
        .alias("ppsm")
    )
    kept, excluded, unscreened = screen_outliers(frame, ForecastConfig())
    assert excluded.height >= 1  # the planted one; N(0, 0.03) noise rarely adds another
    assert (excluded["area_id"] == victim["area_id"]).any()
    assert set(excluded["reason"]) == {"outlier"}
    assert victim["transaction_id"] not in kept["transaction_id"].to_list()
    assert unscreened > 0  # villa groups have ~4 sales a month, under the 10-sale minimum


def test_prepare_rows_reports_drops_and_adds_ids():
    frame = _with(SMALL, 0, price_aed=-1.0)
    rows, quality = prepare_rows(frame, ForecastConfig())
    assert isinstance(quality, DataQuality)
    assert quality.loaded == SMALL.height
    assert quality.kept == rows.height == SMALL.height - sum(quality.dropped.values())
    assert quality.dropped["bad_price"] == 1
    assert rows["row_id"].to_list() == list(range(rows.height))
    assert (rows["ppsm"] > 0).all()
    lines = quality.lines()
    total = sum(quality.dropped.values())
    share = 100 * total / quality.loaded
    assert lines[0] == f"Dropped {total:,} of {quality.loaded:,} rows ({share:.1f}%)"
    assert any(line.strip().startswith("bad_price") for line in lines)
    assert quality.drop_share == pytest.approx(total / quality.loaded)
    assert quality.to_dict()["dropped"]["bad_price"] == 1


def test_sampling_is_seeded():
    config = dataclasses.replace(ForecastConfig(), sample_rows=50)
    first, _ = prepare_rows(SMALL, config)
    again, _ = prepare_rows(SMALL, config)
    other, _ = prepare_rows(SMALL, dataclasses.replace(config, seed=8))
    assert first.height <= 50
    assert first["transaction_id"].to_list() == again["transaction_id"].to_list()
    assert first["transaction_id"].to_list() != other["transaction_id"].to_list()


def test_load_rows_reads_postgres(pg_test_db):
    from pathlib import Path

    from ingestion.pipeline import run_pipeline
    from models.forecast.rows import load_rows

    run_pipeline(Path("tests/fixtures/price_sample.csv"), pg_test_db)
    rows, quality, data_end, areas, aliases = load_rows(pg_test_db, ForecastConfig())
    assert rows.height == quality.kept > 0
    assert data_end == rows["instance_date"].max()
    assert areas.height > 0 and aliases.height > 0
    assert {"ppsm", "row_id", "sub_kind", "reg_type"} <= set(rows.columns)
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_rows.py`

Expected: FAIL with `ModuleNotFoundError`-style import error for `models.forecast.rows`.

- [ ] **Step 6: Write rows.py**

`models/forecast/rows.py`:

```python
"""Home sales for forecasting: validation, dedup, robust outliers, and a quality report."""

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from ingestion.config import DbSettings
from models.forecast.config import ForecastConfig
from models.price.config import TrainConfig
from models.price.data import load_homes

DROP_REASONS = (
    "not_clean", "bad_price", "bad_size", "bad_date", "missing_area",
    "duplicate_transaction_id", "repeat_sale", "outlier",
)  # fmt: skip
EXCLUDED_SCHEMA = {
    "area_id": pl.Int64,
    "sub_kind": pl.Utf8,
    "reg_type": pl.Utf8,
    "month": pl.Date,
    "reason": pl.Utf8,
}
REPEAT_KEY = ("building_name", "instance_date", "area_sqm", "price_aed")
MAD_SCALE = 1.4826


@dataclass(frozen=True)
class DataQuality:
    loaded: int
    dropped: dict[str, int]
    kept: int
    excluded: pl.DataFrame = field(repr=False)
    unscreened_groups: int = 0

    @property
    def drop_share(self) -> float:
        return sum(self.dropped.values()) / self.loaded if self.loaded else 0.0

    def lines(self) -> list[str]:
        total = sum(self.dropped.values())
        out = [f"Dropped {total:,} of {self.loaded:,} rows ({100 * self.drop_share:.1f}%)"]
        out += [f"  {reason:<26}{count:>10,}" for reason, count in self.dropped.items()]
        out.append(f"  outlier groups not screened (<10 sales): {self.unscreened_groups:,}")
        out.append(f"Kept {self.kept:,} rows")
        return out

    def to_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "dropped": dict(self.dropped),
            "kept": self.kept,
            "drop_share": self.drop_share,
            "unscreened_groups": self.unscreened_groups,
        }


def validate_rows(rows: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    dropped = {}
    checks = (
        ("not_clean", pl.col("is_clean").fill_null(False)),
        ("bad_price", pl.col("price_aed").fill_null(0.0) > 0),
        ("bad_size", pl.col("area_sqm").fill_null(0.0) > 0),
        ("bad_date", pl.col("instance_date").is_not_null()),
        ("missing_area", pl.col("area_id").is_not_null()),
    )
    for reason, keep in checks:
        before = rows.height
        rows = rows.filter(keep)
        dropped[reason] = before - rows.height
    before = rows.height
    rows = rows.unique("transaction_id", keep="first", maintain_order=True)
    dropped["duplicate_transaction_id"] = before - rows.height
    before = rows.height
    repeat = rows.filter(pl.col("building_name").is_not_null())
    single = rows.filter(pl.col("building_name").is_null())
    repeat = repeat.unique(list(REPEAT_KEY), keep="last", maintain_order=True)
    rows = pl.concat([repeat, single]).sort("instance_date", "transaction_id")
    dropped["repeat_sale"] = before - rows.height
    return rows, dropped


def screen_outliers(
    rows: pl.DataFrame, config: ForecastConfig
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Robust z of ln ppsm within area x sub_kind x month; |z| > outlier_z is excluded."""
    group = ["area_id", "sub_kind", "_month"]
    frame = rows.with_columns(
        pl.col("instance_date").dt.truncate("1mo").alias("_month"),
        pl.col("ppsm").log().alias("_ln"),
    )
    stats = frame.group_by(group).agg(
        pl.col("_ln").median().alias("_median"),
        (pl.col("_ln") - pl.col("_ln").median()).abs().median().alias("_mad"),
        pl.len().alias("_n"),
    )
    frame = frame.join(stats, on=group, how="left", maintain_order="left")
    screened = (pl.col("_n") >= config.outlier_min_group) & (pl.col("_mad") > 0)
    z = (pl.col("_ln") - pl.col("_median")) / (MAD_SCALE * pl.col("_mad"))
    is_outlier = (screened & (z.abs() > config.outlier_z)).fill_null(False)
    frame = frame.with_columns(is_outlier.alias("_outlier"))
    excluded = frame.filter(pl.col("_outlier")).select(
        "area_id",
        "sub_kind",
        "reg_type",
        pl.col("_month").alias("month"),
        pl.lit("outlier").alias("reason"),
    )
    unscreened = stats.filter((pl.col("_n") < config.outlier_min_group) | (pl.col("_mad") <= 0))
    kept = frame.filter(~pl.col("_outlier")).drop(
        "_month", "_ln", "_median", "_mad", "_n", "_outlier"
    )
    return kept, excluded.cast(EXCLUDED_SCHEMA), unscreened.height


def prepare_rows(rows: pl.DataFrame, config: ForecastConfig) -> tuple[pl.DataFrame, DataQuality]:
    loaded = rows.height
    if config.sample_rows is not None and rows.height > config.sample_rows:
        rows = rows.sample(config.sample_rows, seed=config.seed).sort(
            "instance_date", "transaction_id"
        )
        loaded = rows.height
    kept, dropped = validate_rows(rows)
    kept = kept.with_columns((pl.col("price_aed") / pl.col("area_sqm")).alias("ppsm"))
    kept, excluded, unscreened = screen_outliers(kept, config)
    dropped["outlier"] = excluded.height
    kept = kept.sort("instance_date", "transaction_id").with_row_index("row_id")
    kept = kept.with_columns(pl.col("row_id").cast(pl.Int64))
    quality = DataQuality(loaded, dropped, kept.height, excluded, unscreened)
    return kept, quality


def load_rows(
    settings: DbSettings, config: ForecastConfig
) -> tuple[pl.DataFrame, DataQuality, date, pl.DataFrame, pl.DataFrame]:
    homes = load_homes(settings, TrainConfig())
    rows, quality = prepare_rows(homes.rows, config)
    return rows, quality, rows["instance_date"].max(), homes.areas, homes.aliases
```

About `prepare_rows`: when sampling, `loaded` counts the sampled rows. The report then describes the sample it actually screened.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/models/forecast/test_forecast_rows.py`

Expected: every test passes.

If the expected dictionary in `test_validate_rows_drops_each_bad_value_with_its_reason` doesn't match, check how the edits interact: the duplicate-id row and its original are both valid, so exactly one drop is expected. Fix the code, not the expectation.

- [ ] **Step 8: Lint, test, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/models/forecast
git add .gitignore models/forecast/__init__.py models/forecast/config.py models/forecast/rows.py tests/models/forecast/forecast_fixtures.py tests/models/forecast/conftest.py tests/models/forecast/test_forecast_rows.py
git commit -m "feat(forecast): config, row validation, dedup, robust outliers and a quality report

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Keys, the rolling-window helper, base prices and growth targets

**Files:**
- Create:
  - `models/forecast/targets.py`
  - `tests/models/forecast/test_forecast_targets.py`

**Interfaces:**

*Consumes:*
- `models.forecast.config`: `BASE_DAYS`, `HORIZON_SPECS`, `HORIZONS`, `MIN_LEVEL_SALES`
- `ingestion.normalize`: `map_unique`, `match_key`
- Prepared rows from Task 1. Targets read only `row_id`, `transaction_id`, `instance_date`, `area_id`, `sub_kind`, `building_name` and `ppsm`.

*Produces, from `models.forecast.targets`:*
- `STATUSES = ("usable", "no_base", "no_target", "window_open")`
- `add_keys(frame) -> pl.DataFrame`: adds `building_key`, `area_key`, `city_key`.
- `rolling_stats(frame, key, *, start_days, length_days, closed, prefix) -> pl.DataFrame`
  - Adds `{prefix}_ppsm` (median) and `{prefix}_n` (Int64 count) for same-key rows whose **ppsm is not null** and whose date is in the window.
  - The window starts at T + `start_days` and spans `length_days`. With `closed="left"` it is `[start, start + length)`; with `closed="both"` both ends are included.
  - A null key gives a null median and a count of 0.
  - The output is sorted by `row_id`.
- `add_base(frame) -> pl.DataFrame`: adds `building_base_ppsm`, `building_base_n`, `area_base_ppsm`, `area_base_n`, `base_level` (`"building"`, `"area"` or null), `base_ppsm` and `base_n`.
- `add_targets(frame, data_end) -> pl.DataFrame`: for each horizon h, adds `target_ppsm_<h>`, `target_n_<h>`, `status_<h>` (one of `STATUSES`) and `growth_<h>`. `growth_<h>` is non-null exactly when the status is `usable`.
- `TargetReport(rows: int, horizons: dict[str, dict])`, frozen, with `.lines()` and `.to_dict()`. Each horizon's dict has one count per status plus `last_t` (a date or None).
- `target_report(frame) -> TargetReport`: counts only rows with a non-null `ppsm`.
- `build_targets(rows, data_end) -> tuple[pl.DataFrame, TargetReport]`: runs `add_keys`, then `add_base`, then `add_targets`.

**Design notes:**
- **Why a null `ppsm` never counts.** Task 10 appends "query rows" dated T = data_end + 1 day, with a null `ppsm`, to compute live features. The same code must compute training and serving features, so query rows must never count toward any other row's window.
- **Why an extra status, `window_open`.** A target window that ends after `data_end` is only partly observed, and its median would be biased. Such rows are excluded with their own status, `window_open`, which is added to the spec's `no_base` and `no_target` (ruling 11).

- [ ] **Step 1: Write the failing tests**

`tests/models/forecast/test_forecast_targets.py`:

```python
import math
from datetime import date, timedelta

import polars as pl
import pytest

from models.forecast.targets import (
    STATUSES,
    add_base,
    add_keys,
    build_targets,
    rolling_stats,
)

T = date(2020, 6, 1)
SCHEMA = {
    "row_id": pl.Int64,
    "transaction_id": pl.Utf8,
    "instance_date": pl.Date,
    "area_id": pl.Int64,
    "sub_kind": pl.Utf8,
    "building_name": pl.Utf8,
    "ppsm": pl.Float64,
}


def sale(days, ppsm, building="Tower A", area_id=1, sub_kind="flat", name=None):
    return {
        "transaction_id": name,
        "instance_date": T + timedelta(days=days),
        "area_id": area_id,
        "sub_kind": sub_kind,
        "building_name": building,
        "ppsm": ppsm,
    }


def frame_of(sales):
    rows = [
        {**record, "row_id": index, "transaction_id": record["transaction_id"] or f"t{index}"}
        for index, record in enumerate(sales)
    ]
    return pl.DataFrame(rows, schema=SCHEMA)


def anchor(frame, name="anchor"):
    return frame.filter(pl.col("transaction_id") == name).row(0, named=True)


# Area 1: the anchor at T, five building sales inside [T-92, T), one just outside it,
# five inside the 3m window [T+76, T+107] and one just outside each end.
AREA_ONE = [
    sale(0, 100.0, name="anchor"),
    sale(-93, 999.0),
    sale(-92, 10.0),
    sale(-50, 20.0),
    sale(-30, 30.0),
    sale(-10, 40.0),
    sale(-1, 50.0),
    sale(75, 999.0),
    sale(76, 60.0),
    sale(80, 60.0),
    sale(90, 60.0),
    sale(100, 66.0),
    sale(107, 66.0),
    sale(108, 999.0),
]


def test_add_keys_builds_the_three_levels():
    frame = add_keys(frame_of([sale(0, 1.0, building="Al Noor Tower"), sale(0, 1.0, building=None)]))
    assert frame["building_key"].to_list() == ["1|noor tower", None]
    assert frame["area_key"].to_list() == ["1|flat", "1|flat"]
    assert frame["city_key"].to_list() == ["flat", "flat"]


def test_rolling_stats_window_edges():
    frame = add_keys(frame_of(AREA_ONE))
    base = rolling_stats(
        frame, "building_key", start_days=-92, length_days=92, closed="left", prefix="b"
    )
    row = anchor(base)
    assert row["b_n"] == 5  # -92 is in, -93 is out, the anchor's own day is out
    assert row["b_ppsm"] == 30.0
    ahead = rolling_stats(
        frame, "building_key", start_days=76, length_days=31, closed="both", prefix="f"
    )
    row = anchor(ahead)
    assert row["f_n"] == 5  # 76 and 107 are in, 75 and 108 are out
    assert row["f_ppsm"] == 60.0
    assert base["row_id"].to_list() == sorted(base["row_id"].to_list())


def test_rolling_stats_ignores_null_ppsm_and_null_keys():
    sales = [*AREA_ONE, sale(-5, None, name="query"), sale(-5, 5.0, building=None, name="villa")]
    frame = add_keys(frame_of(sales))
    out = rolling_stats(
        frame, "building_key", start_days=-92, length_days=92, closed="left", prefix="b"
    )
    assert anchor(out)["b_n"] == 5  # the null-ppsm row at -5 does not count
    assert anchor(out, "villa")["b_n"] == 0
    assert anchor(out, "villa")["b_ppsm"] is None


def test_base_prefers_the_building_then_the_area():
    sales = [
        sale(0, 100.0, building="B", area_id=2, name="anchor"),
        *[sale(-d, 10.0 * d, building="B", area_id=2) for d in (5, 10, 15, 20)],
        sale(-25, 250.0, building="C", area_id=2),
    ]
    row = anchor(add_base(add_keys(frame_of(sales))))
    assert row["building_base_n"] == 4
    assert row["base_level"] == "area"
    assert row["base_n"] == 5
    assert row["base_ppsm"] == 150.0  # median of 50, 100, 150, 200, 250
    one = anchor(add_base(add_keys(frame_of(AREA_ONE))))
    assert (one["base_level"], one["base_n"], one["base_ppsm"]) == ("building", 5, 30.0)


def test_no_base_when_the_area_has_fewer_than_five_sales():
    sales = [
        sale(0, 100.0, area_id=4, name="anchor"),
        *[sale(-d, 50.0, area_id=4) for d in (1, 2, 3, 4)],
    ]
    frame, report = build_targets(frame_of(sales), T + timedelta(days=2000))
    assert anchor(frame)["base_level"] is None
    assert anchor(frame)["status_3m"] == "no_base"
    assert report.horizons["3m"]["no_base"] == 5  # no row has five earlier sales
    assert report.rows == 5


def test_growth_is_the_log_ratio_at_the_base_level():
    frame, _ = build_targets(frame_of(AREA_ONE), T + timedelta(days=2000))
    row = anchor(frame)
    assert row["status_3m"] == "usable"
    assert row["target_n_3m"] == 5
    assert row["growth_3m"] == pytest.approx(math.log(60.0 / 30.0))
    assert row["status_1y"] == "no_target"  # nothing in [T+335, T+396]
    assert row["growth_1y"] is None


def test_target_stays_at_the_building_level():
    sales = [
        sale(0, 100.0, building="D", area_id=3, name="anchor"),
        *[sale(-d, 20.0, building="D", area_id=3) for d in (5, 10, 15, 20, 25)],
        *[sale(80 + d, 40.0, building="D", area_id=3) for d in (0, 1, 2)],
        *[sale(80 + d, 40.0, building="E", area_id=3) for d in (0, 1, 2)],
    ]
    frame, _ = build_targets(frame_of(sales), T + timedelta(days=2000))
    row = anchor(frame)
    assert row["base_level"] == "building"
    assert row["target_n_3m"] == 3  # the area has 6 sales in the window; the building has 3
    assert row["status_3m"] == "no_target"
    assert row["growth_3m"] is None


def test_open_windows_are_excluded_and_counted():
    frame, report = build_targets(frame_of(AREA_ONE), T + timedelta(days=100))
    row = anchor(frame)
    assert row["status_3m"] == "window_open"  # T + 107 is after the data end
    assert row["growth_3m"] is None
    assert report.horizons["3m"]["usable"] == 0
    for name in ("3m", "1y", "3y"):
        counts = frame.group_by(f"status_{name}").len()
        expected = dict(counts.iter_rows())
        assert {s: report.horizons[name][s] for s in STATUSES if s in expected} == expected
        assert sum(report.horizons[name][s] for s in STATUSES) == report.rows


def test_report_lines_and_dict():
    frame, report = build_targets(frame_of(AREA_ONE), T + timedelta(days=2000))
    assert report.horizons["3m"]["last_t"] == T
    assert report.lines()[0] == f"Target rows: {len(AREA_ONE):,}"
    assert any(line.strip().startswith("3m:") and str(T) in line for line in report.lines())
    assert report.to_dict()["horizons"]["3m"]["last_t"] == T.isoformat()
    assert report.to_dict()["horizons"]["1y"]["last_t"] is None


def test_history_targets_track_the_known_growth(history):
    data_end = history["instance_date"].max()
    rows = history.with_row_index("row_id").with_columns(
        pl.col("row_id").cast(pl.Int64),
        (pl.col("price_aed") / pl.col("area_sqm")).alias("ppsm"),
    )
    frame, report = build_targets(rows, data_end)
    usable = frame.filter(
        (pl.col("status_1y") == "usable") & (pl.col("area_id") == 1) & (pl.col("sub_kind") == "flat")
    )
    assert usable.height > 1000
    assert (usable["base_level"] == "building").all()
    # area 1 grows 0.005 a month; base centre T-46 d to target centre T+365.5 d is ~13.5 months
    assert usable["growth_1y"].median() == pytest.approx(0.005 * 411.5 / 30.4375, abs=0.01)
    assert report.horizons["3y"]["last_t"] <= data_end - timedelta(days=1126)
```

The `history` fixture has no `row_id` or `ppsm` columns yet, because Task 1's `build_history` returns rows as `load_homes` returns them. So the last test adds both itself.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_targets.py`

Expected: FAIL with an import error for `models.forecast.targets`.

- [ ] **Step 3: Write targets.py**

`models/forecast/targets.py`:

```python
"""Base price and forward growth targets per sale.

Spec: docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md ("Targets").
Every window is an exact day range relative to the sale date T (see config). Rows with a
null ppsm (serving-time query rows) never count toward any window.
"""

from dataclasses import dataclass
from datetime import date

import polars as pl

from ingestion.normalize import map_unique, match_key
from models.forecast.config import BASE_DAYS, HORIZON_SPECS, HORIZONS, MIN_LEVEL_SALES

STATUSES = ("usable", "no_base", "no_target", "window_open")
LEVELS = ("building", "area")


def add_keys(frame: pl.DataFrame) -> pl.DataFrame:
    """building_key ("{area_id}|{match_key}"), area_key ("{area_id}|{sub_kind}"), city_key."""
    building = map_unique(frame["building_name"], match_key, pl.Utf8)
    return (
        frame.with_columns(building.alias("_building"))
        .with_columns(
            pl.when(pl.col("_building").is_not_null())
            .then(pl.format("{}|{}", "area_id", "_building"))
            .alias("building_key"),
            pl.when(pl.col("sub_kind").is_not_null())
            .then(pl.format("{}|{}", "area_id", "sub_kind"))
            .alias("area_key"),
            pl.col("sub_kind").alias("city_key"),
        )
        .drop("_building")
    )


def rolling_stats(
    frame: pl.DataFrame,
    key: str,
    *,
    start_days: int,
    length_days: int,
    closed: str,
    prefix: str,
) -> pl.DataFrame:
    """Median ppsm and count of same-key sales in a day window placed relative to each row.

    The window starts at T + start_days and spans length_days (Polars offset/period).
    closed="left" gives [start, start + length); closed="both" includes both ends.
    """
    median, count = f"{prefix}_ppsm", f"{prefix}_n"
    keyed = frame.filter(pl.col(key).is_not_null()).sort(key, "instance_date", "row_id")
    stats = keyed.rolling(
        index_column="instance_date",
        period=f"{length_days}d",
        offset=f"{start_days}d",
        closed=closed,
        group_by=key,
    ).agg(
        pl.col("ppsm").median().alias(median),
        pl.col("ppsm").count().alias(count),
    )
    aligned = stats[key].equals(keyed[key]) and stats["instance_date"].equals(
        keyed["instance_date"]
    )
    if not aligned:
        raise RuntimeError(f"rolling output for {key} is not aligned with its input")
    keyed = keyed.hstack(stats.select(median, count))
    unkeyed = frame.filter(pl.col(key).is_null()).with_columns(
        pl.lit(None, dtype=pl.Float64).alias(median), pl.lit(0).alias(count)
    )
    return (
        pl.concat([keyed, unkeyed], how="vertical_relaxed")
        .with_columns(pl.col(count).fill_null(0).cast(pl.Int64))
        .sort("row_id")
    )


def add_base(frame: pl.DataFrame) -> pl.DataFrame:
    """Median ppsm over [T-92 d, T): the building's if it has 5+ sales, else the area's."""
    for level in LEVELS:
        frame = rolling_stats(
            frame,
            f"{level}_key",
            start_days=-BASE_DAYS,
            length_days=BASE_DAYS,
            closed="left",
            prefix=f"{level}_base",
        )
    building = pl.col("building_base_n") >= MIN_LEVEL_SALES
    area = pl.col("area_base_n") >= MIN_LEVEL_SALES

    def pick(part: str) -> pl.Expr:
        return (
            pl.when(building)
            .then(pl.col(f"building_base_{part}"))
            .when(area)
            .then(pl.col(f"area_base_{part}"))
        )

    return frame.with_columns(
        pl.when(building)
        .then(pl.lit("building"))
        .when(area)
        .then(pl.lit("area"))
        .alias("base_level"),
        pick("ppsm").alias("base_ppsm"),
        pick("n").alias("base_n"),
    )


def add_targets(frame: pl.DataFrame, data_end: date) -> pl.DataFrame:
    """growth_<h> = ln(median ppsm in the target window / base), at the base's level."""
    has_base = pl.col("base_level").is_not_null()
    at_building = pl.col("base_level") == "building"
    for name in HORIZONS:
        spec = HORIZON_SPECS[name]
        for level in LEVELS:
            frame = rolling_stats(
                frame,
                f"{level}_key",
                start_days=spec.start_days,
                length_days=spec.end_days - spec.start_days,
                closed="both",
                prefix=f"_{level}_{name}",
            )
        window_end = pl.col("instance_date").dt.offset_by(f"{spec.end_days}d")
        enough = pl.col(f"target_n_{name}") >= MIN_LEVEL_SALES
        status = (
            pl.when(~has_base)
            .then(pl.lit("no_base"))
            .when(window_end > pl.lit(data_end))
            .then(pl.lit("window_open"))
            .when(~enough)
            .then(pl.lit("no_target"))
            .otherwise(pl.lit("usable"))
        )
        frame = (
            frame.with_columns(
                pl.when(at_building)
                .then(pl.col(f"_building_{name}_ppsm"))
                .otherwise(pl.col(f"_area_{name}_ppsm"))
                .alias(f"target_ppsm_{name}"),
                pl.when(at_building)
                .then(pl.col(f"_building_{name}_n"))
                .otherwise(pl.col(f"_area_{name}_n"))
                .alias(f"target_n_{name}"),
            )
            .with_columns(status.alias(f"status_{name}"))
            .with_columns(
                pl.when(pl.col(f"status_{name}") == "usable")
                .then((pl.col(f"target_ppsm_{name}") / pl.col("base_ppsm")).log())
                .alias(f"growth_{name}")
            )
            .drop([f"_{level}_{name}_{part}" for level in LEVELS for part in ("ppsm", "n")])
        )
    return frame


@dataclass(frozen=True)
class TargetReport:
    rows: int
    horizons: dict[str, dict]

    def lines(self) -> list[str]:
        out = [f"Target rows: {self.rows:,}"]
        for name, counts in self.horizons.items():
            parts = ", ".join(f"{status} {counts[status]:,}" for status in STATUSES)
            out.append(f"  {name}: {parts}; last usable T {counts['last_t'] or 'none'}")
        return out

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "horizons": {
                name: {
                    **counts,
                    "last_t": counts["last_t"].isoformat() if counts["last_t"] else None,
                }
                for name, counts in self.horizons.items()
            },
        }


def target_report(frame: pl.DataFrame) -> TargetReport:
    real = frame.filter(pl.col("ppsm").is_not_null())
    horizons = {}
    for name in HORIZONS:
        counts: dict = dict.fromkeys(STATUSES, 0)
        counts.update(dict(real.group_by(f"status_{name}").len().iter_rows()))
        usable = real.filter(pl.col(f"status_{name}") == "usable")
        counts["last_t"] = usable["instance_date"].max() if usable.height else None
        horizons[name] = counts
    return TargetReport(real.height, horizons)


def build_targets(rows: pl.DataFrame, data_end: date) -> tuple[pl.DataFrame, TargetReport]:
    frame = add_targets(add_base(add_keys(rows)), data_end)
    return frame, target_report(frame)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/models/forecast/test_forecast_targets.py`

Expected: every test passes.

**If the alignment check raises**, Polars has not returned the rolling output in input order. Do not remove the check. Instead:
1. Add `pl.col("row_id").last().alias("_row")` to the aggregation. Every row sharing a key and date has the same window, so any one of their stats is valid for all of them.
2. Join the stats back on `(key, instance_date)` after taking one row per `(key, instance_date)`.
3. Record what you changed in the report.

**If Polars warns about sortedness** (the tests run with `-W error`), the input is not sorted by key and date. Fix the sort; do not filter the warning.

- [ ] **Step 5: Lint, test, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/models/forecast
git add models/forecast/targets.py tests/models/forecast/test_forecast_targets.py
git commit -m "feat(forecast): exact-window base prices and growth targets with per-horizon status

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Core as-of features and the model matrix

**Files:**
- Create:
  - `models/forecast/features.py`
  - `tests/models/forecast/test_forecast_features.py`
- Modify: `tests/models/forecast/forecast_fixtures.py` (add `prepared_history`)

**Interfaces:**

*Consumes:*
- From Task 1: `config.BASE_DAYS`, `CATEGORICAL`, `FEATURES`, `MIN_LEVEL_SALES`, `MOMENTUM_LAGS`, `TRAILING_DAYS`, `ForecastConfig`, and `rows.prepare_rows`.
- From Task 2: `targets.rolling_stats`, `add_keys`, `add_base`, `build_targets`.

*Produces, from `models.forecast.features`:*
- Constants:
  - `OTHER_PROJECT = "(other)"`
  - `PROPERTY_FEATURES = ("off_plan", "log_area_sqm", "bedrooms")`
  - `CORE_FEATURES`: the entries of `FEATURES` that do not start with `infra_`.
- `add_momentum(frame)`: needs `add_base` columns. Adds:
  - `city_base_ppsm` and `city_base_n`;
  - `area_mom_{3m,12m,36m}` and `city_mom_{3m,12m,36m}`. Each is the ln change between the base window and the same window lagged by 91, 365 or 1096 days, and is null unless both windows have at least 5 sales.
- `add_activity(frame)`: adds:
  - `building_sales_12m` (null when there is no building);
  - `area_sales_12m`;
  - `area_share_12m`;
  - `days_since_building_sale` and `days_since_area_sale` (Float64 days since the latest earlier sale date, strictly before T). `days_since_area_sale` is a helper for confidence labels, not a model feature.
- `add_property(frame)`: adds `building_age_proxy_years`, `ln_base_ppsm`, `base_level_building`, `off_plan`, `log_area_sqm`, `area_code` (the `area_id` as a string) and `project_code` (the `project_name`).
- `add_core_features(frame)`: runs `add_momentum`, then `add_activity`, then `add_property`.
- `fit_categories(frame, config) -> dict[str, list[str]]`:
  - `sub_kind` and `area_code`: the sorted unique values.
  - `project_code`: the sorted projects with at least `min_project_rows` rows, plus `OTHER_PROJECT`.
- `to_matrix(frame, categories, features=FEATURES) -> pd.DataFrame`:
  - Numeric columns become Float64. Categorical columns become `pd.Categorical` with the fitted categories, so unseen values become NaN.
  - A project that isn't among the categories becomes `OTHER_PROJECT`.
  - Raises `KeyError` naming any missing columns.
- `feature_coverage(frame, features) -> dict[str, float]`: the non-null share of each feature.

*Produces, from `forecast_fixtures`:* `prepared_history(**kwargs) -> tuple[pl.DataFrame, date]`, which runs `prepare_rows` on `build_history(**kwargs)` and returns the rows and the data end.

- [ ] **Step 1: Add the prepared-history helper**

Append to `tests/models/forecast/forecast_fixtures.py`:

```python
def prepared_history(**kwargs):
    """build_history rows run through prepare_rows: (rows with ppsm and row_id, data_end)."""
    from models.forecast.config import ForecastConfig
    from models.forecast.rows import prepare_rows

    rows, _ = prepare_rows(build_history(**kwargs), ForecastConfig())
    return rows, rows["instance_date"].max()
```

- [ ] **Step 2: Write the failing tests**

`tests/models/forecast/test_forecast_features.py`:

```python
import math
from datetime import date, timedelta

import pandas as pd
import polars as pl
import pytest
from forecast_fixtures import prepared_history

from models.forecast.config import CATEGORICAL, ForecastConfig
from models.forecast.features import (
    CORE_FEATURES,
    OTHER_PROJECT,
    add_core_features,
    feature_coverage,
    fit_categories,
    to_matrix,
)
from models.forecast.targets import build_targets

ROWS, DATA_END = prepared_history()


def core(rows, data_end=DATA_END):
    frame, _ = build_targets(rows, data_end)
    return add_core_features(frame)


FRAME = core(ROWS)


def test_core_features_exist_and_keep_row_order():
    assert set(CORE_FEATURES) <= set(FRAME.columns)
    assert FRAME["row_id"].to_list() == ROWS["row_id"].to_list()
    assert not any(name.startswith("infra_") for name in CORE_FEATURES)


def test_area_momentum_matches_the_known_growth():
    late = FRAME.filter(
        (pl.col("area_id") == 1)
        & (pl.col("sub_kind") == "flat")
        & (pl.col("instance_date") > date(2019, 1, 1))
    )
    # area 1 grows 0.005 a month in log price
    assert late["area_mom_3m"].median() == pytest.approx(0.005 * 91 / 30.4375, abs=0.01)
    assert late["area_mom_12m"].median() == pytest.approx(0.005 * 365 / 30.4375, abs=0.01)
    assert late["area_mom_36m"].median() == pytest.approx(0.005 * 1096 / 30.4375, abs=0.02)
    early = FRAME.filter(pl.col("instance_date") < date(2017, 1, 1))
    assert early["area_mom_36m"].null_count() == early.height  # no window 3 years back yet
    city = FRAME.filter((pl.col("sub_kind") == "flat") & (pl.col("instance_date") > date(2019, 1, 1)))
    assert 0.005 * 12 - 0.01 < city["city_mom_12m"].median() < 0.009 * 12 + 0.01


def test_activity_counts_and_recency():
    late = FRAME.filter(
        (pl.col("sub_kind") == "flat") & (pl.col("instance_date") > date(2019, 1, 1))
    )
    assert late["building_sales_12m"].median() == pytest.approx(104, abs=6)  # 2 a week
    assert late["area_share_12m"].median() == pytest.approx(1 / 3, abs=0.02)
    assert late["days_since_building_sale"].max() <= 13
    assert late["days_since_building_sale"].min() >= 1
    villas = FRAME.filter(pl.col("sub_kind") == "villa")
    assert villas["building_sales_12m"].null_count() == villas.height
    assert villas["days_since_building_sale"].null_count() == villas.height
    assert villas.filter(pl.col("instance_date") > date(2016, 1, 1))["days_since_area_sale"].max() <= 13


def test_building_age_proxy_uses_only_earlier_sales():
    tower = FRAME.filter(pl.col("building_key") == "1|tower 1 0").sort("instance_date")
    first_day = tower["instance_date"][0]
    assert tower.filter(pl.col("instance_date") == first_day)["building_age_proxy_years"].null_count() > 0
    later = tower.filter(pl.col("instance_date") > date(2017, 1, 1)).row(0, named=True)
    expected = (later["instance_date"] - first_day).days / 365.25
    assert later["building_age_proxy_years"] == pytest.approx(expected)


def test_property_columns():
    row = FRAME.filter(pl.col("base_level").is_not_null()).row(0, named=True)
    assert row["ln_base_ppsm"] == pytest.approx(math.log(row["base_ppsm"]))
    assert row["base_level_building"] in (0.0, 1.0)
    assert row["log_area_sqm"] == pytest.approx(math.log(row["area_sqm"]))
    assert row["off_plan"] == float(row["reg_type"] == "off_plan")
    assert row["area_code"] == str(row["area_id"])
    assert row["project_code"] == row["project_name"]


def test_features_never_look_at_sales_on_or_after_t():
    cutoff = date(2019, 6, 1)
    later = pl.col("instance_date") >= cutoff
    changed = ROWS.with_columns(
        pl.when(later).then(pl.col("ppsm") * 3).otherwise(pl.col("ppsm")).alias("ppsm")
    ).filter(~later | (pl.col("row_id") % 2 == 0))
    before = FRAME.filter(~later).select("transaction_id", *CORE_FEATURES).sort("transaction_id")
    after = (
        core(changed)
        .filter(~later)
        .select("transaction_id", *CORE_FEATURES)
        .sort("transaction_id")
    )
    assert before.equals(after)


def test_query_rows_see_history_but_change_nothing():
    day = DATA_END + timedelta(days=1)
    template = ROWS.filter(pl.col("building_name").is_not_null()).row(-1, named=True)
    query = pl.DataFrame(
        [{**template, "transaction_id": "query", "instance_date": day, "ppsm": None,
          "row_id": ROWS.height}],
        schema=ROWS.schema,
    )  # fmt: skip
    with_query = core(pl.concat([ROWS, query]))
    others = with_query.filter(pl.col("transaction_id") != "query").select(
        "transaction_id", *CORE_FEATURES
    )
    assert others.equals(FRAME.select("transaction_id", *CORE_FEATURES))
    row = with_query.filter(pl.col("transaction_id") == "query").row(0, named=True)
    assert row["base_level"] == "building"
    recent = ROWS.filter(
        (pl.col("building_name") == template["building_name"])
        & (pl.col("instance_date") >= day - timedelta(days=92))
    )
    assert row["base_n"] == recent.height
    assert row["base_ppsm"] == pytest.approx(recent["ppsm"].median())


def test_to_matrix_orders_columns_and_fixes_categories():
    config = ForecastConfig()
    categories = fit_categories(FRAME, config)
    assert categories["area_code"] == ["1", "2", "3"]
    assert categories["project_code"][-1] == OTHER_PROJECT
    odd = FRAME.head(3).with_columns(
        pl.Series("area_code", ["1", "99", None]),
        pl.Series("project_code", ["Project 1", "Nowhere", None]),
    )
    matrix = to_matrix(odd, categories, CORE_FEATURES)
    assert list(matrix.columns) == list(CORE_FEATURES)
    for name in CATEGORICAL:
        assert isinstance(matrix[name].dtype, pd.CategoricalDtype)
        assert list(matrix[name].cat.categories) == categories[name]
    assert matrix["area_code"].isna().tolist() == [False, True, True]
    assert matrix["project_code"].tolist()[:2] == ["Project 1", OTHER_PROJECT]
    assert pd.isna(matrix["project_code"].tolist()[2])
    numeric = [name for name in CORE_FEATURES if name not in CATEGORICAL]
    assert all(matrix[name].dtype == "float64" for name in numeric)
    with pytest.raises(KeyError, match="infra_active_mall"):
        to_matrix(FRAME, categories)


def test_rare_projects_become_other():
    config = ForecastConfig(min_project_rows=10**9)
    assert fit_categories(FRAME, config)["project_code"] == [OTHER_PROJECT]


def test_feature_coverage():
    coverage = feature_coverage(FRAME, CORE_FEATURES)
    assert set(coverage) == set(CORE_FEATURES)
    assert coverage["log_area_sqm"] == 1.0
    assert 0.0 < coverage["area_mom_36m"] < 1.0
```

The building key in `test_building_age_proxy_uses_only_earlier_sales` is `"1|tower 1 0"`, because `match_key("Tower 1-0")` replaces non-alphanumerics with spaces.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_features.py`

Expected: FAIL with an import error for `models.forecast.features`.

- [ ] **Step 4: Write features.py**

`models/forecast/features.py`:

```python
"""As-of features per sale: momentum, base context, activity, property and categoricals.

Every value uses only sales dated strictly before T. Rows with a null ppsm (serving-time
query rows) never count. Spec: "Features" in
docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md.
"""

import pandas as pd
import polars as pl

from models.forecast.config import (
    BASE_DAYS,
    CATEGORICAL,
    FEATURES,
    MIN_LEVEL_SALES,
    MOMENTUM_LAGS,
    TRAILING_DAYS,
    ForecastConfig,
)
from models.forecast.targets import rolling_stats

DAYS_PER_YEAR = 365.25
OTHER_PROJECT = "(other)"
PROPERTY_FEATURES = ("off_plan", "log_area_sqm", "bedrooms")
CORE_FEATURES = tuple(name for name in FEATURES if not name.startswith("infra_"))
MOMENTUM_LEVELS = ("area", "city")


def _is_sale() -> pl.Expr:
    return pl.col("ppsm").is_not_null()


def add_momentum(frame: pl.DataFrame) -> pl.DataFrame:
    frame = rolling_stats(
        frame,
        "city_key",
        start_days=-BASE_DAYS,
        length_days=BASE_DAYS,
        closed="left",
        prefix="city_base",
    )
    for label, lag in MOMENTUM_LAGS.items():
        for level in MOMENTUM_LEVELS:
            lagged = f"_{level}_lag"
            frame = rolling_stats(
                frame,
                f"{level}_key",
                start_days=-(BASE_DAYS + lag),
                length_days=BASE_DAYS,
                closed="left",
                prefix=lagged,
            )
            enough = (pl.col(f"{level}_base_n") >= MIN_LEVEL_SALES) & (
                pl.col(f"{lagged}_n") >= MIN_LEVEL_SALES
            )
            change = (pl.col(f"{level}_base_ppsm") / pl.col(f"{lagged}_ppsm")).log()
            frame = frame.with_columns(
                pl.when(enough).then(change).alias(f"{level}_mom_{label}")
            ).drop(f"{lagged}_ppsm", f"{lagged}_n")
    return frame


def _trailing_count(frame: pl.DataFrame, key: str, name: str) -> pl.DataFrame:
    counted = rolling_stats(
        frame,
        key,
        start_days=-TRAILING_DAYS,
        length_days=TRAILING_DAYS,
        closed="left",
        prefix="_trailing",
    )
    return counted.with_columns(
        pl.when(pl.col(key).is_not_null()).then(pl.col("_trailing_n")).alias(name)
    ).drop("_trailing_ppsm", "_trailing_n")


def _days_since_previous_sale(frame: pl.DataFrame, key: str, name: str) -> pl.DataFrame:
    sales = (
        frame.filter(_is_sale() & pl.col(key).is_not_null())
        .select(key, pl.col("instance_date").alias("_previous"))
        .unique()
        .sort("_previous")
    )
    joined = frame.sort("instance_date").join_asof(
        sales,
        left_on="instance_date",
        right_on="_previous",
        by=key,
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    days = (pl.col("instance_date") - pl.col("_previous")).dt.total_days().cast(pl.Float64)
    return joined.with_columns(days.alias(name)).drop("_previous").sort("row_id")


def add_activity(frame: pl.DataFrame) -> pl.DataFrame:
    frame = _trailing_count(frame, "building_key", "building_sales_12m")
    frame = _trailing_count(frame, "area_key", "area_sales_12m")
    frame = _trailing_count(frame, "city_key", "_city_sales_12m")
    frame = frame.with_columns(
        pl.when(pl.col("_city_sales_12m") > 0)
        .then(pl.col("area_sales_12m") / pl.col("_city_sales_12m"))
        .alias("area_share_12m")
    ).drop("_city_sales_12m")
    frame = _days_since_previous_sale(frame, "building_key", "days_since_building_sale")
    return _days_since_previous_sale(frame, "area_key", "days_since_area_sale")


def add_property(frame: pl.DataFrame) -> pl.DataFrame:
    first = (
        frame.filter(_is_sale() & pl.col("building_key").is_not_null())
        .group_by("building_key")
        .agg(pl.col("instance_date").min().alias("_first_sale"))
    )
    age = (pl.col("instance_date") - pl.col("_first_sale")).dt.total_days() / DAYS_PER_YEAR
    return (
        frame.join(first, on="building_key", how="left")
        .with_columns(
            pl.when(pl.col("_first_sale") < pl.col("instance_date"))
            .then(age)
            .alias("building_age_proxy_years"),
            pl.col("base_ppsm").log().alias("ln_base_ppsm"),
            (pl.col("base_level") == "building").cast(pl.Float64).alias("base_level_building"),
            (pl.col("reg_type") == "off_plan").cast(pl.Float64).alias("off_plan"),
            pl.col("area_sqm").log().alias("log_area_sqm"),
            pl.col("area_id").cast(pl.Utf8).alias("area_code"),
            pl.col("project_name").alias("project_code"),
        )
        .drop("_first_sale")
        .sort("row_id")
    )


def add_core_features(frame: pl.DataFrame) -> pl.DataFrame:
    return add_property(add_activity(add_momentum(frame)))


def fit_categories(frame: pl.DataFrame, config: ForecastConfig) -> dict[str, list[str]]:
    projects = (
        frame.filter(pl.col("project_code").is_not_null())
        .group_by("project_code")
        .len()
        .filter(pl.col("len") >= config.min_project_rows)
    )
    return {
        "sub_kind": sorted(frame["sub_kind"].drop_nulls().unique().to_list()),
        "area_code": sorted(frame["area_code"].drop_nulls().unique().to_list(), key=int),
        "project_code": sorted(projects["project_code"].to_list()) + [OTHER_PROJECT],
    }


def to_matrix(
    frame: pl.DataFrame, categories: dict[str, list[str]], features=FEATURES
) -> pd.DataFrame:
    """Exactly `features`, in order, as XGBoost input (categories fixed at fit time)."""
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise KeyError(f"feature columns missing: {missing}")
    known = categories["project_code"]
    frame = frame.with_columns(
        pl.when(pl.col("project_code").is_null() | pl.col("project_code").is_in(known))
        .then(pl.col("project_code"))
        .otherwise(pl.lit(OTHER_PROJECT))
        .alias("project_code")
    )
    selected = frame.select(
        pl.col(name).cast(pl.Utf8) if name in CATEGORICAL else pl.col(name).cast(pl.Float64)
        for name in features
    ).to_pandas()
    for name in CATEGORICAL:
        if name in features:
            selected[name] = pd.Categorical(selected[name], categories=categories[name])
    return selected


def feature_coverage(frame: pl.DataFrame, features) -> dict[str, float]:
    total = max(frame.height, 1)
    return {name: 1.0 - frame[name].null_count() / total for name in features}
```

Note on `fit_categories`: `area_code` is sorted numerically (`key=int`), so `["1", "2", "3"]` stays in order and `"10"` sorts after `"9"`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/models/forecast/test_forecast_features.py`

Expected: every test passes.

If `join_asof` rejects `allow_exact_matches` or `check_sortedness` in the installed Polars (1.44), check the signature with `uv run python -c "import polars, inspect; print(inspect.signature(polars.DataFrame.join_asof))"`. Then adapt so that:
- an exact same-day match is still excluded;
- no sortedness warning appears.

If you change the approach, record it in the report.

The as-of test must pass unchanged. If it fails, a feature is looking at the future: find which column differs and fix the code.

- [ ] **Step 6: Lint, test, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/models/forecast
git add models/forecast/features.py tests/models/forecast/test_forecast_features.py tests/models/forecast/forecast_fixtures.py
git commit -m "feat(forecast): as-of momentum, activity and property features plus the model matrix

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: The `build` command and stop point 1

**Files:**
- Create:
  - `models/forecast/__main__.py`
  - `tests/models/forecast/test_forecast_cli.py`
- Modify:
  - `models/forecast/rows.py` (add `excluded_summary`)
  - `tests/models/forecast/forecast_fixtures.py` (add `history_areas`, `history_aliases`, `fake_load_rows`)

**Interfaces:**

*Consumes:*
- From Task 1: `rows.load_rows`, `prepare_rows`, `DataQuality`, and `config.DATA_DIR`, `ForecastConfig`.
- From Task 2: `targets.build_targets`, `TargetReport`.
- From Task 3: `features.add_core_features`, `feature_coverage`, `CORE_FEATURES`.

*Produces:*
- `rows.excluded_summary(excluded: pl.DataFrame) -> list[dict]`: counts per `(area_id, sub_kind, reg_type, reason)`, sorted, under the key `count`.
- `models.forecast.__main__`:
  - `main(argv=None) -> int`
  - `_settings() -> DbSettings` (tests monkeypatch it)
  - `Stages`, which records stage timings to `DATA_DIR / "stage_timings.json"`
  - `write_json(name, payload) -> Path`, which writes into `DATA_DIR`
  - Module globals that tests monkeypatch: `DATA_DIR`, `load_rows`, `ForecastConfig`. The parser reads its defaults from `DEFAULTS = ForecastConfig()`, which is captured at import time, so a monkeypatched `ForecastConfig` cannot break argument parsing.
- The `build` command: `build [--sample N] [--seed S]`.
  - It prints the quality lines, the target report and the feature coverage, and writes `DATA_DIR/quality.json`.
  - It exits 1, printing the reason to stderr, when the drop share is above `max_drop_share` or when anything raises.
- From `forecast_fixtures`:
  - `history_areas() -> pl.DataFrame`
  - `history_aliases() -> pl.DataFrame`
  - `fake_load_rows(rows_and_end, quality=None)`, which returns a stand-in for `load_rows`.

Task 6 changes `build` to include infrastructure features. Tasks 5 and 11 add more commands.

- [ ] **Step 1: Add the fixture helpers**

Append to `tests/models/forecast/forecast_fixtures.py`:

```python
def history_areas():
    return pl.DataFrame(
        {"area_id": list(AREA_NAMES), "name_en": list(AREA_NAMES.values())},
        schema={"area_id": pl.Int64, "name_en": pl.Utf8},
    )


def history_aliases():
    from ingestion.normalize import match_key

    return pl.DataFrame(
        {
            "alias_key": [match_key(name) for name in AREA_NAMES.values()],
            "area_id": list(AREA_NAMES),
        },
        schema={"alias_key": pl.Utf8, "area_id": pl.Int64},
    )


def fake_load_rows(rows_and_end, quality=None):
    """A stand-in for models.forecast.rows.load_rows over prepared synthetic rows."""
    from models.forecast.rows import DataQuality, EXCLUDED_SCHEMA

    rows, data_end = rows_and_end
    if quality is None:
        quality = DataQuality(rows.height, {"bad_price": 0}, rows.height, pl.DataFrame(schema=EXCLUDED_SCHEMA))

    def load(settings, config):
        return rows, quality, data_end, history_areas(), history_aliases()

    return load
```

- [ ] **Step 2: Write the failing tests**

`tests/models/forecast/test_forecast_cli.py`:

```python
import json
from datetime import date

import polars as pl
from forecast_fixtures import fake_load_rows, prepared_history

from models.forecast import __main__ as cli
from models.forecast.rows import EXCLUDED_SCHEMA, DataQuality, excluded_summary

HISTORY = prepared_history(areas=2, buildings_per_area=2, start=date(2019, 1, 7))


def quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: None)


def test_build_prints_the_report_and_writes_quality_json(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    seen = {}

    def load(settings, config):
        seen.update(sample=config.sample_rows, seed=config.seed)
        return fake_load_rows(HISTORY)(settings, config)

    monkeypatch.setattr(cli, "load_rows", load)
    assert cli.main(["build", "--sample", "500", "--seed", "3"]) == 0
    assert seen == {"sample": 500, "seed": 3}
    out = capsys.readouterr().out
    assert "Dropped 0 of" in out
    assert "Target rows:" in out
    assert "Feature coverage" in out
    payload = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert payload["data_end"] == HISTORY[1].isoformat()
    assert payload["sample_rows"] == 500
    assert payload["targets"]["rows"] == HISTORY[0].height
    assert set(payload) >= {"quality", "excluded", "targets", "feature_coverage"}
    timings = json.loads((tmp_path / "stage_timings.json").read_text(encoding="utf-8"))
    assert set(timings["build"]) == {"load_rows", "targets", "features"}


def test_build_stops_when_too_many_rows_are_dropped(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    rows, _ = HISTORY
    quality = DataQuality(
        rows.height * 2, {"bad_price": rows.height}, rows.height, pl.DataFrame(schema=EXCLUDED_SCHEMA)
    )
    monkeypatch.setattr(cli, "load_rows", fake_load_rows(HISTORY, quality))
    assert cli.main(["build"]) == 1
    err = capsys.readouterr().err
    assert "Build stopped: 50.0% of rows were dropped, above the 30% limit" in err
    assert not (tmp_path / "quality.json").exists()


def test_build_reports_errors(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)

    def boom(settings, config):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(cli, "load_rows", boom)
    assert cli.main(["build"]) == 1
    assert "Build failed: RuntimeError: database unreachable" in capsys.readouterr().err


def test_excluded_summary_counts_by_segment():
    excluded = pl.DataFrame(
        {
            "area_id": [1, 1, 2],
            "sub_kind": ["flat", "flat", "villa"],
            "reg_type": ["off_plan", "off_plan", "ready"],
            "month": [date(2020, 1, 1)] * 3,
            "reason": ["outlier"] * 3,
        },
        schema=EXCLUDED_SCHEMA,
    )
    assert excluded_summary(excluded) == [
        {"area_id": 1, "sub_kind": "flat", "reg_type": "off_plan", "reason": "outlier", "count": 2},
        {"area_id": 2, "sub_kind": "villa", "reg_type": "ready", "reason": "outlier", "count": 1},
    ]


def test_build_runs_on_the_test_database(pg_test_db, monkeypatch, tmp_path, capsys):
    from pathlib import Path

    from ingestion.pipeline import run_pipeline

    run_pipeline(Path("tests/fixtures/price_sample.csv"), pg_test_db)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: pg_test_db)
    assert cli.main(["build", "--sample", "1000"]) == 0
    assert "Dropped" in capsys.readouterr().out
    payload = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert payload["quality"]["loaded"] <= 1000
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_cli.py`

Expected: FAIL with an import error for `models.forecast.__main__` / `excluded_summary`.

- [ ] **Step 4: Add excluded_summary to rows.py**

Append to `models/forecast/rows.py`:

```python
def excluded_summary(excluded: pl.DataFrame) -> list[dict]:
    """Excluded-row counts per area, kind, registration type and reason (for predict)."""
    keys = ["area_id", "sub_kind", "reg_type", "reason"]
    return excluded.group_by(keys).len(name="count").sort(keys, nulls_last=True).to_dicts()
```

- [ ] **Step 5: Write __main__.py**

`models/forecast/__main__.py`:

```python
"""Command line: python -m models.forecast build|infra-check|train|evaluate|predict."""

import argparse
import dataclasses
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

from ingestion.config import DbSettings
from models.forecast.config import DATA_DIR, ForecastConfig
from models.forecast.features import CORE_FEATURES, add_core_features, feature_coverage
from models.forecast.rows import excluded_summary, load_rows
from models.forecast.targets import build_targets


DEFAULTS = ForecastConfig()  # parser defaults, fixed at import (tests replace ForecastConfig)


def _settings() -> DbSettings:
    return DbSettings.from_env()


def write_json(name: str, payload: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


class Stages:
    """Wall-clock seconds per stage, merged into DATA_DIR/stage_timings.json by command."""

    def __init__(self, command: str):
        self.command = command
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = round(time.perf_counter() - started, 2)

    def save(self) -> None:
        path = DATA_DIR / "stage_timings.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        write_json("stage_timings.json", {**existing, self.command: self.seconds})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m models.forecast",
        description="Build, check, train, evaluate and query the price forecasts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="rows, targets and features plus a quality report")
    build.add_argument("--sample", type=int, help="screen a seeded random sample of N rows")
    build.add_argument("--seed", type=int, default=DEFAULTS.seed)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv()  # before any settings are read: DbSettings.from_env() needs POSTGRES_PORT
    handlers = {"build": _build}
    return handlers[args.command](args)


def _print(lines) -> None:
    for line in lines:
        print(line)


def _build(args: argparse.Namespace) -> int:
    config = dataclasses.replace(ForecastConfig(), sample_rows=args.sample, seed=args.seed)
    stages = Stages("build")
    try:
        with stages.stage("load_rows"):
            rows, quality, data_end, _, _ = load_rows(_settings(), config)
        _print(quality.lines())
        if quality.drop_share > config.max_drop_share:
            print(
                f"Build stopped: {quality.drop_share:.1%} of rows were dropped, above the "
                f"{config.max_drop_share:.0%} limit",
                file=sys.stderr,
            )
            return 1
        with stages.stage("targets"):
            frame, report = build_targets(rows, data_end)
        with stages.stage("features"):
            frame = add_core_features(frame)
        coverage = feature_coverage(frame, CORE_FEATURES)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print(report.lines())
    print("Feature coverage (non-null share):")
    _print(f"  {name:<28}{share:>7.1%}" for name, share in coverage.items())
    write_json(
        "quality.json",
        {
            "data_end": data_end.isoformat(),
            "sample_rows": config.sample_rows,
            "seed": config.seed,
            "quality": quality.to_dict(),
            "excluded": excluded_summary(quality.excluded),
            "targets": report.to_dict(),
            "feature_coverage": coverage,
        },
    )
    stages.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/models/forecast/test_forecast_cli.py`

Expected: every test passes. The database test needs the Docker stack; start it with `docker compose up -d --wait` if it skips.

- [ ] **Step 7: Lint, test, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/models/forecast
git add models/forecast/__main__.py models/forecast/rows.py tests/models/forecast/test_forecast_cli.py tests/models/forecast/forecast_fixtures.py
git commit -m "feat(forecast): build command with the drop guard, quality report and stage timings

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Real sample run (stop point 1)**

```bash
uv run python -m models.forecast build --sample 10000
uv run python -m models.forecast build
```

- **The sample run** is the data-quality check the spec asks for.
- **The full run** reports the real per-horizon counts, the last usable T and the timings.
  - Expected shape: the 3y horizon has very few or no usable rows (ruling 5).
  - In the sample, most rows are `no_base`, because building windows are sparse in a 3% sample (ruling 6).
- **Report:** put both outputs, verbatim, in your report file, together with `data/forecast/stage_timings.json`.
- **Then STOP** and report `DONE`. The controller shows the reports to the user before Task 5 starts.
- **If either run exits 1,** report `BLOCKED` with the output. Do not loosen the guard.

---

## Task 5: The sourced infrastructure table and `infra-check` (stop point 2)

**Files:**
- Create:
  - `models/forecast/infra.py`
  - `models/forecast/reference/infrastructure_projects.csv`
  - `tests/models/forecast/test_forecast_infra.py`
- Modify:
  - `models/forecast/__main__.py` (add `infra-check`)
  - `tests/models/forecast/forecast_fixtures.py` (add `project_record`, `write_projects`, `project_table`)
  - `tests/models/forecast/test_forecast_cli.py` (add the infra-check tests)

**Interfaces:**

*Consumes:*
- `config.INFRA_TYPES`
- `ingestion.config.DbSettings.connect()`
- From Task 4: the `__main__` helpers `_settings`, `_print`, `main`, `_parser`.

*Produces, from `models.forecast.infra`:*
- Constants:
  - `PROJECTS_PATH` (the committed CSV)
  - `INFRA_COLUMNS` (the 10 columns, in order)
  - `DATE_COLUMNS`
  - `MIN_PROJECTS = 15`
- `InfraTableError(ValueError)`
- `load_projects(path=PROJECTS_PATH) -> pl.DataFrame`
  - Text columns are stripped; empty text becomes null.
  - Each date column is parsed to `pl.Date`, with invalid text becoming null, and its original text is kept in `<column>_text`.
  - `area_ids` becomes `list[Int64]`; a non-integer entry becomes null.
  - Raises `InfraTableError` if the columns aren't exactly `INFRA_COLUMNS`.
- `validate_projects(projects, known_area_ids) -> list[str]`: one message per problem, formatted `"<project_id>: <problem>"`. An empty list means the table is valid.
- `fetch_reference(settings) -> tuple[dict[int, str], date]`: `dld.areas` ids mapped to their names, and the clean-data end date. It is read-only.
- `project_lines(projects, area_names) -> list[str]`: a readable listing.

*Produces, from `forecast_fixtures`:*
- `project_record(index, **overrides) -> dict[str, str]`
- `write_projects(directory, records) -> Path`
- `project_table(directory, count=15, **overrides) -> pl.DataFrame`, a valid table covering areas 1 and 2.

*Produces, as a command:* `infra-check`
- It prints the table with area names and says how many projects were announced after the data end (those affect no row).
- It exits 1 and prints the problems to stderr when validation fails.
- Tests monkeypatch the module globals `load_projects` and `fetch_reference`.

- [ ] **Step 1: Add the fixture helpers**

Append to `tests/models/forecast/forecast_fixtures.py`:

```python
def project_record(index, **overrides):
    from models.forecast.config import INFRA_TYPES

    record = {
        "project_id": f"P{index:02d}",
        "name": f"Project number {index}",
        "type": INFRA_TYPES[index % len(INFRA_TYPES)],
        "announced_date": "2016-01-01",
        "planned_completion_date": "2019-01-01",
        "actual_completion_date": "2019-06-01" if index % 2 else "",
        "affected_area_ids": "1;2",
        "source_url": f"https://example.org/projects/{index}",
        "source_accessed": "2026-09-16",
        "notes": "synthetic test row",
    }
    return {**record, **overrides}


def write_projects(directory, records):
    from models.forecast.infra import INFRA_COLUMNS

    path = directory / "projects.csv"
    pl.DataFrame(records, schema=dict.fromkeys(INFRA_COLUMNS, pl.Utf8)).write_csv(path)
    return path


def project_table(directory, count=15, **overrides):
    from models.forecast.infra import load_projects

    records = [project_record(index, **overrides) for index in range(count)]
    return load_projects(write_projects(directory, records))
```

- [ ] **Step 2: Write the failing validator tests**

`tests/models/forecast/test_forecast_infra.py`:

```python
from datetime import date

import polars as pl
import pytest
from forecast_fixtures import project_record, project_table, write_projects

from models.forecast.config import INFRA_TYPES
from models.forecast.infra import (
    INFRA_COLUMNS,
    MIN_PROJECTS,
    InfraTableError,
    load_projects,
    project_lines,
    validate_projects,
)

KNOWN = {1, 2}


def problems_for(tmp_path, **overrides):
    records = [project_record(index) for index in range(MIN_PROJECTS)]
    records[0] = project_record(0, **overrides)
    return validate_projects(load_projects(write_projects(tmp_path, records)), KNOWN)


def test_a_valid_table_has_no_problems(tmp_path):
    projects = project_table(tmp_path)
    assert validate_projects(projects, KNOWN) == []
    row = projects.row(1, named=True)
    assert row["announced_date"] == date(2016, 1, 1)
    assert row["actual_completion_date"] == date(2019, 6, 1)
    assert projects.row(0, named=True)["actual_completion_date"] is None
    assert row["area_ids"] == [1, 2]
    assert row["announced_date_text"] == "2016-01-01"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"source_url": ""}, "P00: source_url must be an https:// link"),
        ({"source_url": "http://example.org"}, "P00: source_url must be an https:// link"),
        (
            {"announced_date": "2020-01-01"},
            "P00: announced_date 2020-01-01 is after planned_completion_date 2019-01-01",
        ),
        (
            {"announced_date": "2019-01-01", "planned_completion_date": "2019-03-01",
             "actual_completion_date": "2018-12-31"},
            "P00: actual_completion_date 2018-12-31 is before announced_date 2019-01-01",
        ),
        ({"affected_area_ids": "1;999"}, "P00: area id 999 is not in dld.areas"),
        ({"affected_area_ids": "1;x"}, "P00: affected_area_ids must be integers separated by ';'"),
        ({"affected_area_ids": ""}, "P00: affected_area_ids is empty"),
        ({"type": "stadium"}, f"P00: type 'stadium' is not one of {', '.join(INFRA_TYPES)}"),
        ({"announced_date": "2016-13-01"}, "P00: announced_date '2016-13-01' is not a YYYY-MM-DD date"),
        ({"planned_completion_date": ""}, "P00: planned_completion_date is missing"),
        ({"source_accessed": ""}, "P00: source_accessed is missing"),
        ({"name": " "}, "P00: name is missing"),
        ({"notes": ""}, "P00: notes must explain the area mapping"),
        ({"project_id": "P01"}, "P01: project_id is used more than once"),
    ],
)  # fmt: skip
def test_each_problem_is_reported(tmp_path, overrides, expected):
    assert expected in problems_for(tmp_path, **overrides)


def test_a_short_table_is_rejected(tmp_path):
    problems = validate_projects(project_table(tmp_path, count=3), KNOWN)
    assert problems == [f"the table has 3 projects; at least {MIN_PROJECTS} are required"]


def test_wrong_columns_raise(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("project_id,name\nP1,x\n", encoding="utf-8")
    with pytest.raises(InfraTableError, match="columns must be"):
        load_projects(path)


def test_the_committed_table_is_well_formed():
    projects = load_projects()
    assert projects.height >= MIN_PROJECTS
    every_id = {value for ids in projects["area_ids"].to_list() for value in (ids or []) if value}
    assert validate_projects(projects, every_id) == []
    assert set(projects["type"].unique()) <= set(INFRA_TYPES)
    assert projects["notes"].null_count() == 0


def test_project_lines_name_the_areas(tmp_path):
    lines = project_lines(project_table(tmp_path, count=2), {1: "Dubai Marina", 2: "Al Barsha"})
    assert len(lines) == 2
    assert lines[0].startswith("P00")
    assert "Dubai Marina (1), Al Barsha (2)" in lines[0]
    assert "done 2019-06-01" in lines[1]
    assert project_lines(project_table(tmp_path, count=1, affected_area_ids="7"), {})[0].endswith(
        "areas: ? (7)"
    )
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_infra.py`

Expected: FAIL with an import error for `models.forecast.infra`.

- [ ] **Step 4: Write infra.py (table half)**

`models/forecast/infra.py`:

```python
"""The hand-curated, sourced infrastructure table: loading, validation and listing.

Spec: "Infrastructure table" in
docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md. A project counts only
from its announced_date; a completion only from its actual_completion_date.
planned_completion_date is the date stated when the project was announced.
"""

from datetime import date
from pathlib import Path

import polars as pl

from ingestion.config import DbSettings
from models.forecast.config import INFRA_TYPES

PROJECTS_PATH = Path(__file__).resolve().parent / "reference" / "infrastructure_projects.csv"
INFRA_COLUMNS = (
    "project_id", "name", "type", "announced_date", "planned_completion_date",
    "actual_completion_date", "affected_area_ids", "source_url", "source_accessed", "notes",
)  # fmt: skip
DATE_COLUMNS = (
    "announced_date", "planned_completion_date", "actual_completion_date", "source_accessed",
)  # fmt: skip
REQUIRED_DATES = ("announced_date", "planned_completion_date", "source_accessed")
MIN_PROJECTS = 15


class InfraTableError(ValueError):
    """The infrastructure CSV cannot be read as a table of projects."""


def load_projects(path: Path = PROJECTS_PATH) -> pl.DataFrame:
    raw = pl.read_csv(path, infer_schema=False)
    if tuple(raw.columns) != INFRA_COLUMNS:
        raise InfraTableError(
            f"{Path(path).name} columns must be {', '.join(INFRA_COLUMNS)}; "
            f"found {', '.join(raw.columns)}"
        )
    stripped = raw.with_columns(
        pl.when(pl.col(name).str.strip_chars() != "")
        .then(pl.col(name).str.strip_chars())
        .alias(name)
        for name in INFRA_COLUMNS
    )
    return stripped.with_columns(
        *(pl.col(name).alias(f"{name}_text") for name in DATE_COLUMNS),
        *(pl.col(name).str.to_date("%Y-%m-%d", strict=False).alias(name) for name in DATE_COLUMNS),
        pl.col("affected_area_ids")
        .str.split(";")
        .list.eval(pl.element().str.strip_chars().cast(pl.Int64, strict=False))
        .alias("area_ids"),
    )


def _date_problem(row: dict, name: str) -> str | None:
    text, value = row[f"{name}_text"], row[name]
    if text is None:
        return f"{name} is missing" if name in REQUIRED_DATES else None
    if value is None:
        return f"{name} {text!r} is not a YYYY-MM-DD date"
    return None


def _row_problems(row: dict, known_area_ids) -> list[str]:
    found = []
    if row["project_id"] is None:
        found.append("project_id is missing")
    if row["name"] is None:
        found.append("name is missing")
    if row["type"] not in INFRA_TYPES:
        found.append(f"type {row['type']!r} is not one of {', '.join(INFRA_TYPES)}")
    found += [problem for name in DATE_COLUMNS if (problem := _date_problem(row, name))]
    announced = row["announced_date"]
    planned = row["planned_completion_date"]
    actual = row["actual_completion_date"]
    if announced and planned and announced > planned:
        found.append(f"announced_date {announced} is after planned_completion_date {planned}")
    if announced and actual and actual < announced:
        found.append(f"actual_completion_date {actual} is before announced_date {announced}")
    if not (row["source_url"] or "").startswith("https://"):
        found.append("source_url must be an https:// link")
    if row["notes"] is None:
        found.append("notes must explain the area mapping")
    ids = row["area_ids"]
    if not ids:
        found.append("affected_area_ids is empty")
    elif any(value is None for value in ids):
        found.append("affected_area_ids must be integers separated by ';'")
    else:
        found += [f"area id {value} is not in dld.areas" for value in ids if value not in known_area_ids]
    return found


def validate_projects(projects: pl.DataFrame, known_area_ids) -> list[str]:
    problems = []
    if projects.height < MIN_PROJECTS:
        problems.append(
            f"the table has {projects.height} projects; at least {MIN_PROJECTS} are required"
        )
    duplicated = projects.filter(
        pl.col("project_id").is_not_null() & pl.col("project_id").is_duplicated()
    )
    problems += [
        f"{project_id}: project_id is used more than once"
        for project_id in sorted(set(duplicated["project_id"].to_list()))
    ]
    for row in projects.iter_rows(named=True):
        label = row["project_id"] or "(missing id)"
        problems += [f"{label}: {problem}" for problem in _row_problems(row, known_area_ids)]
    return problems


def fetch_reference(settings: DbSettings) -> tuple[dict[int, str], date]:
    """dld.areas names by id and the last clean sale date (read-only)."""
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT area_id, name_en FROM dld.areas ORDER BY area_id")
            names = dict(cur.fetchall())
            cur.execute(
                "SELECT max(instance_date) FROM dld.transactions WHERE exclusion_reason IS NULL"
            )
            (data_end,) = cur.fetchone()
    finally:
        conn.close()
    return names, data_end


def project_lines(projects: pl.DataFrame, area_names: dict[int, str]) -> list[str]:
    lines = []
    for row in projects.iter_rows(named=True):
        done = f"done {row['actual_completion_date']}" if row["actual_completion_date"] else "open"
        areas = ", ".join(
            f"{area_names.get(value, '?')} ({value})" for value in (row["area_ids"] or [])
        )
        lines.append(
            f"{row['project_id']}  {row['type']:<10} {row['announced_date']} -> "
            f"{row['planned_completion_date']} ({done})  {row['name']}  areas: {areas}"
        )
    return lines
```

`test_each_problem_is_reported` uses the case `{"project_id": "P01"}`: record 0 takes the id `P01`, which record 1 already has, so the table has a duplicate. That is why the case expects `"P01: project_id is used more than once"`.

- [ ] **Step 5: Add infra-check to the CLI, with tests**

Add these imports to `models/forecast/__main__.py`:

```python
from models.forecast.infra import fetch_reference, load_projects, project_lines, validate_projects
```

In `_parser()`, before `return parser`, add:

```python
    commands.add_parser("infra-check", help="validate and list the infrastructure table")
```

In `main()`, register `"infra-check": _infra_check` in `handlers`, and add:

```python
def _infra_check(args: argparse.Namespace) -> int:
    try:
        projects = load_projects()
        area_names, data_end = fetch_reference(_settings())
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Infrastructure check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print(project_lines(projects, area_names))
    late = projects.filter(pl.col("announced_date") > data_end).height
    print(
        f"{projects.height} projects; {late} announced after the data end ({data_end}) "
        "and so affect no row"
    )
    problems = validate_projects(projects, set(area_names))
    if problems:
        print("Infrastructure table problems:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("Infrastructure table OK")
    return 0
```

It also needs `import polars as pl` at the top of `__main__.py`.

Append to `tests/models/forecast/test_forecast_cli.py`:

```python
def test_infra_check_lists_a_valid_table(monkeypatch, tmp_path, capsys):
    from forecast_fixtures import project_table

    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_projects", lambda: project_table(tmp_path))
    monkeypatch.setattr(
        cli, "fetch_reference", lambda settings: ({1: "Dubai Marina", 2: "Al Barsha"}, date(2017, 1, 1))
    )
    assert cli.main(["infra-check"]) == 0
    out = capsys.readouterr().out
    assert "P00" in out and "Dubai Marina (1)" in out
    assert "15 projects; 0 announced after the data end (2017-01-01)" in out
    assert "Infrastructure table OK" in out


def test_infra_check_fails_on_problems(monkeypatch, tmp_path, capsys):
    from forecast_fixtures import project_table

    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_projects", lambda: project_table(tmp_path, affected_area_ids="5"))
    monkeypatch.setattr(cli, "fetch_reference", lambda settings: ({1: "A"}, date(2015, 1, 1)))
    assert cli.main(["infra-check"]) == 1
    captured = capsys.readouterr()
    assert "15 announced after the data end" in captured.out
    assert "P00: area id 5 is not in dld.areas" in captured.err


def test_infra_check_runs_on_the_test_database(pg_test_db, monkeypatch, tmp_path, capsys):
    from pathlib import Path

    from forecast_fixtures import project_record, write_projects

    from ingestion.pipeline import run_pipeline
    from models.forecast.infra import load_projects

    run_pipeline(Path("tests/fixtures/price_sample.csv"), pg_test_db)
    conn = pg_test_db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT area_id FROM dld.areas ORDER BY area_id LIMIT 2")
            ids = ";".join(str(row[0]) for row in cur.fetchall())
    finally:
        conn.close()
    path = write_projects(tmp_path, [project_record(i, affected_area_ids=ids) for i in range(15)])
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "_settings", lambda: pg_test_db)
    monkeypatch.setattr(cli, "load_projects", lambda: load_projects(path))
    assert cli.main(["infra-check"]) == 0
    assert "Infrastructure table OK" in capsys.readouterr().out
```

- [ ] **Step 6: Research and write the real table**

Create `models/forecast/reference/infrastructure_projects.csv`:
- The header is exactly `project_id,name,type,announced_date,planned_completion_date,actual_completion_date,affected_area_ids,source_url,source_accessed,notes`.
- It needs at least 15 rows, and 18–25 is better.
- Fields containing commas are quoted.

**Listing the DLD areas** (read-only; the database is on port 5433, never 5432):

```bash
uv run python -c "from dotenv import load_dotenv; load_dotenv(); from ingestion.config import DbSettings; c = DbSettings.from_env().connect(); cur = c.cursor(); cur.execute('SELECT a.area_id, a.name_en, count(t.*) FROM dld.areas a LEFT JOIN dld.transactions t USING (area_id) GROUP BY 1, 2 ORDER BY 3 DESC'); [print(*r, sep=' | ') for r in cur.fetchall()]; c.close()"
```

**Research rules.** Use WebSearch and WebFetch. They are non-negotiable, because every number in the README traces back to this table.
- **Sources:** each row's `source_url` is an `https://` page you actually opened, and it states the dates you record.
  - Prefer official sources: RTA (`rta.ae`), Dubai Media Office (`mediaoffice.ae`), WAM (`wam.ae`), and operator or developer press releases.
  - A major UAE newspaper (Gulf News, Khaleej Times, The National) is acceptable when no official page states the dates.
  - Never use Wikipedia or a blog as the source.
- **Dates:**
  - `announced_date` is the first public announcement or contract award.
  - `planned_completion_date` is the completion date stated **at announcement**.
  - `actual_completion_date` is the opening date. Leave it empty if the project isn't open.
  - If a source gives only a month, use the first of that month and write `month precision` in `notes`.
  - If only a year is known for a required date, **leave the project out**. Never guess.
- **`source_accessed`:** today's date, `2026-09-16`.
- **`affected_area_ids`:** the DLD area ids whose names clearly contain or adjoin the project. Explain the mapping in `notes` (for example, `metro stations at Jabal Ali First and Dubai Investment Park First`). Use only ids printed by the query above.
- **`type`:** exactly one of `metro_rail`, `mall`, `school`, `park`, `mixed_use`, `airport`.
- **Project mix:** most projects should have been announced **before 2023-03-17**, so they affect the data. A few later projects (for example, the Blue Line) are fine; they affect no row.

**Candidates to verify.** Include only those whose dates you can source:

| Type | Candidates |
|---|---|
| Metro and rail | Route 2020 (Dubai Metro Red Line extension to Expo); Dubai Tram (Al Sufouh); Dubai Metro Blue Line; Etihad Rail passenger service |
| Mixed use and districts | Expo 2020 site / Expo City Dubai; Dubai Creek Harbour; Bluewaters Island and Ain Dubai; Dubai Water Canal; Dubai Harbour; Dubai Design District (d3) |
| Malls | Dubai Hills Mall; City Walk; Nakheel Mall; Cityland Mall; Al Khail Avenue; Mall of the Emirates expansion |
| Parks | Dubai Safari Park; Quranic Park; Dubai Frame (Zabeel Park); Dubai Parks and Resorts |
| Airport | Al Maktoum International Airport expansion (DWC) |

Then run the validator against the real areas:

```bash
uv run python -m models.forecast infra-check
```

It must exit 0.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q -W error tests/models/forecast/test_forecast_infra.py tests/models/forecast/test_forecast_cli.py`

Expected: every test passes, including `test_the_committed_table_is_well_formed`.

- [ ] **Step 8: Lint, test, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/models/forecast
git add models/forecast/infra.py models/forecast/reference/infrastructure_projects.csv models/forecast/__main__.py tests/models/forecast/test_forecast_infra.py tests/models/forecast/test_forecast_cli.py tests/models/forecast/forecast_fixtures.py
git commit -m "feat(forecast): sourced infrastructure table, validator and infra-check

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 9: Stop point 2**

Put the full `infra-check` output in your report, verbatim. For each project, add one line stating which sentence on the source page gives each date.

**Then STOP** and report `DONE`. The controller shows the table to the user before Task 6 starts.

---

## Task 6: As-of infrastructure features and the full dataset

**Files:**
- Modify:
  - `models/forecast/infra.py` (add `add_infra_features`)
  - `models/forecast/features.py` (add `build_dataset`)
  - `models/forecast/__main__.py` (`build` validates the table and uses `build_dataset`)
  - `tests/models/forecast/test_forecast_infra.py`
  - `tests/models/forecast/test_forecast_features.py`
  - `tests/models/forecast/test_forecast_cli.py`

**Interfaces:**

*Consumes:*
- From Task 5: `load_projects`, `validate_projects`, `project_table`.
- From Tasks 2–3: `build_targets`, `add_core_features`, `to_matrix`, `fit_categories`, `feature_coverage`.

*Produces:*
- `infra.add_infra_features(frame, projects) -> pl.DataFrame`. For each row, as of its `instance_date` T and its `area_id`:
  - `infra_active_<type>`: the count of projects affecting the area with `announced_date <= T` and not completed by T (`actual_completion_date` null or after T).
  - `infra_months_to_next`: over those active projects, the minimum of `max(planned − T, 0) / 30.4375` days. It is null when none is active.
  - `infra_completed_24m`: 1.0 if a project affecting the area has `T − 730 d < actual_completion_date <= T`, else 0.0.
  - `infra_mix_<type>`: `infra_active_<type>` divided by all active projects, or 0.0 when none is active.
  - Counts are Float64. The output keeps `row_id` order.
- `features.build_dataset(rows, data_end, projects) -> tuple[pl.DataFrame, TargetReport]`: runs `build_targets`, then `add_core_features`, then `add_infra_features`.
- The `build` command:
  - validates the table against the area ids returned by `load_rows`, exiting 1 with the problems if it is invalid;
  - then uses `build_dataset`;
  - reports coverage over the full `FEATURES`;
  - records the stages `load_rows` and `dataset`.

- [ ] **Step 1: Write the failing infrastructure-feature tests**

Append to `tests/models/forecast/test_forecast_infra.py`:

```python
from datetime import timedelta  # noqa: E402 — keep with the tests that use it

from models.forecast.infra import add_infra_features  # noqa: E402

ANNOUNCED = date(2018, 1, 1)
PLANNED = date(2020, 1, 1)
ACTUAL = date(2020, 6, 1)


def rows_on(days_and_areas):
    return pl.DataFrame(
        {
            "row_id": list(range(len(days_and_areas))),
            "instance_date": [day for day, _ in days_and_areas],
            "area_id": [area for _, area in days_and_areas],
        },
        schema={"row_id": pl.Int64, "instance_date": pl.Date, "area_id": pl.Int64},
    )


def one_metro(tmp_path, **overrides):
    record = project_record(
        0,
        type="metro_rail",
        announced_date=str(ANNOUNCED),
        planned_completion_date=str(PLANNED),
        actual_completion_date=str(ACTUAL),
        affected_area_ids="1",
        **overrides,
    )
    return load_projects(write_projects(tmp_path, [record]))


def test_a_project_counts_only_from_its_announcement(tmp_path):
    frame = rows_on([(ANNOUNCED - timedelta(days=1), 1), (ANNOUNCED, 1), (date(2020, 3, 1), 1)])
    out = add_infra_features(frame, one_metro(tmp_path))
    assert out["infra_active_metro_rail"].to_list() == [0.0, 1.0, 1.0]
    assert out["infra_mix_metro_rail"].to_list() == [0.0, 1.0, 1.0]
    months = out["infra_months_to_next"].to_list()
    assert months[0] is None
    assert months[1] == pytest.approx((PLANNED - ANNOUNCED).days / 30.4375)
    assert months[2] == 0.0  # planned date passed but not yet open: delayed, not negative


def test_a_completion_counts_only_from_its_opening(tmp_path):
    days = [
        ACTUAL - timedelta(days=1),
        ACTUAL,
        ACTUAL + timedelta(days=729),
        ACTUAL + timedelta(days=730),
    ]
    out = add_infra_features(rows_on([(day, 1) for day in days]), one_metro(tmp_path))
    assert out["infra_active_metro_rail"].to_list() == [1.0, 0.0, 0.0, 0.0]
    assert out["infra_completed_24m"].to_list() == [0.0, 1.0, 1.0, 0.0]
    assert out["infra_months_to_next"].to_list()[1:] == [None, None, None]


def test_unaffected_areas_and_order(tmp_path):
    frame = rows_on([(date(2019, 1, 1), 2), (date(2019, 1, 1), 1), (date(2014, 1, 1), 1)])
    out = add_infra_features(frame, one_metro(tmp_path))
    assert out["row_id"].to_list() == [0, 1, 2]
    assert out["infra_active_metro_rail"].to_list() == [0.0, 1.0, 0.0]
    for kind in INFRA_TYPES:
        assert out[f"infra_mix_{kind}"].null_count() == 0
    assert out["infra_completed_24m"].to_list() == [0.0, 0.0, 0.0]


def test_the_type_mix_splits_active_projects(tmp_path):
    records = [
        project_record(0, type="metro_rail", affected_area_ids="1", actual_completion_date=""),
        project_record(1, type="mall", affected_area_ids="1", actual_completion_date=""),
        project_record(2, type="mall", affected_area_ids="1", actual_completion_date=""),
    ]
    projects = load_projects(write_projects(tmp_path, records))
    out = add_infra_features(rows_on([(date(2017, 1, 1), 1)]), projects)
    assert out["infra_active_mall"][0] == 2.0
    assert out["infra_mix_mall"][0] == pytest.approx(2 / 3)
    assert out["infra_mix_metro_rail"][0] == pytest.approx(1 / 3)
    assert out["infra_mix_park"][0] == 0.0
```

Put the two new imports at the top of the file with the other imports rather than using `noqa` (ruff's isort rule will sort them). The inline form above only marks what this step adds.

- [ ] **Step 2: Write the failing dataset tests**

Append to `tests/models/forecast/test_forecast_features.py`:

```python
def test_build_dataset_matches_the_allowlist(tmp_path):
    from forecast_fixtures import project_table

    from models.forecast.config import FEATURES, FORBIDDEN_FEATURES
    from models.forecast.features import build_dataset

    frame, report = build_dataset(ROWS, DATA_END, project_table(tmp_path))
    assert report.rows == ROWS.height
    matrix = to_matrix(frame, fit_categories(frame, ForecastConfig()))
    assert list(matrix.columns) == list(FEATURES)
    assert not set(matrix.columns) & set(FORBIDDEN_FEATURES)
    assert frame["infra_active_metro_rail"].sum() > 0  # project_table covers areas 1 and 2


def test_full_features_never_look_at_sales_on_or_after_t(tmp_path):
    from forecast_fixtures import project_table

    from models.forecast.config import FEATURES
    from models.forecast.features import build_dataset

    projects = project_table(tmp_path)
    cutoff = date(2019, 6, 1)
    later = pl.col("instance_date") >= cutoff
    changed = ROWS.with_columns(
        pl.when(later).then(pl.col("ppsm") * 3).otherwise(pl.col("ppsm")).alias("ppsm")
    ).filter(~later | (pl.col("row_id") % 3 == 0))
    columns = ["transaction_id", *FEATURES]
    before = build_dataset(ROWS, DATA_END, projects)[0].filter(~later).select(columns)
    after = build_dataset(changed, DATA_END, projects)[0].filter(~later).select(columns)
    assert before.sort("transaction_id").equals(after.sort("transaction_id"))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_infra.py tests/models/forecast/test_forecast_features.py`

Expected: FAIL with import errors for `add_infra_features` / `build_dataset`.

- [ ] **Step 4: Implement**

Append to `models/forecast/infra.py`:

```python
MONTH_DAYS = 30.4375
COMPLETED_WINDOW_DAYS = 730


def add_infra_features(frame: pl.DataFrame, projects: pl.DataFrame) -> pl.DataFrame:
    """As-of infrastructure features for each row's area at its sale date."""
    links = (
        projects.select(
            "type",
            "announced_date",
            "planned_completion_date",
            "actual_completion_date",
            pl.col("area_ids").alias("area_id"),
        )
        .explode("area_id")
        .drop_nulls("area_id")
    )
    day = pl.col("instance_date")
    opened = pl.col("actual_completion_date")
    completed = opened.is_not_null() & (opened <= day)
    active = (pl.col("announced_date") <= day) & ~completed
    months = (
        pl.max_horizontal((pl.col("planned_completion_date") - day).dt.total_days(), pl.lit(0))
        / MONTH_DAYS
    )
    recent = completed & (opened > day.dt.offset_by(f"-{COMPLETED_WINDOW_DAYS}d"))
    stats = (
        frame.select("area_id", "instance_date")
        .unique()
        .join(links, on="area_id", how="inner")
        .group_by("area_id", "instance_date")
        .agg(
            *(
                (active & (pl.col("type") == kind)).sum().cast(pl.Float64).alias(f"infra_active_{kind}")
                for kind in INFRA_TYPES
            ),
            pl.when(active).then(months).min().alias("infra_months_to_next"),
            recent.any().cast(pl.Float64).alias("infra_completed_24m"),
        )
    )
    counts = [f"infra_active_{kind}" for kind in INFRA_TYPES]
    total = pl.sum_horizontal(counts)
    out = frame.join(stats, on=["area_id", "instance_date"], how="left").with_columns(
        pl.col(*counts, "infra_completed_24m").fill_null(0.0)
    )
    return out.with_columns(
        pl.when(total > 0)
        .then(pl.col(f"infra_active_{kind}") / total)
        .otherwise(0.0)
        .alias(f"infra_mix_{kind}")
        for kind in INFRA_TYPES
    ).sort("row_id")
```

Add to `models/forecast/features.py`:
- the imports `from datetime import date`, `from models.forecast.infra import add_infra_features` and `from models.forecast.targets import TargetReport, build_targets`;
- this function:

```python
def build_dataset(
    rows: pl.DataFrame, data_end: date, projects: pl.DataFrame
) -> tuple[pl.DataFrame, TargetReport]:
    """Targets plus every model feature, as of each sale."""
    frame, report = build_targets(rows, data_end)
    return add_infra_features(add_core_features(frame), projects), report
```

In `models/forecast/__main__.py`:
- change the features import to `from models.forecast.features import build_dataset, feature_coverage`;
- import `FEATURES` from config;
- replace the body of `_build` between the drop guard and the `except` with:

```python
        projects = load_projects()
        problems = validate_projects(projects, set(areas["area_id"].to_list()))
        if problems:
            print("Build stopped: the infrastructure table has problems:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        with stages.stage("dataset"):
            frame, report = build_dataset(rows, data_end, projects)
        coverage = feature_coverage(frame, FEATURES)
```

- Change the unpacking to `rows, quality, data_end, areas, _ = load_rows(_settings(), config)`.
- Remove the now-unused `build_targets`, `add_core_features` and `CORE_FEATURES` imports.

- [ ] **Step 5: Update the CLI tests**

In `tests/models/forecast/test_forecast_cli.py`:
- In `quiet()`, add this line, so build tests get a valid table for areas 1 and 2:

```python
    monkeypatch.setattr(cli, "load_projects", lambda: project_table(tmp_path))
```

  Import `project_table` at the top of the file with the other `forecast_fixtures` imports. Delete the local imports of `project_table` in the infra-check tests, which now come from the top-level import. The infra-check tests that set their own `load_projects` still do so after calling `quiet()`, so theirs wins.
- In `test_build_prints_the_report_and_writes_quality_json`, change the timings assertion to `assert set(timings["build"]) == {"load_rows", "dataset"}`, and add `assert "infra_active_metro_rail" in payload["feature_coverage"]`.
- Add:

```python
def test_build_stops_on_an_invalid_infrastructure_table(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_rows", fake_load_rows(HISTORY))
    monkeypatch.setattr(cli, "load_projects", lambda: project_table(tmp_path, affected_area_ids="9"))
    assert cli.main(["build"]) == 1
    err = capsys.readouterr().err
    assert "Build stopped: the infrastructure table has problems" in err
    assert "P00: area id 9 is not in dld.areas" in err
```

- In `test_build_runs_on_the_test_database`, after `run_pipeline(...)`:
  - read two area ids exactly as `test_infra_check_runs_on_the_test_database` does;
  - write a 15-row table with them through `write_projects`;
  - monkeypatch `cli.load_projects` to load it.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/models/forecast`

Expected: every test passes.

If `pl.col(*counts, "infra_completed_24m")` is rejected, use `pl.col([*counts, "infra_completed_24m"])`.

- [ ] **Step 7: Lint, commit, real build**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add models/forecast/infra.py models/forecast/features.py models/forecast/__main__.py tests/models/forecast/test_forecast_infra.py tests/models/forecast/test_forecast_features.py tests/models/forecast/test_forecast_cli.py
git commit -m "feat(forecast): as-of infrastructure features and the full dataset build

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
uv run python -m models.forecast build
```

Put the coverage lines for the `infra_*` features in your report. At least one `infra_active_*` must have a non-zero mean on the real data. If every one is zero, the area mapping or the dates are wrong: report `DONE_WITH_CONCERNS`.

---

## Task 7: Walk-forward folds and the baselines

**Files:**
- Create:
  - `models/forecast/folds.py`
  - `models/forecast/baselines.py`
  - `tests/models/forecast/test_forecast_folds.py`

**Interfaces:**

*Consumes:*
- `config.Horizon`, `HORIZON_SPECS`, `ForecastConfig`
- Frames with `instance_date` and `growth_<h>`; the baselines also need `area_mom_<len>` and `city_mom_<len>`.

*Produces, from `models.forecast.folds`:*
- `MONTH_DAYS`
- `Fold(index, cutoff, end, role)`, frozen, with `.to_dict()`. `role` is `"score"`, `"tune"` or `"test"`, and `end` is exclusive.
- `month_start(day)`, `add_months(first_of_month, months)`
- `usable_rows(frame, horizon)`
- `plan_folds(frame, horizon, config) -> list[Fold]`
- `fold_split(frame, horizon, fold) -> tuple[train, val]`
- `insufficiency(folds, test_rows, config) -> str | None`

*Produces, from `models.forecast.baselines`:*
- `BASELINES = ("no_change", "area_trend")`
- `baseline_growth(frame, horizon) -> dict[str, np.ndarray]`

- [ ] **Step 1: Write the failing tests**

`tests/models/forecast/test_forecast_folds.py`:

```python
import dataclasses
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from forecast_fixtures import prepared_history

from models.forecast.baselines import BASELINES, baseline_growth
from models.forecast.config import HORIZON_SPECS, ForecastConfig
from models.forecast.folds import (
    Fold,
    add_months,
    fold_split,
    insufficiency,
    month_start,
    plan_folds,
)
from models.forecast.targets import build_targets

THREE_M = HORIZON_SPECS["3m"]


def growth_frame(days_and_growth):
    return pl.DataFrame(
        {
            "instance_date": [day for day, _ in days_and_growth],
            "growth_3m": [value for _, value in days_and_growth],
            "growth_1y": [None] * len(days_and_growth),
            "growth_3y": [None] * len(days_and_growth),
        },
        schema={
            "instance_date": pl.Date,
            "growth_3m": pl.Float64,
            "growth_1y": pl.Float64,
            "growth_3y": pl.Float64,
        },
    )


def test_month_helpers():
    assert month_start(date(2017, 4, 27)) == date(2017, 4, 1)
    assert add_months(date(2017, 11, 1), 3) == date(2018, 2, 1)
    assert add_months(date(2017, 5, 1), -5) == date(2016, 12, 1)
    with pytest.raises(ValueError):
        add_months(date(2017, 5, 2), 1)


def test_fold_plan_by_hand():
    # first T 2015-01-10 + 731 d + 107 d = 2017-04-27, so the first cutoff is 2017-05-01
    frame = growth_frame([(date(2015, 1, 10), 0.1), (date(2018, 1, 10), 0.2), (date(2016, 1, 1), None)])
    folds = plan_folds(frame, THREE_M, ForecastConfig())
    assert [fold.cutoff for fold in folds] == [date(2017, 5, 1), date(2017, 8, 1), date(2017, 11, 1)]
    assert [fold.end for fold in folds] == [date(2017, 8, 1), date(2017, 11, 1), date(2018, 2, 1)]
    assert [fold.role for fold in folds] == ["tune", "tune", "test"]
    roles = [fold.role for fold in plan_folds(frame, THREE_M, ForecastConfig(tune_folds=1))]
    assert roles == ["score", "tune", "test"]
    assert folds[0].to_dict() == {
        "index": 0, "cutoff": "2017-05-01", "end": "2017-08-01", "role": "tune",
    }  # fmt: skip
    assert plan_folds(growth_frame([(date(2015, 1, 10), None)]), THREE_M, ForecastConfig()) == []


def test_split_respects_the_target_window_guard():
    fold = Fold(0, date(2017, 5, 1), date(2017, 8, 1), "test")
    last_ok = date(2017, 5, 1) - timedelta(days=108)  # its window ends 2017-04-30
    too_late = date(2017, 5, 1) - timedelta(days=107)  # its window ends on the cutoff
    frame = growth_frame(
        [(last_ok, 0.1), (too_late, 0.1), (date(2017, 5, 1), 0.2), (date(2017, 7, 31), 0.3),
         (date(2017, 8, 1), 0.4), (date(2017, 6, 1), None)]
    )  # fmt: skip
    train, val = fold_split(frame, THREE_M, fold)
    assert train["instance_date"].to_list() == [last_ok]
    assert val["instance_date"].to_list() == [date(2017, 5, 1), date(2017, 7, 31)]


def test_every_history_fold_is_leak_free():
    rows, data_end = prepared_history()
    frame, _ = build_targets(rows, data_end)
    for horizon in HORIZON_SPECS.values():
        folds = plan_folds(frame, horizon, ForecastConfig())
        for fold in folds:
            train, val = fold_split(frame, horizon, fold)
            ends = train["instance_date"].to_list()
            assert all(day + timedelta(days=horizon.end_days) < fold.cutoff for day in ends)
            assert val["instance_date"].is_between(fold.cutoff, fold.end, closed="left").all()
            assert val[f"growth_{horizon.name}"].null_count() == 0
        cutoffs = [fold.cutoff for fold in folds]
        assert cutoffs == sorted(cutoffs)
        assert all(add_months(a, horizon.step_months) == b for a, b in zip(cutoffs, cutoffs[1:]))
    assert len(plan_folds(frame, HORIZON_SPECS["3m"], ForecastConfig())) > 10


def test_insufficiency_reasons():
    config = ForecastConfig()
    fold = Fold(0, date(2020, 1, 1), date(2020, 4, 1), "test")
    assert insufficiency([fold], 5_000, config) == "only 1 walk-forward folds (needs 2)"
    assert insufficiency([fold, fold], 999, config) == "only 999 test rows (needs 1,000)"
    assert insufficiency([fold, fold], 1_000, config) is None
    assert insufficiency([], 0, dataclasses.replace(config, min_folds=0, min_test_rows=0)) is None


def test_baselines_fall_back_from_area_to_city_to_zero():
    frame = pl.DataFrame(
        {
            "area_mom_3m": [0.1, None, None],
            "city_mom_3m": [0.2, 0.3, None],
            "area_mom_12m": [0.5, 0.5, 0.5],
            "city_mom_12m": [0.0, 0.0, 0.0],
        },
        schema=dict.fromkeys(("area_mom_3m", "city_mom_3m", "area_mom_12m", "city_mom_12m"), pl.Float64),
    )
    three = baseline_growth(frame, THREE_M)
    assert list(three) == list(BASELINES)
    assert three["no_change"].tolist() == [0.0, 0.0, 0.0]
    assert three["area_trend"].tolist() == [0.1, 0.3, 0.0]
    year = baseline_growth(frame, HORIZON_SPECS["1y"])
    assert year["area_trend"].tolist() == [0.5, 0.5, 0.5]
    assert isinstance(year["area_trend"], np.ndarray)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_folds.py`

Expected: FAIL with an import error for `models.forecast.folds`.

- [ ] **Step 3: Write folds.py and baselines.py**

`models/forecast/folds.py`:

```python
"""Walk-forward folds per horizon with the target-window leakage guard.

A training row's target window must end before its fold's cutoff: T + end_days < cutoff.
The last fold is the test period; the (up to) tune_folds folds before it tune the model.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from models.forecast.config import ForecastConfig, Horizon

MONTH_DAYS = 30.4375


@dataclass(frozen=True)
class Fold:
    index: int
    cutoff: date
    end: date  # exclusive: the next cutoff
    role: str  # "score", "tune" or "test"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "cutoff": self.cutoff.isoformat(),
            "end": self.end.isoformat(),
            "role": self.role,
        }


def month_start(day: date) -> date:
    return day.replace(day=1)


def add_months(first_of_month: date, months: int) -> date:
    if first_of_month.day != 1:
        raise ValueError(f"add_months needs the first of a month, got {first_of_month}")
    total = first_of_month.year * 12 + first_of_month.month - 1 + months
    return date(total // 12, total % 12 + 1, 1)


def usable_rows(frame: pl.DataFrame, horizon: Horizon) -> pl.DataFrame:
    return frame.filter(pl.col(f"growth_{horizon.name}").is_not_null())


def plan_folds(frame: pl.DataFrame, horizon: Horizon, config: ForecastConfig) -> list[Fold]:
    usable = usable_rows(frame, horizon)
    if usable.height == 0:
        return []
    first_t = usable["instance_date"].min()
    last_t = usable["instance_date"].max()
    history = round(config.min_train_months * MONTH_DAYS)
    anchor = first_t + timedelta(days=history + horizon.end_days)
    cutoff = add_months(month_start(anchor), 1)
    cutoffs = []
    while cutoff <= last_t:
        cutoffs.append(cutoff)
        cutoff = add_months(cutoff, horizon.step_months)
    last = len(cutoffs) - 1
    folds = []
    for index, start in enumerate(cutoffs):
        if index == last:
            role = "test"
        elif index >= last - config.tune_folds:
            role = "tune"
        else:
            role = "score"
        folds.append(Fold(index, start, add_months(start, horizon.step_months), role))
    return folds


def fold_split(
    frame: pl.DataFrame, horizon: Horizon, fold: Fold
) -> tuple[pl.DataFrame, pl.DataFrame]:
    usable = usable_rows(frame, horizon)
    window_end = pl.col("instance_date").dt.offset_by(f"{horizon.end_days}d")
    train = usable.filter(window_end < fold.cutoff)
    val = usable.filter(pl.col("instance_date").is_between(fold.cutoff, fold.end, closed="left"))
    return train, val


def insufficiency(folds: list[Fold], test_rows: int, config: ForecastConfig) -> str | None:
    """Why a horizon cannot be trained and gated, or None when it can."""
    if len(folds) < config.min_folds:
        return f"only {len(folds)} walk-forward folds (needs {config.min_folds})"
    if test_rows < config.min_test_rows:
        return f"only {test_rows:,} test rows (needs {config.min_test_rows:,})"
    return None
```

`models/forecast/baselines.py`:

```python
"""Baseline growth forecasts, computed on exactly the rows the model is scored on."""

import numpy as np
import polars as pl

from models.forecast.config import Horizon

BASELINES = ("no_change", "area_trend")


def baseline_growth(frame: pl.DataFrame, horizon: Horizon) -> dict[str, np.ndarray]:
    """no_change = 0; area_trend = the area's trailing ln change, else the city's, else 0."""
    trend = frame.select(
        pl.coalesce(
            pl.col(f"area_mom_{horizon.momentum}").fill_nan(None),
            pl.col(f"city_mom_{horizon.momentum}").fill_nan(None),
            pl.lit(0.0),
        ).alias("trend")
    )["trend"]
    return {
        "no_change": np.zeros(frame.height),
        "area_trend": trend.to_numpy().astype(float),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/models/forecast/test_forecast_folds.py`

Expected: every test passes.

- [ ] **Step 5: Lint, test, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/models/forecast
git add models/forecast/folds.py models/forecast/baselines.py tests/models/forecast/test_forecast_folds.py
git commit -m "feat(forecast): walk-forward folds with the target-window guard, and baselines

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: Conformal intervals, confidence labels, evaluation and the gate

**Files:**
- Create:
  - `models/forecast/intervals.py`
  - `models/forecast/evaluate.py`
  - `tests/models/forecast/test_forecast_evaluate.py`

**Interfaces:**

*Consumes:*
- `models.price.evaluate.conformal_quantile(abs_errors, alpha) -> float` (inf when there are too few rows)
- `config.ForecastConfig`, `Horizon`, `HORIZON_SPECS`
- `baselines.BASELINES`

*Produces, from `models.forecast.intervals`:*
- Constants: `POOLED = "_pooled"`, `LOW_MESSAGE`, `HIGH_MIN_SALES = 10`, `HIGH_MAX_DAYS = 90`, `MEDIUM_MIN_SALES = 5`, `MEDIUM_MAX_DAYS = 180`.
- `interval_segments(frame) -> pl.Series`, with values `"{reg_type}_{villa|unit}"`.
- `fit_intervals(segments, abs_errors, config) -> dict[str, float]`: log-growth half-widths per segment plus `POOLED`. A segment with fewer than `min_conformal_rows` rows gets the pooled value. Raises `ValueError` when even the pooled value is infinite.
- `half_width(intervals, segment) -> float`
- `coverage_by_segment(segments, predicted, actual, intervals) -> dict[str, float]`: includes `"all"`.
- `confidence(base_level, base_n, comparable_days, area_rows, min_area_rows) -> str`: `"HIGH"`, `"MEDIUM"` or `"LOW"`.
- `low_message(point, low, high) -> str`

*Produces, from `models.forecast.evaluate`:*
- Constants: `MODEL = "model"`, `FLAG_MAPE = 0.25`, `SEGMENTS`, `PRIMARY_SEGMENTS`, `SEGMENT_TEXT`.
- `ape(predicted_growth, actual_growth) -> np.ndarray`
- `top_areas(train, count) -> list[int]`
- `segment_masks(frame, top) -> dict[str, np.ndarray]`
- `segment_table(frame, predictions, actual, top) -> pl.DataFrame`, with columns `segment, model, rows, mape, median_ape, flagged`.
- `bootstrap_mape_upper(errors, n_resamples, seed) -> float`
- `GateResult(passed, reasons: tuple[str, ...], checks: dict[str, float])`
- `table_mape(table, segment, model) -> float`
- `gate(table, horizon, model_upper) -> GateResult`
- `format_table(name, table) -> list[str]`

- [ ] **Step 1: Write the failing tests**

`tests/models/forecast/test_forecast_evaluate.py`:

```python
import dataclasses
import math
import re

import numpy as np
import polars as pl
import pytest

from models.forecast.config import HORIZON_SPECS, ForecastConfig
from models.forecast.evaluate import (
    FLAG_MAPE,
    MODEL,
    PRIMARY_SEGMENTS,
    SEGMENTS,
    ape,
    bootstrap_mape_upper,
    format_table,
    gate,
    segment_masks,
    segment_table,
    table_mape,
    top_areas,
)
from models.forecast.intervals import (
    LOW_MESSAGE,
    POOLED,
    confidence,
    coverage_by_segment,
    fit_intervals,
    half_width,
    interval_segments,
    low_message,
)

CONFIG = ForecastConfig()
THREE_M = HORIZON_SPECS["3m"]


def test_interval_segments():
    frame = pl.DataFrame({"reg_type": ["ready", "off_plan"], "sub_kind": ["villa", "flat"]})
    assert interval_segments(frame).to_list() == ["ready_villa", "off_plan_unit"]


def test_conformal_coverage_is_close_to_eighty_percent():
    rng = np.random.default_rng(0)
    segments = ["ready_unit", "off_plan_unit"] * 2_500
    intervals = fit_intervals(segments, np.abs(rng.normal(0, 0.1, 5_000)), CONFIG)
    actual = rng.normal(0, 0.1, 4_000)
    coverage = coverage_by_segment(["ready_unit"] * 4_000, np.zeros(4_000), actual, intervals)
    assert 0.75 <= coverage["all"] <= 0.85
    assert coverage["ready_unit"] == coverage["all"]


def test_small_segments_use_the_pooled_width():
    errors = np.linspace(0.0, 1.0, 1_000)
    segments = ["ready_unit"] * 990 + ["ready_villa"] * 10
    intervals = fit_intervals(segments, errors, CONFIG)
    assert intervals["ready_villa"] == intervals[POOLED]
    assert intervals["ready_unit"] != intervals[POOLED]
    assert half_width(intervals, "off_plan_villa") == intervals[POOLED]
    with pytest.raises(ValueError, match="too few calibration rows"):
        fit_intervals(["ready_unit"] * 2, [0.1, 0.2], CONFIG)


@pytest.mark.parametrize(
    ("level", "base_n", "days", "area_rows", "expected"),
    [
        ("building", 10, 90, 50, "HIGH"),
        ("building", 9, 90, 50, "MEDIUM"),
        ("building", 10, 91, 50, "MEDIUM"),
        ("area", 30, 10, 500, "MEDIUM"),
        ("area", 5, 180, 500, "MEDIUM"),
        ("area", 5, 181, 500, "LOW"),
        ("area", 4, 10, 500, "LOW"),
        ("building", 30, 5, 49, "LOW"),
        ("building", 30, None, 500, "LOW"),
        (None, None, None, 500, "LOW"),
    ],
)  # fmt: skip
def test_confidence_rules(level, base_n, days, area_rows, expected):
    assert confidence(level, base_n, days, area_rows, CONFIG.low_confidence_area_rows) == expected


def test_low_message_states_the_half_width():
    assert low_message(1_000_000, 820_000, 1_180_000) == (
        "Limited comparable data for this property type/area. "
        "Estimate has wide uncertainty (±18%)."
    )
    assert LOW_MESSAGE.format(pct=5).endswith("(±5%).")


def test_ape_is_the_price_error():
    assert ape([0.1, 0.0], [0.0, 0.1]) == pytest.approx([math.exp(0.1) - 1, 1 - math.exp(-0.1)])


def eval_frame():
    return pl.DataFrame(
        {
            "reg_type": ["ready", "ready", "off_plan", "ready"],
            "area_id": [1, 2, 3, 3],
            "building_age_proxy_years": [0.5, 3.0, 7.0, None],
        },
        schema={"reg_type": pl.Utf8, "area_id": pl.Int64, "building_age_proxy_years": pl.Float64},
    )


def test_top_areas_break_ties_by_id():
    train = pl.DataFrame({"area_id": [3, 3, 1, 2, 2, 5]})
    assert top_areas(train, 2) == [2, 3]
    assert top_areas(train, 10) == [2, 3, 1, 5]


def test_segment_masks():
    masks = segment_masks(eval_frame(), [3])
    as_lists = {name: mask.tolist() for name, mask in masks.items()}
    assert list(masks) == list(SEGMENTS)
    assert as_lists["all"] == [True] * 4
    assert as_lists["ready"] == [True, True, False, True]
    assert as_lists["off_plan"] == [False, False, True, False]
    assert as_lists["top5"] == [False, False, True, True]
    assert as_lists["rest"] == [True, True, False, False]
    assert as_lists["age_lt1"] == [True, False, False, False]
    assert as_lists["age_1to5"] == [False, True, False, False]
    assert as_lists["age_gt5"] == [False, False, True, False]
    assert as_lists["age_unknown"] == [False, False, False, True]
    assert as_lists["age_gt2"] == [False, True, True, False]


def test_segment_table_scores_every_model_per_segment():
    actual = np.zeros(4)
    predictions = {MODEL: np.array([0.0, 0.0, 0.5, 0.0]), "no_change": np.zeros(4)}
    table = segment_table(eval_frame(), predictions, actual, [3])
    assert table.columns == ["segment", "model", "rows", "mape", "median_ape", "flagged"]
    assert table.height == len(SEGMENTS) * 2
    assert table_mape(table, "off_plan", MODEL) == pytest.approx(math.exp(0.5) - 1)
    assert table_mape(table, "all", MODEL) == pytest.approx((math.exp(0.5) - 1) / 4)
    flagged = table.filter(pl.col("flagged"))
    assert flagged["segment"].to_list() == ["all", "off_plan", "top5", "age_gt5", "age_gt2"]
    assert (math.exp(0.5) - 1) / 4 > FLAG_MAPE
    empty = segment_table(eval_frame().head(1), {MODEL: np.zeros(1)}, np.zeros(1), [3])
    assert math.isnan(table_mape(empty, "off_plan", MODEL))
    assert not empty.filter(pl.col("segment") == "off_plan")["flagged"][0]


def test_bootstrap_upper_bound():
    errors = np.random.default_rng(1).exponential(0.1, 2_000)
    upper = bootstrap_mape_upper(errors, 1_000, seed=7)
    assert upper == bootstrap_mape_upper(errors, 1_000, seed=7)
    assert errors.mean() < upper < errors.mean() + 0.01
    assert bootstrap_mape_upper(np.full(10, 0.2), 1_000, seed=7) == pytest.approx(0.2)
    assert math.isnan(bootstrap_mape_upper(np.array([]), 1_000, seed=7))


def table_of(model=0.05, no_change=0.10, area_trend=0.08, ready=None, rows=100):
    records = []
    for segment in SEGMENTS:
        for name, value in (("model", model), ("no_change", no_change), ("area_trend", area_trend)):
            if name == "model" and segment == "ready" and ready is not None:
                value = ready
            records.append(
                {"segment": segment, "model": name, "rows": rows, "mape": value,
                 "median_ape": value / 2, "flagged": value > FLAG_MAPE}
            )  # fmt: skip
    return pl.DataFrame(records)


def test_gate_passes_when_every_condition_holds():
    result = gate(table_of(), THREE_M, model_upper=0.06)
    assert result.passed
    assert result.reasons == ()
    assert result.checks["strongest_baseline"] == 0.08


def test_gate_fails_a_primary_segment_over_the_limit():
    result = gate(table_of(ready=0.16), THREE_M, model_upper=0.06)
    assert not result.passed
    assert result.reasons == ("3m resale MAPE 16.0% exceeds the 15% gate",)


def test_gate_fails_when_only_one_baseline_is_beaten():
    result = gate(table_of(model=0.09), THREE_M, model_upper=0.095)
    assert not result.passed
    assert "3m MAPE 9.0% does not beat the area_trend baseline (8.0%)" in result.reasons
    assert not any("no_change" in reason for reason in result.reasons)


def test_gate_fails_when_the_upper_bound_is_not_below_the_stronger_baseline():
    result = gate(table_of(), THREE_M, model_upper=0.08)
    assert result.reasons == (
        "3m MAPE 95% upper bound 8.0% is not below the stronger baseline (8.0%)",
    )


def test_gate_fails_without_primary_rows():
    table = table_of().with_columns(
        pl.when(pl.col("segment") == "age_gt2").then(0).otherwise(pl.col("rows")).alias("rows")
    )
    result = gate(table, THREE_M, model_upper=0.06)
    assert result.reasons == ("3m has no older-building (>2y) test rows",)
    assert set(PRIMARY_SEGMENTS) == {"ready", "top5", "age_gt2"}


def test_gate_uses_each_horizons_limit():
    three_y = dataclasses.replace(HORIZON_SPECS["3y"])
    result = gate(table_of(model=0.25, no_change=0.4, area_trend=0.35), three_y, model_upper=0.3)
    assert result.passed


def test_format_table_lines():
    lines = format_table("3m", table_of(model=0.3, no_change=0.4, area_trend=0.35))
    assert len(lines) == 1 + len(SEGMENTS)
    assert "no_change" in lines[0] and "area_trend" in lines[0]
    assert re.search(r"all\s+100\s+30\.00%\s+40\.00%\s+35\.00%\s+15\.00%\s+FLAG >25%", lines[1])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_evaluate.py`

Expected: FAIL with an import error for `models.forecast.evaluate`.

- [ ] **Step 3: Write intervals.py**

`models/forecast/intervals.py`:

```python
"""Split-conformal growth ranges, confidence labels and the LOW-confidence message."""

import math

import numpy as np
import polars as pl

from models.forecast.config import ForecastConfig
from models.price.evaluate import conformal_quantile

POOLED = "_pooled"
LOW_MESSAGE = (
    "Limited comparable data for this property type/area. "
    "Estimate has wide uncertainty (±{pct}%)."
)
HIGH_MIN_SALES = 10
HIGH_MAX_DAYS = 90
MEDIUM_MIN_SALES = 5
MEDIUM_MAX_DAYS = 180


def interval_segments(frame: pl.DataFrame) -> pl.Series:
    kind = pl.when(pl.col("sub_kind") == "villa").then(pl.lit("villa")).otherwise(pl.lit("unit"))
    return frame.select(pl.format("{}_{}", "reg_type", kind).alias("segment"))["segment"]


def fit_intervals(segments, abs_errors, config: ForecastConfig) -> dict[str, float]:
    """Half-widths (log growth) from calibration |errors|; small segments use the pooled one."""
    frame = pl.DataFrame(
        {"segment": list(segments), "error": np.asarray(abs_errors, dtype=float)},
        schema={"segment": pl.Utf8, "error": pl.Float64},
    )
    pooled = conformal_quantile(frame["error"].to_numpy(), config.alpha)
    if not math.isfinite(pooled):
        raise ValueError(
            f"too few calibration rows ({frame.height}) for an {1 - config.alpha:.0%} interval"
        )
    out = {POOLED: pooled}
    for segment in sorted(frame["segment"].unique().to_list()):
        errors = frame.filter(pl.col("segment") == segment)["error"].to_numpy()
        enough = errors.size >= config.min_conformal_rows
        out[segment] = conformal_quantile(errors, config.alpha) if enough else pooled
    return out


def half_width(intervals: dict[str, float], segment: str) -> float:
    return intervals.get(segment, intervals[POOLED])


def coverage_by_segment(segments, predicted, actual, intervals) -> dict[str, float]:
    segments = list(segments)
    halves = np.array([half_width(intervals, segment) for segment in segments])
    residuals = np.abs(np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float))
    inside = residuals <= halves
    if inside.size == 0:
        return {}
    frame = pl.DataFrame({"segment": segments, "inside": inside})
    by_segment = frame.group_by("segment").agg(pl.col("inside").mean()).sort("segment")
    return {"all": float(inside.mean()), **dict(by_segment.iter_rows())}


def confidence(base_level, base_n, comparable_days, area_rows: int, min_area_rows: int) -> str:
    if base_level is None or base_n is None or comparable_days is None:
        return "LOW"
    if area_rows < min_area_rows:
        return "LOW"
    if base_level == "building" and base_n >= HIGH_MIN_SALES and comparable_days <= HIGH_MAX_DAYS:
        return "HIGH"
    if base_n >= MEDIUM_MIN_SALES and comparable_days <= MEDIUM_MAX_DAYS:
        return "MEDIUM"
    return "LOW"


def low_message(point: float, low: float, high: float) -> str:
    return LOW_MESSAGE.format(pct=round(100 * (high - low) / 2 / point))
```

- [ ] **Step 4: Write evaluate.py**

`models/forecast/evaluate.py`:

```python
"""Price-error metrics, evaluation segments, the bootstrap bound and the per-horizon gate.

There is deliberately no single headline accuracy number: every table is per segment.
"""

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from models.forecast.baselines import BASELINES
from models.forecast.config import Horizon

MODEL = "model"
FLAG_MAPE = 0.25
SEGMENTS = (
    "all", "off_plan", "ready", "top5", "rest",
    "age_lt1", "age_1to5", "age_gt5", "age_unknown", "age_gt2",
)  # fmt: skip
PRIMARY_SEGMENTS = ("ready", "top5", "age_gt2")
SEGMENT_TEXT = {"ready": "resale", "top5": "top-5-area", "age_gt2": "older-building (>2y)"}
BOOTSTRAP_CHUNK = 100
TABLE_SCHEMA = {
    "segment": pl.Utf8,
    "model": pl.Utf8,
    "rows": pl.Int64,
    "mape": pl.Float64,
    "median_ape": pl.Float64,
    "flagged": pl.Boolean,
}


def ape(predicted_growth, actual_growth) -> np.ndarray:
    """|predicted price / actual price - 1| where both prices share the same base."""
    difference = np.asarray(predicted_growth, dtype=float) - np.asarray(actual_growth, dtype=float)
    return np.abs(np.expm1(difference))


def top_areas(train: pl.DataFrame, count: int) -> list[int]:
    counts = train.group_by("area_id").len().sort(["len", "area_id"], descending=[True, False])
    return counts["area_id"].head(count).to_list()


def segment_masks(frame: pl.DataFrame, top: list[int]) -> dict[str, np.ndarray]:
    age = pl.col("building_age_proxy_years")
    in_top = pl.col("area_id").is_in(top)
    expressions = {
        "off_plan": pl.col("reg_type") == "off_plan",
        "ready": pl.col("reg_type") == "ready",
        "top5": in_top,
        "rest": ~in_top,
        "age_lt1": age < 1,
        "age_1to5": age.is_between(1, 5),
        "age_gt5": age > 5,
        "age_unknown": age.is_null(),
        "age_gt2": age > 2,
    }
    masks = frame.select(
        expression.fill_null(False).alias(name) for name, expression in expressions.items()
    )
    out = {"all": np.ones(frame.height, dtype=bool)}
    out.update({name: masks[name].to_numpy() for name in expressions})
    return {name: out[name] for name in SEGMENTS}


def segment_table(frame, predictions: dict[str, np.ndarray], actual, top) -> pl.DataFrame:
    actual = np.asarray(actual, dtype=float)
    records = []
    for segment, mask in segment_masks(frame, top).items():
        for name, predicted in predictions.items():
            errors = ape(np.asarray(predicted)[mask], actual[mask])
            mape = float(errors.mean()) if errors.size else math.nan
            records.append(
                {
                    "segment": segment,
                    "model": name,
                    "rows": int(mask.sum()),
                    "mape": mape,
                    "median_ape": float(np.median(errors)) if errors.size else math.nan,
                    "flagged": bool(errors.size) and mape > FLAG_MAPE,
                }
            )
    return pl.DataFrame(records, schema=TABLE_SCHEMA)


def bootstrap_mape_upper(errors, n_resamples: int, seed: int) -> float:
    """Upper end of the 95% percentile interval of the mean, from row-level resamples."""
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return math.nan
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_resamples, BOOTSTRAP_CHUNK):
        size = min(BOOTSTRAP_CHUNK, n_resamples - start)
        picks = rng.integers(0, errors.size, size=(size, errors.size))
        means.append(errors[picks].mean(axis=1))
    return float(np.quantile(np.concatenate(means), 0.975))


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]
    checks: dict[str, float]


def table_mape(table: pl.DataFrame, segment: str, model: str) -> float:
    row = table.filter((pl.col("segment") == segment) & (pl.col("model") == model))
    if row.height == 0 or row["rows"][0] == 0:
        return math.nan
    return float(row["mape"][0])


def gate(table: pl.DataFrame, horizon: Horizon, model_upper: float) -> GateResult:
    reasons = []
    checks = {}
    for segment in PRIMARY_SEGMENTS:
        value = table_mape(table, segment, MODEL)
        checks[f"{segment}_mape"] = value
        if math.isnan(value):
            reasons.append(f"{horizon.name} has no {SEGMENT_TEXT[segment]} test rows")
        elif value > horizon.mape_gate:
            reasons.append(
                f"{horizon.name} {SEGMENT_TEXT[segment]} MAPE {value:.1%} exceeds the "
                f"{horizon.mape_gate:.0%} gate"
            )
    model_all = table_mape(table, "all", MODEL)
    baseline_all = {name: table_mape(table, "all", name) for name in BASELINES}
    for name, value in baseline_all.items():
        checks[f"{name}_mape"] = value
        if not model_all < value:
            reasons.append(
                f"{horizon.name} MAPE {model_all:.1%} does not beat the {name} baseline "
                f"({value:.1%})"
            )
    strongest = min(baseline_all.values())
    checks.update(model_mape=model_all, model_upper=model_upper, strongest_baseline=strongest)
    if not model_upper < strongest:
        reasons.append(
            f"{horizon.name} MAPE 95% upper bound {model_upper:.1%} is not below the stronger "
            f"baseline ({strongest:.1%})"
        )
    return GateResult(not reasons, tuple(reasons), checks)


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2%}"


def format_table(name: str, table: pl.DataFrame) -> list[str]:
    models = table["model"].unique(maintain_order=True).to_list()
    header = f"{name:<4}{'segment':<13}{'rows':>8}" + "".join(f"{m:>13}" for m in models)
    lines = [header + f"{'model MdAPE':>13}  flag"]
    for segment in SEGMENTS:
        part = table.filter(pl.col("segment") == segment)
        mapes = dict(zip(part["model"].to_list(), part["mape"].to_list(), strict=True))
        mine = part.filter(pl.col("model") == MODEL).row(0, named=True)
        cells = "".join(f"{_pct(mapes[m]):>13}" for m in models)
        flag = "FLAG >25%" if mine["flagged"] else ""
        lines.append(
            f"{'':<4}{segment:<13}{part['rows'][0]:>8,}{cells}{_pct(mine['median_ape']):>13}  {flag}"
        )
    return lines
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/models/forecast/test_forecast_evaluate.py`

Expected: every test passes.

- [ ] **Step 6: Lint, test, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/models/forecast
git add models/forecast/intervals.py models/forecast/evaluate.py tests/models/forecast/test_forecast_evaluate.py
git commit -m "feat(forecast): conformal ranges, confidence labels, segment tables and the gate

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: The horizon model, per-horizon training and MLflow registration

**Files:**
- Create:
  - `models/forecast/model.py`
  - `models/forecast/train.py`
  - `tests/models/forecast/test_forecast_train.py`

**Interfaces:**

*Consumes:*
- From Tasks 1–8: `features.to_matrix`, `fit_categories`, `build_dataset`; `folds.plan_folds`, `fold_split`, `insufficiency`, `Fold`; `baselines.baseline_growth`; `intervals.fit_intervals`, `interval_segments`, `coverage_by_segment`, `half_width`; `evaluate.ape`, `segment_table`, `top_areas`, `bootstrap_mape_upper`, `gate`; `infra.INFRA_COLUMNS`.
- From earlier phases:
  - `models.price.boosting.make_dmatrix`, `resolve_device`
  - `models.price.pyfunc.pip_requirements`
  - `models.price.registry.configure`, `log_metrics`, `register_champion`, `CHAMPION_ALIAS`
  - `listings.fraud.resolve_price_model_version(uri) -> str | None`, which also resolves `models:/name@alias` for any registered model.

*Produces, from `models.forecast.model`:*
- `ForecastModel(horizon, booster, rounds, categories, intervals, area_rows, metadata)`, a dataclass with:
  - `.predict_growth(frame) -> np.ndarray`
  - `.contributions(frame) -> np.ndarray`, of shape `(n, len(FEATURES) + 1)`; the last column is the bias.
  - `.half_width(segment) -> float`
  - `.importance() -> pl.DataFrame`, with columns `feature, gain`.
  - `.save(directory) -> Path` and the classmethod `.load(directory)`. Loading forces `device=cpu`.
- `ForecastPyfunc`
- `log_forecast_model(model, directory, artifact_path) -> str` (a model URI; no `code_paths`)
- `load_champion(name) -> tuple[ForecastModel | None, str | None]`
- `latest_gates(experiment) -> dict[str, dict[str, str]]`: `{h: {"status", "reason"}}`, read from the tags of the latest run.

*Produces, from `models.forecast.train`:*
- `suggest_params(trial) -> dict`
- `fit_booster(train, val, params, categories, device, config, label, num_rounds=None) -> tuple[xgb.Booster, int]`, which returns the booster and the number of rounds used.
- `predict_rounds(booster, frame, categories, rounds) -> np.ndarray`
- `HorizonResult`, with:
  - fields `horizon, status, reasons, folds, table, fold_scores, coverage, upper, model, seconds`;
  - `.metrics() -> dict[str, float]`.
- `run_horizon(frame, horizon, device, config, data_end) -> HorizonResult`
- `TrainingSummary(device, run_id, results, versions, seconds)`
- `run_training(frame, report, quality, projects, data_end, config, *, device="auto", register=True, horizons=None, tracking_uri=None, artifact_location=None) -> TrainingSummary`

**Training flow for one horizon** (spec "Models", plus rulings 3, 4, 12 and 14):
1. **Plan and check the folds.** Plan the folds; the last one is the test fold. If `insufficiency(...)` returns a reason, the status is `insufficient_data` and the flow stops.
2. **Fit the categories.** `categories = fit_categories(test-fold training rows)`. These are used for every fold (ruling 12).
3. **Tune.** Optuna runs `n_trials` seeded trials on the `tune` folds. Trial 0 is the base parameters. Each trial fits with early stopping on the fold's validation rows and scores the mean validation MAPE.
4. **Score with the best parameters.** Refit every non-test fold with the best parameters and record each fold's MAPE for the model and both baselines. The `tune` folds' |log errors| become the conformal calibration, and their round counts give the final `rounds`.
5. **Fit the final model** on the test fold's training rows, with `rounds` fixed and no evaluation set.
6. **Score the test fold.** Predict the test fold, build the segment table with the baselines, compute the bootstrap upper bound, then apply the gate and compute coverage.

- [ ] **Step 1: Write the failing tests**

`tests/models/forecast/test_forecast_train.py`:

```python
import dataclasses

import numpy as np
import pytest
from forecast_fixtures import fake_load_rows, prepared_history, project_table

from models.forecast.config import FEATURES, HORIZON_SPECS, ForecastConfig
from models.forecast.features import build_dataset
from models.forecast.train import run_horizon, run_training

FAST = ForecastConfig(
    n_trials=1,
    n_estimators=200,
    learning_rate=0.1,
    tune_folds=2,
    min_test_rows=50,
    n_bootstrap=200,
    device="cpu",
)
THREE_M = HORIZON_SPECS["3m"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    rows, data_end = prepared_history(per_building_per_week=4)
    projects = project_table(tmp_path_factory.mktemp("projects"))
    frame, report = build_dataset(rows, data_end, projects)
    return frame, report, projects, data_end


@pytest.fixture(scope="module")
def passed(dataset):
    frame, _, _, data_end = dataset
    return run_horizon(frame, THREE_M, "cpu", FAST, data_end)


def test_a_learnable_horizon_passes_its_gate(passed):
    assert passed.status == "passed", passed.reasons
    assert passed.reasons == ()
    assert [fold.role for fold in passed.folds][-1] == "test"
    assert passed.table is not None and passed.table.height > 0
    assert 0.5 <= passed.coverage["all"] <= 1.0
    assert passed.model is not None and passed.model.rounds >= 1
    metrics = passed.metrics()
    assert metrics["gate.3m.passed"] == 1.0
    assert "3m.test.all.model.mape" in metrics
    assert "3m.folds.area_trend.mape_mean" in metrics
    assert set(passed.fold_scores["role"].unique()) <= {"score", "tune"}
    assert passed.model.metadata["test_cutoff"] == passed.folds[-1].cutoff.isoformat()


def test_a_strict_gate_fails_but_still_reports(dataset):
    frame, _, _, data_end = dataset
    strict = dataclasses.replace(THREE_M, mape_gate=0.0)
    result = run_horizon(frame, strict, "cpu", FAST, data_end)
    assert result.status == "failed"
    assert any("exceeds the 0% gate" in reason for reason in result.reasons)
    assert result.table is not None
    assert result.metrics()["gate.3m.passed"] == 0.0


def test_a_short_history_is_insufficient(dataset):
    frame, _, _, data_end = dataset
    result = run_horizon(frame, HORIZON_SPECS["3y"], "cpu", FAST, data_end)
    assert result.status == "insufficient_data"
    assert result.reasons[0].startswith("3y: only ")
    assert result.model is None and result.table is None
    assert result.metrics()["gate.3y.passed"] == 0.0


def test_the_model_round_trips_and_explains_itself(passed, dataset, tmp_path):
    from models.forecast.model import ForecastModel

    frame = dataset[0].filter(dataset[0]["growth_3m"].is_not_null()).head(50)
    model = passed.model
    loaded = ForecastModel.load(model.save(tmp_path / "model"))
    assert np.allclose(loaded.predict_growth(frame), model.predict_growth(frame), atol=1e-6)
    contributions = loaded.contributions(frame)
    assert contributions.shape == (50, len(FEATURES) + 1)
    assert np.allclose(contributions.sum(axis=1), loaded.predict_growth(frame), atol=1e-4)
    assert loaded.half_width("ready_unit") > 0
    assert loaded.area_rows == model.area_rows
    assert loaded.importance().columns == ["feature", "gain"]


def test_run_training_registers_only_passing_horizons(dataset, temp_mlflow):
    from models.forecast.model import latest_gates, load_champion

    frame, report, projects, data_end = dataset
    quality = fake_load_rows(prepared_history(per_building_per_week=4))(None, None)[1]
    strict_year = dataclasses.replace(HORIZON_SPECS["1y"], mape_gate=0.0)
    summary = run_training(
        frame, report, quality, projects, data_end, FAST,
        device="cpu", horizons=(THREE_M, strict_year),
        tracking_uri=temp_mlflow["tracking_uri"],
        artifact_location=temp_mlflow["artifact_location"],
    )  # fmt: skip
    assert summary.versions == {"3m": "1"}
    assert summary.results["1y"].status == "failed"
    model, version = load_champion("zestimator-forecast-3m")
    assert version == "1" and model is not None
    assert model.horizon == "3m"
    assert load_champion("zestimator-forecast-1y") == (None, None)
    gates = latest_gates(FAST.experiment)
    assert gates["3m"] == {"status": "passed", "reason": "passed"}
    assert gates["1y"]["status"] == "failed"
    assert "exceeds the 0% gate" in gates["1y"]["reason"]
    assert gates["3y"]["status"] == "unknown"
```

**If `test_a_learnable_horizon_passes_its_gate` fails on the synthetic history,** first print `passed.table` and `passed.reasons`. Then fix it with exactly one of these, without weakening the gate and without editing thresholds:
- make the history larger (`per_building_per_week=6`);
- raise `n_estimators` in `FAST`.

The synthetic growth is learnable exactly from `area_code`, while `area_trend` is biased by about 1.5 months of growth, so the model should win. Record the change in your report.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_train.py`

Expected: FAIL with an import error for `models.forecast.train`.

- [ ] **Step 3: Write model.py**

`models/forecast/model.py`:

```python
"""A trained horizon model (booster, categories, intervals) and its MLflow pyfunc.

Logged without `code_paths`: the model is only loaded in-process from this repository, which
must be importable (a bundled copy of `models/` would shadow the live one on sys.path).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import xgboost as xgb

from models.forecast.config import HORIZONS
from models.forecast.features import to_matrix
from models.forecast.intervals import half_width
from models.price.boosting import make_dmatrix
from models.price.pyfunc import pip_requirements
from models.price.registry import CHAMPION_ALIAS

LOGGER = logging.getLogger(__name__)
BOOSTER_FILE = "booster.ubj"
META_FILE = "forecast_model.json"


@dataclass
class ForecastModel:
    horizon: str
    booster: xgb.Booster
    rounds: int
    categories: dict[str, list[str]]
    intervals: dict[str, float]
    area_rows: dict[int, int]
    metadata: dict = field(default_factory=dict)

    def _dmatrix(self, frame: pl.DataFrame) -> xgb.DMatrix:
        return make_dmatrix(to_matrix(frame, self.categories))

    def predict_growth(self, frame: pl.DataFrame) -> np.ndarray:
        return self.booster.predict(self._dmatrix(frame), iteration_range=(0, self.rounds))

    def contributions(self, frame: pl.DataFrame) -> np.ndarray:
        return self.booster.predict(
            self._dmatrix(frame), pred_contribs=True, iteration_range=(0, self.rounds)
        )

    def half_width(self, segment: str) -> float:
        return half_width(self.intervals, segment)

    def importance(self) -> pl.DataFrame:
        gains = self.booster.get_score(importance_type="gain")
        return pl.DataFrame(
            {"feature": list(gains), "gain": [float(value) for value in gains.values()]},
            schema={"feature": pl.Utf8, "gain": pl.Float64},
        ).sort("gain", descending=True)

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / BOOSTER_FILE)
        meta = {
            "horizon": self.horizon,
            "rounds": self.rounds,
            "categories": self.categories,
            "intervals": self.intervals,
            "area_rows": {str(key): value for key, value in self.area_rows.items()},
            "metadata": self.metadata,
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2, default=str), "utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path) -> "ForecastModel":
        directory = Path(directory)
        booster = xgb.Booster()
        booster.load_model(directory / BOOSTER_FILE)
        booster.set_param({"device": "cpu"})
        meta = json.loads((directory / META_FILE).read_text("utf-8"))
        return cls(
            horizon=meta["horizon"],
            booster=booster,
            rounds=int(meta["rounds"]),
            categories=meta["categories"],
            intervals=meta["intervals"],
            area_rows={int(key): value for key, value in meta["area_rows"].items()},
            metadata=meta["metadata"],
        )


class ForecastPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = ForecastModel.load(Path(context.artifacts["model_dir"]))

    def predict(self, context, model_input, params=None):
        return self.model.predict_growth(pl.from_pandas(model_input))


def log_forecast_model(model: ForecastModel, directory: Path, artifact_path: str) -> str:
    info = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=ForecastPyfunc(),
        artifacts={"model_dir": str(model.save(directory))},
        pip_requirements=pip_requirements(),
    )
    return info.model_uri


def load_champion(name: str) -> tuple[ForecastModel | None, str | None]:
    """Resolve the champion alias to a version first, then load exactly that version."""
    from listings.fraud import resolve_price_model_version  # resolves any models:/ URI

    version = resolve_price_model_version(f"models:/{name}@{CHAMPION_ALIAS}")
    if version is None:
        return None, None
    try:
        pyfunc = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    except Exception as exc:  # noqa: BLE001 — no champion is an expected state
        LOGGER.warning("forecast model %s v%s could not be loaded (%s)", name, version, exc)
        return None, None
    return pyfunc.unwrap_python_model().model, version


def latest_gates(experiment: str) -> dict[str, dict[str, str]]:
    """Gate status and reason per horizon from the most recent training run's tags."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment],
            order_by=["attributes.start_time DESC"],
            max_results=1,
            output_format="list",
        )
    except Exception as exc:  # noqa: BLE001 — no experiment yet is an expected state
        LOGGER.warning("no forecast runs found in %s (%s)", experiment, exc)
        runs = []
    tags = runs[0].data.tags if runs else {}
    return {
        name: {
            "status": tags.get(f"gate.{name}.status", "unknown"),
            "reason": tags.get(f"gate.{name}.reason", ""),
        }
        for name in HORIZONS
    }
```

- [ ] **Step 4: Write train.py**

`models/forecast/train.py`:

```python
"""Per-horizon training: walk-forward tuning, conformal calibration, final fit and the gate.

Spec: "Models", "Intervals" and "Evaluation and gate" in
docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md.
"""

import dataclasses
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import optuna
import polars as pl
import xgboost as xgb

from models.forecast.baselines import baseline_growth
from models.forecast.config import HORIZON_SPECS, ForecastConfig, Horizon
from models.forecast.evaluate import MODEL, ape, bootstrap_mape_upper, gate, segment_table, top_areas
from models.forecast.features import fit_categories, to_matrix
from models.forecast.folds import Fold, fold_split, insufficiency, plan_folds
from models.forecast.infra import INFRA_COLUMNS
from models.forecast.intervals import coverage_by_segment, fit_intervals, interval_segments
from models.forecast.model import ForecastModel, log_forecast_model
from models.forecast.rows import DataQuality
from models.forecast.targets import TargetReport
from models.price.boosting import make_dmatrix, resolve_device
from models.price.registry import configure, log_metrics, register_champion

FOLD_SCORE_SCHEMA = {
    "fold": pl.Int64, "role": pl.Utf8, "model": pl.Utf8, "rows": pl.Int64, "mape": pl.Float64,
}  # fmt: skip


def base_params(config: ForecastConfig) -> dict:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
    }


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 64.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def fit_booster(train, val, params, categories, device, config, label, num_rounds=None):
    full = {**base_params(config), **params, "device": device, "seed": config.seed}
    dtrain = make_dmatrix(to_matrix(train, categories), label=train[label].to_numpy())
    if val is None:
        return xgb.train(full, dtrain, num_boost_round=num_rounds), num_rounds
    dval = make_dmatrix(to_matrix(val, categories), label=val[label].to_numpy())
    booster = xgb.train(
        full,
        dtrain,
        num_boost_round=config.n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )
    return booster, int(booster.best_iteration) + 1


def predict_rounds(booster, frame, categories, rounds) -> np.ndarray:
    return booster.predict(make_dmatrix(to_matrix(frame, categories)), iteration_range=(0, rounds))


def _run_fold(frame, horizon, fold, params, categories, device, config):
    """(validation rows, predictions, rounds) or None when the fold has no validation rows."""
    label = f"growth_{horizon.name}"
    train, val = fold_split(frame, horizon, fold)
    if val.height == 0 or train.height == 0:
        return None
    booster, rounds = fit_booster(train, val, params, categories, device, config, label)
    return val, predict_rounds(booster, val, categories, rounds), rounds


def tune(frame, horizon, folds, categories, device, config) -> dict:
    label = f"growth_{horizon.name}"
    tuning = [fold for fold in folds if fold.role == "tune"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    base = base_params(config)
    study.enqueue_trial({name: base[name] for name in (
        "max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree",
        "reg_lambda",
    )})  # fmt: skip

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores = []
        for fold in tuning:
            outcome = _run_fold(frame, horizon, fold, params, categories, device, config)
            if outcome is not None:
                val, predicted, _ = outcome
                scores.append(float(np.mean(ape(predicted, val[label].to_numpy()))))
        return float(np.mean(scores)) if scores else float("inf")

    study.optimize(objective, n_trials=config.n_trials)
    return dict(study.best_params)


@dataclass
class HorizonResult:
    horizon: str
    status: str  # "passed", "failed" or "insufficient_data"
    reasons: tuple[str, ...]
    folds: list[Fold]
    table: pl.DataFrame | None = None
    fold_scores: pl.DataFrame | None = None
    coverage: dict[str, float] = dataclasses.field(default_factory=dict)
    upper: float | None = None
    model: ForecastModel | None = None
    seconds: float = 0.0

    def metrics(self) -> dict[str, float]:
        name = self.horizon
        out = {f"gate.{name}.passed": float(self.status == "passed"), f"{name}.folds": len(self.folds)}
        if self.table is not None:
            for row in self.table.iter_rows(named=True):
                prefix = f"{name}.test.{row['segment']}.{row['model']}"
                out[f"{prefix}.mape"] = row["mape"]
                out[f"{prefix}.median_ape"] = row["median_ape"]
                out[f"{prefix}.rows"] = row["rows"]
        if self.fold_scores is not None and self.fold_scores.height:
            means = self.fold_scores.group_by("model").agg(pl.col("mape").mean())
            for model, value in means.iter_rows():
                out[f"{name}.folds.{model}.mape_mean"] = value
        for segment, share in self.coverage.items():
            out[f"{name}.coverage.{segment}"] = share
        if self.upper is not None:
            out[f"{name}.test.all.model.mape_upper95"] = self.upper
        out[f"{name}.seconds"] = self.seconds
        return out


def run_horizon(
    frame: pl.DataFrame, horizon: Horizon, device: str, config: ForecastConfig, data_end: date
) -> HorizonResult:
    started = time.perf_counter()
    label = f"growth_{horizon.name}"
    folds = plan_folds(frame, horizon, config)
    train, test = fold_split(frame, horizon, folds[-1]) if folds else (frame.head(0), frame.head(0))
    reason = insufficiency(folds, test.height, config)
    if reason is not None:
        return HorizonResult(
            horizon.name, "insufficient_data", (f"{horizon.name}: {reason}",), folds,
            seconds=time.perf_counter() - started,
        )  # fmt: skip
    categories = fit_categories(train, config)
    params = tune(frame, horizon, folds, categories, device, config)

    scores, calibration_segments, calibration_errors, rounds = [], [], [], []
    for fold in folds[:-1]:
        outcome = _run_fold(frame, horizon, fold, params, categories, device, config)
        if outcome is None:
            continue
        val, predicted, used = outcome
        actual = val[label].to_numpy()
        for name, values in {MODEL: predicted, **baseline_growth(val, horizon)}.items():
            scores.append(
                {"fold": fold.index, "role": fold.role, "model": name, "rows": val.height,
                 "mape": float(np.mean(ape(values, actual)))}
            )  # fmt: skip
        if fold.role == "tune":
            calibration_segments += interval_segments(val).to_list()
            calibration_errors += np.abs(predicted - actual).tolist()
            rounds.append(used)
    intervals = fit_intervals(calibration_segments, calibration_errors, config)
    final_rounds = max(1, round(float(np.mean(rounds)))) if rounds else config.n_estimators

    booster, _ = fit_booster(train, None, params, categories, device, config, label, final_rounds)
    actual = test[label].to_numpy()
    predicted = predict_rounds(booster, test, categories, final_rounds)
    top = top_areas(train, config.top_areas)
    table = segment_table(test, {MODEL: predicted, **baseline_growth(test, horizon)}, actual, top)
    upper = bootstrap_mape_upper(ape(predicted, actual), config.n_bootstrap, config.seed)
    verdict = gate(table, horizon, upper)
    coverage = coverage_by_segment(interval_segments(test), predicted, actual, intervals)
    test_fold = folds[-1]
    model = ForecastModel(
        horizon=horizon.name,
        booster=booster,
        rounds=final_rounds,
        categories=categories,
        intervals=intervals,
        area_rows=dict(train.group_by("area_id").len().iter_rows()),
        metadata={
            "params": params,
            "data_end": data_end.isoformat(),
            "test_cutoff": test_fold.cutoff.isoformat(),
            "test_end": test_fold.end.isoformat(),
            "top_areas": top,
            "train_rows": train.height,
            "test_rows": test.height,
            "gate": dict(verdict.checks),
        },
    )
    return HorizonResult(
        horizon=horizon.name,
        status="passed" if verdict.passed else "failed",
        reasons=verdict.reasons,
        folds=folds,
        table=table,
        fold_scores=pl.DataFrame(scores, schema=FOLD_SCORE_SCHEMA),
        coverage=coverage,
        upper=upper,
        model=model,
        seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class TrainingSummary:
    device: str
    run_id: str
    results: dict[str, HorizonResult]
    versions: dict[str, str]
    seconds: float


def _log_horizon(result: HorizonResult) -> None:
    name = result.horizon
    log_metrics(result.metrics())
    mlflow.set_tags(
        {
            f"gate.{name}.status": result.status,
            f"gate.{name}.reason": result.reasons[0] if result.reasons else "passed",
        }
    )
    mlflow.log_dict({"folds": [fold.to_dict() for fold in result.folds]}, f"{name}/folds.json")
    if result.table is not None:
        mlflow.log_text(result.table.write_csv(), f"{name}/test_segments.csv")
        mlflow.log_text(result.fold_scores.write_csv(), f"{name}/fold_scores.csv")
    if result.model is not None:
        mlflow.log_text(result.model.importance().write_csv(), f"{name}/importance.csv")
        mlflow.log_dict(result.model.intervals, f"{name}/intervals.json")


def run_training(
    frame: pl.DataFrame,
    report: TargetReport,
    quality: DataQuality,
    projects: pl.DataFrame,
    data_end: date,
    config: ForecastConfig,
    *,
    device: str = "auto",
    register: bool = True,
    horizons: tuple[Horizon, ...] | None = None,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> TrainingSummary:
    started = time.perf_counter()
    device = resolve_device(device)
    horizons = horizons or tuple(HORIZON_SPECS.values())
    configure(config.experiment, tracking_uri, artifact_location)
    results, versions = {}, {}
    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as scratch:
        params = {key: str(value) for key, value in dataclasses.asdict(config).items()}
        mlflow.log_params({**params, "resolved_device": device, "data_end": data_end.isoformat()})
        mlflow.log_dict({"quality": quality.to_dict(), "targets": report.to_dict()}, "data_quality.json")
        mlflow.log_text(projects.select(INFRA_COLUMNS).write_csv(), "infrastructure_projects.csv")
        for horizon in horizons:
            result = run_horizon(frame, horizon, device, config, data_end)
            results[horizon.name] = result
            _log_horizon(result)
            if register and result.status == "passed":
                uri = log_forecast_model(
                    result.model, Path(scratch) / horizon.name, f"model_{horizon.name}"
                )
                versions[horizon.name] = register_champion(
                    uri, f"{config.model_prefix}-{horizon.name}"
                )
        mlflow.set_tag("registered", ",".join(sorted(versions)) or "none")
    return TrainingSummary(device, run.info.run_id, results, versions, time.perf_counter() - started)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/models/forecast/test_forecast_train.py`

Expected: every test passes, in under 3 minutes on the CPU.

If MLflow or Optuna emits a warning that `-W error` turns into a failure, find where it comes from:
- If our call is wrong (a deprecated argument), fix the call.
- If the warning comes from the library itself, add a narrowly scoped `filterwarnings` mark to that test, with the exact message, and record why in your report.

Do not add a blanket filter.

- [ ] **Step 6: Lint, test, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/models/forecast
git add models/forecast/model.py models/forecast/train.py tests/models/forecast/test_forecast_train.py
git commit -m "feat(forecast): walk-forward tuned horizon models, gates and MLflow registration

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 10: The Forecaster (JSON output, ranges, confidence and key drivers)

**Files:**
- Create:
  - `models/forecast/predict.py`
  - `tests/models/forecast/test_forecast_predict.py`
- Modify: `tests/models/forecast/forecast_fixtures.py` (add `FakePrice`, `small_model`)

**Interfaces:**

*Consumes:*
- From Task 1: `rows.load_rows`, `excluded_summary`, `config.HORIZONS`, `FEATURES`, `CATEGORICAL`, `MOMENTUM_LAGS`, `INFRA_TYPES`, `ForecastConfig`.
- From Tasks 2, 3, 5 and 6: `targets.add_keys`, `add_base`; `features.add_core_features`, `PROPERTY_FEATURES`; `infra.load_projects`, `add_infra_features`.
- From Tasks 8 and 9: `intervals.confidence`, `low_message`; `model.ForecastModel`, `load_champion`, `latest_gates`.
- From earlier phases:
  - `models.price.predictor.PriceRequest`, `PriceInputError`, `KIND_TO_TYPE`
  - `ingestion.normalize.match_key`
  - `listings.fraud.load_price_predictor(uri)`, which returns a `PricePredictor` or None.
  - `.predict_one(request)` returns a `PriceEstimate`, which has `estimate_aed`, `range_80` and `model_version`.

*Produces, from `models.forecast.predict`:*
- `SNAPSHOT_COLUMNS`
- `query_rows(rows, day) -> pl.DataFrame`
- `build_snapshot(rows, data_end, projects) -> pl.DataFrame`: one row per `(area_id, sub_kind, building_key)` at T = data_end + 1 day, plus an area-level row where `building_key` is null.
- `resolve_area(request, areas, aliases) -> int`
- `describe(feature, value) -> str` and `driver_text(feature, value, contribution, horizon) -> str`
- `Forecaster`:
  - the constructor `Forecaster(snapshot, data_end, areas, aliases, models, gates, price, excluded, config)`;
  - `Forecaster.build(rows, data_end, projects, areas, aliases, models, gates, price, excluded, config)`;
  - `Forecaster.from_registry(settings, config)`;
  - `.forecast(property: dict, as_of: date | None = None) -> dict`.
  - `models` is a `dict[h, tuple[ForecastModel | None, str | None]]`, and `gates` is the output of `latest_gates`.

*Produces, from `forecast_fixtures`:*
- `FakePrice`, with `predict_one` returning an estimate of 1,000,000 with a range of (900,000, 1,100,000) and `model_version` "7".
- `small_model(frame, horizon="3m", rounds=30) -> ForecastModel`, a quick CPU fit.

**Rulings carried here:**
- **Ruling 7.** Forecasts are made at T = data_end + 1 day, so any other `as_of` is rejected with `PriceInputError("as_of", ...)`. `as_of` in the JSON is `data_end`.
- **Ruling 8.** Drivers come from the 1y model, then the 3m model, otherwise the list is empty.
- **Ruling 15.** A forecast's range is `estimate × exp(growth ± half_width)`. It covers growth uncertainty only; the current estimate's own range is reported separately in `current_range_80`, as the spec describes.
- **Ruling 16.** A missing champion is reported as `not_deployed`. The reason is taken from the latest run's gate tag when that run failed or had too little data; otherwise the reason is `"no registered <h> model"`.

- [ ] **Step 1: Add the fixture helpers**

Append to `tests/models/forecast/forecast_fixtures.py`:

```python
class FakePrice:
    """Stands in for the Phase 3 PricePredictor: a fixed 1,000,000 AED estimate."""

    def __init__(self):
        self.requests = []

    def predict_one(self, request):
        from types import SimpleNamespace

        self.requests.append(request)
        return SimpleNamespace(
            estimate_aed=1_000_000.0, range_80=(900_000.0, 1_100_000.0), model_version="7"
        )


def small_model(frame, horizon="3m", rounds=30):
    from models.forecast.config import ForecastConfig
    from models.forecast.features import fit_categories
    from models.forecast.model import ForecastModel
    from models.forecast.train import fit_booster

    config = ForecastConfig(device="cpu")
    train = frame.filter(pl.col(f"growth_{horizon}").is_not_null())
    categories = fit_categories(train, config)
    booster, used = fit_booster(
        train, None, {}, categories, "cpu", config, f"growth_{horizon}", rounds
    )
    return ForecastModel(
        horizon=horizon,
        booster=booster,
        rounds=used,
        categories=categories,
        intervals={"_pooled": 0.05, "ready_unit": 0.04},
        area_rows=dict(train.group_by("area_id").len().iter_rows()),
        metadata={},
    )
```

- [ ] **Step 2: Write the failing tests**

`tests/models/forecast/test_forecast_predict.py`:

```python
import dataclasses
import math
import re
from datetime import date, timedelta

import polars as pl
import pytest
from forecast_fixtures import (
    FakePrice,
    history_aliases,
    history_areas,
    prepared_history,
    project_table,
    small_model,
)

from models.forecast.config import ForecastConfig
from models.forecast.features import build_dataset
from models.forecast.predict import Forecaster, build_snapshot, describe, driver_text
from models.forecast.rows import EXCLUDED_SCHEMA
from models.price.predictor import PriceInputError

CONFIG = ForecastConfig(device="cpu")
MARINA_FLAT = {
    "property_id": "p-1",
    "area": "Dubai Marina",
    "building": "Tower 1-0",
    "property_kind": "apartment",
    "status": "ready",
    "size_sqm": 80.0,
    "bedrooms": 1,
}
GATES = {
    "3m": {"status": "passed", "reason": "passed"},
    "1y": {"status": "failed", "reason": "1y resale MAPE 31.2% exceeds the 20% gate"},
    "3y": {"status": "insufficient_data", "reason": "3y: only 0 walk-forward folds (needs 2)"},
}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rows, data_end = prepared_history()
    projects = project_table(tmp_path_factory.mktemp("projects"))
    frame, _ = build_dataset(rows, data_end, projects)
    return rows, data_end, projects, small_model(frame)


def forecaster(world, models=None, gates=GATES, excluded=None, config=CONFIG):
    rows, data_end, projects, model = world
    models = models if models is not None else {"3m": (model, "4")}
    excluded = excluded if excluded is not None else pl.DataFrame(schema=EXCLUDED_SCHEMA)
    return Forecaster.build(
        rows, data_end, projects, history_areas(), history_aliases(), models, gates,
        FakePrice(), excluded, config,
    )  # fmt: skip


def test_snapshot_matches_a_direct_computation(world):
    rows, data_end, projects, _ = world
    snapshot = build_snapshot(rows, data_end, projects)
    tower = snapshot.filter(pl.col("building_key") == "1|tower 1 0").row(0, named=True)
    since = data_end + timedelta(days=1) - timedelta(days=92)
    recent = rows.filter((pl.col("building_name") == "Tower 1-0") & (pl.col("instance_date") >= since))
    assert tower["base_level"] == "building"
    assert tower["base_n"] == recent.height
    assert tower["base_ppsm"] == pytest.approx(recent["ppsm"].median())
    assert snapshot.filter(pl.col("building_key").is_null()).height == 6  # 3 areas x flat/villa
    assert snapshot.select("area_id", "sub_kind", "building_key").is_unique().all()


def test_forecast_json_shape(world):
    result = forecaster(world).forecast(MARINA_FLAT)
    assert list(result) == [
        "property_id", "as_of", "current_estimate_aed", "current_range_80",
        "forecast_3m", "forecast_1y", "forecast_3y",
        "key_drivers", "exclusions_applied", "model_versions",
    ]  # fmt: skip
    assert result["property_id"] == "p-1"
    assert result["as_of"] == world[1].isoformat()
    assert result["current_estimate_aed"] == 1_000_000
    assert result["current_range_80"] == [900_000, 1_100_000]
    three = result["forecast_3m"]
    assert set(three) == {"point", "ci_low", "ci_high", "confidence"}
    assert three["ci_low"] < three["point"] < three["ci_high"]
    growth = math.log(three["point"] / 1_000_000)
    assert three["ci_high"] == pytest.approx(1_000_000 * math.exp(growth + 0.04), abs=2)
    assert three["confidence"] == "HIGH"
    assert result["forecast_1y"] == {
        "status": "not_deployed", "reason": "1y resale MAPE 31.2% exceeds the 20% gate",
    }  # fmt: skip
    assert result["forecast_3y"]["reason"] == "3y: only 0 walk-forward folds (needs 2)"
    assert result["model_versions"] == {"price": "7", "forecast_3m": "4"}


def test_no_number_without_a_range(world):
    result = forecaster(world).forecast(MARINA_FLAT)
    for name in ("forecast_3m", "forecast_1y", "forecast_3y"):
        block = result[name]
        assert ("point" in block) == ("ci_low" in block) == ("ci_high" in block)
    assert len(result["current_range_80"]) == 2


def test_unknown_building_falls_back_to_the_area(world):
    result = forecaster(world).forecast({**MARINA_FLAT, "building": "Nowhere Tower"})
    assert result["forecast_3m"]["confidence"] == "MEDIUM"


def test_thin_areas_get_the_low_message(world):
    rows, data_end, projects, model = world
    thin = dataclasses.replace(model, area_rows={})
    result = forecaster(world, models={"3m": (thin, "4")}).forecast(MARINA_FLAT)
    block = result["forecast_3m"]
    assert block["confidence"] == "LOW"
    pct = round(100 * (block["ci_high"] - block["ci_low"]) / 2 / block["point"])
    assert block["message"] == (
        "Limited comparable data for this property type/area. "
        f"Estimate has wide uncertainty (±{pct}%)."
    )
    assert re.fullmatch(r".*\(±\d+%\)\.", block["message"])


def test_key_drivers_come_from_shap(world):
    drivers = forecaster(world).forecast(MARINA_FLAT)["key_drivers"]
    assert 1 <= len(drivers) <= 3
    assert all(re.search(r"\([+-]\d+\.\d% to the 3m forecast\)$", text) for text in drivers)
    silent = dataclasses.replace(CONFIG, min_driver_contribution=10.0)
    assert forecaster(world, config=silent).forecast(MARINA_FLAT)["key_drivers"] == []
    nothing = forecaster(world, models={}).forecast(MARINA_FLAT)
    assert nothing["key_drivers"] == []
    assert nothing["forecast_3m"] == {"status": "not_deployed", "reason": "no registered 3m model"}


def test_driver_text_templates():
    assert describe("area_mom_12m", math.log(1.14)) == "Area prices rose 14% over the last 12 months"
    assert describe("city_mom_3m", math.log(0.9)) == (
        "Dubai-wide prices for this kind fell 10% over the last 3 months"
    )
    assert describe("base_level_building", 1.0) == "Recent prices come from the same building"
    assert describe("infra_active_metro_rail", 2.0) == "2 metro or rail projects under way in the area"
    assert describe("days_since_building_sale", None) == "Days since the building's last sale: unknown"
    assert driver_text("area_mom_12m", math.log(1.14), 0.0305, "1y") == (
        "Area prices rose 14% over the last 12 months (+3.1% to the 1y forecast)"
    )
    assert driver_text("off_plan", 1.0, -0.02, "3m").endswith("(-2.0% to the 3m forecast)")


def test_exclusions_name_the_area_and_segment(world):
    excluded = pl.DataFrame(
        {
            "area_id": [1, 1, 1, 2],
            "sub_kind": ["flat", "flat", "villa", "flat"],
            "reg_type": ["off_plan", "off_plan", "ready", "off_plan"],
            "month": [date(2022, 1, 1)] * 4,
            "reason": ["outlier"] * 4,
        },
        schema=EXCLUDED_SCHEMA,
    )
    result = forecaster(world, excluded=excluded).forecast(MARINA_FLAT)
    assert result["exclusions_applied"] == ["Dropped 2 off-plan outliers in Dubai Marina"]


def test_invalid_requests_raise_price_input_errors(world):
    engine = forecaster(world)
    with pytest.raises(PriceInputError, match="as_of"):
        engine.forecast(MARINA_FLAT, as_of=date(2020, 1, 1))
    with pytest.raises(PriceInputError, match="area"):
        engine.forecast({**MARINA_FLAT, "area": "Atlantis"})
    with pytest.raises(PriceInputError, match="size_sqm"):
        engine.forecast({**MARINA_FLAT, "size_sqm": -1})
    with pytest.raises(PriceInputError, match="no sales history"):
        engine.forecast({**MARINA_FLAT, "property_kind": "townhouse"})
    assert engine.forecast(MARINA_FLAT, as_of=world[1])["as_of"] == world[1].isoformat()
```

The fixture's `sub_kind` values are `flat` and `villa`, so `townhouse` has no snapshot row. `Dubai Marina` resolves to area 1 through `history_aliases`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_predict.py`

Expected: FAIL with an import error for `models.forecast.predict`.

- [ ] **Step 4: Write predict.py**

`models/forecast/predict.py`:

```python
"""Forecasts: the current estimate x exp(predicted growth), with ranges, labels and drivers.

Spec: "Prediction" in docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md.
Features for a request come from a snapshot computed once, at T = data_end + 1 day, by the
same code that builds training features (query rows carry a null ppsm and never count).
"""

import math
from datetime import date, timedelta

import numpy as np
import polars as pl

from ingestion.config import DbSettings
from ingestion.normalize import match_key
from models.forecast.config import CATEGORICAL, FEATURES, HORIZONS, INFRA_TYPES, ForecastConfig
from models.forecast.features import PROPERTY_FEATURES, add_core_features
from models.forecast.infra import add_infra_features, load_projects
from models.forecast.intervals import confidence, low_message
from models.forecast.model import ForecastModel, latest_gates, load_champion
from models.forecast.rows import excluded_summary, load_rows
from models.forecast.targets import add_base, add_keys
from models.price.predictor import KIND_TO_TYPE, PriceInputError, PriceRequest

SNAPSHOT_COLUMNS = (
    "base_level", "base_ppsm", "days_since_area_sale",
    *(name for name in FEATURES if name not in PROPERTY_FEATURES and name not in CATEGORICAL),
)  # fmt: skip
DRIVER_HORIZONS = ("1y", "3m")
LAG_TEXT = {"3m": "3 months", "12m": "12 months", "36m": "36 months"}
TYPE_TEXT = {
    "metro_rail": "metro or rail",
    "mall": "mall",
    "school": "school",
    "park": "park",
    "mixed_use": "mixed-use",
    "airport": "airport",
}
LABELS = {
    "ln_base_ppsm": "Recent price level",
    "base_level_building": "Where recent prices come from",
    "base_n": "Comparable sales in the last 3 months",
    "area_share_12m": "The area's share of Dubai sales",
    "days_since_building_sale": "Days since the building's last sale",
    "building_sales_12m": "Sales in this building in the last 12 months",
    "area_sales_12m": "Sales in this area in the last 12 months",
    "off_plan": "Registration status",
    "log_area_sqm": "Size",
    "bedrooms": "Bedrooms",
    "building_age_proxy_years": "Years since the building's first sale",
    "infra_months_to_next": "Months until the next nearby project completes",
    "infra_completed_24m": "Nearby project completions",
    "sub_kind": "Property kind",
    "area_code": "Location",
    "project_code": "Project",
}


def _moved(value: float) -> str:
    change = math.expm1(value)
    return f"{'rose' if change >= 0 else 'fell'} {abs(change):.0%}"


def describe(feature: str, value) -> str:
    """A plain sentence about one feature's value (never about its effect)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        label = LABELS.get(feature, feature.replace("_", " ").capitalize())
        return f"{label}: unknown"
    for level, subject in (("area", "Area prices"), ("city", "Dubai-wide prices for this kind")):
        for lag, text in LAG_TEXT.items():
            if feature == f"{level}_mom_{lag}":
                return f"{subject} {_moved(value)} over the last {text}"
    for kind in INFRA_TYPES:
        if feature == f"infra_active_{kind}":
            return f"{value:.0f} {TYPE_TEXT[kind]} projects under way in the area"
        if feature == f"infra_mix_{kind}":
            return f"{value:.0%} of the area's active projects are {TYPE_TEXT[kind]}"
    templates = {
        "ln_base_ppsm": lambda v: f"Recent price level is AED {math.exp(v):,.0f}/m²",
        "base_level_building": lambda v: "Recent prices come from the same building"
        if v >= 0.5
        else "Recent prices come from the wider area",
        "base_n": lambda v: f"{v:.0f} comparable sales in the last 3 months",
        "area_share_12m": lambda v: f"This area had {v:.1%} of Dubai sales of this kind last year",
        "days_since_building_sale": lambda v: f"The building's last sale was {v:.0f} days ago",
        "building_sales_12m": lambda v: f"{v:.0f} sales in this building in the last 12 months",
        "area_sales_12m": lambda v: f"{v:.0f} sales in this area in the last 12 months",
        "off_plan": lambda v: "Off-plan" if v >= 0.5 else "Ready (resale)",
        "log_area_sqm": lambda v: f"Size {math.exp(v):,.0f} m²",
        "bedrooms": lambda v: "Studio" if v == 0 else f"{v:.0f} bedrooms",
        "building_age_proxy_years": lambda v: f"Building first sold {v:.1f} years ago",
        "infra_months_to_next": lambda v: f"Next nearby project completes in {v:.0f} months",
        "infra_completed_24m": lambda v: "A nearby project opened in the last 2 years"
        if v >= 0.5
        else "No nearby project opened in the last 2 years",
    }
    if feature in templates:
        return templates[feature](float(value))
    return f"{LABELS.get(feature, feature)}: {value}"


def driver_text(feature: str, value, contribution: float, horizon: str) -> str:
    effect = math.expm1(contribution)
    sign = "+" if effect >= 0 else "-"
    return f"{describe(feature, value)} ({sign}{abs(effect):.1%} to the {horizon} forecast)"


def query_rows(rows: pl.DataFrame, day: date) -> pl.DataFrame:
    """One ppsm-less row per building and per area x kind, dated `day`."""
    buildings = (
        rows.filter(pl.col("building_name").is_not_null())
        .select("area_id", "sub_kind", "building_name")
        .unique()
    )
    areas = (
        rows.select("area_id", "sub_kind")
        .unique()
        .with_columns(pl.lit(None, dtype=pl.Utf8).alias("building_name"))
    )
    queries = pl.concat([buildings, areas]).sort(
        "area_id", "sub_kind", "building_name", nulls_last=True
    )
    start = int(rows["row_id"].max()) + 1 if rows.height else 0
    index = pl.int_range(pl.len(), dtype=pl.Int64)
    return queries.with_columns(
        pl.lit(day).alias("instance_date"),
        pl.lit(None, dtype=pl.Float64).alias("ppsm"),
        (index + start).alias("row_id"),
        pl.format("query-{}", index).alias("transaction_id"),
    )


def build_snapshot(rows: pl.DataFrame, data_end: date, projects: pl.DataFrame) -> pl.DataFrame:
    day = data_end + timedelta(days=1)
    frame = pl.concat([rows, query_rows(rows, day)], how="diagonal_relaxed")
    frame = add_infra_features(add_core_features(add_base(add_keys(frame))), projects)
    return (
        frame.filter(pl.col("ppsm").is_null())
        .select("area_id", "sub_kind", "building_key", *SNAPSHOT_COLUMNS)
        .unique(["area_id", "sub_kind", "building_key"], keep="first", maintain_order=True)
    )


def resolve_area(request: PriceRequest, areas: pl.DataFrame, aliases: pl.DataFrame) -> int:
    if (request.area is None) == (request.area_id is None):
        raise PriceInputError("area", "give exactly one of area or area_id")
    if request.area_id is not None:
        if request.area_id not in set(areas["area_id"].to_list()):
            raise PriceInputError("area_id", f"unknown area_id {request.area_id}")
        return request.area_id
    key = match_key(request.area)
    ids = sorted(set(aliases.filter(pl.col("alias_key") == key)["area_id"].to_list())) if key else []
    if not ids:
        raise PriceInputError("area", f"unknown area {request.area!r}")
    if len(ids) > 1:
        raise PriceInputError("area", f"area {request.area!r} is ambiguous: use area_id, one of {ids}")
    return ids[0]


class Forecaster:
    def __init__(self, snapshot, data_end, areas, aliases, models, gates, price, excluded, config):
        self.snapshot = snapshot
        self.data_end = data_end
        self.areas = areas
        self.aliases = aliases
        self.models = models
        self.gates = gates
        self.price = price
        self.excluded = excluded
        self.config = config

    @classmethod
    def build(cls, rows, data_end, projects, areas, aliases, models, gates, price, excluded, config):
        snapshot = build_snapshot(rows, data_end, projects)
        return cls(snapshot, data_end, areas, aliases, models, gates, price, excluded, config)

    @classmethod
    def from_registry(cls, settings: DbSettings, config: ForecastConfig) -> "Forecaster":
        from listings.fraud import load_price_predictor

        price = load_price_predictor(config.price_model_uri)
        if price is None:
            raise RuntimeError(
                f"the price model {config.price_model_uri} is unavailable; forecasts need the "
                "current estimate"
            )
        rows, quality, data_end, areas, aliases = load_rows(settings, config)
        models = {name: load_champion(f"{config.model_prefix}-{name}") for name in HORIZONS}
        return cls.build(
            rows, data_end, load_projects(), areas, aliases, models,
            latest_gates(config.experiment), price, quality.excluded, config,
        )  # fmt: skip

    def forecast(self, property: dict, as_of: date | None = None) -> dict:
        if as_of is not None and as_of != self.data_end:
            raise PriceInputError(
                "as_of", f"forecasts are only available as of the data end, {self.data_end}"
            )
        request = PriceRequest.parse({k: v for k, v in property.items() if k != "property_id"})
        estimate = self.price.predict_one(request)
        area_id = resolve_area(request, self.areas, self.aliases)
        _, sub_kind = KIND_TO_TYPE[request.property_kind]
        features = self._features(request, area_id, sub_kind)
        base = features.row(0, named=True)
        building = base["base_level"] == "building"
        context = {
            "estimate": estimate.estimate_aed,
            "segment": f"{request.status}_{'villa' if sub_kind == 'villa' else 'unit'}",
            "base": base,
            "days": base["days_since_building_sale" if building else "days_since_area_sale"],
            "area_id": area_id,
        }
        out = {
            "property_id": property.get("property_id"),
            "as_of": self.data_end.isoformat(),
            "current_estimate_aed": round(estimate.estimate_aed),
            "current_range_80": [round(value) for value in estimate.range_80],
        }
        for name in HORIZONS:
            out[f"forecast_{name}"] = self._horizon(name, features, context)
        out["key_drivers"] = self._drivers(features)
        out["exclusions_applied"] = self._exclusions(area_id, sub_kind)
        out["model_versions"] = {
            "price": estimate.model_version,
            **{
                f"forecast_{name}": version
                for name, (model, version) in self.models.items()
                if model is not None
            },
        }
        return out

    def _features(self, request: PriceRequest, area_id: int, sub_kind: str) -> pl.DataFrame:
        same = (pl.col("area_id") == area_id) & (pl.col("sub_kind") == sub_kind)
        key = match_key(request.building) if request.building else None
        found = self.snapshot.filter(same & (pl.col("building_key") == f"{area_id}|{key}"))
        if key is None or found.height == 0:
            found = self.snapshot.filter(same & pl.col("building_key").is_null())
        if found.height == 0:
            raise PriceInputError(
                "property_kind", f"no sales history for a {request.property_kind} in this area"
            )
        if found["base_level"][0] is None:
            raise PriceInputError(
                "area", "not enough sales in the last 3 months to forecast this area and kind"
            )
        return found.head(1).with_columns(
            pl.lit(float(request.status == "off_plan")).alias("off_plan"),
            pl.lit(math.log(request.size_sqm)).alias("log_area_sqm"),
            pl.lit(request.bedrooms, dtype=pl.Float64).alias("bedrooms"),
            pl.lit(str(area_id)).alias("area_code"),
            pl.lit(request.project, dtype=pl.Utf8).alias("project_code"),
        )

    def _model(self, name: str) -> ForecastModel | None:
        return self.models.get(name, (None, None))[0]

    def _horizon(self, name: str, features: pl.DataFrame, context: dict) -> dict:
        model = self._model(name)
        if model is None:
            gate = self.gates.get(name, {})
            failed = gate.get("status") in ("failed", "insufficient_data")
            reason = gate.get("reason") if failed else f"no registered {name} model"
            return {"status": "not_deployed", "reason": reason}
        growth = float(model.predict_growth(features)[0])
        half = model.half_width(context["segment"])
        point, low, high = (context["estimate"] * math.exp(growth + d) for d in (0.0, -half, half))
        base = context["base"]
        label = confidence(
            base["base_level"],
            base["base_n"],
            context["days"],
            model.area_rows.get(context["area_id"], 0),
            self.config.low_confidence_area_rows,
        )
        block = {"point": round(point), "ci_low": round(low), "ci_high": round(high)}
        block["confidence"] = label
        if label == "LOW":
            block["message"] = low_message(point, low, high)
        return block

    def _drivers(self, features: pl.DataFrame) -> list[str]:
        name = next((h for h in DRIVER_HORIZONS if self._model(h) is not None), None)
        if name is None:
            return []
        contributions = self._model(name).contributions(features)[0][: len(FEATURES)]
        row = features.row(0, named=True)
        drivers = []
        for index in np.argsort(-np.abs(contributions))[:3]:
            value = float(contributions[index])
            if abs(value) < self.config.min_driver_contribution:
                break
            feature = FEATURES[index]
            drivers.append(driver_text(feature, row[feature], value, name))
        return drivers

    def _exclusions(self, area_id: int, sub_kind: str) -> list[str]:
        mine = self.excluded.filter(
            (pl.col("area_id") == area_id) & (pl.col("sub_kind") == sub_kind)
        )
        names = dict(self.areas.iter_rows())
        area_name = names.get(area_id, f"area {area_id}")
        out = []
        for row in excluded_summary(mine):
            kind = f"{row['reg_type'].replace('_', '-')} " if row["reg_type"] else ""
            reason = row["reason"].replace("_", " ") + ("s" if row["count"] != 1 else "")
            out.append(f"Dropped {row['count']:,} {kind}{reason} in {area_name}")
        return out
```

`dict(self.areas.iter_rows())` works because `areas` has exactly two columns: `area_id` and `name_en`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/models/forecast/test_forecast_predict.py`

Expected: every test passes.

**If `test_key_drivers_come_from_shap` finds no driver above 0.005,** the 30-round model is too flat. Raise `rounds` in the test's `small_model(frame)` call (for example, to 100). Do not lower the threshold.

**If `test_unknown_building_falls_back_to_the_area` does not get MEDIUM:** first check that the area-level snapshot row has `base_level == "area"` and `days_since_area_sale <= 180`. Fix the code, not the expectation.

- [ ] **Step 6: Lint, test, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/models/forecast
git add models/forecast/predict.py tests/models/forecast/test_forecast_predict.py tests/models/forecast/forecast_fixtures.py
git commit -m "feat(forecast): Forecaster JSON with ranges, confidence, SHAP drivers and exclusions

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 11: The `train`, `evaluate` and `predict` commands, the end-to-end test and the real run (stop point 3)

**Files:**
- Modify:
  - `models/forecast/__main__.py`
  - `tests/models/forecast/test_forecast_cli.py`

**Interfaces:**

*Consumes:*
- `train.run_training`, `TrainingSummary`
- `model.load_champion`, `latest_gates`
- `evaluate.format_table`, `segment_table`, `top_areas`, `MODEL`
- `baselines.baseline_growth`
- `folds.usable_rows`
- `features.build_dataset`
- `predict.Forecaster`
- `config.HORIZON_SPECS`, `HORIZONS`

*Produces, as commands:*
- **`train [--trials N] [--device auto|cuda|cpu] [--no-register]`**
  - Before training, it runs the same row guard and infrastructure guard as `build`.
  - It prints, per horizon: the status line, the reasons, the test segment table, the test coverage, the mean fold MAPE per model and the timings.
  - It exits 0 when at least one horizon passed, 2 when none did, and 1 on errors.
- **`evaluate`**
  - For each registered champion, it rebuilds the dataset and re-scores the champion on the test period stored in its metadata, `[test_cutoff, test_end)`, alongside both baselines and the stored top areas. It prints the table.
  - For each horizon without a champion, it prints `<h>: not deployed (<reason>)`.
  - It exits 0, or 1 on errors.
- **`predict`**
  - Arguments: `--area | --area-id`, `--kind`, `--status`, `--size`, `[--size-basis] [--bedrooms] [--building] [--project] [--penthouse] [--parking yes|no] [--property-id]`.
  - It prints the JSON.
  - It exits 1 with `Invalid input: ...` on a `PriceInputError`, and 1 with `Forecast failed: ...` on other errors.
- **Module globals** that tests monkeypatch: `run_training`, `load_champion`, `latest_gates`, `Forecaster`.
- `_dataset(config)`: the helper `train` and `evaluate` share. It returns `(frame, report, quality, projects, data_end)` or raises `GuardError(message)`, and applies the drop guard and the infrastructure guard.

- [ ] **Step 1: Write the failing CLI tests**

Append to `tests/models/forecast/test_forecast_cli.py`:

```python
def _fast_config(**overrides):
    from models.forecast.config import ForecastConfig

    values = {
        "n_trials": 1, "n_estimators": 200, "learning_rate": 0.1, "tune_folds": 2,
        "min_test_rows": 50, "n_bootstrap": 200, **overrides,
    }  # fmt: skip
    return lambda: ForecastConfig(**values)


def test_train_end_to_end_then_evaluate_and_predict(monkeypatch, tmp_path, capsys, temp_mlflow):
    import mlflow
    from forecast_fixtures import FakePrice

    from models.forecast import predict as predict_module

    rich = prepared_history(per_building_per_week=4)
    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_rows", fake_load_rows(rich))
    monkeypatch.setattr(cli, "ForecastConfig", _fast_config())
    monkeypatch.setattr(predict_module, "load_rows", fake_load_rows(rich))
    monkeypatch.setattr(predict_module, "load_projects", lambda: project_table(tmp_path))
    monkeypatch.setattr("listings.fraud.load_price_predictor", lambda uri: FakePrice())
    mlflow.create_experiment("price-forecast", artifact_location=temp_mlflow["artifact_location"])

    assert cli.main(["train", "--trials", "1", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "3m: passed" in out
    assert "3y: insufficient_data" in out
    assert "segment" in out and "area_trend" in out
    assert "Registered zestimator-forecast-3m version 1 as @champion" in out

    assert cli.main(["evaluate"]) == 0
    out = capsys.readouterr().out
    assert "3m champion v1" in out
    assert "3y: not deployed (3y: only 0 walk-forward folds (needs 2))" in out

    code = cli.main(
        ["predict", "--area", "Dubai Marina", "--kind", "apartment", "--status", "ready",
         "--size", "80", "--bedrooms", "1", "--building", "Tower 1-0", "--property-id", "p-9"]
    )  # fmt: skip
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["property_id"] == "p-9"
    assert set(result["forecast_3m"]) >= {"point", "ci_low", "ci_high", "confidence"}
    assert result["forecast_3y"]["status"] == "not_deployed"
    assert result["model_versions"]["forecast_3m"] == "1"


def test_train_exits_2_when_nothing_passes(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_rows", fake_load_rows(HISTORY))
    result = SimpleNamespace(
        horizon="3m", status="failed", reasons=("3m resale MAPE 16.0% exceeds the 15% gate",),
        folds=[], table=None, fold_scores=None, coverage={}, upper=None, seconds=1.0,
    )  # fmt: skip
    summary = SimpleNamespace(
        device="cpu", run_id="r1", results={"3m": result}, versions={}, seconds=2.0
    )
    seen = {}

    def fake_run(frame, report, quality, projects, data_end, config, **kwargs):
        seen.update(trials=config.n_trials, kwargs=kwargs)
        return summary

    monkeypatch.setattr(cli, "run_training", fake_run)
    assert cli.main(["train", "--trials", "3", "--device", "cpu", "--no-register"]) == 2
    assert seen == {"trials": 3, "kwargs": {"device": "cpu", "register": False}}
    captured = capsys.readouterr()
    assert "3m: failed" in captured.out
    assert "3m resale MAPE 16.0% exceeds the 15% gate" in captured.out
    assert "No horizon passed its gate; nothing was registered." in captured.err


def test_train_reports_errors(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_rows", fake_load_rows(HISTORY))

    def boom(*args, **kwargs):
        raise RuntimeError("GPU lost")

    monkeypatch.setattr(cli, "run_training", boom)
    assert cli.main(["train"]) == 1
    assert "Training failed: RuntimeError: GPU lost" in capsys.readouterr().err


def test_predict_reports_invalid_input(monkeypatch, tmp_path, capsys):
    from models.price.predictor import PriceInputError

    quiet(monkeypatch, tmp_path)

    class Refuses:
        @classmethod
        def from_registry(cls, settings, config):
            return cls()

        def forecast(self, request):
            raise PriceInputError("area", "unknown area 'Atlantis'")

    monkeypatch.setattr(cli, "Forecaster", Refuses)
    code = cli.main(["predict", "--area", "Atlantis", "--kind", "apartment", "--status", "ready",
                     "--size", "100"])  # fmt: skip
    assert code == 1
    assert "Invalid input: area: unknown area 'Atlantis'" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/models/forecast/test_forecast_cli.py`

Expected: FAIL. The parser rejects `train`, `evaluate` and `predict`.

- [ ] **Step 3: Extend __main__.py**

Add these imports:

```python
from models.forecast.baselines import baseline_growth
from models.forecast.config import HORIZON_SPECS, HORIZONS
from models.forecast.evaluate import MODEL, format_table, segment_table
from models.forecast.folds import usable_rows
from models.forecast.model import latest_gates, load_champion
from models.forecast.predict import Forecaster
from models.forecast.train import run_training
from models.price.predictor import PriceInputError
```

Add these constants:

```python
KINDS = ("apartment", "hotel_apartment", "townhouse", "villa")


class GuardError(RuntimeError):
    """A stop-point guardrail refused the data."""
```

In `_parser()`, add:

```python
    train = commands.add_parser("train", help="tune, evaluate, gate and register each horizon")
    train.add_argument("--trials", type=int, default=DEFAULTS.n_trials)
    train.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    train.add_argument("--no-register", action="store_true")
    commands.add_parser("evaluate", help="re-score the registered champions on their test periods")
    predict = commands.add_parser("predict", help="forecast one home as JSON")
    area = predict.add_mutually_exclusive_group(required=True)
    area.add_argument("--area")
    area.add_argument("--area-id", type=int)
    predict.add_argument("--kind", choices=KINDS, required=True)
    predict.add_argument("--status", choices=("ready", "off_plan"), required=True)
    predict.add_argument("--size", type=float, required=True, help="size in m²")
    predict.add_argument("--size-basis", choices=("built_up", "plot"), default="built_up")
    predict.add_argument("--bedrooms", type=int)
    predict.add_argument("--building")
    predict.add_argument("--project")
    predict.add_argument("--penthouse", action="store_true")
    predict.add_argument("--parking", choices=("yes", "no"))
    predict.add_argument("--property-id")
```

Register `"train": _train`, `"evaluate": _evaluate` and `"predict": _predict` in `handlers`.

Refactor `_build` to use a shared `_dataset` helper, so all three commands apply the same guards. Replace `_build` with:

```python
def _dataset(config: ForecastConfig, stages: Stages):
    with stages.stage("load_rows"):
        rows, quality, data_end, areas, _ = load_rows(_settings(), config)
    _print(quality.lines())
    if quality.drop_share > config.max_drop_share:
        raise GuardError(
            f"{quality.drop_share:.1%} of rows were dropped, above the "
            f"{config.max_drop_share:.0%} limit"
        )
    projects = load_projects()
    problems = validate_projects(projects, set(areas["area_id"].to_list()))
    if problems:
        raise GuardError(
            "the infrastructure table has problems:\n" + "\n".join(f"  {p}" for p in problems)
        )
    with stages.stage("dataset"):
        frame, report = build_dataset(rows, data_end, projects)
    return frame, report, quality, projects, data_end


def _build(args: argparse.Namespace) -> int:
    config = dataclasses.replace(ForecastConfig(), sample_rows=args.sample, seed=args.seed)
    stages = Stages("build")
    try:
        frame, report, quality, _, data_end = _dataset(config, stages)
    except GuardError as exc:
        print(f"Build stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    coverage = feature_coverage(frame, FEATURES)
    _print(report.lines())
    print("Feature coverage (non-null share):")
    _print(f"  {name:<28}{share:>7.1%}" for name, share in coverage.items())
    write_json(
        "quality.json",
        {
            "data_end": data_end.isoformat(),
            "sample_rows": config.sample_rows,
            "seed": config.seed,
            "quality": quality.to_dict(),
            "excluded": excluded_summary(quality.excluded),
            "targets": report.to_dict(),
            "feature_coverage": coverage,
        },
    )
    stages.save()
    return 0
```

Check that the build tests still hold:
- The guard messages keep their wording: `"Build stopped: 50.0% of rows were dropped, above the 30% limit"`, and `"Build stopped: the infrastructure table has problems:"` followed by the indented problems.
- The build timings are still `{"load_rows", "dataset"}`.

Add the three handlers:

```python
def _train(args: argparse.Namespace) -> int:
    config = dataclasses.replace(ForecastConfig(), n_trials=args.trials)
    stages = Stages("train")
    try:
        frame, report, quality, projects, data_end = _dataset(config, stages)
        with stages.stage("training"):
            summary = run_training(
                frame, report, quality, projects, data_end, config,
                device=args.device, register=not args.no_register,
            )  # fmt: skip
    except GuardError as exc:
        print(f"Training stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Device {summary.device}; MLflow run {summary.run_id}")
    for name, result in summary.results.items():
        print(f"{name}: {result.status} ({len(result.folds)} folds, {result.seconds:,.0f}s)")
        _print(f"  {reason}" for reason in result.reasons)
        if result.table is not None:
            _print(format_table(name, result.table))
            coverage = ", ".join(f"{k} {v:.1%}" for k, v in result.coverage.items())
            print(f"  80% range coverage on test: {coverage}")
            means = result.fold_scores.group_by("model").agg(pl.col("mape").mean()).sort("model")
            print("  mean fold MAPE: " + ", ".join(f"{m} {v:.2%}" for m, v in means.iter_rows()))
    for name, version in summary.versions.items():
        print(f"Registered {config.model_prefix}-{name} version {version} as @champion")
    stages.save()
    if not any(result.status == "passed" for result in summary.results.values()):
        print("No horizon passed its gate; nothing was registered.", file=sys.stderr)
        return 2
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    config = ForecastConfig()
    stages = Stages("evaluate")
    try:
        champions = {name: load_champion(f"{config.model_prefix}-{name}") for name in HORIZONS}
        gates = latest_gates(config.experiment)
        frame = None
        if any(model is not None for model, _ in champions.values()):
            frame = _dataset(config, stages)[0]
        for name, (model, version) in champions.items():
            if model is None:
                reason = gates[name]["reason"] or f"no registered {name} model"
                print(f"{name}: not deployed ({reason})")
                continue
            horizon = HORIZON_SPECS[name]
            start = date.fromisoformat(model.metadata["test_cutoff"])
            end = date.fromisoformat(model.metadata["test_end"])
            test = usable_rows(frame, horizon).filter(
                pl.col("instance_date").is_between(start, end, closed="left")
            )
            predictions = {MODEL: model.predict_growth(test), **baseline_growth(test, horizon)}
            actual = test[f"growth_{name}"].to_numpy()
            table = segment_table(test, predictions, actual, model.metadata["top_areas"])
            print(f"{name} champion v{version}: test period {start} to {end}, {test.height:,} rows")
            _print(format_table(name, table))
    except GuardError as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    stages.save()
    return 0


def _predict(args: argparse.Namespace) -> int:
    request = {
        "property_id": args.property_id,
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
    }
    try:
        forecaster = Forecaster.from_registry(_settings(), ForecastConfig())
        result = forecaster.forecast({k: v for k, v in request.items() if v is not None})
    except PriceInputError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Forecast failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
```

Add `from datetime import date` to the imports.

**Note on the end-to-end test.** It monkeypatches `ForecastConfig` in `cli` with a zero-argument factory. `_train` calls `dataclasses.replace(ForecastConfig(), ...)`, which works on the instance the factory returns. The parser takes its defaults from `DEFAULTS` (Task 4), so the monkeypatch never reaches argument parsing.

**Windows console.** `predict` prints `±` and `²`. Phase 5 hit a cp1252 console crash on output like this. Apply the same fix: at the start of `main()`, call `sys.stdout.reconfigure(encoding="utf-8")` when `sys.stdout` has a `reconfigure` method, and do the same for `sys.stderr`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/models/forecast`

Expected: every test passes. The end-to-end test takes about 1–2 minutes on the CPU.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add models/forecast/__main__.py tests/models/forecast/test_forecast_cli.py
git commit -m "feat(forecast): train, evaluate and predict commands with an end-to-end test

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: The real run (stop point 3)**

Run from Git Bash. MLflow uses the project's `.env` tracking URI; never write to `./mlruns`.

```bash
uv run python -m models.forecast train --device cuda 2>&1 | tee data/forecast/train_output.txt
uv run python -m models.forecast evaluate 2>&1 | tee data/forecast/evaluate_output.txt
uv run python -m models.forecast predict --area "Dubai Marina" --kind apartment --status ready --size 85 --bedrooms 1 | tee data/forecast/predict_example.json
```

- **Expected runtime:** 20–60 minutes for `train` on the GPU. Run it in the foreground with a long timeout, or in the background, and poll no more often than every 10 minutes.
- **Expected outcome:** the 3y horizon is `insufficient_data` (ruling 5). The 3m and 1y horizons pass or fail on their merits. Both outcomes are acceptable results; a failure is reported, not tuned away.
- **Report:** put the full `train` and `evaluate` outputs, verbatim, in your report, along with the MLflow run id and the example JSON.
- **Then STOP** and report `DONE`, or `DONE_WITH_CONCERNS` if any horizon failed. The controller shows the segmented tables to the user before Task 12.
- **If `train` exits 1,** report `BLOCKED` with the output.

---

## Task 12: README section and spec amendments

**Files:**
- Modify:
  - `README.md` (add `## Price forecasting` after `## Property search`; update `## Architecture (current)` and `## Module layout`)
  - `docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md` (fill "Amendments during implementation")

**Interfaces:**
- Consumes: the real outputs of the Task 11 run, which are in `data/forecast/` and the MLflow run, plus the ledger's rulings, which the controller passes in the dispatch.
- Produces: documentation only.

- [ ] **Step 1: Write the README section**

Add `## Price forecasting` with these subsections. Every number must be copied from `data/forecast/train_output.txt`, `evaluate_output.txt`, `quality.json` or `stage_timings.json`, with the MLflow run id cited once.
1. **What it answers.** Three horizons, each as the current estimate × e^growth, with a conformal 80% range and a confidence label. Say plainly that a horizon that fails its gate is not served.
2. **Data and limits.**
   - The data ends 2023-03-17.
   - Give the last usable T per horizon, from `quality.json`.
   - Explain why the 3y horizon is `insufficient_data`, and that a newer DLD file (the Kaggle mirror or data.dubai) fixes it by re-running Phases 2–6.
   - Infrastructure is mapped at area level; DLD has no coordinates.
   - The planned completion date is the one stated at announcement.
3. **Rows and exclusions.** Each drop reason with its real count, the 30% guard, the outlier rule, and the target statuses (`no_base`, `no_target`, `window_open`) with their real counts.
4. **Features.** A table with one row per feature group, giving its as-of rule. Name the forbidden columns and say why (`nearest_metro` and `nearest_mall` are snapshots).
5. **Validation.** The walk-forward fold plan per horizon (fold counts and test periods), the target-window guard, and the tuning folds.
6. **Results.** For each horizon: the test segment table exactly as printed, the baselines, the gate verdict and its reasons, the coverage, and the mean fold MAPE. There is no single overall accuracy number.
7. **Example.** The example JSON from `predict_example.json`.
8. **Commands.** `build`, `infra-check`, `train`, `evaluate`, `predict`.
9. **Infrastructure table.** How many projects, how many were announced after the data end, and a pointer to the CSV and its sources.

Update `## Architecture (current)` and `## Module layout` so they list `models/forecast/`, in the same style as the existing entries for `search/`.

- [ ] **Step 2: Write the spec amendments**

Replace `(none yet)` under "Amendments during implementation" with a bulleted list. Each bullet gives the change and why:
- The day-count windows.
- The median of ppsm, then ln.
- The `window_open` status.
- Tuning on the last 4 folds.
- The fixed-round final model.
- Categories fitted on the test-fold training rows.
- Early stopping and scoring sharing a fold's validation rows.
- Ranges covering growth uncertainty only.
- `as_of` fixed to the data end.
- The not-deployed reasons.
- Any ruling the ledger added during execution. The controller lists these in the dispatch.

- [ ] **Step 3: Check, commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add README.md docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md
git commit -m "docs(forecast): README price-forecasting section and spec amendments

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
