# Phase 3 — Price Estimation Model

Status: design approved in chat 2026-09-15; awaiting written-spec review
Date: 2026-09-15
Part of: Dubai Real Estate ML Platform (sub-project 3 of 10)
Builds on: `docs/superpowers/specs/2026-09-15-phase2-ingestion-design.md`

## Goal

Train a model that estimates the fair market price of a Dubai home from its
location, type, size and status. Evaluate it honestly on a held-out future
period against a comps baseline, attach calibrated price ranges so it can flag
mis-priced listings, log everything to MLflow, and register the model as one
self-contained artifact (`models:/zestimator-price@champion`) that the Phase 6
API can load and call without a database.

## Decisions (user, 2026-09-15)

- **Scope: homes only.** Apartments (`unit` / `Flat`), hotel apartments
  (`unit` / `Hotel Apartment`), stacked townhouses (`unit` /
  `Stacked Townhouses`), villas (`villa`, any sub-type). Offices, shops, hotel
  rooms, other commercial units, land and whole buildings are out of scope.
- **Window: sales from 2015-01-01 on, recency-weighted.** This skips the
  2008 crash and the 2009–2010 pre-registration spike (66k
  `Sell - Pre registration` sales at ~1.5× the 2008 price per m²).
- **Off-plan and ready: one model**, with the status as a feature and
  every metric also reported per segment.
- **Approach A:** hierarchical location priors plus a market index as
  features, XGBoost on the GPU tuned with Optuna. The comps rule (B0) and
  LightGBM (B1) are logged as baselines.

## Data (profiled 2026-09-15, homes from 2015 on)

| Split | Period | Clean sales | Statistically excluded* |
|---|---|---|---|
| train | 2015-01-01 → 2022-06-30 | 260,078 | 5,550 |
| val | 2022-07-01 → 2022-10-31 | 27,543 | 454 |
| test | 2022-11-01 → 2023-03-17 | 37,804 | 539 |

\* Rows excluded in Phase 2 only by `price_outlier_low`,
`price_outlier_high` or `suspected_sqft_entry`. They are used only in the
honest test set (see Evaluation).

Facts the design depends on:
- Villa `area_sqm` is the **plot** when `property_sub_type` is null
  (median 512 m²) and the **built-up** area when it is `Villa`
  (median 244 m²).
- `building_name` is present for every unit and absent for every villa;
  `project_name` is present for 77% of units and about 60% of villas.
- 124 areas have home sales since 2015; 38 of them have fewer than 30.
- Bulk groups (identical sales: same date, location, type and price)
  cover about 112k rows; the largest has 94 members.
- Some clean sales have implausible sizes (as small as 0.5 m²), so the
  training scope applies the same size bounds the predictor enforces.
- The data ends on 2023-03-17 (`DATA_END`, read from the data and asserted,
  never hard-coded into predictions).

## Architecture

```
models/price/
  __init__.py
  __main__.py     CLI: python -m models.price train|predict   (load_dotenv first)
  config.py       TrainConfig — every constant below, overridable in tests
  data.py         load_homes(settings, config) -> HomesData   (scope, bounds, lineage)
  features.py     derive_segments(), MarketIndex, LocationPriors, build_features()
  split.py        assign_split(), bulk_group_ids(), sample_weights()
  baselines.py    comps_b0(), lightgbm_b1()
  tune.py         tune_xgboost(...) -> TuneResult   (Optuna, GPU or CPU)
  evaluate.py     metrics, slices, conformal quantiles, coverage
  predictor.py    PriceRequest, PriceEstimate, PriceInputError, PricePredictor
  pyfunc.py       PricePyfunc (mlflow.pyfunc.PythonModel wrapping PricePredictor)
  registry.py     log runs, log/register the model, move the champion alias
  train.py        run_training(settings, config) -> TrainingSummary (orchestration)
scripts/build_price_fixture.py
tests/fixtures/price_sample.csv
tests/models/price/...
```

`models/__init__.py` (the Phase 1 placeholder) stays. Training runs on the
host through `uv run`, not in Airflow. A retraining DAG and schedule are
Phase 8.

### Environment changes

- Pin the project to Python 3.11 (`uv python pin 3.11`; `requires-python`
  stays `>=3.11`). The venv currently runs 3.13; the Airflow image and the
  MLflow server's era are both 3.11.
- Pin `mlflow==2.17.2` on the client to match the server.
- New dependencies: `xgboost` (the Windows wheel includes CUDA), `lightgbm`,
  `optuna`, `scikit-learn`, `pandas`, `pyarrow`, `matplotlib`, `pydantic`
  (v2). Exact versions are pinned in `pyproject.toml` and `uv.lock` when they
  are added.

## Scope and cleaning (data.py)

`load_homes` reads from Postgres (`DbSettings.from_env()`) and returns:
- `sales`: clean in-scope sales (`exclusion_reason IS NULL`) with
  `instance_date >= 2014-01-01`. Sales from 2014 are used **only** to seed
  the market index and never as training rows.
- `stat_excluded`: in-scope rows with an `exclusion_reason` of
  `price_outlier_low`, `price_outlier_high` or `suspected_sqft_entry`, and
  `instance_date >= 2022-11-01`.
- `lineage`: `ingest_run_id` (must be a single value across the rows) and
  that run's `source_sha256` from `dld.ingestion_runs`.
- `drop_counts`: the rows dropped by each rule below, per segment.

The SQL select list is explicit. It never selects `price_per_sqm_aed`,
`price_robust_z`, `peer_tier` or `source_row`, and `exclusion_reason` is
used only to separate the two frames.

Derived fields (features.py `derive_segments`, pure):
- `size_basis`: `plot` for villas with a null sub-type; `built_up` otherwise.
- `sub_kind`: `flat` | `hotel_apartment` | `townhouse` | `villa`.
- `room_kind` and `bedrooms` from `rooms`:
  - `Studio` → `studio`, 0 bedrooms
  - `N B/R` → `bedrooms`, N bedrooms
  - `Penthouse` → `penthouse`, bedrooms null
  - `Single Room` → `single_room`, bedrooms null
  - null or anything else → `unknown`, bedrooms null
- `segment` = `{property_type}_{reg_type}_{size_basis}`, for example
  `unit_ready_built_up`.

Both frames get the same derivations and drop rules. Rows dropped from
scope, each counted:
1. **Size bounds** (m², inclusive), the same bounds the predictor enforces:

   | Kind | Min | Max |
   |---|---|---|
   | flat, hotel apartment, townhouse (built-up) | 12 | 2,000 |
   | villa, built-up | 40 | 3,000 |
   | villa, plot | 60 | 20,000 |
2. **Unsupported segment:** any segment with fewer than 200 train-period
   rows. Today that is only `villa_off_plan_plot` (2 rows). The supported
   segment list is saved with the model, and the predictor rejects the rest.

## Split and weights (split.py)

- Split by `instance_date`:
  - train: 2015-01-01 ≤ d < 2022-07-01
  - val: 2022-07-01 ≤ d < 2022-11-01
  - test: 2022-11-01 ≤ d ≤ `DATA_END`

  Each bulk group shares one date, so no group can straddle two splits;
  a test asserts this.
- **Bulk group id:** rows sharing
  (`instance_date`, `area_id`, `project_name`, `building_name`,
  `property_type`, `property_sub_type`, `price_aed`), with nulls compared as
  equal. `group_size` is the group's size.
- **Bulk weight** = 1 / `group_size`: a bulk sale is one pricing decision.
- **Recency weight** = 0.5 ^ (age_days / 730.5), where age is measured from
  the last date of the data being fitted (2-year half-life).
- **Sample weight** = recency weight × bulk weight. It is used for model
  fitting and early stopping, not for the reported metrics.

## Target and features (features.py)

### Market index

`I(segment, month)` is the median of `ln(price_aed / area_sqm)` over clean
in-scope sales of that segment in calendar months [m−3, m−1], i.e. strictly
before month m. When the window has fewer than 30 sales:
1. widen to [m−6, m−1], then to [m−12, m−1];
2. if still short, pool over `reg_type` (same `property_type` and
   `size_basis`), using the same window sequence;
3. if still short, carry forward the segment's last defined value.

A training row's own month never feeds its index. Index values for the
val and test months are computed from all clean sales before that month,
including earlier val and test months. That is legitimate past market data,
the way a published index would be, and it never includes the row itself.
For serving, the as-of month is the month containing `DATA_END`.

### Target

`y = ln(price_aed / area_sqm) − I(segment, month)`: the premium over the
current market level. The price is recovered as
`exp(ŷ + I) × area_sqm`. The tree model never has to extrapolate a price
level it hasn't seen.

### Location priors (LocationPriors)

Levels and keys (every key includes `property_type`):
- level 0, city: `property_type`
- level 1, area: + `area_id`
- level 2, project: + `match_key(project_name)`; absent when the project is
  null
- level 3, building: + `match_key(building_name)`; absent when the
  building is null

`match_key` is Phase 2's `ingestion.normalize.match_key`.

The fit stores, per key, the bulk-weighted sum of `y` and the effective
count `n_eff` (the sum of bulk weights). Transform shrinks each row down
the chain:
- `est_city = weighted mean y` for the property type
- `est_area = (n·mean + K·est_city) / (n + K)`
- `est_project = (n·mean + K·est_area) / (n + K)`, or `est_area` when the
  project is absent or unseen
- `est_building = (n·mean + K·parent) / (n + K)`, where `parent` is
  `est_project` if the row has a project and `est_area` otherwise; or
  `parent` itself when the building is absent or unseen

`K = 10` (`TrainConfig.shrink_k`). `loc_level` is the deepest level whose
`n_eff ≥ 3` (`min_level_n`), otherwise 0.

Leak-proofing:
- **Training rows** get out-of-fold priors: 5-fold `GroupKFold` on the bulk
  group id, each fold transformed using statistics fitted on the other four.
  Identical sales never see each other's price.
- **Val, test and stat-excluded rows** (evaluation model) get priors fitted
  on all training rows.
- **Production model:** out-of-fold for its training rows. The serving
  tables are fitted on all its data.

### Feature allowlist (exact, in order; a test asserts equality)

| Feature | Type |
|---|---|
| `property_type` | categorical |
| `reg_type` | categorical |
| `size_basis` | categorical |
| `sub_kind` | categorical |
| `room_kind` | categorical |
| `bedrooms` | float (NaN when unknown) |
| `log_area_sqm` | float |
| `has_parking` | float 0/1 (NaN when unknown) |
| `area_id` | categorical |
| `market_index` | float |
| `prior_area` | float |
| `prior_project` | float |
| `prior_building` | float |
| `n_area` | float, `log1p(n_eff)`, 0 when unseen |
| `n_project` | float, `log1p(n_eff)`, 0 when unseen |
| `n_building` | float, `log1p(n_eff)`, 0 when unseen |
| `loc_level` | int 0–3 |

Categoricals use pandas `Categorical` with category lists fixed at fit time
and saved with the model. Values unseen at inference become missing, which
XGBoost's native categorical support handles.

**Never features:** `price_per_sqm_aed`, `price_robust_z`, `peer_tier`,
`exclusion_reason`, `source_row`, `ingest_run_id`, `procedure_name`
(a person estimating a price wouldn't know it), `price_aed` itself, and
`nearest_metro`, `nearest_mall`, `nearest_landmark` (out of scope; the
location priors capture them).

## Training (tune.py, baselines.py, train.py)

Each item below is an MLflow run in experiment `price-model`:
1. **`b0-comps`:** `ŷ` = weighted median of training `y` over
   (`area_id`, `segment`, room bucket), where the room bucket is
   `bedrooms` for the `studio` and `bedrooms` kinds and `room_kind`
   otherwise, requiring `n_eff ≥ 5`. It falls back to (`area_id`,
   `segment`), then to `segment`. The price comes from the same index
   formula, so B0 is an index-adjusted comps estimate.
2. **`b1-lightgbm`:** LightGBM on the CPU, same features and weights,
   `learning_rate=0.05`, up to 4,000 rounds, early stopping after 100
   rounds on val, other parameters at their defaults.
3. **`xgb-tune`** (parent run; each trial is a nested child run):
   - Optuna `TPESampler(seed=42)`, `TrainConfig.n_trials` trials (60 by
     default), minimising the weighted RMSE of `y` on val.
   - XGBoost `tree_method="hist"`, `device` from `--device`: `auto` picks
     `cuda` when XGBoost has CUDA support and a GPU is visible, `cpu`
     otherwise.
   - Up to 4,000 boosting rounds, early stopping after 100.
   - Search space:
     - `max_depth` 4–12
     - `learning_rate` 0.02–0.3 (log scale)
     - `min_child_weight` 1–64 (log scale)
     - `subsample` 0.6–1.0
     - `colsample_bytree` 0.5–1.0
     - `reg_lambda` 1e-3–10 (log scale)
     - `reg_alpha` 1e-3–10 (log scale)
4. **`xgb-champion-eval`:** the best parameters, trained on train with early
   stopping on val, then scored once on test. This is the run that decides
   acceptance.
5. **`xgb-production`:** the same parameters and the eval model's
   `best_iteration`, refit on train + val + test through `DATA_END`. The
   priors, index, bounds and percentiles are refit on the same data. This
   is the version that gets registered.

**Acceptance gate:** `xgb-champion-eval`'s `test_clean.mdape` must be at most
0.90 × `b0-comps`' `test_clean.mdape`. If it isn't, nothing is registered
and the CLI exits with code 2 and a message.

## Evaluation (evaluate.py)

Metrics, computed on prices and unweighted:
- `APE = |pred − actual| / actual`
- `mdape` = median APE
- `ppe10` = share with APE ≤ 0.10
- `ppe20` = share with APE ≤ 0.20
- `rmse_log` = RMSE of ln(price)

Evaluation sets:
- `val` (every model)
- `test_clean`: test-period clean sales
- `test_honest`: `test_clean` plus the stat-excluded rows. The Phase 2
  outlier filter used the price, so `test_clean` alone flatters every
  model.

Slices, logged as `{set}.{slice}.{metric}` (for example
`test_clean.segment.villa_ready_plot.mdape`):
- `all`
- `segment.<segment>`
- `loc_level.<0-3>`
- `dedup`: one row per bulk group

**Prediction intervals** (split conformal, evaluation model):
- On val, take `e = |ŷ − y|`, the absolute log error.
- For each supported segment with at least 200 val rows,
  `q_α` = the ⌈(n+1)(1−α)⌉-th smallest `e`, for α = 0.20 (80%) and
  α = 0.05 (95%). Segments with fewer val rows use the pooled quantile.
- Interval = `exp(ŷ + I ± q) × area_sqm`.
- Test coverage (the share of actual prices inside the interval) is logged
  overall and per segment. Coverage is reported, not gated.
- The production model reuses the evaluation model's quantiles. This is
  conservative, because the refit's errors are no larger.

**Artifacts** logged on `xgb-champion-eval` (matplotlib, Agg backend):
- `feature_importance.png` (gain)
- `residuals_by_segment.png`
- `error_by_loc_level.png`
- `pred_vs_actual.png` (log–log hexbin)
- `metrics.json`, the full metric table

## Predictor (predictor.py, pyfunc.py)

`PricePredictor.from_dir(path)` loads a model directory, so tests and the
API need no MLflow. It contains:
- `booster.json`
- `tables/*.parquet`: prior stats per level, market index, plausibility
  bounds, size percentiles, the alias table snapshot and area names
- `metadata.json`: feature list, category lists, supported segments,
  conformal quantiles, `DATA_END`, lineage, hyperparameters and training
  metrics

`PricePyfunc` wraps it for MLflow. It is logged with
`code_paths=["models", "ingestion"]`, and the API calls
`predictor.predict_one(PriceRequest)`.

### PriceRequest (pydantic v2)

| Field | Type | Rule |
|---|---|---|
| `area` | str, optional | name or alias, resolved through the alias snapshot by `match_key` |
| `area_id` | int, optional | exactly one of `area` / `area_id` is required |
| `project` | str, optional | |
| `building` | str, optional | |
| `property_kind` | `apartment` \| `hotel_apartment` \| `townhouse` \| `villa` | required |
| `status` | `ready` \| `off_plan` | required |
| `size_sqm` | float | > 0 and within the size bounds above |
| `size_basis` | `built_up` \| `plot` | default `built_up`; `plot` only for a villa |
| `bedrooms` | int 0–8, optional | 0 means studio |
| `is_penthouse` | bool | default false; only for an apartment or hotel apartment |
| `has_parking` | bool, optional | |
| `asking_price_aed` | float > 0, optional | |

A request raises `PriceInputError(field, message)` (a `ValueError`) when:
- a field-level rule fails (pydantic validation, re-raised as
  `PriceInputError`)
- the area is unknown or ambiguous
- the `area_id` isn't in the area snapshot (`dld.areas`). A known area with
  no training sales is allowed; it resolves at city level with low
  confidence
- the combination is unsupported (a segment not in the supported list,
  for example villa + off-plan + plot)
- `size_basis=plot` is given for a non-villa
- `is_penthouse` is given for a villa or townhouse

Phase 6 maps this error to HTTP 422.

### PriceEstimate

| Field | Meaning |
|---|---|
| `estimate_aed` | the price estimate |
| `range_80`, `range_95` | `(low, high)` in AED |
| `price_per_sqm_aed` | output only |
| `location_level` | `building` \| `project` \| `area` \| `city` |
| `confidence` | `high` (building or project), `medium` (area), `low` (city) |
| `market_label` | `below_market` \| `fair` \| `above_market` (null without an asking price), from the 80% range |
| `asking_vs_estimate_pct` | the asking price's difference from the estimate |
| `flags` | list of strings, below |
| `as_of` | `DATA_END` |
| `model_version` | the registered version |

Guardrails (flags):
- **`implausible_clipped`:** `ŷ` is clipped to [p1 − ln 3, p99 + ln 3] of
  training `y`, taken per (`property_type`, `size_basis`, `area_id`) when
  the area has `n_eff ≥ 30` and per (`property_type`, `size_basis`)
  otherwise. So a price can land at most 3× beyond the normal range, and
  it is never negative, since the model works in logs.
- **`unusual_size`:** the size is outside the segment's p0.5–p99.5 in the
  training data (the request is still answered).
- **`location_fallback`:** a given project or building wasn't found, so a
  coarser level was used.

## CLI (__main__.py)

- `python -m models.price train [--trials N] [--device auto|cuda|cpu] [--no-register]`
  - calls `load_dotenv()` before anything touches the database
  - prints a metric table for every model and the registered version
  - exit codes: 0 success, 1 error, 2 acceptance gate failed
- `python -m models.price predict --area "JVC" --kind apartment --status ready --size 75 [--bedrooms 1] [--asking 900000]`
  - loads `models:/zestimator-price@champion` and prints the estimate as
    JSON, for manual checks and the README

## Testing

Unit tests, at least one per master-prompt edge case:
- **Leakage:**
  - The feature columns equal the allowlist exactly, and the loader's
    SQL never selects a forbidden column.
  - Perturbing `price_aed` for val and test rows leaves their features
    unchanged.
  - The market index for month m is unchanged when prices in month m and
    later are perturbed.
  - Two identical bulk-sale rows with an extreme price, in a building with
    other sales: neither row's out-of-fold `prior_building` reflects the
    extreme price.
- **Unseen or sparse locations:**
  - An unseen building takes its parent's value, and `loc_level` drops.
  - An area with `n_eff < 3` resolves to level 0.
  - The shrinkage arithmetic matches hand-computed values.
  - "JVC", "Dubai Marina" and "JBR" resolve through the alias snapshot;
    an unknown area name raises `PriceInputError`.
- **Implausible predictions:** a booster stub that returns an extreme `ŷ`
  gives a clipped estimate with `implausible_clipped`, and the estimate is
  always > 0.
- **Invalid input:** each rejection rule above has a test. `unusual_size`
  is flagged, not rejected.
- **Off-plan and ready:** `reg_type` is in the feature set, and the
  metrics include one slice per segment.
- **Derivations:** `size_basis`, `sub_kind`, `room_kind` and `bedrooms`
  for every `rooms` value, including `Office` → `unknown`.
- **Split and weights:**
  - Split boundaries are exact.
  - No bulk group spans two splits.
  - The recency weight is 0.5 at 730.5 days.
  - The bulk weight is 1/c.
- **Scope:** offices, shops, land and buildings are excluded; size-bound
  and unsupported-segment drops are counted per segment.
- **Metrics:** MdAPE, PPE10, PPE20, RMSE, the conformal quantile with its
  finite-sample rank, and coverage, all against hand-computed values.
- **B0 fallback chain:** each fallback level is exercised.
- **Market labels:** below, fair and above at the interval edges;
  `as_of == DATA_END`.
- **CLI:** `main()` loads `.env` before building `DbSettings` (the
  port-5432 lesson).

Integration:
- `tests/fixtures/price_sample.csv`: about 3,000 real DLD rows, built by
  `scripts/build_price_fixture.py` (seed 42, stratified so every split and
  every supported segment has rows). Phase 2 reclassifies the smaller file,
  so the test doesn't depend on how many rows end up stat-excluded.
- The test loads the fixture into the throwaway Postgres database
  (`pg_test_db`) with Phase 2's `run_pipeline`, then runs `run_training`
  with a scaled-down `TrainConfig`:
  - 2 trials, CPU
  - a lower segment minimum
  - the gate disabled
  - MLflow on a temporary SQLite store
- It checks the registered model loads through `mlflow.pyfunc.load_model`
  and `predict_one` returns a positive estimate.
- A GPU smoke test trains a tiny booster with `device="cuda"`; it is
  skipped when CUDA isn't available.

## Deliverables beyond code

- README:
  - a "Price model" section: how to train and predict, the real test
    metrics (clean and honest, per segment, versus B0 and B1), and GPU
    versus CPU training time
  - the half-page write-up the master prompt asks for: the business
    problem, the metric optimised, and what would change with production
    data
- The Phase 2 spec's stale formula line is fixed (done alongside this
  spec).

## Out of scope

- API endpoints (Phase 6)
- A retraining DAG and schedule (Phase 8)
- Drift monitoring (Phase 9)
- Commercial units, land and buildings
- The nearest-metro, mall and landmark features
- Floor, view and unit-level features (not in the DLD data)
- Estimates for dates after `DATA_END`
