# Phase 6 — Multi-Horizon Price Forecasting

- **Status:** design approved in chat on 2026-09-16; the written spec is awaiting review.
- **Date:** 2026-09-16
- **Part of:** Dubai Real Estate ML Platform (sub-project 6 of 11)
- **Builds on:**
  - `docs/superpowers/briefs/2026-09-16-price-forecasting-brief.md` (the user's request, adapted)
  - `docs/superpowers/specs/2026-09-15-phase3-price-model-design.md`
  - `docs/superpowers/specs/2026-09-15-phase2-ingestion-design.md`

## Goal

Buyers and investors ask: "What will this property be worth in 3 months, 1 year and 3 years?" Phase 6 answers with three separate XGBoost models. Each model predicts **market growth** over its horizon. A forecast is Phase 3's current estimate × e^(predicted growth).

Every forecast comes with an 80% range and a confidence label. Validation is walk-forward only. A horizon is served only if it passes its own accuracy gate and beats two simple baselines. Otherwise it is published as `not_deployed`, with its numbers.

There is no UI, no LLM explanation layer, and no network access during training.

## Decisions (user, 2026-09-16)

- **Placement:** this is a new Phase 6, between search (5) and the API (now 7). The later phases were renumbered to API 7, Demo 8, CI/CD 9, Deploy and monitor 10, Docs 11.
- **Infrastructure geography:** each project is mapped by hand to the DLD areas it affects.
  - DLD has no coordinates, so the features are area-level counts, not distances.
  - Area centroids and geocoding were declined.
- **Data:** use the DLD data already loaded, which ends 2023-03-17.
  - A newer file exists: the Kaggle mirror was reportedly updated on 2026-02-03 with about 1.51M rows.
  - The user was given the links. If they drop a new `data/raw/Transactions.csv`, Phases 2–6 re-run on it. This phase needs no code change for that, unless DLD changed its columns.
- **Gates:** MAPE ≤ 15% for 3 months, ≤ 20% for 1 year and ≤ 30% for 3 years, measured on the primary segments. Each horizon must also beat both baselines (see Gate).
- **Approach A: growth per sale.**
  - Predicting the future price per m² directly (approach B) was declined.
  - Area-level series (approach C) were declined.

## Deviations from the original request (recorded in the brief)

These are the changes from the user's original wording:
- Floor, total floors, view and unit are dropped, because DLD has none of them.
- The building age is a proxy.
- DLD's `nearest_metro` and `nearest_mall` fields are excluded. They are present-day snapshots and would leak into older rows.
- The infrastructure table gains `announced_date` and `source_url`.
- "Metro Line 3" is replaced by official project names.
- MAPE is reported together with median APE.
- Gates are set per horizon.
- The code lives in a package, `models/forecast/`, rather than four loose scripts.

## Architecture

The new package `models/forecast/` reuses Phase 2 cleaning and Phase 3 data loading, prediction and registry code.

| Module | Job |
|---|---|
| `config.py` | `ForecastConfig`, `HORIZONS`, `FEATURES`, `FORBIDDEN_FEATURES`, gates |
| `rows.py` | Load home sales (via `models.price.data.load_homes`), row validation, dedup, area-month robust outliers, drop report |
| `targets.py` | Base price, target windows, building→area fallback, per-horizon exclusion |
| `infra.py` | Load and validate `reference/infrastructure_projects.csv`; as-of area features |
| `features.py` | Momentum, base context, property, location and infrastructure features, all as of T |
| `folds.py` | Walk-forward folds with the target-window guard |
| `baselines.py` | No-change and area-trend growth |
| `train.py` | XGBoost per horizon, Optuna on folds, final fit, SHAP contributions |
| `intervals.py` | Split-conformal 80% ranges per horizon and segment; confidence labels |
| `evaluate.py` | Segmented MAPE and median APE, coverage, baselines, gates, MLflow logging |
| `predict.py` | `forecast(property) -> dict` (the JSON), key drivers, pyfunc |
| `__main__.py` | `python -m models.forecast build\|infra-check\|train\|evaluate\|predict` |
| `reference/infrastructure_projects.csv` | The hand-curated, sourced project table (committed) |

`models/forecast` imports `models.price` for the data loader, the market-index helpers, the predictor and the registry. `models.price` never imports `models.forecast`. Because `models/price/__init__.py` already preloads the DLL, the Windows DLL order holds.

There are no new dependencies: XGBoost's `pred_contribs=True` provides the SHAP values.

## Rows (rows.py)

- **Input:** Phase 3's `load_homes` rows: homes only, sales from 2015-01-01 onward, Phase 2 exclusions and size bounds applied.
- **Validation:** keep only rows with `price_aed > 0`, `size_sqm > 0`, a parseable `instance_date` and a non-null `area_id`. Drop the rest.
  - Print `Dropped X of Y rows (Z%)`, broken down by reason.
- **Deduplication:**
  - Drop repeated `transaction_id`s.
  - Flag exact repeats of (building, date, size, price), keep the latest by source row, and count them.
- **Outliers:**
  - Compute log price/m² robust z-scores within area × property kind × month, using the median and MAD (×1.4826).
  - Exclude rows with |z| > 3, and log each exclusion with its area and month.
  - Groups with fewer than 10 sales are not screened. They are counted instead.
- **Report:** a data-quality report (JSON plus printed text) with row counts at each step.

## Targets (targets.py)

- **Base.** For a sale at T, the base is the median price/m² in the same building over [T − 3 months, T).
  - If the building has fewer than 5 sales in that window, the base comes from the same area and property kind over the same window.
  - If that also has fewer than 5, the row is excluded, with reason `no_base`.
  - `base_level` (building or area) and `base_n` are recorded.
- **Target windows:**

  | Horizon | Window |
  |---|---|
  | 3m | [T + 2.5 months, T + 3.5 months] |
  | 1y | [T + 11 months, T + 13 months] |
  | 3y | [T + 35 months, T + 37 months] |

- **Target.** `growth_h = ln(median price/m² in the window ÷ base)`, computed at the **same level as the base**. It needs at least 5 sales in the window.
  - Otherwise the row is excluded for that horizon, with reason `no_target`.
  - Nothing is imputed, and the current price is never used as a fallback.
- **Reporting.** Row counts and exclusion reasons are reported per horizon.
- **Last usable T.** Also reported per horizon; with the 2023 data it is about 2022-12, 2022-03 and 2020-02.

## Features (features.py, infra.py)

Every feature is computed only from sales dated **before T**, or from infrastructure rows whose relevant dates are **on or before T**. The README documents each feature with its as-of rule.

- **Momentum:**
  - ln change of the area × kind median price/m² over the trailing 3, 12 and 36 months;
  - the same changes city-wide by kind;
  - the area's share of city sales over the last 12 months.
- **Base context:**
  - ln base price/m², `base_level`, `base_n`;
  - days since the building's previous sale;
  - building and area sale counts over the trailing 12 months.
- **Property:**
  - property kind, off-plan or ready, size in m², bedrooms;
  - `building_age_proxy_years`: years since the building's first DLD sale before T. It is NaN if the building has no earlier sale.
- **Location:** area and project as categoricals. Category handling follows Phase 3.
- **Infrastructure (as of T, for the row's area):**
  - announced and not yet completed: a count per type (metro/rail, mall, school, park, mixed use, airport);
  - `months_to_next_completion`: months until the nearest planned completion, NaN if none;
  - `completed_last_24m`: whether a project affecting the area completed in the last 24 months, as 0 or 1;
  - a one-hot type mix.
- **Forbidden columns.** `FORBIDDEN_FEATURES` includes `nearest_metro`, `nearest_mall`, `nearest_landmark`, `price_robust_z`, `peer_tier`, `exclusion_reason` and every target column. A test enforces the allowlist.

### Infrastructure table

`models/forecast/reference/infrastructure_projects.csv` has these columns:

| Column | Notes |
|---|---|
| `project_id` | |
| `name` | |
| `type` | one of `metro_rail`, `mall`, `school`, `park`, `mixed_use`, `airport` |
| `announced_date` | |
| `planned_completion_date` | |
| `actual_completion_date` | empty if not completed |
| `affected_area_ids` | semicolon-separated DLD `area_id`s |
| `source_url` | |
| `source_accessed` | |
| `notes` | |

Rules:
- The table holds **at least 15** real projects.
- **Every row cites a public source**, such as RTA, Dubai Media Office, WAM or an operator or developer press release. A row whose dates can't be verified is left out, never guessed.
- The area mapping is proposed by hand from `dld.areas` names, with the reasoning in `notes`.
- `infra-check` validates the table:
  - sources are present;
  - `announced ≤ planned`;
  - the area ids exist in `dld.areas`;
  - the types are known.
- A project announced after the data ends is kept in the table but affects no row. The README says so.
- **Stop point 2:** the table is shown to the user before training.

## Validation (folds.py)

- **Walk-forward folds per horizon.** Fold k has a cutoff C_k. Consecutive cutoffs are 3 months apart for 3m, 6 months for 1y and 12 months for 3y.
- **Training rows** for a fold have T < C_k **and a target window that ends before C_k**. This is the leakage guard.
- **Validation rows** have C_k ≤ T < C_k + step.
- **Test period.** The last validation period of each horizon is its **test period**. It is never used to tune or select a model.
- **Earlier folds** are used for Optuna, early stopping and conformal calibration.
- **Fold count.** Folds start once at least 24 months of training rows exist. The fold count per horizon is reported.

## Models (train.py)

- **Model.** XGBoost `reg:squarederror` on growth, trained on CUDA.
- **Starting parameters:** `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`, with early stopping after 50 rounds on the fold's validation rows.
- **Tuning.** Optuna runs 30 trials per horizon, seeded. The objective is the mean validation MAPE over the tuning folds.
- **Final model.** Each horizon is refit on all rows whose target windows end before the test period starts, with the best parameters and iteration count.
- **Predicted price.** For MAPE and median APE, the predicted price/m² is base × e^(predicted growth), and the actual is the median price/m² in the target window.

## Baselines (baselines.py)

Both baselines are computed on exactly the same rows as the model.

- **No change:** growth = 0.
- **Area trend:** growth = the area × kind trailing ln change over the same length as the horizon, as known at T. It is NaN, and falls back to the city × kind trend, when the area has too few sales.

## Intervals and confidence (intervals.py)

- **80% range.** A split-conformal interval per horizon and segment, from the absolute log errors on the calibration folds, using the (1 − 0.2)(1 + 1/n) quantile.
  - Segments with fewer than 200 calibration rows fall back to the horizon's pooled quantile.
  - Test-period coverage is reported per segment.
- **Confidence labels:**
  - **HIGH:** `base_n ≥ 10`, the base is building-level, and the latest comparable sale is at most 90 days old.
  - **MEDIUM:** `base_n ≥ 5`, and the latest comparable is at most 180 days old.
  - **LOW:** anything else, or any area with fewer than 50 training rows.
- **LOW-confidence message.** Every LOW output carries the text: `"Limited comparable data for this property type/area. Estimate has wide uncertainty (±X%)."` Here X is the half-width of the range as a percentage of the point estimate.

## Evaluation and gate (evaluate.py)

### Metrics

Both metrics are on predicted versus actual price/m²:
- MAPE;
- median APE.

They are reported for each horizon on the test period, and averaged over the validation folds.

### Segments

Results are reported for:
- off-plan vs ready;
- top 5 areas (by training sales) vs the rest;
- building age proxy under 1 year vs over 5 years (1–5 years is reported as its own row).

Any segment with MAPE above 25% is flagged. There is **no single overall accuracy number**.

### Primary segments

The gate is computed on these three segments:
- ready (resale);
- the top 5 areas;
- building age proxy over 2 years.

### Gate per horizon

A horizon passes only if **all** of these hold:
1. Its test MAPE on each primary segment is at most 15% (3m), 20% (1y) or 30% (3y).
2. Its test MAPE is below both baselines' MAPE on the same rows.
3. The upper bound of the bootstrap 95% CI of the model's MAPE (1,000 row-level resamples, seeded) is below the stronger baseline's MAPE.

If it passes, the model is registered as `zestimator-forecast-<h>` with alias `champion`. If it fails, `gate.<h>.passed = 0`, nothing is registered, and the reason is logged.

### MLflow

Everything goes to the MLflow experiment `price-forecast`:
- params and metrics;
- the segment tables as CSV;
- the drop report;
- the infrastructure table snapshot;
- the fold plan;
- the feature importance.

Non-finite metrics are dropped.

## Prediction (predict.py)

**Signature.** `forecast(property: dict, as_of: date | None = None) -> dict`. `as_of` defaults to the last date in the data.

**Input.** The same property fields as Phase 3's `PriceRequest`, plus an optional `property_id`.

**Output JSON:**

```json
{
  "property_id": "...",
  "as_of": "2023-03-17",
  "current_estimate_aed": 1250000,
  "current_range_80": [1150000, 1350000],
  "forecast_3m": {"point": 1270000, "ci_low": 1220000, "ci_high": 1320000, "confidence": "HIGH"},
  "forecast_1y": {"point": 0, "ci_low": 0, "ci_high": 0, "confidence": "LOW",
                  "message": "Limited comparable data for this property type/area. Estimate has wide uncertainty (±18%)."},
  "forecast_3y": {"status": "not_deployed", "reason": "3y resale MAPE 31.2% exceeds the 30% gate"},
  "key_drivers": ["Area prices rose 14% over the last 12 months", "..."],
  "exclusions_applied": ["Dropped 3 off-plan outliers in JVC"],
  "model_versions": {"price": "2", "forecast_3m": "1", "forecast_1y": "1"}
}
```

**How each part is produced:**
- **`current_estimate_aed` and `current_range_80`** come from the Phase 3 champion.
- **Each forecast** is `current_estimate_aed × exp(growth)`, and its range is the conformal range applied the same way.
- **`key_drivers`** are the three largest absolute SHAP contributions (`pred_contribs`), rendered from a fixed template for each feature. Contributions below 0.005 in log growth are omitted, and a driver is never invented.
- **`exclusions_applied`** lists the data rules that removed rows in the property's area and kind segment, from the drop report.

**Rules the prediction always follows:**
- It never outputs a number without a range.
- If the property can't be priced, or its area has no base, it raises the Phase 3 `PriceInputError` family.
- Models are served as MLflow pyfuncs **without `code_paths`**. The repo must be importable, as with the Phase 5 ruling.

## CLI (`python -m models.forecast`)

| Command | What it does |
|---|---|
| `build [--sample N] [--seed S]` | Builds rows, targets and features, then prints the data-quality report and per-horizon counts. **Stop point 1:** the real run first uses `--sample 10000` and shows the report to the user. |
| `infra-check` | Validates and prints the infrastructure table. **Stop point 2.** |
| `train [--trials T] [--device auto\|cuda\|cpu]` | Runs folds, tuning, final fits, intervals, evaluation and gates, logs to MLflow, and registers the horizons that pass. |
| `evaluate` | Re-scores the registered champions on their test periods and prints the segment tables. **Stop point 3:** the segmented table is shown to the user. |
| `predict --area ... --kind ... --status ... --size ... [--bedrooms ...] [--building ...]` | Prints the JSON. |

All commands:
- call `load_dotenv()` first;
- use `DbSettings.from_env()`, which refuses to run without a port;
- record per-stage timings in `data/forecast/stage_timings.json`, which is gitignored.

## Error handling

- **A stop point breaks its guardrail:** the command exits non-zero with the reason. Example: more than 30% of rows dropped in the sample, or an invalid infrastructure row. The pipeline never continues on degraded data.
- **A horizon has too few rows** (fewer than 2 folds or fewer than 1,000 test rows): it is reported as `insufficient_data` and not deployed.
- **The Phase 3 champion is unavailable:** `predict` fails loudly. Training does not need it.

## Testing

Tests run sequentially against `zestimator_test`, use the `temp_mlflow` pattern, and never need a GPU or the network.

- **Rows:**
  - validation drops, with the report and its counts;
  - dedup;
  - the robust outlier rule, including groups with fewer than 10 sales.
- **Targets**, hand-checked on a small synthetic sales frame:
  - base windows and the building→area fallback;
  - each horizon's window edges;
  - exclusion when a window is empty;
  - the same-level rule.
- **Leakage:**
  - **fold guard:** no training row's target window ends on or after its fold cutoff;
  - **as-of features:** altering only sales dated on or after T leaves every feature of that row unchanged;
  - **allowlist:** the feature allowlist and the forbidden-column scan.
- **Infrastructure:**
  - the validator catches a missing source, dates out of order, an unknown area id and an unknown type;
  - a project counts only from its announcement date, and completions only from their completion date.
- **Baselines:** hand-checked.
- **Intervals:**
  - conformal coverage on synthetic data lands within ±5 points of 80%;
  - the pooled fallback works.
- **Confidence:** each label's rules, and the LOW message text with its X.
- **Gate:** passes and fails for each condition, including the case where the model beats only one baseline.
- **Predict:**
  - JSON shape and key names;
  - no number without a range;
  - the `not_deployed` path;
  - key-driver text from SHAP, using a small fitted model.
- **CLI:**
  - `build --sample` and `infra-check` end to end on the test database;
  - `train` on a small synthetic history with 1 trial on the CPU, confirming both the pass and the fail paths of the gate.
- **Lint:** `ruff check`, `ruff format --check` and `pytest -W error` must be clean.

## Deliverables beyond code

- **Real run** on the loaded DLD data, with the three stop points shown to the user.
- **README section "## Price forecasting":**
  - every feature and exclusion rule, with the assumptions and data limits (data end, the 3-year horizon's latest usable T, area-level infrastructure);
  - the segmented tables per horizon with both baselines, gate results and coverage;
  - an example JSON.
  - Every number traces to a log or an MLflow run.
- **The infrastructure table,** with its sources.
- **The architecture page,** republished.
- **Spec amendments** at the bottom of this file.

## Out of scope

- UI and the LLM explanation layer.
- Geocoding and distance features.
- Rent forecasting.
- Any network call during training.
- Serving over HTTP, which is Phase 7.

## Amendments during implementation

**Planning rulings 1–16** (controller, before implementation; see the
ledger `.superpowers/sdd/2026-09-16-phase6-price-forecasting/progress.md`):

- **Day-count windows (ruling 1).** Every window (base, target, momentum,
  trailing) is a fixed day count, not a calendar month: 1 month =
  30.4375 days, rounded. Base `[T − 92d, T)`; 3m target `[T + 76d,
  T + 107d]`; 1y target `[T + 335d, T + 396d]`; 3y target `[T + 1065d,
  T + 1126d]`; momentum lags 91d/365d/1096d; trailing counts 365d. This
  makes every window exact and lets Polars `rolling` compute them.
- **Median of ppsm, then ln (ruling 2).** Base and target growth take the
  median of price/m² first, then ln, rather than the median of ln(ppsm).
  The two differ only for even sale counts; this matches how the spec
  phrases it.
- **Tuning capped at the last 4 folds (ruling 3).** Optuna tunes on at
  most the last 4 non-test, non-`gap` folds per horizon
  (`config.tune_folds`). Every fold is still scored and reported with the
  final parameters, and the tuning-fold refits provide conformal
  calibration.
- **Fixed-round final model (ruling 4).** The final model is refit with
  `n_estimators` = the mean best iteration over the tuning-fold refits,
  with no early stopping, because the test fold must never guide
  training.
- **The 3y horizon was expected to be `insufficient_data` (ruling 5).**
  With 24 months of history before the first cutoff, the first 3y cutoff
  is about 2019-02, and its validation targets need to reach 2022–23 —
  past the 2023-03-17 data end. Reported, not worked around; the README
  explains that a newer DLD file fixes it.
- **The 10,000-row `--sample` build is seeded and random (ruling 6).** It
  reports data quality only; because building windows are sparse in a
  sample, most sampled rows end up `no_base`, and the report says so.
- **`as_of` fixed to the data end (ruling 7).** Forecasts are computed at
  T = data_end + 1 day, so the base window includes the last day of data.
  `as_of` in the JSON is always `data_end`; any other `as_of` is rejected
  (ruling 16).
- **Key drivers come from the 1y model (ruling 8)** when it is deployed,
  otherwise from the 3m model, otherwise the list is empty.
- **The CLI end-to-end test monkeypatches `load_rows` and the area-id
  lookup (ruling 9)**, using synthetic history, matching Phase 3's CLI
  tests. The real SQL path is covered by a separate database test that
  loads `tests/fixtures/price_sample.csv` through Phase 2's
  `run_pipeline`.
- **Project names become a categorical only above 20 training rows
  (ruling 10).** All others map to `"(other)"`, keeping XGBoost's
  category count bounded.
- **The `window_open` status (ruling 11).** A target window that ends
  after the data end gets its own status, alongside `no_base` and
  `no_target`. A partly observed window would bias the median.
- **Categories fitted once per horizon (ruling 12),** on the test fold's
  training rows, and used for every fold. Unused category levels are
  harmless; this keeps one encoding per model.
- **`planned_completion_date` is the date stated at announcement (ruling
  13).** Later revisions would leak hindsight into older rows.
- **Early stopping and fold scoring share each fold's validation rows
  (ruling 14),** as the spec states. The test fold never guides training
  (ruling 4).
- **Ranges cover growth uncertainty only (ruling 15).** A forecast's range
  is `estimate × exp(growth ± half-width)`; the current estimate's own
  range is reported separately as `current_range_80`.
- **Not-deployed reasons (ruling 16).** A missing champion is reported as
  `not_deployed`. The reason is the latest run's gate reason when that
  run failed or had too little data; otherwise it is `"no registered <h>
  model"`.

**Task 2 rulings.**
- **T2-a:** `rolling_stats` joins its window stats back on `(key,
  instance_date)` instead of a positional hstack, because Polars can
  reorder group blocks under parallelism; rows sharing key and date share
  a window, so the join is exact.
- **T2-b:** the history test's all-building assertion excludes the first
  `BASE_DAYS`, because cold-start rows legitimately fall back to area
  level.

**Task 5 rulings (infrastructure table).**
- **T5-a (USER, 2026-09-17):** `planned_completion_date` may carry year or
  quarter precision, mapped to that period's midpoint (year → YYYY-07-01;
  quarter → the 1st of its middle month: Q1 02-01, Q2 05-01, Q3 08-01, Q4
  11-01; "late YYYY"/"end of YYYY" → YYYY-11-01; "early YYYY" →
  YYYY-02-01; "mid YYYY" → YYYY-07-01), and flagged in `notes` as
  "planned precision: year|quarter|part-year". `announced_date` and
  `actual_completion_date` still need at least month precision. Cost:
  `months_to_next_completion` off by up to ~6 months for flagged rows.
- **T5-b:** accept removal of 2 unused imports in
  `test_forecast_infra.py` (ruff); no behavioural cost.
- **T5-c:** "second half of YYYY" maps to YYYY-10-01 (the H2 midpoint,
  consistent with T5-a), and Coca-Cola Arena is filed as `mixed_use`
  because it has no dedicated venue type.

**Final-review fix round (F1–F5).** The final whole-feature review found
4 Important issues; all were fixed before the real run:
- **F1 — never serve a horizon whose latest gate did not pass.**
  `run_training` removes the `champion` alias
  (`MlflowClient().delete_registered_model_alias`, missing model/alias
  treated as a no-op) whenever a horizon's status is not `passed`.
  `latest_gates` reads only runs with `attributes.status = 'FINISHED' and
  tags.gate_run = 'true'`, so a crashed or non-training run is never
  "latest". The `Forecaster` treats a loaded champion as `not_deployed`
  when its gate is `failed`/`insufficient_data` (reason: the gate reason)
  or when `model.metadata["data_end"]` differs from the current data end
  (reason: `"the {h} model was trained on data ending {model_end};
  retrain it on data ending {data_end}"`).
- **F2 — purge tuning, calibration and round selection away from the test
  targets.** A new fold role, `gap`: every non-test fold whose target
  window can still reach the test period
  (`fold.end + timedelta(days=horizon.end_days) > test.cutoff`) is scored
  and reported only, never used for Optuna, conformal calibration or
  `final_rounds`. Roles are assigned in order: `test` (the last fold),
  `gap`, `tune` (up to `config.tune_folds` of the latest remaining
  folds), `score` (the rest). `insufficiency` also reports "no tuning
  fold ends before the test period's target windows" when there is no
  `tune` fold.
- **F3 — plot-priced villas get their own market.** Real data has 30,870
  plot-basis villa sales against 42,430 built-up ones (42% of villa
  sales). `rows.prepare_rows` adds `market_kind` (`"villa_plot"` for
  plot-basis villas, `sub_kind` otherwise) before outlier screening;
  outlier groups, `area_key`/`city_key` and the model's `CATEGORICAL`
  column all key on `market_kind` in place of `sub_kind`; the interval
  segment stays `{reg_type}_{villa|unit}`, computed from `sub_kind`; the
  `Forecaster` computes `market_kind` from the request's kind and
  `size_basis`.
- **F4 — anchor the fold grid backwards from the last usable T.**
  `last = add_months(month_start(last_t), -(horizon.step_months - 1))`,
  so the test period `[last, last + step)` is a full step containing
  `last_t`; cutoffs step backward from `last` by `step`, keeping those
  `>= earliest` (the original forward-computed first cutoff), in
  ascending order.
- **F5 — batched minor fixes:** `bad_price`/`bad_size` also drop
  non-finite values; `validate_rows` sorts by `(instance_date,
  transaction_id)` before dedup, so `keep="first"`/`keep="last"` are
  deterministic ("latest" = the largest `transaction_id` on that date,
  since DLD has no source-row number); `repeat_sale` exclusions are
  logged with the same `EXCLUDED_SCHEMA` as outliers, written to
  `excluded.csv` and logged as an MLflow artifact; `fold_scores` also
  records `median_ape`; `fit_intervals` raising `ValueError` (too few
  calibration rows) is reported as `insufficient_data` with reason
  `f"{h}: too few calibration rows for the 80% range"` instead of
  crashing; the full gate reason list is logged as `{h}/gate.json`;
  `evaluate.gate`'s `strongest` baseline is NaN-safe; `train.suggest_params`'
  learning-rate floor is 0.03 so early stopping can fire within 500
  rounds, and `metadata["capped_rounds"]` records how many tune-fold fits
  hit `n_estimators`; `predict.driver_text` renders `area_code`,
  `market_kind` and `project_code` readably; `Forecaster._features` nulls
  `bedrooms` for a penthouse request, matching training; a snapshot-parity
  test and the missing `predict` tests (`PriceInputError`, an
  `area_id`-only request) were added; a same-day-sale test confirms a
  same-day sale never enters the anchor's own base window.

**USER-2 (2026-09-17).** Per-task code review is skipped from Task 6
onward, at the user's request ("check everything at the end"). The
controller runs the full forecast test suite plus ruff after each task
lands; one whole-Phase-6 review happens at the end, followed by a fix
round and a re-review. Cost: defects surface later than a per-task review
would catch them.

**Documented-only items** (already ruled; out of scope for further
implementation, recorded here for completeness):
- **LOW confidence.** Given the base confidence rules, LOW effectively
  triggers only when an area has fewer than 50 training rows.
- **Project inference.** A request's project is never inferred from its
  building.
- **Early stopping and calibration.** Conformal calibration shares the
  `tune` folds' validation rows with early stopping.
- **Score folds.** `score` folds are scored with parameters tuned on
  later folds.

**Not yet adopted — flagged for user approval (STOP 3, 2026-09-17).** The
3m champion registered from the real run has only 3 boosting rounds
(early stopping hit its minimum in every tuning fold), making it close to
a drift predictor. The controller accepted the passing gate result as-is
(it beats both baselines and every primary segment) rather than treat
this as a failure, and flagged for the user's future decision: adding a
"trailing mean growth" baseline to the 3m gate. This would be a spec
change, since it could cause a future champion to fail against a
stronger, still-trivial baseline — not adopted without the user's
approval.
