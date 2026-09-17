# Multi-horizon price forecasting: brief

Status: queued. The user added this brief on 2026-09-16.
Before any code is written, it goes through the project's usual process: brainstorm, then spec, then plan, then subagent-driven build and review.
Placement: a new phase after Phase 5 (search ranking) and before the API, so the API can serve forecasts. See "Where it fits".

The original request is a generic XGBoost forecasting spec. This version adapts it to what dubimator already has and to what the DLD data actually contains.

## What changed from the original request, and why

| Original | Adapted | Why |
|---|---|---|
| Primary data: the Dubai Pulse Open API | Default: the `dld.market_sales` data already loaded (Phase 2), which ends 2023-03-17. Optional: a fresh Dubai Pulse CSV loaded offline through the existing ingestion pipeline. | Training must stay offline (the original also requires this). A newer extract is a data-source decision for the user. |
| Features `floor_number`, `total_floors`, `view_type` | Dropped. | DLD `Transactions.csv` has no floor, floor count or view columns. The spec must not invent them. |
| `building_age_years` | Dropped. The spec may use a proxy: years since the building's first recorded transaction in DLD, computed only from transactions **before** T. | DLD has no completion dates. |
| Deduplicate on (building, unit, date) | Deduplicate on `transaction_id` first. Then flag exact repeats of (building, date, size, price). | DLD has no unit column. |
| `dist_to_*_km` from property lat/lon | **Area-level** infrastructure features. The spec picks one approach (see Open decisions). | DLD has no coordinates. With the original rule ("missing lat/lon → NaN"), every row would be NaN. |
| DLD `nearest_metro` / `nearest_mall` as features | **Excluded.** They are a snapshot taken when the data was exported, not the state at T. | A 2016 sale lists a Route 2020 station that opened in 2021. That is leakage. |
| Infrastructure table: completion_date only | Adds `announced_date`, `status_history`, `source_url` and `affected_area_ids` (see below). | A feature may count a project only if it was announced by T. Every row needs a citable source. |
| "Metro Line 3" | Use official names: Route 2020 (Red Line extension), Blue Line, Etihad Rail passenger, Expo City, Dubai South, Al Maktoum airport expansion, and RTA projects, each with its announcement date. | "Metro Line 3" is not an official Dubai project name. |
| Pass/fail gate: MAPE < 15% on primary segments | Keep MAPE, but also report median APE (matching Phase 3). Gate **each horizon separately**. A horizon that misses its gate ships as "not deployed", with its numbers published. | 3-year forecasts will almost certainly miss 15%. That is a result to report, not a reason to stop the whole phase. |
| 3-year horizon, walk-forward sliding 12 months | Keep it, but state that a 3-year target needs T ≤ 2020-02. The 3-year model's latest training data therefore spans COVID and never sees the 2021–23 boom. | This follows from where the data ends. Say so in the README. |
| New files `data_pipeline.py`, `train_models.py`, `evaluate.py`, `predict.py` | A package `models/forecast/`, run as `python -m models.forecast build\|train\|evaluate\|predict`. It reuses Phase 2's cleaning and Phase 3's market index, conformal intervals, confidence labels, MLflow registry and pyfunc pattern. | Matches the project layout. It avoids rebuilding the cleaning and outlier rules that Phase 2 already has, with a stated reason for each exclusion. |
| 80% interval via quantile regression or conformal | Use split-conformal on walk-forward residuals, calibrated per horizon and segment. Report coverage per segment. | Same method as Phase 3's calibrated ranges. |
| Output JSON | Keep the JSON exactly. Add `as_of` (the last data date) and `model_versions`. | Every forecast must say which date it is made from. |

## Scope

Three separate XGBoost regressors, one per horizon: `forecast-3m`, `forecast-1y`, `forecast-3y`. Each is registered as `dubimator-forecast-<h>@champion` only if it passes its gate. There is no UI and no LLM layer, and training makes no network calls.

### Targets (unchanged in spirit)

For a sale at time T in building B, the target for horizon h is the **median price/sqm** of market sales in B inside the window below.
- If B has fewer than 5 sales in that window, use the same area and property kind instead.
- If the window is empty, **exclude the row**. Never impute, and never fall back to the current price.

| Horizon | Target window |
|---|---|
| 3 months | [T+2.5 months, T+3.5 months] |
| 1 year | [T+11 months, T+13 months] |
| 3 years | [T+35 months, T+37 months] |

Log how many rows each horizon excludes, and the share. The prediction is the target ratio **relative to** the current market-index level; the Phase 3 index already exists. The spec decides whether the model predicts the ratio or the level. Predicting the ratio is recommended.

### Features

Every feature must be computable from data dated before T. Each one is documented in the README with its as-of rule.

- **Core:**
  - price/sqm at T;
  - area, project and building priors (Phase 3's leak-free priors, recomputed as-of T);
  - `size_sqm`, `bedrooms`, property kind, `reg_type` (off-plan or ready);
  - building "first seen" age proxy;
  - `days_since_last_sale_in_building`;
  - building and area sale counts over the trailing 12 months;
  - market-index momentum over the trailing 3, 12 and 36 months.
- **Infrastructure pipeline** (the differentiator). All of these are area-level and as-of T:
  - count of announced, not-yet-completed projects affecting the area, by type;
  - months until the nearest such project completes;
  - a flag for whether a project affecting the area completed in the last 24 months;
  - a one-hot mix of project types (metro, rail, mall, school, park, mixed-use, airport).
- **Hard rule:** missing values stay NaN, and XGBoost handles them. Never fill with 0.

### Infrastructure table: `models/forecast/reference/infrastructure_projects.csv`

Columns:
- `project_id`, `name`, `type`
- `announced_date`, `planned_completion_date`, `actual_completion_date`
- `status_history` (JSON of date and status pairs)
- `affected_area_ids` (DLD area ids, hand-mapped)
- `radius_note`, `source_url`, `source_accessed`

Rules:
- At least 15 projects.
- **Every row needs a citable public source** (RTA, Dubai Media Office, DLD, operator press releases). Rows whose dates can't be verified are left out, not guessed.
- A project announced after 2023-03-17 is kept in the table but can't affect any training row. The README says so.

### Guardrails (kept from the original)

- **Row validation.** Keep `price > 0`, `sqm > 0`, a parseable date and a non-null area. Everything else is dropped with a printed `Dropped X of Y rows (Z%)`, extending Phase 2's exclusion reasons.
- **Outliers.** Exclude rows more than 3 standard deviations from the area-month median. Use a robust z-score, consistent with Phase 2, and log every exclusion.
- **Validation.**
  - Walk-forward only, sliding 3 months (3m horizon), 6 months (1y) and 12 months (3y).
  - No random splits. A final untouched test fold is reserved for each horizon.
  - Any feature that could be known only after T is a stop-and-flag event.
- **Starting parameters:** `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50`. Tune with Optuna on walk-forward folds only.
- **Metrics.**
  - Report MAPE and median absolute error. Never publish a single accuracy number.
  - Always report by segment:
    - off-plan vs ready;
    - top 5 areas vs the rest;
    - building age proxy under 1 year vs over 5 years.
  - Flag any segment with MAPE above 25%.
- **Baselines** that each model must beat, per horizon:
  - "no change" (price/sqm at T);
  - the area-level market-index drift over the same horizon.
- **Confidence labels.** Each estimate gets HIGH, MEDIUM or LOW, based on:
  - the number of comparable sales in the training window;
  - how well represented the area is;
  - how recent the latest comparable sale is.

  LOW outputs must say: `"Limited comparable data for this property type/area. Estimate has wide uncertainty (±X%)."`
- **Every point estimate carries its 80% interval.**

### Output

```json
{
  "property_id": "...",
  "as_of": "2023-03-17",
  "current_estimate_aed": 1250000,
  "forecast_3m": {"point": 1270000, "ci_low": 1220000, "ci_high": 1320000, "confidence": "HIGH"},
  "forecast_1y": {"point": 0, "ci_low": 0, "ci_high": 0, "confidence": "MEDIUM"},
  "forecast_3y": {"status": "not_deployed", "reason": "3y MAPE 31% on resale exceeds the 15% gate"},
  "key_drivers": ["Route 2020 extension completed 14 months before the valuation date"],
  "exclusions_applied": ["Dropped 3 off-plan outliers in JVC"],
  "model_versions": {"price": "2", "forecast_3m": "1", "forecast_1y": "1"}
}
```

`current_estimate_aed` comes from the Phase 3 champion. `key_drivers` are generated from the largest feature contributions (SHAP values from XGBoost) and never invented.

### Order of work (with stop points)

1. **Data pipeline.** Run it on a 10,000-row sample and show the data-quality report: drops by reason, and the excluded share per horizon.
2. **Infrastructure table.** Build it with at least 15 sourced projects and show it for review.
3. **Train and validate.** Show the segmented MAPE and median-APE table against both baselines.
4. **Registration and prediction.** Only for horizons that pass their gate (MAPE < 15% on resale, top-5 areas and buildings older than 2 years): register the model and build prediction. Horizons that fail are reported and not deployed.

If any step breaks its guardrail, stop and report. Never proceed on degraded data.

## Open decisions (ask the user when brainstorming)

1. **Infrastructure geography**, since DLD has no coordinates:
   - (a) hand-map each project to the DLD areas it affects (recommended: small, auditable, honest);
   - (b) hand-curate a centroid for each of the ~250 areas and compute distances (larger, and error-prone);
   - (c) geocode buildings from an external source (breaks the offline rule unless it's done once and saved).
2. **Data freshness:**
   - keep the 2023-03 mirror (the 3-year model then ends at 2020 training data); or
   - load a newer DLD extract from Dubai Pulse offline through Phase 2's pipeline. This may need a Dubai Pulse account, which only the user can create.
3. **Gate per horizon:** keep 15% for all three, or loosen it for 1y and 3y (for example 20% and 30%).

## Where it fits

Proposed renumbering, with forecasting as Phase 6:

| Phase | Content |
|---|---|
| 6 | Price forecasting |
| 7 | API, which serves `/forecast` too |
| 8 | Demo UI |
| 9 | CI/CD |
| 10 | Deploy and monitor (local Docker plus Cloudflare Tunnel) |
| 11 | Documentation |

Alternative: append forecasting as Phase 11, which leaves the current numbering untouched but keeps forecasting out of the API and demo.
