# Phase 2 — DLD Ingestion Pipeline

Status: approved for implementation planning
Date: 2026-09-15
Part of: Dubai Real Estate ML Platform (sub-project 2 of 10)
Builds on: `docs/superpowers/specs/2026-09-14-phase1-foundation-design.md`

## Goal

Read the Dubai Land Department (DLD) transactions CSV from `data/raw/`,
validate it strictly, normalize it, classify every row as a clean market sale
or excluded with exactly one reason code, and load all rows into Postgres
atomically, with a `market_sales` view for Phase 3 and an area alias table for
Phases 5–7. Runs from the host CLI and as a manually triggered Airflow DAG.

## Source data (profiled 2026-09-15)

- File: `data/raw/Transactions.csv`, 637,241,734 bytes, UTF-8 without BOM.
- Provenance: Kaggle mirror `alexefimik/dubai-real-estate-transactions-dataset`
  of DLD's open `Transactions.csv` (Dubai Pulse). Real data, not synthetic.
- 1,047,965 rows × 46 columns. `transaction_id` unique across all rows.
- `instance_date` format `DD-MM-YYYY`; range 1995-03-07 → 2023-03-17; 5 rows empty.
- Missing numerics are the literal token `null`; missing text is `""`.
- `trans_group_en`: Sales 787,892 · Mortgages 224,183 · Gifts 35,890.
- `reg_type_en`: Existing Properties 739,266 · Off-Plan Properties 308,699.
- `property_type_en`: Unit 721,070 · Villa 214,855 · Land 79,710 · Building 32,330.
- `procedure_area` is square metres for every row: `meter_sale_price` equals
  `actual_worth / procedure_area` within 1% for 786,797 of 787,892 sales; the
  256 disagreements are `actual_worth = 1` placeholders.
- 253 `area_id` values, 252 `area_name_en` values (`Mushrif` has ids 404 and
  420). Names are DLD official names; familiar names live in
  `master_project_en` (e.g. `Marsa Dubai` ← "Dubai Marina" and
  "Jumeriah Beach Residence  - JBR").

**Decision (user, 2026-09-15):** accept that data ends 2023-03-17. Estimates are
"as of Q1 2023"; README and later UI state this. Adding newer DLD exports is a
separate future task, not designed for here.

## Architecture

Python + Polars transforms in code, then load to Postgres (not SQL-side ELT —
every cleaning function must be unit-testable against malformed inputs).
Flat top-level package `ingestion/` (Phase 1 layout), run with `python -m`.

```
ingestion/
  __main__.py        CLI: python -m ingestion [--csv PATH]
  config.py          DbSettings from env (load_dotenv, no override)
  schema.py          expected columns, closed domains, SchemaDriftError, read_raw()
  normalize.py       pure value parsers + to_typed(raw_df) -> typed_df
  rules.py           classify(typed_df) -> adds exclusion_reason, peer_tier, price_robust_z
  areas.py           build_areas(), build_area_aliases()
  load.py            apply DDL, run bookkeeping, COPY
  pipeline.py        run_pipeline(csv_path, settings) -> RunSummary
  sql/schema.sql     idempotent DDL
  reference/curated_area_aliases.csv
dags/dld_ingestion.py
Dockerfile.airflow
scripts/build_dld_fixture.py
tests/conftest.py
tests/ingestion/test_schema.py, test_normalize.py, test_rules.py,
               test_areas.py, test_pipeline_integration.py
tests/fixtures/dld_sample.csv
tests/test_airflow_dag.py
```

Dependencies: add `polars`, `psycopg2-binary`, `python-dotenv` to
`[project] dependencies` (move the latter two out of the dev group).

## Stage 1 — Read and validate (`schema.py`)

- `EXPECTED_COLUMNS`: the exact 46 source column names (order-agnostic):
  `transaction_id, procedure_id, trans_group_id, trans_group_ar, trans_group_en,
  procedure_name_ar, procedure_name_en, instance_date, property_type_id,
  property_type_ar, property_type_en, property_sub_type_id, property_sub_type_ar,
  property_sub_type_en, property_usage_ar, property_usage_en, reg_type_id,
  reg_type_ar, reg_type_en, area_id, area_name_ar, area_name_en, building_name_ar,
  building_name_en, project_number, project_name_ar, project_name_en,
  master_project_en, master_project_ar, nearest_landmark_ar, nearest_landmark_en,
  nearest_metro_ar, nearest_metro_en, nearest_mall_ar, nearest_mall_en, rooms_ar,
  rooms_en, has_parking, procedure_area, actual_worth, meter_sale_price,
  rent_value, meter_rent_price, no_of_parties_role_1, no_of_parties_role_2,
  no_of_parties_role_3`.
- `read_raw(path) -> pl.DataFrame`: all columns read as strings
  (`infer_schema=False`), `null_values=["null"]`, strict UTF-8 (invalid bytes
  raise). Missing file → `FileNotFoundError` with the path.
- `validate_columns(columns)`: raises `SchemaDriftError` naming the missing and
  unexpected columns if the set differs from `EXPECTED_COLUMNS`.
- Closed domains, checked on the raw values; any value outside raises
  `SchemaDriftError` naming the column and the unexpected values:
  - `trans_group_en` ∈ {Sales, Mortgages, Gifts}
  - `reg_type_en` ∈ {Existing Properties, Off-Plan Properties}
  - `property_type_en` ∈ {Unit, Villa, Land, Building}
- All validation happens before any database connection is opened.

## Stage 2 — Normalize (`normalize.py`)

Pure functions, each unit-tested with malformed inputs, applied by
`to_typed(raw) -> typed` (Polars expressions or `map_elements` where needed):

| Function | Behaviour |
|---|---|
| `parse_date(s)` | `DD-MM-YYYY` → `date`; `""`, `None`, or invalid (e.g. `31-02-2020`) → `None` |
| `parse_float(s)` | numeric string → float; `None`, `""`, non-numeric → `None` |
| `parse_rooms(s) -> (label, bedrooms)` | `"Studio"`→`("Studio", 0)`; `"N B/R"`→`("N B/R", N)` for N 1–9; `"PENTHOUSE"`→`("Penthouse", None)`; `"GYM"`→`("Gym", None)`; `Office`, `Shop`, `Single Room`, `Store` → `(label, None)`; `""`/`None` → `(None, None)`; any other value → `(stripped value, None)` |
| `clean_display_name(s)` | collapse internal whitespace, strip; if the value is ALL CAPS (has letters, no lowercase) **and longer than 5 characters** convert to title case (so short acronyms like `DIFC`, `JBR` stay); `""` → `None`. `"Al Khairan  Second"`→`"Al Khairan Second"`, `"MADINAT HIND 2"`→`"Madinat Hind 2"`, `"DIFC"`→`"DIFC"`; `"Al-Nahdah"` unchanged |
| `match_key(s)` | lowercase; every non-alphanumeric char → space; split; drop tokens equal to `al`; join with one space. `"Al-Nahdah"`→`"nahdah"`, `"Jumeriah Beach Residence  - JBR"`→`"jumeriah beach residence jbr"` |
| `trans_group` | Sales→`sales`, Mortgages→`mortgages`, Gifts→`gifts` |
| `reg_type` | Existing Properties→`ready`, Off-Plan Properties→`off_plan` |
| `property_type` | lowercase of source value |
| `has_parking` | `"1"`→True, `"0"`→False, else None |

Typed frame columns (these become `dld.transactions` columns, plus the rule
columns from Stage 3): `transaction_id, source_row` (1-based data-row index in
file order), `trans_group, procedure_name` (source English, stripped),
`instance_date, year, property_type, property_sub_type, property_usage,
reg_type, area_id (int), area_name` (display), `area_name_ar, building_name,
project_number (int), project_name, master_project` (display),
`nearest_landmark, nearest_metro, nearest_mall, rooms, bedrooms, has_parking,
area_sqm, price_aed, price_per_sqm_aed, parties_role_1, parties_role_2,
parties_role_3`.

- `area_sqm` = `procedure_area` (canonical unit is m²; no conversion — the
  profile shows the source is m² throughout).
- `price_aed` = `actual_worth`.
- `price_per_sqm_aed` = `price_aed / area_sqm` when both present and
  `area_sqm > 0`, else `None`. Recomputed, not taken from `meter_sale_price`.
  **It is derived from the target — Phase 3 must never use it as a feature.**
- Text columns use `clean_display_name`; empty → `None`.
- Not loaded (kept only in the source file): Arabic columns other than
  `area_name_ar`, the `*_id` code columns except `area_id`, `meter_sale_price`,
  `rent_value`, `meter_rent_price`.

## Stage 3 — Classify (`rules.py`)

`MARKET_SALE_PROCEDURES = {"Sell", "Sell - Pre registration", "Delayed Sell",
"Sale On Payment Plan"}` (730,651 rows in the profiled file).

Every row gets `exclusion_reason` = `None` (clean market sale) or the **first**
matching reason, evaluated in this order:

| # | Reason | Condition |
|---|---|---|
| 1 | `duplicate_transaction_id` | a previous row (lower `source_row`) has the same `transaction_id` |
| 2 | `mortgage` | `trans_group == "mortgages"` |
| 3 | `gift` | `trans_group == "gifts"` |
| 4 | `non_market_procedure` | `procedure_name` not in `MARKET_SALE_PROCEDURES` (includes unknown procedures) |
| 5 | `missing_date` | `instance_date` is null |
| 6 | `missing_price` | `price_aed` is null |
| 7 | `invalid_area` | `area_sqm` null or `<= 0` |
| 8 | `price_below_floor` | `price_aed < 10_000` |
| 9 | `suspected_sqft_entry` | see outlier rules |
| 10 | `price_outlier_low` | see outlier rules |
| 11 | `price_outlier_high` | see outlier rules |

**Outlier rules (reasons 9–11).** Candidates = rows passing rules 1–8.
`x = ln(price_per_sqm_aed)`. Peer tiers, first qualifying tier wins per row:

| Tier | Group key |
|---|---|
| 1 | `area_id, property_type, reg_type, year` |
| 2 | `area_id, property_type, year` |
| 3 | `property_type, reg_type, year` |
| 4 | `property_type, year` |
| 5 | `property_type` |

A group qualifies when it has ≥ 30 candidate rows and `MAD > 0`, where
`MAD = median(|x − median(x)|)` over the group's candidates.
`price_robust_z = (x − median) / scale` with `scale = max(1.4826 × MAD, 0.12)`;
the sqft-corrected z uses the same scale. `peer_tier` records the tier
used. If no tier qualifies for a row, raise (cannot happen with real data;
tested with a synthetic frame). Then:
- `z < −3.5` and the sqft-corrected value
  `(ln(price_per_sqm_aed × 10.7639) − median) / scale` lies in
  `[−3.5, 3.5]` → `suspected_sqft_entry` (size was probably entered in sqft).
- else `z < −3.5` → `price_outlier_low`.
- `z > 3.5` → `price_outlier_high`.
Rows excluded by rules 1–8 have `peer_tier` and `price_robust_z` null.

## Stage 4 — Areas and aliases (`areas.py`)

- `dld.areas`: one row per `area_id`: `area_id, name_en` (display),
  `name_ar, match_key, market_sales` (count of rows with null
  `exclusion_reason`). If an `area_id` appears with several names, use the most
  frequent.
- `dld.area_aliases (alias_key, alias, area_id, source)`, primary key
  `(alias_key, area_id)` — one alias may map to several areas (e.g. `mushrif`
  → 404 and 420):
  1. `official`: each area's `name_en` → its `area_id`.
  2. `master_project`: each `master_project` with ≥ 50 rows whose most common
     `area_id` holds ≥ 80% of its rows → that `area_id`.
  3. `curated`: rows of `ingestion/reference/curated_area_aliases.csv`
     (`alias,area_name_en`), resolved by the official name's `match_key`.
     Initial content, each mapping evidenced by the profile:
     `Dubai Marina,Marsa Dubai` · `Marina,Marsa Dubai` · `JBR,Marsa Dubai` ·
     `Jumeirah Beach Residence,Marsa Dubai` · `JLT,Al Thanyah Fifth` ·
     `JVC,Al Barsha South Fourth` · `Downtown,Burj Khalifa` ·
     `Downtown Dubai,Burj Khalifa` · `International City,Al Warsan First` ·
     `Sports City,Al Hebiah Fourth` · `Dubai Hills,Hadaeq Sheikh Mohammed Bin Rashid`.
- `alias_key = match_key(alias)`; duplicate `(alias_key, area_id)` pairs keep
  the first source in the order official → master_project → curated.
- Curated rows whose target area isn't present are **not** an error (a small
  fixture won't contain every area); they are returned as
  `unresolved_curated_aliases` and recorded in the run details. The full-file
  verification requires that list to be empty.

## Stage 5 — Load (`load.py`, `sql/schema.sql`)

- Schema `dld`. `sql/schema.sql` is idempotent (`CREATE SCHEMA IF NOT EXISTS`,
  `CREATE TABLE IF NOT EXISTS`, `CREATE OR REPLACE VIEW`) and is applied at the
  start of every run. No migration framework.
- Tables: `dld.ingestion_runs, dld.transactions, dld.areas,
  dld.area_aliases`; view `dld.market_sales` = `dld.transactions WHERE
  exclusion_reason IS NULL`.
- `dld.transactions`: typed columns from Stage 2 + `exclusion_reason text`,
  `peer_tier smallint`, `price_robust_z double precision`, `ingest_run_id int
  references dld.ingestion_runs`. Primary key `(ingest_run_id, source_row)`
  (so duplicate source ids can still be stored). Index on `exclusion_reason`,
  and on `(area_id, property_type, instance_date)`.
- `dld.ingestion_runs`: `run_id serial PK, started_at timestamptz, finished_at
  timestamptz, source_path text, source_sha256 text, rows_read int,
  rows_loaded int, rows_market_sale int, status text` (`running` /
  `succeeded` / `failed`), `error text, details jsonb` (keys:
  `reason_counts`, `peer_tier_counts`, `unresolved_curated_aliases`).
- Run flow: (1) validate + transform in memory; (2) insert run row with
  `status='running'` and commit; (3) in **one transaction**: `TRUNCATE
  dld.transactions, dld.areas, dld.area_aliases`, then `COPY` all rows in
  chunks of 100,000, then insert areas and aliases; commit; (4) update the run
  row to `succeeded` with counts/details. On any exception in (3)/(4): roll
  back, mark the run `failed` with the error text, re-raise. A failed run
  leaves the previous data intact.
- Invariant checked before commit: `rows_loaded == rows_read`.

## Configuration (`config.py`)

`DbSettings(host, port, user, password, dbname)` from env with `load_dotenv()`
(no override): `POSTGRES_HOST` (default `127.0.0.1`), `POSTGRES_PORT`
(default `5432`), `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
(defaults `zestimator`, `changeme`, `zestimator`).

## CLI

`uv run python -m ingestion [--csv PATH]` (default
`data/raw/Transactions.csv`). Prints rows read, market sales, and a table of
reason counts; exit code 0 on success, 1 on failure with the error on stderr.

## Airflow

- `Dockerfile.airflow`: `FROM apache/airflow:2.10.3-python3.11`, then
  `pip install --no-cache-dir` exact versions of `polars`, `psycopg2-binary`,
  `python-dotenv` matching `uv.lock`.
- Compose `airflow` service: `build: Dockerfile.airflow` (replaces `image:`);
  `command: bash -c "rm -f /opt/airflow/*.pid && exec airflow standalone"`
  (clears stale PID files left in the `airflow_data` volume — Phase 1
  residual); `depends_on: postgres: condition: service_healthy`; volumes add
  `./dags:/opt/airflow/dags`, `./ingestion:/opt/zestimator/ingestion:ro`,
  `./data:/opt/zestimator/data:ro`; environment adds
  `PYTHONPATH=/opt/zestimator`, `POSTGRES_HOST=postgres`, `POSTGRES_PORT=5432`
  (container-internal port, not the host-mapped one), `POSTGRES_USER`,
  `POSTGRES_PASSWORD`, `POSTGRES_DB` from `.env`,
  `DLD_CSV_PATH=/opt/zestimator/data/raw/Transactions.csv`,
  `AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=false`.
- `dags/dld_ingestion.py`: `dag_id="dld_ingestion"`, `schedule=None`,
  `catchup=False`, `start_date=2026-01-01`, one task `ingest` calling
  `ingestion.pipeline.run_pipeline(Path(os.environ["DLD_CSV_PATH"]),
  DbSettings.from_env())`.

## Tests

- `tests/conftest.py`: `load_dotenv()` once for all tests (moved from
  `tests/test_infra_smoke.py`), plus a `pg_test_db` fixture that creates a fresh
  database `zestimator_test` on the running Postgres, yields its `DbSettings`,
  and drops it afterwards; skips (socket check, as in the smoke tests) when
  Postgres isn't reachable.
- Unit tests (no database), each using malformed DLD-style inputs:
  `test_schema.py` (missing / extra / renamed column; unexpected
  `trans_group_en`, `reg_type_en`, `property_type_en` values; invalid UTF-8
  file; `null` token handling), `test_normalize.py` (every function in the
  Stage 2 table, including the listed examples and malformed values),
  `test_rules.py` (each reason with a synthetic frame, first-match ordering,
  tier fallback including `MAD == 0`, the sqft rule, no-qualifying-tier raise),
  `test_areas.py` (Mushrif-style multi-id alias, master-project 80% threshold,
  curated resolution and unresolved reporting).
- `tests/fixtures/dld_sample.csv`: between 250 and 400 real rows sampled
  deterministically (seed 42) by `scripts/build_dld_fixture.py` from the real
  file, stratified to include rows of every `trans_group`, every market-sale
  procedure, several non-market procedures, both reg types, all four property
  types, the 5 empty-date rows, whitelisted-procedure sales with
  `actual_worth` null, and whitelisted-procedure sales with
  `actual_worth < 10000`. Real rows only; no edited values. (All 5 empty-date
  rows are mortgages or non-market procedures, so `missing_date` never fires
  on real data — it is covered by unit tests only.)
- `test_pipeline_integration.py` (uses `pg_test_db`): runs `run_pipeline` on
  the fixture and asserts `rows_loaded == rows_read == fixture rows`; the view
  count equals rows with null reason; the deterministic rules hold on the
  loaded rows, checked as SQL properties (every non-duplicate `mortgages` row
  has reason `mortgage`; every non-duplicate `gifts` row has `gift`; every
  non-duplicate `sales` row with a procedure outside `MARKET_SALE_PROCEDURES`
  has `non_market_procedure`; every whitelisted sale with a date and null
  price has `missing_price`; every whitelisted sale with date, price and
  positive area and price < 10,000 has `price_below_floor`), and each of those
  five reasons occurs at least once; the 5 empty-date rows load with null
  `instance_date`; `details.reason_counts` sums to `rows_read`; a second run
  leaves the same counts (full refresh, no duplication) and records a second
  succeeded run; a schema-drift CSV fails without creating any run row.
- `tests/test_airflow_dag.py`: `docker compose exec -T airflow airflow dags
  list-import-errors -o json` returns no errors, and `dld_ingestion` is listed;
  skips when the Airflow container isn't running.
- Existing Phase 1 tests keep passing.

## Verification (full file)

1. `uv run python -m ingestion` on the full file succeeds; record runtime,
   `rows_read` (expect 1,047,965), market sales, reason counts, peer-tier
   counts; `unresolved_curated_aliases` is empty.
2. `docker compose up -d --build --wait` succeeds with the new Airflow image;
   `docker compose exec airflow airflow dags test dld_ingestion` succeeds
   against the real file.
3. README gains a "Data" section: provenance, coverage 1995-03-07 →
   2023-03-17, "estimates as of Q1 2023", real data only (no synthetic data
   in Phase 2), the run's counts, and how to run ingestion (CLI and Airflow).
   `data/README.md` documents the expected file.

## Housekeeping folded into Phase 2

- Airflow stale-PID fix (above).
- `[tool.ruff] extend-exclude = ["docs"]` so `ruff format --check .` ignores
  the markdown plans.
- `load_dotenv()` moves to `tests/conftest.py`.

## Out of scope

Model training and training-window/recency choices (Phase 3), sparse-area
prediction fallback (Phase 3), search use of aliases (Phase 5), newer DLD
data, scheduling (the DAG is manual-trigger only), CI (Phase 8).

## Amendments during implementation (2026-09-15)

- Runs are serialized with a session-level Postgres advisory lock taken right after connecting; a concurrent run raises `IngestionInProgressError` before writing anything. While holding the lock, leftover `running` rows are marked `failed` ("abandoned: ingestion process ended before finishing").
- `finished_at` uses `clock_timestamp()` (not `now()`, which is the transaction start).
- The load sets `lock_timeout = 60s` before its `TRUNCATE`, so an idle reader holding a lock fails the run instead of hanging it.
- The DAG has `max_active_runs=1` and is verified with `airflow dags trigger`, never `dags test`/`tasks test` (those create DagRuns that the unpaused scheduler also executes).
- Outlier robust scale has a floor of 0.12 (log units), so prices within about 1.5x of the peer median are never flagged; bulk-sale blocks can make a group's MAD near zero.
- `price_per_sqm_aed`, `price_robust_z`, `peer_tier` and `exclusion_reason` carry database comments warning they are not model features.
