# Phase 1 — Foundation: Repo Scaffolding + Local Docker Compose Stack

Status: approved for implementation planning
Date: 2026-09-14
Part of: Dubai Real Estate ML Platform (sub-project 1 of 10 — see decomposition below)

## Context

This is the first of 10 planned sub-projects building a portfolio-grade ML
platform for Dubai real estate (price estimation, duplicate/fraud listing
detection, and search ranking), backed by real Dubai Land Department (DLD)
transaction data. The full decomposition, in build order:

1. **Foundation** — repo scaffolding, Docker Compose (this spec)
2. Ingestion — DLD CSV cleaning pipeline → Postgres
3. Price estimation model — train/eval/MLflow registry
4. Duplicate/fraud detection — needs a decision on supplementary data
5. Search ranking model — LTR + semantic search
6. Unified FastAPI service
7. Streamlit demo UI
8. CI/CD (GitHub Actions)
9. Deployment + monitoring (Cloud Run, Evidently)
10. Docs/README/diagrams — folded into each phase, plus a final pass

Each sub-project gets its own brainstorm → spec → plan → implementation
cycle. This document covers sub-project 1 only.

Tech stack is fixed by the master build prompt (Python 3.11+, Postgres +
pgvector, MLflow, Airflow, FastAPI, Streamlit, Docker, GCP Cloud Run) — not
open for substitution without asking.

## Goal

Stand up a working local dev environment: a git repo with clear module
separation, a `uv`-managed Python project, and a Docker Compose stack
(Postgres+pgvector, MLflow, Airflow) that all start cleanly and pass a
smoke test — with nothing else built on top yet.

## Repo layout

```
zestimator/
├── ingestion/       # (Phase 2+) DLD CSV cleaning pipeline — empty placeholder now
├── models/          # (Phase 3+) price/fraud/ranking model code — empty placeholder now
├── api/             # (Phase 6) FastAPI service — empty placeholder now
├── demo/            # (Phase 7) Streamlit app — empty placeholder now
├── data/raw/        # gitignored — DLD CSVs dropped here by the user
├── data/README.md   # documents expected file naming/schema (placeholder until Phase 2)
├── tests/           # pytest suite, starting with infra smoke tests
├── docker-compose.yml
├── Dockerfile.mlflow
├── pyproject.toml   # uv-managed
├── .env.example
├── .gitignore
└── README.md        # architecture diagram (Mermaid) + cost breakdown, built up over phases
```

Each of `ingestion/`, `models/`, `api/`, `demo/` gets an `__init__.py` and a
one-line placeholder so the structure is visible in the initial commit, but
no logic is implemented in Phase 1.

## Docker Compose services

- **postgres** — `pgvector/pgvector:pg16` image. App database `zestimator`.
  Exposes `5432`. Credentials from `.env` (`POSTGRES_USER`,
  `POSTGRES_PASSWORD`, `POSTGRES_DB`). This is the one database used by
  ingestion/models/API in later phases — pgvector extension enabled at
  container init via an init SQL script (`docker/postgres-init.sql`
  running `CREATE EXTENSION IF NOT EXISTS vector;`).

- **mlflow** — custom `Dockerfile.mlflow` (`python:3.11-slim` + `mlflow`
  package). SQLite backend store (`/mlflow/mlflow.db`) and local artifact
  root (`/mlflow/artifacts`), both bind-mounted to `./mlruns` on the host
  so runs survive container restarts. Exposes `5000`.

- **airflow** — official `apache/airflow:2.10-python3.11` image, run via
  the `airflow standalone` command (bundles webserver + scheduler +
  triggerer + its own SQLite metadata DB in a single container/process).
  This is a deliberate simplification: Airflow's docs recommend separate
  webserver/scheduler/init containers with a dedicated Postgres metadata
  DB for production, but this stack is local-dev-only per the master
  prompt ("local Docker for dev"), and `standalone` is the documented way
  to run a single-node dev instance with minimal setup. Exposes `8080`,
  admin credentials auto-generated on first run and printed to logs (we
  will NOT try to pin a fixed admin password in Phase 1 — acceptable for
  local dev; revisit if this becomes a real pain point in later phases).

  Airflow's metadata store is NOT the app Postgres container — it stays
  fully isolated (its own SQLite inside the container), avoiding any
  schema collision with app data.

All three services are defined in one `docker-compose.yml` at repo root.
`docker compose up -d` brings up the full stack.

## Python tooling

Single `uv`-managed `pyproject.toml` at repo root (not split into
per-module packages — that would be premature for a repo with no logic
yet). Phase 1 dependencies are limited to what's needed to write and run
the smoke test:

- `pytest`
- `ruff` (lint, dev dependency)
- `psycopg2-binary` (Postgres connectivity check)
- `requests` (HTTP health checks for MLflow/Airflow)

Later phases add `pandas`/`polars`, `scikit-learn`, `xgboost`, etc. as
each phase actually needs them — no speculative pre-declaration.

## Secrets

`.env.example` at repo root with placeholder values:

```
POSTGRES_USER=zestimator
POSTGRES_PASSWORD=changeme
POSTGRES_DB=zestimator
POSTGRES_PORT=5432
MLFLOW_TRACKING_URI=http://localhost:5000
AIRFLOW_PORT=8080
```

Real `.env` is gitignored. `docker-compose.yml` reads from `.env` via
Compose's built-in `.env` support (no extra tooling needed).

## Testing

No application logic exists yet in Phase 1, so "tests" means infra smoke
tests confirming the stack is healthy — this becomes the first step
GitHub Actions CI runs in Phase 8.

`tests/test_infra_smoke.py`, run manually after `docker compose up -d`
(not run automatically as part of `docker compose up` itself, since CI
wiring is Phase 8's job):

- Postgres: connect via `psycopg2` using `.env` credentials, run
  `CREATE EXTENSION IF NOT EXISTS vector;` and confirm no error (proves
  both connectivity and that pgvector is actually installed, not just
  that the container is up).
- MLflow: `GET http://localhost:5000/health` returns 200. (If MLflow's
  built-in health endpoint doesn't exist in the pinned version, fall back
  to `GET /` returning 200 — verify which during implementation and
  document the choice in the test docstring.)
- Airflow: `GET http://localhost:8080/health` returns 200.

Each test skips with a clear message (not a hard failure) if the
corresponding service isn't reachable at all, so `pytest` run without
Docker running fails clearly rather than hanging on connection timeouts —
achieved with a short explicit connection timeout (a few seconds) per
check, not a bare `pytest.skip` with no signal.

## Out of scope for Phase 1

- Any DLD data ingestion logic (Phase 2)
- Any model code (Phase 3+)
- CI/CD wiring — GitHub Actions doesn't run this smoke test yet (Phase 8)
- Cloud deployment of any kind (Phase 9)
- Airflow DAGs — the Airflow container runs with zero DAGs defined; DAG
  authoring starts when a phase actually needs orchestration

## Open questions

None — tooling (uv, Docker, Docker Compose) confirmed already installed
and working on the target machine.
