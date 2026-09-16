# Phase 9: CI/CD

- **Status:** designed on 2026-09-17 under the user's overnight delegation. Every decision below is the controller's recommended option, marked (C).
- **Process:** the lighter one the user agreed to for Phases 9–11: a short design, bigger tasks, and one review at the end of the phase.
- **Date:** 2026-09-17
- **Part of:** Dubai Real Estate ML Platform (sub-project 9 of 11)

## Goal

Every change is checked automatically: lint, the unit and database tests, and container builds. Retraining can run on a schedule, and every stage keeps its gate. The repository has **no git remote yet**, so everything must also run locally with one command, and the GitHub workflows must be valid and ready for the first push.

## Decisions (all (C))

| Topic | Decision |
|---|---|
| CI host | GitHub Actions, in `.github/workflows/ci.yml`. It runs on push and pull_request to `master`. |
| CI jobs | (1) `lint`: `uv run ruff check .` and `ruff format --check .`. (2) `test`: an Ubuntu runner with a `pgvector/pgvector:pg16` service on port 5433, the `.env` values written from the workflow's `env:`, and `uv sync`, then `uv run pytest -q -W error -m "not gpu and not live"`. (3) `docker`: builds the API and demo images from Phase 10's Dockerfiles once they exist; until then it builds the existing `Dockerfile.mlflow` and `Dockerfile.airflow`. |
| GPU and live tests | CI has no GPU and no MLflow server with champions. New pytest markers: `gpu` (needs CUDA) and `live` (needs the running stack and registered models). They are registered in `pyproject.toml`. `tests/test_infra_smoke.py` is marked `live`. The workflow deselects both. |
| Torch on Linux CI | The lockfile pins the cu128 torch wheel, which also runs on CPU-only runners; tests never need CUDA. CI installs the locked environment unchanged (`uv sync --frozen`), with `astral-sh/setup-uv` caching enabled so the large wheel downloads once. Swapping to a CPU wheel would need a second lock, which isn't worth maintaining. The README flags the download size as a first-push note. |
| Infra guards (Phase 3 carry-over) | `pg_test_db` skips when `POSTGRES_PORT` is unset instead of erroring. `test_infra_smoke` no longer defaults the port to 5432. `REQUIRE_INFRA=1` turns those skips into failures, and CI sets it so the DB tests really run. |
| Local runner | `scripts/ci.py` (`uv run python scripts/ci.py [--fast]`) runs the same steps in order: ruff check, format check, pytest with the same markers, and the compose config check. It prints a pass/fail table and exits non-zero on failure. `--fast` skips the DB tests. |
| Workflow validation | `actionlint`, run through `uvx --from actionlint-py actionlint`, validates `.github/workflows/*.yml`. `scripts/ci.py` runs it when available and skips it with a note otherwise. |
| Scheduled retraining | CI runners have neither the 637 MB DLD file nor a GPU, so retraining stays local. `python -m pipelines retrain` runs the stages in order, each keeping its own gate: ingestion (skipped if the source hash is unchanged), price train, listings detect (registers the pair model), search queries + train, forecast build + train. It stops at the first failure and writes `data/pipelines/retrain_<timestamp>.json` with each stage's status, duration and exit code. The README documents a Windows Task Scheduler entry (weekly) and a cron line. |
| Release | `.github/workflows/release.yml`: on a `v*` tag, it builds and pushes the API and demo images to GHCR. It is inactive until a remote exists. |
| Dependabot | `.github/dependabot.yml`: weekly updates for `uv` and `github-actions`. |

## Deliverables

- `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `.github/dependabot.yml`
- Markers and guards: `pyproject.toml` markers; `tests/conftest.py` with `REQUIRE_INFRA` and the unset-port skip; `tests/test_infra_smoke.py` fixes
- `scripts/ci.py`
- The `pipelines/` package: `__init__.py`, `__main__.py`, `retrain.py`, and tests with fake stage runners
- A README "## CI/CD" section: local runner, workflows, markers, retraining schedule, first-push checklist

## Verification

- `uv run python scripts/ci.py` passes locally.
- `actionlint` passes.
- `python -m pipelines retrain --dry-run` lists the stages. A real `retrain --only forecast` run completes.
- The one-review-per-phase rule applies: a single review of the whole phase diff at the end.

## Amendments during implementation

(none yet)
