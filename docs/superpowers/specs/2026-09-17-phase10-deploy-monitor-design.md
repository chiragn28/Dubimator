# Phase 10: Deploy and Monitor

- **Status:** designed on 2026-09-17 under the user's overnight delegation. The user decided the hosting on 2026-09-16: local Docker Compose plus a Cloudflare Tunnel, with no cloud billing. Every other decision here is the controller's recommendation and is marked (C).
- **Process:** lighter. A design document, 2–3 build tasks, and one review at the end.
- **Date:** 2026-09-17
- **Part of:** Dubai Real Estate ML Platform (sub-project 10 of 11)

## Goal

One `docker compose` command runs the whole product: the API and demo alongside Postgres, MLflow and Airflow. A public link comes up while the PC is on. Metrics are scraped and graphed. Data and model drift can be reported on demand. Everything stays free.

## Decisions

| Topic | Decision |
|---|---|
| (C) API image | `Dockerfile.api`: `python:3.11-slim` plus uv.<br>• Runs `uv sync --frozen --no-dev --no-install-package torch`, then installs the **CPU** torch 2.9.1 wheel from the PyTorch CPU index. This keeps the image well under the cu128 size, and the project memory notes that serving should use a CPU build.<br>• XGBoost 3.2's Linux wheel runs on CPU; predictors already force `device=cpu`.<br>• Copies the repository code, which must be importable because models are pyfuncs without `code_paths`.<br>• Runs as a non-root user with a `HEALTHCHECK` on `/health`.<br>• Command: `python -m api serve --host 0.0.0.0 --port 8000`. |
| (C) Demo image | `Dockerfile.demo`: slim, with only `streamlit`, `httpx` and `python-dotenv` installed via `uv pip` (it imports nothing else), plus `demo/`. `HEALTHCHECK` on `/_stcore/health`. |
| (C) Model cache | A named volume `hf_cache`, mounted at `/home/app/.cache/huggingface` in the API container, so MiniLM downloads once. |
| (C) Compose services | Added to `docker-compose.yml`:<br>• **`api`**: depends on healthy postgres and mlflow; env `POSTGRES_HOST=postgres`, `POSTGRES_PORT=5432`, `MLFLOW_TRACKING_URI=http://mlflow:5000`, plus `API_KEYS` etc. from `.env`; port `127.0.0.1:${API_PORT:-8000}:8000`.<br>• **`demo`**: env `DEMO_API_URL=http://api:8000` and `DEMO_API_KEY`; port `127.0.0.1:${DEMO_PORT:-8501}:8501`.<br>• All ports stay bound to 127.0.0.1. |
| Public link (user decision; mechanics (C)) | Profile `tunnel`, running `cloudflare/cloudflared` in one of two modes:<br>• **Quick tunnel** (default): `tunnel --no-autoupdate --url http://demo:8501`. No account needed; it gives a random `*.trycloudflare.com` URL, which `scripts/tunnel_url.py` reads from the container logs.<br>• **Named tunnel**: when `CLOUDFLARE_TUNNEL_TOKEN` is set, `tunnel --no-autoupdate run --token …`, for a stable hostname the user configures in the Cloudflare dashboard.<br>Only the **demo** is exposed. The API stays internal, and the demo holds the key server-side. |
| (C) Monitoring stack | Profile `monitoring`:<br>• **`prometheus`**: scrapes `api:8000/metrics` every 15 s with `authorization: credentials_file`. The key file is generated at start from `.env` by a small init container, and is never committed.<br>• **`grafana`**: anonymous viewer, an admin password from `.env`, and a provisioned datasource plus the dashboard `monitoring/grafana/dashboards/api.json` (request rate, error rate, p50/p95 latency per route, component up, model versions).<br>• Ports bound to localhost: 9090 and 3000. |
| (C) Drift reports | `python -m monitoring drift`, a small numeric report with no Evidently dependency (it would add many transitive packages for one HTML file):<br>• **Price-model inputs:** population stability index (PSI) and Kolmogorov–Smirnov per feature (size, bedrooms, price per m², reg type, sub kind), comparing the training period with the latest 3 months of sales.<br>• **Forecast residuals:** 3m model error on the newest rows whose targets have fully happened, against the gate's test MAPE.<br>• **Search traffic:** query-length and zero-result rate from the API logs, if a log file is given.<br>• **Output:** `data/monitoring/drift_<date>.json` plus a self-contained HTML table. PSI > 0.2 is flagged. |
| (C) Airflow | A DAG `dubimator_drift`, run manually (the same hand-trigger rule as ingestion), runs the drift report. The Airflow image gains no ML dependencies: the task runs `docker exec dubimator-api python -m monitoring drift` via a BashOperator. Because that needs the Docker socket, this is documented as optional. Simpler default: document a Task Scheduler entry and skip the DAG. **Ruling: no DAG; schedule it with Task Scheduler or cron next to `pipelines retrain`.** |
| (C) Phase 9 link | The CI `docker` job now builds `Dockerfile.api` and `Dockerfile.demo`, and `release.yml` pushes both images. |
| (C) Secrets | `.env.example` gains `API_PORT`, `DEMO_PORT`, `CLOUDFLARE_TUNNEL_TOKEN` (optional) and `GRAFANA_ADMIN_PASSWORD`. Real values live only in `.env`, and no secret is ever printed. |

## Verification (real)

1. `docker compose build api demo` succeeds; record the image sizes.
2. `docker compose up -d api demo` brings both to healthy.
3. The stack is reachable:
   - `curl localhost:8000/health` returns ok;
   - `/v1/ready`, called with the key, shows the components up;
   - `curl localhost:8501/_stcore/health` returns ok.
4. The live demo smoke run (Phase 8 Task 4's script) passes against the container URLs.
5. `docker compose --profile monitoring up -d`: Prometheus shows the API target as UP, and Grafana's health endpoint returns ok.
6. `docker compose --profile tunnel up -d cloudflared`: a quick tunnel URL appears in the logs, and `curl <url>/_stcore/health` returns ok.
   - If outbound access to Cloudflare is blocked, record that and leave the service defined.
   - Stop the tunnel afterwards, so nothing stays public while the user sleeps.
7. `python -m monitoring drift` writes its report.
8. Never run `docker compose down -v`. Stop only the services this phase added, or leave the core stack running as it was.

## Deliverables

- `Dockerfile.api`, `Dockerfile.demo` and `.dockerignore`
- The compose services and profiles
- `monitoring/` (the prometheus config template, grafana provisioning, the dashboard JSON, and a `drift.py` + `__main__.py` package with tests)
- `scripts/tunnel_url.py`
- A README "## Deploy and monitor" section: start and stop, tunnel modes, dashboards, drift, and troubleshooting
- An update to the architecture page

## Amendments during implementation

(none yet)
