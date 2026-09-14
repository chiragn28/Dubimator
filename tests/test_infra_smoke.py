"""Infra smoke tests for the Phase 1 Docker Compose stack.

Run manually after `docker compose up -d`:
    uv run pytest tests/test_infra_smoke.py -v

Each test uses a short connection timeout and SKIPS (not fails) with a
clear message if the corresponding service isn't reachable, so running
this suite without Docker running gives a clear signal instead of a
mysterious hang or a red failure.
"""
import os

import psycopg2
import pytest
import requests
from dotenv import load_dotenv

load_dotenv()

CONNECT_TIMEOUT = 5


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def test_postgres_pgvector():
    port = _env("POSTGRES_PORT", "5432")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=port,
            user=_env("POSTGRES_USER", "zestimator"),
            password=_env("POSTGRES_PASSWORD", "changeme"),
            dbname=_env("POSTGRES_DB", "zestimator"),
            connect_timeout=CONNECT_TIMEOUT,
        )
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres not reachable on localhost:{port} — start it with `docker compose up -d postgres`. ({exc})")

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
    finally:
        conn.close()


def test_mlflow_health():
    try:
        resp = requests.get("http://localhost:5000/", timeout=CONNECT_TIMEOUT)
    except requests.exceptions.ConnectionError as exc:
        pytest.skip(f"MLflow not reachable on localhost:5000 — start it with `docker compose up -d mlflow`. ({exc})")

    assert resp.status_code == 200


def test_mlflow_proxies_artifacts():
    mlflow_url = _env("MLFLOW_TRACKING_URI", "http://localhost:5000").rstrip("/")
    try:
        resp = requests.get(
            f"{mlflow_url}/api/2.0/mlflow/experiments/get",
            params={"experiment_id": "0"},
            timeout=CONNECT_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        pytest.skip(f"MLflow not reachable at {mlflow_url} — start it with `docker compose up -d mlflow`. ({exc})")

    resp.raise_for_status()
    assert resp.json()["experiment"]["artifact_location"].startswith("mlflow-artifacts:")


def test_airflow_health():
    port = _env("AIRFLOW_PORT", "8080")
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=CONNECT_TIMEOUT)
    except requests.exceptions.ConnectionError as exc:
        pytest.skip(
            f"Airflow not reachable on localhost:{port} — start it with `docker compose up -d airflow`. "
            f"Note: Airflow standalone can take 30-60s to become healthy on first start. ({exc})"
        )

    assert resp.status_code == 200
