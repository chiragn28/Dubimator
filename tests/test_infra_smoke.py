"""Infra smoke tests for the Phase 1 Docker Compose stack.

Run after `docker compose up -d --wait`:
    uv run pytest tests/test_infra_smoke.py -v

A test SKIPS only when its service isn't reachable at all (nothing
listening), so running without Docker gives a clear signal instead of a
hang. A reachable service that misbehaves (wrong credentials, unhealthy
components, misconfigured artifact store) FAILS.
"""

import os
import socket

import psycopg2
import pytest
import requests
from dotenv import load_dotenv

load_dotenv()  # no override: shell env wins, matching docker compose's own precedence

CONNECT_TIMEOUT = 5
HOST = "127.0.0.1"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _mlflow_url() -> str:
    return _env("MLFLOW_TRACKING_URI", f"http://{HOST}:5000").rstrip("/")


def _http_get_or_skip(url: str, service: str, **kwargs) -> requests.Response:
    try:
        return requests.get(url, timeout=CONNECT_TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        pytest.skip(
            f"{service} not reachable at {url} — start the stack with `docker compose up -d --wait`. ({exc})"
        )


def test_postgres_pgvector():
    port = int(_env("POSTGRES_PORT", "5432"))
    try:
        socket.create_connection((HOST, port), timeout=CONNECT_TIMEOUT).close()
    except OSError as exc:
        pytest.skip(
            f"Postgres not reachable at {HOST}:{port} — start the stack with `docker compose up -d --wait`. ({exc})"
        )

    conn = psycopg2.connect(
        host=HOST,
        port=port,
        user=_env("POSTGRES_USER", "zestimator"),
        password=_env("POSTGRES_PASSWORD", "changeme"),
        dbname=_env("POSTGRES_DB", "zestimator"),
        connect_timeout=CONNECT_TIMEOUT,
    )
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
    finally:
        conn.close()


def test_mlflow_health():
    resp = _http_get_or_skip(f"{_mlflow_url()}/health", "MLflow")
    assert resp.status_code == 200


def test_mlflow_proxies_artifacts():
    resp = _http_get_or_skip(
        f"{_mlflow_url()}/api/2.0/mlflow/experiments/get", "MLflow", params={"experiment_id": "0"}
    )
    resp.raise_for_status()
    assert resp.json()["experiment"]["artifact_location"].startswith("mlflow-artifacts:")


def test_airflow_health():
    port = _env("AIRFLOW_PORT", "8080")
    resp = _http_get_or_skip(f"http://{HOST}:{port}/health", "Airflow")
    assert resp.status_code == 200
    health = resp.json()
    assert health["metadatabase"]["status"] == "healthy"
    assert health["scheduler"]["status"] == "healthy"
