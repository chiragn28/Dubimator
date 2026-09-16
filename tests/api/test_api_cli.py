"""`python -m api serve|smoke`: tests/api/api_fixtures' fakes stand in for `default_loaders`
and `_sample_listing()` stands in for the database lookup, so nothing here touches a real
model or a real Postgres.
"""

import json

import pytest
from api_fixtures import KEY, fake_loaders

from api import __main__ as cli

SAMPLE = {
    "listing_id": 42,
    "title": "Spacious 1BR in Dubai Marina",
    "description": "Sea view, close to the metro.",
    "asking_price_aed": 1_450_000.0,
    "area_id": 10,
    "building_name": "Marina Tower",
    "project_name": None,
    "property_kind": "apartment",
    "status": "ready",
    "size_sqm": 85.0,
    "photo_ids": [],
}

REQUIRED_ENDPOINTS = {
    "GET /v1/ready", "POST /v1/price", "POST /v1/forecast", "GET /v1/search",
    f"GET /v1/listings/{SAMPLE['listing_id']}/flags", "POST /v1/listings/check", "GET /metrics",
}  # fmt: skip


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("API_KEYS", KEY)
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.delenv("API_COMPONENTS", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


def _results(tmp_path) -> list[dict]:
    payload = json.loads((tmp_path / "smoke.json").read_text(encoding="utf-8"))
    return payload["results"]


def test_smoke_all_endpoints_200_and_exits_0(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "default_loaders", fake_loaders)
    monkeypatch.setattr(cli, "_sample_listing", lambda: SAMPLE)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    exit_code = cli.main(["smoke"])

    assert exit_code == 0
    out = capsys.readouterr().out
    for endpoint in REQUIRED_ENDPOINTS:
        assert endpoint in out

    results = _results(tmp_path)
    assert {row["endpoint"] for row in results} == REQUIRED_ENDPOINTS
    assert all(row["status"] == 200 for row in results)


def test_smoke_exits_1_when_a_component_fails_to_load(monkeypatch, tmp_path):
    def broken_search(settings, context):
        raise RuntimeError("search index unavailable")

    monkeypatch.setattr(cli, "default_loaders", lambda: fake_loaders(search=broken_search))
    monkeypatch.setattr(cli, "_sample_listing", lambda: SAMPLE)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    exit_code = cli.main(["smoke"])

    assert exit_code == 1
    search_row = next(r for r in _results(tmp_path) if r["endpoint"] == "GET /v1/search")
    assert search_row["status"] == 503


def test_smoke_allows_404_on_the_deliberate_missing_listing_call(monkeypatch, tmp_path):
    """`api_fixtures.FakeChecker.stored` treats listing id 404 as the "unknown" sentinel
    (its own docstring), so pointing the sample at it exercises exactly the 404 that
    `smoke`'s exit-code rule carves out for the flags endpoint, without a real 404 ever
    coming from a genuinely missing listing.
    """
    monkeypatch.setattr(cli, "default_loaders", fake_loaders)
    monkeypatch.setattr(cli, "_sample_listing", lambda: {**SAMPLE, "listing_id": 404})
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    exit_code = cli.main(["smoke"])

    assert exit_code == 0
    flags_row = next(r for r in _results(tmp_path) if r["endpoint"].endswith("/flags"))
    assert flags_row["status"] == 404


def test_smoke_skips_listing_calls_when_no_sample_listing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "default_loaders", fake_loaders)
    monkeypatch.setattr(cli, "_sample_listing", lambda: None)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    exit_code = cli.main(["smoke"])

    assert exit_code == 0
    assert "skipping" in capsys.readouterr().out
    endpoints = {row["endpoint"] for row in _results(tmp_path)}
    assert "flags" not in " ".join(endpoints)
    assert "POST /v1/listings/check" not in endpoints
    assert endpoints == REQUIRED_ENDPOINTS - {
        f"GET /v1/listings/{SAMPLE['listing_id']}/flags", "POST /v1/listings/check",
    }  # fmt: skip


def test_smoke_writes_json_to_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "default_loaders", fake_loaders)
    monkeypatch.setattr(cli, "_sample_listing", lambda: SAMPLE)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "nested" / "api")

    cli.main(["smoke"])

    written = tmp_path / "nested" / "api" / "smoke.json"
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert "startup_seconds" in payload
    assert len(payload["results"]) == len(REQUIRED_ENDPOINTS)


def test_serve_calls_uvicorn_run_with_host_and_port(monkeypatch):
    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(cli, "default_loaders", fake_loaders)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    exit_code = cli.main(["serve", "--host", "0.0.0.0", "--port", "9000"])

    assert exit_code == 0
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000
    assert captured["workers"] == 1
    assert captured["log_config"] is None
    assert captured["app"] is not None


def test_serve_defaults_to_localhost_and_8000(monkeypatch):
    captured = {}

    def fake_run(app, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "default_loaders", fake_loaders)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    cli.main(["serve"])

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
