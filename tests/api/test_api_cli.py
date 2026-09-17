"""`python -m api serve|smoke`: tests/api/api_fixtures' fakes stand in for `default_loaders`
and `_sample_listing()` stands in for the database lookup, so nothing here touches a real
model or a real Postgres.
"""

import json

import pytest
from api_fixtures import KEY, FakeAreas, FakeEngine, fake_loaders

from api import __main__ as cli
from api.state import LoadedComponent

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
    "size_basis": "built_up",
    "bedrooms": 1,
    "photo_ids": [],
}

LISTING_ENDPOINTS = {f"GET /v1/listings/{SAMPLE['listing_id']}/flags", "POST /v1/listings/check"}
AREA_ENDPOINTS = {"GET /v1/areas", "GET /v1/areas/1/history"}  # FakeAreas' first area is 1
REQUIRED_ENDPOINTS = {
    "GET /v1/ready", "POST /v1/price", "POST /v1/forecast", "GET /v1/search",
    *LISTING_ENDPOINTS, *AREA_ENDPOINTS, "GET /metrics",
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
    assert all(row["first_status"] == row["warm_status"] == 200 for row in results)
    assert "first_status" in out and "warm_status" in out
    assert f"wrote {tmp_path / 'smoke.json'}" in out


def test_smoke_exits_1_when_a_component_fails_to_load(monkeypatch, tmp_path):
    def broken_search(settings, context):
        raise RuntimeError("search index unavailable")

    monkeypatch.setattr(cli, "default_loaders", lambda: fake_loaders(search=broken_search))
    monkeypatch.setattr(cli, "_sample_listing", lambda: SAMPLE)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    exit_code = cli.main(["smoke"])

    assert exit_code == 1
    search_row = next(r for r in _results(tmp_path) if r["endpoint"] == "GET /v1/search")
    assert search_row["first_status"] == search_row["warm_status"] == 503


def test_smoke_exits_1_on_a_404_listing(monkeypatch, tmp_path):
    """`api_fixtures.FakeChecker.stored` treats listing id 404 as "unknown": smoke needs a
    200 from every endpoint, so a sample listing the API can't find fails the run."""
    monkeypatch.setattr(cli, "default_loaders", fake_loaders)
    monkeypatch.setattr(cli, "_sample_listing", lambda: {**SAMPLE, "listing_id": 404})
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    exit_code = cli.main(["smoke"])

    assert exit_code == 1
    flags_row = next(r for r in _results(tmp_path) if r["endpoint"].endswith("/flags"))
    assert flags_row["warm_status"] == 404


def test_smoke_exits_1_when_only_the_warm_call_fails(monkeypatch, tmp_path, capsys):
    class OnceEngine(FakeEngine):
        def search(self, text, k):
            if self.calls:
                raise RuntimeError("second call breaks")
            return super().search(text, k)

    def search(settings, context):
        return LoadedComponent(value=OnceEngine(), version="v")

    monkeypatch.setattr(cli, "default_loaders", lambda: fake_loaders(search=search))
    monkeypatch.setattr(cli, "_sample_listing", lambda: SAMPLE)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    assert cli.main(["smoke"]) == 1
    row = next(r for r in _results(tmp_path) if r["endpoint"] == "GET /v1/search")
    assert (row["first_status"], row["warm_status"]) == (200, 500)


def test_smoke_requires_the_area_endpoints(monkeypatch, tmp_path):
    def broken(settings, context):
        raise RuntimeError("no homes")

    monkeypatch.setattr(cli, "default_loaders", lambda: fake_loaders(areas=broken))
    monkeypatch.setattr(cli, "_sample_listing", lambda: SAMPLE)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    assert cli.main(["smoke"]) == 1
    rows = {r["endpoint"]: r for r in _results(tmp_path)}
    assert rows["GET /v1/areas"]["first_status"] == 503
    assert not any("/history" in endpoint for endpoint in rows)  # no area id to ask for


def test_smoke_fails_when_there_is_no_area_to_ask_about(monkeypatch, tmp_path, capsys):
    class NoAreas(FakeAreas):
        def summary_records(self):
            return []

    def areas(settings, context):
        return LoadedComponent(value=NoAreas(), version="v")

    monkeypatch.setattr(cli, "default_loaders", lambda: fake_loaders(areas=areas))
    monkeypatch.setattr(cli, "_sample_listing", lambda: SAMPLE)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    assert cli.main(["smoke"]) == 1
    assert "no area" in capsys.readouterr().out


def test_smoke_skips_area_calls_when_areas_is_not_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "default_loaders", fake_loaders)
    monkeypatch.setattr(cli, "_sample_listing", lambda: SAMPLE)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    exit_code = cli.main(["smoke", "--components", "price", "forecast", "search", "listings"])

    assert exit_code == 0
    assert {r["endpoint"] for r in _results(tmp_path)} == REQUIRED_ENDPOINTS - AREA_ENDPOINTS


def test_check_body_mirrors_the_price_request_fields():
    body = cli._check_body(SAMPLE)
    assert body["size_basis"] == "built_up"
    assert body["bedrooms"] == 1
    assert "listing_id" not in body


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
    assert endpoints == REQUIRED_ENDPOINTS - LISTING_ENDPOINTS


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
