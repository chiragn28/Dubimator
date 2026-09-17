import io
import json
import logging
import threading
from contextlib import contextmanager

from api_fixtures import ADMIN_KEY, KEY, admin_settings, client, fake_loaders, settings

from api.logging import JsonFormatter
from api.state import LoadedComponent


def _headers(key=KEY):
    return {"X-API-Key": key}


def test_health_is_public_and_carries_a_request_id():
    with client(settings(), fake_loaders()) as c:
        response = c.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_ready_requires_a_key():
    with client(settings(), fake_loaders()) as c:
        no_key = c.get("/v1/ready")
        assert no_key.status_code == 401
        body = no_key.json()
        assert body["error"]["code"] == "unauthorized"
        assert body["request_id"] == no_key.headers["X-Request-ID"]

        wrong_key = c.get("/v1/ready", headers=_headers("nope"))
        assert wrong_key.status_code == 401

        header_key = c.get("/v1/ready", headers=_headers())
        assert header_key.status_code == 200

        bearer = c.get("/v1/ready", headers={"Authorization": f"Bearer {KEY}"})
        assert bearer.status_code == 200


def test_ready_body_lists_components_and_survives_a_broken_one():
    def broken(settings, context):
        raise RuntimeError("no champion")

    with client(settings(), fake_loaders(search=broken)) as c:
        response = c.get("/v1/ready", headers=_headers())
    assert response.status_code == 200
    components = response.json()["components"]
    assert components["price"]["up"] is True
    assert components["price"]["version"]
    assert components["search"]["up"] is False
    assert "no champion" in components["search"]["error"]


def test_rate_limit_returns_429_with_retry_after():
    with client(settings(API_RATE_LIMIT_PER_MINUTE="2"), fake_loaders()) as c:
        first = c.get("/v1/ready", headers=_headers())
        second = c.get("/v1/ready", headers=_headers())
        third = c.get("/v1/ready", headers=_headers())
    assert (first.status_code, second.status_code) == (200, 200)
    assert third.status_code == 429
    assert third.json()["error"]["code"] == "rate_limited"
    retry_after = int(third.headers["Retry-After"])
    assert retry_after >= 1


def test_request_id_is_echoed_or_generated():
    with client(settings(), fake_loaders()) as c:
        echoed = c.get("/health", headers={"X-Request-ID": "abc-123"})
        replaced = c.get("/health", headers={"X-Request-ID": "not a valid id!!"})
    assert echoed.headers["X-Request-ID"] == "abc-123"
    assert replaced.headers["X-Request-ID"] != "not a valid id!!"
    assert replaced.headers["X-Request-ID"]


def test_metrics_needs_a_key_and_reports_requests_and_components():
    with client(settings(), fake_loaders()) as c:
        no_key = c.get("/metrics")
        assert no_key.status_code == 401

        c.get("/v1/ready", headers=_headers())
        response = c.get("/metrics", headers=_headers())
    assert response.status_code == 200
    text = response.text
    assert "api_requests_total" in text
    assert 'route="/v1/ready"' in text
    assert "api_component_up" in text


def test_reload_rebuilds_the_state_with_new_versions():
    with client(admin_settings(), fake_loaders()) as c:
        before = c.get("/v1/ready", headers=_headers()).json()["components"]["price"]["version"]
        reloaded = c.post("/v1/admin/reload", headers=_headers(ADMIN_KEY))
        after = reloaded.json()["components"]["price"]["version"]
    assert reloaded.status_code == 200
    assert after != before


def test_reload_needs_an_admin_key():
    with client(admin_settings(), fake_loaders()) as c:
        plain = c.post("/v1/admin/reload", headers=_headers())
        missing = c.post("/v1/admin/reload")
    assert plain.status_code == 403
    assert plain.json()["error"]["code"] == "forbidden"
    assert plain.json()["error"]["message"] == "an admin key is required"
    assert missing.status_code == 401


def test_reload_is_disabled_without_admin_keys():
    with client(settings(), fake_loaders()) as c:
        response = c.post("/v1/admin/reload", headers=_headers())
    assert response.status_code == 403
    assert response.json()["error"] == {
        "code": "forbidden",
        "message": "admin reload is disabled: set API_ADMIN_KEYS",
        "field": None,
    }


def test_a_second_reload_while_one_runs_is_a_409():
    entered, release = threading.Event(), threading.Event()
    loads = []

    def price(settings, context):
        loads.append(1)
        if len(loads) == 2:  # the first reload (the lifespan load is the first call)
            entered.set()
            release.wait(5)
        return LoadedComponent(value=object(), version=f"v{len(loads)}")

    with client(admin_settings(), fake_loaders(price=price)) as c:
        results = {}

        def first():
            results["first"] = c.post("/v1/admin/reload", headers=_headers(ADMIN_KEY))

        worker = threading.Thread(target=first)
        worker.start()
        try:
            assert entered.wait(5)
            second = c.post("/v1/admin/reload", headers=_headers(ADMIN_KEY))
        finally:
            release.set()
            worker.join(5)
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "conflict"
    assert results["first"].status_code == 200


def test_unknown_path_and_unhandled_exception_map_to_the_error_envelope():
    def boom():
        raise RuntimeError("kaboom")

    with client(settings(), fake_loaders()) as c:
        c.app.get("/boom")(boom)  # test-only route

        missing = c.get("/nope")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "not_found"

        broken = c.get("/boom")
    assert broken.status_code == 500
    body = broken.json()
    assert body["error"]["code"] == "internal"
    assert body["error"]["message"] == "internal error"
    assert "kaboom" not in broken.text
    assert "Traceback" not in broken.text


@contextmanager
def captured(name="api"):
    """JSON lines from the api logger tree (it doesn't propagate to root, so no caplog)."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)


def _records(stream) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_an_unhandled_exception_is_logged_with_its_traceback_once():
    def boom():
        raise RuntimeError("kaboom")

    with client(settings(), fake_loaders()) as c:
        c.app.get("/boom")(boom)  # test-only route
        with captured() as stream:
            c.get("/boom")
    records = _records(stream)
    assert sum("exception" in record for record in records) == 1
    (access,) = [r for r in records if r["logger"] == "api.access"]
    assert access["level"] == "ERROR" and access["status"] == 500
    assert "exception" not in access


def test_a_rate_limited_request_is_attributed_to_its_key():
    with client(settings(API_RATE_LIMIT_PER_MINUTE="1"), fake_loaders()) as c:
        c.get("/v1/ready", headers=_headers())
        with captured("api.access") as stream:
            limited = c.get("/v1/ready", headers=_headers())
    assert limited.status_code == 429
    (record,) = _records(stream)
    assert record["status"] == 429
    assert record["key_id"] and record["key_id"] != "open"


def test_wrong_method_is_a_405_with_its_own_code():
    with client(settings(), fake_loaders()) as c:
        response = c.post("/health")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_redoc_is_off_and_docs_stay_on():
    with client(settings(), fake_loaders()) as c:
        assert c.get("/redoc").status_code == 404
        assert c.get("/docs").status_code == 200


def test_open_mode_logs_a_warning():
    open_settings = settings(API_KEYS="", API_ALLOW_NO_KEYS="1")
    with captured() as stream, client(open_settings, fake_loaders()):
        pass
    warnings = [r for r in _records(stream) if r["level"] == "WARNING"]
    assert any("authentication is disabled" in r["message"] for r in warnings)


def test_a_request_body_never_reaches_the_logs():
    marker = "SECRET-BODY-MARKER-7f3a"
    body = {
        "title": marker, "description": marker, "asking_price_aed": 1.0, "area_id": 5,
        "property_kind": "apartment", "status": "ready", "size_sqm": 80.0,
    }  # fmt: skip
    with client(settings(), fake_loaders()) as c, captured() as stream:
        response = c.post("/v1/listings/check", json=body, headers=_headers())
        invalid = c.post("/v1/listings/check", json={**body, "size_sqm": -1}, headers=_headers())
    assert (response.status_code, invalid.status_code) == (200, 422)
    raw = stream.getvalue()
    assert raw  # the access lines were written
    assert marker not in raw


def test_access_log_line_is_json_and_never_carries_the_key():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    access_logger = logging.getLogger("api.access")
    access_logger.addHandler(handler)
    try:
        with client(settings(), fake_loaders()) as c:
            c.get("/v1/ready", headers=_headers())
    finally:
        access_logger.removeHandler(handler)

    raw = stream.getvalue()
    assert KEY not in raw
    lines = [line for line in raw.splitlines() if line.strip()]
    assert lines
    record = json.loads(lines[-1])
    assert record["status"] == 200
    assert record["path"] == "/v1/ready"
    assert record["key_id"]


def test_keyless_mode_allows_ready_with_no_key():
    open_settings = settings(API_KEYS="", API_ALLOW_NO_KEYS="1")
    with client(open_settings, fake_loaders()) as c:
        response = c.get("/v1/ready")
    assert response.status_code == 200
