import io
import json
import logging

from api_fixtures import KEY, client, fake_loaders, settings

from api.logging import JsonFormatter


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
    with client(settings(), fake_loaders()) as c:
        before = c.get("/v1/ready", headers=_headers()).json()["components"]["price"]["version"]
        reloaded = c.post("/v1/admin/reload", headers=_headers())
        after = reloaded.json()["components"]["price"]["version"]
    assert reloaded.status_code == 200
    assert after != before


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
