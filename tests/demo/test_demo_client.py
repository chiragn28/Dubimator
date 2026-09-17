"""Tests for demo.client — the demo's HTTP client for the Phase 7 API."""

import ast
import json
from pathlib import Path

import httpx
import pytest

from demo.client import ApiClient, ApiProblem, client_from_env, get_client

BASE_URL = "http://testserver"


def recording_transport(response_factory):
    """A MockTransport that also records every request it receives."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return response_factory(request)

    return httpx.MockTransport(handler), calls


def json_response(status_code, payload, headers=None):
    def factory(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload, headers=headers or {})

    return factory


# ---------------------------------------------------------------------------
# Request shape per method
# ---------------------------------------------------------------------------


def test_ready_request():
    transport, calls = recording_transport(json_response(200, {"status": "ok"}))
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    result = client.ready()

    assert result == {"status": "ok"}
    assert len(calls) == 1
    assert calls[0].method == "GET"
    assert calls[0].url.path == "/v1/ready"


def test_price_request():
    transport, calls = recording_transport(json_response(200, {"estimate": 100}))
    client = ApiClient(BASE_URL, key="secret", transport=transport)
    body = {"area_id": 1, "size_sqm": 120}

    result = client.price(body)

    assert result == {"estimate": 100}
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/v1/price"
    assert json.loads(calls[0].content) == body


def test_forecast_request():
    transport, calls = recording_transport(json_response(200, {"forecast": []}))
    client = ApiClient(BASE_URL, key="secret", transport=transport)
    body = {"area_id": 1, "horizon_months": 12}

    result = client.forecast(body)

    assert result == {"forecast": []}
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/v1/forecast"
    assert json.loads(calls[0].content) == body


def test_search_request_default_k():
    transport, calls = recording_transport(json_response(200, {"results": []}))
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    result = client.search("villa in marina")

    assert result == {"results": []}
    assert calls[0].method == "GET"
    assert calls[0].url.path == "/v1/search"
    assert calls[0].url.params["q"] == "villa in marina"
    assert calls[0].url.params["k"] == "10"


def test_search_request_explicit_k():
    transport, calls = recording_transport(json_response(200, {"results": []}))
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    client.search("townhouse", k=5)

    assert calls[0].url.params["q"] == "townhouse"
    assert calls[0].url.params["k"] == "5"


def test_listing_flags_request():
    transport, calls = recording_transport(json_response(200, {"flags": []}))
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    result = client.listing_flags(42)

    assert result == {"flags": []}
    assert calls[0].method == "GET"
    assert calls[0].url.path == "/v1/listings/42/flags"


def test_check_listing_request():
    transport, calls = recording_transport(json_response(200, {"duplicates": []}))
    client = ApiClient(BASE_URL, key="secret", transport=transport)
    body = {"title": "3BR villa", "price": 2_000_000}

    result = client.check_listing(body)

    assert result == {"duplicates": []}
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/v1/listings/check"
    assert json.loads(calls[0].content) == body


def test_areas_request():
    transport, calls = recording_transport(json_response(200, {"areas": []}))
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    result = client.areas()

    assert result == {"areas": []}
    assert calls[0].method == "GET"
    assert calls[0].url.path == "/v1/areas"


def test_area_history_request():
    transport, calls = recording_transport(json_response(200, {"history": []}))
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    result = client.area_history(7)

    assert result == {"history": []}
    assert calls[0].method == "GET"
    assert calls[0].url.path == "/v1/areas/7/history"


# ---------------------------------------------------------------------------
# X-API-Key header
# ---------------------------------------------------------------------------


def test_api_key_header_sent():
    transport, calls = recording_transport(json_response(200, {}))
    client = ApiClient(BASE_URL, key="my-secret-key", transport=transport)

    client.ready()

    assert calls[0].headers["X-API-Key"] == "my-secret-key"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_envelope_error_maps_to_api_problem():
    payload = {
        "error": {"code": "not_found", "message": "area not found", "field": "area_id"},
        "request_id": "req-123",
    }
    transport, _ = recording_transport(json_response(404, payload))
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    with pytest.raises(ApiProblem) as exc_info:
        client.area_history(999)

    problem = exc_info.value
    assert problem.status == 404
    assert problem.code == "not_found"
    assert problem.message == "area not found"
    assert problem.field == "area_id"
    assert problem.request_id == "req-123"


def test_envelope_error_request_id_falls_back_to_header():
    payload = {"error": {"code": "bad_request", "message": "bad input", "field": None}}
    transport, _ = recording_transport(
        json_response(400, payload, headers={"X-Request-ID": "hdr-req-1"})
    )
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    with pytest.raises(ApiProblem) as exc_info:
        client.ready()

    assert exc_info.value.request_id == "hdr-req-1"
    assert exc_info.value.field is None


def test_non_json_error_gives_http_error():
    def factory(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    transport, _ = recording_transport(factory)
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    with pytest.raises(ApiProblem) as exc_info:
        client.ready()

    problem = exc_info.value
    assert problem.status == 502
    assert problem.code == "http_error"
    assert problem.message == "HTTP 502"


def test_connect_error_gives_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    with pytest.raises(ApiProblem) as exc_info:
        client.ready()

    problem = exc_info.value
    assert problem.status == 0
    assert problem.code == "unreachable"
    assert BASE_URL in problem.message


def test_timeout_exception_gives_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    transport = httpx.MockTransport(handler)
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    with pytest.raises(ApiProblem) as exc_info:
        client.ready()

    problem = exc_info.value
    assert problem.status == 0
    assert problem.code == "unreachable"
    assert BASE_URL in problem.message


@pytest.mark.parametrize(
    "error",
    [httpx.ReadError, httpx.RemoteProtocolError, httpx.WriteError],
    ids=lambda e: e.__name__,
)
def test_other_transport_errors_give_unreachable(error):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("connection dropped", request=request)

    client = ApiClient(BASE_URL, key="secret", transport=httpx.MockTransport(handler))

    with pytest.raises(ApiProblem) as exc_info:
        client.ready()

    assert exc_info.value.code == "unreachable"
    assert exc_info.value.status == 0


def test_non_json_success_gives_bad_response():
    def factory(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy page</html>", headers={"X-Request-ID": "r9"})

    transport, _ = recording_transport(factory)
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    with pytest.raises(ApiProblem) as exc_info:
        client.ready()

    problem = exc_info.value
    assert problem.status == 200
    assert problem.code == "bad_response"
    assert problem.request_id == "r9"


def test_non_object_success_gives_bad_response():
    transport, _ = recording_transport(json_response(200, [1, 2, 3]))
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    with pytest.raises(ApiProblem) as exc_info:
        client.areas()

    assert exc_info.value.code == "bad_response"


@pytest.mark.parametrize(
    "payload",
    [["not", "a", "dict"], "just a string", {"error": "plain string"}, {"detail": "Not Found"}],
    ids=["list", "string", "error-string", "fastapi-detail"],
)
def test_error_bodies_that_are_not_envelopes_give_http_error(payload):
    transport, _ = recording_transport(
        json_response(404, payload, headers={"X-Request-ID": "hdr-2"})
    )
    client = ApiClient(BASE_URL, key="secret", transport=transport)

    with pytest.raises(ApiProblem) as exc_info:
        client.ready()

    problem = exc_info.value
    assert problem.status == 404
    assert problem.code == "http_error"
    assert problem.message == "HTTP 404"
    assert problem.request_id == "hdr-2"


def test_no_key_raises_without_sending_request():
    transport, calls = recording_transport(json_response(200, {}))
    client = ApiClient(BASE_URL, key=None, transport=transport)

    with pytest.raises(ApiProblem) as exc_info:
        client.ready()

    problem = exc_info.value
    assert problem.status == 0
    assert problem.code == "no_key"
    assert problem.message == "DEMO_API_KEY is not set"
    assert calls == []


def test_no_key_empty_string_also_raises():
    transport, calls = recording_transport(json_response(200, {}))
    client = ApiClient(BASE_URL, key="", transport=transport)

    with pytest.raises(ApiProblem) as exc_info:
        client.areas()

    assert exc_info.value.code == "no_key"
    assert calls == []


# ---------------------------------------------------------------------------
# client_from_env
# ---------------------------------------------------------------------------


def test_client_from_env_defaults():
    client = client_from_env(environ={})

    assert isinstance(client, ApiClient)
    assert client._base_url == "http://127.0.0.1:8000"
    assert client._key is None


def test_client_from_env_reads_environ():
    environ = {"DEMO_API_URL": "http://example.test:9000", "DEMO_API_KEY": "abc123"}

    client = client_from_env(environ=environ)

    assert client._base_url == "http://example.test:9000"
    assert client._key == "abc123"


# ---------------------------------------------------------------------------
# Trailing slash handling
# ---------------------------------------------------------------------------


def test_trailing_slash_on_base_url_is_handled():
    transport, calls = recording_transport(json_response(200, {}))
    client = ApiClient("http://testserver/", key="secret", transport=transport)

    client.ready()

    assert str(calls[0].url) == "http://testserver/v1/ready"
    assert client._base_url == "http://testserver"


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------


def test_get_client_is_cached():
    get_client.cache_clear()
    try:
        first = get_client()
        second = get_client()
        assert first is second
    finally:
        get_client.cache_clear()


# ---------------------------------------------------------------------------
# Source scan: the demo must never import the DB/model/search internals
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES = {"psycopg2", "mlflow", "models", "search", "listings"}


def _imported_top_level_modules(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0]


def test_demo_never_imports_db_or_model_internals():
    demo_root = Path(__file__).resolve().parents[2] / "demo"
    violations = []

    for path in demo_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _imported_top_level_modules(tree):
            if module in FORBIDDEN_MODULES:
                violations.append(f"{path}: imports {module!r}")

    assert not violations, "\n".join(violations)
