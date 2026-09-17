"""Route tests for /v1/price, /v1/forecast, /v1/search and /v1/listings/*.

Every route needs the key; api_fixtures' fakes give predictable, fast responses so these
tests never load a real model or touch a database.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import psycopg2
import pytest
from api_fixtures import KEY, FakeChecker, FakeEngine, client, fake_loaders, settings

from api.state import LoadedComponent

PRICE_BODY = {
    "area": "Dubai Marina",
    "property_kind": "apartment",
    "status": "ready",
    "size_sqm": 80.0,
}
CHECK_BODY = {
    "title": "Nice flat",
    "description": "Sea view",
    "asking_price_aed": 1_000_000.0,
    "area_id": 5,
    "property_kind": "apartment",
    "status": "ready",
    "size_sqm": 80.0,
}


def _headers():
    return {"X-API-Key": KEY}


# --- price -----------------------------------------------------------------------------


def test_price_valid_body_returns_the_estimate():
    with client(settings(), fake_loaders()) as c:
        response = c.post("/v1/price", json=PRICE_BODY, headers=_headers())
    assert response.status_code == 200
    assert response.json()["estimate_aed"] == 1_000_000.0


def test_price_unknown_area_is_a_422_naming_the_field():
    with client(settings(), fake_loaders()) as c:
        response = c.post("/v1/price", json={**PRICE_BODY, "area": "Atlantis"}, headers=_headers())
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "area"


def test_price_negative_size_is_a_422_naming_the_field():
    with client(settings(), fake_loaders()) as c:
        response = c.post("/v1/price", json={**PRICE_BODY, "size_sqm": -1}, headers=_headers())
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "size_sqm"


# --- forecast --------------------------------------------------------------------------


def test_forecast_echoes_the_property_id():
    with client(settings(), fake_loaders()) as c:
        response = c.post(
            "/v1/forecast", json={**PRICE_BODY, "property_id": "p1"}, headers=_headers()
        )
    assert response.status_code == 200
    assert response.json()["property_id"] == "p1"


# --- search ----------------------------------------------------------------------------


def test_search_returns_200():
    with client(settings(), fake_loaders()) as c:
        response = c.get("/v1/search", params={"q": "flat"}, headers=_headers())
    assert response.status_code == 200
    assert response.json()["query"] == "flat"


def test_search_empty_query_is_a_422_naming_the_field():
    with client(settings(), fake_loaders()) as c:
        response = c.get("/v1/search", params={"q": ""}, headers=_headers())
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "q"


def test_search_k_too_large_is_a_422():
    with client(settings(), fake_loaders()) as c:
        response = c.get("/v1/search", params={"q": "flat", "k": 51}, headers=_headers())
    assert response.status_code == 422


def test_search_serializes_concurrent_calls():
    engine = FakeEngine(delay=0.05)

    def load(settings, context):
        return LoadedComponent(value=engine, version="v")

    with client(settings(), fake_loaders(search=load)) as c:

        def call(i):
            return c.get("/v1/search", params={"q": f"flat {i}"}, headers=_headers())

        with ThreadPoolExecutor(max_workers=5) as pool:
            responses = list(pool.map(call, range(5)))
    assert all(response.status_code == 200 for response in responses)
    assert engine.calls == 5
    assert engine.max_active == 1


# --- listings --------------------------------------------------------------------------


def test_listing_flags_found_and_missing():
    with client(settings(), fake_loaders()) as c:
        found = c.get("/v1/listings/1/flags", headers=_headers())
        missing = c.get("/v1/listings/404/flags", headers=_headers())
    assert found.status_code == 200
    assert found.json()["listing_id"] == 1
    assert missing.status_code == 404
    assert missing.json()["error"]["message"] == "listing 404 not found"


def test_listing_check_returns_200():
    with client(settings(), fake_loaders()) as c:
        response = c.post("/v1/listings/check", json=CHECK_BODY, headers=_headers())
    assert response.status_code == 200
    assert response.json() == {"duplicates": [], "flags": [], "notes": [], "model_versions": {}}


def test_listing_check_unknown_photo_id_is_a_422():
    with client(settings(), fake_loaders()) as c:
        response = c.post(
            "/v1/listings/check", json={**CHECK_BODY, "photo_ids": [5]}, headers=_headers()
        )
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "photo_ids"


def test_listing_check_rejects_an_infinite_number_sent_as_json():
    raw = json.dumps(CHECK_BODY).replace('"size_sqm": 80.0', '"size_sqm": 1e309')
    assert "1e309" in raw
    with client(settings(), fake_loaders()) as c:
        response = c.post(
            "/v1/listings/check",
            content=raw,
            headers={**_headers(), "Content-Type": "application/json"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "size_sqm"


def test_forecast_rejects_an_infinite_number_sent_as_json():
    raw = json.dumps(PRICE_BODY).replace('"size_sqm": 80.0', '"size_sqm": 1e309')
    with client(settings(), fake_loaders()) as c:
        response = c.post(
            "/v1/forecast",
            content=raw,
            headers={**_headers(), "Content-Type": "application/json"},
        )
    assert response.status_code == 422


def test_listing_check_rejects_unknown_fields():
    with client(settings(), fake_loaders()) as c:
        response = c.post(
            "/v1/listings/check", json={**CHECK_BODY, "area": "Dubai Marina"}, headers=_headers()
        )
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "area"


# --- lost database connections ----------------------------------------------------------


class _DroppedEngine:
    def search(self, text, k):
        raise psycopg2.OperationalError("server closed the connection unexpectedly")


class _DroppedChecker:
    def check(self, request):
        raise psycopg2.InterfaceError("connection already closed")

    def stored(self, listing_id):
        raise psycopg2.OperationalError("SSL SYSCALL error: EOF detected")


def _flaky(broken_factory, healthy_factory):
    """A loader whose first load gives a component with a dead connection."""
    calls = []

    def load(settings, context):
        calls.append(1)
        value = broken_factory() if len(calls) == 1 else healthy_factory()
        return LoadedComponent(value=value, version=f"v{len(calls)}")

    load.calls = calls
    return load


def _join_rebuilds(c):
    for thread in threading.enumerate():
        if thread.name.startswith("rebuild-"):
            thread.join(5)


def _assert_reconnecting(response, name):
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "unavailable"
    assert error["message"] == (
        f"{name} lost its database connection; it is reconnecting — retry shortly"
    )


def test_search_lost_connection_is_a_503_and_triggers_a_rebuild():
    search = _flaky(_DroppedEngine, FakeEngine)
    with client(settings(), fake_loaders(search=search)) as c:
        lost = c.get("/v1/search", params={"q": "flat"}, headers=_headers())
        _assert_reconnecting(lost, "search")
        _join_rebuilds(c)
        assert len(search.calls) == 2
        again = c.get("/v1/search", params={"q": "flat"}, headers=_headers())
        ready = c.get("/v1/ready", headers=_headers()).json()["components"]["search"]
    assert again.status_code == 200
    assert ready["version"] == "v2"


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/v1/listings/check", {"json": CHECK_BODY}),
        ("get", "/v1/listings/1/flags", {}),
    ],
)
def test_listings_lost_connection_is_a_503_and_triggers_a_rebuild(method, path, kwargs):
    listings = _flaky(_DroppedChecker, FakeChecker)
    with client(settings(), fake_loaders(listings=listings)) as c:
        lost = getattr(c, method)(path, headers=_headers(), **kwargs)
        _assert_reconnecting(lost, "listings")
        _join_rebuilds(c)
        assert len(listings.calls) == 2
        again = getattr(c, method)(path, headers=_headers(), **kwargs)
    assert again.status_code == 200


def test_ready_reports_a_component_whose_probe_fails_as_down():
    def search(settings, context):
        return LoadedComponent(value=FakeEngine(), version="v", check=lambda: False)

    with client(settings(), fake_loaders(search=search)) as c:
        components = c.get("/v1/ready", headers=_headers()).json()["components"]
        metrics = c.get("/metrics", headers=_headers()).text
    assert components["search"]["up"] is False
    assert components["search"]["error"] == "database connection lost"
    assert components["price"]["up"] is True
    assert 'api_component_up{component="search"} 0.0' in metrics


def test_a_503_never_shows_connection_details():
    def leaky(settings, context):
        raise psycopg2.OperationalError(
            'connection to server at "db.internal" (10.1.2.3), port 5432 failed: '
            'FATAL: password authentication failed for user "zest"'
        )

    with client(settings(), fake_loaders(search=leaky)) as c:
        response = c.get("/v1/search", params={"q": "flat"}, headers=_headers())
        ready = c.get("/v1/ready", headers=_headers()).text
    assert response.status_code == 503
    for text in (response.text, ready):
        assert "db.internal" not in text
        assert "10.1.2.3" not in text
        assert '"zest"' not in text
    assert "OperationalError: " in response.json()["error"]["message"]


# --- partial availability ---------------------------------------------------------------


def test_a_failed_component_gives_503_and_the_rest_still_work():
    def broken(settings, context):
        raise RuntimeError("no champion")

    with client(settings(), fake_loaders(price=broken)) as c:
        broken_response = c.post("/v1/price", json=PRICE_BODY, headers=_headers())
        forecast_response = c.post(
            "/v1/forecast", json={**PRICE_BODY, "property_id": "p1"}, headers=_headers()
        )
        search_response = c.get("/v1/search", params={"q": "flat"}, headers=_headers())
    assert broken_response.status_code == 503
    assert "no champion" in broken_response.json()["error"]["message"]
    assert forecast_response.status_code == 200
    assert search_response.status_code == 200
