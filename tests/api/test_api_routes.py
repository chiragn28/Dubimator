"""Route tests for /v1/price, /v1/forecast, /v1/search and /v1/listings/*.

Every route needs the key; api_fixtures' fakes give predictable, fast responses so these
tests never load a real model or touch a database.
"""

from concurrent.futures import ThreadPoolExecutor

from api_fixtures import KEY, client, fake_loaders, settings

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
    with client(settings(), fake_loaders()) as c:

        def call(i):
            return c.get("/v1/search", params={"q": f"flat {i}"}, headers=_headers())

        with ThreadPoolExecutor(max_workers=5) as pool:
            responses = list(pool.map(call, range(5)))
    assert all(response.status_code == 200 for response in responses)


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
