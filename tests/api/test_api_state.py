import pytest

from api.errors import ApiError
from api.metrics import Metrics
from api.schemas import ForecastRequest, ListingCheckRequest
from api.settings import ApiSettings
from api.state import LoadedComponent, StateHolder, build_state

SETTINGS = ApiSettings.from_env({"API_KEYS": "k", "API_COMPONENTS": "price,search"})


def good(name):
    closed = []

    def loader(settings, context):
        loader.context = context
        context.setdefault("seen", []).append(name)
        return LoadedComponent(
            value=f"{name}-object", version="v1", close=lambda: closed.append(name)
        )

    loader.closed = closed
    return loader


def broken(settings, context):
    raise FileNotFoundError("no champion")


def test_build_state_isolates_failures():
    state = build_state(
        SETTINGS, {"price": good("price"), "search": broken, "forecast": good("forecast")}
    )
    assert state.get("price") == "price-object"
    with pytest.raises(ApiError) as caught:
        state.get("search")
    assert caught.value.status == 503
    assert caught.value.message == "search is unavailable: FileNotFoundError: no champion"
    with pytest.raises(ApiError, match="forecast is not enabled"):
        state.get("forecast")  # not in settings.components, so never loaded
    status = state.status()
    assert status["price"] == {"up": True, "version": "v1", "error": None}
    assert status["search"]["up"] is False
    assert "forecast" not in status


def test_loaders_share_a_context():
    price, search = good("price"), good("search")
    build_state(SETTINGS, {"price": price, "search": search})
    assert price.context is search.context
    assert price.context["seen"] == ["price", "search"]  # COMPONENTS order


def test_reload_swaps_then_closes_the_old_state():
    price = good("price")
    holder = StateHolder(build_state(SETTINGS, {"price": price, "search": broken}))
    old = holder.current
    new = holder.reload(SETTINGS, {"price": good("price"), "search": good("search")})
    assert holder.current is new and new is not old
    assert new.get("search") == "search-object"
    assert price.closed == ["price"]  # the old state's connections were closed after the swap


def test_locks_are_per_component():
    state = build_state(SETTINGS, {"price": good("price"), "search": good("search")})
    assert state.lock("search") is state.lock("search")
    assert state.lock("search") is not state.lock("price")


def test_metrics_render_prometheus_text():
    metrics = Metrics()
    metrics.observe("/v1/price", "POST", 200, 0.05)
    metrics.observe("/v1/price", "POST", 422, 0.01)
    metrics.set_components(build_state(SETTINGS, {"price": good("price"), "search": broken}))
    body, content_type = metrics.render()
    text = body.decode()
    assert content_type.startswith("text/plain")
    assert 'api_requests_total{method="POST",route="/v1/price",status="200"} 1.0' in text
    assert "api_request_seconds_bucket" in text
    assert 'api_component_up{component="search"} 0.0' in text
    assert 'api_model_info{component="price",version="v1"} 1.0' in text


def test_request_schemas():
    body = {"area": "Dubai Marina", "property_kind": "apartment", "status": "ready", "size_sqm": 80}
    assert ForecastRequest.model_validate({**body, "property_id": "p1"}).property_id == "p1"
    check = ListingCheckRequest.model_validate(
        {**body, "title": "Nice flat", "description": "Sea view", "asking_price_aed": 1e6,
         "area_id": 5, "photo_ids": [1, 2]}
    )  # fmt: skip
    assert check.photo_ids == [1, 2]
    with pytest.raises(ValueError):
        ListingCheckRequest.model_validate({**body, "title": "t", "description": "d",
                                            "asking_price_aed": 0, "area_id": 5})  # fmt: skip
    with pytest.raises(ValueError):
        ListingCheckRequest.model_validate({**body, "title": "t", "description": "d",
                                            "asking_price_aed": 1, "area_id": 5,
                                            "photo_ids": list(range(11))})  # fmt: skip
