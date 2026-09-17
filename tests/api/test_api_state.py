import threading
import time

import pytest

from api.errors import ApiError, redact
from api.metrics import Metrics
from api.schemas import ForecastRequest, ListingCheckRequest
from api.settings import ApiSettings
from api.state import LoadedComponent, StateHolder, build_state

SETTINGS = ApiSettings.from_env({"API_KEYS": "k", "API_COMPONENTS": "price,search"})
PRICE_ONLY = ApiSettings.from_env({"API_KEYS": "k", "API_COMPONENTS": "price"})


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
    check_body = {key: value for key, value in body.items() if key != "area"}
    check = ListingCheckRequest.model_validate(
        {**check_body, "title": "Nice flat", "description": "Sea view", "asking_price_aed": 1e6,
         "area_id": 5, "photo_ids": [1, 2]}
    )  # fmt: skip
    assert check.photo_ids == [1, 2]
    with pytest.raises(ValueError):
        ListingCheckRequest.model_validate({**check_body, "title": "t", "description": "d",
                                            "asking_price_aed": 0, "area_id": 5})  # fmt: skip
    with pytest.raises(ValueError):
        ListingCheckRequest.model_validate({**check_body, "title": "t", "description": "d",
                                            "asking_price_aed": 1, "area_id": 5,
                                            "photo_ids": list(range(11))})  # fmt: skip


def test_listing_check_request_forbids_extra_fields_and_non_finite_numbers():
    body = {"title": "t", "description": "d", "asking_price_aed": 1e6, "area_id": 5,
            "property_kind": "apartment", "status": "ready", "size_sqm": 80}  # fmt: skip
    ListingCheckRequest.model_validate(body)
    with pytest.raises(ValueError):
        ListingCheckRequest.model_validate({**body, "area": "Dubai Marina"})
    forecast_body = {"area": "Dubai Marina", "property_kind": "apartment", "status": "ready"}
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ListingCheckRequest.model_validate({**body, "size_sqm": bad})
        with pytest.raises(ValueError):
            ForecastRequest.model_validate({**forecast_body, "size_sqm": bad})


# --- reload locking (F1) -----------------------------------------------------------------


def test_reload_waits_for_in_flight_operations_before_closing():
    events = []

    def loader(settings, context):
        return LoadedComponent(value="p", version="v", close=lambda: events.append("close"))

    holder = StateHolder(build_state(PRICE_ONLY, {"price": loader}))
    old = holder.current
    inside = threading.Event()

    def operation():
        with old.lock("price"):
            inside.set()
            time.sleep(0.2)
            events.append("operation done")

    worker = threading.Thread(target=operation)
    worker.start()
    assert inside.wait(5)
    holder.reload(PRICE_ONLY, {"price": loader})
    worker.join(5)
    assert events == ["operation done", "close"]


def test_reload_while_a_reload_runs_is_a_409():
    release = threading.Event()
    entered = threading.Event()

    def slow(settings, context):
        entered.set()
        release.wait(5)
        return LoadedComponent(value="p", version="v")

    holder = StateHolder(build_state(PRICE_ONLY, {"price": good("price")}))
    worker = threading.Thread(target=holder.reload, args=(PRICE_ONLY, {"price": slow}))
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(ApiError) as caught:
            holder.reload(PRICE_ONLY, {"price": good("price")})
        assert (caught.value.status, caught.value.code) == (409, "conflict")
    finally:
        release.set()
        worker.join(5)


# --- liveness probes and per-component rebuild (F2) ---------------------------------------


def probed(check):
    def loader(settings, context):
        return LoadedComponent(value="s", version="v", check=check)

    return loader


def test_live_status_reports_a_failed_probe_as_down():
    state = build_state(SETTINGS, {"price": good("price"), "search": probed(lambda: False)})
    assert state.status()["search"]["up"] is True  # probes run only when asked
    live = state.status(live=True)
    assert live["search"] == {"up": False, "version": "v", "error": "database connection lost"}
    assert live["price"]["up"] is True


def test_live_status_treats_a_raising_probe_as_down():
    def boom():
        raise RuntimeError("socket gone")

    state = build_state(SETTINGS, {"price": good("price"), "search": probed(boom)})
    assert state.status(live=True)["search"]["error"] == "database connection lost"


def test_live_status_skips_the_probe_while_the_component_is_busy():
    calls = []

    def check():
        calls.append(1)
        return False

    state = build_state(SETTINGS, {"price": good("price"), "search": probed(check)})
    with state.lock("search"):
        assert state.status(live=True)["search"]["up"] is True
    assert calls == []


def test_metrics_use_live_status():
    alive = [True]
    metrics = Metrics()
    state = build_state(SETTINGS, {"price": good("price"), "search": probed(lambda: alive[0])})
    metrics.set_components(state)
    assert 'api_component_up{component="search"} 1.0' in metrics.render()[0].decode()
    alive[0] = False
    metrics.set_components(state)
    assert 'api_component_up{component="search"} 0.0' in metrics.render()[0].decode()


def test_rebuild_swaps_one_component_and_reuses_the_embedder():
    closed, contexts = [], []
    counts = {"search": 0}

    def search(settings, context):
        counts["search"] += 1
        n = counts["search"]
        contexts.append(dict(context))
        context.setdefault("embedder", object())
        return LoadedComponent(value=f"search-{n}", version=f"v{n}", close=lambda: closed.append(n))

    price = good("price")
    holder = StateHolder(build_state(SETTINGS, {"price": price, "search": search}))
    state = holder.current
    embedder = state.context["embedder"]
    holder.rebuild("search", SETTINGS, {"price": price, "search": search})
    assert holder.current is state  # only the one component was swapped
    assert state.get("search") == "search-2"
    assert state.versions["search"] == "v2"
    assert contexts[-1] == {"embedder": embedder}  # a fresh context, the shared embedder
    assert closed == [1]
    assert price.closed == []  # the other component is untouched
    state.close()
    assert closed == [1, 2]


def test_rebuild_closes_the_old_component_only_after_in_flight_work():
    events = []

    def search(settings, context):
        return LoadedComponent(value="s", version="v", close=lambda: events.append("close"))

    holder = StateHolder(build_state(SETTINGS, {"price": good("price"), "search": search}))
    state = holder.current
    inside = threading.Event()

    def operation():
        with state.lock("search"):
            inside.set()
            time.sleep(0.2)
            events.append("operation done")

    worker = threading.Thread(target=operation)
    worker.start()
    assert inside.wait(5)
    holder.rebuild("search", SETTINGS, {"search": search})
    worker.join(5)
    assert events == ["operation done", "close"]


def test_a_failed_rebuild_keeps_the_old_component():
    holder = StateHolder(build_state(SETTINGS, {"price": good("price"), "search": good("search")}))
    holder.rebuild("search", SETTINGS, {"search": broken})
    assert holder.current.get("search") == "search-object"


def test_background_rebuilds_run_one_at_a_time_per_component():
    release = threading.Event()
    calls = []

    def slow(settings, context):
        calls.append(1)
        release.wait(5)
        return LoadedComponent(value="s2", version="v2")

    holder = StateHolder(build_state(SETTINGS, {"price": good("price"), "search": good("search")}))
    first = holder.rebuild_in_background("search", SETTINGS, {"search": slow})
    second = holder.rebuild_in_background("search", SETTINGS, {"search": slow})
    assert first is not None and second is None
    release.set()
    first.join(5)
    assert calls == [1]
    assert holder.current.get("search") == "s2"
    third = holder.rebuild_in_background("search", SETTINGS, {"search": slow})
    assert third is not None  # the per-component guard is released afterwards
    third.join(5)


# --- redaction ----------------------------------------------------------------------------

LEAKY = (
    "connection failed: host=db.internal port=5432 user=zest password='hunter 2' "
    "via postgresql://zest:hunter2@db.internal:5432/zest "
)


def test_redact_hides_connection_details():
    text = redact(ConnectionError(LEAKY))
    assert text.startswith("ConnectionError: connection failed: host=***")
    for secret in ("hunter", "db.internal", "zest"):
        assert secret not in text


def test_redact_truncates_to_120_characters_of_message():
    text = redact(ValueError("x" * 300))
    assert text == "ValueError: " + "x" * 120
    assert redact(RuntimeError("no champion")) == "RuntimeError: no champion"


def test_load_errors_are_redacted_and_truncated():
    def leaky(settings, context):
        raise ConnectionError(LEAKY + "x" * 300)

    state = build_state(PRICE_ONLY, {"price": leaky})
    error = state.status()["price"]["error"]
    assert error.startswith("ConnectionError: ")
    assert len(error) <= len("ConnectionError: ") + 120
    for secret in ("hunter", "db.internal"):
        assert secret not in error
    with pytest.raises(ApiError) as caught:
        state.get("price")
    assert "hunter" not in caught.value.message
