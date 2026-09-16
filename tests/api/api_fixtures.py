"""Shared fakes for API tests (Task 6 builds its route tests on these same fakes)."""

import threading
import warnings
from contextlib import contextmanager
from datetime import date

from starlette.exceptions import StarletteDeprecationWarning

# Importing `starlette.testclient` trips two library-internal warnings on this stack,
# neither of which is ours to fix, so both are silenced right at the import instead of
# failing `-W error`:
# 1. This installed Starlette prefers a package called `httpx2` for its TestClient and
#    only warns (doesn't fail) when falling back to plain `httpx`, which is what this
#    project actually has (a transitive dependency of huggingface-hub). httpx2 isn't a
#    project dependency, and adding one is out of scope for this task.
# 2. Starlette's testclient module itself references the deprecated
#    `anyio.abc.BlockingPortal` alias at import time (anyio's own suggested fix,
#    `anyio.from_thread.BlockingPortal`, is Starlette's call to make, not ours).
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*httpx2.*", category=StarletteDeprecationWarning)
    warnings.filterwarnings("ignore", message=r".*BlockingPortal.*", category=DeprecationWarning)
    from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import ApiSettings
from api.state import LoadedComponent
from listings.check import UnknownPhotos
from models.price.predictor import PriceInputError

KEY = "test-key"


def settings(**env):
    return ApiSettings.from_env({"API_KEYS": KEY, "API_RATE_LIMIT_PER_MINUTE": "1000", **env})


@contextmanager
def client(settings, loaders):
    """A TestClient over `create_app`, used as a context manager so the lifespan runs.

    `raise_server_exceptions=False`: without it, Starlette's ServerErrorMiddleware
    re-raises an unhandled exception into the test after sending our 500 response
    (its own documented behaviour, so a real server can still log it), which would
    blow up any test that exercises the generic Exception handler.
    """
    app = create_app(settings, loaders)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


class _FakeEstimate:
    """A stand-in for the Phase 3 PriceEstimate: same `model_dump(mode="json")` shape."""

    def model_dump(self, mode="json"):
        return {
            "estimate_aed": 1000000.0,
            "range_80": [900000.0, 1100000.0],
            "range_95": [800000.0, 1200000.0],
            "price_per_sqm_aed": 12500.0,
            "location_level": "area",
            "confidence": "medium",
            "market_label": "fair",
        }


class FakePrice:
    def predict_one(self, request):
        if getattr(request, "area", None) == "Atlantis":
            raise PriceInputError("area", "unknown area 'Atlantis'")
        return _FakeEstimate()


class FakeForecaster:
    def forecast(self, body_dict):
        return {
            "forecast_3m": {"estimate_aed": 1000000.0},
            "property_id": body_dict.get("property_id"),
        }


class _FakeSearchResult:
    def __init__(self, query, k):
        self.query = query
        self.k = k

    def to_dict(self):
        return {"query": self.query, "hits": [], "k": self.k}


class FakeEngine:
    """Records the thread that called `search`, so tests can check the lock is used."""

    def __init__(self):
        self.threads = []

    def search(self, text, k):
        self.threads.append(threading.current_thread())
        return _FakeSearchResult(text, k)


class FakeChecker:
    def check(self, request):
        if getattr(request, "photo_ids", None) == [5]:
            raise UnknownPhotos([5])
        return {"duplicates": [], "flags": [], "notes": [], "model_versions": {}}

    def stored(self, listing_id):
        """None for an unknown listing (the route's 404 case); a dict otherwise.

        `listing_id == 404` is the fixture's "unknown" sentinel, so a test reads as
        `checker.stored(404)` gives 404.
        """
        if listing_id == 404:
            return None
        return {"listing_id": listing_id, "detect_run_id": 1, "duplicates": [], "flags": []}


class FakeAreas:
    """A stand-in for `api.areas.AreaStats`: one area with a history a 404 sentinel."""

    data_end = date(2023, 3, 17)

    def summary_records(self):
        return [
            {
                "area_id": 1,
                "name": "Dubai Marina",
                "median_ppsm_12m": 15250.0,
                "change_12m": 0.032,
                "sales_12m": 412,
            }
        ]

    def history_records(self, area_id):
        # `area_id == 404` is the fixture's "unknown" sentinel (matching FakeChecker.stored).
        if area_id == 404:
            return None
        return {
            "area_id": area_id,
            "name": "Dubai Marina",
            "data_end": self.data_end.isoformat(),
            "series": [
                {
                    "market_kind": "apartment",
                    "points": [{"month": "2023-03", "median_ppsm": 15250.0, "sales": 40}],
                }
            ],
        }


def fake_loaders(**overrides):
    """Loaders producing the fakes above, each versioned by how many times it has run.

    Versions change on every `build_state` call (startup, then each reload), so a
    reload test can tell the new state apart from the old one.
    """
    counts = {"price": 0, "forecast": 0, "search": 0, "listings": 0, "areas": 0}

    def _loader(name, value_factory):
        def load(settings, context):
            counts[name] += 1
            return LoadedComponent(value=value_factory(), version=f"{name}-v{counts[name]}")

        return load

    loaders = {
        "price": _loader("price", FakePrice),
        "forecast": _loader("forecast", FakeForecaster),
        "search": _loader("search", FakeEngine),
        "listings": _loader("listings", FakeChecker),
        "areas": _loader("areas", FakeAreas),
    }
    loaders.update(overrides)
    return loaders
