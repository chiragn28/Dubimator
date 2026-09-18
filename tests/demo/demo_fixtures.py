"""A fake ApiClient for testing the demo's Streamlit pages without the network.

Mirrors the Phase 7 API response shapes from the Phase 8 design (see
`docs/superpowers/specs/2026-09-17-phase8-demo-design.md`) closely enough for
the pages under test. Request bodies, though, are validated with the API's real
request models (`PriceRequest`, `ForecastRequest`, `ListingCheckRequest`), so a page that
builds a body the API would reject fails its test. Only the test side imports `api`/`models`;
the demo package itself never does.
"""

from __future__ import annotations

import io
from collections import defaultdict

from api.schemas import ForecastRequest, ListingCheckRequest
from demo.client import ApiProblem
from models.price.predictor import PriceRequest


def _tiny_jpeg() -> bytes:
    """A 4x3 JPEG, so pages that render a photo get bytes Streamlit can actually decode."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (200, 160, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


TINY_JPEG = _tiny_jpeg()

READY_PAYLOAD = {
    "status": "ok",
    "components": {
        "price": {"up": True, "version": "price-v3", "error": None},
        "forecast": {"up": True, "version": "forecast-v1", "error": None},
        "search": {"up": True, "version": "search-v2", "error": None},
        "listings": {"up": True, "version": "listings-v1", "error": None},
        "areas": {"up": True, "version": "areas-v1", "error": None},
    },
}

PRICE_PAYLOAD = {
    "estimate_aed": 1_455_490.0,
    "range_80": [1_320_000.0, 1_590_000.0],
    "range_95": [1_240_000.0, 1_680_000.0],
    "price_per_sqm_aed": 12_900.0,
    "location_level": "area",
    "confidence": "high",
    "market_label": "villa",
    "asking_vs_estimate_pct": 0.021,
    "flags": [],
    "as_of": "2023-03-17",
    "model_version": "price-v3",
}

FORECAST_PAYLOAD = {
    "property_id": None,
    "as_of": "2023-03-17",
    "current_estimate_aed": 1_455_490.0,
    "current_range_80": [1_320_000.0, 1_590_000.0],
    "forecast_3m": {
        "point": 1_468_000.0,
        "ci_low": 1_390_000.0,
        "ci_high": 1_545_000.0,
        "confidence": "MEDIUM",
    },
    "forecast_1y": {
        "status": "not_deployed",
        "reason": "1-year forecasting is not yet deployed",
    },
    "forecast_3y": {
        "status": "not_deployed",
        "reason": "3-year forecasting is not yet deployed",
    },
    "key_drivers": [
        "Area prices rose 14% over the last 12 months (+0.8% to the 3m forecast)",
        "Size 85 m² (-0.3% to the 3m forecast)",
    ],
    "exclusions_applied": ["outlier listings excluded"],
    "model_versions": {"price": "price-v3", "forecast": "forecast-v1"},
}

SEARCH_PAYLOAD = {
    "parsed": {
        "area_ids": [1],
        "bedrooms": 3,
        "budget_max": 3_000_000,
        "property_type": "villa",
    },
    "results": [
        {
            "listing_id": 501,
            "score": 0.93,
            "title": "3BR Villa, Dubai Marina",
            "area_name": "Dubai Marina",
            "bedrooms": 3,
            "asking_price_aed": 2_850_000,
            "size_sqm": 210.0,
            "reasons": ["matches area", "within budget", "3 bedrooms"],
            "duplicates_hidden": 2,
        },
        {
            "listing_id": 502,
            "score": 0.88,
            "title": "3BR Villa, Dubai Marina - Sea View",
            "area_name": "Dubai Marina",
            "bedrooms": 3,
            "asking_price_aed": 2_950_000,
            "size_sqm": 225.0,
            "reasons": ["matches area", "within budget"],
            "duplicates_hidden": 0,
        },
    ],
    "notes": [],
    "timings_ms": {"parse": 1.2, "retrieve": 8.4, "rank": 2.1},
    "ranker": "learned_v2",
}

# Shaped like the real API: `detail` and `signals` are objects, not prose (see
# listings/check.py's STORED_PAIRS_SQL / STORED_FLAGS_SQL).
LISTING_FLAGS_PAYLOAD = {
    "listing_id": 501,
    "detect_run_id": 4,
    "duplicates": [
        {
            "listing_id": 777,
            "score": 0.95,
            "signals": {
                "same_area": 1,
                "same_building": 1,
                "same_project": 0,
                "same_agent": 0,
                "bedrooms_equal": 1,
                "days_apart": 11.0,
                "text_cosine": 0.986,
                "image_max_cosine": 0.99,
                "shared_photo_count": 3,
                "abs_log_price_ratio": 0.0114,
            },
        },
    ],
    "flags": [
        {"flag": "photo_reuse", "detail": {"photo_set_id": 185, "listings": 122, "areas": 33}},
    ],
}

CHECK_LISTING_PAYLOAD = {
    "duplicates": [
        {"listing_id": 777, "score": 0.95, "signals": {"same_area": 1, "text_cosine": 0.93}},
    ],
    "flags": [
        {
            "flag": "bait_price",
            "detail": {
                "asking_price_aed": 350000.0,
                "range_80_low": 418330.5,
                "estimate_aed": 576331.0,
                "below_low_pct": 16.3,
            },
        },
    ],
    "notes": ["checked against the live corpus"],
    "model_versions": {"listings": "listings-v1"},
}

AREAS_PAYLOAD = {
    "data_end": "2023-03-17",
    "areas": [
        {
            "area_id": 1,
            "name": "Dubai Marina",
            "median_ppsm_12m": 15250.0,
            "change_12m": 0.032,
            "sales_12m": 412,
        },
        {
            "area_id": 2,
            "name": "Dubai Hills Estate",
            "median_ppsm_12m": 13100.0,
            "change_12m": -0.011,
            "sales_12m": 265,
        },
        {
            "area_id": 3,
            "name": "Jumeirah Village Circle",
            "median_ppsm_12m": 9800.0,
            "change_12m": 0.061,
            "sales_12m": 341,
        },
    ],
}


def _history_points(base_ppsm: float) -> list[dict]:
    months = [f"2022-{m:02d}" for m in range(1, 13)] + ["2023-01", "2023-02", "2023-03"]
    return [
        {"month": month, "median_ppsm": base_ppsm * (1 + 0.002 * i), "sales": 20 + i}
        for i, month in enumerate(months)
    ]


AREA_HISTORY_PAYLOADS = {
    1: {
        "area_id": 1,
        "name": "Dubai Marina",
        "data_end": "2023-03-17",
        "series": [
            {"market_kind": "apartment", "points": _history_points(15000.0)},
        ],
    },
    2: {
        "area_id": 2,
        "name": "Dubai Hills Estate",
        "data_end": "2023-03-17",
        "series": [
            {"market_kind": "villa", "points": _history_points(13000.0)},
            {"market_kind": "villa_plot", "points": _history_points(9000.0)},
        ],
    },
    3: {
        "area_id": 3,
        "name": "Jumeirah Village Circle",
        "data_end": "2023-03-17",
        "series": [
            {"market_kind": "apartment", "points": _history_points(9800.0)},
        ],
    },
}


class FakeClient:
    """Stands in for `demo.client.ApiClient` in page tests.

    `fail` maps a method name (e.g. `"forecast"`) to the `ApiProblem` it
    should raise instead of returning its usual payload, so a test can make
    any single call fail without touching the others. `areas_payload`
    replaces the `/v1/areas` response.

    Every call is recorded in `calls[name]` (the body for POSTs, the argument
    otherwise). POST bodies are validated against the API's request models
    first, so an invalid body raises `pydantic.ValidationError`.
    """

    def __init__(
        self,
        fail: dict[str, ApiProblem] | None = None,
        areas_payload: dict | None = None,
    ) -> None:
        self._fail = fail or {}
        self._areas_payload = areas_payload if areas_payload is not None else AREAS_PAYLOAD
        self.calls: dict[str, list] = defaultdict(list)

    def _maybe_fail(self, name: str) -> None:
        if name in self._fail:
            raise self._fail[name]

    def ready(self) -> dict:
        self.calls["ready"].append(None)
        self._maybe_fail("ready")
        return READY_PAYLOAD

    def price(self, body: dict) -> dict:
        PriceRequest.model_validate(body)
        self.calls["price"].append(body)
        self._maybe_fail("price")
        return PRICE_PAYLOAD

    def forecast(self, body: dict) -> dict:
        ForecastRequest.model_validate(body)
        self.calls["forecast"].append(body)
        self._maybe_fail("forecast")
        return FORECAST_PAYLOAD

    def search(self, q: str, k: int = 10) -> dict:
        self._maybe_fail("search")
        payload = dict(SEARCH_PAYLOAD)
        payload["query"] = q
        return payload

    def listing_flags(self, listing_id) -> dict:
        self.calls["listing_flags"].append(listing_id)
        self._maybe_fail("listing_flags")
        payload = dict(LISTING_FLAGS_PAYLOAD)
        payload["listing_id"] = listing_id
        return payload

    def listing_photos(self, listing_id) -> dict:
        self.calls["listing_photos"].append(listing_id)
        self._maybe_fail("listing_photos")
        return {
            "listing_id": listing_id,
            "photos": [{"photo_id": 529, "room": "frontal", "position": 0}],
        }

    def photo(self, photo_id) -> bytes | None:
        self.calls["photo"].append(photo_id)
        self._maybe_fail("photo")
        return TINY_JPEG

    def check_listing(self, body: dict) -> dict:
        ListingCheckRequest.model_validate(body)
        self.calls["check_listing"].append(body)
        self._maybe_fail("check_listing")
        return CHECK_LISTING_PAYLOAD

    def areas(self) -> dict:
        self.calls["areas"].append(None)
        self._maybe_fail("areas")
        return self._areas_payload

    def area_history(self, area_id) -> dict:
        self.calls["area_history"].append(area_id)
        self._maybe_fail("area_history")
        key = int(area_id)
        return AREA_HISTORY_PAYLOADS.get(key, AREA_HISTORY_PAYLOADS[1])
