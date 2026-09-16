"""A fake ApiClient for testing the demo's Streamlit pages without the network.

Mirrors the Phase 7 API response shapes from the Phase 8 design (see
`docs/superpowers/specs/2026-09-17-phase8-demo-design.md`) closely enough for
the pages under test, without importing anything from `api/*`.
"""

from __future__ import annotations

from demo.client import ApiProblem

READY_PAYLOAD = {
    "price": {"up": True, "version": "price-v3", "error": None},
    "forecast": {"up": True, "version": "forecast-v1", "error": None},
    "search": {"up": True, "version": "search-v2", "error": None},
    "listings": {"up": True, "version": "listings-v1", "error": None},
    "areas": {"up": True, "version": "areas-v1", "error": None},
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
        "confidence": "medium",
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
        {"feature": "area trend", "impact_pct": 0.008},
        {"feature": "size", "impact_pct": -0.003},
    ],
    "exclusions_applied": ["outlier listings excluded"],
    "model_versions": {"price": "price-v3", "forecast": "forecast-v1"},
}

SEARCH_PAYLOAD = {
    "query": "",
    "parsed": {
        "area_ids": [1],
        "bedrooms": 3,
        "budget_max": 3_000_000,
        "property_type": "villa",
    },
    "hits": [
        {
            "listing_id": 501,
            "score": 0.93,
            "title": "3BR Villa, Dubai Marina",
            "area_name": "Dubai Marina",
            "bedrooms": 3,
            "asking_price_aed": 2_850_000,
            "size_sqm": 210.0,
            "reasons": ["matches area", "within budget", "3 bedrooms"],
            "duplicate": True,
            "fraud_flags": ["photo reuse"],
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
        },
    ],
    "notes": [],
    "timings_ms": {"parse": 1.2, "retrieve": 8.4, "rank": 2.1},
    "ranker": "learned_v2",
}

LISTING_FLAGS_PAYLOAD = {
    "listing_id": 501,
    "detect_run_id": "run-2023-03-17",
    "duplicates": [
        {"listing_id": 777, "score": 0.95, "signals": ["photo hash", "same phone"]},
    ],
    "flags": [
        {"flag": "photo_reuse", "detail": "3 photos match listing 777"},
    ],
}

CHECK_LISTING_PAYLOAD = {
    "duplicates": [
        {"listing_id": 777, "score": 0.95, "signals": ["photo hash"]},
    ],
    "flags": [
        {"flag": "bait_price", "detail": "asking price is 40% below the area median"},
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
    any single call fail without touching the others.
    """

    def __init__(self, fail: dict[str, ApiProblem] | None = None) -> None:
        self._fail = fail or {}

    def _maybe_fail(self, name: str) -> None:
        if name in self._fail:
            raise self._fail[name]

    def ready(self) -> dict:
        self._maybe_fail("ready")
        return READY_PAYLOAD

    def price(self, body: dict) -> dict:
        self._maybe_fail("price")
        return PRICE_PAYLOAD

    def forecast(self, body: dict) -> dict:
        self._maybe_fail("forecast")
        return FORECAST_PAYLOAD

    def search(self, q: str, k: int = 10) -> dict:
        self._maybe_fail("search")
        payload = dict(SEARCH_PAYLOAD)
        payload["query"] = q
        return payload

    def listing_flags(self, listing_id) -> dict:
        self._maybe_fail("listing_flags")
        payload = dict(LISTING_FLAGS_PAYLOAD)
        payload["listing_id"] = listing_id
        return payload

    def check_listing(self, body: dict) -> dict:
        self._maybe_fail("check_listing")
        return CHECK_LISTING_PAYLOAD

    def areas(self) -> dict:
        self._maybe_fail("areas")
        return AREAS_PAYLOAD

    def area_history(self, area_id) -> dict:
        self._maybe_fail("area_history")
        key = int(area_id)
        return AREA_HISTORY_PAYLOADS.get(key, AREA_HISTORY_PAYLOADS[1])
