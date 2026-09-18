"""Tests for the demo's Streamlit pages, driven with `AppTest` against a fake client."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import demo.client
from demo.client import ApiProblem
from demo.ui import EXAMPLE_LISTING_IDS
from tests.demo.demo_fixtures import FakeClient

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"

PAGE_FILES = [
    DEMO_DIR / "app.py",
    DEMO_DIR / "pages" / "1_Price_and_forecast.py",
    DEMO_DIR / "pages" / "2_Search.py",
    DEMO_DIR / "pages" / "3_Listing_check.py",
    DEMO_DIR / "pages" / "4_Area_explorer.py",
    DEMO_DIR / "pages" / "5_About.py",
]


@pytest.fixture(autouse=True)
def _fresh_areas_cache():
    """`cached_areas()` uses `st.cache_data`, which is a process-wide cache.

    Clear it around every test so one test's FakeClient (or failure) can't
    leak cached `/v1/areas` data into the next.
    """
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()


def patch_client(monkeypatch, fail=None, areas_payload=None):
    """Patch `demo.client.get_client` so pages get a `FakeClient` with the given failures."""
    client = FakeClient(fail=fail, areas_payload=areas_payload)
    monkeypatch.setattr(demo.client, "get_client", lambda: client)
    return client


# ---------------------------------------------------------------------------
# 1. Every page, plus app.py, runs without exception.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page_file", PAGE_FILES, ids=lambda p: p.name)
def test_page_runs_without_exception(monkeypatch, page_file):
    patch_client(monkeypatch)

    at = AppTest.from_file(str(page_file))
    at.run()

    assert not at.exception


# ---------------------------------------------------------------------------
# 2. Price page: submitting the form shows an estimate containing "AED".
# ---------------------------------------------------------------------------


def test_price_form_submission_shows_estimate(monkeypatch):
    patch_client(monkeypatch)

    at = AppTest.from_file(str(DEMO_DIR / "pages" / "1_Price_and_forecast.py"))
    at.run()
    at.button[0].click().run()

    assert not at.exception
    assert "AED" in at.metric[0].value


# ---------------------------------------------------------------------------
# 3. Price page shows the 1y not-deployed reason text.
# ---------------------------------------------------------------------------


def test_price_page_shows_1y_reason(monkeypatch):
    patch_client(monkeypatch)

    at = AppTest.from_file(str(DEMO_DIR / "pages" / "1_Price_and_forecast.py"))
    at.run()
    at.button[0].click().run()

    assert not at.exception
    table_text = at.table[0].value.to_string()
    assert "1-year forecasting is not yet deployed" in table_text


# ---------------------------------------------------------------------------
# 4. Search page: typing a query shows 2 results.
# ---------------------------------------------------------------------------


def test_search_shows_results(monkeypatch):
    patch_client(monkeypatch)

    at = AppTest.from_file(str(DEMO_DIR / "pages" / "2_Search.py"))
    at.run()
    at.text_input(key="search_query").set_value("3br villa in dubai marina").run()

    assert not at.exception
    headings = [el.value for el in at.subheader]
    assert any("2 result" in heading for heading in headings)
    # the real API returns "results" with duplicates_hidden (search/engine.py SearchResult.to_dict)
    assert any("2 likely duplicate" in el.value for el in at.warning)
    assert any("learned_v2" in el.value for el in at.caption)


# ---------------------------------------------------------------------------
# 5. Listing check, existing tab: an id with flags shows the flag names.
# ---------------------------------------------------------------------------


def test_listing_check_existing_tab_shows_flag_names(monkeypatch):
    patch_client(monkeypatch)

    at = AppTest.from_file(str(DEMO_DIR / "pages" / "3_Listing_check.py"))
    at.run()
    at.text_input(key="existing_listing_id").set_value("501").run()
    at.button(key="lookup_existing").click().run()

    assert not at.exception
    markdown_text = " ".join(el.value for el in at.markdown)
    assert "photo_reuse" in markdown_text


# ---------------------------------------------------------------------------
# 5b. Listing check: an example button fills the ID box and looks that ID up,
#     so the page is usable without knowing any listing ID.
# ---------------------------------------------------------------------------


def test_listing_check_example_button_looks_up_that_id(monkeypatch):
    client = patch_client(monkeypatch)

    at = AppTest.from_file(str(DEMO_DIR / "pages" / "3_Listing_check.py"))
    at.run()
    example_id, _ = EXAMPLE_LISTING_IDS[0]
    at.button(key=f"example_{example_id}").click().run()

    assert not at.exception
    assert client.calls["listing_flags"] == [example_id]
    assert at.text_input(key="existing_listing_id").value == example_id


def test_listing_check_offers_one_example_per_outcome(monkeypatch):
    patch_client(monkeypatch)

    at = AppTest.from_file(str(DEMO_DIR / "pages" / "3_Listing_check.py"))
    at.run()

    assert not at.exception
    keys = {el.key for el in at.button}
    assert {f"example_{i}" for i, _ in EXAMPLE_LISTING_IDS} <= keys
    # Distinct ids, and each says what it demonstrates.
    ids = [i for i, _ in EXAMPLE_LISTING_IDS]
    assert len(set(ids)) == len(ids) >= 2
    assert all(outcome for _, outcome in EXAMPLE_LISTING_IDS)


# ---------------------------------------------------------------------------
# 6. Area explorer: the table has 3 rows, and selecting an area renders a chart.
# ---------------------------------------------------------------------------


def test_area_explorer_table_and_chart(monkeypatch):
    patch_client(monkeypatch)

    at = AppTest.from_file(str(DEMO_DIR / "pages" / "4_Area_explorer.py"))
    at.run()

    assert not at.exception
    assert len(at.dataframe[0].value) == 3

    at.selectbox(key=at.selectbox[0].key).select("Dubai Hills Estate").run()

    assert not at.exception
    assert len(at.get("vega_lite_chart")) >= 1


# ---------------------------------------------------------------------------
# 7. A failing call shows the API's message and request id.
# ---------------------------------------------------------------------------


def test_forecast_failure_shows_message_and_request_id(monkeypatch):
    problem = ApiProblem(
        503, "unavailable", "forecast is unavailable: try again shortly", None, "rid1"
    )
    patch_client(monkeypatch, fail={"forecast": problem})

    at = AppTest.from_file(str(DEMO_DIR / "pages" / "1_Price_and_forecast.py"))
    at.run()
    at.button[0].click().run()

    assert not at.exception
    error_text = " ".join(el.value for el in at.error)
    caption_text = " ".join(el.value for el in at.caption)
    assert "forecast is unavailable: try again shortly" in error_text
    assert "rid1" in caption_text


# ---------------------------------------------------------------------------
# 8. A `no_key` problem shows the setup hint.
# ---------------------------------------------------------------------------


def test_no_key_shows_setup_hint(monkeypatch):
    problem = ApiProblem(0, "no_key", "DEMO_API_KEY is not set")
    patch_client(monkeypatch, fail={"ready": problem})

    at = AppTest.from_file(str(DEMO_DIR / "app.py"))
    at.run()

    assert not at.exception
    info_text = " ".join(el.value for el in at.info)
    assert "DEMO_API_URL" in info_text
    assert "DEMO_API_KEY" in info_text


def test_unauthorized_shows_setup_hint(monkeypatch):
    problem = ApiProblem(401, "unauthorized", "missing or invalid API key", None, "rid2")
    patch_client(monkeypatch, fail={"ready": problem})

    at = AppTest.from_file(str(DEMO_DIR / "app.py"))
    at.run()

    assert not at.exception
    info_text = " ".join(el.value for el in at.info)
    assert "DEMO_API_KEY" in info_text


def test_app_renders_nested_component_status(monkeypatch):
    patch_client(monkeypatch)

    at = AppTest.from_file(str(DEMO_DIR / "app.py"))
    at.run()

    assert not at.exception
    markdown_text = " ".join(el.value for el in at.markdown)
    assert "**price** (price-v3)" in markdown_text
    assert "**status**" not in markdown_text


def test_ready_is_cached_across_reruns(monkeypatch):
    client = patch_client(monkeypatch)

    at = AppTest.from_file(str(DEMO_DIR / "app.py"))
    at.run()
    at.run()

    assert not at.exception
    assert len(client.calls["ready"]) == 1


# ---------------------------------------------------------------------------
# Price page: the body the page sends is a valid PriceRequest / ForecastRequest.
# ---------------------------------------------------------------------------

PRICE_PAGE = DEMO_DIR / "pages" / "1_Price_and_forecast.py"


def test_price_form_sends_a_valid_body(monkeypatch):
    client = patch_client(monkeypatch)

    at = AppTest.from_file(str(PRICE_PAGE))
    at.run()
    at.selectbox(key="price_area").select("Dubai Marina").run()
    at.text_input(key="price_building").set_value("Marina Gate 1").run()
    at.button[0].click().run()

    assert not at.exception
    body = client.calls["price"][0]
    assert body == {
        "area_id": 1,
        "property_kind": "apartment",
        "status": "ready",
        "size_sqm": 120.0,
        "size_basis": "built_up",
        "bedrooms": 2,
        "building": "Marina Gate 1",
    }
    assert client.calls["forecast"][0] == body


def test_price_form_offers_plot_basis_only_for_villas(monkeypatch):
    client = patch_client(monkeypatch)

    at = AppTest.from_file(str(PRICE_PAGE))
    at.run()
    assert not [el for el in at.selectbox if el.key == "price_size_basis"]

    at.selectbox(key="price_kind").select("villa").run()
    at.selectbox(key="price_size_basis").select("plot").run()
    at.selectbox(key="price_status").select("off_plan").run()
    at.selectbox(key="price_bedrooms").select_index(0).run()  # "not specified"
    at.text_input(key="price_project").set_value("Emaar South").run()
    at.button[0].click().run()

    assert not at.exception
    body = client.calls["price"][0]
    assert body["property_kind"] == "villa"
    assert body["size_basis"] == "plot"
    assert body["status"] == "off_plan"
    assert body["project"] == "Emaar South"
    assert "bedrooms" not in body
    assert "building" not in body


def test_price_form_uses_free_text_area_when_areas_are_unavailable(monkeypatch):
    client = patch_client(monkeypatch, fail={"areas": ApiProblem(503, "unavailable", "down")})

    at = AppTest.from_file(str(PRICE_PAGE))
    at.run()
    at.text_input(key="price_area_name").set_value("Dubai Marina").run()
    at.button[0].click().run()

    assert not at.exception
    body = client.calls["price"][0]
    assert body["area"] == "Dubai Marina"
    assert "area_id" not in body


# ---------------------------------------------------------------------------
# Listing check, new tab: the body is a valid ListingCheckRequest.
# ---------------------------------------------------------------------------

LISTING_PAGE = DEMO_DIR / "pages" / "3_Listing_check.py"


def _fill_new_listing(at, photo_ids=""):
    at.text_input(key="new_title").set_value("2BR apartment with marina view").run()
    at.text_area(key="new_description").set_value("Bright, high floor, vacant.").run()
    at.selectbox(key="new_area").select("Dubai Marina").run()
    at.text_input(key="new_building").set_value("Marina Gate 1").run()
    at.text_input(key="new_photo_ids").set_value(photo_ids).run()


def test_new_listing_sends_a_valid_body(monkeypatch):
    client = patch_client(monkeypatch)

    at = AppTest.from_file(str(LISTING_PAGE))
    at.run()
    _fill_new_listing(at, photo_ids="11, 12,13")
    at.button(key="check_new_listing").click().run()

    assert not at.exception
    body = client.calls["check_listing"][0]
    assert body == {
        "title": "2BR apartment with marina view",
        "description": "Bright, high floor, vacant.",
        "asking_price_aed": 1_000_000.0,
        "area_id": 1,
        "building_name": "Marina Gate 1",
        "property_kind": "apartment",
        "status": "ready",
        "size_sqm": 100.0,
        "size_basis": "built_up",
        "bedrooms": 2,
        "photo_ids": [11, 12, 13],
    }
    markdown_text = " ".join(el.value for el in at.markdown)
    assert "bait_price" in markdown_text


def test_new_listing_rejects_bad_photo_ids_without_calling_the_api(monkeypatch):
    client = patch_client(monkeypatch)

    at = AppTest.from_file(str(LISTING_PAGE))
    at.run()
    _fill_new_listing(at, photo_ids="11, twelve")
    at.button(key="check_new_listing").click().run()

    assert not at.exception
    assert client.calls["check_listing"] == []
    assert any("photo" in el.value.lower() for el in at.error)


def test_new_listing_rejects_more_than_ten_photo_ids(monkeypatch):
    client = patch_client(monkeypatch)

    at = AppTest.from_file(str(LISTING_PAGE))
    at.run()
    _fill_new_listing(at, photo_ids=",".join(str(i) for i in range(11)))
    at.button(key="check_new_listing").click().run()

    assert not at.exception
    assert client.calls["check_listing"] == []
    assert any("photo" in el.value.lower() for el in at.error)


def test_new_listing_requires_title_and_description(monkeypatch):
    client = patch_client(monkeypatch)

    at = AppTest.from_file(str(LISTING_PAGE))
    at.run()
    at.button(key="check_new_listing").click().run()

    assert not at.exception
    assert client.calls["check_listing"] == []
    assert at.error


def test_new_listing_has_no_prefill_and_explains_why(monkeypatch):
    patch_client(monkeypatch)

    at = AppTest.from_file(str(LISTING_PAGE))
    at.run()

    assert not [el for el in at.text_input if el.key == "prefill_listing_id"]
    caption_text = " ".join(el.value for el in at.caption)
    assert "doesn't return the listing text" in caption_text


def test_new_listing_needs_the_areas_list(monkeypatch):
    client = patch_client(monkeypatch, fail={"areas": ApiProblem(503, "unavailable", "down")})

    at = AppTest.from_file(str(LISTING_PAGE))
    at.run()

    assert not at.exception
    assert client.calls["check_listing"] == []
    assert any(el.value == "down" for el in at.error)


# ---------------------------------------------------------------------------
# Area explorer: numeric columns, a missing change, cached history.
# ---------------------------------------------------------------------------

AREA_PAGE = DEMO_DIR / "pages" / "4_Area_explorer.py"


def test_area_explorer_keeps_numbers_numeric_and_handles_missing_change(monkeypatch):
    areas_payload = {
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
                "area_id": 4,
                "name": "Al Barsha South",
                "median_ppsm_12m": 8000.0,
                "change_12m": None,
                "sales_12m": 51,
            },
        ],
    }
    patch_client(monkeypatch, areas_payload=areas_payload)

    at = AppTest.from_file(str(AREA_PAGE))
    at.run()

    assert not at.exception
    frame = at.dataframe[0].value
    assert len(frame) == 2
    assert frame["Change (12m)"].dtype.kind == "f"
    assert frame["Median AED/m² (12m)"].dtype.kind == "f"
    assert frame["Sales (12m)"].dtype.kind in "iu"
    by_area = frame.set_index("Area")["Change (12m)"]
    assert by_area["Dubai Marina"] == pytest.approx(0.032)  # shown via format="percent"
    assert by_area.isna()["Al Barsha South"]


def test_area_history_is_cached(monkeypatch):
    client = patch_client(monkeypatch)

    at = AppTest.from_file(str(AREA_PAGE))
    at.run()
    at.run()

    assert not at.exception
    assert len(client.calls["area_history"]) == 1
