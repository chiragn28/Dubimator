"""Tests for the demo's Streamlit pages, driven with `AppTest` against a fake client."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import demo.client
from demo.client import ApiProblem
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


def patch_client(monkeypatch, fail=None):
    """Patch `demo.client.get_client` so pages get a `FakeClient` with the given failures."""
    client = FakeClient(fail=fail)
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
