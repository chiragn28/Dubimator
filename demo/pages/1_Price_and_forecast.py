"""Price & forecast — estimate a home's value and see its trend."""

from __future__ import annotations

import streamlit as st

from demo.client import ApiProblem, get_client
from demo.ui import DATA_NOTE, cached_areas, forecast_rows, money, show_problem

st.set_page_config(page_title="Price & forecast — Dubimator", layout="wide")
st.title("Price & forecast")
st.caption(DATA_NOTE)

PROPERTY_KINDS = ["apartment", "hotel_apartment", "townhouse", "villa"]
STATUSES = ["ready", "off_plan"]
SIZE_BASES = ["built_up", "plot"]
BEDROOM_CHOICES = [None, *range(9)]  # the API accepts 0-8; None leaves it unspecified


def _label(value: str) -> str:
    return value.replace("_", " ")


def _bedrooms_label(value: int | None) -> str:
    if value is None:
        return "not specified"
    return "studio" if value == 0 else str(value)


area_names: dict[str, int] = {}
areas_available = True
try:
    areas_payload = cached_areas()
    area_names = {area["name"]: area["area_id"] for area in areas_payload.get("areas", [])}
except ApiProblem:
    areas_available = False

# Outside the form so choosing "villa" immediately offers the size basis below.
kind = st.selectbox("Property type", PROPERTY_KINDS, format_func=_label, key="price_kind")

with st.form("price_form"):
    area_id: int | None = None
    area_text: str | None = None
    if areas_available and area_names:
        area_choice = st.selectbox("Area", sorted(area_names), key="price_area")
        area_id = area_names[area_choice]
    else:
        area_text = st.text_input("Area (name)", key="price_area_name").strip() or None
        st.caption("Couldn't load the areas list — type the area name instead.")

    status = st.selectbox("Status", STATUSES, format_func=_label, key="price_status")
    size_sqm = st.number_input(
        "Size (sqm)", min_value=20.0, max_value=2000.0, value=120.0, key="price_size"
    )
    if kind == "villa":
        size_basis = st.selectbox(
            "Size measured as",
            SIZE_BASES,
            format_func=lambda basis: "built-up area" if basis == "built_up" else "plot area",
            key="price_size_basis",
        )
    else:
        size_basis = "built_up"
    bedrooms = st.selectbox(
        "Bedrooms", BEDROOM_CHOICES, index=3, format_func=_bedrooms_label, key="price_bedrooms"
    )
    building = st.text_input("Building (optional)", key="price_building").strip()
    project = st.text_input("Project (optional)", key="price_project").strip()

    submitted = st.form_submit_button("Estimate")

if submitted:
    body = {
        "area_id": area_id,
        "area": area_text,
        "property_kind": kind,
        "status": status,
        "size_sqm": float(size_sqm),
        "size_basis": size_basis,
        "bedrooms": bedrooms,
        "building": building or None,
        "project": project or None,
    }
    body = {key: value for key, value in body.items() if value is not None}

    try:
        price = get_client().price(body)
    except ApiProblem as problem:
        show_problem(problem)
    else:
        low, high = price["range_80"]
        st.metric("Estimate", money(price["estimate_aed"]))
        st.write(f"80% range: {money(low)} – {money(high)}")
        st.caption(
            f"Confidence: {price.get('confidence', 'n/a')} · "
            f"{money(price.get('price_per_sqm_aed', 0))}/m²"
        )
        flags = price.get("flags") or []
        for flag in flags:
            st.warning(flag)

        try:
            forecast = get_client().forecast(body)
        except ApiProblem as problem:
            show_problem(problem)
        else:
            st.subheader("Forecast")
            st.table(forecast_rows(forecast))

            st.subheader("Key drivers")
            drivers = forecast.get("key_drivers") or []
            if drivers:
                for driver in drivers:
                    if isinstance(driver, dict):
                        label = driver.get("feature", "factor")
                        impact = driver.get("impact_pct")
                        st.write(f"- {label}" + (f": {impact:+.1%}" if impact is not None else ""))
                    else:
                        st.write(f"- {driver}")
            else:
                st.write("No single factor moved this forecast by 0.5% or more.")

            exclusions = forecast.get("exclusions_applied") or []
            if exclusions:
                st.subheader("Exclusions applied")
                for exclusion in exclusions:
                    st.write(f"- {exclusion}")
