"""Price & forecast — estimate a home's value and see its trend."""

from __future__ import annotations

import streamlit as st

from demo.client import ApiProblem, get_client
from demo.ui import DATA_NOTE, cached_areas, forecast_rows, money, show_problem

st.set_page_config(page_title="Price & forecast — Zestimator", layout="wide")
st.title("Price & forecast")
st.caption(DATA_NOTE)

PROPERTY_KINDS = ["apartment", "villa", "townhouse"]
STATUSES = ["ready", "offplan"]

area_names: dict[str, int] = {}
areas_available = True
try:
    areas_payload = cached_areas()
    area_names = {area["name"]: area["area_id"] for area in areas_payload.get("areas", [])}
except ApiProblem:
    areas_available = False

with st.form("price_form"):
    if areas_available and area_names:
        area_choice = st.selectbox("Area", sorted(area_names))
        area_id = area_names[area_choice]
    else:
        area_choice = st.text_input("Area (name)")
        area_id = None
        st.caption("Couldn't load the areas list — type the area name instead.")

    kind = st.selectbox("Property type", PROPERTY_KINDS)
    status = st.selectbox("Status", STATUSES)
    size_sqm = st.number_input("Size (sqm)", min_value=20.0, max_value=2000.0, value=120.0)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
    building = st.text_input("Building (optional)")
    project = st.text_input("Project (optional)")

    submitted = st.form_submit_button("Estimate")

if submitted:
    body = {
        "area_id": area_id,
        "area_name": area_choice,
        "kind": kind,
        "status": status,
        "size_sqm": size_sqm,
        "bedrooms": int(bedrooms),
        "building": building or None,
        "project": project or None,
    }

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
