"""Dubimator demo — Streamlit entry point.

Talks to the Phase 7 API over HTTP only (see `demo/client.py`); it never
imports the database, MLflow or the model/search/listings packages
directly.
"""

from __future__ import annotations

import streamlit as st

from demo.client import ApiProblem
from demo.ui import DATA_NOTE, cached_ready, render_component_status, show_problem

st.set_page_config(page_title="Dubimator", layout="wide")

st.title("Dubimator")
st.write(
    "Price a Dubai home and see its forecast, search listings in plain English, "
    "check a listing for duplicates and fraud, and explore how an area's prices "
    "have moved — all backed by the Dubimator API."
)

with st.sidebar:
    st.subheader("API status")
    try:
        ready = cached_ready()
    except ApiProblem as problem:
        show_problem(problem)
    else:
        render_component_status(ready)

st.caption(DATA_NOTE)

st.markdown("### Pages")
st.page_link("pages/1_Price_and_forecast.py", label="Price & forecast")
st.page_link("pages/2_Search.py", label="Search")
st.page_link("pages/3_Listing_check.py", label="Listing check")
st.page_link("pages/4_Area_explorer.py", label="Area explorer")
st.page_link("pages/5_About.py", label="About")
