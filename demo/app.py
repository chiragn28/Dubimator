"""Dubimator demo — Streamlit entry point.

Talks to the Phase 7 API over HTTP only (see `demo/client.py`); it never
imports the database, MLflow or the model/search/listings packages
directly.
"""

from __future__ import annotations

import streamlit as st

from demo import theme
from demo.client import ApiProblem
from demo.ui import DATA_NOTE, cached_ready, render_component_status, show_problem

st.set_page_config(page_title="Dubimator", layout="wide")
theme.apply()

# The four pages, in the order the product tells its story: value a home, then find one, then
# check it is real, then look at the market behind all three. Each says which phase built it,
# because the point of the demo is to show the system, not just the answers.
ENTRIES = [
    (
        "pages/1_Price_and_forecast.py",
        "Price & forecast",
        "Phases 3 & 6",
        (
            "An estimate with its 80% range, then 3-month, 1-year and 3-year growth. "
            "Horizons that failed their gate return the reason, not a number."
        ),
    ),
    (
        "pages/2_Search.py",
        "Search",
        "Phase 5",
        (
            "“2BR in Dubai Marina under 1.5M”, parsed into slots, retrieved two ways "
            "and re-ranked, with a reason on every result."
        ),
    ),
    (
        "pages/3_Listing_check.py",
        "Listing check",
        "Phase 4",
        (
            "Reposted listings found from text and image similarity, plus bait prices "
            "and inconsistent relists."
        ),
    ),
    (
        "pages/4_Area_explorer.py",
        "Area explorer",
        "Phase 2",
        (
            "Twelve-month medians and monthly history per area and property kind, "
            "straight from the DLD record."
        ),
    ),
]

theme.page_header(
    "Dubai real estate ML platform",
    "Dubimator",
    "Four models over 1,047,965 Dubai Land Department sales. Every one had to beat a plain "
    "baseline before it was allowed to serve — and the one that didn’t is still here, saying so.",
)

columns = st.columns(2, gap="medium")
for index, (page, title, phase, blurb) in enumerate(ENTRIES):
    with columns[index % 2], st.container(border=True):
        st.markdown(f'<div class="dbm-eyebrow">{phase}</div>', unsafe_allow_html=True)
        st.markdown(f"### {title}")
        st.write(blurb)
        st.page_link(page, label=f"Open {title}  →")

st.markdown("## The honest part")
st.write(
    "The transactions are real. The listings are not: DLD publishes completed sales, not "
    "listings, so the corpus behind Search and Listing check is generated over those real sales. "
    "The 1-year forecast missed its accuracy gate and is not served — the About page has the "
    "full list of what this demo cannot tell you."
)
st.page_link("pages/5_About.py", label="Read the limitations  →")

st.caption(DATA_NOTE)

with st.sidebar:
    st.markdown('<div class="dbm-eyebrow">Live status</div>', unsafe_allow_html=True)
    st.markdown("### API")
    try:
        ready = cached_ready()
    except ApiProblem as problem:
        show_problem(problem)
    else:
        render_component_status(ready)
