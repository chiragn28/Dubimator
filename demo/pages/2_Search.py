"""Search — find listings in plain English."""

from __future__ import annotations

import streamlit as st

from demo.client import ApiProblem, get_client
from demo.ui import DATA_NOTE, SYNTHETIC_NOTE, money, show_problem

st.set_page_config(page_title="Search — Zestimator", layout="wide")
st.title("Search")
st.caption(DATA_NOTE)
st.caption(SYNTHETIC_NOTE)

EXAMPLES = [
    "3br villa in dubai marina under 3m",
    "studio apartment in JVC",
    "townhouse with a pool near the metro",
]

if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""

st.write("Try an example:")
example_cols = st.columns(len(EXAMPLES))
for col, example in zip(example_cols, EXAMPLES):
    if col.button(example, key=f"example_{EXAMPLES.index(example)}"):
        st.session_state["search_query"] = example

query = st.text_input("Search", key="search_query")
k = st.slider("Number of results", min_value=5, max_value=20, value=10)

if query:
    try:
        result = get_client().search(query, k=k)
    except ApiProblem as problem:
        show_problem(problem)
    else:
        parsed = result.get("parsed") or {}
        if parsed:
            st.subheader("Understood")
            st.json(parsed)

        for note in result.get("notes") or []:
            st.info(note)

        hits = result.get("results") or []
        st.subheader(f"{len(hits)} result(s)")
        for hit in hits:
            with st.container(border=True):
                st.markdown(
                    f"**{hit.get('title', 'Untitled')}**  ·  listing #{hit.get('listing_id')}"
                )
                details = [money(hit["asking_price_aed"])] if "asking_price_aed" in hit else []
                if hit.get("size_sqm"):
                    details.append(f"{hit['size_sqm']:,.0f} m²")
                if hit.get("bedrooms") is not None:
                    details.append("studio" if hit["bedrooms"] == 0 else f"{hit['bedrooms']} bed")
                if hit.get("area_name"):
                    details.append(hit["area_name"])
                st.write("  ·  ".join(details))
                reasons = hit.get("reasons") or []
                if reasons:
                    st.write("Why: " + ", ".join(reasons))
                hidden = hit.get("duplicates_hidden") or 0
                if hidden:
                    st.warning(f"{hidden} likely duplicate listing(s) of this one were hidden")
        if result.get("ranker"):
            st.caption(f"Ranked by {result['ranker']}")
