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

        hits = result.get("hits") or []
        st.subheader(f"{len(hits)} result(s)")
        for hit in hits:
            with st.container(border=True):
                st.markdown(f"**{hit.get('title', 'Untitled')}**")
                if "asking_price_aed" in hit:
                    st.write(money(hit["asking_price_aed"]))
                if hit.get("area_name"):
                    st.caption(hit["area_name"])
                reasons = hit.get("reasons") or []
                if reasons:
                    st.write("Why: " + ", ".join(reasons))
                if hit.get("duplicate"):
                    st.warning("Possible duplicate listing")
                fraud_flags = hit.get("fraud_flags")
                if fraud_flags:
                    st.error("Fraud flags: " + ", ".join(fraud_flags))
