"""Search — find listings in plain English."""

from __future__ import annotations

import streamlit as st

from demo.client import ApiProblem, get_client
from demo.ui import (
    DATA_NOTE,
    SYNTHETIC_NOTE,
    chips,
    cover_photo,
    listing_card,
    parsed_chips,
    show_problem,
)

st.set_page_config(page_title="Search — Dubimator", layout="wide")
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
        understood = parsed_chips(parsed)
        if understood:
            st.caption("Searching for")
            chips(understood)
        for error in parsed.get("errors") or []:
            st.warning(error)

        for note in result.get("notes") or []:
            st.info(note)

        hits = result.get("results") or []
        st.subheader(f"{len(hits)} result{'' if len(hits) == 1 else 's'}")
        for hit in hits:
            listing_id = hit.get("listing_id")
            photo = cover_photo(int(listing_id)) if listing_id is not None else None
            listing_card(hit, photo=photo)

        with st.expander("How these results were found"):
            if result.get("ranker"):
                st.write(f"Ranked by `{result['ranker']}`.")
            timings = result.get("timings_ms") or {}
            if timings:
                st.write(
                    "  ·  ".join(
                        f"{name} {value:.0f} ms"
                        for name, value in timings.items()
                        if isinstance(value, (int, float))
                    )
                )
            st.caption("The parsed query, as the API returned it:")
            st.json(parsed)
