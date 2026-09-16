"""Listing check — look up duplicates and fraud flags."""

from __future__ import annotations

import streamlit as st

from demo.client import ApiProblem, get_client
from demo.ui import DATA_NOTE, SYNTHETIC_NOTE, show_problem

st.set_page_config(page_title="Listing check — Zestimator", layout="wide")
st.title("Listing check")
st.caption(DATA_NOTE)
st.caption(SYNTHETIC_NOTE)


def _render_duplicates(duplicates: list[dict]) -> None:
    st.subheader("Duplicates")
    if duplicates:
        st.table(duplicates)
    else:
        st.write("No duplicates found.")


def _render_flags(flags: list[dict]) -> None:
    st.subheader("Flags")
    if flags:
        for flag in flags:
            with st.container(border=True):
                st.markdown(f"**{flag.get('flag', 'flag')}**")
                if flag.get("detail"):
                    st.write(flag["detail"])
    else:
        st.write("No flags raised.")


existing_tab, new_tab = st.tabs(["Existing listing", "New listing"])

with existing_tab:
    listing_id = st.text_input("Listing ID", key="existing_listing_id")
    if st.button("Look up", key="lookup_existing") and listing_id:
        try:
            result = get_client().listing_flags(listing_id)
        except ApiProblem as problem:
            show_problem(problem)
        else:
            _render_duplicates(result.get("duplicates") or [])
            _render_flags(result.get("flags") or [])

with new_tab:
    st.caption(
        "The flags endpoint doesn't return listing text, so prefilling from an "
        "existing listing only fills the id-related notes below — enter the rest "
        "of the listing yourself."
    )
    prefill_id = st.text_input("Prefill from listing ID (optional)", key="prefill_listing_id")
    if st.button("Load", key="prefill_button") and prefill_id:
        try:
            prefill_result = get_client().listing_flags(prefill_id)
        except ApiProblem as problem:
            show_problem(problem)
        else:
            st.session_state["new_listing_notes"] = (
                f"Prefilled from listing {prefill_id} "
                f"(detect run {prefill_result.get('detect_run_id', 'n/a')})."
            )

    with st.form("new_listing_form"):
        title = st.text_input("Title")
        price = st.number_input("Asking price (AED)", min_value=0.0, value=1_000_000.0)
        area_name = st.text_input("Area")
        size_sqm = st.number_input("Size (sqm)", min_value=0.0, value=100.0)
        notes = st.text_area("Notes", value=st.session_state.get("new_listing_notes", ""))
        submitted = st.form_submit_button("Check listing")

    if submitted:
        body = {
            "title": title,
            "asking_price_aed": price,
            "area_name": area_name,
            "size_sqm": size_sqm,
            "notes": notes,
        }
        try:
            result = get_client().check_listing(body)
        except ApiProblem as problem:
            show_problem(problem)
        else:
            _render_duplicates(result.get("duplicates") or [])
            _render_flags(result.get("flags") or [])
            for note in result.get("notes") or []:
                st.caption(note)
