"""Listing check — look up duplicates and fraud flags."""

from __future__ import annotations

import streamlit as st

from demo.client import ApiProblem, get_client
from demo.ui import (
    DATA_NOTE,
    EXAMPLE_LISTING_IDS,
    NO_PHOTO_BOX,
    PHOTO_DISCLAIMER,
    SYNTHETIC_NOTE,
    cached_areas,
    chips,
    cover_photo,
    duplicate_chips,
    flag_summary,
    photo_strip,
    show_problem,
)

st.set_page_config(page_title="Listing check — Dubimator", layout="wide")
st.title("Listing check")
st.caption(DATA_NOTE)
st.caption(SYNTHETIC_NOTE)


PROPERTY_KINDS = ["apartment", "hotel_apartment", "townhouse", "villa"]
STATUSES = ["ready", "off_plan"]
SIZE_BASES = ["built_up", "plot"]
BEDROOM_CHOICES = [None, *range(9)]  # the API accepts 0-8; None leaves it unspecified
MAX_PHOTOS = 10


def _label(value: str) -> str:
    return value.replace("_", " ")


def _bedrooms_label(value: int | None) -> str:
    if value is None:
        return "not specified"
    return "studio" if value == 0 else str(value)


def parse_photo_ids(text: str) -> list[int]:
    """`"11, 12,13"` -> `[11, 12, 13]`. Raises ValueError on a non-integer or too many ids."""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    try:
        ids = [int(part) for part in parts]
    except ValueError:
        raise ValueError("Photo IDs must be whole numbers separated by commas.") from None
    if len(ids) > MAX_PHOTOS:
        raise ValueError(f"At most {MAX_PHOTOS} photo IDs can be checked at once.")
    return ids


def _render_duplicates(duplicates: list[dict]) -> None:
    st.subheader("Possible reposts")
    if not duplicates:
        st.success("No duplicates found.")
        return
    count = len(duplicates)
    st.caption(
        f"{count} other listing looks like the same home."
        if count == 1
        else f"{count} other listings look like the same home."
    )
    for duplicate in duplicates:
        other_id = duplicate.get("listing_id")
        with st.container(border=True):
            image_col, text_col = st.columns([1, 2.6], vertical_alignment="top")
            with image_col:
                photo = cover_photo(int(other_id)) if other_id is not None else None
                if photo:
                    st.image(photo, use_container_width=True)
                else:
                    st.markdown(NO_PHOTO_BOX, unsafe_allow_html=True)
            with text_col:
                score = duplicate.get("score")
                if score is not None:
                    st.markdown(f"### {min(float(score), 1.0) * 100:.1f}% match")
                st.markdown(f"**Listing #{other_id}**")
                chips(duplicate_chips(duplicate.get("signals") or {}))


def _render_flags(flags: list[dict]) -> None:
    st.subheader("Warnings")
    if not flags:
        st.success("No flags raised.")
        return
    for flag in flags:
        name = flag.get("flag", "flag")
        heading, sentence, tags = flag_summary(name, flag.get("detail") or {})
        with st.container(border=True):
            st.markdown(f"**⚠️ {heading}**")
            if sentence:
                st.write(sentence)
            chips(tags)
            if flag.get("detail"):
                with st.expander("Why the detector said so"):
                    st.json(flag["detail"])


def _use_example(listing_id: str) -> None:
    """Fill the ID box and look it up, from the example buttons."""
    st.session_state.existing_listing_id = listing_id
    st.session_state.run_existing_lookup = True


existing_tab, new_tab = st.tabs(["Existing listing", "New listing"])

with existing_tab:
    listing_id = st.text_input(
        "Listing ID", key="existing_listing_id", placeholder="a number from 1 to 20000"
    )
    st.caption("Or try an example:")
    # One trailing spacer keeps the buttons compact on the wide layout without
    # squeezing the labels to an ellipsis.
    columns = st.columns(len(EXAMPLE_LISTING_IDS) + 1)
    for column, (example_id, outcome) in zip(columns, EXAMPLE_LISTING_IDS):
        column.button(
            f"{example_id} · {outcome}",
            key=f"example_{example_id}",
            on_click=_use_example,
            args=(example_id,),
            use_container_width=True,
        )
    looked_up = st.button("Look up", key="lookup_existing")
    if (looked_up or st.session_state.pop("run_existing_lookup", False)) and listing_id:
        try:
            result = get_client().listing_flags(listing_id)
        except ApiProblem as problem:
            show_problem(problem)
        else:
            st.subheader(f"Listing #{listing_id}")
            if photo_strip(int(listing_id)):
                st.caption(PHOTO_DISCLAIMER)
            else:
                st.caption("No photos for this listing on this server.")
            _render_duplicates(result.get("duplicates") or [])
            _render_flags(result.get("flags") or [])

with new_tab:
    st.caption(
        "Describe a listing that isn't in the corpus yet. The API doesn't return the "
        "listing text for existing listings, so there is no prefill — enter it here."
    )
    try:
        area_names = {area["name"]: area["area_id"] for area in cached_areas().get("areas") or []}
    except ApiProblem as problem:
        st.write("Checking a new listing needs the areas list:")
        show_problem(problem)
        area_names = {}

    if area_names:
        # Outside the form so choosing "villa" immediately offers the size basis below.
        kind = st.selectbox("Property type", PROPERTY_KINDS, format_func=_label, key="new_kind")

        with st.form("new_listing_form"):
            title = st.text_input("Title", max_chars=300, key="new_title").strip()
            description = st.text_area("Description", max_chars=5000, key="new_description")
            price = st.number_input(
                "Asking price (AED)", min_value=1.0, value=1_000_000.0, key="new_price"
            )
            area_choice = st.selectbox("Area", sorted(area_names), key="new_area")
            building = st.text_input(
                "Building (optional)", max_chars=200, key="new_building"
            ).strip()
            project = st.text_input("Project (optional)", max_chars=200, key="new_project").strip()
            status = st.selectbox("Status", STATUSES, format_func=_label, key="new_status")
            size_sqm = st.number_input(
                "Size (sqm)", min_value=1.0, max_value=20000.0, value=100.0, key="new_size"
            )
            if kind == "villa":
                size_basis = st.selectbox(
                    "Size measured as",
                    SIZE_BASES,
                    format_func=lambda basis: "built-up area" if basis == "built_up" else "plot",
                    key="new_size_basis",
                )
            else:
                size_basis = "built_up"
            bedrooms = st.selectbox(
                "Bedrooms",
                BEDROOM_CHOICES,
                index=3,
                format_func=_bedrooms_label,
                key="new_bedrooms",
            )
            photo_text = st.text_input(
                f"Photo IDs (optional, comma-separated, up to {MAX_PHOTOS})", key="new_photo_ids"
            )
            submitted = st.form_submit_button("Check listing", key="check_new_listing")

        if submitted:
            problems = []
            if not title:
                problems.append("Enter a title.")
            if not description.strip():
                problems.append("Enter a description.")
            try:
                photo_ids = parse_photo_ids(photo_text)
            except ValueError as exc:
                problems.append(str(exc))
                photo_ids = []
            for message in problems:
                st.error(message)

            if not problems:
                body = {
                    "title": title,
                    "description": description.strip(),
                    "asking_price_aed": float(price),
                    "area_id": area_names[area_choice],
                    "building_name": building or None,
                    "project_name": project or None,
                    "property_kind": kind,
                    "status": status,
                    "size_sqm": float(size_sqm),
                    "size_basis": size_basis,
                    "bedrooms": bedrooms,
                    "photo_ids": photo_ids,
                }
                body = {key: value for key, value in body.items() if value is not None}
                try:
                    result = get_client().check_listing(body)
                except ApiProblem as problem:
                    show_problem(problem)
                else:
                    _render_duplicates(result.get("duplicates") or [])
                    _render_flags(result.get("flags") or [])
                    for note in result.get("notes") or []:
                        st.caption(note)
