"""Area explorer — see how an area's prices have moved."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from demo import theme
from demo.client import ApiProblem
from demo.ui import DATA_NOTE, cached_area_history, cached_areas, show_problem

st.set_page_config(page_title="Area explorer — Dubimator", layout="wide")
theme.apply()
theme.page_header(
    "Phase 2",
    "Area explorer",
    "Twelve-month medians and how they moved, per area and property kind, straight from the "
    "Dubai Land Department record — no model in the way.",
)
st.caption(DATA_NOTE)

try:
    areas_payload = cached_areas()
except ApiProblem as problem:
    show_problem(problem)
else:
    areas = areas_payload.get("areas") or []
    if not areas:
        st.info("No areas met the minimum sales threshold.")
    else:
        # Keep the columns numeric so they sort properly; formatting happens in column_config.
        # change_12m can be None (too few sales a year earlier) and shows as an empty cell.
        summary = pd.DataFrame(
            {
                "Area": [area["name"] for area in areas],
                "Median AED/m² (12m)": pd.Series(
                    [area["median_ppsm_12m"] for area in areas], dtype="float64"
                ),
                "Change (12m)": pd.Series(
                    [area.get("change_12m") for area in areas], dtype="float64"
                ),
                "Sales (12m)": pd.Series([area["sales_12m"] for area in areas], dtype="int64"),
            }
        )
        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
            column_config={
                "Median AED/m² (12m)": st.column_config.NumberColumn(format="%,.0f"),
                "Change (12m)": st.column_config.NumberColumn(format="percent"),
                "Sales (12m)": st.column_config.NumberColumn(format="%,d"),
            },
        )

        area_by_name = {area["name"]: area["area_id"] for area in areas}
        selected_name = st.selectbox("Area", sorted(area_by_name))
        selected_area_id = area_by_name[selected_name]

        try:
            history = cached_area_history(selected_area_id)
        except ApiProblem as problem:
            show_problem(problem)
        else:
            series = history.get("series") or []
            kinds = [item["market_kind"] for item in series]
            if kinds:
                selected_kind = st.selectbox("Market kind", kinds)
                selected_series = next(
                    item for item in series if item["market_kind"] == selected_kind
                )
                points = selected_series.get("points") or []
                if points:
                    chart_df = pd.DataFrame(points).set_index("month")
                    st.subheader(f"{selected_name} — {selected_kind}")
                    st.line_chart(chart_df[["median_ppsm"]])
                    st.bar_chart(chart_df[["sales"]])
                else:
                    st.info("No history available for this area and market kind.")
            else:
                st.info("No market kinds available for this area.")
