"""Area explorer — see how an area's prices have moved."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from demo.client import ApiProblem, get_client
from demo.ui import DATA_NOTE, cached_areas, pct, show_problem

st.set_page_config(page_title="Area explorer — Zestimator", layout="wide")
st.title("Area explorer")
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
        summary_records = [
            {
                "Area": area["name"],
                "Median AED/m² (12m)": area["median_ppsm_12m"],
                "Change (12m)": pct(area["change_12m"]),
                "Sales (12m)": area["sales_12m"],
            }
            for area in areas
        ]
        st.dataframe(pd.DataFrame(summary_records), use_container_width=True)

        area_by_name = {area["name"]: area["area_id"] for area in areas}
        selected_name = st.selectbox("Area", sorted(area_by_name))
        selected_area_id = area_by_name[selected_name]

        try:
            history = get_client().area_history(selected_area_id)
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
