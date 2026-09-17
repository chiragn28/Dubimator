"""About — data, limits and how the models are gated."""

from __future__ import annotations

import streamlit as st

from demo.client import ApiProblem
from demo.ui import (
    DATA_NOTE,
    SYNTHETIC_NOTE,
    cached_ready,
    render_component_status,
    show_problem,
)

st.set_page_config(page_title="About — Zestimator", layout="wide")
st.title("About")

st.markdown(
    f"""
### Data and its limits
{DATA_NOTE}

Sales come from the Dubai Land Department (DLD). Only clean home sales from
2015 onward feed pricing, forecasting and the area statistics. DLD publishes
no coordinates, so there is no map — the area explorer uses a table and
history charts instead.

{SYNTHETIC_NOTE} Duplicate and fraud detection run over that synthetic
corpus, never over a real seller's data.

### How each model is gated
- **Price** is served whenever the price model is deployed and its inputs
  validate against the area, size and property type it was trained on.
- **Forecast** serves a 3-month horizon once the forecast model is
  deployed. The 1-year and 3-year horizons are gated separately and, until
  they ship, the page shows the API's own reason rather than a number.
- **Search** falls back to retrieval order (with a note saying so) if no
  ranking model is registered.
- **Listing checks** (duplicates and fraud flags) run against the
  synthetic listings corpus once the `listings` component is up.

### Links
- [Architecture](https://claude.ai/artifact/6keTdjksn2nWLnkVmNbMRw)
"""
)

st.subheader("Current deployment status")
try:
    ready = cached_ready()
except ApiProblem as problem:
    show_problem(problem)
else:
    render_component_status(ready)
