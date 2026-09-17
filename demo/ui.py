"""Shared rendering helpers for the Streamlit demo pages.

Every page imports from here instead of duplicating formatting or error
handling, so the "every page shows its data-as-of date, never shows a
number without its range, and shows a synthetic-listings note" rules in
the Phase 8 global constraints stay true in one place.
"""

from __future__ import annotations

import streamlit as st

from demo import client as api_client
from demo.client import ApiProblem

DATA_NOTE = "Data as of 2023-03-17 (Dubai Land Department)."
SYNTHETIC_NOTE = "Listings are synthetic, generated over real DLD sales, and labelled."

_SETUP_CODES = frozenset({"no_key", "unreachable", "unauthorized"})
_SETUP_HINT = (
    "Set DEMO_API_URL and DEMO_API_KEY (in a .env file or your shell) and reload the page."
)


def show_problem(problem: ApiProblem) -> None:
    """Render an `ApiProblem` as a friendly error, with its request id.

    `no_key`, `unreachable` and `unauthorized` are the problems whoever runs
    the demo can fix themselves (a missing or wrong `DEMO_API_URL` /
    `DEMO_API_KEY`), so those get an extra setup hint.
    """
    st.error(problem.message)
    st.caption(f"request id: {problem.request_id or 'n/a'}")
    if problem.code in _SETUP_CODES:
        st.info(_SETUP_HINT)


def money(aed: float) -> str:
    """Format an AED amount, e.g. `money(1455490)` -> `"AED 1,455,490"`."""
    return f"AED {aed:,.0f}"


def forecast_rows(forecast: dict) -> list[dict]:
    """One row per horizon: horizon, point, low, high, confidence, note.

    The note is the horizon's `message` (e.g. a low-confidence caveat) when
    it is deployed, or its `reason` for not being deployed otherwise.
    """
    rows = []
    for horizon, key in (("3m", "forecast_3m"), ("1y", "forecast_1y"), ("3y", "forecast_3y")):
        data = forecast.get(key) or {}
        if data.get("status") == "not_deployed":
            rows.append(
                {
                    "horizon": horizon,
                    "point": None,
                    "low": None,
                    "high": None,
                    "confidence": None,
                    "note": data.get("reason", "not deployed"),
                }
            )
        else:
            rows.append(
                {
                    "horizon": horizon,
                    "point": data.get("point"),
                    "low": data.get("ci_low"),
                    "high": data.get("ci_high"),
                    "confidence": data.get("confidence"),
                    "note": data.get("message", ""),
                }
            )
    return rows


def render_component_status(ready: dict) -> None:
    """Render each API component's up/down state and version.

    `/v1/ready` nests them under `"components"`.
    """
    components = ready.get("components")
    if not isinstance(components, dict):
        return
    for name, info in components.items():
        if not isinstance(info, dict):
            continue
        up = info.get("up")
        version = info.get("version")
        icon = "🟢" if up else "🔴"
        label = f"{icon} **{name}**"
        if version:
            label += f" ({version})"
        st.markdown(label)
        if not up and info.get("error"):
            st.caption(info["error"])


@st.cache_data(ttl=600)
def cached_areas() -> dict:
    """`GET /v1/areas`, cached for 10 minutes (the spec's caching decision)."""
    return api_client.get_client().areas()


@st.cache_data(ttl=600)
def cached_area_history(area_id: int) -> dict:
    """`GET /v1/areas/{id}/history`, cached for 10 minutes."""
    return api_client.get_client().area_history(area_id)


@st.cache_data(ttl=600)
def cached_ready() -> dict:
    """`GET /v1/ready`, cached for 10 minutes."""
    return api_client.get_client().ready()
