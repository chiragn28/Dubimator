# Phase 8: Streamlit Demo

- **Status:** designed on 2026-09-17 while the user was asleep. The user had delegated every decision ("take all decisions yourself (your own recommended decisions)"), so each one below is the controller's recommended option and is marked (C).
- **Date:** 2026-09-17
- **Part of:** Dubai Real Estate ML Platform (sub-project 8 of 11)
- **Builds on:** Phase 7 (API), docs/superpowers/specs/2026-09-17-phase7-api-design.md

## Goal

A Streamlit app that a recruiter or a dubizzle/Bayut engineer can open through the Phase 10 tunnel and use in two minutes to:
- price a home and see its forecast;
- search listings in plain English;
- check a listing for duplicates and fraud;
- explore how an area's prices moved.

The app only talks to the Phase 7 API over HTTP. It never reads the database or MLflow directly, so it deploys as its own container.

## Decisions (all (C))

| Topic | Decision |
|---|---|
| Transport | An `httpx` client (`demo/client.py`) calls the API with the key from `DEMO_API_KEY` and the URL from `DEMO_API_URL` (default `http://127.0.0.1:8000`). The key stays on the server side of Streamlit and is never sent to the browser. |
| Area explorer data | DLD has no coordinates, and the user declined geocoding, so there is **no map**. Instead the API gains two read-only endpoints: `GET /v1/areas` (id, name, 12-month median price/m², 12-month change, sales count) and `GET /v1/areas/{area_id}/history` (the monthly median price/m² and count per `market_kind`). A new API component, `areas`, precomputes both from `dld.transactions` at startup. The demo draws a sortable table and a line chart. |
| Pages | A Streamlit multipage app: **Price & forecast**, **Search**, **Listing check**, **Area explorer** and **About**. About covers the data limits, the synthetic listings and the model cards, with links to the README and the architecture page. |
| Charts | Streamlit's built-in `st.line_chart` and `st.bar_chart`, so there is no plotting dependency. |
| Honesty in the UI | Every number carries its range or a "not deployed" reason. The data-as-of date (2023-03-17) shows on every page. Listing pages carry a "synthetic listings" badge. Forecast horizons that are not deployed show the API's reason text. |
| Errors | API errors are rendered as friendly messages that carry the API's `message` and `request_id`. A 401 or connection error shows a setup hint. |
| Caching | `st.cache_data`, with a TTL of 10 minutes, for `/v1/areas`, area history and `/v1/ready`. Other calls are not cached. |
| Running it | `uv run python -m demo` starts `streamlit run demo/app.py` on port 8501. CORS isn't needed because calls happen server-side. |
| Tests | Streamlit's `streamlit.testing.v1.AppTest` runs each page against a fake client. Client tests use `httpx.MockTransport`. The API additions get TestClient tests and a DB test for the `areas` component. |

## Architecture

| File | Job |
|---|---|
| `api/areas.py` | `AreaStats.from_connection(conn)` loads the monthly medians per area × market_kind, using the Phase 6 row rules: clean home sales, `market_kind` as defined there. It exposes `.summary()` and `.history(area_id)`. |
| `api/routes/areas.py` | `GET /v1/areas` and `GET /v1/areas/{area_id}/history`; an unknown area returns 404 |
| `api/state.py` | Adds the `areas` component and its loader, plus `"areas"` in `COMPONENTS` |
| `demo/client.py` | `ApiClient(base_url, key, transport=None)` with `ready()`, `price(body)`, `forecast(body)`, `search(q, k)`, `listing_flags(id)`, `check_listing(body)`, `areas()` and `area_history(id)`. It raises `ApiProblem(status, message, field, request_id)`. |
| `demo/app.py` | Entry point. It sets the page config, builds the client from the environment and shows a sidebar with API status. |
| `demo/pages/1_Price_and_forecast.py` | A form (area select from `/v1/areas`, kind, status, size, bedrooms, building, project) → estimate card, 80% range bar, a forecast table and drivers |
| `demo/pages/2_Search.py` | A query box with examples → "understood" chips, results with reasons and notes, and duplicate and fraud badges |
| `demo/pages/3_Listing_check.py` | A form, or "load listing #id" to prefill → duplicates table with signals, flags, notes |
| `demo/pages/4_Area_explorer.py` | A sortable areas table plus a history chart for the selected area and kind |
| `demo/pages/5_About.py` | Static text: data, limits, models and their gates |
| `demo/__main__.py` | Launcher |

`demo/__init__.py`'s stale docstring ("Phase 7") becomes "Phase 8".

## Area statistics

| Setting | Value |
|---|---|
| Sales included | Clean home sales from 2015 onward (the same filters as Phase 6 `load_rows`, without outlier screening: a median is robust) |
| Month grain | Calendar month |
| Summary | Median price/m² over the last 12 months before the data end, the change against the 12 months before that (as a ratio − 1), and the sales count |
| Omitted areas | Areas with fewer than 20 sales in the last 12 months are left out of the summary |
| History | The last 8 years of monthly medians and counts per `market_kind`; months with fewer than 5 sales are null |

## Testing

- **Client:** each method's URL, method and body; the error mapping to `ApiProblem`; the key header; timeouts.
- **Pages (AppTest):**
  - each page renders with the fake client;
  - the price form submits and shows the estimate and range;
  - a not-deployed horizon shows its reason;
  - a search shows results;
  - an API error shows the message with its request id;
  - a missing key shows the setup hint.
- **API additions:** the routes (200, and 404 for an unknown area) and `AreaStats` against the Phase 2 fixture in `pg_test_db`.
- **Real run:** start the API and `python -m demo`, then fetch `http://127.0.0.1:8501` and the Streamlit health endpoint (`/_stcore/health`). Also run AppTest with the real client against the live API, one smoke call per page.

## Deliverables

- **Code:** the files above, plus the `streamlit` dependency.
- **README:** a "## Demo" section with run instructions and page descriptions, plus any screenshots captured headlessly. If screenshot tooling is unavailable, text is enough.
- **Architecture page:** republished.

## Amendments during implementation

(none yet)
