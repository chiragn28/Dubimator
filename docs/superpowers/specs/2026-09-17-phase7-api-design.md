# Phase 7: Unified API

- **Status:** design decisions made by the user on 2026-09-17: the endpoints, auth, model loading and observability, plus "whichever you recommend" for the listing check. The user was asleep when this spec was written and had delegated the remaining decisions ("take all decisions yourself"). The controller made them with its recommended options, and each is recorded under Decisions.
- **Date:** 2026-09-17
- **Part of:** Dubai Real Estate ML Platform (sub-project 7 of 11)
- **Builds on:** Phase 3 (price), Phase 4 (duplicates and fraud), Phase 5 (search) and Phase 6 (forecast).

## Goal

A single FastAPI service puts every model behind one versioned HTTP API: the price estimate, the multi-horizon forecast, ranked search, and listing checks.

It must be safe to expose through a Cloudflare Tunnel (Phase 10). That means API keys, per-key rate limits and no stack traces in responses. It must also be observable, with JSON logs, request IDs and Prometheus metrics.

Out of scope: Docker packaging and the tunnel (Phase 10), the UI (Phase 8), and CI (Phase 9).

## Decisions

The first five are the user's; the rest are the controller's, marked (C).

| Topic | Decision |
|---|---|
| Endpoints | Price, forecast, search and listing check, all under `/v1` |
| Auth | An `X-API-Key` header (also accepted as `Authorization: Bearer <key>`). Keys come from `.env` `API_KEYS`, comma-separated, and are compared with `hmac.compare_digest`. |
| Rate limit | Per key, a token bucket held in process: `API_RATE_LIMIT_PER_MINUTE` (default 60), burst = limit. An exhausted bucket gets 429 with `Retry-After`. |
| Model loading | Everything loads at startup (FastAPI lifespan). `POST /v1/admin/reload` rebuilds the state and swaps it in atomically. |
| Observability | JSON logs to stdout; an `X-Request-ID` header (echoed back, or generated); Prometheus `/metrics` |
| Listing check | Both: `GET /v1/listings/{id}/flags` (stored flags) and `POST /v1/listings/check` (a new listing scored on the fly) |
| (C) Pair model | Phase 4 refits its logistic pair model on every `detect` and never saves it. `python -m listings detect` now also logs it to MLflow as the pyfunc `zestimator-duplicate-pair@champion` (no `code_paths`), holding the pipeline, threshold, feature names and detect run id. A one-off `python -m listings detect` re-run registers the first version. |
| (C) Photos in `POST /check` | A new listing may reference existing corpus `photo_ids`. Photo similarity and shared-photo features use them. Uploaded images are not embedded per request, because that would add latency and download risk. With no photos, image features are 0 and the response says so in `notes`. |
| (C) `/metrics` access | Needs the same API key, because it would otherwise be public through the tunnel. Prometheus sends `Authorization: Bearer`. `/health` is public. |
| (C) Server | uvicorn, 1 worker, sync endpoints on FastAPI's threadpool. The SearchEngine is not thread-safe, so a `threading.Lock` serializes access to it. The price predictor, Forecaster and pair model are read-only and safe to share. |
| (C) Partial availability | A component that fails to load (for example, no forecast champion) does not stop the service. Its endpoint returns 503 with the reason, and `/ready` lists each component's state. |
| (C) Versioning | Routes live under `/v1`. The OpenAPI docs stay at `/docs` and `/openapi.json`, which are public and leak nothing sensitive. |

## Architecture

The new package is `api/`. The only change outside it is `listings/pairmodel.py` plus an addition to `listings detect`.

| Module | Job |
|---|---|
| `api/settings.py` | `ApiSettings.from_env()`: keys, rate limit, the component toggles and model URIs. It refuses to start with no API key unless `API_ALLOW_NO_KEYS=1`, which is for tests. |
| `api/logging.py` | JSON log formatter; request-ID context variable |
| `api/security.py` | The `require_key` dependency and the `RateLimiter` token bucket (the clock is injectable) |
| `api/metrics.py` | Prometheus registry: `api_requests_total{route,method,status}`, `api_request_seconds{route}` histogram, `api_model_info{component,version}` gauge, `api_component_up{component}` gauge |
| `api/state.py` | `AppState` holds the components and their load errors. `load_state(settings)` builds it, and each component has its own loader, so tests inject fakes. |
| `api/schemas.py` | Pydantic request and response models |
| `api/routes/*.py` | `health`, `price`, `forecast`, `search`, `listings`, `admin` |
| `api/app.py` | `create_app(settings, state_loader)`: lifespan, middleware (request ID, logging, metrics), exception handlers, routers |
| `api/__main__.py` | `python -m api serve [--host --port]` and `python -m api smoke` (an in-process real run of every endpoint, with a latency report) |
| `listings/pairmodel.py` | `PairModel` (pipeline, threshold, features, detect_run_id): save, load, pyfunc, `register_pair_model`, `load_pair_model` |
| `listings/check.py` | `ListingChecker`: scores a new listing against the corpus |

## Endpoints

All `/v1` routes need a key. Every response carries `X-Request-ID`. Errors use `{"error": {"code", "message", "field"?}, "request_id"}`.

| Method and path | Body or query | Returns |
|---|---|---|
| `GET /health` | | `{"status":"ok"}`. Public; liveness only. |
| `GET /v1/ready` | | Component states, with version or error each; 200 even when a component is down |
| `POST /v1/price` | The `PriceRequest` fields | `PriceEstimate` JSON (Phase 3) |
| `POST /v1/forecast` | The `PriceRequest` fields plus optional `property_id` | The Phase 6 forecast JSON |
| `GET /v1/search` | `q` (1–300 chars), `k` (1–50, default 10) | `SearchResult.to_dict()` |
| `GET /v1/listings/{listing_id}/flags` | | The latest detect run's stored flags: duplicate pairs with score and signals, and fraud flags with detail. 404 if the listing is unknown. |
| `POST /v1/listings/check` | `ListingCheckRequest` | `{duplicates:[{listing_id, score, signals}], flags:[{flag, detail}], notes:[...], model_versions}` |
| `POST /v1/admin/reload` | | Reloads all components; returns the new component states |
| `GET /metrics` | | Prometheus text format (key required) |

**Status codes:**

| Code | Meaning |
|---|---|
| 401 | Missing or wrong key |
| 422 | Invalid input: a Pydantic validation error or a `PriceInputError`, with its `field` |
| 404 | Unknown listing |
| 429 | Rate limit exceeded |
| 503 | Component unavailable, with its load error |
| 500 | Anything else, as a generic message; the traceback goes to the logs only |

**`ListingCheckRequest`:** `title`, `description`, `asking_price_aed > 0`, `area_id`, `building_name?`, `project_name?`, `property_kind`, `status`, `size_sqm > 0`, `size_basis`, `bedrooms?`, `agent_id?`, `posted_at?` (default: the corpus's latest date), `photo_ids?` (at most 10, existing ids only; unknown ids give 422).

## Listing check (`listings/check.py`)

`ListingChecker.from_connection(conn, embedder, pair_model, price_predictor, config)` loads once: listing attributes, listing vectors, photo sets and photo vectors. It holds its own connection for pgvector queries, serialized by a lock.

`check(request)` runs these steps:

1. **Embed the text.** Embed `title + "\n" + description` with the same text model Phase 4 used (`embed_texts`).
2. **Find candidates.**
   - Take the top `text_top_k` nearest listings by text vector (pgvector query with `hnsw.ef_search`).
   - Add the top `photo_top_k` by mean image vector when photos are given.
   - Add every listing that shares a given photo, capped by `max_photo_fanout`.
3. **Build the pair features.**
   - Use the Phase 4 `build_features`, with the new listing as a temporary attribute row whose `listing_id` is −1, joined to the candidates.
   - Its vectors are the new text vector, the mean of the given photos' vectors (normalized) and the photo list.
4. **Score and decide.** Score with the registered pair model and compare to its threshold. The duplicates returned are those with `decision` true, sorted by score, with their 12 signals.
5. **Fraud flags:**
   - **`bait_price`:** as in Phase 4 (the asking price is more than `bait_margin` below the price model's 80% low), when the listing can be priced. If it can't, a note gives the `PriceInputError` message.
   - **`photo_reuse`:** as in Phase 4. A given photo is used by listings in at least `photo_reuse_min_areas` distinct areas.
   - **`inconsistent_relist`:** a flagged duplicate whose asking price differs by more than `relist_price_spread`. This is the Phase 4 rule applied to the new listing's duplicate pairs, using the price-blind decision (the same model and threshold with `abs_log_price_ratio` set to 0).
6. **Report versions.** `model_versions` lists the pair model and the price model.

The checker reads only detection columns and never touches `DETECTION_FORBIDDEN_COLUMNS`; a test enforces this with the existing forbidden-column scan pattern.

## State and reload

`AppState` holds these components:
- `price`: the `PricePredictor`, via `listings.fraud.load_price_predictor`;
- `forecaster`: `Forecaster.from_registry`;
- `search`: a `SearchEngine` on its own connection, with its lock;
- `listings`: the `ListingChecker` on its own connection, plus a reader for stored flags.

Each component is loaded inside its own try block. On failure the component is `None`, and `errors[name]` holds `"<Type>: <message>"`.

`reload` builds a new state completely, then swaps the reference, then closes the old connections. Requests in flight keep the old state object.

## Security notes

- Keys are never logged. The logged key identifier is the first 8 characters of the key's SHA-256.
- Request bodies are not logged; paths, statuses and timings are.
- A search `q` is logged truncated to 100 characters.
- CORS is off by default. `API_CORS_ORIGINS` enables it for the Phase 8 demo.
- No stack traces appear in responses.

## Testing

- **Unit tests** use `fastapi.testclient.TestClient`, with fake components in place of the real ones. They cover:
  - auth: none, wrong, header, bearer;
  - the rate limit, with an injected clock;
  - request IDs: echoed or generated;
  - the error envelope and status codes;
  - each route's happy path and error path;
  - 503 on a missing component, `/ready`, reload's atomic swap, and `/metrics` content;
  - a log line with no key or body.
- **`ListingChecker` on synthetic data** (`FakeEmbedder`, a fitted tiny pair model, `pg_test_db`):
  - an exact repost of a corpus listing is flagged as its duplicate;
  - a fresh listing gets no duplicates;
  - bait price;
  - photo reuse;
  - an unknown photo id gives 422;
  - no forbidden columns are read.
- **`pairmodel`:** a save/load round trip, and registration with `temp_mlflow`.
- **The real run:** `python -m listings detect` (registers the pair model), then `python -m api smoke` against the real stack. It prints per-endpoint status and latency and must show 200 for every endpoint.
- Every test runs under `-W error` and needs neither a GPU nor the network.

## Deliverables

- **Code:** `api/`, `listings/pairmodel.py`, `listings/check.py`, and the `detect` registration change.
- **Dependencies:** `fastapi`, `uvicorn`, `prometheus-client`.
- **Settings:** `.env.example` gains `API_KEYS`, `API_RATE_LIMIT_PER_MINUTE` and `API_CORS_ORIGINS`.
- **README:** a "## API" section with endpoints, auth, examples (curl), the smoke-run latency table, and the engine and thread-safety notes.
- **Architecture page:** republished.

## Amendments during implementation

(none yet)
