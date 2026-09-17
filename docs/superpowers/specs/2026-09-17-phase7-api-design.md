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
| `POST /v1/admin/reload` | | Reloads all components; returns the new component states. Needs a key listed in `API_ADMIN_KEYS`. |
| `GET /metrics` | | Prometheus text format (key required) |

**Status codes:**

| Code | Meaning |
|---|---|
| 401 | Missing or wrong key (`unauthorized`) |
| 403 | Not an admin key, or admin reload disabled (`forbidden`) |
| 405 | Wrong method for the path (`method_not_allowed`) |
| 409 | A reload or reconnect is already running (`conflict`) |
| 422 | Invalid input: a Pydantic validation error or a `PriceInputError`, with its `field` |
| 404 | Unknown listing, area or path (`not_found`) |
| 429 | Rate limit exceeded (`rate_limited`) |
| 503 | Component unavailable, with its redacted load error, or a lost database connection (`unavailable`) |
| 500 | Anything else, as a generic message (`internal`); the traceback goes to the logs only |

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

Made during implementation and the end-of-phase review (2026-09-17):

- **Areas component.** A fifth component, `areas` (`api/areas.py`, `api/routes/areas.py`), serves `GET /v1/areas` (a 12-month summary per area) and `GET /v1/areas/{area_id}/history` (up to 96 monthly medians per market kind) for the Phase 8 demo. Its rows come from `load_homes`, filtered by Phase 6's `validate_rows` (clean rows, positive price and size, a date and an area, no duplicate transaction ids or repeat sales) and by a non-null `area_id`, with no outlier screening. The loader reuses `context["homes"]` when an earlier loader left one there. The forecast loader can't share that read, because `Forecaster.from_registry` loads its own rows.
- **New settings:**
  - `API_COMPONENTS`: which components load (blank means all);
  - `API_EMBED_DEVICE`: `auto`, `cpu` or `cuda`, for the one text embedder that search and listings share;
  - `API_ADMIN_KEYS`: see "Admin keys" below.

  All three are in `.env.example`.
- **Reload locking.** Routes hold the component's lock (`AppState.lock(name)`) for the whole operation and fetch the component inside it. Each component's closer is wrapped so it takes the same lock. So a reload, which builds the new state, swaps it in and then closes the old one, waits for in-flight requests before closing their connections.
- **Database reconnect.**
  - **Liveness probe.** `LoadedComponent` has an optional `check`. For `search` and `listings` it confirms the connection isn't closed and that `SELECT 1` succeeds (rolled back).
  - **Live status.** `AppState.status(live=True)` runs each probe under the component's lock with a non-blocking acquire. A busy component counts as up, and a failed probe reports `up: false` with the error `database connection lost`. `/v1/ready` and `/metrics` use the live status.
  - **Dropped connection.** When the search or listings routes hit `psycopg2.OperationalError` or `InterfaceError`, they answer 503 `unavailable` ("<name> lost its database connection; it is reconnecting — retry shortly").
  - **Rebuild.** The same error starts a daemon thread running `StateHolder.rebuild(name)`, at most one per component at a time. The rebuild re-runs only that component's loader, with a fresh context that reuses the shared embedder (`AppState.context`). It swaps the result in and closes the old component under its lock. If the rebuild fails, the old component stays, so the next failing request retries.
  - **Serialized with reload.** Rebuilds and reloads share one lock.
  - **Rollbacks.** The checker's `finally` rollbacks swallow `psycopg2.Error`, so a dead connection's failed rollback can't replace the original error.
- **Admin keys.**
  - `POST /v1/admin/reload` needs a key listed in `API_ADMIN_KEYS`, and each admin key must also be in `API_KEYS`, or startup fails.
  - A valid non-admin key gets 403 `forbidden`.
  - With `API_ADMIN_KEYS` empty, reload is disabled: 403, "admin reload is disabled: set API_ADMIN_KEYS".
  - A reload while another reload or a rebuild runs gets 409 `conflict`.
- **Error exposure and logging.**
  - **503s and `/v1/ready`.** Load errors appear only as `"<ExceptionType>: <first 120 characters>"`, after `api.errors.redact()` masks `host=`/`user=`/`password=` values, credential URIs and libpq's quoted server and user names. Full errors go to the logs.
  - **Unhandled exceptions.** Each one is logged once with its traceback (by the exception handler); the access line is an error-level line without it.
  - **Other codes.** A 405 has the code `method_not_allowed`. A 429 is attributed to its key in the access log.
  - **Auth.** API keys are matched by comparing SHA-256 digests against every configured key, with no early return. Starting with auth disabled logs a warning.
  - **ReDoc** is off (`redoc_url=None`); `/docs` stays.
- **Request bodies.** `ListingCheckRequest` forbids unknown fields, and it and `ForecastRequest` reject infinite and NaN numbers (422). The unused `area` field was dropped from `ListingCheckRequest`.
- **Smoke rules.** `python -m api smoke` requires 200 from every endpoint on both the first and the warm call; there is no 404 exception.
  - **Areas.** When `areas` is enabled, the run includes `GET /v1/areas`, then `GET /v1/areas/{id}/history` for the first summary area; no area there fails the run.
  - **Sample listing.** The check body mirrors `listings.fraud.price_request` (including `size_basis` and `bedrooms`).
  - **Output.** The table shows both statuses, and the command prints the path of the `smoke.json` it writes.
- **`photo_reuse` own-area rule.** In `POST /v1/listings/check`, the new listing's own `area_id` is added to each photo set's area set, and the listing itself to its listing count. The flag fires when that total reaches `photo_reuse_min_areas`. The detail carries `photo_set_id`, `areas` and `listings`, as in Phase 4.
- **Direct-neighbour relist.** `inconsistent_relist` in the check uses the new listing's direct neighbours: the candidates whose price-blind score clears the threshold. Phase 4 used transitive clusters over all decided pairs, and a new listing has no stored pairs to chain through.
- **Stored flags and the corpus.** `GET /v1/listings/{id}/flags` reads the latest detect run whose `corpus_run_id` is the latest corpus run. With none, it returns the listing with `detect_run_id: null` and empty lists.
- **Model versions.** `model_versions.price` in the check is the loaded price model's registry version (from `resolve_price_model_version`), even when this listing couldn't be priced.
