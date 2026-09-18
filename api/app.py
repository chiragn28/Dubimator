"""The app factory: lifespan, middleware, exception handlers, and the routers.

Spec: docs/superpowers/specs/2026-09-17-phase7-api-design.md ("Architecture", "Endpoints").

Middleware order, outermost first: request ID, then access log + metrics, then CORS
(when configured). Both custom middlewares subclass `BaseHTTPMiddleware`. That is safe
here because the only contextvar traffic is forward — `request_id_var` is set before
`call_next()` and only ever read downstream of that point (including in a sync route
run on FastAPI's threadpool, since `anyio.to_thread.run_sync` copies the current
context) — never read back in the outer middleware *after* `call_next()` returns. The
one path that would dodge a `call_next`-based header write — an exception that
`ServerErrorMiddleware` turns into a response, since that middleware wraps even our
own middleware — is handled separately: the `Exception` handler below reads
`request.state.request_id` off the same (shared, mutable) `scope` dict and sets the
header itself, and the access-log middleware wraps `call_next` in a try/except so that
path is still timed, logged and counted before the exception continues on to
`ServerErrorMiddleware`.
"""

import logging
import math
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from api.errors import ApiError
from api.logging import configure_logging, new_request_id, request_id_var
from api.metrics import Metrics
from api.security import RateLimiter, check_key, extract_key, key_id
from api.settings import ApiSettings
from api.state import AppState, StateHolder, build_state
from listings.check import UnknownPhotos
from models.price.predictor import PriceInputError

ACCESS_LOGGER = logging.getLogger("api.access")
LOGGER = logging.getLogger("api")


def current_state(request: Request) -> AppState:
    return request.app.state.holder.current


def require_key(request: Request) -> str:
    """The auth + rate-limit dependency. Stores `request.state.key_id` for the access log."""
    settings: ApiSettings = request.app.state.settings
    limiter: RateLimiter = request.app.state.limiter
    if settings.auth_enabled:
        matched = check_key(extract_key(request.headers), settings)
        if matched is None:
            raise ApiError(401, "unauthorized", "missing or invalid API key")
        identity = key_id(matched)
    else:
        identity = "open"
    request.state.key_id = identity  # set first, so a 429 is attributed too
    wait = limiter.acquire(identity)
    if wait is not None:
        raise ApiError(
            429,
            "rate_limited",
            "rate limit exceeded",
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )
    return identity


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return route.path if route is not None else "unmatched"


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = new_request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response


def _access_extra(request: Request, status: int, duration: float) -> dict:
    return {
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "duration_ms": round(duration * 1000, 2),
        "key_id": getattr(request.state, "key_id", None),
    }


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        metrics: Metrics = request.app.state.metrics
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            duration = time.monotonic() - start
            metrics.observe(_route_template(request), request.method, 500, duration)
            # No traceback here: the Exception handler below logs it, once.
            ACCESS_LOGGER.error("request failed", extra=_access_extra(request, 500, duration))
            raise
        duration = time.monotonic() - start
        metrics.observe(_route_template(request), request.method, response.status_code, duration)
        ACCESS_LOGGER.info("request", extra=_access_extra(request, response.status_code, duration))
        return response


def _envelope(request: Request, code: str, message: str, field: str | None = None) -> dict:
    request_id = getattr(request.state, "request_id", None) or request_id_var.get()
    return {"error": {"code": code, "message": message, "field": field}, "request_id": request_id}


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return JSONResponse(
            _envelope(request, exc.code, exc.message, exc.field),
            status_code=exc.status,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        error = exc.errors()[0]
        parts = [str(part) for part in error["loc"] if part not in ("body", "query")]
        field = ".".join(parts) if parts else None
        return JSONResponse(
            _envelope(request, "invalid_input", error["msg"], field), status_code=422
        )

    @app.exception_handler(PriceInputError)
    async def price_input_error_handler(request: Request, exc: PriceInputError):
        return JSONResponse(
            _envelope(request, "invalid_input", exc.message, exc.field), status_code=422
        )

    @app.exception_handler(UnknownPhotos)
    async def unknown_photos_handler(request: Request, exc: UnknownPhotos):
        return JSONResponse(
            _envelope(request, "invalid_input", str(exc), "photo_ids"), status_code=422
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        headers = getattr(exc, "headers", None)
        if exc.status_code == 404:
            return JSONResponse(
                _envelope(request, "not_found", exc.detail or "not found"),
                status_code=404,
                headers=headers,
            )
        if exc.status_code == 405:
            return JSONResponse(
                _envelope(request, "method_not_allowed", exc.detail or "method not allowed"),
                status_code=405,
                headers=headers,
            )
        return JSONResponse(
            _envelope(request, "internal", exc.detail or "error"),
            status_code=exc.status_code,
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # This handler is invoked by Starlette's ServerErrorMiddleware, which sits
        # outside every app middleware (including ours), so it never sees
        # RequestIdMiddleware's post-call_next header write. `request.state` still
        # has what RequestIdMiddleware stored on `scope` before the exception was
        # raised (the same mutable dict, unaffected by that middleware's contextvar
        # reset), so the header and body still carry the right request id.
        request_id = getattr(request.state, "request_id", None) or request_id_var.get()
        LOGGER.error(
            "unhandled exception",
            exc_info=exc,
            extra={"request_id": request_id, "path": request.url.path},
        )
        response = JSONResponse(_envelope(request, "internal", "internal error"), status_code=500)
        response.headers["X-Request-ID"] = request_id
        return response


def create_app(settings: ApiSettings, loaders: dict) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = build_state(settings, loaders)
        app.state.holder = StateHolder(state)
        try:
            yield
        finally:
            app.state.holder.current.close()

    if not settings.auth_enabled:
        LOGGER.warning(
            "API authentication is disabled (API_ALLOW_NO_KEYS=1): every endpoint is open"
        )

    app = FastAPI(lifespan=lifespan, redoc_url=None)
    app.state.settings = settings
    app.state.metrics = Metrics()
    app.state.limiter = RateLimiter(per_minute=settings.rate_limit_per_minute)
    app.state.loaders = loaders

    # add_middleware() inserts at the front of the user-middleware list, so the class
    # added last ends up outermost — CORS first (innermost of the three), then the
    # access log, then request ID last (outermost), matching the order above.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIdMiddleware)

    _register_exception_handlers(app)

    # Deferred import: api.routes.* import `current_state`/`require_key` from this
    # module, so importing them at module scope here would be circular. By the time
    # create_app() runs, this module is already fully defined.
    from api.routes import admin, areas, forecast, health, listings, photos, price, search

    app.include_router(health.router)
    app.include_router(admin.router)
    app.include_router(price.router)
    app.include_router(forecast.router)
    app.include_router(search.router)
    app.include_router(listings.router)
    app.include_router(photos.router)
    app.include_router(areas.router)

    return app


__all__ = ["create_app", "current_state", "require_key"]
