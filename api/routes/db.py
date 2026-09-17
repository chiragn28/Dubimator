"""Lost database connections: a 503 for the caller and a background rebuild of the component."""

import logging
from contextlib import contextmanager

import psycopg2
from fastapi import Request

from api.errors import ApiError

LOGGER = logging.getLogger("api.db")
CONNECTION_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)


@contextmanager
def reconnect_on_loss(request: Request, name: str):
    """Turns a dropped connection inside the block into a 503 and starts (at most one)
    background rebuild of `name`. Use it outside `state.lock(name)`."""
    try:
        yield
    except CONNECTION_ERRORS as exc:
        LOGGER.warning(
            "%s lost its database connection",
            name,
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        app_state = request.app.state
        app_state.holder.rebuild_in_background(name, app_state.settings, app_state.loaders)
        raise ApiError(
            503,
            "unavailable",
            f"{name} lost its database connection; it is reconnecting — retry shortly",
        ) from exc
