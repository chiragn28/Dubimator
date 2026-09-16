"""GET /v1/search — the Phase 5 ranked search, serialized under `state.lock("search")`
because SearchEngine holds one non-thread-safe pgvector connection."""

import logging

from fastapi import APIRouter, Depends, Query

from api.app import current_state, require_key
from api.state import AppState

LOGGER = logging.getLogger("api.search")

router = APIRouter()


@router.get("/v1/search")
def search(
    q: str = Query(min_length=1, max_length=300),
    k: int = Query(10, ge=1, le=50),
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    engine = state.get("search")
    LOGGER.debug("search", extra={"q": q[:100]})
    with state.lock("search"):
        result = engine.search(q, k)
    return result.to_dict()
