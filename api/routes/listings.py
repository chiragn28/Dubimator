"""GET /v1/listings/{listing_id}/flags and POST /v1/listings/check, serialized under
`state.lock("listings")` because ListingChecker holds one non-thread-safe pgvector
connection (its own internal lock only serializes its own database reads, not the checks
built on top of them). A dropped connection is a 503 plus a background rebuild."""

from fastapi import APIRouter, Depends, Request

from api.app import current_state, require_key
from api.errors import ApiError
from api.routes.db import reconnect_on_loss
from api.schemas import ListingCheckRequest
from api.state import AppState

router = APIRouter()


@router.get("/v1/listings/{listing_id}/flags")
def listing_flags(
    request: Request,
    listing_id: int,
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    with reconnect_on_loss(request, "listings"), state.lock("listings"):
        found = state.get("listings").stored(listing_id)
    if found is None:
        raise ApiError(404, "not_found", f"listing {listing_id} not found")
    return found


@router.post("/v1/listings/check")
def check_listing(
    request: Request,
    body: ListingCheckRequest,
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    with reconnect_on_loss(request, "listings"), state.lock("listings"):
        return state.get("listings").check(body)
