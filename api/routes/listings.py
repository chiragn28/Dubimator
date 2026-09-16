"""GET /v1/listings/{listing_id}/flags and POST /v1/listings/check, serialized under
`state.lock("listings")` because ListingChecker holds one non-thread-safe pgvector
connection (its own internal lock only serializes its own database reads, not the checks
built on top of them)."""

from fastapi import APIRouter, Depends

from api.app import current_state, require_key
from api.errors import ApiError
from api.schemas import ListingCheckRequest
from api.state import AppState

router = APIRouter()


@router.get("/v1/listings/{listing_id}/flags")
def listing_flags(
    listing_id: int,
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    checker = state.get("listings")
    with state.lock("listings"):
        found = checker.stored(listing_id)
    if found is None:
        raise ApiError(404, "not_found", f"listing {listing_id} not found")
    return found


@router.post("/v1/listings/check")
def check_listing(
    body: ListingCheckRequest,
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    checker = state.get("listings")
    with state.lock("listings"):
        return checker.check(body)
