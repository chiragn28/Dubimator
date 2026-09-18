"""GET /v1/listings/{listing_id}/photos and GET /v1/photos/{photo_id} — the corpus images,
so a client can show a listing the way a property site does instead of a row of numbers.

Both read through the listings component (it owns the pgvector connection), so they take the
same lock and reconnect handling as the other listings routes. The image bytes are served by
the API rather than read off disk by the client: that keeps the API the only thing holding the
corpus, so a client stays a pure API consumer.

The files are gitignored (the Houses-dataset licence is unstated, see listings/photos.py), so
a deployment without them gets a 404 from `/v1/photos/{id}` while the rest of the API works.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, Request, Response

from api.app import current_state, require_key
from api.errors import ApiError
from api.routes.db import reconnect_on_loss
from api.state import AppState

router = APIRouter()

# The corpus only ever holds JPEGs (listings/photos.py writes .jpg for both the pool and the
# variants), so the media type is fixed rather than guessed from the path.
MEDIA_TYPE = "image/jpeg"
# Browsers may cache a photo for a day: a photo_id's bytes never change once the corpus is built.
CACHE_CONTROL = "public, max-age=86400"


@router.get("/v1/listings/{listing_id}/photos")
def listing_photos(
    request: Request,
    listing_id: int,
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    with reconnect_on_loss(request, "listings"), state.lock("listings"):
        found = state.get("listings").photos(listing_id)
    if found is None:
        raise ApiError(404, "not_found", f"listing {listing_id} not found")
    return found


def _resolved(root: Path, relative: str) -> Path:
    """`root / relative`, refusing anything that escapes `root`.

    The path comes from the database rather than the request, so this guards against a corpus
    row with a traversing path, not against a hostile caller.
    """
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ApiError(404, "not_found", "photo path is outside the photo root")
    return target


@router.get(
    "/v1/photos/{photo_id}",
    response_class=Response,
    responses={200: {"content": {MEDIA_TYPE: {}}, "description": "The photo, as JPEG bytes."}},
)
def photo(
    request: Request,
    photo_id: int,
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> Response:
    root = request.app.state.settings.photo_root
    if not root:
        raise ApiError(404, "not_found", "this deployment serves no photos")
    with reconnect_on_loss(request, "listings"), state.lock("listings"):
        relative = state.get("listings").photo_path(photo_id)
    if relative is None:
        raise ApiError(404, "not_found", f"photo {photo_id} not found")
    target = _resolved(Path(root), relative)
    try:
        data = target.read_bytes()
    except OSError:
        # The row exists but the file doesn't: the corpus images aren't on this host.
        raise ApiError(404, "not_found", f"photo {photo_id} is not on this server") from None
    return Response(content=data, media_type=MEDIA_TYPE, headers={"Cache-Control": CACHE_CONTROL})
