"""GET /v1/areas and GET /v1/areas/{area_id}/history — area price statistics for the demo."""

from fastapi import APIRouter, Depends

from api.app import current_state, require_key
from api.errors import ApiError
from api.state import AppState

router = APIRouter()


@router.get("/v1/areas")
def areas(
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    stats = state.get("areas")
    return {"data_end": stats.data_end.isoformat(), "areas": stats.summary_records()}


@router.get("/v1/areas/{area_id}/history")
def area_history(
    area_id: int,
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    stats = state.get("areas")
    found = stats.history_records(area_id)
    if found is None:
        raise ApiError(404, "not_found", f"area {area_id} not found")
    return found
