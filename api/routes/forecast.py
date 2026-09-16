"""POST /v1/forecast — the Phase 6 multi-horizon forecast."""

from fastapi import APIRouter, Depends

from api.app import current_state, require_key
from api.schemas import ForecastRequest
from api.state import AppState

router = APIRouter()


@router.post("/v1/forecast")
def forecast(
    body: ForecastRequest,
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    forecaster = state.get("forecast")
    return forecaster.forecast(body.model_dump(exclude_none=True))
