"""POST /v1/price — the Phase 3 price estimate, unchanged."""

from fastapi import APIRouter, Depends

from api.app import current_state, require_key
from api.state import AppState
from models.price.predictor import PriceRequest

router = APIRouter()


@router.post("/v1/price")
def price(
    body: PriceRequest,
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    predictor = state.get("price")
    return predictor.predict_one(body).model_dump(mode="json")
