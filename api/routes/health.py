"""GET /health (public liveness) and GET /v1/ready (keyed readiness)."""

from fastapi import APIRouter, Depends

from api.app import current_state, require_key
from api.state import AppState

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/v1/ready")
def ready(
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> dict:
    return {"components": state.status(live=True)}
