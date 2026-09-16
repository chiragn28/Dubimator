"""POST /v1/admin/reload and GET /metrics — both need a key."""

from fastapi import APIRouter, Depends, Request, Response

from api.app import current_state, require_key
from api.state import AppState

router = APIRouter()


@router.post("/v1/admin/reload")
def reload_components(
    request: Request,
    _key: str = Depends(require_key),
) -> dict:
    holder = request.app.state.holder
    new_state = holder.reload(request.app.state.settings, request.app.state.loaders)
    return {"components": new_state.status()}


@router.get("/metrics")
def metrics_endpoint(
    request: Request,
    state: AppState = Depends(current_state),  # noqa: B008 — FastAPI's own dependency idiom
    _key: str = Depends(require_key),
) -> Response:
    metrics = request.app.state.metrics
    metrics.set_components(state)
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
