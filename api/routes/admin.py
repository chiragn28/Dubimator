"""POST /v1/admin/reload (an admin key: one listed in API_ADMIN_KEYS) and GET /metrics
(any key)."""

from fastapi import APIRouter, Depends, Request, Response

from api.app import current_state, require_key
from api.errors import ApiError
from api.security import check_admin, extract_key
from api.state import AppState

router = APIRouter()


@router.post("/v1/admin/reload")
def reload_components(
    request: Request,
    _key: str = Depends(require_key),
) -> dict:
    settings = request.app.state.settings
    if not settings.admin_keys:
        raise ApiError(403, "forbidden", "admin reload is disabled: set API_ADMIN_KEYS")
    if not check_admin(extract_key(request.headers), settings):
        raise ApiError(403, "forbidden", "an admin key is required")
    holder = request.app.state.holder
    new_state = holder.reload(settings, request.app.state.loaders)  # 409 if one is running
    return {"components": new_state.status(live=True)}


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
