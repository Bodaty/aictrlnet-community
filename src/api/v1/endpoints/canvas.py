"""A2UI Canvas endpoints — render, auto-detect, and list templates.

Community edition provides chart, table, text, metric, and status block types.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends

from core.dependencies import get_current_user_safe
from schemas.canvas import (
    AutoDetectRequest,
    AutoDetectResponse,
    CanvasRenderRequest,
    CanvasRenderResponse,
    CanvasTemplateListResponse,
)
from services.canvas_service import CanvasRenderService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["canvas"])


@router.post("/render", response_model=CanvasRenderResponse)
async def render_canvas(
    request: CanvasRenderRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Render a canvas from specified blocks and layout."""
    service = CanvasRenderService()
    return service.render(request)


@router.post("/auto-detect", response_model=AutoDetectResponse)
async def auto_detect(
    request: AutoDetectRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Auto-detect the best block type for given data."""
    service = CanvasRenderService()
    return service.auto_detect(request)


@router.get("/templates", response_model=CanvasTemplateListResponse)
async def list_templates(
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """List available canvas templates."""
    service = CanvasRenderService()
    return service.get_templates()
