"""Control plane API routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse

from core.security import get_current_active_user
from .models import (
    ComponentRegistrationRequest, ComponentRegistrationResponse,
    Component, ComponentHeartbeat, ComponentEvent,
    ComponentStatus, ComponentType
)
from .services import control_plane_service
from .auth import component_auth


router = APIRouter(prefix="/control-plane", tags=["control-plane"])


@router.post("/components/register", response_model=ComponentRegistrationResponse)
async def register_component(
    request: ComponentRegistrationRequest,
    current_user: dict = Depends(get_current_active_user)
) -> ComponentRegistrationResponse:
    """Register a new component in the control plane."""
    try:
        response = await control_plane_service.register_component(
            request,
            current_user["id"]
        )
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register component: {str(e)}"
        )


@router.post("/components/{component_id}/heartbeat", response_model=Component)
async def send_heartbeat(
    component_id: str,
    heartbeat: ComponentHeartbeat,
    component_info: dict = Depends(component_auth.get_current_component)
) -> Component:
    """Send a heartbeat from a component."""
    # Verify component is sending its own heartbeat
    if component_info["id"] != component_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Components can only send their own heartbeat"
        )
    
    try:
        component = await control_plane_service.process_heartbeat(heartbeat)
        return component
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/components", response_model=List[Component])
async def get_components(
    component_type: Optional[ComponentType] = Query(None),
    status: Optional[ComponentStatus] = Query(None),
    edition: Optional[str] = Query(None),
    capability: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_active_user)
) -> List[Component]:
    """Get registered components with optional filtering."""
    # Filter by user's edition
    user_edition = current_user.get("edition", "community")
    if edition and edition != user_edition:
        # User can only see components from their edition or lower
        edition_hierarchy = ["community", "business", "enterprise"]
        user_idx = edition_hierarchy.index(user_edition)
        requested_idx = edition_hierarchy.index(edition)
        if requested_idx > user_idx:
            edition = user_edition
    else:
        edition = user_edition
    
    components = await control_plane_service.get_components(
        component_type=component_type,
        status=status,
        edition=edition,
        capability=capability
    )
    
    return components


@router.get("/components/{component_id}", response_model=Component)
async def get_component(
    component_id: str,
    current_user: dict = Depends(get_current_active_user)
) -> Component:
    """Get a specific component by ID."""
    component = await control_plane_service.get_component(component_id)
    if not component:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component {component_id} not found"
        )
    
    # Check edition access
    user_edition = current_user.get("edition", "community")
    edition_hierarchy = ["community", "business", "enterprise"]
    user_idx = edition_hierarchy.index(user_edition)
    component_idx = edition_hierarchy.index(component.required_edition)
    
    if component_idx > user_idx:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Component requires {component.required_edition} edition"
        )
    
    return component


@router.put("/components/{component_id}/status")
async def update_component_status(
    component_id: str,
    new_status: ComponentStatus,
    reason: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user)
) -> Component:
    """Update a component's status (admin only)."""
    # Check admin role
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        component = await control_plane_service.update_component_status(
            component_id, new_status, reason
        )
        return component
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.post("/components/{component_id}/result")
async def record_component_result(
    component_id: str,
    success: bool,
    error_message: Optional[str] = None,
    metrics: Optional[dict] = None,
    component_info: dict = Depends(component_auth.get_current_component)
) -> Component:
    """Record the result of a component operation."""
    # Verify component is recording its own result
    if component_info["id"] != component_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Components can only record their own results"
        )
    
    try:
        component = await control_plane_service.record_component_result(
            component_id, success, error_message, metrics
        )
        return component
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/components/{component_id}/events", response_model=List[ComponentEvent])
async def get_component_events(
    component_id: str,
    event_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_current_active_user)
) -> List[ComponentEvent]:
    """Get events for a specific component."""
    events = await control_plane_service.get_component_events(
        component_id=component_id,
        event_type=event_type,
        limit=limit
    )
    return events


@router.post("/components/{component_id}/refresh-token")
async def refresh_component_token(
    component_id: str,
    old_token: str,
    component_info: dict = Depends(component_auth.get_current_component)
) -> dict:
    """Refresh a component's JWT token."""
    # Verify component is refreshing its own token
    if component_info["id"] != component_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Components can only refresh their own token"
        )
    
    try:
        new_token, expires_at = await control_plane_service.refresh_component_token(
            component_id, old_token
        )
        return {
            "token": new_token,
            "expires_at": expires_at
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/health", response_model=dict)
async def get_health_summary(
    current_user: dict = Depends(get_current_active_user)
) -> dict:
    """Get overall health summary of the control plane."""
    summary = await control_plane_service.get_component_health_summary()
    return summary


@router.post("/cleanup")
async def cleanup_inactive_components(
    current_user: dict = Depends(get_current_active_user)
) -> dict:
    """Clean up inactive components (admin only)."""
    # Check admin role
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    inactive_ids = await control_plane_service.cleanup_inactive_components()
    return {
        "cleaned_up": len(inactive_ids),
        "component_ids": inactive_ids
    }