"""Community Runtime Gateway API endpoints (audit-only mode).

Provides registration, evaluation (always ALLOW), reporting, heartbeat,
and listing for external AI agent runtimes. Community edition provides
an audit trail without full Q/G/S/M policy evaluation.

Mounted at prefix ``/runtime`` by the community router.
"""

import hashlib
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.dependencies import get_current_user_safe
from models.runtime_gateway import RuntimeInstance
from schemas.runtime_gateway_basic import (
    RuntimeRegistrationRequest,
    RuntimeRegistrationResponse,
    ActionEvaluationRequest,
    ActionEvaluationResponse,
    ActionReportRequest,
    ActionReportResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    RuntimeListResponse,
    EvaluationListResponse,
)
from services.runtime_audit_service import RuntimeAuditService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runtime-gateway-basic"])


# ── API Key Auth Dependency ──────────────────────────────────────────────────


async def get_runtime_from_api_key(
    request: Request,
    db: AsyncSession = Depends(get_db),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> RuntimeInstance:
    """Authenticate a runtime via ``X-API-Key`` header (preferred) or
    ``Authorization: Bearer <key>`` (backwards-compat).
    """
    raw_key: Optional[str] = None

    if x_api_key:
        raw_key = x_api_key
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token.startswith("rtgw_"):
            raw_key = token

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header or Authorization: Bearer rtgw_...",
        )

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    result = await db.execute(
        select(RuntimeInstance).where(RuntimeInstance.api_key_hash == key_hash)
    )
    instance = result.scalar_one_or_none()

    if instance is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if instance.status == "suspended":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Runtime instance is suspended",
        )

    if instance.status == "deregistered":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Runtime instance is deregistered",
        )

    return instance


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post(
    "/register",
    response_model=RuntimeRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_runtime(
    request: RuntimeRegistrationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Register a new external AI agent runtime (Community: audit-only)."""
    service = RuntimeAuditService(db)
    try:
        return await service.register_runtime(request, current_user)
    except Exception as exc:
        logger.error(f"Runtime registration failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(exc)}",
        )


@router.post(
    "/evaluate",
    response_model=ActionEvaluationResponse,
)
async def evaluate_action(
    request: ActionEvaluationRequest,
    db: AsyncSession = Depends(get_db),
    runtime: RuntimeInstance = Depends(get_runtime_from_api_key),
):
    """Evaluate an action — Community always returns ALLOW with audit trail."""
    service = RuntimeAuditService(db)
    try:
        return await service.audit_evaluate(request, runtime_instance=runtime)
    except Exception as exc:
        logger.error(f"Action evaluation failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(exc)}",
        )


@router.post(
    "/report",
    response_model=ActionReportResponse,
    status_code=status.HTTP_201_CREATED,
)
async def report_action(
    request: ActionReportRequest,
    db: AsyncSession = Depends(get_db),
    runtime: RuntimeInstance = Depends(get_runtime_from_api_key),
):
    """Report the outcome of an executed action."""
    service = RuntimeAuditService(db)
    try:
        return await service.report_action(request, runtime_instance=runtime)
    except Exception as exc:
        logger.error(f"Action report failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report failed: {str(exc)}",
        )


@router.post(
    "/heartbeat",
    response_model=HeartbeatResponse,
)
async def heartbeat(
    request: HeartbeatRequest = None,
    db: AsyncSession = Depends(get_db),
    runtime: RuntimeInstance = Depends(get_runtime_from_api_key),
):
    """Send a heartbeat to keep the runtime marked as active."""
    service = RuntimeAuditService(db)
    try:
        return await service.heartbeat(runtime, request)
    except Exception as exc:
        logger.error(f"Heartbeat failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Heartbeat failed: {str(exc)}",
        )


@router.get(
    "/instances",
    response_model=RuntimeListResponse,
)
async def list_instances(
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """List registered runtime instances."""
    org_id = current_user.get("organization_id") or getattr(
        current_user, "organization_id", None
    )
    service = RuntimeAuditService(db)
    return await service.list_instances(organization_id=org_id)


@router.delete(
    "/instances/{instance_id}",
    status_code=status.HTTP_200_OK,
)
async def deregister_runtime(
    instance_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Deregister a runtime instance.

    Sets the instance status to ``deregistered``. The instance's API key
    will no longer be accepted for evaluate/report/heartbeat calls.
    """
    result = await db.execute(
        select(RuntimeInstance).where(RuntimeInstance.id == instance_id)
    )
    instance = result.scalar_one_or_none()
    if instance is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Runtime instance '{instance_id}' not found",
        )
    instance.status = "deregistered"
    await db.commit()
    return {"detail": f"Runtime instance '{instance_id}' deregistered", "instance_id": instance_id}


@router.get(
    "/evaluations",
    response_model=EvaluationListResponse,
)
async def list_evaluations(
    runtime_instance_id: Optional[str] = Query(None, description="Filter by runtime instance"),
    decision: Optional[str] = Query(None, description="Filter by decision"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Query action evaluations with optional filters."""
    service = RuntimeAuditService(db)
    return await service.list_evaluations(
        runtime_instance_id=runtime_instance_id,
        decision=decision,
        limit=limit,
        offset=offset,
    )
