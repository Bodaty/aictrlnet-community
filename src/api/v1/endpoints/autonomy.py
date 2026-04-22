"""AI Control Spectrum — Community autonomy subset.

Pure-Community deploys need a minimal autonomy surface so the frontend's
autonomy dial still functions end-to-end. Business edition overrides this
with the full hierarchy (`aictrlnet_business.api.v1.endpoints.autonomy`)
mounted at the same path.

Exposed paths (registered at `/api/v1/autonomy` by community_router):
    GET  /resolve            — effective policy for (workflow, user)
    GET  /workflow/{id}      — per-workflow policy
    PUT  /workflow/{id}      — owner-only update
    POST /preview            — gate/auto-approve counts per workflow
    GET  /phases             — phase catalog

Org/Dept/Agent endpoints are Business-only and not registered here.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from models.community_complete import WorkflowDefinition
from services.autonomy_resolver import CommunityAutonomyResolver
from services.autonomy_taxonomy import (
    PHASES,
    level_to_auto_approve_threshold,
    level_to_phase,
    node_risk as static_node_risk,
    phase_descriptor,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AutonomyView(BaseModel):
    level: Optional[int] = Field(None, ge=0, le=100)
    phase: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    auto_approve_threshold: Optional[float] = None


class WorkflowAutonomyPayload(BaseModel):
    autonomy_level: Optional[int] = Field(None, ge=0, le=100)
    autonomy_locked: Optional[bool] = None


class PreviewRequest(BaseModel):
    workflow_id: str
    level: int = Field(..., ge=0, le=100)


class PreviewNodeResult(BaseModel):
    node_id: str
    node_type: str
    risk: float
    would_gate: bool
    reason: Optional[str] = None


class PreviewResponse(BaseModel):
    level: int
    phase: str
    auto_approve_threshold: float
    total_nodes: int
    would_gate_count: int
    auto_approve_count: int
    nodes: List[PreviewNodeResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _view(level: Optional[int]) -> AutonomyView:
    if level is None:
        return AutonomyView()
    level = max(0, min(100, int(level)))
    phase_key = level_to_phase(level)
    descriptor = phase_descriptor(phase_key)
    return AutonomyView(
        level=level,
        phase=phase_key,
        label=descriptor.label,
        description=descriptor.tagline,
        auto_approve_threshold=level_to_auto_approve_threshold(level),
    )


def _safe_user_id(current_user: Any) -> str:
    if current_user is None:
        return "system"
    uid = (
        getattr(current_user, "id", None)
        or getattr(current_user, "user_id", None)
        or (current_user.get("id") if isinstance(current_user, dict) else None)
        or (current_user.get("user_id") if isinstance(current_user, dict) else None)
    )
    return str(uid) if uid else "system"


def _safe_tenant(current_user: Any) -> str:
    tid = (
        getattr(current_user, "tenant_id", None)
        or (current_user.get("tenant_id") if isinstance(current_user, dict) else None)
    )
    return str(tid) if tid else "default-tenant"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/phases")
async def list_phases() -> List[Dict[str, Any]]:
    return [
        {
            "key": p.key,
            "label": p.label,
            "tagline": p.tagline,
            "min_level": p.min_level,
            "max_level": p.max_level,
            "midpoint": p.midpoint,
        }
        for p in PHASES
    ]


@router.get("/resolve")
async def resolve_autonomy(
    workflow_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
) -> Dict[str, Any]:
    user_id = _safe_user_id(current_user)
    tenant_id = _safe_tenant(current_user)
    resolver = CommunityAutonomyResolver(db)
    resolved = await resolver.resolve(
        tenant_id=tenant_id,
        user_id=user_id,
        workflow_id=workflow_id,
    )
    descriptor = phase_descriptor(resolved.phase)
    return {
        "level": resolved.level,
        "phase": resolved.phase,
        "label": descriptor.label,
        "description": descriptor.tagline,
        "source": resolved.source,
        "auto_approve_threshold": resolved.auto_approve_threshold,
        "approver_ids": list(resolved.approver_ids),
        "trace": [[src, lvl] for src, lvl in (resolved.trace or [])],
    }


@router.get("/workflow/{workflow_id}")
async def get_workflow_autonomy(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
) -> Dict[str, Any]:
    result = await db.execute(
        select(
            WorkflowDefinition.autonomy_level,
            WorkflowDefinition.autonomy_locked,
            WorkflowDefinition.tenant_id,
        ).where(WorkflowDefinition.id == workflow_id)
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    level, locked, tenant_id = row
    return {
        "workflow_id": workflow_id,
        "tenant_id": tenant_id,
        "view": _view(level).model_dump(),
        "autonomy_level": level,
        "autonomy_locked": bool(locked or False),
    }


@router.put("/workflow/{workflow_id}")
async def update_workflow_autonomy(
    workflow_id: str,
    payload: WorkflowAutonomyPayload,
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
) -> Dict[str, Any]:
    existing = await db.execute(
        select(WorkflowDefinition.tenant_id).where(WorkflowDefinition.id == workflow_id)
    )
    tenant_id = existing.scalar_one_or_none()
    if tenant_id is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    caller_tenant = _safe_tenant(current_user)
    if str(tenant_id) != str(caller_tenant):
        raise HTTPException(status_code=403, detail="Cross-tenant workflow update denied")

    data = payload.model_dump(exclude_unset=True)
    values: Dict[str, Any] = {}
    if "autonomy_level" in data:
        values["autonomy_level"] = data["autonomy_level"]
    if "autonomy_locked" in data:
        values["autonomy_locked"] = bool(data["autonomy_locked"])
    if not values:
        raise HTTPException(status_code=400, detail="No fields to update")

    await db.execute(
        update(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id).values(**values)
    )
    await db.commit()
    return await get_workflow_autonomy(workflow_id, db=db, current_user=current_user)


@router.post("/preview", response_model=PreviewResponse)
async def preview_autonomy(
    payload: PreviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
) -> PreviewResponse:
    result = await db.execute(
        select(WorkflowDefinition.definition).where(WorkflowDefinition.id == payload.workflow_id)
    )
    definition = result.scalar_one_or_none()
    if definition is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if isinstance(definition, str):
        try:
            definition = _json.loads(definition)
        except _json.JSONDecodeError:
            definition = {}

    threshold = level_to_auto_approve_threshold(payload.level)
    nodes: List[PreviewNodeResult] = []
    would_gate = 0
    auto_approve = 0

    for node in definition.get("nodes", []) if isinstance(definition, dict) else []:
        node_id = str(node.get("id", ""))
        node_type = str(node.get("type", ""))
        params = node.get("parameters") or {}
        override = params.get("risk_override") if isinstance(params, dict) else None
        risk = static_node_risk(node_type, risk_override=override)
        gated = risk > threshold
        reason = None
        if gated:
            reason = (
                f"Risk {risk:.2f} > auto-approve threshold {threshold:.2f} "
                f"at level {payload.level}"
            )
            would_gate += 1
        else:
            auto_approve += 1
        nodes.append(
            PreviewNodeResult(
                node_id=node_id,
                node_type=node_type,
                risk=risk,
                would_gate=gated,
                reason=reason,
            )
        )

    return PreviewResponse(
        level=payload.level,
        phase=level_to_phase(payload.level),
        auto_approve_threshold=threshold,
        total_nodes=len(nodes),
        would_gate_count=would_gate,
        auto_approve_count=auto_approve,
        nodes=nodes,
    )


__all__ = ["router"]
