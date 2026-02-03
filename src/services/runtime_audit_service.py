"""Community Runtime Audit Service — audit-only mode (always ALLOW).

Provides runtime registration, heartbeat, and audit trail for action evaluations.
Community edition always returns ALLOW with audit logging.
Business/Enterprise editions override with full Q/G/S/M policy evaluation.
"""

import hashlib
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from models.runtime_gateway import (
    ActionEvaluation,
    ActionReport,
    RuntimeInstance,
)
from schemas.runtime_gateway_basic import (
    ActionEvaluationRequest,
    ActionEvaluationResponse,
    ActionReportRequest,
    ActionReportResponse,
    EvaluationListResponse,
    EvaluationSummary,
    HeartbeatRequest,
    HeartbeatResponse,
    RuntimeInstanceResponse,
    RuntimeListResponse,
    RuntimeRegistrationRequest,
    RuntimeRegistrationResponse,
)

logger = logging.getLogger(__name__)

HEARTBEAT_TIMEOUT_SECONDS = int(
    os.environ.get("RUNTIME_HEARTBEAT_TIMEOUT_SECONDS", "300")
)


def _hash_api_key(raw_key: str) -> str:
    """SHA-256 hex digest."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _generate_api_key() -> Tuple[str, str]:
    """Return (raw_key, sha256_hash)."""
    raw = f"rtgw_{secrets.token_urlsafe(32)}"
    return raw, _hash_api_key(raw)


class RuntimeAuditService:
    """Community audit-only runtime service. Always returns ALLOW."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Registration ─────────────────────────────────────────────────────

    async def register_runtime(
        self, request: RuntimeRegistrationRequest, user: Dict[str, Any]
    ) -> RuntimeRegistrationResponse:
        """Register a new runtime instance and return the one-time API key."""
        raw_key, key_hash = _generate_api_key()
        instance_id = str(uuid.uuid4())
        now = datetime.utcnow()

        user_id = user.get("id") or getattr(user, "id", None)
        org_id = user.get("organization_id") or getattr(user, "organization_id", None)

        instance = RuntimeInstance(
            id=instance_id,
            runtime_type=request.runtime_type,
            instance_name=request.instance_name,
            organization_id=org_id,
            user_id=str(user_id) if user_id else None,
            capabilities=request.capabilities,
            status="active",
            last_heartbeat=now,
            api_key_hash=key_hash,
            config=request.config,
            resource_metadata=request.resource_metadata,
            total_evaluations=0,
            allowed_count=0,
            denied_count=0,
            escalated_count=0,
            created_at=now,
            updated_at=now,
        )
        self.db.add(instance)
        await self.db.commit()
        await self.db.refresh(instance)

        return RuntimeRegistrationResponse(
            id=instance.id,
            runtime_type=instance.runtime_type,
            instance_name=instance.instance_name,
            status=instance.status,
            api_key=raw_key,
            created_at=instance.created_at,
        )

    # ── Evaluate (audit-only: always ALLOW) ──────────────────────────────

    async def audit_evaluate(
        self,
        request: ActionEvaluationRequest,
        runtime_instance: Optional[RuntimeInstance] = None,
    ) -> ActionEvaluationResponse:
        """Evaluate an action — Community always returns ALLOW for audit trail."""
        start = time.monotonic()

        if runtime_instance is not None:
            instance = runtime_instance
            if not request.runtime_instance_id:
                request.runtime_instance_id = instance.id
        else:
            if not request.runtime_instance_id:
                return ActionEvaluationResponse(
                    evaluation_id="rejected",
                    decision="ALLOW",
                    reasons=["runtime_instance_id is required"],
                    risk_score=0.0,
                    risk_level="low",
                    evaluation_duration_ms=0,
                )
            result = await self.db.execute(
                select(RuntimeInstance).where(RuntimeInstance.id == request.runtime_instance_id)
            )
            instance = result.scalar_one_or_none()

        if instance is None or instance.status in ("suspended", "deregistered"):
            return ActionEvaluationResponse(
                evaluation_id="rejected",
                decision="ALLOW",
                reasons=["Runtime instance not found or inactive"],
                risk_score=0.0,
                risk_level="low",
                evaluation_duration_ms=0,
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Persist evaluation for audit trail
        eval_id = str(uuid.uuid4())
        evaluation = ActionEvaluation(
            id=eval_id,
            runtime_instance_id=request.runtime_instance_id,
            action_type=request.action.action_type,
            action_target=request.action.target,
            action_description=request.action.description,
            risk_score=0.0,
            risk_level="low",
            decision="ALLOW",
            decision_reasons=["Community edition: audit-only mode (always ALLOW)"],
            policies_evaluated=[],
            evaluation_duration_ms=elapsed_ms,
            context_data=request.context.model_dump() if request.context else {},
            risk_hints=request.risk_hints.model_dump() if request.risk_hints else {},
            conditions=[],
            created_at=datetime.utcnow(),
        )
        self.db.add(evaluation)

        # Update counters
        await self.db.execute(
            update(RuntimeInstance)
            .where(RuntimeInstance.id == request.runtime_instance_id)
            .values(
                total_evaluations=RuntimeInstance.total_evaluations + 1,
                allowed_count=RuntimeInstance.allowed_count + 1,
                last_heartbeat=datetime.utcnow(),
            )
        )
        await self.db.commit()

        return ActionEvaluationResponse(
            evaluation_id=eval_id,
            decision="ALLOW",
            reasons=["Community edition: audit-only mode (always ALLOW)"],
            conditions=[],
            risk_score=0.0,
            risk_level="low",
            evaluation_duration_ms=elapsed_ms,
        )

    # ── Report ───────────────────────────────────────────────────────────

    async def report_action(
        self,
        request: ActionReportRequest,
        runtime_instance: Optional[RuntimeInstance] = None,
    ) -> ActionReportResponse:
        """Persist a post-execution action report."""
        if runtime_instance is not None and not request.runtime_instance_id:
            request.runtime_instance_id = runtime_instance.id
        report_id = str(uuid.uuid4())
        now = datetime.utcnow()

        report = ActionReport(
            id=report_id,
            evaluation_id=request.evaluation_id,
            runtime_instance_id=request.runtime_instance_id,
            action_type=request.action_type,
            status=request.status,
            result_summary=request.result_summary,
            quality_score=None,
            duration_ms=request.duration_ms,
            resource_metadata=request.resource_metadata,
            created_at=now,
        )
        self.db.add(report)
        await self.db.commit()

        return ActionReportResponse(
            report_id=report_id,
            evaluation_id=request.evaluation_id,
            status=request.status,
            quality_score=None,
            created_at=now,
        )

    # ── Heartbeat ────────────────────────────────────────────────────────

    async def heartbeat(
        self,
        runtime_instance: RuntimeInstance,
        request: Optional[HeartbeatRequest] = None,
    ) -> HeartbeatResponse:
        """Update heartbeat timestamp."""
        now = datetime.utcnow()
        values = {
            "last_heartbeat": now,
            "status": "active",
            "updated_at": now,
        }
        if request and request.resource_metadata:
            values["resource_metadata"] = request.resource_metadata
        await self.db.execute(
            update(RuntimeInstance)
            .where(RuntimeInstance.id == runtime_instance.id)
            .values(**values)
        )
        await self.db.commit()
        return HeartbeatResponse(
            instance_id=runtime_instance.id,
            status="active",
            last_heartbeat=now,
            next_heartbeat_deadline_seconds=HEARTBEAT_TIMEOUT_SECONDS,
        )

    # ── Queries ──────────────────────────────────────────────────────────

    async def list_instances(
        self, organization_id: Optional[str] = None
    ) -> RuntimeListResponse:
        """List registered runtime instances."""
        # Check for stale runtimes
        cutoff = datetime.utcnow() - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)
        await self.db.execute(
            update(RuntimeInstance)
            .where(
                RuntimeInstance.status == "active",
                RuntimeInstance.last_heartbeat < cutoff,
            )
            .values(status="stale", updated_at=datetime.utcnow())
        )

        query = select(RuntimeInstance)
        if organization_id:
            query = query.where(RuntimeInstance.organization_id == organization_id)
        query = query.order_by(RuntimeInstance.created_at.desc())
        result = await self.db.execute(query)
        instances = result.scalars().all()

        return RuntimeListResponse(
            instances=[
                RuntimeInstanceResponse(
                    id=i.id,
                    runtime_type=i.runtime_type,
                    instance_name=i.instance_name,
                    organization_id=i.organization_id,
                    capabilities=i.capabilities or [],
                    status=i.status,
                    last_heartbeat=i.last_heartbeat,
                    total_evaluations=i.total_evaluations,
                    allowed_count=i.allowed_count,
                    denied_count=i.denied_count,
                    escalated_count=i.escalated_count,
                    config=i.config or {},
                    resource_metadata=i.resource_metadata or {},
                    created_at=i.created_at,
                    updated_at=i.updated_at,
                )
                for i in instances
            ],
            total=len(instances),
        )

    async def list_evaluations(
        self,
        runtime_instance_id: Optional[str] = None,
        decision: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> EvaluationListResponse:
        """Query action evaluations."""
        query = select(ActionEvaluation)
        if runtime_instance_id:
            query = query.where(ActionEvaluation.runtime_instance_id == runtime_instance_id)
        if decision:
            query = query.where(ActionEvaluation.decision == decision)
        query = query.order_by(ActionEvaluation.created_at.desc())

        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        query = query.offset(offset).limit(limit)
        result = await self.db.execute(query)
        evals = result.scalars().all()

        return EvaluationListResponse(
            evaluations=[
                EvaluationSummary(
                    id=e.id,
                    runtime_instance_id=e.runtime_instance_id,
                    action_type=e.action_type,
                    action_target=e.action_target,
                    risk_score=e.risk_score or 0.0,
                    risk_level=e.risk_level or "low",
                    decision=e.decision,
                    decision_reasons=e.decision_reasons or [],
                    conditions=e.conditions or [],
                    evaluation_duration_ms=e.evaluation_duration_ms,
                    created_at=e.created_at,
                )
                for e in evals
            ],
            total=total,
        )
