"""Community Runtime Gateway schemas (audit-only, no delegation/webhooks)."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Registration ─────────────────────────────────────────────────────────────


class RuntimeRegistrationRequest(BaseModel):
    runtime_type: str = Field(..., description="Runtime type: openclaw, claude_code, custom")
    instance_name: str = Field(..., description="Human-readable name for this instance")
    capabilities: List[str] = Field(default_factory=list, description="List of capabilities")
    config: Dict[str, Any] = Field(default_factory=dict, description="Runtime configuration")
    resource_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RuntimeRegistrationResponse(BaseModel):
    id: str
    runtime_type: str
    instance_name: str
    status: str
    api_key: str = Field(..., description="One-time displayed API key — store securely")
    created_at: datetime

    class Config:
        from_attributes = True


# ── Evaluate ─────────────────────────────────────────────────────────────────


class RiskHints(BaseModel):
    """Caller-supplied risk hints (Community: recorded for audit, not enforced)."""
    is_production: bool = False
    is_financial: bool = False
    is_pii: bool = False
    is_destructive: bool = False
    is_external: bool = False
    custom: Dict[str, Any] = Field(default_factory=dict)


class ActionDetail(BaseModel):
    """Describes the action the runtime wants to perform."""
    action_type: str = Field(..., description="Action type: file_write, api_call, shell_exec, etc.")
    target: Optional[str] = Field(None, description="Target resource/path")
    description: Optional[str] = Field(None, description="Human-readable description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


class ActionContext(BaseModel):
    """Contextual information about the execution environment."""
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    user_intent: Optional[str] = None
    environment: str = "development"
    extra: Dict[str, Any] = Field(default_factory=dict)


class ActionEvaluationRequest(BaseModel):
    runtime_instance_id: Optional[str] = Field(
        None, description="Registered runtime instance ID (optional when using API key auth)"
    )
    action: ActionDetail
    context: ActionContext = Field(default_factory=ActionContext)
    risk_hints: RiskHints = Field(default_factory=RiskHints)


class ActionEvaluationResponse(BaseModel):
    evaluation_id: str
    decision: str = "ALLOW"
    reasons: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    risk_score: float = 0.0
    risk_level: str = "low"
    evaluation_duration_ms: int = 0


# ── Report ───────────────────────────────────────────────────────────────────


class ActionReportRequest(BaseModel):
    evaluation_id: str = Field(..., description="Evaluation ID from the evaluate response")
    runtime_instance_id: Optional[str] = Field(
        None, description="Runtime instance ID (optional when using API key auth)"
    )
    action_type: str
    status: str = Field(..., description="Outcome: success, failure, partial, error")
    result_summary: Optional[str] = None
    duration_ms: Optional[int] = None
    resource_metadata: Dict[str, Any] = Field(default_factory=dict)


class ActionReportResponse(BaseModel):
    report_id: str
    evaluation_id: str
    status: str
    quality_score: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ── List / Detail ────────────────────────────────────────────────────────────


class RuntimeInstanceResponse(BaseModel):
    id: str
    runtime_type: str
    instance_name: str
    organization_id: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    status: str
    last_heartbeat: Optional[datetime] = None
    total_evaluations: int = 0
    allowed_count: int = 0
    denied_count: int = 0
    escalated_count: int = 0
    config: Dict[str, Any] = Field(default_factory=dict)
    resource_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RuntimeListResponse(BaseModel):
    instances: List[RuntimeInstanceResponse]
    total: int


class EvaluationSummary(BaseModel):
    id: str
    runtime_instance_id: str
    action_type: str
    action_target: Optional[str] = None
    risk_score: float = 0.0
    risk_level: str = "low"
    decision: str
    decision_reasons: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    evaluation_duration_ms: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EvaluationListResponse(BaseModel):
    evaluations: List[EvaluationSummary]
    total: int


# ── Heartbeat ────────────────────────────────────────────────────────────────


class HeartbeatRequest(BaseModel):
    """Optional payload for heartbeat."""
    resource_metadata: Dict[str, Any] = Field(default_factory=dict)


class HeartbeatResponse(BaseModel):
    instance_id: str
    status: str
    last_heartbeat: datetime
    next_heartbeat_deadline_seconds: int = Field(
        default=300, description="Send next heartbeat within this many seconds to stay active"
    )
