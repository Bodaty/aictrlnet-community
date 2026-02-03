"""Community Runtime Gateway database models.

Stores registered runtime instances, action evaluations (audit-only ALLOW),
and post-execution action reports. Community edition provides audit trail
without full Q/G/S/M policy evaluation.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, JSON, DateTime,
    ForeignKey, Index,
)
from models.base import Base


class RuntimeInstance(Base):
    """A registered external AI agent runtime (Community: audit-only mode)."""

    __tablename__ = "runtime_instances"

    id = Column(String, primary_key=True)
    runtime_type = Column(String(100), nullable=False)  # openclaw, claude_code, custom
    instance_name = Column(String(255), nullable=False)
    organization_id = Column(String, nullable=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    capabilities = Column(JSON, default=[])
    status = Column(String(50), default="active", nullable=False)  # active, suspended, deregistered
    last_heartbeat = Column(DateTime, nullable=True)
    api_key_hash = Column(String(64), nullable=False)  # SHA-256 hex
    config = Column(JSON, default={})
    resource_metadata = Column("resource_metadata", JSON, default={})

    # Denormalized counters
    total_evaluations = Column(Integer, default=0, nullable=False)
    allowed_count = Column(Integer, default=0, nullable=False)
    denied_count = Column(Integer, default=0, nullable=False)
    escalated_count = Column(Integer, default=0, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_runtime_instances_status", "status"),
        Index("ix_runtime_instances_org", "organization_id"),
        Index("ix_runtime_instances_user", "user_id"),
        {"extend_existing": True},
    )


class ActionEvaluation(Base):
    """Result of evaluating an action (Community: always ALLOW for audit)."""

    __tablename__ = "action_evaluations"

    id = Column(String, primary_key=True)
    runtime_instance_id = Column(
        String, ForeignKey("runtime_instances.id"), nullable=False
    )
    action_type = Column(String(100), nullable=False)
    action_target = Column(String(500), nullable=True)
    action_description = Column(Text, nullable=True)

    risk_score = Column(Float, default=0.0)
    risk_level = Column(String(50), default="low")
    decision = Column(String(20), nullable=False)  # Community: always ALLOW
    decision_reasons = Column(JSON, default=[])
    policies_evaluated = Column(JSON, default=[])

    evaluation_duration_ms = Column(Integer, nullable=True)
    context_data = Column(JSON, default={})
    risk_hints = Column(JSON, default={})
    conditions = Column(JSON, default=[])

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_action_evaluations_runtime", "runtime_instance_id"),
        Index("ix_action_evaluations_decision", "decision"),
        Index("ix_action_evaluations_created", "created_at"),
        {"extend_existing": True},
    )


class ActionReport(Base):
    """Post-execution outcome report from the runtime."""

    __tablename__ = "action_reports"

    id = Column(String, primary_key=True)
    evaluation_id = Column(
        String, ForeignKey("action_evaluations.id"), nullable=False
    )
    runtime_instance_id = Column(
        String, ForeignKey("runtime_instances.id"), nullable=False
    )
    action_type = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)  # success, failure, partial, error
    result_summary = Column(Text, nullable=True)
    quality_score = Column(Float, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    resource_metadata = Column("resource_metadata", JSON, default={})

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_action_reports_evaluation", "evaluation_id"),
        Index("ix_action_reports_runtime", "runtime_instance_id"),
        Index("ix_action_reports_status", "status"),
        {"extend_existing": True},
    )
