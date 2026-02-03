"""add_runtime_gateway_basic

Community Runtime Gateway tables — audit-only mode.
Uses inspector pattern to avoid conflicts with Business edition tables.

Revision ID: b4c7d9e1f234
Revises: a3f9c8e71d42
Create Date: 2026-02-02 10:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b4c7d9e1f234"
down_revision: Union[str, None] = "a3f9c8e71d42"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    existing_tables = inspector.get_table_names()

    # ── runtime_instances ────────────────────────────────────────────────
    if "runtime_instances" not in existing_tables:
        op.create_table(
            "runtime_instances",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("runtime_type", sa.String(100), nullable=False),
            sa.Column("instance_name", sa.String(255), nullable=False),
            sa.Column("organization_id", sa.String(), nullable=True),
            sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("capabilities", sa.JSON(), default=[]),
            sa.Column("status", sa.String(50), nullable=False, server_default="active"),
            sa.Column("last_heartbeat", sa.DateTime(), nullable=True),
            sa.Column("api_key_hash", sa.String(64), nullable=False),
            sa.Column("config", sa.JSON(), default={}),
            sa.Column("resource_metadata", sa.JSON(), default={}),
            sa.Column("total_evaluations", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("allowed_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("denied_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("escalated_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index("ix_runtime_instances_status", "runtime_instances", ["status"])
        op.create_index("ix_runtime_instances_org", "runtime_instances", ["organization_id"])
        op.create_index("ix_runtime_instances_user", "runtime_instances", ["user_id"])

    # ── action_evaluations ───────────────────────────────────────────────
    if "action_evaluations" not in existing_tables:
        op.create_table(
            "action_evaluations",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("runtime_instance_id", sa.String(), sa.ForeignKey("runtime_instances.id"), nullable=False),
            sa.Column("action_type", sa.String(100), nullable=False),
            sa.Column("action_target", sa.String(500), nullable=True),
            sa.Column("action_description", sa.Text(), nullable=True),
            sa.Column("risk_score", sa.Float(), server_default="0.0"),
            sa.Column("risk_level", sa.String(50), server_default="'low'"),
            sa.Column("decision", sa.String(20), nullable=False),
            sa.Column("decision_reasons", sa.JSON(), default=[]),
            sa.Column("policies_evaluated", sa.JSON(), default=[]),
            sa.Column("evaluation_duration_ms", sa.Integer(), nullable=True),
            sa.Column("context_data", sa.JSON(), default={}),
            sa.Column("risk_hints", sa.JSON(), default={}),
            sa.Column("conditions", sa.JSON(), default=[]),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index("ix_action_evaluations_runtime", "action_evaluations", ["runtime_instance_id"])
        op.create_index("ix_action_evaluations_decision", "action_evaluations", ["decision"])
        op.create_index("ix_action_evaluations_created", "action_evaluations", ["created_at"])

    # ── action_reports ───────────────────────────────────────────────────
    if "action_reports" not in existing_tables:
        op.create_table(
            "action_reports",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("evaluation_id", sa.String(), sa.ForeignKey("action_evaluations.id"), nullable=False),
            sa.Column("runtime_instance_id", sa.String(), sa.ForeignKey("runtime_instances.id"), nullable=False),
            sa.Column("action_type", sa.String(100), nullable=False),
            sa.Column("status", sa.String(50), nullable=False),
            sa.Column("result_summary", sa.Text(), nullable=True),
            sa.Column("quality_score", sa.Float(), nullable=True),
            sa.Column("duration_ms", sa.Integer(), nullable=True),
            sa.Column("resource_metadata", sa.JSON(), default={}),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index("ix_action_reports_evaluation", "action_reports", ["evaluation_id"])
        op.create_index("ix_action_reports_runtime", "action_reports", ["runtime_instance_id"])
        op.create_index("ix_action_reports_status", "action_reports", ["status"])


def downgrade() -> None:
    op.drop_table("action_reports")
    op.drop_table("action_evaluations")
    op.drop_table("runtime_instances")
