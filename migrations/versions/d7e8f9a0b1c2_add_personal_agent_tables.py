"""add_personal_agent_tables

Personal Agent Hub tables for Community Edition.
Uses inspector pattern to avoid conflicts with higher editions.

Revision ID: d7e8f9a0b1c2
Revises: c5d6e7f8a9b0
Create Date: 2026-02-02 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d7e8f9a0b1c2"
down_revision: Union[str, None] = "c5d6e7f8a9b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    existing_tables = inspector.get_table_names()

    # ── personal_agent_configs ────────────────────────────────────────
    if "personal_agent_configs" not in existing_tables:
        op.create_table(
            "personal_agent_configs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
                unique=True,
            ),
            sa.Column("agent_name", sa.String(255), nullable=False, server_default="My Assistant"),
            sa.Column("personality", sa.JSON(), nullable=True),
            sa.Column("preferences", sa.JSON(), nullable=True),
            sa.Column("active_workflows", sa.JSON(), nullable=True),
            sa.Column("max_workflows", sa.Integer(), nullable=False, server_default="5"),
            sa.Column("status", sa.String(20), nullable=False, server_default="active"),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index(
            "ix_personal_agent_configs_user_id",
            "personal_agent_configs",
            ["user_id"],
        )
        op.create_index(
            "ix_personal_agent_configs_status",
            "personal_agent_configs",
            ["status"],
        )

    # ── personal_agent_memories ───────────────────────────────────────
    if "personal_agent_memories" not in existing_tables:
        op.create_table(
            "personal_agent_memories",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "config_id",
                sa.String(36),
                sa.ForeignKey("personal_agent_configs.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("memory_type", sa.String(50), nullable=False, server_default="interaction"),
            sa.Column("content", sa.JSON(), nullable=False),
            sa.Column("importance_score", sa.Float(), nullable=False, server_default="0.5"),
            sa.Column("expires_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index(
            "ix_personal_agent_memories_config_id",
            "personal_agent_memories",
            ["config_id"],
        )
        op.create_index(
            "ix_personal_agent_memories_memory_type",
            "personal_agent_memories",
            ["memory_type"],
        )
        op.create_index(
            "ix_personal_agent_memories_importance",
            "personal_agent_memories",
            ["importance_score"],
        )
        op.create_index(
            "ix_personal_agent_memories_created_at",
            "personal_agent_memories",
            ["created_at"],
        )


def downgrade() -> None:
    op.drop_table("personal_agent_memories")
    op.drop_table("personal_agent_configs")
