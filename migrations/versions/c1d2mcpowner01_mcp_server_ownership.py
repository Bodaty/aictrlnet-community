"""Add ownership columns to mcp_servers (per-owner authorization).

Adds nullable owner_user_id + tenant_id to mcp_servers so MCP server mutation
can be scoped to the creating user (or a superuser) instead of any authenticated
user. Pre-existing rows keep owner_user_id = NULL and are treated as
"shared/system" servers: readable by all, mutable only by a superuser.

This migration ALSO merges the two open Alembic heads
(b7w7c3d4e5f6 = mcp_wave7_tables, a3f9c8e71d42 = enable_rls_policies) so the
community history has a single head again.

Revision ID: c1d2mcpowner01
Revises: b7w7c3d4e5f6, a3f9c8e71d42
Create Date: 2026-07-02
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c1d2mcpowner01"
down_revision = ("b7w7c3d4e5f6", "a3f9c8e71d42")  # merge both heads
branch_labels = None
depends_on = None


TABLE = "mcp_servers"


def _existing_columns(connection) -> set:
    inspector = sa.inspect(connection)
    return {col["name"] for col in inspector.get_columns(TABLE)}


def upgrade() -> None:
    connection = op.get_bind()
    existing = _existing_columns(connection)

    if "owner_user_id" not in existing:
        op.add_column(TABLE, sa.Column("owner_user_id", sa.String(length=64), nullable=True))
        op.create_index("ix_mcp_servers_owner_user_id", TABLE, ["owner_user_id"])

    if "tenant_id" not in existing:
        op.add_column(TABLE, sa.Column("tenant_id", sa.String(length=64), nullable=True))
        op.create_index("ix_mcp_servers_tenant_id", TABLE, ["tenant_id"])


def downgrade() -> None:
    connection = op.get_bind()
    existing = _existing_columns(connection)

    if "tenant_id" in existing:
        op.drop_index("ix_mcp_servers_tenant_id", table_name=TABLE)
        op.drop_column(TABLE, "tenant_id")

    if "owner_user_id" in existing:
        op.drop_index("ix_mcp_servers_owner_user_id", table_name=TABLE)
        op.drop_column(TABLE, "owner_user_id")
