"""mcp meters and idempotency tables

Creates the two MCP-specific tables:

- ``mcp_meters`` — atomic per-tenant per-meter counters used by
  ``mcp_server.metering.charge_atomic``. One row per
  ``(tenant_id, meter, period_start)``.
- ``mcp_idempotency_keys`` — dedup table for ``idempotency_key``
  argument on mutating MCP tools.

Both tables are owned by the community edition since the MCP server
lives there; Business/Enterprise share them.

Revision ID: e1a2b3c4d5f6
Revises: a7b8c9d0e1f2
Create Date: 2026-04-22 13:15:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "e1a2b3c4d5f6"
down_revision = "a7b8c9d0e1f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    existing_tables = set(inspector.get_table_names())

    # ----- mcp_meters -----
    if "mcp_meters" not in existing_tables:
        op.create_table(
            "mcp_meters",
            sa.Column("id", sa.String(), primary_key=True,
                      server_default=sa.text("gen_random_uuid()::text")),
            sa.Column("tenant_id", sa.String(), nullable=False),
            sa.Column("meter", sa.String(length=64), nullable=False),
            sa.Column("counter", sa.BigInteger(), nullable=False,
                      server_default="0"),
            sa.Column("limit_override", sa.BigInteger(), nullable=True),
            sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
            sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.UniqueConstraint(
                "tenant_id", "meter", "period_start",
                name="uq_mcp_meters_tenant_meter_period",
            ),
        )
        op.create_index(
            "idx_mcp_meters_tenant_meter",
            "mcp_meters",
            ["tenant_id", "meter"],
        )

    # ----- mcp_idempotency_keys -----
    if "mcp_idempotency_keys" not in existing_tables:
        op.create_table(
            "mcp_idempotency_keys",
            sa.Column("id", sa.String(), primary_key=True,
                      server_default=sa.text("gen_random_uuid()::text")),
            sa.Column("tenant_id", sa.String(), nullable=False),
            sa.Column("tool_name", sa.String(length=128), nullable=False),
            sa.Column("idempotency_key", sa.String(length=256), nullable=False),
            sa.Column("args_hash", sa.String(length=64), nullable=False),
            sa.Column("response_json", sa.Text(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.UniqueConstraint(
                "tenant_id", "tool_name", "idempotency_key",
                name="uq_mcp_idempotency_key",
            ),
        )
        op.create_index(
            "idx_mcp_idempotency_created",
            "mcp_idempotency_keys",
            ["created_at"],
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    existing_tables = set(inspector.get_table_names())

    if "mcp_idempotency_keys" in existing_tables:
        op.drop_index("idx_mcp_idempotency_created", table_name="mcp_idempotency_keys")
        op.drop_table("mcp_idempotency_keys")

    if "mcp_meters" in existing_tables:
        op.drop_index("idx_mcp_meters_tenant_meter", table_name="mcp_meters")
        op.drop_table("mcp_meters")
