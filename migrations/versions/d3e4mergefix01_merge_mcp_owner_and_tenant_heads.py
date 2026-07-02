"""Merge the two open community heads back to one.

c1d2mcpowner01 (mcp_servers ownership, from the security remediation branch)
and b8a1c2d3e4f5 (tenant_id on adapter_configs) both descend from
b7w7c3d4e5f6, leaving two heads and breaking `alembic upgrade head`.
No-op merge revision; no schema changes.

Revision ID: d3e4mergefix01
Revises: b8a1c2d3e4f5, c1d2mcpowner01
Create Date: 2026-07-02
"""


# revision identifiers, used by Alembic.
revision = "d3e4mergefix01"
down_revision = ("b8a1c2d3e4f5", "c1d2mcpowner01")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
