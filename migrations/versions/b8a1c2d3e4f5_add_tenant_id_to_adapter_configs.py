"""add_tenant_id_to_adapter_configs

Add a nullable tenant_id column to adapter_configs so per-org (tenant) adapter
credentials can be resolved through get_adapter_credentials_for_tenant (GEO
Phase B2 tiered model). Nullable + additive: existing rows (and the
default-tenant / Bodaty free-tier path, which falls back to the env key) are
unaffected. Tenant filtering happens in the getter; the existing per-user
unique constraint is intentionally left unchanged.

Revision ID: b8a1c2d3e4f5
Revises: b7w7c3d4e5f6
Create Date: 2026-06-09 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b8a1c2d3e4f5"
down_revision: Union[str, None] = "b7w7c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    existing_tables = inspector.get_table_names()
    if "adapter_configs" not in existing_tables:
        # Table created by a later/initial migration — nothing to alter yet.
        return

    existing_columns = [col["name"] for col in inspector.get_columns("adapter_configs")]
    if "tenant_id" not in existing_columns:
        op.add_column(
            "adapter_configs",
            sa.Column("tenant_id", sa.String(length=36), nullable=True),
        )

    existing_indexes = [ix["name"] for ix in inspector.get_indexes("adapter_configs")]
    if "ix_adapter_configs_tenant_id" not in existing_indexes:
        op.create_index(
            "ix_adapter_configs_tenant_id",
            "adapter_configs",
            ["tenant_id"],
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    existing_tables = inspector.get_table_names()
    if "adapter_configs" not in existing_tables:
        return

    existing_indexes = [ix["name"] for ix in inspector.get_indexes("adapter_configs")]
    if "ix_adapter_configs_tenant_id" in existing_indexes:
        op.drop_index("ix_adapter_configs_tenant_id", table_name="adapter_configs")

    existing_columns = [col["name"] for col in inspector.get_columns("adapter_configs")]
    if "tenant_id" in existing_columns:
        op.drop_column("adapter_configs", "tenant_id")
