"""data quality tables keep the tenant id as a string

The seven data-quality tables declared ``tenant_id`` as a UUID, but tenant ids
are strings (``default-tenant``). The handlers fell back to the all-zero UUID,
which the tenant-isolation policy then rejected, so no data-quality write could
succeed. The policies reference the column, so they are dropped and recreated
around the type change, exactly as they were.

Revision ID: dq7tenantstr01
Revises: e5f6convjobs01
Create Date: 2026-10-25 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects import postgresql


revision = "dq7tenantstr01"
down_revision = "e5f6convjobs01"
branch_labels = None
depends_on = None

TABLES = (
    "data_quality_assessments",
    "quality_rules",
    "quality_profiles",
    "data_lineage",
    "quality_slas",
    "quality_improvements",
    "quality_usage_tracking",
)


def _policies(conn, table):
    rows = conn.execute(
        text("SELECT policyname FROM pg_policies WHERE tablename = :t"), {"t": table}
    )
    return {row[0] for row in rows}


def _drop_policies(conn, table):
    existing = _policies(conn, table)
    for name in (f"tenant_isolation_{table}", f"admin_bypass_{table}"):
        if name in existing:
            op.execute(text(f"DROP POLICY {name} ON {table}"))
    return existing


def _restore_policies(table, existing):
    if f"tenant_isolation_{table}" in existing:
        op.execute(text(f"""
            CREATE POLICY tenant_isolation_{table} ON {table}
            FOR ALL
            USING (tenant_id::text = current_setting('app.current_tenant_id', true))
            WITH CHECK (tenant_id::text = current_setting('app.current_tenant_id', true))
        """))
    if f"admin_bypass_{table}" in existing:
        op.execute(text(f"""
            CREATE POLICY admin_bypass_{table} ON {table}
            FOR ALL
            USING (current_setting('app.is_admin', true)::boolean = true)
        """))


def _column_type(conn, table):
    return conn.execute(
        text(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = 'tenant_id'"
        ),
        {"t": table},
    ).scalar()


def upgrade() -> None:
    conn = op.get_bind()
    for table in TABLES:
        if _column_type(conn, table) != "uuid":
            continue
        existing = _drop_policies(conn, table)
        op.alter_column(
            table,
            "tenant_id",
            type_=sa.String(100),
            existing_type=postgresql.UUID(),
            postgresql_using="tenant_id::text",
        )
        _restore_policies(table, existing)


def downgrade() -> None:
    conn = op.get_bind()
    for table in TABLES:
        if _column_type(conn, table) == "uuid":
            continue
        existing = _drop_policies(conn, table)
        # Rows written for string tenants cannot become UUIDs; they are dropped.
        op.execute(text(
            f"DELETE FROM {table} WHERE tenant_id !~ "
            "'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'"
        ))
        op.alter_column(
            table,
            "tenant_id",
            type_=postgresql.UUID(),
            existing_type=sa.String(100),
            postgresql_using="tenant_id::uuid",
        )
        _restore_policies(table, existing)
