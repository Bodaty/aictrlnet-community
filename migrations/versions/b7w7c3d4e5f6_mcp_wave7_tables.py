"""mcp wave 7 tables — workflow_versions, slas, roles, file_versions, template_versions, credentials_extended

Track C migration for Wave 7. Six new tables (or columns) that back
the tools in Track B2 + B3. Each section is independent and uses
inspector-pattern existence checks — safe to re-run.

Tables created:
- mcp_workflow_versions — snapshot history per workflow (B2.3)
- mcp_slas — SLA definitions + targets (B2.2)
- mcp_sla_violations — breach records (B2.2)
- mcp_roles + mcp_role_permissions + mcp_user_roles — RBAC (B2.4)
- mcp_file_versions — staged-file version history (B3.1)
- mcp_template_versions — workflow-template version history (B3.4)

Columns added:
- credentials: backend_type, last_validated_at, rotation_count — if the
  `credentials` or `staged_credentials` table exists. These are for
  B1.2's rotate/validate tools (otherwise feature_pending).

Revision ID: b7w7c3d4e5f6
Revises: a3b4c5d6e7f8
Create Date: 2026-04-23 16:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "b7w7c3d4e5f6"
down_revision = "a3b4c5d6e7f8"
branch_labels = None
depends_on = None


def _has_table(conn, name: str) -> bool:
    return name in sa.inspect(conn).get_table_names()


def _has_column(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    if table not in inspector.get_table_names():
        return False
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    conn = op.get_bind()

    # ------------------------------------------------------------
    # B2.3 — workflow_versions
    # ------------------------------------------------------------
    if not _has_table(conn, "mcp_workflow_versions"):
        op.create_table(
            "mcp_workflow_versions",
            sa.Column("id", sa.String(), primary_key=True,
                      server_default=sa.text("gen_random_uuid()::text")),
            sa.Column("workflow_id", sa.String(), nullable=False, index=True),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("definition_json", sa.Text(), nullable=False),
            sa.Column("tenant_id", sa.String(), nullable=False, index=True),
            sa.Column("created_by", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      nullable=False, server_default=sa.text("now()")),
            sa.Column("change_summary", sa.String(length=500), nullable=True),
            sa.UniqueConstraint("workflow_id", "version",
                                name="uq_mcp_workflow_versions_wid_v"),
        )

    # ------------------------------------------------------------
    # B2.2 — SLAs + violations
    # ------------------------------------------------------------
    if not _has_table(conn, "mcp_slas"):
        op.create_table(
            "mcp_slas",
            sa.Column("id", sa.String(), primary_key=True,
                      server_default=sa.text("gen_random_uuid()::text")),
            sa.Column("tenant_id", sa.String(), nullable=False, index=True),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("resource_type", sa.String(length=64), nullable=False),  # workflow/agent/tool
            sa.Column("resource_id", sa.String(), nullable=True),
            sa.Column("metric", sa.String(length=64), nullable=False),  # latency_p95/success_rate/...
            sa.Column("target_value", sa.Float(), nullable=False),
            sa.Column("window_seconds", sa.Integer(), nullable=False,
                      server_default="3600"),
            sa.Column("severity", sa.String(length=16), nullable=False,
                      server_default="medium"),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("created_by", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      nullable=False, server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(timezone=True),
                      nullable=False, server_default=sa.text("now()")),
        )

    if not _has_table(conn, "mcp_sla_violations"):
        op.create_table(
            "mcp_sla_violations",
            sa.Column("id", sa.String(), primary_key=True,
                      server_default=sa.text("gen_random_uuid()::text")),
            sa.Column("sla_id", sa.String(), nullable=False, index=True),
            sa.Column("tenant_id", sa.String(), nullable=False, index=True),
            sa.Column("observed_value", sa.Float(), nullable=False),
            sa.Column("target_value", sa.Float(), nullable=False),
            sa.Column("severity", sa.String(length=16), nullable=False),
            sa.Column("resolved", sa.Boolean(), nullable=False, server_default="false"),
            sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("detected_at", sa.DateTime(timezone=True),
                      nullable=False, server_default=sa.text("now()")),
            sa.Column("meta_data", sa.JSON(), nullable=True),
        )

    # ------------------------------------------------------------
    # B2.4 — RBAC (if Business edition doesn't already have these)
    # ------------------------------------------------------------
    if not _has_table(conn, "mcp_roles"):
        op.create_table(
            "mcp_roles",
            sa.Column("id", sa.String(), primary_key=True,
                      server_default=sa.text("gen_random_uuid()::text")),
            sa.Column("tenant_id", sa.String(), nullable=False, index=True),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("is_system", sa.Boolean(), nullable=False, server_default="false"),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      nullable=False, server_default=sa.text("now()")),
            sa.UniqueConstraint("tenant_id", "name", name="uq_mcp_roles_tenant_name"),
        )

    if not _has_table(conn, "mcp_role_permissions"):
        op.create_table(
            "mcp_role_permissions",
            sa.Column("id", sa.String(), primary_key=True,
                      server_default=sa.text("gen_random_uuid()::text")),
            sa.Column("role_id", sa.String(), nullable=False, index=True),
            sa.Column("resource", sa.String(length=64), nullable=False),
            sa.Column("action", sa.String(length=64), nullable=False),
            sa.Column("scope", sa.String(length=64), nullable=True),
            sa.UniqueConstraint("role_id", "resource", "action", "scope",
                                name="uq_mcp_role_perm"),
        )

    if not _has_table(conn, "mcp_user_roles"):
        op.create_table(
            "mcp_user_roles",
            sa.Column("id", sa.String(), primary_key=True,
                      server_default=sa.text("gen_random_uuid()::text")),
            sa.Column("user_id", sa.String(), nullable=False, index=True),
            sa.Column("role_id", sa.String(), nullable=False, index=True),
            sa.Column("tenant_id", sa.String(), nullable=False, index=True),
            sa.Column("granted_by", sa.String(), nullable=True),
            sa.Column("granted_at", sa.DateTime(timezone=True),
                      nullable=False, server_default=sa.text("now()")),
            sa.UniqueConstraint("user_id", "role_id", "tenant_id",
                                name="uq_mcp_user_role"),
        )

    # ------------------------------------------------------------
    # B3.1 — file_versions
    # ------------------------------------------------------------
    if not _has_table(conn, "mcp_file_versions"):
        op.create_table(
            "mcp_file_versions",
            sa.Column("id", sa.String(), primary_key=True,
                      server_default=sa.text("gen_random_uuid()::text")),
            sa.Column("file_id", sa.String(), nullable=False, index=True),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("storage_path", sa.String(length=1024), nullable=False),
            sa.Column("file_size", sa.BigInteger(), nullable=False),
            sa.Column("content_type", sa.String(length=128), nullable=True),
            sa.Column("checksum_sha256", sa.String(length=64), nullable=True),
            sa.Column("created_by", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      nullable=False, server_default=sa.text("now()")),
            sa.UniqueConstraint("file_id", "version", name="uq_mcp_file_versions"),
        )

    # ------------------------------------------------------------
    # B3.4 — template_versions
    # ------------------------------------------------------------
    if not _has_table(conn, "mcp_template_versions"):
        op.create_table(
            "mcp_template_versions",
            sa.Column("id", sa.String(), primary_key=True,
                      server_default=sa.text("gen_random_uuid()::text")),
            sa.Column("template_id", sa.String(), nullable=False, index=True),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("definition_json", sa.Text(), nullable=False),
            sa.Column("parameters_schema", sa.JSON(), nullable=True),
            sa.Column("change_summary", sa.String(length=500), nullable=True),
            sa.Column("created_by", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      nullable=False, server_default=sa.text("now()")),
            sa.UniqueConstraint("template_id", "version",
                                name="uq_mcp_template_versions"),
        )

    # ------------------------------------------------------------
    # B1.2 — extended credential metadata (if table present)
    # ------------------------------------------------------------
    if _has_table(conn, "credentials"):
        for col, spec in (
            ("backend_type", sa.Column("backend_type", sa.String(length=32), nullable=True)),
            ("last_validated_at", sa.Column("last_validated_at", sa.DateTime(timezone=True), nullable=True)),
            ("rotation_count", sa.Column("rotation_count", sa.Integer(), nullable=False, server_default="0")),
        ):
            if not _has_column(conn, "credentials", col):
                op.add_column("credentials", spec)


def downgrade() -> None:
    conn = op.get_bind()

    # Drop columns first
    if _has_column(conn, "credentials", "rotation_count"):
        op.drop_column("credentials", "rotation_count")
    if _has_column(conn, "credentials", "last_validated_at"):
        op.drop_column("credentials", "last_validated_at")
    if _has_column(conn, "credentials", "backend_type"):
        op.drop_column("credentials", "backend_type")

    # Drop tables in reverse dependency order
    for t in (
        "mcp_template_versions",
        "mcp_file_versions",
        "mcp_user_roles",
        "mcp_role_permissions",
        "mcp_roles",
        "mcp_sla_violations",
        "mcp_slas",
        "mcp_workflow_versions",
    ):
        if _has_table(conn, t):
            op.drop_table(t)
