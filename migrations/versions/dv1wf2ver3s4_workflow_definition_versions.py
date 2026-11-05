"""Workflow definition version history, with row-level security, backfilled.

Revision ID: dv1wf2ver3s4
Revises: dq7tenantstr01
"""

import sqlalchemy as sa
from alembic import op

revision = "dv1wf2ver3s4"
down_revision = "dq7tenantstr01"
branch_labels = None
depends_on = None

TABLE = "workflow_definition_versions"


def upgrade():
    if TABLE not in set(sa.inspect(op.get_bind()).get_table_names()):
        op.create_table(
            TABLE,
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("tenant_id", sa.String(36), nullable=False, server_default="default-tenant"),
            sa.Column("workflow_id", sa.String(36), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("definition", sa.JSON(), nullable=False),
            sa.Column("template_name", sa.String(256), nullable=True),
            sa.Column("template_version", sa.String(32), nullable=True),
            sa.Column("reason", sa.String(16), nullable=False),
            sa.Column("created_by", sa.String(64), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
            sa.UniqueConstraint("workflow_id", "version", name="uq_workflow_definition_version"),
        )
        op.create_index(f"ix_{TABLE}_tenant_id", TABLE, ["tenant_id"])
        op.create_index(f"ix_{TABLE}_workflow_id", TABLE, ["workflow_id"])

    op.execute(f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY")
    op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{TABLE} ON {TABLE}")
    op.execute(f"""
        CREATE POLICY tenant_isolation_{TABLE} ON {TABLE}
        FOR ALL
        USING (tenant_id::text = current_setting('app.current_tenant_id', true))
        WITH CHECK (tenant_id::text = current_setting('app.current_tenant_id', true))
    """)
    op.execute(f"DROP POLICY IF EXISTS admin_bypass_{TABLE} ON {TABLE}")
    op.execute(f"""
        CREATE POLICY admin_bypass_{TABLE} ON {TABLE}
        FOR ALL
        USING (coalesce(nullif(current_setting('app.is_admin', true), ''), 'false')::boolean)
    """)

    op.execute("SELECT set_config('app.is_admin', 'true', true)")
    op.execute(f"""
        INSERT INTO {TABLE}
            (id, tenant_id, workflow_id, version, definition, template_name, template_version, reason, created_by)
        SELECT gen_random_uuid()::text, d.tenant_id, d.id, COALESCE(d.version, 1), d.definition,
               LEFT(d.definition::jsonb ->> 'name', 256), LEFT(d.definition::jsonb ->> 'version', 32), 'backfill', NULL
        FROM workflow_definitions d
        WHERE NOT EXISTS (SELECT 1 FROM {TABLE} v WHERE v.workflow_id = d.id)
    """)


def downgrade():
    op.execute(f"DROP POLICY IF EXISTS admin_bypass_{TABLE} ON {TABLE}")
    op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{TABLE} ON {TABLE}")
    op.drop_table(TABLE)
