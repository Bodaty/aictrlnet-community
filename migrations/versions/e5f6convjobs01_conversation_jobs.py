"""conversation_jobs: long-running conversation tools as background jobs (T5).

Tenant-isolated by RLS like the other tenant tables. The admin bypass uses the
form that tolerates an empty `app.is_admin` (a bare `::boolean` cast errors on
'' — see reference_rls_admin_guc_poisoning).

Revision ID: e5f6convjobs01
Revises: d3e4mergefix01
Create Date: 2026-09-30
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "e5f6convjobs01"
down_revision = "d3e4mergefix01"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    if "conversation_jobs" in sa.inspect(connection).get_table_names():
        return
    op.create_table(
        "conversation_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(36), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("conversation_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("progress", sa.Text(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("owner_token", sa.String(64), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=False),
        sa.Column("harvested_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_conversation_jobs_tenant_id", "conversation_jobs", ["tenant_id"])
    op.create_index("ix_conversation_jobs_session_id", "conversation_jobs", ["session_id"])
    op.create_index("ix_conversation_jobs_status_lease", "conversation_jobs", ["status", "lease_expires_at"])
    op.execute("ALTER TABLE conversation_jobs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE conversation_jobs FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY tenant_isolation_conversation_jobs ON conversation_jobs
        FOR ALL
        USING (tenant_id::text = current_setting('app.current_tenant_id', true))
        WITH CHECK (tenant_id::text = current_setting('app.current_tenant_id', true))
    """)
    op.execute("""
        CREATE POLICY admin_bypass_conversation_jobs ON conversation_jobs
        FOR ALL
        USING (coalesce(nullif(current_setting('app.is_admin', true), ''), 'false')::boolean)
    """)


def downgrade():
    op.drop_table("conversation_jobs")
