"""IAM agents, messages, sessions, events and metrics belong to a tenant (G45).

Every existing row starts parked in the tenant ``iam-unowned``, which no user,
signup or session holds, so forced RLS hides it from everyone but the admin
bypass. Rows are parked, never deleted (Bobby, 5 Nov 2026); a purge is a later
follow-up once the parked rows have been looked at. Agents whose creator is
knowable move to that creator's tenant in the Business migration
``ia2ow3ne4rs5`` (the link lives on the Business ``enhanced_agents`` table), and
their children follow them.

Agent names become unique per owner in a tenant (system agents: per tenant)
instead of across the platform.

Revision ID: ia1te2na3nt4
Revises: dv2ex3pin4s5
"""

import sqlalchemy as sa
from alembic import op

revision = "ia1te2na3nt4"
down_revision = "dv2ex3pin4s5"
branch_labels = None
depends_on = None

UNOWNED = "iam-unowned"
TABLES = ("iam_agents", "iam_messages", "iam_sessions", "iam_event_logs", "iam_metrics")
OWNER_NAME_UNIQUE = "uq_iam_agents_owner_name"
SYSTEM_NAME_UNIQUE = "uq_iam_agents_system_name"


def _global_name_unique(connection):
    for uc in sa.inspect(connection).get_unique_constraints("iam_agents"):
        if uc["column_names"] == ["name"]:
            return uc["name"]
    return None


def upgrade():
    connection = op.get_bind()
    columns = {t: {c["name"] for c in sa.inspect(connection).get_columns(t)} for t in TABLES}

    for table in TABLES:
        if "tenant_id" not in columns[table]:
            op.add_column(table, sa.Column(
                "tenant_id", sa.String(100), nullable=False, server_default=UNOWNED))
            op.alter_column(table, "tenant_id", server_default=None)
            op.create_index(f"ix_{table}_tenant_id", table, ["tenant_id"])
    if "owner_user_id" not in columns["iam_agents"]:
        op.add_column("iam_agents", sa.Column("owner_user_id", sa.String(36), nullable=True))
        op.create_index("ix_iam_agents_owner_user_id", "iam_agents", ["owner_user_id"])

    old_unique = _global_name_unique(connection)
    if old_unique:
        op.drop_constraint(old_unique, "iam_agents", type_="unique")
    indexes = {ix["name"] for ix in sa.inspect(connection).get_indexes("iam_agents")}
    if OWNER_NAME_UNIQUE not in indexes:
        op.create_index(OWNER_NAME_UNIQUE, "iam_agents", ["tenant_id", "owner_user_id", "name"],
                        unique=True, postgresql_where=sa.text("owner_user_id IS NOT NULL"))
    if SYSTEM_NAME_UNIQUE not in indexes:
        op.create_index(SYSTEM_NAME_UNIQUE, "iam_agents", ["tenant_id", "name"],
                        unique=True, postgresql_where=sa.text("owner_user_id IS NULL"))

    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{table} ON {table}")
        op.execute(f"""
            CREATE POLICY tenant_isolation_{table} ON {table}
            FOR ALL
            USING (tenant_id::text = current_setting('app.current_tenant_id', true))
            WITH CHECK (tenant_id::text = current_setting('app.current_tenant_id', true))
        """)
        op.execute(f"DROP POLICY IF EXISTS admin_bypass_{table} ON {table}")
        op.execute(f"""
            CREATE POLICY admin_bypass_{table} ON {table}
            FOR ALL
            USING (coalesce(nullif(current_setting('app.is_admin', true), ''), 'false')::boolean)
        """)

    op.execute("SET LOCAL app.is_admin = 'true'")
    counts = {t: connection.execute(sa.text(f"SELECT count(*) FROM {t}")).scalar() for t in TABLES}
    print(f"G45: parked in {UNOWNED} pending owner assignment: {counts}")


def downgrade():
    connection = op.get_bind()
    for table in TABLES:
        op.execute(f"DROP POLICY IF EXISTS admin_bypass_{table} ON {table}")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{table} ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    indexes = {ix["name"] for ix in sa.inspect(connection).get_indexes("iam_agents")}
    for name in (OWNER_NAME_UNIQUE, SYSTEM_NAME_UNIQUE):
        if name in indexes:
            op.drop_index(name, table_name="iam_agents")
    duplicated = connection.execute(sa.text(
        "SELECT count(*) > count(DISTINCT name) FROM iam_agents")).scalar()
    if duplicated:
        print("G45 downgrade: agent names repeat across tenants; the platform-wide "
              "unique constraint on iam_agents.name is not restored")
    elif not _global_name_unique(connection):
        op.create_unique_constraint("iam_agents_name_key", "iam_agents", ["name"])

    op.drop_index("ix_iam_agents_owner_user_id", table_name="iam_agents")
    op.drop_column("iam_agents", "owner_user_id")
    for table in TABLES:
        op.drop_index(f"ix_{table}_tenant_id", table_name=table)
        op.drop_column(table, "tenant_id")
