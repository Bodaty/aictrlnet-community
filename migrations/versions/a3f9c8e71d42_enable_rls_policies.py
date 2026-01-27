"""enable_rls_policies

Revision ID: a3f9c8e71d42
Revises: 92efc1dbf4f0
Create Date: 2026-01-15

Enable PostgreSQL Row-Level Security (RLS) on all tenant-aware tables.
This provides database-level tenant isolation as a second layer of defense
beyond application-level filtering.

RLS ensures:
1. Queries automatically filter by current tenant
2. Inserts/updates can only affect current tenant's data
3. No cross-tenant data leakage even if app code has bugs
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text


# revision identifiers, used by Alembic.
revision = 'a3f9c8e71d42'
down_revision = '92efc1dbf4f0'
branch_labels = None
depends_on = None


# Tables with tenant_id that require RLS policies
# Grouped by domain for clarity
RLS_TABLES = [
    # Core workflow tables
    'tasks',
    'workflow_definitions',
    'workflow_instances',
    'workflow_steps',
    'workflow_executions',
    'workflow_templates',

    # Billing and subscription
    'subscriptions',
    'usage_tracking',
    'payment_methods',
    'billing_history',

    # Enforcement and limits
    'usage_metrics',
    'tenant_limit_overrides',
    'feature_trials',
    'upgrade_prompts',
    'license_cache',
    'billing_events',
    'usage_summaries',

    # Data quality
    'data_quality_assessments',
    'quality_rules',
    'quality_profiles',
    'data_lineage',
    'quality_slas',
    'quality_improvements',
    'quality_usage_tracking',

    # API and integrations
    'api_keys',
    'webhooks',
    'webhook_deliveries',
    'mcp_servers',
    'mcp_async_tasks',
    'adapter_configs',

    # Agents and automation
    'agents',
    'agent_runs',
    'agent_logs',

    # Conversations
    'conversations',
    'messages',

    # Knowledge system
    'knowledge_sources',
    'knowledge_embeddings',
    'knowledge_chunks',

    # Audit and notifications
    'audit_logs',
    'notifications',
]

# Tables with nullable tenant_id (special handling)
NULLABLE_TENANT_TABLES = [
    'users',  # System users may not have tenant
]


def upgrade() -> None:
    """Enable RLS and create tenant isolation policies."""
    connection = op.get_bind()
    inspector = inspect(connection)
    existing_tables = inspector.get_table_names()

    # Get existing policies to avoid duplicates
    result = connection.execute(text(
        "SELECT tablename, policyname FROM pg_policies WHERE schemaname = 'public'"
    ))
    existing_policies = {(row[0], row[1]) for row in result}

    # Process tables with required tenant_id
    for table in RLS_TABLES:
        if table not in existing_tables:
            continue

        # Check if table has tenant_id column
        columns = [col['name'] for col in inspector.get_columns(table)]
        if 'tenant_id' not in columns:
            continue

        # Enable RLS on table
        op.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))

        # Force RLS even for table owner (important for security)
        op.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))

        # Create tenant isolation policy (if not exists)
        policy_name = f"tenant_isolation_{table}"
        if (table, policy_name) not in existing_policies:
            op.execute(text(f"""
                CREATE POLICY {policy_name} ON {table}
                FOR ALL
                USING (tenant_id::text = current_setting('app.current_tenant_id', true))
                WITH CHECK (tenant_id::text = current_setting('app.current_tenant_id', true))
            """))

        # Create admin bypass policy (if not exists)
        admin_policy_name = f"admin_bypass_{table}"
        if (table, admin_policy_name) not in existing_policies:
            op.execute(text(f"""
                CREATE POLICY {admin_policy_name} ON {table}
                FOR ALL
                USING (current_setting('app.is_admin', true)::boolean = true)
            """))

    # Process tables with nullable tenant_id (different policy)
    for table in NULLABLE_TENANT_TABLES:
        if table not in existing_tables:
            continue

        columns = [col['name'] for col in inspector.get_columns(table)]
        if 'tenant_id' not in columns:
            continue

        # Enable RLS
        op.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        op.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))

        # Create policy allowing NULL tenant_id OR matching tenant
        policy_name = f"tenant_isolation_{table}"
        if (table, policy_name) not in existing_policies:
            op.execute(text(f"""
                CREATE POLICY {policy_name} ON {table}
                FOR ALL
                USING (
                    tenant_id IS NULL
                    OR tenant_id::text = current_setting('app.current_tenant_id', true)
                )
                WITH CHECK (
                    tenant_id IS NULL
                    OR tenant_id::text = current_setting('app.current_tenant_id', true)
                )
            """))

        # Admin bypass
        admin_policy_name = f"admin_bypass_{table}"
        if (table, admin_policy_name) not in existing_policies:
            op.execute(text(f"""
                CREATE POLICY {admin_policy_name} ON {table}
                FOR ALL
                USING (current_setting('app.is_admin', true)::boolean = true)
            """))

    # Handle tenants table specially - everyone can see their own tenant
    if 'tenants' in existing_tables:
        op.execute(text("ALTER TABLE tenants ENABLE ROW LEVEL SECURITY"))
        op.execute(text("ALTER TABLE tenants FORCE ROW LEVEL SECURITY"))

        if ('tenants', 'tenant_isolation_tenants') not in existing_policies:
            op.execute(text("""
                CREATE POLICY tenant_isolation_tenants ON tenants
                FOR ALL
                USING (id = current_setting('app.current_tenant_id', true))
                WITH CHECK (id = current_setting('app.current_tenant_id', true))
            """))

        if ('tenants', 'admin_bypass_tenants') not in existing_policies:
            op.execute(text("""
                CREATE POLICY admin_bypass_tenants ON tenants
                FOR ALL
                USING (current_setting('app.is_admin', true)::boolean = true)
            """))


def downgrade() -> None:
    """Remove RLS policies and disable RLS."""
    connection = op.get_bind()
    inspector = inspect(connection)
    existing_tables = inspector.get_table_names()

    # Get existing policies
    result = connection.execute(text(
        "SELECT tablename, policyname FROM pg_policies WHERE schemaname = 'public'"
    ))
    existing_policies = {(row[0], row[1]) for row in result}

    all_tables = RLS_TABLES + NULLABLE_TENANT_TABLES + ['tenants']

    for table in all_tables:
        if table not in existing_tables:
            continue

        # Drop policies
        policy_name = f"tenant_isolation_{table}"
        if (table, policy_name) in existing_policies:
            op.execute(text(f"DROP POLICY IF EXISTS {policy_name} ON {table}"))

        admin_policy_name = f"admin_bypass_{table}"
        if (table, admin_policy_name) in existing_policies:
            op.execute(text(f"DROP POLICY IF EXISTS {admin_policy_name} ON {table}"))

        # Disable RLS
        op.execute(text(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY"))
