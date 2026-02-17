"""Alembic environment configuration."""

import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import settings and models
from src.core.config import get_settings

# Import Base from models.base which is what all models actually use
from models.base import Base

# Import ALL Community models to ensure all tables are detected
try:
    # Import each model file directly to register tables
    import models.user
    import models.adapter_config  # NEW: Adapter configurations
    import models.api_key
    import models.community
    import models.community_complete
    import models.conversation
    import models.data_quality
    import models.enforcement
    import models.iam
    import models.platform_integration
    import models.subscription
    import models.usage_metrics
    import models.webhook
    import models.workflow_execution
    import models.workflow_templates
    import models.knowledge  # NEW: Knowledge system for intelligent assistant
    import models.staged_file  # NEW: File upload staging for workflows
    import models.personal_agent  # Personal agent config + memories
    # Note: base.py is excluded as it contains the Base class itself
    print("✅ Successfully imported all Community models")
except Exception as e:
    print(f"❌ Error importing models: {e}")
    # Continue anyway - but migrations may be incomplete

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Get settings instance
settings = get_settings()

# Set SQLAlchemy URL from settings or environment variable
# SQLALCHEMY_DATABASE_URI bypasses Pydantic validation (needed for Cloud SQL Unix sockets)
database_url = os.getenv("SQLALCHEMY_DATABASE_URI")
if not database_url:
    database_url = str(settings.DATABASE_URL)
config.set_main_option("sqlalchemy.url", database_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

# Define Community tables that this edition manages
COMMUNITY_TABLES = {
    # Core tables
    'users',
    'tasks',
    'workflow_definitions',
    'workflow_instances',
    'workflow_steps',
    'workflow_executions',
    'node_executions',
    'workflow_checkpoints',
    'workflow_triggers',
    'workflow_schedules',
    'workflow_templates',
    'workflow_template_permissions',
    'workflow_template_usage',
    'workflow_template_reviews',
    # MCP tables
    'mcp_servers',
    'mcp_server_capabilities',
    'mcp_tools',
    'mcp_invocations',
    'mcp_task_associations',
    'mcp_context_storage',
    'mcp_async_tasks',  # NEW: MCP async tasks per SEP-1686
    # Adapter/Bridge tables
    'adapters',
    'adapter_configs',  # NEW: User adapter configurations
    'bridge_connections',
    'bridge_syncs',
    # Resource pool
    'resource_pool_configs',
    # API keys
    'api_keys',
    'api_key_logs',
    # Webhooks
    'webhooks',
    'webhook_deliveries',
    # Enforcement
    'usage_metrics',
    'tenant_limit_overrides',
    'feature_trials',
    'upgrade_prompts',
    'license_caches',
    'billing_events',
    'usage_summaries',
    # Subscription
    'subscription_plans',
    'subscriptions',
    'usage_tracking',
    'payment_methods',
    'billing_history',
    # IAM
    'iam_agents',
    'iam_messages',
    'iam_sessions',
    'iam_event_log',
    'iam_metrics',
    # Basic agents (Community Edition)
    'basic_agents',
    # Platform integration
    'platform_credentials',
    'platform_executions',
    'platform_adapters',
    'platform_health',
    'platform_webhooks',
    # Data quality
    'data_quality_checks',
    'data_quality_results',
    'data_quality_alerts',
    'data_quality_rules',
    # Usage metrics
    'usage_limits',
    # Conversation tables (NEW)
    'conversation_sessions',
    'conversation_messages',
    'conversation_actions',
    'conversation_intents',
    'conversation_patterns',
    # Knowledge system tables (NEW)
    'knowledge_items',
    'knowledge_index',
    'knowledge_queries',
    'system_manifests',
    'learned_patterns',
    # File staging
    'staged_files',
    # Personal agent
    'personal_agent_configs',
    'personal_agent_memories',
}

def include_object(object, name, type_, reflected, compare_to):
    """
    Filter to include only Community tables and prevent dropping Business/Enterprise tables.
    This is critical for the accretive model where higher editions extend lower ones.
    """
    if type_ == "table":
        # Only manage Community tables
        return name in COMMUNITY_TABLES
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,  # Enable filtering to prevent dropping Business/Enterprise tables
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    # Pre-create the version table if it doesn't exist to avoid issues
    from sqlalchemy import text
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS alembic_version (
            version_num VARCHAR(32) NOT NULL,
            CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
        )
    """))
    connection.commit()
    
    context.configure(
        connection=connection, 
        target_metadata=target_metadata,
        include_object=include_object,  # Enable filtering to prevent dropping Business/Enterprise tables
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()