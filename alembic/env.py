from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context
import os
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, parent_dir)

# Add src directory to Python path (this is how the app runs)
src_dir = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_dir)

# Import settings
from core.config import get_settings

# Import Base metadata - using same import path as models use
from models.base import Base

# Import all models to ensure they're registered with Base.metadata
# This is critical for autogenerate to work
from models.user import User
from models.community_complete import (
    Task, WorkflowDefinition, WorkflowInstance, WorkflowStep,
    MCPServer, MCPServerCapability, MCPTool, MCPInvocation,
    TaskMCP, MCPContextStorage, Adapter, BridgeConnection,
    BridgeSync, ResourcePoolConfig
)
from models.enforcement import (
    UsageMetric as EnforcementUsageMetric, TenantLimitOverride,
    FeatureTrial, UpgradePrompt, LicenseCache, BillingEvent, UsageSummary
)
from models.subscription import (
    SubscriptionPlan, Subscription, UsageTracking,
    PaymentMethod, BillingHistory
)
from models.data_quality import (
    DataQualityAssessment, QualityRule, QualityProfile,
    DataLineage, QualityDimension, QualityImprovement,
    QualitySLA, QualityUsageTracking
)
from models.iam import (
    IAMAgent, IAMMessage, IAMSession, IAMEventLog, IAMMetric
)
from models.api_key import APIKey, APIKeyLog
from models.webhook import Webhook, WebhookDelivery
from models.workflow_execution import (
    WorkflowExecution, NodeExecution, WorkflowCheckpoint,
    WorkflowTrigger, WorkflowSchedule
)
from models.usage_metrics import UsageMetric as BasicUsageMetric, UsageLimit
from models.platform_integration import (
    PlatformCredential, PlatformExecution, PlatformAdapter,
    PlatformHealth, PlatformWebhook
)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override sqlalchemy.url with value from settings
# Convert async URL to sync for Alembic
settings = get_settings()
sync_url = str(settings.DATABASE_URL).replace("postgresql+asyncpg://", "postgresql://")
config.set_main_option("sqlalchemy.url", sync_url)

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
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()