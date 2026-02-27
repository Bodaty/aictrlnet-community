#!/usr/bin/env python3
"""Seed data script for Community Edition."""

import asyncio
import logging
from datetime import datetime
from typing import Optional
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.security import get_password_hash

# Import SQLAlchemy components
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text

# Import models directly to avoid circular import issues
# IMPORTANT: Import UserAdapterConfig BEFORE User to ensure it's available for the relationship
from src.models.adapter_config import UserAdapterConfig  # Must be imported before User
from src.models.user import User
from src.models.community_complete import Adapter, ResourcePoolConfig
from src.models.api_key import APIKey
from src.models.subscription import SubscriptionPlan
from src.models.usage_metrics import UsageLimit
from src.models.data_quality import QualityDimensionModel

# Database URL - use environment variables for Cloud Run compatibility
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "aictrlnet")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# Cloud Run uses Unix socket path, local uses hostname
if POSTGRES_SERVER.startswith("/cloudsql/"):
    # Cloud SQL Unix socket connection
    DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@/{POSTGRES_DB}?host={POSTGRES_SERVER}"
else:
    # Standard TCP connection (local Docker)
    DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logger.info(f"🌍 Running seed script for environment: {ENVIRONMENT}")


async def seed_users(session):
    """Create default users."""
    users_data = []
    
    # Default tier preferences for all users
    default_tier_preferences = {
        "preferredFastModel": "llama3.2:1b",
        "preferredBalancedModel": "llama3.2:3b",
        "preferredQualityModel": "llama3.1:8b-instruct-q4_K_M"
    }

    # Always seed full test data for F&F beta (removed environment checks)
    users_data = [
        {
            "id": str(uuid.uuid4()),
            "email": "admin@aictrlnet.com",
            "username": "admin",
            "hashed_password": get_password_hash("admin123"),
            "full_name": "System Administrator",
            "is_active": True,
            "is_superuser": True,
            "tenant_id": "default-tenant",
            "edition": "community",
            "preferences": default_tier_preferences
        },
        {
            "id": str(uuid.uuid4()),
            "email": "dev@aictrlnet.com",
            "username": "developer",
            "hashed_password": get_password_hash("testpass123"),
            "full_name": "Developer User",
            "is_active": True,
            "is_superuser": True,
            "tenant_id": "default-tenant",
            "edition": "enterprise",
            "preferences": default_tier_preferences
        },
        {
            "id": str(uuid.uuid4()),
            "email": "test@example.com",
            "username": "testuser",
            "hashed_password": get_password_hash("testpass123"),
            "full_name": "Test User",
            "is_active": True,
            "is_superuser": False,
            "tenant_id": "default-tenant",
            "edition": "community",
            "preferences": default_tier_preferences
        }
    ]
    
    created_count = 0
    for user_data in users_data:
        existing = await session.get(User, user_data["id"])
        if not existing:
            # Check if email exists
            result = await session.execute(
                select(User).where(User.email == user_data["email"])
            )
            if not result.scalar_one_or_none():
                user = User(**user_data)
                session.add(user)
                created_count += 1
                logger.info(f"Created user: {user_data['email']}")
    
    if created_count > 0:
        await session.commit()
    
    return created_count


# NOTE: Adapter seeding is handled by scripts/seed_adapters.py
# which is called via subprocess below in main()
# DO NOT duplicate adapter seeding here!

async def seed_adapters_REMOVED(session):
    """Create adapters ONLY for implementations that actually exist."""
    adapters_data = [
        # ============= AI AGENT ADAPTERS (have implementation) =============
        {
            "id": str(uuid.uuid4()),
            "name": "LangChain",
            "category": "ai_agent",
            "description": "LangChain framework for building AI agents",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "framework": "langchain",
                "supported_models": ["openai", "anthropic", "ollama"],
                "features": ["chains", "agents", "memory", "tools"]
            },
            "config_schema": {"type": "object"},
            "capabilities": ["chat", "chains", "agents"],
            "tags": ["langchain", "agents", "community"],
            "available": True,
            "installed": True,
            "install_count": 0
        },
        
        # ============= AI ADAPTERS (have implementations) =============
        {
            "id": str(uuid.uuid4()),
            "name": "OpenAI",
            "category": "ai",
            "description": "OpenAI GPT models integration",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "provider": "openai",
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "api_version": "v1"
            },
            "config_schema": {"type": "object"},
            "capabilities": ["chat", "completion", "embeddings"],
            "tags": ["openai", "gpt", "ai"],
            "available": True,
            "installed": True,
            "install_count": 0
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Claude",
            "category": "ai",
            "description": "Anthropic Claude AI integration",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "provider": "anthropic",
                "models": ["claude-3-opus", "claude-3-sonnet"],
                "api_version": "v1"
            },
            "config_schema": {"type": "object"},
            "capabilities": ["chat", "analysis", "reasoning"],
            "tags": ["claude", "anthropic", "ai"],
            "available": True,
            "installed": True,
            "install_count": 0
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Ollama",
            "category": "ai",
            "description": "Local AI models via Ollama",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "provider": "ollama",
                "local": True,
                "models": ["llama2", "mistral", "codellama"]
            },
            "config_schema": {"type": "object"},
            "capabilities": ["chat", "completion"],
            "tags": ["ollama", "local", "privacy"],
            "available": True,
            "installed": False,
            "install_count": 0
        },
        {
            "id": str(uuid.uuid4()),
            "name": "HuggingFace",
            "category": "ai",
            "description": "HuggingFace models and inference",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "provider": "huggingface",
                "api_type": "inference"
            },
            "config_schema": {"type": "object"},
            "capabilities": ["inference", "embeddings"],
            "tags": ["huggingface", "transformers", "ai"],
            "available": True,
            "installed": False,
            "install_count": 0
        },
        
        # ============= COMMUNICATION ADAPTERS (have implementations) =============
        {
            "id": str(uuid.uuid4()),
            "name": "Email",
            "category": "communication",
            "description": "Email notifications via SMTP",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "protocols": ["smtp", "smtps"],
                "features": ["html", "attachments"]
            },
            "config_schema": {"type": "object"},
            "capabilities": ["send", "bulk"],
            "tags": ["email", "smtp", "notifications"],
            "available": True,
            "installed": True,
            "install_count": 0
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Slack",
            "category": "communication",
            "description": "Slack workspace integration",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "api_version": "v2",
                "features": ["messages", "channels"]
            },
            "config_schema": {"type": "object"},
            "capabilities": ["send", "receive"],
            "tags": ["slack", "chat", "team"],
            "available": True,
            "installed": False,
            "install_count": 0
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Discord",
            "category": "communication",
            "description": "Discord server integration",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "api_version": "v10",
                "features": ["messages", "embeds"]
            },
            "config_schema": {"type": "object"},
            "capabilities": ["send", "webhooks"],
            "tags": ["discord", "chat"],
            "available": True,
            "installed": False,
            "install_count": 0
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Webhook",
            "category": "communication",
            "description": "Generic webhook for HTTP notifications",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "methods": ["POST", "GET", "PUT"],
                "features": ["retry", "auth"]
            },
            "config_schema": {"type": "object"},
            "capabilities": ["send", "batch"],
            "tags": ["webhook", "http", "api"],
            "available": True,
            "installed": True,
            "install_count": 0
        },
        
        # ============= HUMAN ADAPTERS (have implementations) =============
        {
            "id": str(uuid.uuid4()),
            "name": "Upwork",
            "category": "human",
            "description": "Upwork freelancer platform",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "platform": "upwork",
                "features": ["jobs", "freelancers"]
            },
            "config_schema": {"type": "object"},
            "capabilities": ["post", "search"],
            "tags": ["upwork", "freelance", "human"],
            "available": True,
            "installed": False,
            "install_count": 0
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Fiverr",
            "category": "human",
            "description": "Fiverr gig marketplace",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "platform": "fiverr",
                "features": ["gigs", "orders"]
            },
            "config_schema": {"type": "object"},
            "capabilities": ["search", "order"],
            "tags": ["fiverr", "gigs", "human"],
            "available": True,
            "installed": False,
            "install_count": 0
        },
        
        # ============= PAYMENT ADAPTERS (have implementation) =============
        {
            "id": str(uuid.uuid4()),
            "name": "Stripe",
            "category": "payment",
            "description": "Stripe payment processing",
            "version": "1.0.0",
            "min_edition": "community",
            "enabled": True,
            "metadata": {
                "provider": "stripe",
                "features": ["payments", "subscriptions"],
                "api_version": "2023-10-16"
            },
            "config_schema": {"type": "object"},
            "capabilities": ["charge", "subscribe"],
            "tags": ["stripe", "payment"],
            "available": True,
            "installed": False,
            "install_count": 0
        }
        
        # NOTE: Removed PostgreSQL, CSV, Logger, GitHub, Jira adapters 
        # because they have NO IMPLEMENTATIONS in the codebase!
    ]
    
    created_count = 0
    for adapter_data in adapters_data:
        # Check if adapter with same name exists
        result = await session.execute(
            select(Adapter).where(Adapter.name == adapter_data["name"])
        )
        if not result.scalar_one_or_none():
            adapter = Adapter(**adapter_data)
            session.add(adapter)
            created_count += 1
            logger.info(f"Created adapter: {adapter_data['name']}")
    
    if created_count > 0:
        await session.commit()
    
    return created_count


async def seed_workflow_templates(session):
    """Create default workflow templates for Community."""
    # Templates are loaded from JSON files at startup via initialize_system_templates()
    # as per WORKFLOW_TEMPLATE_SYSTEM_SPEC.md architecture
    # 
    # The actual templates are loaded from:
    # - /aictrlnet-fastapi/workflow-templates/system/ (1 Community template)
    # - /aictrlnet-fastapi-business/workflow-templates/system/ (176 Business templates)
    # - /aictrlnet-fastapi-enterprise/workflow-templates/system/ (6 Enterprise templates)
    # Total: 183 templates via accretion model
    # 
    # This seed function now loads templates during seeding to avoid race condition
    # where app startup tries to load before migrations have created tables
    
    try:
        from services.workflow_template_service import WorkflowTemplateService
        
        template_service = WorkflowTemplateService()
        count = await template_service.initialize_system_templates(session)
        logger.info(f"✅ Initialized {count} Community workflow templates from JSON files")
        return count
    except Exception as e:
        logger.error(f"Failed to initialize workflow templates: {e}")
        # Don't fail the entire seeding if templates can't load
        # They'll be loaded on app startup as fallback
        logger.warning("Templates will be loaded on application startup instead")
        return 0


async def seed_api_keys(session):
    """Create default API keys for testing."""
    # Model already imported at top: api_key import APIKey
    # Model already imported at top: user import User
    # Get admin user
    from sqlalchemy import select
    result = await session.execute(
        select(User).where(User.email == "admin@aictrlnet.com")
    )
    admin_user = result.scalar_one_or_none()
    
    if not admin_user:
        logger.warning("Admin user not found, skipping API key creation")
        return 0
    
    api_keys_data = [
        {
            "id": str(uuid.uuid4()),
            "name": "Default Development Key",
            "description": "Development and testing API key",
            "key_prefix": "dev",
            "key_suffix": "test",
            "key_hash": get_password_hash("test-api-key-hash"),  # In production, use proper key generation
            "key_salt": get_password_hash("salt"),  # In production, use proper salt
            "user_id": admin_user.id,
            "is_active": True,
            "scopes": ["read", "write", "admin"],
            "allowed_ips": [],  # Allow all IPs
            "expires_at": None
        }
    ]
    
    created_count = 0
    for key_data in api_keys_data:
        # Check if key with same name exists
        result = await session.execute(
            select(APIKey).where(APIKey.name == key_data["name"])
        )
        if not result.scalar_one_or_none():
            api_key = APIKey(**key_data)
            session.add(api_key)
            created_count += 1
            logger.info(f"Created API key: {key_data['name']}")
    
    if created_count > 0:
        await session.commit()
    
    return created_count


async def seed_subscription_plans(session):
    """Create subscription plans based on PRICING_AND_LICENSING_STRATEGY_2025.md."""
    # Model already imported at top: subscription import SubscriptionPlan
    plans_data = [
        {
            "id": "community-free",
            "name": "community",
            "display_name": "Community Edition",
            "description": "Open source edition for individuals and small projects",
            "price_monthly": 0.0,
            "price_annual": 0.0,
            "currency": "USD",
            "features": {
                "max_users": 1,
                "max_workflows": 10,
                "max_adapters": 5,
                "api_calls_per_month": 10000,
                "storage_gb": 1,
                "support_level": "community",
                "ml_features": False,
                "ai_governance": False,
                "advanced_analytics": False,
                "custom_integrations": False,
                "mfa_enabled": False,
                "sso_enabled": False
            },
            "limits": {
                "api_requests_per_hour": 100,
                "concurrent_workflows": 2,
                "executions_per_month": 1000,
                "analytics_retention_days": 7
            },
            "stripe_price_id_monthly": None,
            "stripe_price_id_annual": None,
            "is_active": True
        },
        {
            "id": "community-cloud",
            "name": "community_cloud",
            "display_name": "Community Cloud",
            "description": "Hosted Community Edition with HitLai UI",
            "price_monthly": 49.0,
            "price_annual": 490.0,  # ~17% discount
            "currency": "USD",
            "features": {
                "max_users": 1,
                "max_workflows": 10,
                "max_adapters": 5,
                "api_calls_per_month": 10000,
                "storage_gb": 1,
                "support_level": "community",
                "ml_features": False,
                "ai_governance": False,
                "advanced_analytics": False,
                "custom_integrations": False,
                "mfa_enabled": False,
                "sso_enabled": False,
                "hitlai_ui": True  # Key differentiator
            },
            "limits": {
                "api_requests_per_hour": 100,
                "concurrent_workflows": 2,
                "executions_per_month": 1000,
                "analytics_retention_days": 7
            },
            "stripe_price_id_monthly": "price_community_cloud_monthly",  # Will be set when Stripe configured
            "stripe_price_id_annual": "price_community_cloud_annual",
            "is_active": True
        }
    ]
    
    created_count = 0
    for plan_data in plans_data:
        # Check if plan exists
        result = await session.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.id == plan_data["id"])
        )
        if not result.scalar_one_or_none():
            plan = SubscriptionPlan(**plan_data)
            session.add(plan)
            created_count += 1
            logger.info(f"Created subscription plan: {plan_data['display_name']}")
    
    if created_count > 0:
        await session.commit()
    
    return created_count


async def seed_usage_limits(session):
    """Create default usage limits for Community Edition."""
    # Model already imported at top: usage_metrics import UsageLimit
    # Based on the UsageLimit model structure and PRICING_AND_LICENSING_STRATEGY_2025.md
    limits_data = {
        "id": str(uuid.uuid4()),
        "edition": "community",
        "max_workflows": 10,
        "max_adapters": 5,
        "max_users": 1,
        "max_api_calls_month": 10000,
        "max_storage_bytes": 1073741824  # 1GB in bytes
    }
    
    created_count = 0
    # Check if limit exists for this edition
    from sqlalchemy import select
    result = await session.execute(
        select(UsageLimit).where(UsageLimit.edition == "community")
    )
    if not result.scalar_one_or_none():
        usage_limit = UsageLimit(**limits_data)
        session.add(usage_limit)
        created_count = 1
        logger.info(f"Created usage limits for Community edition")
        logger.info(f"  - Workflows: {limits_data['max_workflows']}")
        logger.info(f"  - Adapters: {limits_data['max_adapters']}")
        logger.info(f"  - Users: {limits_data['max_users']}")
        logger.info(f"  - API calls/month: {limits_data['max_api_calls_month']}")
        logger.info(f"  - Storage: {limits_data['max_storage_bytes'] / 1073741824}GB")
        await session.commit()
    
    return created_count


async def seed_quality_dimensions(session):
    """Create quality dimension definitions for Community Edition (only 2 dimensions)."""
    # Model already imported at top: data_quality import QualityDimensionModel
    # Based on ISO_25012_DATA_QUALITY_IMPLEMENTATION_SPEC.md
    # Community gets only accuracy and completeness
    dimensions_data = [
        {
            "id": str(uuid.uuid4()),
            "name": "accuracy",
            "category": "inherent",
            "description": "The degree to which data correctly represents the real-world entity or event being described",
            "edition_required": "community",
            "measurement_method": "Compare data values against known correct values or trusted sources. Calculate accuracy score as percentage of correct values."
        },
        {
            "id": str(uuid.uuid4()),
            "name": "completeness",
            "category": "inherent", 
            "description": "The degree to which all required data is present and no mandatory fields are missing",
            "edition_required": "community",
            "measurement_method": "Check for null/empty values in required fields. Calculate completeness score as percentage of populated required fields."
        }
    ]
    
    created_count = 0
    for dim_data in dimensions_data:
        # Check if dimension exists
        result = await session.execute(
            select(QualityDimensionModel).where(
                QualityDimensionModel.name == dim_data["name"]
            )
        )
        if not result.scalar_one_or_none():
            dimension = QualityDimensionModel(**dim_data)
            session.add(dimension)
            created_count += 1
            logger.info(f"Created quality dimension: {dim_data['name']}")
    
    if created_count > 0:
        await session.commit()
    
    return created_count


async def seed_resource_pool_configs(session):
    """Create default resource pool configurations for Community Edition."""
    # Model already imported at top: community_complete import ResourcePoolConfig
    pools_data = [
        {
            "id": str(uuid.uuid4()),
            "name": "default_compute_pool",
            "resource_type": "compute",
            "min_size": 1,
            "max_size": 2,  # Limited for Community
            "acquire_timeout": 30.0,
            "idle_timeout": 300.0,
            "max_lifetime": 3600.0,
            "health_check_interval": 60.0,
            "scale_up_threshold": 0.8,
            "scale_down_threshold": 0.2,
            "enabled": True,
            "config_metadata": {
                "cpu_limit": "1",
                "memory_limit": "512Mi",
                "edition": "community",
                "priority": 100,
                "tags": ["community", "default"]
            }
        },
        {
            "id": str(uuid.uuid4()),
            "name": "default_storage_pool",
            "resource_type": "storage",
            "min_size": 1,
            "max_size": 1,  # Fixed for Community
            "acquire_timeout": 10.0,
            "idle_timeout": 600.0,
            "max_lifetime": 86400.0,
            "health_check_interval": 300.0,
            "scale_up_threshold": 0.9,
            "scale_down_threshold": 0.1,
            "enabled": True,
            "config_metadata": {
                "storage_class": "standard",
                "max_size_gb": 1,
                "edition": "community",
                "priority": 100,
                "tags": ["community", "default"]
            }
        }
    ]
    
    created_count = 0
    for pool_data in pools_data:
        # Check if pool exists
        result = await session.execute(
            select(ResourcePoolConfig).where(
                ResourcePoolConfig.name == pool_data["name"]
            )
        )
        if not result.scalar_one_or_none():
            pool = ResourcePoolConfig(**pool_data)
            session.add(pool)
            created_count += 1
            logger.info(f"Created resource pool: {pool_data['name']}")
    
    if created_count > 0:
        await session.commit()
    
    return created_count


async def main():
    """Run all seed functions."""
    logger.info("🌱 Starting Community Edition seed data loading...")
    
    # Create database engine and session
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Bypass Row-Level Security for seeding operations
        await session.execute(text("SET app.is_admin = 'true'"))
        try:
            # Seed users
            user_count = await seed_users(session)
            logger.info(f"✅ Created {user_count} users")
            
            # Seed adapters using the adapter seeding module (matching Business/Enterprise pattern)
            logger.info("🔌 Seeding adapters for Community Edition...")
            from seed_adapters import seed_adapters_for_edition
            adapter_results = await seed_adapters_for_edition("community")
            adapter_count = adapter_results.get("seeded_to_db", 0)
            logger.info(f"✅ Created {adapter_count} adapters")
            
            # Seed workflow templates
            template_count = await seed_workflow_templates(session)
            logger.info(f"✅ Created {template_count} workflow templates")
            
            # Seed API keys
            key_count = await seed_api_keys(session)
            logger.info(f"✅ Created {key_count} API keys")
            
            # Seed subscription plans
            plan_count = await seed_subscription_plans(session)
            logger.info(f"✅ Created {plan_count} subscription plans")
            
            # Seed usage limits
            limit_count = await seed_usage_limits(session)
            logger.info(f"✅ Created {limit_count} usage limits")
            
            # Seed quality dimensions
            dimension_count = await seed_quality_dimensions(session)
            logger.info(f"✅ Created {dimension_count} quality dimensions")
            
            # Seed resource pool configs
            pool_count = await seed_resource_pool_configs(session)
            logger.info(f"✅ Created {pool_count} resource pools")
            
            logger.info("🎉 Seed data loading complete!")
            
        except Exception as e:
            logger.error(f"❌ Error during seeding: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())