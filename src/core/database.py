"""Database configuration with async SQLAlchemy."""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, DateTime, func, text
import uuid
from datetime import datetime
import logging

from .config import get_settings
from .tenant_context import get_current_tenant_id

logger = logging.getLogger(__name__)


# Create engine and session maker lazily
_engine = None
_async_session_maker = None


def get_engine():
    """Get or create the async engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            str(settings.DATABASE_URL),
            echo=False,
            future=True,
            pool_size=settings.MAX_CONNECTIONS_COUNT,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_timeout=10,
        )
    return _engine


def get_session_maker():
    """Get or create the async session maker."""
    global _async_session_maker
    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_maker

# Base, TimestampMixin, and UUIDMixin are imported from models.base above

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session with tenant context.

    Sets PostgreSQL session variable for Row-Level Security (RLS).
    All queries will be automatically filtered by tenant_id.
    """
    async with get_session_maker()() as session:
        try:
            # Set tenant context at PostgreSQL session level for RLS
            # Using set_config() instead of SET because asyncpg doesn't support
            # parameterized SET statements
            tenant_id = get_current_tenant_id()
            if tenant_id:
                await session.execute(
                    text("SELECT set_config('app.current_tenant_id', :tenant_id, false)"),
                    {"tenant_id": tenant_id}
                )
            yield session
            # Don't auto-commit - let the service layer handle it
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_admin_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get admin database session (bypasses RLS).

    Use this for:
    - Migrations
    - System operations
    - Admin tasks that need cross-tenant access

    WARNING: Only use for legitimate admin operations!
    """
    async with get_session_maker()() as session:
        try:
            # Set admin flag to bypass RLS policies
            await session.execute(text("SET app.is_admin = 'true'"))
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database - migrations must be run separately."""
    # Import all models here to ensure they're registered
    from models import community  # noqa
    # Business and Enterprise models are in their respective editions
    
    # Don't create tables - rely on migrations
    logger.info("Database initialization - relying on migrations")
    logger.info("Run migrations with: ./run-migrations.sh")


async def close_db() -> None:
    """Close database connections."""
    if _engine is not None:
        await _engine.dispose()