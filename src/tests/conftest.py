"""Pytest configuration for tests."""

import pytest
import asyncio
import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["EDITION"] = "community"
# Don't override DATABASE_URL - let it use the container's existing value


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    # Use the production database for integration tests (will rollback changes)
    database_url = os.environ.get("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@postgres:5432/aictrlnet")

    engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=NullPool,  # Use NullPool for tests to avoid connection issues
    )

    yield engine

    # NullPool holds no connections; dispose on a fresh loop at session end.
    asyncio.new_event_loop().run_until_complete(engine.dispose())


@pytest.fixture(scope="function")
async def db(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test."""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        try:
            yield session
            await session.rollback()  # Rollback after each test
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Clean up adapter registry between tests
    try:
        from adapters.registry import adapter_registry
        # Clear any registered adapters
        if hasattr(adapter_registry, '_adapters'):
            adapter_registry._adapters.clear()
        if hasattr(adapter_registry, '_adapter_classes'):
            adapter_registry._adapter_classes.clear()
        if hasattr(adapter_registry, '_factories'):
            adapter_registry._factories.clear()
    except (ImportError, AttributeError):
        pass  # Registry may not exist or have different structure