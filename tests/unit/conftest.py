"""Real-DB fixtures for the community unit tree (copied from
editions/business/tests/conftest.py:64-104 on 2026-09-10; that tree had none).
NullPool keeps this safe across pytest-asyncio's per-test event loops."""
import asyncio
import os
from typing import AsyncGenerator
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


@pytest.fixture(scope="session")
def test_engine():
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@postgres:5432/aictrlnet",
    )
    engine = create_async_engine(database_url, echo=False, poolclass=NullPool)
    yield engine
    loop = asyncio.new_event_loop()
    loop.run_until_complete(engine.dispose())
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db(test_engine) -> AsyncGenerator[AsyncSession, None]:
    async_session = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        try:
            yield session
            await session.rollback()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
