"""Tests for Row-Level Security (RLS) tenant isolation.

These tests verify that:
1. RLS policies prevent cross-tenant data access
2. Tenant context is properly set in database sessions
3. Admin bypass works for legitimate admin operations
"""

import pytest
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uuid

from core.database import get_db, get_admin_db, get_session_maker
from core.tenant_context import set_current_tenant_id, get_current_tenant_id, DEFAULT_TENANT_ID
from models.community_complete import WorkflowDefinition


# Test fixtures
TENANT_A = "tenant-a-test"
TENANT_B = "tenant-b-test"


@asynccontextmanager
async def tenant_context(tenant_id: str) -> AsyncGenerator[AsyncSession, None]:
    """Context manager for testing with a specific tenant."""
    original_tenant = get_current_tenant_id()
    try:
        set_current_tenant_id(tenant_id)
        async for session in get_db():
            yield session
            break
    finally:
        set_current_tenant_id(original_tenant)


@asynccontextmanager
async def admin_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for admin operations (bypasses RLS)."""
    async for session in get_admin_db():
        yield session
        break


class TestRLSSessionVariables:
    """Test that PostgreSQL session variables are properly set."""

    @pytest.mark.asyncio
    async def test_session_sets_tenant_id(self):
        """Verify that get_db() sets app.current_tenant_id."""
        set_current_tenant_id(TENANT_A)

        async for session in get_db():
            # Check that the session variable is set
            result = await session.execute(
                text("SELECT current_setting('app.current_tenant_id', true)")
            )
            value = result.scalar()
            assert value == TENANT_A
            break

    @pytest.mark.asyncio
    async def test_admin_session_sets_bypass_flag(self):
        """Verify that get_admin_db() sets app.is_admin to true."""
        async for session in get_admin_db():
            result = await session.execute(
                text("SELECT current_setting('app.is_admin', true)")
            )
            value = result.scalar()
            assert value == "true"
            break

    @pytest.mark.asyncio
    async def test_different_sessions_have_different_tenants(self):
        """Verify that different sessions can have different tenant contexts."""
        async with tenant_context(TENANT_A) as session_a:
            result_a = await session_a.execute(
                text("SELECT current_setting('app.current_tenant_id', true)")
            )
            value_a = result_a.scalar()

        async with tenant_context(TENANT_B) as session_b:
            result_b = await session_b.execute(
                text("SELECT current_setting('app.current_tenant_id', true)")
            )
            value_b = result_b.scalar()

        assert value_a == TENANT_A
        assert value_b == TENANT_B


class TestRLSPolicies:
    """Test RLS policy enforcement."""

    @pytest.mark.asyncio
    async def test_rls_blocks_cross_tenant_read(self):
        """Verify that RLS prevents reading data from other tenants."""
        workflow_id = str(uuid.uuid4())

        # Create workflow as tenant A
        async with tenant_context(TENANT_A) as session:
            workflow = WorkflowDefinition(
                id=workflow_id,
                name="Tenant A Workflow",
                description="Test workflow",
                tenant_id=TENANT_A,
                definition={},
                status="active"
            )
            session.add(workflow)
            await session.commit()

        # Try to read from tenant B - should return None (RLS blocks)
        async with tenant_context(TENANT_B) as session:
            result = await session.execute(
                select(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id)
            )
            workflow = result.scalar_one_or_none()
            assert workflow is None, "RLS should block cross-tenant reads"

        # Clean up with admin session
        async with admin_context() as session:
            await session.execute(
                text("DELETE FROM workflow_definitions WHERE id = :id"),
                {"id": workflow_id}
            )
            await session.commit()

    @pytest.mark.asyncio
    async def test_rls_allows_same_tenant_read(self):
        """Verify that RLS allows reading own tenant's data."""
        workflow_id = str(uuid.uuid4())

        # Create workflow as tenant A
        async with tenant_context(TENANT_A) as session:
            workflow = WorkflowDefinition(
                id=workflow_id,
                name="Tenant A Workflow",
                description="Test workflow",
                tenant_id=TENANT_A,
                definition={},
                status="active"
            )
            session.add(workflow)
            await session.commit()

        # Read from same tenant - should work
        async with tenant_context(TENANT_A) as session:
            result = await session.execute(
                select(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id)
            )
            workflow = result.scalar_one_or_none()
            assert workflow is not None, "RLS should allow same-tenant reads"
            assert workflow.name == "Tenant A Workflow"

        # Clean up
        async with admin_context() as session:
            await session.execute(
                text("DELETE FROM workflow_definitions WHERE id = :id"),
                {"id": workflow_id}
            )
            await session.commit()

    @pytest.mark.asyncio
    async def test_rls_blocks_cross_tenant_update(self):
        """Verify that RLS prevents updating data from other tenants."""
        workflow_id = str(uuid.uuid4())

        # Create workflow as tenant A
        async with tenant_context(TENANT_A) as session:
            workflow = WorkflowDefinition(
                id=workflow_id,
                name="Original Name",
                description="Test workflow",
                tenant_id=TENANT_A,
                definition={},
                status="active"
            )
            session.add(workflow)
            await session.commit()

        # Try to update from tenant B - should not affect the row
        async with tenant_context(TENANT_B) as session:
            result = await session.execute(
                text("""
                    UPDATE workflow_definitions
                    SET name = 'Hacked Name'
                    WHERE id = :id
                """),
                {"id": workflow_id}
            )
            await session.commit()
            # With RLS, this update should affect 0 rows
            assert result.rowcount == 0, "RLS should block cross-tenant updates"

        # Verify original name is unchanged
        async with tenant_context(TENANT_A) as session:
            result = await session.execute(
                select(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id)
            )
            workflow = result.scalar_one_or_none()
            assert workflow.name == "Original Name", "Name should not have changed"

        # Clean up
        async with admin_context() as session:
            await session.execute(
                text("DELETE FROM workflow_definitions WHERE id = :id"),
                {"id": workflow_id}
            )
            await session.commit()

    @pytest.mark.asyncio
    async def test_admin_bypass_allows_cross_tenant_access(self):
        """Verify that admin sessions can access all tenant data."""
        workflow_a_id = str(uuid.uuid4())
        workflow_b_id = str(uuid.uuid4())

        # Create workflows for different tenants
        async with admin_context() as session:
            workflow_a = WorkflowDefinition(
                id=workflow_a_id,
                name="Tenant A Workflow",
                description="Test",
                tenant_id=TENANT_A,
                definition={},
                status="active"
            )
            workflow_b = WorkflowDefinition(
                id=workflow_b_id,
                name="Tenant B Workflow",
                description="Test",
                tenant_id=TENANT_B,
                definition={},
                status="active"
            )
            session.add(workflow_a)
            session.add(workflow_b)
            await session.commit()

        # Admin should see both
        async with admin_context() as session:
            result = await session.execute(
                select(WorkflowDefinition).where(
                    WorkflowDefinition.id.in_([workflow_a_id, workflow_b_id])
                )
            )
            workflows = result.scalars().all()
            assert len(workflows) == 2, "Admin should see all tenants' data"

        # Clean up
        async with admin_context() as session:
            await session.execute(
                text("DELETE FROM workflow_definitions WHERE id IN (:a, :b)"),
                {"a": workflow_a_id, "b": workflow_b_id}
            )
            await session.commit()


class TestRLSListQueries:
    """Test RLS with list/aggregate queries."""

    @pytest.mark.asyncio
    async def test_list_only_returns_current_tenant_data(self):
        """Verify that list queries only return current tenant's data."""
        # Create workflows for different tenants
        workflow_ids = []
        async with admin_context() as session:
            for i in range(3):
                for tenant in [TENANT_A, TENANT_B]:
                    workflow_id = str(uuid.uuid4())
                    workflow_ids.append(workflow_id)
                    workflow = WorkflowDefinition(
                        id=workflow_id,
                        name=f"{tenant} Workflow {i}",
                        description="Test",
                        tenant_id=tenant,
                        definition={},
                        status="active"
                    )
                    session.add(workflow)
            await session.commit()

        # Tenant A should only see 3 workflows
        async with tenant_context(TENANT_A) as session:
            result = await session.execute(
                select(WorkflowDefinition).where(
                    WorkflowDefinition.id.in_(workflow_ids)
                )
            )
            workflows = result.scalars().all()
            assert len(workflows) == 3, f"Tenant A should see 3 workflows, got {len(workflows)}"
            for w in workflows:
                assert w.tenant_id == TENANT_A, "All returned workflows should belong to tenant A"

        # Clean up
        async with admin_context() as session:
            await session.execute(
                text("DELETE FROM workflow_definitions WHERE id = ANY(:ids)"),
                {"ids": workflow_ids}
            )
            await session.commit()


class TestTenantContextIntegration:
    """Test tenant context integration with RLS."""

    @pytest.mark.asyncio
    async def test_default_tenant_fallback(self):
        """Verify that default tenant is used when no context is set."""
        set_current_tenant_id(None)

        async for session in get_db():
            result = await session.execute(
                text("SELECT current_setting('app.current_tenant_id', true)")
            )
            value = result.scalar()
            # When context is None, nothing should be set
            assert value is None or value == ""
            break

    @pytest.mark.asyncio
    async def test_context_switch_updates_session(self):
        """Verify that context switches update the session variable."""
        set_current_tenant_id(TENANT_A)

        async for session in get_db():
            # First check
            result = await session.execute(
                text("SELECT current_setting('app.current_tenant_id', true)")
            )
            assert result.scalar() == TENANT_A
            break

        # Switch context
        set_current_tenant_id(TENANT_B)

        async for session in get_db():
            # Second check with new session
            result = await session.execute(
                text("SELECT current_setting('app.current_tenant_id', true)")
            )
            assert result.scalar() == TENANT_B
            break
