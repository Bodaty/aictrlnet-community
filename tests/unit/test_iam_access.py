"""Who an IAM caller is: superuser, tenant administrator, or member (G45).

IAM agents belong to a tenant and an owner. A member sees their own agents and
the tenant's system agents; a tenant administrator and a superuser see every
agent in the tenant. Community has no tenant-administrator role, so a higher
edition supplies one through ``register_tenant_admin_check``. The shared
``default-tenant`` holds every self-signup and never has a tenant administrator.
"""

import pytest

from core import iam_access
from core.iam_access import IAMCaller, iam_caller


@pytest.fixture
def admin_check():
    calls = []

    async def check(db, user_id, tenant_id):
        calls.append((user_id, tenant_id))
        return True

    iam_access.register_tenant_admin_check(check)
    yield calls
    iam_access._TENANT_ADMIN_CHECKS.remove(check)


@pytest.mark.asyncio
async def test_superuser_sees_all():
    caller = await iam_caller(None, {"id": "u-1", "tenant_id": "t-1", "is_superuser": True})
    assert caller == IAMCaller(user_id="u-1", tenant_id="t-1", is_superuser=True)
    assert caller.sees_all


@pytest.mark.asyncio
async def test_member_does_not_see_all():
    caller = await iam_caller(None, {"id": "u-1", "tenant_id": "t-1", "is_superuser": False})
    assert not caller.sees_all and not caller.is_tenant_admin


@pytest.mark.asyncio
async def test_registered_check_makes_a_tenant_admin(admin_check):
    caller = await iam_caller(None, {"id": "u-1", "tenant_id": "practice-1"})
    assert caller.is_tenant_admin and caller.sees_all
    assert admin_check == [("u-1", "practice-1")]


@pytest.mark.asyncio
async def test_default_tenant_never_has_a_tenant_admin(admin_check):
    caller = await iam_caller(None, {"id": "u-1", "tenant_id": "default-tenant"})
    assert not caller.is_tenant_admin
    assert admin_check == []


@pytest.mark.asyncio
async def test_sentinel_user_id_is_no_identity(admin_check):
    caller = await iam_caller(None, {"id": "unknown", "tenant_id": "practice-1"})
    assert caller.user_id is None and not caller.sees_all
    assert admin_check == []
