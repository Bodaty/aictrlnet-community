"""Find the tenant a pre-tenant request belongs to, and bind the request to it.

Sign-in steps that arrive without a JWT (an API key, an OAuth2 client token,
a social sign-in, a password reset link) start on the default tenant's
session, and row-level security hides every other tenant's rows from it.
Each helper here reads only a tenant id, on a session allowed to see every
tenant, from a value the server stored (never a client-supplied header or
body); the rest of the step then runs bound to that tenant.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import bind_session_tenant, get_session_maker
from core.tenant_context import set_current_tenant_id


async def bind_request_tenant(db: AsyncSession, tenant_id: Optional[str]) -> None:
    """Run the rest of the request under this tenant: its session and its context."""
    if not tenant_id:
        return
    bind_session_tenant(db, str(tenant_id))
    if db.in_transaction():
        await db.commit()  # the binding applies from the next transaction
    set_current_tenant_id(str(tenant_id))


def _single(tenants) -> Optional[str]:
    tenants = {t for t in tenants if t}
    return next(iter(tenants)) if len(tenants) == 1 else None


async def tenant_of_account(identifier: str) -> Optional[str]:
    """The tenant of the one account with this email or username (None if none or several)."""
    from models.user import User

    async with get_session_maker()() as admin:
        admin.info["is_admin"] = True
        tenants = (await admin.execute(
            select(User.tenant_id).where((User.email == identifier) | (User.username == identifier))
        )).scalars().all()
    return _single(tenants)


async def tenant_of_api_key(raw_key: str) -> Optional[str]:
    """The tenant of the user who owns this API key (None if no active key matches).

    API keys carry no tenant column; the key's user does. The key's hash is
    verified here too, so only the key's real owner decides the tenant.
    """
    from models.api_key import APIKey
    from models.user import User
    from services.api_key_service import APIKeyService

    if not raw_key or not raw_key.startswith(f"{APIKeyService.KEY_PREFIX}_"):
        return None
    parts = raw_key.split("_")
    if len(parts) < 3:
        return None
    key_prefix = f"{parts[0]}_{parts[1]}_{parts[2][:2]}"

    async with get_session_maker()() as admin:
        admin.info["is_admin"] = True
        rows = (await admin.execute(
            select(APIKey, User.tenant_id)
            .join(User, User.id == APIKey.user_id)
            .where(APIKey.key_prefix == key_prefix, APIKey.is_active == True)  # noqa: E712
        )).all()
        verifier = APIKeyService(admin)
        tenants = []
        now = datetime.utcnow()
        for key, tenant_id in rows:
            if key.expires_at and key.expires_at < now:
                continue
            if verifier._verify_key_hash(raw_key, key.key_hash, bytes.fromhex(key.key_salt)):
                tenants.append(tenant_id)
    return _single(tenants)


async def tenant_of_oauth2_token(token: str) -> Optional[str]:
    """The tenant a live OAuth2 access token acts in: its user's own tenant.

    Tokens are stored hashed. None when the token is unknown, expired or
    revoked, or when its client names a different tenant than its user.
    """
    import sys

    if "/workspace/editions/business/src" not in sys.path:
        sys.path.insert(0, "/workspace/editions/business/src")
    try:
        from aictrlnet_business.models.oauth2 import OAuth2AccessToken, OAuth2Client  # type: ignore
        from aictrlnet_business.services.oauth2_service_async import OAuth2ServiceAsync  # type: ignore
    except ImportError:
        return None
    from models.user import User

    async with get_session_maker()() as admin:
        admin.info["is_admin"] = True
        token_hash = OAuth2ServiceAsync(admin)._hash_token(token)
        row = (await admin.execute(
            select(OAuth2Client.tenant_id, User.tenant_id, OAuth2Client.created_by, OAuth2Client.client_metadata)
            .select_from(OAuth2AccessToken)
            .outerjoin(OAuth2Client, OAuth2Client.client_id == OAuth2AccessToken.client_id)
            .outerjoin(User, User.id == OAuth2AccessToken.user_id)
            .where(
                OAuth2AccessToken.token_hash == token_hash,
                OAuth2AccessToken.expires_at > datetime.utcnow(),
                OAuth2AccessToken.revoked == False,  # noqa: E712
            )
        )).first()
    if not row:
        return None
    client_tenant, user_tenant, created_by, client_metadata = row
    # The user row is server-stored. A user-created client that names a
    # tenant other than its user's own is refused rather than trusted. MCP
    # dynamic-registration clients sit in a sandbox tenant and carry tokens
    # the user consented to: they act in the consenting user's tenant.
    if not user_tenant:
        return None
    is_dcr_client = created_by == "__dcr__" or (client_metadata or {}).get("dcr") is True
    if client_tenant and not is_dcr_client and str(client_tenant) != str(user_tenant):
        return None
    return str(user_tenant)
