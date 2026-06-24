"""Template resolution and credential lookup utilities for workflow nodes.

Resolves {{dotted.path}} patterns from a context dict and {{env.VAR}} from
os.environ.  Works recursively on strings, dicts, and lists so it can be
applied to entire parameter trees in one call.

Also provides ``get_adapter_credentials`` to fetch user-configured adapter
API keys from the database so workflow nodes honour UI-configured settings.
"""

import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def resolve_templates(value: Any, context: Dict[str, Any]) -> Any:
    """Resolve ``{{path.to.var}}`` placeholders in *value*.

    Supported patterns:
    - ``{{env.VAR}}``        – looked up in ``os.environ``
    - ``{{today}}`` / ``{{today+Nd}}`` / ``{{today-Nd}}`` – a UTC date (YYYY-MM-DD), N days
      from today; for relative reminders (e.g. a calendar follow-up "3 days out")
    - ``{{dotted.path}}``    – walked through *context* dict/list
    - ``{{simple_key}}``     – top-level key in *context*

    Unresolved placeholders are left as-is so downstream code can still
    detect missing data.
    """
    if isinstance(value, str):
        # Relative-date tokens first (they contain +/- which the path regex won't match).
        def _date_replacer(m: "re.Match") -> str:
            from datetime import datetime, timedelta
            sign, num = m.group(1), m.group(2)
            days = (int(num) * (1 if sign == "+" else -1)) if (sign and num) else 0
            return (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")

        value = re.sub(r"\{\{today(?:([+-])(\d+)d)?\}\}", _date_replacer, value)

        def _replacer(match: re.Match) -> str:
            path = match.group(1)

            # Environment variable lookup
            if path.startswith("env."):
                return os.environ.get(path[4:], match.group(0))

            # Walk the dotted path through context
            parts = path.split(".")
            current: Any = context
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, (list, tuple)) and part.isdigit():
                    idx = int(part)
                    current = current[idx] if idx < len(current) else None
                else:
                    return match.group(0)  # unresolvable
                if current is None:
                    return match.group(0)  # unresolvable
            return str(current)

        return re.sub(r"\{\{(\w+(?:\.\w+)*)\}\}", _replacer, value)

    if isinstance(value, dict):
        return {k: resolve_templates(v, context) for k, v in value.items()}

    if isinstance(value, list):
        return [resolve_templates(item, context) for item in value]

    return value


async def get_adapter_credentials(adapter_type: str) -> Optional[Dict[str, Any]]:
    """Look up user-configured credentials for *adapter_type* from the DB.

    Returns the decrypted credentials dict if an enabled config exists,
    otherwise ``None``.  Falls back gracefully on any error so workflow
    execution is never blocked by a missing config row.
    """
    try:
        from core.database import get_session_maker
        from models.adapter_config import UserAdapterConfig
        from core.crypto import decrypt_data
        from sqlalchemy import select

        async with get_session_maker()() as session:
            query = (
                select(UserAdapterConfig)
                .where(
                    UserAdapterConfig.adapter_type == adapter_type,
                    UserAdapterConfig.enabled == True,
                )
                .order_by(UserAdapterConfig.updated_at.desc())
                .limit(1)
            )
            result = await session.execute(query)
            config = result.scalar_one_or_none()

            if config and config.credentials:
                credentials = decrypt_data(config.credentials)
                logger.info(
                    f"Loaded UI-configured credentials for adapter '{adapter_type}'"
                )
                return credentials

    except Exception as exc:
        logger.debug(
            f"Could not load adapter credentials for '{adapter_type}': {exc}"
        )

    return None


# Sentinel tenant the workflow runtime injects when no real org is in scope
# (workflow_execution.py defaults context["tenant_id"] to this). Treated like
# "no tenant" for credential resolution so it falls through to the shared key.
_DEFAULT_TENANT = "default-tenant"


async def get_adapter_credentials_for_tenant(
    adapter_type: str, tenant_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Resolve adapter credentials scoped to a tenant (GEO Phase B2 tiered model).

    Resolution order:
      1. The org's own row (``tenant_id`` matches a real tenant) — their key,
         their billing.
      2. A shared/global row (``tenant_id IS NULL`` — the Bodaty free-tier key).
      3. ``None`` — the adapter falls back to its env key (e.g. PERPLEXITY_API_KEY).

    Unlike ``get_adapter_credentials``, this never returns another tenant's row:
    step 2 matches NULL explicitly, not "any tenant", so there is no cross-tenant
    credential leak. Falls back gracefully (returns None) on any error.
    """
    try:
        from core.database import get_session_maker
        from models.adapter_config import UserAdapterConfig
        from core.crypto import decrypt_data
        from sqlalchemy import select

        async def _load(session, tenant_filter):
            query = (
                select(UserAdapterConfig)
                .where(
                    UserAdapterConfig.adapter_type == adapter_type,
                    UserAdapterConfig.enabled == True,
                    tenant_filter,
                )
                .order_by(UserAdapterConfig.updated_at.desc())
                .limit(1)
            )
            config = (await session.execute(query)).scalar_one_or_none()
            if config and config.credentials:
                return decrypt_data(config.credentials)
            return None

        async with get_session_maker()() as session:
            if tenant_id and tenant_id != _DEFAULT_TENANT:
                creds = await _load(session, UserAdapterConfig.tenant_id == tenant_id)
                if creds:
                    logger.info(
                        f"Loaded tenant '{tenant_id}' credentials for adapter '{adapter_type}'"
                    )
                    return creds
            # Shared / free-tier row, never another org's. Rows saved on
            # single-tenant deployments before tenant normalization carry the
            # literal "default-tenant" — treat those as shared too so
            # existing saved keys keep working without a data migration.
            creds = await _load(
                session,
                (UserAdapterConfig.tenant_id.is_(None))
                | (UserAdapterConfig.tenant_id == _DEFAULT_TENANT),
            )
            if creds:
                logger.info(
                    f"Loaded shared (free-tier) credentials for adapter '{adapter_type}'"
                )
                return creds

    except Exception as exc:
        logger.debug(
            f"Could not load tenant credentials for '{adapter_type}'/'{tenant_id}': {exc}"
        )

    return None


class CredentialsUnavailable(Exception):
    """An adapter's credentials could not be resolved.

    ``connected`` distinguishes "this OAuth integration isn't connected (yet)" — which a
    workflow may legitimately treat as a dry-run fallback — from a hard config error.
    """

    def __init__(self, message: str, *, connected: bool = False):
        super().__init__(message)
        self.connected = connected


async def resolve_adapter_credentials(
    adapter_type: str,
    adapter_class: Any = None,
    *,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve an adapter's credentials by its declared auth type.

    Auth type is read from the adapter class attribute ``AUTH_TYPE`` (default ``"api_key"``):
      - ``api_key`` → existing per-tenant lookup (UserAdapterConfig). UNCHANGED behaviour for
        every adapter that doesn't opt into another type: returns {} when none is stored so the
        adapter falls back to its env key exactly as before, and never raises.
      - ``oauth2`` → a per-user OAuth access token (refreshed) from the token manager, flattened
        with its account_metadata (e.g. QuickBooks ``realm_id``). Provider is read from the
        class attribute ``OAUTH_PROVIDER``.

    Scope is deliberately a policy HERE, not in the adapters: this takes both ``tenant_id`` and
    ``user_id`` (+ ``owner_id`` for unattended runs). Today oauth2 resolves per-user (``user_id``,
    falling back to ``owner_id`` when a workflow runs as "system"); moving to workspace-scoped
    connections later is a change in THIS function, not in any adapter. Raises
    ``CredentialsUnavailable(connected=False)`` when an oauth2 integration isn't connected.
    """
    auth_type = getattr(adapter_class, "AUTH_TYPE", "api_key") if adapter_class else "api_key"

    if auth_type != "oauth2":
        # api_key (default): unchanged per-tenant resolution; never raises.
        return await get_adapter_credentials_for_tenant(adapter_type, tenant_id) or {}

    # oauth2: per-user fresh token, with run-as-owner fallback for unattended/system runs.
    provider = getattr(adapter_class, "OAUTH_PROVIDER", None) or adapter_type
    effective_user = user_id if (user_id and user_id != "system") else owner_id
    if not effective_user:
        raise CredentialsUnavailable(
            f"{adapter_type}: no user to resolve an OAuth token for "
            f"(unattended run with no workflow owner).",
            connected=False,
        )
    try:
        # Business-only service; optional import keeps Community-only deployments clean.
        from aictrlnet_business.services.oauth_token_manager import (
            OAuthNotConnected,
            OAuthRefreshFailed,
            OAuthTokenManager,
        )
    except Exception as exc:  # community-only: no OAuth integrations available
        raise CredentialsUnavailable(
            f"{adapter_type}: OAuth integrations require the Business edition ({exc}).",
            connected=False,
        )

    from core.database import get_session_maker

    async with get_session_maker()() as session:
        try:
            tok = await OAuthTokenManager(session).get_fresh_token(effective_user, provider)
        except (OAuthNotConnected, OAuthRefreshFailed) as exc:
            raise CredentialsUnavailable(str(exc), connected=False) from exc

    creds: Dict[str, Any] = {"access_token": tok["access_token"]}
    if tok.get("email"):
        creds["email"] = tok["email"]
    # Flatten account_metadata (e.g. QuickBooks realm_id) to the top level.
    creds.update(tok.get("account_metadata") or {})
    return creds
