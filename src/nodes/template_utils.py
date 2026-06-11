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
    - ``{{dotted.path}}``    – walked through *context* dict/list
    - ``{{simple_key}}``     – top-level key in *context*

    Unresolved placeholders are left as-is so downstream code can still
    detect missing data.
    """
    if isinstance(value, str):
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
