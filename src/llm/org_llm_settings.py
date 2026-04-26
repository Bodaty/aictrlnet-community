"""Organization-level LLM settings.

Stores LLM configuration in the tenant.settings["llm"] JSON field,
avoiding the need for a new database table or migration.

Fallback chain: Node-specific → Org default → System default → Error
"""

import logging
from typing import Optional, Dict, List, Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

SETTINGS_KEY = "llm"


class OrgLLMSettings(BaseModel):
    """LLM configuration for an organization/tenant."""

    # Provider & model preferences
    preferred_provider: Optional[str] = Field(
        None,
        description="Default LLM provider (ollama, vllm, openai, anthropic, vertex_ai, gemini, deepseek)"
    )
    preferred_model: Optional[str] = Field(
        None,
        description="Default model name (e.g. gpt-4o, claude-sonnet-4-20250514, llama3.1:8b)"
    )
    fallback_provider: Optional[str] = Field(
        None,
        description="Fallback provider if preferred is unavailable"
    )
    fallback_model: Optional[str] = None

    # API key references — maps provider name to credential ID (not raw keys)
    api_key_refs: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps provider -> credential_id from OrganizationAPIKey table"
    )

    # Provider whitelist (empty = all allowed)
    allowed_providers: List[str] = Field(
        default_factory=list,
        description="If set, only these providers can be used"
    )

    # Trial mode
    trial_mode: bool = Field(
        True,
        description="When true, org uses system default only (Bodaty pays). Set false after BYOK."
    )

    # Cost controls
    max_cost_per_call: Optional[float] = Field(
        None, description="Max cost in USD per individual LLM call"
    )
    monthly_budget: Optional[float] = Field(
        None, description="Monthly LLM spend cap in USD"
    )
    monthly_call_limit: Optional[int] = Field(
        5000, description="Max LLM calls per month (default trial limit)"
    )

    def has_own_key(self, provider: Optional[str] = None) -> bool:
        """Check if the org has configured their own API key."""
        if provider:
            return provider in self.api_key_refs
        return len(self.api_key_refs) > 0


async def get_org_llm_settings(
    tenant_id: str,
    db: AsyncSession
) -> Optional[OrgLLMSettings]:
    """Read LLM settings from tenant.settings['llm']."""
    from models.tenant import Tenant

    result = await db.execute(
        select(Tenant).where(Tenant.id == tenant_id)
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        return None

    settings = tenant.settings or {}
    llm_data = settings.get(SETTINGS_KEY)
    if not llm_data:
        # Return defaults — trial mode on, system default
        return OrgLLMSettings()

    return OrgLLMSettings(**llm_data)


async def save_org_llm_settings(
    tenant_id: str,
    llm_settings: OrgLLMSettings,
    db: AsyncSession
) -> OrgLLMSettings:
    """Write LLM settings to tenant.settings['llm']."""
    from models.tenant import Tenant

    result = await db.execute(
        select(Tenant).where(Tenant.id == tenant_id)
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise ValueError(f"Tenant {tenant_id} not found")

    current_settings = dict(tenant.settings or {})
    current_settings[SETTINGS_KEY] = llm_settings.model_dump()
    tenant.settings = current_settings

    await db.flush()
    logger.info(f"Saved LLM settings for tenant {tenant_id}: provider={llm_settings.preferred_provider}, trial={llm_settings.trial_mode}")
    return llm_settings


async def get_tenant_trial_status(
    tenant_id: str,
    db: AsyncSession
) -> bool:
    """Check if tenant is in trial mode (uses system LLM, Bodaty pays)."""
    from models.tenant import Tenant

    result = await db.execute(
        select(Tenant.status).where(Tenant.id == tenant_id)
    )
    row = result.scalar_one_or_none()
    if row == "trial":
        return True

    # Also check LLM settings
    settings = await get_org_llm_settings(tenant_id, db)
    return settings.trial_mode if settings else True
