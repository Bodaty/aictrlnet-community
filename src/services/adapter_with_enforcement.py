"""Example of adapter service with enforcement integrated."""

from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession

from core.enforcement import LicenseEnforcer, LimitType, LimitExceededException
from core.usage_tracker import get_usage_tracker
from adapters.factory import AdapterFactory
from adapters.models import AdapterConfig


class AdapterServiceWithEnforcement:
    """Adapter service with license enforcement."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.enforcer = LicenseEnforcer(db)
        self.factory = AdapterFactory()
    
    async def create_adapter_instance(
        self,
        tenant_id: str,
        adapter_type: str,
        config: Dict[str, Any],
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an adapter instance with limit checking."""
        
        # Check adapter limit
        await self.enforcer.check_limit(
            tenant_id=tenant_id,
            limit_type=LimitType.ADAPTERS,
            increment=1
        )
        
        # Check if adapter type is available for tenant's edition
        tenant_info = await self.enforcer._get_tenant_info(tenant_id)
        edition = tenant_info["edition"]
        
        # Business adapters require business/enterprise edition
        business_adapters = ["azure_openai", "aws_bedrock", "salesforce", "hubspot", "jira"]
        if adapter_type in business_adapters and edition == "community":
            raise LimitExceededException(
                limit_type=LimitType.ADAPTERS,
                current=0,
                limit=0,
                upgrade_path=self.enforcer._get_upgrade_path(edition)
            )
        
        # Create the adapter
        adapter_config = AdapterConfig(
            adapter_type=adapter_type,
            name=name or f"{adapter_type}_adapter",
            config=config
        )
        
        adapter = self.factory.create_adapter(
            adapter_type=adapter_type,
            config=config
        )
        
        # Track usage
        tracker = await get_usage_tracker(self.db)
        await tracker.track_usage(
            tenant_id=tenant_id,
            metric_type="adapters_created",
            metadata={
                "adapter_type": adapter_type,
                "name": adapter_config.name
            }
        )
        
        return {
            "id": adapter.id,
            "type": adapter_type,
            "name": adapter_config.name,
            "status": "active"
        }
    
    async def list_available_adapters(
        self,
        tenant_id: str
    ) -> List[Dict[str, Any]]:
        """List adapters available for tenant's edition."""
        
        tenant_info = await self.enforcer._get_tenant_info(tenant_id)
        edition = tenant_info["edition"]
        
        # Community adapters (always available)
        available = [
            {"type": "openai", "category": "ai", "name": "OpenAI"},
            {"type": "claude", "category": "ai", "name": "Claude"},
            {"type": "ollama", "category": "ai", "name": "Ollama"},
            {"type": "huggingface", "category": "ai", "name": "HuggingFace"},
            {"type": "slack", "category": "communication", "name": "Slack"},
            {"type": "email", "category": "communication", "name": "Email"},
            {"type": "webhook", "category": "communication", "name": "Webhook"},
            {"type": "discord", "category": "communication", "name": "Discord"},
            {"type": "stripe", "category": "payment", "name": "Stripe"},
        ]
        
        # Business adapters
        if edition in ["business_starter", "business_growth", "business_scale", "enterprise"]:
            available.extend([
                {"type": "azure_openai", "category": "ai", "name": "Azure OpenAI"},
                {"type": "aws_bedrock", "category": "ai", "name": "AWS Bedrock"},
                {"type": "google_gemini", "category": "ai", "name": "Google Gemini"},
                {"type": "teams", "category": "communication", "name": "Microsoft Teams"},
                {"type": "salesforce", "category": "crm", "name": "Salesforce"},
                {"type": "hubspot", "category": "crm", "name": "HubSpot"},
                {"type": "jira", "category": "project_management", "name": "Jira"},
                {"type": "postgresql", "category": "database", "name": "PostgreSQL"},
                {"type": "mysql", "category": "database", "name": "MySQL"},
            ])
        
        # Enterprise adapters
        if edition == "enterprise":
            available.extend([
                {"type": "cohere", "category": "ai", "name": "Cohere"},
                {"type": "redis", "category": "database", "name": "Redis"},
                {"type": "aws_s3", "category": "cloud", "name": "AWS S3"},
            ])
        
        return available
    
    async def check_adapter_quota(
        self,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Check current adapter usage against limits."""
        
        # Get current usage
        from models.community import AdapterInstance
        from sqlalchemy import select, func
        
        result = await self.db.execute(
            select(func.count(AdapterInstance.id)).where(
                AdapterInstance.tenant_id == tenant_id
            )
        )
        current_count = result.scalar() or 0
        
        # Get tenant info and limits
        tenant_info = await self.enforcer._get_tenant_info(tenant_id)
        edition = tenant_info["edition"]
        limit = self.enforcer.EDITION_LIMITS.get(edition, {}).get(LimitType.ADAPTERS, 5)
        
        return {
            "current": current_count,
            "limit": limit,
            "remaining": max(0, limit - current_count),
            "percentage": (current_count / limit * 100) if limit > 0 else 0,
            "edition": edition
        }