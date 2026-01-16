"""Basic agent configuration service for Community Edition.

Provides simple agent configuration using user preferences (no encryption).
Limited to 3 agents and single API provider.
"""

from typing import Dict, Any, Optional, List
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.user import User

logger = logging.getLogger(__name__)


class AgentConfigService:
    """Basic agent configuration for Community Edition."""
    
    MAX_AGENTS = 3  # Community limit
    ALLOWED_PROVIDERS = ["ollama", "openai"]  # Single provider at a time
    ALLOWED_AGENTS = ["basic_nlp", "basic_workflow", "basic_assistant"]
    
    async def get_user_agent_config(
        self,
        db: AsyncSession,
        user_id: str
    ) -> Dict[str, Any]:
        """Get user's agent configuration from preferences."""
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user or not user.preferences:
            return self._default_config()
        
        return user.preferences.get("agent_config", self._default_config())
    
    async def update_api_key(
        self,
        db: AsyncSession,
        user_id: str,
        provider: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update API key for a provider (only one allowed)."""
        if provider not in self.ALLOWED_PROVIDERS:
            raise ValueError(f"Provider {provider} not allowed in Community Edition")
        
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise ValueError("User not found")
        
        # Initialize preferences if needed
        if not user.preferences:
            user.preferences = {}
        
        # Update agent config (single provider only)
        if "agent_config" not in user.preferences:
            user.preferences["agent_config"] = self._default_config()
            
        user.preferences["agent_config"]["api_provider"] = provider
        user.preferences["agent_config"]["api_config"] = config
        
        # Mark preferences as modified for SQLAlchemy to detect change
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(user, "preferences")
        
        await db.commit()
        return {"success": True, "provider": provider}
    
    async def can_execute_agent(
        self,
        db: AsyncSession,
        user_id: str,
        agent_name: str
    ) -> bool:
        """Check if user can execute this agent."""
        if agent_name not in self.ALLOWED_AGENTS:
            return False
            
        config = await self.get_user_agent_config(db, user_id)
        enabled = config.get("enabled_agents", [])
        
        # Check agent limit
        if agent_name not in enabled and len(enabled) >= self.MAX_AGENTS:
            return False
        
        # Check if API is configured
        api_provider = config.get("api_provider")
        api_config = config.get("api_config", {})
        
        if api_provider == "ollama":
            return bool(api_config.get("ollama_url"))
        elif api_provider == "openai":
            return bool(api_config.get("api_key"))
        
        return False
    
    async def enable_agent(
        self,
        db: AsyncSession,
        user_id: str,
        agent_name: str
    ) -> Dict[str, Any]:
        """Enable an agent for the user (up to limit)."""
        if agent_name not in self.ALLOWED_AGENTS:
            return {
                "success": False,
                "error": f"Agent {agent_name} not available in Community Edition"
            }
        
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return {"success": False, "error": "User not found"}
        
        # Initialize preferences if needed
        if not user.preferences:
            user.preferences = {}
        if "agent_config" not in user.preferences:
            user.preferences["agent_config"] = self._default_config()
        
        enabled = user.preferences["agent_config"].get("enabled_agents", [])
        
        if agent_name in enabled:
            return {"success": True, "message": "Agent already enabled"}
        
        if len(enabled) >= self.MAX_AGENTS:
            return {
                "success": False,
                "error": f"Community Edition limited to {self.MAX_AGENTS} agents",
                "upgrade_url": "/upgrade/business"
            }
        
        enabled.append(agent_name)
        user.preferences["agent_config"]["enabled_agents"] = enabled
        
        # Mark preferences as modified
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(user, "preferences")
        
        await db.commit()
        return {"success": True, "enabled_agents": enabled}
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for new users."""
        return {
            "enabled_agents": ["basic_nlp", "basic_workflow", "basic_assistant"],  # All 3 enabled by default
            "api_provider": "ollama",
            "api_config": {
                "ollama_url": "http://localhost:11434"
            },
            "model_tier": "fast"
        }
    
    def get_upgrade_benefits(self) -> Dict[str, Any]:
        """Get benefits of upgrading to Business Edition."""
        return {
            "business_edition": {
                "agents": "All 33 AI agents",
                "api_keys": "Unlimited encrypted API keys",
                "models": "All model tiers (fast/balanced/quality)",
                "memory": "Context-aware execution with learning",
                "frameworks": "LangChain, AutoGPT integration",
                "tools": "MCP and adapter integration"
            },
            "upgrade_url": "/upgrade/business"
        }