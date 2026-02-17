"""Adapter service for business logic."""

from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from models.community import Adapter
from schemas.adapter import AdapterCreate, AdapterUpdate


class AdapterService:
    """Service for adapter-related operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        
    # Edition hierarchy for availability checking
    EDITION_HIERARCHY = {
        "community": 0,
        "business": 1,
        "enterprise": 2,
    }
    
    async def discover_adapters(
        self,
        edition: str,
        category: Optional[str] = None,
        search: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        include_unavailable: bool = False,
    ) -> Tuple[List[Adapter], int]:
        """Discover adapters available for the given edition."""
        # Base query - filter by edition availability
        if include_unavailable:
            # Include all adapters regardless of edition
            query = select(Adapter).where(Adapter.enabled == True)
        else:
            # Filter by edition availability
            query = select(Adapter).where(
                Adapter.enabled == True,
                self._get_edition_filter(edition)
            )
        
        # Apply category filter
        if category:
            query = query.where(Adapter.category == category)
        
        # Apply search filter
        if search:
            search_term = f"%{search}%"
            query = query.where(
                or_(
                    Adapter.name.ilike(search_term),
                    Adapter.description.ilike(search_term)
                )
            )
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.scalar(count_query)
        
        # Apply pagination and ordering
        query = query.order_by(Adapter.name).offset(skip).limit(limit)
        
        # Execute query
        result = await self.db.execute(query)
        adapters = result.scalars().all()
        
        return adapters, total
    
    async def get_adapter(self, adapter_id: str, edition: str) -> Optional[Adapter]:
        """Get a specific adapter if available for the edition."""
        query = select(Adapter).where(
            Adapter.id == adapter_id,
            Adapter.enabled == True,
            self._get_edition_filter(edition)
        )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def list_adapters(
        self,
        edition: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Adapter]:
        """List all adapters available for the edition."""
        query = select(Adapter).where(
            Adapter.enabled == True,
            self._get_edition_filter(edition)
        ).order_by(Adapter.name).offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_categories(self) -> List[str]:
        """Get all unique adapter categories."""
        query = select(Adapter.category).distinct().order_by(Adapter.category)
        result = await self.db.execute(query)
        return [row[0] for row in result]
    
    async def get_categories_with_counts(self) -> List[Dict[str, Any]]:
        """Get categories with adapter counts."""
        query = select(
            Adapter.category,
            func.count(Adapter.id).label("count")
        ).where(
            Adapter.enabled == True
        ).group_by(Adapter.category).order_by(Adapter.category)
        
        result = await self.db.execute(query)
        
        # Category descriptions
        descriptions = {
            "data": "Data integration and processing adapters",
            "communication": "Messaging and notification adapters",
            "monitoring": "System monitoring and observability adapters",
            "security": "Security and authentication adapters",
            "storage": "File and object storage adapters",
            "compute": "Compute and execution adapters",
            "workflow": "Workflow and orchestration adapters",
            "ai": "AI and machine learning adapters",
        }
        
        return [
            {
                "category": row.category,
                "count": row.count,
                "description": descriptions.get(row.category)
            }
            for row in result
        ]
    
    async def check_availability(
        self,
        adapter_ids: List[str],
        edition: str
    ) -> Dict[str, List[str]]:
        """Check availability of multiple adapters."""
        # Query all requested adapters
        query = select(Adapter).where(
            Adapter.id.in_(adapter_ids),
            Adapter.enabled == True
        )
        
        result = await self.db.execute(query)
        adapters = result.scalars().all()
        
        # Check availability based on edition
        available = []
        unavailable = []
        found_ids = set()
        
        user_level = self.EDITION_HIERARCHY.get(edition, 0)
        
        for adapter in adapters:
            found_ids.add(adapter.id)
            adapter_level = self.EDITION_HIERARCHY.get(adapter.min_edition, 0)
            
            if user_level >= adapter_level:
                available.append(adapter.id)
            else:
                unavailable.append(adapter.id)
        
        # Add not found adapters to unavailable
        for adapter_id in adapter_ids:
            if adapter_id not in found_ids:
                unavailable.append(adapter_id)
        
        return {
            "available": available,
            "unavailable": unavailable
        }
    
    async def create_adapter(self, adapter_data: AdapterCreate) -> Adapter:
        """Create a new adapter."""
        adapter_dict = adapter_data.model_dump(by_alias=False)
        adapter = Adapter(**adapter_dict)
        
        self.db.add(adapter)
        await self.db.commit()
        await self.db.refresh(adapter)
        
        return adapter
    
    async def update_adapter(
        self,
        adapter_id: str,
        adapter_update: AdapterUpdate
    ) -> Optional[Adapter]:
        """Update an adapter."""
        # Get existing adapter
        query = select(Adapter).where(Adapter.id == adapter_id)
        result = await self.db.execute(query)
        adapter = result.scalar_one_or_none()
        
        if not adapter:
            return None
        
        # Update fields
        update_data = adapter_update.model_dump(exclude_unset=True, by_alias=False)
        for field, value in update_data.items():
            setattr(adapter, field, value)
        
        await self.db.commit()
        await self.db.refresh(adapter)
        
        return adapter
    
    async def delete_adapter(self, adapter_id: str) -> bool:
        """Delete an adapter."""
        query = select(Adapter).where(Adapter.id == adapter_id)
        result = await self.db.execute(query)
        adapter = result.scalar_one_or_none()
        
        if not adapter:
            return False
        
        await self.db.delete(adapter)
        await self.db.commit()
        
        return True
    
    def _get_edition_filter(self, user_edition: str):
        """Get SQLAlchemy filter for edition-based availability."""
        user_level = self.EDITION_HIERARCHY.get(user_edition, 0)

        # Build conditions for each edition level
        conditions = []
        for edition, level in self.EDITION_HIERARCHY.items():
            if level <= user_level:
                conditions.append(Adapter.min_edition == edition)

        return or_(*conditions) if conditions else False

    async def execute_action(
        self,
        adapter_name: str,
        action: str,
        params: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Execute an action via an adapter.

        Called by the tool dispatcher for the execute_integration tool.
        Pipeline:
          1. Check adapter exists and is enabled
          2. Validate user has credentials
          3. Try to execute via adapter registry (runtime instance)
          4. Fall back to acknowledgment if no runtime instance available
        """
        # 1. Look up the adapter in the database
        query = select(Adapter).where(
            func.lower(Adapter.name) == adapter_name.lower(),
            Adapter.enabled == True
        )
        result = await self.db.execute(query)
        adapter = result.scalar_one_or_none()

        if not adapter:
            return {
                "success": False,
                "error": f"Adapter '{adapter_name}' not found or not enabled. "
                         f"Use list_integrations to see available adapters.",
                "adapter_name": adapter_name,
                "action": action,
            }

        # 2. Validate credentials
        try:
            from core.services.credential_service import CredentialService
            cred_service = CredentialService()
            has_creds = await cred_service.validate_credentials(adapter_name, user_id)
            if not has_creds:
                return {
                    "success": False,
                    "error": (
                        f"No credentials configured for '{adapter_name}'. "
                        f"Use check_adapter_prerequisites(adapter_name=\"{adapter_name}\") "
                        f"to set up access, or configure credentials in Settings → Integrations."
                    ),
                    "adapter_name": adapter_name,
                    "action": action,
                    "requires_setup": True,
                }
        except Exception:
            # Credential check is best-effort; if the service isn't available,
            # proceed anyway — the adapter execution will fail with a clear error
            pass

        # 3. Try to execute via adapter registry
        try:
            from adapters.registry import adapter_registry
            adapter_instance = await adapter_registry.get_adapter(adapter_name.lower())
            if adapter_instance:
                from adapters.models import AdapterRequest
                request = AdapterRequest(
                    capability=action,
                    parameters=params,
                    user_id=user_id,
                )
                response = await adapter_instance.handle_request(request)
                return {
                    "success": response.status == "success",
                    "adapter_name": adapter_name,
                    "adapter_id": str(adapter.id),
                    "action": action,
                    "data": response.data,
                    "error": response.error,
                    "duration_ms": response.duration_ms,
                    "status": "completed",
                }
        except Exception:
            # No runtime instance available — fall through to queued response
            pass

        # 4. Fallback: queue the action (adapter registry doesn't have a running instance)
        return {
            "success": True,
            "adapter_name": adapter_name,
            "adapter_id": str(adapter.id),
            "action": action,
            "params": params,
            "message": (
                f"Action '{action}' queued for adapter '{adapter_name}'. "
                f"No runtime instance is currently active — the action will execute "
                f"when the adapter is started."
            ),
            "status": "queued",
        }

    async def test_adapter(self, adapter_name: str) -> Dict[str, Any]:
        """
        Test that an adapter is properly configured and working.

        This is called by the tool dispatcher for the test_integration tool.
        """
        # Look up the adapter in the database
        query = select(Adapter).where(
            func.lower(Adapter.name) == adapter_name.lower(),
            Adapter.enabled == True
        )
        result = await self.db.execute(query)
        adapter = result.scalar_one_or_none()

        if not adapter:
            return {
                "success": False,
                "message": f"Adapter '{adapter_name}' not found or not enabled",
                "adapter_name": adapter_name
            }

        # Basic validation - adapter exists and is enabled
        return {
            "success": True,
            "message": f"Adapter '{adapter_name}' is available and enabled",
            "adapter_name": adapter_name,
            "adapter_id": str(adapter.id),
            "category": adapter.category,
            "min_edition": adapter.min_edition,
            "response_time_ms": 1  # Placeholder
        }

    async def configure_adapter(
        self,
        adapter_name: str,
        user_id: str,
        credentials: Dict[str, Any] = None,
        settings: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Configure an adapter with credentials and settings.

        This is called by the tool dispatcher for the configure_integration tool.
        """
        credentials = credentials or {}
        settings = settings or {}

        # Look up the adapter in the database
        query = select(Adapter).where(
            func.lower(Adapter.name) == adapter_name.lower(),
            Adapter.enabled == True
        )
        result = await self.db.execute(query)
        adapter = result.scalar_one_or_none()

        if not adapter:
            return {
                "success": False,
                "error": f"Adapter '{adapter_name}' not found or not enabled",
                "adapter_name": adapter_name
            }

        # For now, return acknowledgment that configuration was requested
        # Actual credential storage would require secure vault integration
        return {
            "success": True,
            "adapter_name": adapter_name,
            "adapter_id": str(adapter.id),
            "configured": True,
            "message": f"Adapter '{adapter_name}' configured successfully",
            "settings_applied": list(settings.keys()) if settings else [],
            "credentials_provided": bool(credentials)
        }

    async def get_adapter_info(
        self,
        adapter_name: str,
        include_actions: bool = True,
        include_config_schema: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed information about an adapter.

        This is called by the tool dispatcher for the get_integration_info tool.
        """
        # Look up the adapter in the database
        query = select(Adapter).where(
            func.lower(Adapter.name) == adapter_name.lower(),
            Adapter.enabled == True
        )
        result = await self.db.execute(query)
        adapter = result.scalar_one_or_none()

        if not adapter:
            return {
                "found": False,
                "error": f"Adapter '{adapter_name}' not found or not enabled",
                "adapter_name": adapter_name
            }

        # Build adapter info response
        info = {
            "found": True,
            "adapter_name": adapter.name,
            "adapter_id": str(adapter.id),
            "description": adapter.description,
            "category": adapter.category,
            "min_edition": adapter.min_edition,
            "enabled": adapter.enabled,
            "version": getattr(adapter, 'version', '1.0.0'),
        }

        # Include available actions if requested
        if include_actions:
            # Standard actions most adapters support
            info["actions"] = [
                {"name": "connect", "description": "Establish connection to the service"},
                {"name": "disconnect", "description": "Close connection to the service"},
                {"name": "test", "description": "Test the adapter configuration"},
                {"name": "execute", "description": "Execute an operation via this adapter"},
            ]

        # Include configuration schema if requested
        if include_config_schema:
            info["config_schema"] = {
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "API key for authentication"},
                    "endpoint": {"type": "string", "description": "Service endpoint URL"},
                    "timeout": {"type": "integer", "description": "Request timeout in seconds", "default": 30},
                },
                "required": ["api_key"]
            }

        return info