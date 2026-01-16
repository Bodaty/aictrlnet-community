"""Integration module for new infrastructure components."""

import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Import our new components
from events.event_bus import event_bus
from events.handlers import handler_registry
from adapters.registry import adapter_registry
from control_plane.routes import router as control_plane_router
from control_plane.registry import component_registry
from nodes.registry import node_registry

# Import existing components
from core.config import get_settings


logger = logging.getLogger(__name__)


async def initialize_infrastructure():
    """Initialize all new infrastructure components."""
    settings = get_settings()
    
    logger.info("Initializing new infrastructure components...")
    
    try:
        # Start event bus
        await event_bus.start()
        logger.info("Event bus started")
        
        # Start adapter health checks
        await adapter_registry.start_health_checks(interval_seconds=60)
        logger.info("Adapter health checks started")
        
        # Initialize default adapters based on environment
        await _initialize_default_adapters(settings)
        
        # Register system components with control plane
        await _register_system_components()
        
        logger.info("Infrastructure initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize infrastructure: {str(e)}")
        raise


async def shutdown_infrastructure():
    """Shutdown all infrastructure components."""
    logger.info("Shutting down infrastructure components...")
    
    try:
        # Stop event bus
        await event_bus.stop()
        
        # Stop adapter health checks and cleanup
        await adapter_registry.shutdown()
        
        # Any other cleanup
        logger.info("Infrastructure shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during infrastructure shutdown: {str(e)}")


async def _initialize_default_adapters(settings):
    """Initialize default adapters based on configuration."""
    from adapters.factory import AdapterFactory
    from adapters.models import AdapterConfig
    
    # Initialize internal service adapters (always available)
    
    # LLM Service Adapter (for internal LLM service)
    # Only initialize if LLM_SERVICE_URL is configured
    if settings.LLM_SERVICE_URL:
        try:
            llm_config = AdapterConfig(
                name="llm-service",
                category="ai",
                version="1.0.0",
                description="Internal LLM Service Adapter",
                required_edition="community",
                parameters={
                    "service_url": settings.LLM_SERVICE_URL,
                    "timeout": 30,
                    "api_key": "dev-token-for-testing"
                }
            )
            await AdapterFactory.create_and_register_adapter("llm-service", llm_config.dict())
            logger.info(f"LLM Service adapter initialized with URL: {settings.LLM_SERVICE_URL}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM Service adapter: {str(e)}")
    else:
        logger.info("LLM Service adapter disabled (LLM_SERVICE_URL not configured)")
    
    # ML Service Adapter (for internal ML service)
    try:
        ml_config = AdapterConfig(
            name="ml-service",
            category="ai",
            version="1.0.0",
            description="Internal ML Service Adapter",
            required_edition="business",
            base_url="http://ml-service:8003",
            timeout_seconds=60,
            parameters={}
        )
        await AdapterFactory.create_and_register_adapter("ml-service", ml_config.dict())
        logger.info("ML Service adapter initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize ML Service adapter: {str(e)}")
    
    # MCP Client Adapter (for connecting to external MCP servers)
    try:
        mcp_config = AdapterConfig(
            name="mcp-client",
            category="ai",
            version="1.0.0",
            description="MCP Client Adapter for external servers",
            required_edition="community",
            parameters={}
        )
        await AdapterFactory.create_and_register_adapter("mcp-client", mcp_config.dict())
        logger.info("MCP Client adapter initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize MCP Client adapter: {str(e)}")
    
    # Initialize external service adapters based on API keys
    
    # OpenAI adapter
    if settings.OPENAI_API_KEY:
        try:
            await AdapterFactory.create_and_register_adapter(
                "openai",
                {
                    "api_key": settings.OPENAI_API_KEY,
                    "version": "1.0.0",
                    "required_edition": "community"
                }
            )
            logger.info("OpenAI adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI adapter: {str(e)}")
    
    # Claude adapter
    if hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY:
        try:
            await AdapterFactory.create_and_register_adapter(
                "claude",
                {
                    "api_key": settings.ANTHROPIC_API_KEY,
                    "version": "1.0.0",
                    "required_edition": "community"
                }
            )
            logger.info("Claude adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Claude adapter: {str(e)}")
    
    # Slack adapter
    if hasattr(settings, 'SLACK_BOT_TOKEN') and settings.SLACK_BOT_TOKEN:
        try:
            from adapters.factory import AdapterFactory
            
            await AdapterFactory.create_and_register_adapter(
                "slack",
                {
                    "credentials": {
                        "bot_token": settings.SLACK_BOT_TOKEN
                    },
                    "version": "1.0.0",
                    "required_edition": "community"
                }
            )
            logger.info("Slack adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Slack adapter: {str(e)}")


async def _register_system_components():
    """Register system components with the control plane."""
    from control_plane.models import ComponentRegistrationRequest, ComponentType
    
    # Register event bus as a system component
    try:
        await control_plane_service.register_component(
            ComponentRegistrationRequest(
                name="event-bus",
                type=ComponentType.SERVICE,
                version="1.0.0",
                description="Central event bus for system events",
                capabilities=[
                    {
                        "name": "publish",
                        "description": "Publish events",
                        "parameters": {"event_type": "string", "data": "object"}
                    },
                    {
                        "name": "subscribe",
                        "description": "Subscribe to events",
                        "parameters": {"event_types": "array"}
                    }
                ],
                edition="community"
            ),
            user_id="system"
        )
    except Exception as e:
        logger.warning(f"Failed to register event bus: {str(e)}")
    
    # Register node executor
    try:
        await control_plane_service.register_component(
            ComponentRegistrationRequest(
                name="node-executor",
                type=ComponentType.SERVICE,
                version="1.0.0",
                description="Workflow node execution engine",
                capabilities=[
                    {
                        "name": "execute_workflow",
                        "description": "Execute a workflow",
                        "parameters": {"workflow_id": "string"}
                    }
                ],
                edition="community"
            ),
            user_id="system"
        )
    except Exception as e:
        logger.warning(f"Failed to register node executor: {str(e)}")


def integrate_routes(app: FastAPI, edition: str = "community"):
    """Integrate new routes into the FastAPI application."""
    # Add control plane routes for all editions
    # (they already exist in Business/Enterprise, but we can enhance them)
    
    if edition == "community":
        # Add enhanced control plane routes to community edition
        app.include_router(
            control_plane_router,
            prefix="/api/v1",
            tags=["control-plane-enhanced"]
        )
        logger.info("Enhanced control plane routes added to Community edition")
    
    # Routes are already included in Business/Enterprise editions


def get_infrastructure_lifespan(original_lifespan=None):
    """Create a lifespan context manager that includes infrastructure."""
    @asynccontextmanager
    async def infrastructure_lifespan(app: FastAPI):
        # Run original startup if provided
        if original_lifespan:
            async with original_lifespan(app) as _:
                # Initialize our infrastructure
                await initialize_infrastructure()
                
                yield
                
                # Shutdown our infrastructure
                await shutdown_infrastructure()
        else:
            # Just our infrastructure
            await initialize_infrastructure()
            
            yield
            
            await shutdown_infrastructure()
    
    return infrastructure_lifespan


# Helper functions for testing
async def get_infrastructure_status():
    """Get status of all infrastructure components."""
    return {
        "event_bus": {
            "stats": await event_bus.get_stats()
        },
        "adapters": {
            "registered": len(await adapter_registry.list_adapters()),
            "healthy": len([
                a for a in await adapter_registry.list_adapters()
                if a.status == "ready"
            ])
        },
        "control_plane": {
            "components": len(component_registry.components),
            "active": len([
                c for c in component_registry.components.values()
                if c.status == "active"
            ])
        },
        "nodes": {
            "types_available": len(node_registry.get_available_node_types())
        }
    }