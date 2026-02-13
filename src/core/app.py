"""
AICtrlNet Community Edition - Core Application

This module provides the base FastAPI application that can be extended
by Business and Enterprise editions.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .config import get_settings
from .database import init_db, close_db
from .cache import get_cache

# Configure logging
logger = logging.getLogger(__name__)


class AICtrlNetApp:
    """Base application class that can be extended by other editions."""
    
    def __init__(self):
        """Initialize the base application."""
        self.settings = get_settings()
        self.app = None
        self._create_app()
    
    def _create_app(self):
        """Create the FastAPI application instance."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan manager."""
            # Startup
            logger.info(f"Starting AICtrlNet {self.settings.EDITION} Edition v{self.settings.VERSION}")
            
            # Initialize database
            await init_db()
            logger.info("Database initialized")

            # Initialize default tenant (for self-hosted mode)
            try:
                from services.tenant_service import TenantService
                from .database import get_session_maker

                async with get_session_maker()() as db:
                    tenant_service = TenantService(db)
                    default_tenant = await tenant_service.get_or_create_default_tenant()
                    logger.info(f"Default tenant ready: {default_tenant.name} (id={default_tenant.id})")
            except Exception as e:
                logger.warning(f"Could not initialize default tenant: {e}")
                # Don't fail startup - migration may not have run yet

            # Initialize cache
            cache = await get_cache()
            logger.info("Cache service initialized")
            
            # Initialize usage collector
            from services.usage_collector import usage_collector
            logger.info("Usage collector service initialized")
            
            # Initialize workflow templates
            try:
                from services.workflow_template_service import WorkflowTemplateService
                from .database import get_session_maker
                
                async with get_session_maker()() as db:
                    template_service = WorkflowTemplateService()
                    count = await template_service.initialize_system_templates(db)
                    logger.info(f"Initialized {count} Community workflow templates")
            except Exception as e:
                logger.error(f"Failed to initialize workflow templates: {e}")
                # Don't fail startup if templates can't be loaded
            
            # Register all adapter classes (not instances) for discovery
            try:
                from adapters.factory import AdapterFactory
                from adapters.registry import adapter_registry
                from adapters.models import AdapterCategory
                
                # Get all available adapter mappings from factory
                available_mappings = AdapterFactory.get_available_adapters()
                logger.info(f"Found {len(available_mappings)} adapter mappings in factory")
                
                # Register each adapter class with the registry
                registered_count = 0
                failed_count = 0
                
                for adapter_type, class_path in available_mappings.items():
                    try:
                        # Load the adapter class
                        adapter_class = AdapterFactory._load_adapter_class(class_path)
                        
                        # Determine category from adapter type
                        category = AdapterFactory._determine_category(adapter_type, {})
                        
                        # Register the class (not an instance) with the registry
                        adapter_registry.register_adapter_class(
                            adapter_type,
                            adapter_class,
                            category,
                            description=f"{adapter_type} adapter"
                        )
                        registered_count += 1
                        logger.debug(f"Registered adapter class: {adapter_type}")
                        
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Failed to register adapter class {adapter_type}: {e}")
                
                logger.info(f"Adapter class registration complete: {registered_count} succeeded, {failed_count} failed")
                
                # Register MCP adapter manually (it's not in the database but needed for MCP protocol)
                try:
                    from adapters.implementations.ai.mcp_client_adapter import MCPClientAdapter
                    from adapters.models import AdapterCategory
                    adapter_registry.register_adapter_class(
                        "mcp-client",
                        MCPClientAdapter,
                        AdapterCategory.AI,
                        description="MCP Protocol Client Adapter"
                    )
                    logger.info("Registered MCP client adapter class")
                except Exception as e:
                    logger.warning(f"Failed to register MCP adapter: {e}")
                
                # Now create instances for critical internal service adapters that should always be running
                # These are special adapters that connect to our internal services
                critical_adapters = []
                
                # LLM Service Adapter (only if we have the class registered)
                if "llm-service" in adapter_registry.get_available_adapter_classes():
                    try:
                        from adapters.models import AdapterConfig
                        llm_config = AdapterConfig(
                            name="llm-service",
                            category="ai",
                            version="1.0.0",
                            description="Internal LLM Service Adapter",
                            required_edition="community",
                            parameters={
                                "service_url": "http://localhost:8000",
                                "timeout": 30,
                                "api_key": "dev-token-for-testing"
                            }
                        )
                        adapter = await adapter_registry.create_adapter("llm-service", llm_config)
                        critical_adapters.append("llm-service")
                        logger.info("LLM Service adapter instance created")
                    except Exception as e:
                        logger.warning(f"Failed to create LLM Service adapter instance: {e}")
                
                # ML Service Adapter (only if class registered and we're Business/Enterprise)
                if self.settings.EDITION in ["business", "enterprise"]:
                    if "ml-service" in adapter_registry.get_available_adapter_classes():
                        try:
                            from adapters.models import AdapterConfig
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
                            adapter = await adapter_registry.create_adapter("ml-service", ml_config)
                            critical_adapters.append("ml-service")
                            logger.info("ML Service adapter instance created")
                        except Exception as e:
                            logger.warning(f"Failed to create ML Service adapter instance: {e}")
                
                if critical_adapters:
                    logger.info(f"Created instances for critical adapters: {critical_adapters}")
                
                # No discovery mode - adapters are discovered from the database
                # Only runtime instances for configured adapters are kept in the registry
                    
            except Exception as e:
                logger.error(f"Failed to register adapter classes: {e}")
                # Don't fail startup if adapters can't be registered
            
            yield
            
            # Shutdown
            await close_db()
            logger.info("Database connections closed")
        
        self.app = FastAPI(
            title=f"{self.settings.PROJECT_NAME} {self.settings.EDITION.title()} Edition",
            description=self._get_description(),
            version=self.settings.VERSION,
            openapi_url=f"{self.settings.API_V1_STR}/openapi.json",
            docs_url=f"{self.settings.API_V1_STR}/docs",
            redoc_url=f"{self.settings.API_V1_STR}/redoc",
            lifespan=lifespan,
        )
        
        self._add_middleware()
        self._add_routes()
    
    def _get_description(self):
        """Get edition-specific description."""
        descriptions = {
            "community": "Open source AI workflow orchestration platform",
            "business": "Enterprise workflow orchestration with advanced features",
            "enterprise": "Full-featured platform with federation and compliance"
        }
        return descriptions.get(self.settings.EDITION, "AI Workflow Orchestration")
    
    def _add_middleware(self):
        """Add middleware to the application."""
        # CORS: include FRONTEND_URL so non-localhost deployments work
        cors_origins = list(self.settings.BACKEND_CORS_ORIGINS)
        frontend = self.settings.FRONTEND_URL.rstrip("/")
        if frontend and frontend not in cors_origins:
            cors_origins.append(frontend)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add tenant middleware FIRST (sets tenant context for all other middleware)
        from middleware.tenant import TenantMiddleware
        self.app.add_middleware(TenantMiddleware)

        # Add enforcement middleware (order matters - add after CORS and Tenant)
        from middleware.enforcement import EnforcementMiddleware, UpgradePromptMiddleware
        self.app.add_middleware(EnforcementMiddleware)
        self.app.add_middleware(UpgradePromptMiddleware)
        
        # Request timing and rate limit headers middleware
        @self.app.middleware("http")
        async def add_process_time_header(request, call_next):
            import time
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Edition"] = self.settings.EDITION
            
            # Add rate limit headers for A2A endpoints
            if request.url.path.startswith("/api/v1/a2a/"):
                # Get rate limit info from request state if available
                if hasattr(request.state, "rate_limit_info"):
                    rate_info = request.state.rate_limit_info
                    response.headers["X-RateLimit-Limit"] = str(rate_info.get("limit", 100))
                    response.headers["X-RateLimit-Remaining"] = str(rate_info.get("remaining", 100))
                    response.headers["X-RateLimit-Reset"] = str(rate_info.get("reset", 0))
            
            return response
    
    def _add_routes(self):
        """Add routes to the application. Override in subclasses to add more."""
        # Root endpoint
        @self.app.get("/")
        async def root():
            return {
                "name": self.settings.PROJECT_NAME,
                "edition": self.settings.EDITION,
                "version": self.settings.VERSION,
                "docs": f"{self.settings.API_V1_STR}/docs",
            }
        
        # Health check
        @self.app.get("/health")
        async def health():
            return {
                "status": "ok",
                "edition": self.settings.EDITION,
                "version": self.settings.VERSION,
            }
        
        # Edition info
        @self.app.get("/api/v1/edition")
        async def edition():
            return {
                "edition": self.settings.EDITION,
                "features": self._get_features(),
                "version": self.settings.VERSION,
            }
        
        # Well-known agent configuration
        @self.app.get("/.well-known/agent.json")
        async def well_known_agent():
            return {
                "name": f"AICtrlNet {self.settings.EDITION.title()} Edition",
                "description": self._get_description(),
                "version": self.settings.VERSION,
                "capabilities": {
                    "workflow_orchestration": True,
                    "ai_governance": self.settings.EDITION in ["business", "enterprise"],
                    "multi_tenancy": self.settings.EDITION == "enterprise",
                    "federation": self.settings.EDITION == "enterprise",
                },
                "api": {
                    "version": "v1",
                    "base_url": self.settings.API_V1_STR,
                    "docs_url": f"{self.settings.API_V1_STR}/docs",
                    "openapi_url": f"{self.settings.API_V1_STR}/openapi.json",
                },
                "edition": self.settings.EDITION,
                "features": self._get_features(),
            }
        
        # Add Community routes
        self._add_community_routes()
    
    def _add_community_routes(self):
        """Add Community edition routes."""
        from api.v1 import api_router
        self.app.include_router(api_router, prefix=self.settings.API_V1_STR)
        
        # Add WebSocket routes
        self._add_websocket_routes()
    
    def _add_websocket_routes(self):
        """Add WebSocket routes for real-time features."""
        from api.v1.websocket.workflow_ws import (
            workflow_execution_websocket,
            workflow_catalog_websocket
        )
        
        # Workflow execution updates
        self.app.websocket("/ws/workflows/{execution_id}")(workflow_execution_websocket)
        
        # Catalog updates
        self.app.websocket("/ws/workflows/catalog")(workflow_catalog_websocket)
    
    def _get_features(self):
        """Get edition-specific features."""
        features = {
            "community": {
                "max_workflows": 10,
                "max_adapters": 5,
                "max_users": 1,
                "ai_enabled": True,
                "custom_branding": False,
            },
            "business": {
                "max_workflows": 100,
                "max_adapters": 20,
                "max_users": 50,
                "ai_enabled": True,
                "custom_branding": True,
                "approval_workflows": True,
                "rbac": True,
            },
            "enterprise": {
                "max_workflows": -1,  # Unlimited
                "max_adapters": -1,
                "max_users": -1,
                "ai_enabled": True,
                "custom_branding": True,
                "approval_workflows": True,
                "rbac": True,
                "multi_tenant": True,
                "federation": True,
                "audit_logging": True,
            }
        }
        return features.get(self.settings.EDITION, features["community"])
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    app_instance = AICtrlNetApp()
    return app_instance.get_app()