"""License enforcement middleware for request-level validation."""

import time
import logging
from typing import Callable, Optional, Dict, Any
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from core.database import get_db
from core.enforcement_simple import (
    LicenseEnforcer, LimitType, LimitExceededException,
    Edition, get_feature_upgrade_info,
)
from core.usage_tracker import get_usage_tracker
from core.security import get_current_active_user
from core.config import get_settings
from core.tenant_context import get_current_tenant_id, DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)


class EnforcementMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce license limits and track usage."""
    
    # Endpoints that don't require enforcement
    EXEMPT_PATHS = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/auth/login",
        "/api/auth/register",
        "/api/upgrade",
        "/api/billing",
        "/api/usage",
        "/.well-known"
    }
    
    # Map endpoints to limit types
    ENDPOINT_LIMITS = {
        "/api/workflows": LimitType.WORKFLOWS,
        "/api/adapters": LimitType.ADAPTERS,
        "/api/users": LimitType.USERS,
        "/api/sessions": LimitType.SESSIONS,
        "/api/agents": LimitType.AGENTS,
    }
    
    # Features required by endpoint pattern
    ENDPOINT_FEATURES = {
        "/api/business/": ["business_adapters", "approval_workflows"],
        "/api/enterprise/": ["enterprise_adapters", "multi_tenant"],
        "/api/approvals/": ["approval_workflows"],
        "/api/compliance/": ["compliance"],
        "/api/federation/": ["federation"],
        "/api/analytics/advanced": ["advanced_analytics"],
        "/api/ai-governance/": ["ai_governance"],
        "/api/a2a/": ["a2a_protocol"],
        "/api/sla/": ["sla_monitoring"],
        "/api/organizations/": ["organization_management"],
        "/api/teams/": ["team_management"],
        "/api/oauth/": ["oauth2_oidc"],
        "/api/saml/": ["saml_sso"],
        "/api/tenants/": ["multi_tenant"],
        "/api/geo-routing/": ["geographic_routing"],
        "/api/audit/": ["audit_logging"],
        "/api/subscriptions/": ["subscription_licensing"],
        "/api/learning-loops/": ["learning_loops"],
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with enforcement checks."""
        
        start_time = time.time()
        
        # Skip enforcement for exempt paths
        if self._is_exempt_path(request.url.path):
            response = await call_next(request)
            return response
        
        # Skip OPTIONS requests
        if request.method == "OPTIONS":
            response = await call_next(request)
            return response
        
        try:
            # Get current user/tenant
            tenant_id = await self._get_tenant_id(request)
            if not tenant_id:
                # No tenant context, allow request (anonymous endpoints)
                response = await call_next(request)
                return response
            
            # Store tenant_id in request state for other middleware
            request.state.tenant_id = tenant_id
            
            # Check feature access
            feature_error = await self._check_feature_access(request, tenant_id)
            if feature_error:
                return JSONResponse(
                    status_code=403,
                    content=feature_error
                )
            
            # Check resource limits for creation endpoints
            if request.method in ["POST", "PUT"] and self._is_resource_endpoint(request.url.path):
                limit_error = await self._check_resource_limits(request, tenant_id)
                if limit_error:
                    return JSONResponse(
                        status_code=402,
                        content=limit_error
                    )
            
            # Process request
            response = await call_next(request)
            
            # Track usage
            await self._track_usage(request, response, tenant_id, start_time)
            
            # Add usage headers to response
            await self._add_usage_headers(response, tenant_id)
            
            return response
            
        except LimitExceededException as e:
            # Already formatted properly
            return JSONResponse(
                status_code=e.status_code,
                content=e.detail
            )
        except Exception as e:
            logger.error(f"Enforcement middleware error: {e}")
            # Don't block requests on enforcement errors - let the endpoint handle them
            # The enforcement middleware should only enforce limits, not handle app errors
            # Pass the error through to the endpoint
            raise
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from enforcement."""
        
        # Exact matches
        if path in self.EXEMPT_PATHS:
            return True
        
        # Prefix matches
        for exempt in self.EXEMPT_PATHS:
            if path.startswith(exempt):
                return True
        
        return False
    
    def _is_resource_endpoint(self, path: str) -> bool:
        """Check if endpoint creates resources that count against limits."""
        
        for endpoint in self.ENDPOINT_LIMITS:
            if path.startswith(endpoint):
                return True
        
        return False
    
    async def _get_tenant_id(self, request: Request) -> Optional[str]:
        """Extract tenant ID from request context."""
        
        # Try to get from user context
        try:
            # Check if user is already in request state (from auth middleware)
            if hasattr(request.state, "user") and request.state.user:
                # Handle both dict and User object
                if isinstance(request.state.user, dict):
                    return request.state.user.get('tenant_id') or get_current_tenant_id()
                else:
                    return getattr(request.state.user, 'tenant_id', None) or get_current_tenant_id()

            # Try to decode token ourselves
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                # This would normally decode JWT and extract tenant_id
                # DEV_ONLY_START
                # Development token for testing - removed in production builds
                if token == "dev-token-for-testing":
                    return get_current_tenant_id()
                # DEV_ONLY_END
        except Exception as e:
            logger.debug(f"Could not extract tenant_id: {e}")

        return None
    
    async def _check_feature_access(
        self,
        request: Request,
        tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """Check if tenant has access to requested feature."""
        
        path = request.url.path
        
        # Check each feature pattern
        for pattern, required_features in self.ENDPOINT_FEATURES.items():
            if pattern in path:
                async for db in get_db():
                    enforcer = LicenseEnforcer(db)
                    
                    for feature in required_features:
                        if not await enforcer.check_feature(tenant_id, feature):
                            tenant_info = await enforcer._get_tenant_info(tenant_id)
                            current_edition = Edition(tenant_info["edition"])
                            response = get_feature_upgrade_info(feature, current_edition)
                            response["upgrade_options"] = enforcer._get_upgrade_path(
                                tenant_info["edition"]
                            )
                            return response
                    break
        
        return None
    
    async def _check_resource_limits(
        self,
        request: Request,
        tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """Check if creating resource would exceed limits."""
        
        path = request.url.path
        
        # Determine limit type
        limit_type = None
        for endpoint, lt in self.ENDPOINT_LIMITS.items():
            if path.startswith(endpoint):
                limit_type = lt
                break
        
        if not limit_type:
            return None
        
        # Check limit
        async for db in get_db():
            enforcer = LicenseEnforcer(db)
            
            try:
                result = await enforcer.check_limit(
                    tenant_id=tenant_id,
                    limit_type=limit_type,
                    increment=1
                )
                
                # Log warnings
                if result.get("warning"):
                    logger.warning(f"Limit warning for {tenant_id}: {result['warning']}")
                
                # In soft mode, allow but return warning header
                if result.get("warning") and not result.get("allowed", True):
                    request.state.limit_warning = result["warning"]
                
            except LimitExceededException as e:
                # Re-raise to be caught by outer handler
                raise
            
            break
        
        return None
    
    async def _track_usage(
        self,
        request: Request,
        response: Response,
        tenant_id: str,
        start_time: float
    ):
        """Track API usage metrics."""
        
        try:
            async for db in get_db():
                tracker = await get_usage_tracker(db)
                
                await tracker.track_api_call(
                    tenant_id=tenant_id,
                    endpoint=request.url.path,
                    method=request.method,
                    response_time_ms=(time.time() - start_time) * 1000,
                    status_code=response.status_code,
                    metadata={
                        "query_params": dict(request.query_params),
                        "path_params": request.path_params
                    }
                )
                
                break
                
        except Exception as e:
            logger.error(f"Failed to track usage: {e}")
    
    async def _add_usage_headers(self, response: Response, tenant_id: str):
        """Add usage information headers to response."""
        
        try:
            # Get current usage for common limits
            async for db in get_db():
                enforcer = LicenseEnforcer(db)
                
                # Get API calls usage for current period
                api_result = await enforcer.check_limit(
                    tenant_id=tenant_id,
                    limit_type=LimitType.API_CALLS,
                    increment=0  # Just check, don't increment
                )
                
                if api_result:
                    response.headers["X-Usage-API-Calls"] = str(api_result.get("current", 0))
                    response.headers["X-Limit-API-Calls"] = str(api_result.get("limit", 0))
                    
                    # Add rate limit style headers
                    response.headers["X-RateLimit-Limit"] = str(api_result.get("limit", 0))
                    response.headers["X-RateLimit-Remaining"] = str(
                        max(0, api_result.get("limit", 0) - api_result.get("current", 0))
                    )
                
                break
                
        except Exception as e:
            logger.debug(f"Could not add usage headers: {e}")


class UpgradePromptMiddleware(BaseHTTPMiddleware):
    """Middleware to inject upgrade prompts based on usage patterns."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and potentially add upgrade prompts."""
        
        response = await call_next(request)
        
        # Only add prompts for successful responses
        if response.status_code >= 300:
            return response
        
        # Check if we have tenant context
        tenant_id = getattr(request.state, "tenant_id", None)
        if not tenant_id:
            return response
        
        # Check if there's a limit warning
        warning = getattr(request.state, "limit_warning", None)
        
        # For JSON responses, add upgrade prompt if approaching limits
        if warning and "application/json" in response.headers.get("content-type", ""):
            # This is a bit hacky but works for adding prompts
            # In production, would use proper response manipulation
            response.headers["X-Upgrade-Prompt"] = warning
            response.headers["X-Upgrade-URL"] = "/api/upgrade/options"
        
        return response


# Helper function to check limits before operations
async def check_operation_limit(
    tenant_id: str,
    limit_type: LimitType,
    increment: int = 1,
    db: Optional[Any] = None
) -> Dict[str, Any]:
    """Check if an operation would exceed limits."""
    
    if not db:
        async for db in get_db():
            break
    
    enforcer = LicenseEnforcer(db)
    return await enforcer.check_limit(
        tenant_id=tenant_id,
        limit_type=limit_type,
        increment=increment
    )


# Decorator for endpoint-level enforcement
def require_feature(feature: str):
    """Decorator to require a specific feature for an endpoint."""
    
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            tenant_id = getattr(request.state, "tenant_id", None)
            if not tenant_id:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            async for db in get_db():
                enforcer = LicenseEnforcer(db)

                if not await enforcer.check_feature(tenant_id, feature):
                    tenant_info = await enforcer._get_tenant_info(tenant_id)
                    current_edition = Edition(tenant_info["edition"])
                    detail = get_feature_upgrade_info(feature, current_edition)
                    detail["upgrade_options"] = enforcer._get_upgrade_path(
                        tenant_info["edition"]
                    )
                    raise HTTPException(status_code=403, detail=detail)

                break
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    
    return decorator