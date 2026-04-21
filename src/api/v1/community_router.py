"""
Community Edition API v1 router.

This contains only the endpoints that belong in the open source Community edition.
"""

from fastapi import APIRouter

from .endpoints import (
    auth,
    tasks_simple as tasks,
    tasks_mcp,
    workflows,
    workflow_templates,
    adapters,
    adapter_registry,  # New adapter registry endpoints
    adapter_config,   # New adapter configuration endpoints
    mcp,
    mcp_agent,  # MCP Agent Integration - Phase 5
    bridge,
    nlp,
    conversation,  # Multi-turn conversation endpoints
    channel_webhook,  # Channel-agnostic webhook endpoint
    channel_link,  # Channel account linking (auth for messaging platforms)
    file_upload,  # File upload for workflow processing
    websocket,
    health,
    users,
    upgrade,
    resource_pools,
    data_quality,
    usage,
    license,
    billing,  # Stripe billing integration (portal, invoices)
    iam,
    ai_agent_basic,
    agent_execution,  # New agent execution endpoints
    memory_basic,
    cache_basic,
    google_a2a_basic,
    mfa,
    knowledge,  # Knowledge service for intelligent assistant
    runtime_gateway_basic,  # Community runtime gateway (audit-only)
    canvas,  # A2UI Canvas (render, auto-detect, templates)
    marketplace,  # Marketplace (browse, search, install)
    personal_agent,  # Personal Agent Hub
    pricing,  # Public pricing/plans endpoint
    subscription_plans,  # Subscription plan listing for frontend
    oauth2_login,  # OAuth2 social login (delegates to Business if available)
)
from . import platform_integration

# Create the main API router for Community edition
api_router = APIRouter()

# Core endpoints
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(mfa.router, tags=["mfa"])  # MFA router already has /users prefix
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
api_router.include_router(tasks_mcp.router, prefix="/tasks", tags=["tasks", "mcp"])
api_router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
api_router.include_router(workflow_templates.router, tags=["workflow-templates"])  # Router already has prefix
api_router.include_router(adapters.router, prefix="/adapters", tags=["adapters"])
api_router.include_router(adapter_registry.router, prefix="/adapters", tags=["adapter-registry"])  # Registry endpoints
api_router.include_router(adapter_config.router, prefix="/adapter-config", tags=["adapter-config"])  # Configuration endpoints

# Knowledge service endpoints (Intelligent Assistant foundation)
api_router.include_router(knowledge.router, tags=["knowledge"])  # Router already has /knowledge prefix

# Integration endpoints
api_router.include_router(mcp.router, prefix="/mcp", tags=["mcp"])
api_router.include_router(bridge.router, prefix="/bridge", tags=["bridge"])
api_router.include_router(platform_integration.router, prefix="/platform-integration", tags=["platform-integration"])

# MCP Agent Integration endpoints (Phase 5 - first-class agent tools)
api_router.include_router(mcp_agent.router, tags=["mcp-agent"])  # Router already has /mcp/agent prefix

# MCP Server endpoints (expose AICtrlNet capabilities via MCP)
try:
    from .endpoints import mcp_server
    api_router.include_router(mcp_server.router, prefix="/mcp-server", tags=["mcp-server"])
except ImportError:
    # MCP server endpoints not available in this edition
    pass

# MCP LLM endpoints (expose LLM service via MCP for AI frameworks)
try:
    from mcp_server import llm_endpoints
    api_router.include_router(llm_endpoints.router, tags=["mcp-llm"])
except ImportError:
    # MCP LLM endpoints not available
    pass

# MCP Streamable HTTP transport (JSON-RPC 2.0 — Claude Code connects here)
from mcp_server.http_transport import router as mcp_transport_router
api_router.include_router(mcp_transport_router, tags=["mcp-transport"])

# API Key management endpoints (routes already include /api-keys prefix)
from .endpoints import api_keys
api_router.include_router(api_keys.router, tags=["api-keys"])

# AI/NLP endpoints
api_router.include_router(nlp.router, prefix="/nlp", tags=["nlp"])

# Multi-turn conversation endpoints (works alongside NLP for backward compatibility)
api_router.include_router(conversation.router, prefix="/conversation", tags=["conversation"])

# Channel-agnostic webhook endpoints (Slack, Telegram, WhatsApp, SMS, Discord)
api_router.include_router(channel_webhook.router, tags=["channels"])

# Channel account linking (authenticated users link their messaging identities)
api_router.include_router(channel_link.router, prefix="/channels", tags=["channels"])

# File upload endpoints
api_router.include_router(file_upload.router, prefix="/files", tags=["files"])

# LLM endpoints (unified LLM service)
from llm.api import endpoints as llm_endpoints
api_router.include_router(llm_endpoints.router, prefix="/llm", tags=["llm"])

# Resource pool endpoints
api_router.include_router(resource_pools.router, prefix="/resource-pools", tags=["resource-pools"])

# Data quality endpoints (ISO 25012)
api_router.include_router(data_quality.router, prefix="/quality", tags=["data-quality"])
# Also mount at /data-quality for backward compatibility
api_router.include_router(data_quality.router, prefix="/data-quality", tags=["data-quality"])

# Basic usage tracking endpoints (for upgrade prompts)
api_router.include_router(usage.router, prefix="/usage", tags=["usage"])

# License management endpoints
api_router.include_router(license.router, prefix="/license", tags=["license"])

# Billing endpoints (Stripe integration - portal, invoices)
api_router.include_router(billing.router, prefix="/billing", tags=["billing"])

# Internal Agent Messaging endpoints
api_router.include_router(iam.router, prefix="/iam", tags=["iam"])

# Basic AI, Memory, and Cache endpoints (Community features with limits)
api_router.include_router(ai_agent_basic.router, prefix="/ai-agent", tags=["ai-agent-basic"])
api_router.include_router(agent_execution.router, prefix="/agent-execution", tags=["agent-execution"])
api_router.include_router(memory_basic.router, prefix="/memory", tags=["memory-basic"])
api_router.include_router(cache_basic.router, prefix="/cache", tags=["cache-basic"])

# Google A2A Protocol endpoints (Community features - discovery only)
api_router.include_router(google_a2a_basic.router, prefix="/a2a", tags=["a2a-basic"])

# Community Runtime Gateway (audit-only mode)
api_router.include_router(runtime_gateway_basic.router, prefix="/runtime", tags=["runtime-gateway-basic"])

# A2UI Canvas endpoints
api_router.include_router(canvas.router, prefix="/canvas", tags=["canvas"])

# Credentials alias — re-mount platform-integration credentials at /credentials
from fastapi import Depends as _Depends
from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession
from core.database import get_db as _get_db
from core.dependencies import get_current_user_safe as _get_current_user_safe

_credentials_router = APIRouter(tags=["credentials"])

@_credentials_router.get("/")
async def list_credentials_alias(
    current_user=_Depends(_get_current_user_safe),
    db: _AsyncSession = _Depends(_get_db),
):
    """List credentials (alias for /platform-integration/credentials)."""
    from services.platform_credential_service import PlatformCredentialService
    from core.user_utils import get_safe_user_id
    if not current_user:
        return []
    user_id = get_safe_user_id(current_user)
    if not user_id:
        return []
    service = PlatformCredentialService(db)
    return await service.list_credentials(user_id=user_id)

api_router.include_router(_credentials_router, prefix="/credentials")

# Integrations alias — list available platform integration types
_integrations_router = APIRouter(tags=["integrations"])

@_integrations_router.get("/")
async def list_integrations(
    current_user=_Depends(_get_current_user_safe),
    db: _AsyncSession = _Depends(_get_db),
):
    """List available platform integrations."""
    from services.platform_adapters import PlatformAdapterService
    service = PlatformAdapterService(db)
    await service.sync_adapters_to_db()
    return await service.list_adapters(active_only=True)

api_router.include_router(_integrations_router, prefix="/integrations")

# Marketplace endpoints (Community: browse, search, install with limit)
api_router.include_router(marketplace.router, prefix="/marketplace", tags=["marketplace"])

# Personal Agent Hub endpoints
api_router.include_router(personal_agent.router, prefix="/personal-agent", tags=["personal-agent"])

# Pricing/plans endpoint (public)
api_router.include_router(pricing.router, prefix="/pricing", tags=["pricing"])

# Subscription plan listing (public — frontend subscriptionService.js uses /subscription/plans)
api_router.include_router(subscription_plans.router, prefix="/subscription", tags=["subscription"])

# OAuth2 social login (delegates to Business edition service if available)
api_router.include_router(oauth2_login.router, prefix="/oauth2", tags=["oauth2"])

# WebSocket endpoints
api_router.include_router(websocket.router, tags=["websocket"])

# Upgrade/billing endpoints
api_router.include_router(upgrade.router, prefix="/upgrade", tags=["upgrade", "billing"])

__all__ = ["api_router"]