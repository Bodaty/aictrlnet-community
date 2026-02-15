"""Tool Dispatcher for AICtrlNet Intelligent Assistant v4.

This module provides the core tool-calling infrastructure for the Intelligent Assistant.
It is part of the Community edition and provides 35 core tools.

Business and Enterprise editions extend this with additional tools.

Tool Categories (Community - 35 tools):
- Workflow Tools (6): create, discover, instantiate, list, execute, status
- Task Management (4): create, list, update, status
- Template Discovery (4): search, list, detail, categories
- Agent Management (5): create, list, status, update, delete
- Integration Tools (5): list, info, configure, execute, test - exposes 51+ adapters
- MCP Tools (4): list_servers, discover_tools, execute, configure
- Help/System (2): get_help, system_status
- API Introspection (3): list_api_endpoints, get_endpoint_detail, search_api_capabilities
"""

import logging
import time
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# Import from Community edition
from llm.models import ToolDefinition, ToolResult, ToolRecoveryStrategy, ChainResult

logger = logging.getLogger(__name__)


class Edition(str, Enum):
    """Edition enumeration for tool filtering."""
    COMMUNITY = "community"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


# =============================================================================
# CORE TOOLS REGISTRY (Community Edition - 35 tools)
# =============================================================================

CORE_TOOLS: Dict[str, ToolDefinition] = {
    # -------------------------------------------------------------------------
    # Workflow Tools (6)
    # -------------------------------------------------------------------------
    "create_workflow": ToolDefinition(
        name="create_workflow",
        description=(
            "Create a workflow to automate a business process. Use for any action/process request "
            "(route, send, process, notify, approve, schedule, monitor, etc.)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Workflow name derived from the user's request"},
                "description": {"type": "string", "description": "What the workflow should do"},
                "industry": {"type": "string", "description": "Industry context (optional)"}
            },
            "required": ["name"]
        },
        editions=["community", "business", "enterprise"],
        handler="workflow_service.create_workflow",
        requires_confirmation=True
    ),
    "discover_workflows": ToolDefinition(
        name="discover_workflows",
        description="Search and browse existing workflows using semantic matching",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "status": {"type": "string", "enum": ["active", "paused", "draft", "all"]},
                "limit": {"type": "integer", "default": 10}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="workflow_service.search_workflows"
    ),
    "instantiate_template": ToolDefinition(
        name="instantiate_template",
        description="Create a workflow from a discovered template with customization",
        parameters={
            "type": "object",
            "properties": {
                "template_id": {"type": "string", "description": "Template ID to instantiate"},
                "workflow_name": {"type": "string", "description": "Name for the new workflow"},
                "customizations": {"type": "object", "description": "Override default values"}
            },
            "required": ["template_id", "workflow_name"]
        },
        editions=["community", "business", "enterprise"],
        handler="template_service.instantiate_template",
        requires_confirmation=True
    ),
    "list_workflows": ToolDefinition(
        name="list_workflows",
        description="List user's existing workflows with filtering and sorting",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "paused", "draft", "all"]},
                "limit": {"type": "integer", "default": 10},
                "offset": {"type": "integer", "default": 0}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="workflow_service.list_workflows"
    ),
    "execute_workflow": ToolDefinition(
        name="execute_workflow",
        description="Execute a workflow manually or with provided inputs",
        parameters={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "description": "Workflow ID to execute"},
                "inputs": {"type": "object", "description": "Input parameters"},
                "async_execution": {"type": "boolean", "default": True}
            },
            "required": ["workflow_id"]
        },
        editions=["community", "business", "enterprise"],
        handler="workflow_service.execute_workflow",
        requires_confirmation=True,
        is_destructive=False
    ),
    "get_workflow_status": ToolDefinition(
        name="get_workflow_status",
        description="Get status and execution history of a workflow",
        parameters={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "description": "Workflow ID"},
                "include_history": {"type": "boolean", "default": False}
            },
            "required": ["workflow_id"]
        },
        editions=["community", "business", "enterprise"],
        handler="workflow_service.get_status"
    ),
    "update_workflow": ToolDefinition(
        name="update_workflow",
        description="Update an existing workflow's settings or configuration.",
        parameters={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "description": "Workflow ID to update (use the most recently created if not specified)"},
                "description": {"type": "string", "description": "New or updated description for the workflow"},
                "name": {"type": "string", "description": "New name for the workflow (optional)"},
                "settings": {"type": "object", "description": "Workflow settings to update"}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="workflow_service.update_workflow"
    ),

    # -------------------------------------------------------------------------
    # Task Management Tools (Internal - NOT exposed to conversational UI)
    # -------------------------------------------------------------------------
    # NOTE: create_task, list_tasks, update_task, get_task_status are INTERNAL tools
    # used for MCP and agent orchestration, NOT for user-facing conversational workflow creation.
    # For user requests to automate processes, use create_workflow instead.
    # These tools are kept for API compatibility but hidden from LLM tool selection.
    # -------------------------------------------------------------------------
    "list_tasks": ToolDefinition(
        name="list_tasks",
        description="[INTERNAL] List tasks - for API use only, not conversational UI",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "limit": {"type": "integer", "default": 20}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="task_service.list_tasks"
    ),
    "update_task": ToolDefinition(
        name="update_task",
        description="[INTERNAL] Update an existing task - for API use only",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to update"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]}
            },
            "required": ["task_id"]
        },
        editions=["community", "business", "enterprise"],
        handler="task_service.update_task"
    ),
    "get_task_status": ToolDefinition(
        name="get_task_status",
        description="[INTERNAL] Get task status - for API use only",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID"}
            },
            "required": ["task_id"]
        },
        editions=["community", "business", "enterprise"],
        handler="task_service.get_task"
    ),

    # -------------------------------------------------------------------------
    # Template Discovery Tools (4)
    # -------------------------------------------------------------------------
    "search_templates": ToolDefinition(
        name="search_templates",
        description="Search workflow templates using semantic matching",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "industry": {"type": "string", "description": "Filter by industry"},
                "category": {"type": "string", "description": "Filter by category"},
                "limit": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        },
        editions=["community", "business", "enterprise"],
        handler="template_service.search_templates"
    ),
    "list_templates": ToolDefinition(
        name="list_templates",
        description="List available workflow templates",
        parameters={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "industry": {"type": "string"},
                "limit": {"type": "integer", "default": 20}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="template_service.list_templates"
    ),
    "get_template_detail": ToolDefinition(
        name="get_template_detail",
        description="Get detailed information about a specific template",
        parameters={
            "type": "object",
            "properties": {
                "template_id": {"type": "string", "description": "Template ID"}
            },
            "required": ["template_id"]
        },
        editions=["community", "business", "enterprise"],
        handler="template_service.get_template"
    ),
    "list_template_categories": ToolDefinition(
        name="list_template_categories",
        description="List available template categories and industries",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="template_service.list_categories"
    ),

    # -------------------------------------------------------------------------
    # Agent Management Tools (5)
    # -------------------------------------------------------------------------
    "create_agent": ToolDefinition(
        name="create_agent",
        description="Create a new AI agent with specified capabilities",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Agent name"},
                "agent_type": {"type": "string", "description": "Type of agent"},
                "capabilities": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"}
            },
            "required": ["name", "agent_type"]
        },
        editions=["community", "business", "enterprise"],
        handler="agent_service.create_agent",
        requires_confirmation=True
    ),
    "list_agents": ToolDefinition(
        name="list_agents",
        description="List available AI agents",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "idle", "offline"]},
                "agent_type": {"type": "string"},
                "limit": {"type": "integer", "default": 20}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="agent_service.list_agents"
    ),
    "get_agent_status": ToolDefinition(
        name="get_agent_status",
        description="Get status and metrics for an agent",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID"},
                "include_metrics": {"type": "boolean", "default": False}
            },
            "required": ["agent_id"]
        },
        editions=["community", "business", "enterprise"],
        handler="agent_service.get_status"
    ),
    "update_agent": ToolDefinition(
        name="update_agent",
        description="Update agent configuration",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID"},
                "name": {"type": "string"},
                "capabilities": {"type": "array", "items": {"type": "string"}},
                "settings": {"type": "object"}
            },
            "required": ["agent_id"]
        },
        editions=["community", "business", "enterprise"],
        handler="agent_service.update_agent"
    ),
    "delete_agent": ToolDefinition(
        name="delete_agent",
        description="Delete an AI agent",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID to delete"}
            },
            "required": ["agent_id"]
        },
        editions=["community", "business", "enterprise"],
        handler="agent_service.delete_agent",
        requires_confirmation=True,
        is_destructive=True
    ),

    # -------------------------------------------------------------------------
    # MCP Tools (4)
    # -------------------------------------------------------------------------
    "list_mcp_servers": ToolDefinition(
        name="list_mcp_servers",
        description="List registered MCP servers with connection status",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["healthy", "unhealthy", "all"]},
                "capability": {"type": "string", "description": "Filter by capability"}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="mcp_service.list_servers"
    ),
    "discover_mcp_tools": ToolDefinition(
        name="discover_mcp_tools",
        description="Browse and search available tools across MCP servers",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you need to do"},
                "server": {"type": "string", "description": "Filter by server name"},
                "limit": {"type": "integer", "default": 10}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="mcp_service.discover_tools"
    ),
    "execute_mcp_tool": ToolDefinition(
        name="execute_mcp_tool",
        description="Execute a tool from a connected MCP server",
        parameters={
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "MCP server name"},
                "tool": {"type": "string", "description": "Tool name"},
                "arguments": {"type": "object", "description": "Tool arguments"}
            },
            "required": ["server", "tool", "arguments"]
        },
        editions=["community", "business", "enterprise"],
        handler="mcp_service.execute_tool",
        requires_confirmation=True
    ),
    "configure_mcp_server": ToolDefinition(
        name="configure_mcp_server",
        description="Register or update an MCP server configuration",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["register", "update", "remove"]},
                "name": {"type": "string", "description": "Server name"},
                "url": {"type": "string", "description": "Server URL"},
                "api_key": {"type": "string", "description": "API key (optional)"},
                "settings": {"type": "object"}
            },
            "required": ["action", "name"]
        },
        editions=["community", "business", "enterprise"],
        handler="mcp_service.configure_server",
        requires_confirmation=True
    ),

    # -------------------------------------------------------------------------
    # Integration/Adapter Tools (5) - Exposes 51+ adapters to LLM
    # -------------------------------------------------------------------------
    "list_integrations": ToolDefinition(
        name="list_integrations",
        description="List available integrations/adapters with their capabilities and status",
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category",
                    "enum": ["ai_models", "communication", "crm", "payment", "databases", "human_services", "ai_agents"]
                },
                "status": {
                    "type": "string",
                    "description": "Filter by configuration status",
                    "enum": ["configured", "available", "all"]
                },
                "limit": {"type": "integer", "default": 20}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="adapter_service.list_integrations"
    ),
    "get_integration_info": ToolDefinition(
        name="get_integration_info",
        description="Get detailed information about a specific integration (Slack, Stripe, OpenAI, etc.)",
        parameters={
            "type": "object",
            "properties": {
                "adapter_name": {
                    "type": "string",
                    "description": "Name of the adapter (e.g., 'slack', 'stripe', 'openai', 'salesforce')"
                },
                "include_actions": {
                    "type": "boolean",
                    "description": "Include available actions/capabilities",
                    "default": True
                }
            },
            "required": ["adapter_name"]
        },
        editions=["community", "business", "enterprise"],
        handler="adapter_service.get_integration_info"
    ),
    "configure_integration": ToolDefinition(
        name="configure_integration",
        description="Configure an integration with credentials and settings",
        parameters={
            "type": "object",
            "properties": {
                "adapter_name": {
                    "type": "string",
                    "description": "Name of the adapter to configure"
                },
                "credentials": {
                    "type": "object",
                    "description": "Credentials (API keys, tokens, etc.)"
                },
                "settings": {
                    "type": "object",
                    "description": "Additional configuration settings"
                }
            },
            "required": ["adapter_name"]
        },
        editions=["community", "business", "enterprise"],
        handler="adapter_service.configure_integration",
        requires_confirmation=True
    ),
    "execute_integration": ToolDefinition(
        name="execute_integration",
        description="Execute an action via an integration (send email, create lead, charge payment, etc.)",
        parameters={
            "type": "object",
            "properties": {
                "adapter_name": {
                    "type": "string",
                    "description": "Name of the adapter (e.g., 'slack', 'email', 'stripe')"
                },
                "action": {
                    "type": "string",
                    "description": "Action to perform (e.g., 'send_message', 'create_lead', 'charge')"
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the action"
                }
            },
            "required": ["adapter_name", "action"]
        },
        editions=["community", "business", "enterprise"],
        handler="adapter_service.execute_integration",
        requires_confirmation=True
    ),
    "test_integration": ToolDefinition(
        name="test_integration",
        description="Test that an integration is properly configured and working",
        parameters={
            "type": "object",
            "properties": {
                "adapter_name": {
                    "type": "string",
                    "description": "Name of the adapter to test"
                }
            },
            "required": ["adapter_name"]
        },
        editions=["community", "business", "enterprise"],
        handler="adapter_service.test_integration"
    ),

    # -------------------------------------------------------------------------
    # Help/System Tools (2)
    # -------------------------------------------------------------------------
    "get_help": ToolDefinition(
        name="get_help",
        description="Get contextual help for features, tools, or concepts",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Help topic"},
                "context": {"type": "string", "description": "Current context"}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="help_service.get_help"
    ),
    "get_system_status": ToolDefinition(
        name="get_system_status",
        description="Get overall system health and status",
        parameters={
            "type": "object",
            "properties": {
                "include_services": {"type": "boolean", "default": False}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="system_service.get_status"
    ),

    # -------------------------------------------------------------------------
    # Live System Context Tools (2) — Phase C
    # -------------------------------------------------------------------------
    "get_platform_metrics": ToolDefinition(
        name="get_platform_metrics",
        description="Get platform-wide metrics: workflow count, success rate, active users.",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="system_service.get_platform_metrics"
    ),
    "get_recent_activity": ToolDefinition(
        name="get_recent_activity",
        description="Get recent platform activity: workflow runs, errors, and user actions.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10, "description": "Max events to return"}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="system_service.get_recent_activity"
    ),

    # -------------------------------------------------------------------------
    # API Introspection Tools (3) — Phase E
    # -------------------------------------------------------------------------
    "list_api_endpoints": ToolDefinition(
        name="list_api_endpoints",
        description="List all API endpoints. Filter by method, path prefix, or tag.",
        parameters={
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
                "path_prefix": {"type": "string", "description": "Filter by path prefix, e.g. '/api/v1/workflows'"},
                "tag": {"type": "string", "description": "Filter by OpenAPI tag"},
                "limit": {"type": "integer", "default": 20}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="api_introspection_service.list_endpoints"
    ),
    "get_endpoint_detail": ToolDefinition(
        name="get_endpoint_detail",
        description="Get full detail for a specific API endpoint (schema, parameters).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "e.g. '/api/v1/workflows'"},
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"], "default": "GET"}
            },
            "required": ["path"]
        },
        editions=["community", "business", "enterprise"],
        handler="api_introspection_service.get_endpoint_detail"
    ),
    "search_api_capabilities": ToolDefinition(
        name="search_api_capabilities",
        description="Search API capabilities by natural language query.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["query"]
        },
        editions=["community", "business", "enterprise"],
        handler="api_introspection_service.search_capabilities"
    ),

    # -------------------------------------------------------------------------
    # File Access Tools (2) - Agent can read staged files
    # -------------------------------------------------------------------------
    "list_user_files": ToolDefinition(
        name="list_user_files",
        description="List the current user's staged/uploaded files. Returns file IDs, names, types, and sizes.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20, "description": "Max files to return"}
            },
            "required": []
        },
        editions=["community", "business", "enterprise"],
        handler="file_access.list_user_files"
    ),
    "access_staged_file": ToolDefinition(
        name="access_staged_file",
        description="Read the extracted data from a staged file by its ID. Returns structured content (text, tables, metadata).",
        parameters={
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "The staged file ID to access"}
            },
            "required": ["file_id"]
        },
        editions=["community", "business", "enterprise"],
        handler="file_access.access_staged_file"
    ),

    # -------------------------------------------------------------------------
    # Browser Automation Tools (3) - Agent can drive a headless browser
    # -------------------------------------------------------------------------
    "browser_execute": ToolDefinition(
        name="browser_execute",
        description=(
            "Execute a sequence of browser actions (navigate, click, fill, screenshot, extract_text, "
            "run_script, wait_for, download). Max 10 actions per call."
        ),
        parameters={
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": "Ordered list of browser actions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["navigate", "click", "fill", "screenshot", "extract_text", "run_script", "wait_for", "download"]},
                            "url": {"type": "string"},
                            "selector": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["type"]
                    }
                },
                "timeout_ms": {"type": "integer", "default": 30000, "description": "Max timeout in ms (max 60000)"}
            },
            "required": ["actions"]
        },
        editions=["community", "business", "enterprise"],
        handler="browser_tool.execute",
        requires_confirmation=True
    ),
    "browser_screenshot": ToolDefinition(
        name="browser_screenshot",
        description="Navigate to a URL and take a screenshot. Convenience wrapper around browser_execute.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to navigate to and screenshot"}
            },
            "required": ["url"]
        },
        editions=["community", "business", "enterprise"],
        handler="browser_tool.screenshot"
    ),
    "browser_extract": ToolDefinition(
        name="browser_extract",
        description="Navigate to a URL and extract text content from one or more CSS selectors.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to navigate to"},
                "selectors": {
                    "type": "array",
                    "description": "CSS selectors to extract text from",
                    "items": {"type": "string"}
                }
            },
            "required": ["url"]
        },
        editions=["community", "business", "enterprise"],
        handler="browser_tool.extract"
    ),
}


def get_tools_for_edition(edition: Edition, include_internal: bool = False) -> List[ToolDefinition]:
    """Get all tools available for the given edition.

    This returns only CORE_TOOLS for Community edition.
    Business and Enterprise editions extend this with additional tools.

    Args:
        edition: The edition to get tools for
        include_internal: If False (default), excludes tools marked as [INTERNAL]
                         These are API-only tools not meant for conversational UI.
    """
    edition_value = edition.value if isinstance(edition, Edition) else edition
    tools = [
        tool for tool in CORE_TOOLS.values()
        if edition_value in tool.editions
    ]

    # Filter out internal tools unless explicitly requested
    if not include_internal:
        tools = [t for t in tools if "[INTERNAL]" not in t.description]

    return tools


def get_tool_counts() -> Dict[str, int]:
    """Get tool counts by edition."""
    return {
        "community": len([t for t in CORE_TOOLS.values() if "community" in t.editions]),
        "business": len([t for t in CORE_TOOLS.values() if "business" in t.editions]),
        "enterprise": len([t for t in CORE_TOOLS.values() if "enterprise" in t.editions]),
    }


class ToolDispatcher:
    """Base tool dispatcher for Community edition.

    Routes tool calls to appropriate service methods.
    Business and Enterprise editions extend this class to add
    edition-specific tools and ML-enhanced features.
    """

    # Map tool names to their handler method names
    TOOL_HANDLER_MAP: Dict[str, str] = {
        # Workflow tools
        "create_workflow": "_create_workflow",
        "discover_workflows": "_discover_workflows",
        "list_workflows": "_list_workflows",
        "execute_workflow": "_execute_workflow",
        "get_workflow_status": "_get_workflow_status",
        "instantiate_template": "_instantiate_template",
        "update_workflow": "_update_workflow",
        "configure_workflow": "_update_workflow",  # Alias for update_workflow
        # Task tools
        "create_task": "_create_task",
        "list_tasks": "_list_tasks",
        "update_task": "_update_task",
        "get_task_status": "_get_task_status",
        # Template tools
        "search_templates": "_search_templates",
        "list_templates": "_list_templates",
        "get_template_detail": "_get_template_detail",
        "list_template_categories": "_list_template_categories",
        # Agent tools
        "create_agent": "_create_agent",
        "list_agents": "_list_agents",
        "get_agent_status": "_get_agent_status",
        "update_agent": "_update_agent",
        "delete_agent": "_delete_agent",
        # MCP tools
        "list_mcp_servers": "_list_mcp_servers",
        "discover_mcp_tools": "_discover_mcp_tools",
        "execute_mcp_tool": "_execute_mcp_tool",
        "configure_mcp_server": "_configure_mcp_server",
        # Integration/Adapter tools
        "list_integrations": "_list_integrations",
        "get_integration_info": "_get_integration_info",
        "configure_integration": "_configure_integration",
        "execute_integration": "_execute_integration",
        "test_integration": "_test_integration",
        # Help/System tools
        "get_help": "_get_help",
        "get_system_status": "_get_system_status",
        # Live System Context (Phase C)
        "get_platform_metrics": "_get_platform_metrics",
        "get_recent_activity": "_get_recent_activity",
        # API Introspection (Phase E)
        "list_api_endpoints": "_list_api_endpoints",
        "get_endpoint_detail": "_get_endpoint_detail",
        "search_api_capabilities": "_search_api_capabilities",
        # File Access tools
        "list_user_files": "_list_user_files",
        "access_staged_file": "_access_staged_file",
        # Browser Automation tools
        "browser_execute": "_browser_execute",
        "browser_screenshot": "_browser_screenshot",
        "browser_extract": "_browser_extract",
    }

    def __init__(self, db: AsyncSession, edition: Edition = Edition.COMMUNITY):
        """Initialize the tool dispatcher.

        Args:
            db: Database session for service instantiation
            edition: Current edition (affects available tools)
        """
        self.db = db
        self.edition = edition
        self._services: Dict[str, Any] = {}
        self._initialized = False

        logger.info(f"[v4] ToolDispatcher initialized for {edition.value} edition")

        # Validate tool handlers at initialization
        self._validate_tool_handlers()

    def _lazy_load_services(self) -> None:
        """Lazy load services on first use."""
        if self._initialized:
            return

        try:
            # Import and instantiate services
            from services.workflow_service import WorkflowService
            from services.task_service import TaskService
            from services.nlp import NLPService
            from services.workflow_execution import WorkflowExecutionService

            self._services['workflow_service'] = WorkflowService(self.db)
            self._services['task_service'] = TaskService(self.db)
            self._services['nlp_service'] = NLPService(self.db)
            self._services['workflow_execution_service'] = WorkflowExecutionService(self.db)

            # Optional services (may not exist in all configurations)
            try:
                from services.mcp_unified import UnifiedMCPService
                self._services['mcp_service'] = UnifiedMCPService()
            except ImportError:
                logger.warning("[v4] MCP service not available")

            # Adapter service for integration tools
            try:
                from services.adapter import AdapterService
                self._services['adapter_service'] = AdapterService(self.db)
            except ImportError:
                logger.warning("[v4] Adapter service not available")

            # API Introspection service (Phase E)
            try:
                from services.knowledge.api_introspection_service import ApiIntrospectionService
                self._services['api_introspection_service'] = ApiIntrospectionService()
            except ImportError:
                logger.warning("[v4] API Introspection service not available")

            self._initialized = True
            logger.info(f"[v4] Services loaded: {list(self._services.keys())}")

        except Exception as e:
            logger.error(f"[v4] Error loading services: {e}")
            raise

    def _validate_tool_handlers(self) -> None:
        """Validate that all registered tools have corresponding handler methods.

        This runs at startup to catch missing handler methods early, preventing
        runtime AttributeError when tools are invoked.
        """
        missing_handlers = []
        tools_registry = self._get_tools_registry()

        for tool_name in tools_registry.keys():
            handler_name = self.TOOL_HANDLER_MAP.get(tool_name)

            if not handler_name:
                # Tool exists but no handler mapping
                missing_handlers.append({
                    "tool": tool_name,
                    "issue": "no_handler_mapping",
                    "message": f"Tool '{tool_name}' has no entry in TOOL_HANDLER_MAP"
                })
                continue

            # Check if handler method exists on this class
            if not hasattr(self, handler_name):
                missing_handlers.append({
                    "tool": tool_name,
                    "handler": handler_name,
                    "issue": "missing_method",
                    "message": f"Tool '{tool_name}' handler method '{handler_name}' not found"
                })

        if missing_handlers:
            logger.error(f"[v4] TOOL VALIDATION FAILED - {len(missing_handlers)} missing handlers:")
            for item in missing_handlers:
                logger.error(f"[v4]   - {item['message']}")

            # In development, we want this to be very visible
            # Log a summary that will stand out
            logger.error(
                f"[v4] ⚠️ CRITICAL: {len(missing_handlers)} tools have no handler implementation! "
                f"Tools will fail at runtime. Missing: {[m['tool'] for m in missing_handlers]}"
            )
        else:
            logger.info(f"[v4] ✓ All {len(tools_registry)} tool handlers validated successfully")

    def _get_tools_registry(self) -> Dict[str, Any]:
        """Get the tools registry for this dispatcher.

        Override in subclasses to include edition-specific tools.
        """
        return CORE_TOOLS

    def get_available_tools(self) -> List[ToolDefinition]:
        """Get tools available for the current edition."""
        return get_tools_for_edition(self.edition)

    async def invoke(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Invoke a tool by name.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Tool arguments
            user_id: User ID for authorization
            context: Additional context (session info, etc.)

        Returns:
            ToolResult with success/failure and data
        """
        self._lazy_load_services()
        start_time = time.time()

        # Check if tool exists
        tool = CORE_TOOLS.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
                error_type="not_found"
            )

        # Check edition access
        if self.edition.value not in tool.editions:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not available in {self.edition.value} edition",
                error_type="edition_restriction"
            )

        logger.info(f"[v4] Invoking tool: {tool_name}")

        try:
            # Re-set tenant context on DB session before tool execution.
            # Intermediate commits in the conversation flow can release the connection
            # back to the pool; a new connection won't have app.current_tenant_id set,
            # causing RLS policy violations on INSERT/UPDATE.
            try:
                from core.tenant_context import get_current_tenant_id
                tenant_id = get_current_tenant_id()
                if tenant_id:
                    await self.db.execute(
                        text("SELECT set_config('app.current_tenant_id', :tid, false)"),
                        {"tid": tenant_id}
                    )
            except Exception as e:
                logger.warning(f"[v4] Could not re-set tenant context: {e}")

            # Route to appropriate handler
            result = await self._route_tool_call(tool_name, arguments, user_id, context)

            execution_time = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time

            logger.info(f"[v4] Tool {tool_name} completed in {execution_time}ms")
            return result

        except Exception as e:
            logger.error(f"[v4] Tool {tool_name} failed: {e}")
            # Rollback the session to clear any failed transaction state.
            # Without this, subsequent DB operations (e.g. _store_message)
            # will fail with PendingRollbackError.
            try:
                await self.db.rollback()
            except Exception:
                pass
            return ToolResult(
                success=False,
                error=str(e),
                error_type="execution_error",
                recovery_strategy=ToolRecoveryStrategy.RETRY,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

    async def _route_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> ToolResult:
        """Route tool call to appropriate service method.

        This follows the ActionOrchestrator pattern of explicit routing.
        """
        # Workflow tools
        if tool_name == "create_workflow":
            return await self._create_workflow(arguments, user_id, context)
        elif tool_name == "discover_workflows":
            return await self._discover_workflows(arguments, user_id)
        elif tool_name == "list_workflows":
            return await self._list_workflows(arguments, user_id)
        elif tool_name == "execute_workflow":
            return await self._execute_workflow(arguments, user_id)
        elif tool_name == "get_workflow_status":
            return await self._get_workflow_status(arguments)
        elif tool_name == "instantiate_template":
            return await self._instantiate_template(arguments, user_id)
        elif tool_name == "update_workflow":
            return await self._update_workflow(arguments, user_id, context)

        # Task tools
        elif tool_name == "create_task":
            return await self._create_task(arguments, user_id)
        elif tool_name == "list_tasks":
            return await self._list_tasks(arguments, user_id)
        elif tool_name == "update_task":
            return await self._update_task(arguments, user_id)
        elif tool_name == "get_task_status":
            return await self._get_task_status(arguments)

        # Template tools
        elif tool_name == "search_templates":
            return await self._search_templates(arguments)
        elif tool_name == "list_templates":
            return await self._list_templates(arguments)
        elif tool_name == "get_template_detail":
            return await self._get_template_detail(arguments)
        elif tool_name == "list_template_categories":
            return await self._list_template_categories()

        # Agent tools
        elif tool_name == "create_agent":
            return await self._create_agent(arguments, user_id)
        elif tool_name == "list_agents":
            return await self._list_agents(arguments, user_id)
        elif tool_name == "get_agent_status":
            return await self._get_agent_status(arguments)
        elif tool_name == "update_agent":
            return await self._update_agent(arguments, user_id)
        elif tool_name == "delete_agent":
            return await self._delete_agent(arguments, user_id)

        # MCP tools
        elif tool_name == "list_mcp_servers":
            return await self._list_mcp_servers(arguments)
        elif tool_name == "discover_mcp_tools":
            return await self._discover_mcp_tools(arguments)
        elif tool_name == "execute_mcp_tool":
            return await self._execute_mcp_tool(arguments, user_id)
        elif tool_name == "configure_mcp_server":
            return await self._configure_mcp_server(arguments, user_id)

        # Integration/Adapter tools
        elif tool_name == "list_integrations":
            return await self._list_integrations(arguments)
        elif tool_name == "get_integration_info":
            return await self._get_integration_info(arguments)
        elif tool_name == "configure_integration":
            return await self._configure_integration(arguments, user_id)
        elif tool_name == "execute_integration":
            return await self._execute_integration(arguments, user_id)
        elif tool_name == "test_integration":
            return await self._test_integration(arguments)

        # Help/System tools
        elif tool_name == "get_help":
            return await self._get_help(arguments)
        elif tool_name == "get_system_status":
            return await self._get_system_status(arguments)

        # Live System Context tools (Phase C)
        elif tool_name == "get_platform_metrics":
            return await self._get_platform_metrics(arguments)
        elif tool_name == "get_recent_activity":
            return await self._get_recent_activity(arguments)

        # API Introspection tools (Phase E)
        elif tool_name == "list_api_endpoints":
            return await self._list_api_endpoints(arguments)
        elif tool_name == "get_endpoint_detail":
            return await self._get_endpoint_detail(arguments)
        elif tool_name == "search_api_capabilities":
            return await self._search_api_capabilities(arguments)

        # File access tools
        elif tool_name == "list_user_files":
            return await self._list_user_files(arguments, user_id)
        elif tool_name == "access_staged_file":
            return await self._access_staged_file(arguments, user_id)

        # Browser automation tools
        elif tool_name == "browser_execute":
            return await self._browser_execute(arguments, user_id)
        elif tool_name == "browser_screenshot":
            return await self._browser_screenshot(arguments, user_id)
        elif tool_name == "browser_extract":
            return await self._browser_extract(arguments, user_id)

        else:
            return ToolResult(
                success=False,
                error=f"No handler for tool: {tool_name}",
                error_type="not_implemented"
            )

    # =========================================================================
    # Tool Handler Methods
    # =========================================================================

    async def _create_workflow(self, args: Dict, user_id: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Create a workflow using NLP service.

        Per WORKFLOW_GENERATION_FIX_IMPLEMENTATION.md Change 1.2:
        Now accepts context with templates for domain-aware generation.
        """
        import re

        nlp_service = self._services.get('nlp_service')
        if not nlp_service:
            return ToolResult(success=False, error="NLP service not available")

        try:
            # Clean up the name in case LLM passed verbose instructions
            name = args.get('name', '')

            # Use regex to remove common instructional prefixes
            # Pattern matches: "Create a/A workflow named 'X'" or similar
            prefixes_pattern = r"^(?:Create\s+)?(?:a\s+)?(?:A\s+)?[Ww]orkflow\s+(?:[Nn]amed|[Cc]alled)\s+['\"]?"
            name = re.sub(prefixes_pattern, '', name, flags=re.IGNORECASE)

            # Strip leading/trailing quotes and whitespace
            name = name.strip("'\" \t")

            # CRITICAL: Reject generic/placeholder names - force user to provide actual name
            generic_names = {'new workflow', 'workflow', 'new', 'untitled', 'test', ''}
            if len(name) < 3 or name.lower() in generic_names:
                return ToolResult(
                    success=False,
                    error="missing_name",
                    data={
                        "message": "Please provide a specific name for your workflow. What would you like to call it?",
                        "needs_clarification": True,
                        "field": "name"
                    }
                )

            description = args.get('description', '')

            prompt = f"Create a workflow named '{name}' that {description}"
            if args.get('industry'):
                prompt += f" for the {args['industry']} industry"

            # Build NLP context with templates for domain-aware generation
            # Per WORKFLOW_GENERATION_FIX_IMPLEMENTATION.md Change 1.2
            nlp_context = {"user_id": user_id}
            if context:
                # Pass templates if available for domain guidance
                if context.get('templates'):
                    nlp_context['templates'] = context['templates']
                    # Extract domain hints from template names/descriptions
                    template_hints = [t.get('name', '') + ': ' + t.get('description', '')[:50]
                                     for t in context['templates'][:3]]
                    nlp_context['domain_hints'] = template_hints

            result = await nlp_service.process_natural_language(
                prompt=prompt,
                context=nlp_context
            )

            # Get the actual workflow name from result, fallback to cleaned name
            actual_name = result.get('workflow', {}).get('name', name)

            return ToolResult(
                success=True,
                data={
                    "workflow_id": str(result.get('workflow', {}).get('id', '')),
                    "workflow_name": actual_name,
                    "message": f"Created workflow '{actual_name}' successfully"
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _discover_workflows(self, args: Dict, user_id: str) -> ToolResult:
        """Search workflows."""
        workflow_service = self._services.get('workflow_service')
        if not workflow_service:
            return ToolResult(success=False, error="Workflow service not available")

        try:
            # Build filters from search parameters
            filters = {}
            if args.get('query'):
                filters['name'] = args.get('query')
            if args.get('status'):
                filters['status'] = args.get('status')

            workflows = await workflow_service.list_workflows(
                filters=filters,
                limit=args.get('limit', 10),
                offset=0
            )

            return ToolResult(
                success=True,
                data={
                    "workflows": [{"id": str(w.id), "name": w.name, "status": getattr(w, 'status', 'active')} for w in workflows],
                    "count": len(workflows),
                    "query": args.get('query', '')
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _list_workflows(self, args: Dict, user_id: str) -> ToolResult:
        """List user's workflows."""
        workflow_service = self._services.get('workflow_service')
        if not workflow_service:
            return ToolResult(success=False, error="Workflow service not available")

        try:
            # Build filters from parameters
            filters = {}
            if args.get('status'):
                filters['status'] = args.get('status')

            workflows = await workflow_service.list_workflows(
                filters=filters,
                limit=args.get('limit', 10),
                offset=args.get('offset', 0)
            )

            return ToolResult(
                success=True,
                data={
                    "workflows": [{"id": str(w.id), "name": w.name, "status": getattr(w, 'status', 'active')} for w in workflows],
                    "count": len(workflows)
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _update_workflow(
        self, args: Dict, user_id: str, context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Update an existing workflow's description, settings, or configuration.

        v5 spec requirement: Turn 3+ follow-up actions - users should be able
        to refine and update workflows they just created.
        """
        workflow_service = self._services.get('workflow_service')
        if not workflow_service:
            return ToolResult(success=False, error="Workflow service not available")

        try:
            # Get workflow_id - either from args or from context (last created workflow)
            workflow_id = args.get('workflow_id')

            # Check if workflow_id is a truncated/partial ID (contains ... or is too short)
            is_truncated = workflow_id and ('...' in str(workflow_id) or len(str(workflow_id)) < 32)

            if (not workflow_id or is_truncated) and context:
                # Try to get from session context v5_parameters
                params = context.get('v5_parameters', {})
                context_workflow_id = params.get('workflow_id') or params.get('last_created_workflow_id')

                if context_workflow_id:
                    # If we have a truncated ID, verify it matches the context workflow
                    if is_truncated and workflow_id:
                        # Extract the non-truncated portion and check if it matches
                        truncated_portion = str(workflow_id).replace('...', '').strip()
                        if str(context_workflow_id).startswith(truncated_portion):
                            workflow_id = context_workflow_id
                            logger.info(f"[v4] update_workflow resolved truncated ID '{args.get('workflow_id')}' to full ID: {workflow_id}")
                        else:
                            logger.warning(f"[v4] Truncated ID '{truncated_portion}' doesn't match context workflow '{context_workflow_id}'")
                    else:
                        workflow_id = context_workflow_id
                        logger.info(f"[v4] update_workflow using workflow_id from context: {workflow_id}")

            if not workflow_id:
                return ToolResult(
                    success=False,
                    error="No workflow specified. Please specify which workflow to update.",
                    data={"needs_clarification": True, "field": "workflow_id"}
                )

            # Build metadata update
            metadata_updates = {}
            description_val = args.get('description', '') or ''
            name_val = args.get('name', '') or ''

            # Log what the LLM is trying to set
            logger.info(f"[v4] update_workflow args: description='{description_val[:50] if description_val else 'None'}...', name='{name_val}'")

            # Get the existing workflow name from context for comparison
            v5_params = context.get('v5_parameters', {}) if context else {}
            context_workflow_name = v5_params.get('workflow_name', '') or v5_params.get('last_created_workflow', '') or ''

            # Check if this looks like a "configure" request without real user input:
            # 1. No description provided AND name matches existing workflow name
            # 2. Description is auto-generated/generic
            needs_user_input = False

            # Case 1: No description and just renaming to same/similar name
            if not description_val.strip():
                if not name_val.strip() or name_val.lower().strip() == context_workflow_name.lower().strip():
                    needs_user_input = True
                    logger.info(f"[v4] update_workflow: no description and name unchanged, asking for user input")

            # Case 2: Description is auto-generated (generic workflow terms)
            if description_val:
                auto_generated_patterns = [
                    'customer onboarding', 'email marketing', 'marketing workflow',
                    'onboarding workflow', 'workflow for', 'automated workflow',
                    'this workflow', 'a workflow'
                ]
                desc_lower = description_val.lower()
                for pattern in auto_generated_patterns:
                    if pattern in desc_lower and len(description_val) < 100:
                        needs_user_input = True
                        logger.info(f"[v4] update_workflow detected auto-generated description pattern: {pattern}")
                        break

            # Case 3: Description just echoes the name
            if description_val and name_val and description_val.lower().strip() == name_val.lower().strip():
                needs_user_input = True
                logger.info(f"[v4] update_workflow: description just echoes name, asking for user input")

            # If we need actual user input, ask for clarification
            if needs_user_input:
                return ToolResult(
                    success=False,
                    error="What description would you like to add to this workflow?",
                    data={
                        "needs_clarification": True,
                        "field": "description",
                        "prompt": "Please tell me what description you'd like to add to this workflow.",
                        "workflow_id": workflow_id,
                        "ask_user": True
                    }
                )

            if description_val.strip():
                metadata_updates['description'] = description_val
            if name_val.strip() and name_val.lower().strip() != context_workflow_name.lower().strip():
                metadata_updates['name'] = name_val
            if args.get('settings'):
                metadata_updates['settings'] = args.get('settings')

            # If no update values provided, ask for clarification
            if not metadata_updates:
                return ToolResult(
                    success=False,
                    error="What would you like to update? Please provide a description or name for the workflow.",
                    data={
                        "needs_clarification": True,
                        "field": "description",
                        "prompt": "What description would you like to add to this workflow?",
                        "workflow_id": workflow_id
                    }
                )

            # Call workflow service to update
            workflow = await workflow_service.update_workflow(
                workflow_id=workflow_id,
                metadata=metadata_updates
            )

            # Also try to update name if provided (some models store name separately)
            if args.get('name') and hasattr(workflow, 'name'):
                workflow.name = args.get('name')
                await self.db.commit()
                await self.db.refresh(workflow)

            return ToolResult(
                success=True,
                data={
                    "workflow_id": str(workflow.id),
                    "workflow_name": getattr(workflow, 'name', 'Unknown'),
                    "updated_fields": list(metadata_updates.keys()),
                    "message": f"Updated workflow '{getattr(workflow, 'name', workflow_id)}' successfully",
                    "tool_name": "update_workflow"
                }
            )
        except Exception as e:
            logger.error(f"[v4] update_workflow failed: {e}")
            return ToolResult(success=False, error=str(e))

    async def _execute_workflow(self, args: Dict, user_id: str) -> ToolResult:
        """Execute a workflow using the real execution engine."""
        workflow_exec_service = self._services.get('workflow_execution_service')
        if not workflow_exec_service:
            return ToolResult(success=False, error="Workflow execution service not available")

        try:
            execution = await workflow_exec_service.create_execution(
                workflow_id=args['workflow_id'],
                input_data=args.get('inputs', {}),
                triggered_by="conversation",
                trigger_metadata={"user_id": user_id}
            )

            started = await workflow_exec_service.start_execution(execution.id)

            return ToolResult(
                success=True,
                data={
                    "execution_id": str(started.id),
                    "status": started.status,
                    "message": "Workflow execution started"
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _get_workflow_status(self, args: Dict) -> ToolResult:
        """Get workflow status."""
        workflow_service = self._services.get('workflow_service')
        if not workflow_service:
            return ToolResult(success=False, error="Workflow service not available")

        try:
            workflow = await workflow_service.get_workflow(args['workflow_id'])
            if not workflow:
                return ToolResult(success=False, error="Workflow not found")

            return ToolResult(
                success=True,
                data={
                    "workflow_id": str(workflow.id),
                    "name": workflow.name,
                    "status": workflow.status,
                    "created_at": str(workflow.created_at)
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _instantiate_template(self, args: Dict, user_id: str) -> ToolResult:
        """Instantiate a template."""
        # TODO: Implement template instantiation
        return ToolResult(
            success=True,
            data={
                "message": f"Template instantiation for '{args.get('workflow_name')}' would be created here",
                "template_id": args.get('template_id')
            }
        )

    async def _create_task(self, args: Dict, user_id: str) -> ToolResult:
        """Create a task."""
        task_service = self._services.get('task_service')
        if not task_service:
            return ToolResult(success=False, error="Task service not available")

        try:
            # Map tool parameters to service parameters
            # Tool uses 'title', service expects 'name'
            task_name = args.get('title') or args.get('name', 'Untitled Task')

            # Store priority and user_id in metadata
            task_metadata = {
                "priority": args.get('priority', 'medium'),
                "user_id": user_id,
                "due_date": args.get('due_date')
            }

            task = await task_service.create_task(
                name=task_name,
                description=args.get('description', ''),
                metadata=task_metadata
            )

            return ToolResult(
                success=True,
                data={
                    "task_id": str(task.id),
                    "title": task.name,
                    "message": f"Task '{task.name}' created"
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _list_tasks(self, args: Dict, user_id: str) -> ToolResult:
        """List tasks."""
        task_service = self._services.get('task_service')
        if not task_service:
            return ToolResult(success=False, error="Task service not available")

        try:
            # Build filters from args - TaskService.list_tasks expects filters dict
            filters = {}
            if args.get('status'):
                filters['status'] = args.get('status')
            # Store user_id filter if needed
            filters['metadata.user_id'] = user_id

            tasks = await task_service.list_tasks(
                filters=filters,
                limit=args.get('limit', 20),
                offset=0
            )

            return ToolResult(
                success=True,
                data={
                    # Task model uses 'name' not 'title'
                    "tasks": [{"id": str(t.id), "title": t.name, "status": t.status} for t in tasks],
                    "count": len(tasks)
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _update_task(self, args: Dict, user_id: str) -> ToolResult:
        """Update a task."""
        task_service = self._services.get('task_service')
        if not task_service:
            return ToolResult(success=False, error="Task service not available")

        try:
            # Map tool parameters to service parameters
            # Tool uses 'title', service expects 'name'
            update_args = {
                "task_id": args['task_id']
            }

            if args.get('title'):
                update_args['name'] = args['title']
            if args.get('description'):
                update_args['description'] = args['description']
            if args.get('status'):
                update_args['status'] = args['status']

            # Store priority in metadata
            if args.get('priority'):
                update_args['metadata'] = {'priority': args['priority']}

            task = await task_service.update_task(**update_args)

            return ToolResult(
                success=True,
                data={
                    "task_id": str(task.id),
                    "message": f"Task '{task.name}' updated"
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _get_task_status(self, args: Dict) -> ToolResult:
        """Get task status."""
        task_service = self._services.get('task_service')
        if not task_service:
            return ToolResult(success=False, error="Task service not available")

        try:
            task = await task_service.get_task(args['task_id'])
            if not task:
                return ToolResult(success=False, error="Task not found")

            # Task model uses 'name' not 'title', and 'priority' is in task_metadata
            task_metadata = task.task_metadata or {}
            status_value = task.status.value if hasattr(task.status, 'value') else str(task.status)
            return ToolResult(
                success=True,
                data={
                    "task_id": str(task.id),
                    "title": task.name,
                    "status": status_value,
                    "priority": task_metadata.get('priority', 'medium')
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _search_templates(self, args: Dict) -> ToolResult:
        """Search templates."""
        # TODO: Connect to actual template service
        return ToolResult(
            success=True,
            data={
                "templates": [],
                "message": f"Template search for '{args.get('query')}' would return results here",
                "count": 0
            }
        )

    async def _list_templates(self, args: Dict) -> ToolResult:
        """List templates."""
        return ToolResult(
            success=True,
            data={
                "templates": [],
                "message": "Template listing would return results here",
                "count": 0
            }
        )

    async def _get_template_detail(self, args: Dict) -> ToolResult:
        """Get template detail."""
        return ToolResult(
            success=True,
            data={
                "template_id": args.get('template_id'),
                "message": "Template details would be returned here"
            }
        )

    async def _list_template_categories(self) -> ToolResult:
        """List template categories."""
        return ToolResult(
            success=True,
            data={
                "categories": ["automation", "integration", "analytics", "communication"],
                "industries": ["healthcare", "finance", "legal", "technology", "retail"]
            }
        )

    async def _create_agent(self, args: Dict, user_id: str) -> ToolResult:
        """Create an agent."""
        return ToolResult(
            success=True,
            data={
                "agent_id": "new-agent-id",
                "name": args.get('name'),
                "message": f"Agent '{args.get('name')}' would be created here"
            }
        )

    async def _list_agents(self, args: Dict, user_id: str) -> ToolResult:
        """List agents."""
        return ToolResult(
            success=True,
            data={
                "agents": [],
                "count": 0,
                "message": "Agent listing would return results here"
            }
        )

    async def _get_agent_status(self, args: Dict) -> ToolResult:
        """Get agent status."""
        return ToolResult(
            success=True,
            data={
                "agent_id": args.get('agent_id'),
                "status": "active",
                "message": "Agent status would be returned here"
            }
        )

    async def _update_agent(self, args: Dict, user_id: str) -> ToolResult:
        """Update an agent."""
        return ToolResult(
            success=True,
            data={
                "agent_id": args.get('agent_id'),
                "message": "Agent would be updated here"
            }
        )

    async def _delete_agent(self, args: Dict, user_id: str) -> ToolResult:
        """Delete an agent."""
        return ToolResult(
            success=True,
            data={
                "agent_id": args.get('agent_id'),
                "message": "Agent would be deleted here"
            }
        )

    async def _list_mcp_servers(self, args: Dict) -> ToolResult:
        """List MCP servers."""
        mcp_service = self._services.get('mcp_service')
        if not mcp_service:
            return ToolResult(
                success=True,
                data={"servers": [], "count": 0, "message": "MCP service not configured"}
            )

        try:
            servers = await mcp_service.list_servers()
            return ToolResult(
                success=True,
                data={
                    "servers": [{"id": s.id, "name": s.name, "status": s.status} for s in servers],
                    "count": len(servers)
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _discover_mcp_tools(self, args: Dict) -> ToolResult:
        """Discover MCP tools."""
        mcp_service = self._services.get('mcp_service')
        if not mcp_service:
            return ToolResult(
                success=True,
                data={"tools": [], "count": 0, "message": "MCP service not configured"}
            )

        try:
            tools = await mcp_service.discover_tools(
                query=args.get('query'),
                server=args.get('server'),
                limit=args.get('limit', 10)
            )
            return ToolResult(
                success=True,
                data={"tools": tools, "count": len(tools)}
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _execute_mcp_tool(self, args: Dict, user_id: str) -> ToolResult:
        """Execute an MCP tool."""
        mcp_service = self._services.get('mcp_service')
        if not mcp_service:
            return ToolResult(success=False, error="MCP service not configured")

        try:
            result = await mcp_service.execute_tool(
                server=args['server'],
                tool=args['tool'],
                arguments=args['arguments']
            )
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _configure_mcp_server(self, args: Dict, user_id: str) -> ToolResult:
        """Configure an MCP server."""
        mcp_service = self._services.get('mcp_service')
        if not mcp_service:
            return ToolResult(success=False, error="MCP service not configured")

        try:
            action = args['action']
            if action == 'register':
                result = await mcp_service.register_server(
                    name=args['name'],
                    url=args.get('url'),
                    api_key=args.get('api_key')
                )
            elif action == 'update':
                result = await mcp_service.update_server(
                    name=args['name'],
                    settings=args.get('settings', {})
                )
            elif action == 'remove':
                result = await mcp_service.remove_server(name=args['name'])
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")

            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _get_help(self, args: Dict) -> ToolResult:
        """Get help."""
        topic = args.get('topic', 'general')

        help_content = {
            "general": "I can help you with workflows, tasks, agents, and MCP integrations. What would you like to do?",
            "workflows": "Workflows automate your business processes. You can create, list, execute, and monitor them.",
            "tasks": "Tasks help you track work items. You can create, update, and manage task status.",
            "agents": "AI agents perform specialized tasks. You can create custom agents or use pre-built specialists.",
            "mcp": "MCP (Model Context Protocol) connects to external AI tools. You can list servers, discover tools, and execute them."
        }

        return ToolResult(
            success=True,
            data={
                "topic": topic,
                "help": help_content.get(topic, help_content["general"])
            }
        )

    async def _get_system_status(self, args: Dict) -> ToolResult:
        """Get system status."""
        return ToolResult(
            success=True,
            data={
                "status": "healthy",
                "edition": self.edition.value,
                "tools_available": len(self.get_available_tools()),
                "services_loaded": list(self._services.keys()) if self._initialized else []
            }
        )

    async def _get_platform_metrics(self, args: Dict) -> ToolResult:
        """Get platform-wide metrics: workflow count, success rate, active users."""
        try:
            from sqlalchemy import select, func
            from models.workflow import Workflow
            from models.conversation import ConversationSession

            # Workflow counts
            wf_stmt = select(func.count(Workflow.id))
            wf_result = await self.db.execute(wf_stmt)
            total_workflows = wf_result.scalar() or 0

            active_stmt = select(func.count(Workflow.id)).where(Workflow.status == "active")
            active_result = await self.db.execute(active_stmt)
            active_workflows = active_result.scalar() or 0

            # Session count as proxy for active users
            session_stmt = select(func.count(ConversationSession.id))
            session_result = await self.db.execute(session_stmt)
            total_sessions = session_result.scalar() or 0

            return ToolResult(
                success=True,
                data={
                    "total_workflows": total_workflows,
                    "active_workflows": active_workflows,
                    "total_sessions": total_sessions,
                    "edition": self.edition.value,
                    "message": (
                        f"Platform has {total_workflows} workflows "
                        f"({active_workflows} active), {total_sessions} conversation sessions"
                    )
                }
            )
        except Exception as e:
            logger.warning(f"[v4] Platform metrics query failed: {e}")
            return ToolResult(
                success=True,
                data={"status": "healthy", "edition": self.edition.value,
                      "message": "Metrics query unavailable, platform is running"}
            )

    async def _get_recent_activity(self, args: Dict) -> ToolResult:
        """Get recent platform activity events."""
        try:
            from sqlalchemy import select, desc
            from models.conversation import ConversationSession

            limit = min(args.get("limit", 10), 25)

            stmt = (
                select(ConversationSession)
                .order_by(desc(ConversationSession.last_activity))
                .limit(limit)
            )
            result = await self.db.execute(stmt)
            sessions = result.scalars().all()

            events = []
            for s in sessions:
                events.append({
                    "type": "conversation",
                    "name": getattr(s, "name", None) or "Conversation",
                    "intent": s.primary_intent,
                    "state": s.state,
                    "last_activity": s.last_activity.isoformat() if s.last_activity else None,
                })

            return ToolResult(
                success=True,
                data={
                    "events": events,
                    "count": len(events),
                    "message": f"Found {len(events)} recent activity events"
                }
            )
        except Exception as e:
            logger.warning(f"[v4] Recent activity query failed: {e}")
            return ToolResult(
                success=True,
                data={"events": [], "count": 0, "message": "Activity query unavailable"}
            )

    # =========================================================================
    # API Introspection Tool Handlers (Phase E)
    # =========================================================================

    async def _list_api_endpoints(self, args: Dict) -> ToolResult:
        """List API endpoints with optional filtering."""
        svc = self._services.get('api_introspection_service')
        if not svc:
            return ToolResult(
                success=True,
                data={"endpoints": [], "count": 0, "message": "API Introspection service not available"}
            )
        try:
            await svc._ensure_initialized()
            result = await svc.list_endpoints(
                method=args.get("method"),
                path_prefix=args.get("path_prefix"),
                tag=args.get("tag"),
                limit=args.get("limit", 20),
            )
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.warning(f"[v4] list_api_endpoints failed: {e}")
            return ToolResult(success=False, error=str(e))

    async def _get_endpoint_detail(self, args: Dict) -> ToolResult:
        """Get full detail for a specific API endpoint."""
        svc = self._services.get('api_introspection_service')
        if not svc:
            return ToolResult(
                success=True,
                data={"error": "API Introspection service not available"}
            )
        try:
            await svc._ensure_initialized()
            result = await svc.get_endpoint_detail(
                path=args.get("path", ""),
                method=args.get("method", "GET"),
            )
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.warning(f"[v4] get_endpoint_detail failed: {e}")
            return ToolResult(success=False, error=str(e))

    async def _search_api_capabilities(self, args: Dict) -> ToolResult:
        """Search API endpoints by natural language query."""
        svc = self._services.get('api_introspection_service')
        if not svc:
            return ToolResult(
                success=True,
                data={"results": [], "count": 0, "message": "API Introspection service not available"}
            )
        try:
            await svc._ensure_initialized()
            result = await svc.search_capabilities(
                query=args.get("query", ""),
                limit=args.get("limit", 10),
            )
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.warning(f"[v4] search_api_capabilities failed: {e}")
            return ToolResult(success=False, error=str(e))

    # =========================================================================
    # Integration/Adapter Tool Handlers
    # =========================================================================

    async def _list_integrations(self, args: Dict) -> ToolResult:
        """List available integrations/adapters."""
        adapter_service = self._services.get('adapter_service')

        # Get filter parameters
        category = args.get('category')
        limit = args.get('limit', 20)
        edition = self.edition.value if self.edition else 'community'

        try:
            if adapter_service:
                # Use adapter service to get adapters from factory/registry
                adapters = await adapter_service.list_adapters(
                    edition=edition,
                    limit=limit
                )
                # Convert Adapter objects to dicts for JSON serialization
                adapter_dicts = [
                    {
                        "name": getattr(a, 'name', str(a)),
                        "category": getattr(a, 'category', 'unknown'),
                        "status": "available" if getattr(a, 'enabled', True) else "disabled",
                        "description": getattr(a, 'description', '')
                    }
                    for a in adapters
                ]
                return ToolResult(
                    success=True,
                    data={
                        "integrations": adapter_dicts,
                        "count": len(adapter_dicts),
                        "categories": ["ai_models", "communication", "crm", "payment", "databases", "human_services", "ai_agents"],
                        "tool_name": "list_integrations"
                    }
                )
            else:
                # Fallback: Return known adapter categories from factory
                return ToolResult(
                    success=True,
                    data={
                        "integrations": [
                            {"name": "openai", "category": "ai_models", "status": "available", "description": "OpenAI GPT models"},
                            {"name": "anthropic", "category": "ai_models", "status": "available", "description": "Anthropic Claude models"},
                            {"name": "ollama", "category": "ai_models", "status": "available", "description": "Local Ollama models"},
                            {"name": "slack", "category": "communication", "status": "available", "description": "Slack messaging"},
                            {"name": "email", "category": "communication", "status": "available", "description": "Email via SMTP/SendGrid"},
                            {"name": "discord", "category": "communication", "status": "available", "description": "Discord messaging"},
                            {"name": "salesforce", "category": "crm", "status": "available", "description": "Salesforce CRM"},
                            {"name": "hubspot", "category": "crm", "status": "available", "description": "HubSpot CRM"},
                            {"name": "stripe", "category": "payment", "status": "available", "description": "Stripe payments"},
                            {"name": "postgresql", "category": "databases", "status": "available", "description": "PostgreSQL database"},
                            {"name": "mongodb", "category": "databases", "status": "available", "description": "MongoDB database"},
                            {"name": "upwork", "category": "human_services", "status": "available", "description": "Upwork freelancer platform"},
                            {"name": "fiverr", "category": "human_services", "status": "available", "description": "Fiverr freelancer platform"},
                            {"name": "langchain", "category": "ai_agents", "status": "available", "description": "LangChain agent framework"},
                            {"name": "crewai", "category": "ai_agents", "status": "available", "description": "CrewAI multi-agent framework"},
                        ],
                        "count": 15,
                        "categories": ["ai_models", "communication", "crm", "payment", "databases", "human_services", "ai_agents"],
                        "message": "Showing subset of 51+ available integrations",
                        "tool_name": "list_integrations"
                    }
                )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _get_integration_info(self, args: Dict) -> ToolResult:
        """Get detailed information about a specific integration."""
        adapter_service = self._services.get('adapter_service')
        adapter_name = args.get('adapter_name', '').lower()
        include_actions = args.get('include_actions', True)

        try:
            if adapter_service:
                info = await adapter_service.get_adapter_info(
                    adapter_name=adapter_name,
                    include_actions=include_actions
                )
                return ToolResult(success=True, data=info)
            else:
                # Fallback: Return known adapter info
                adapter_info = {
                    "slack": {
                        "name": "slack",
                        "category": "communication",
                        "description": "Send messages and interact with Slack workspaces",
                        "actions": ["send_message", "create_channel", "upload_file", "list_channels", "add_reaction"],
                        "required_config": ["bot_token", "workspace_id"],
                        "documentation_url": "https://api.slack.com/"
                    },
                    "email": {
                        "name": "email",
                        "category": "communication",
                        "description": "Send and manage emails via SMTP or SendGrid",
                        "actions": ["send_email", "send_template", "add_attachment"],
                        "required_config": ["smtp_host", "smtp_port", "username", "password"],
                        "documentation_url": "https://sendgrid.com/docs/"
                    },
                    "stripe": {
                        "name": "stripe",
                        "category": "payment",
                        "description": "Process payments, manage subscriptions, and handle invoices",
                        "actions": ["create_charge", "create_customer", "create_subscription", "send_invoice", "refund"],
                        "required_config": ["api_key", "webhook_secret"],
                        "documentation_url": "https://stripe.com/docs/api"
                    },
                    "salesforce": {
                        "name": "salesforce",
                        "category": "crm",
                        "description": "Manage leads, contacts, opportunities, and accounts",
                        "actions": ["create_lead", "update_lead", "create_contact", "create_opportunity", "query_records"],
                        "required_config": ["client_id", "client_secret", "username", "password", "security_token"],
                        "documentation_url": "https://developer.salesforce.com/docs/"
                    },
                    "openai": {
                        "name": "openai",
                        "category": "ai_models",
                        "description": "Access GPT-4, GPT-3.5, DALL-E, and other OpenAI models",
                        "actions": ["generate_text", "generate_image", "create_embedding", "moderate_content"],
                        "required_config": ["api_key"],
                        "documentation_url": "https://platform.openai.com/docs/"
                    },
                }

                info = adapter_info.get(adapter_name, {
                    "name": adapter_name,
                    "status": "unknown",
                    "message": f"Integration '{adapter_name}' info not found. Use list_integrations to see available options."
                })

                return ToolResult(success=True, data=info)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _configure_integration(self, args: Dict, user_id: str) -> ToolResult:
        """Configure an integration with credentials and settings."""
        adapter_service = self._services.get('adapter_service')
        adapter_name = args.get('adapter_name', '').lower()
        credentials = args.get('credentials', {})
        settings = args.get('settings', {})

        try:
            if adapter_service:
                result = await adapter_service.configure_adapter(
                    adapter_name=adapter_name,
                    user_id=user_id,
                    credentials=credentials,
                    settings=settings
                )
                return ToolResult(
                    success=True,
                    data={
                        "adapter_name": adapter_name,
                        "configured": True,
                        "message": f"Integration '{adapter_name}' configured successfully",
                        **result
                    }
                )
            else:
                # Fallback: Acknowledge but note service unavailable
                return ToolResult(
                    success=True,
                    data={
                        "adapter_name": adapter_name,
                        "configured": False,
                        "message": f"Configuration for '{adapter_name}' would be saved. Adapter service not fully initialized."
                    }
                )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _execute_integration(self, args: Dict, user_id: str) -> ToolResult:
        """Execute an action via an integration."""
        adapter_service = self._services.get('adapter_service')
        adapter_name = args.get('adapter_name', '').lower()
        action = args.get('action', '')
        params = args.get('params', {})

        try:
            if adapter_service:
                result = await adapter_service.execute_action(
                    adapter_name=adapter_name,
                    action=action,
                    params=params,
                    user_id=user_id
                )
                return ToolResult(
                    success=True,
                    data={
                        "adapter_name": adapter_name,
                        "action": action,
                        "result": result,
                        "message": f"Action '{action}' executed on '{adapter_name}'"
                    }
                )
            else:
                # Fallback: Acknowledge the request
                return ToolResult(
                    success=False,
                    error=f"Adapter service not available. Cannot execute '{action}' on '{adapter_name}'.",
                    recovery_strategy=ToolRecoveryStrategy.RETRY,
                    retry_after=5
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                recovery_strategy=ToolRecoveryStrategy.CLARIFY
            )

    async def _test_integration(self, args: Dict) -> ToolResult:
        """Test that an integration is properly configured and working."""
        adapter_service = self._services.get('adapter_service')
        adapter_name = args.get('adapter_name', '').lower()

        try:
            if adapter_service:
                test_result = await adapter_service.test_adapter(adapter_name)
                return ToolResult(
                    success=True,
                    data={
                        "adapter_name": adapter_name,
                        "test_passed": test_result.get('success', False),
                        "response_time_ms": test_result.get('response_time_ms'),
                        "message": test_result.get('message', 'Test completed'),
                        "details": test_result
                    }
                )
            else:
                return ToolResult(
                    success=True,
                    data={
                        "adapter_name": adapter_name,
                        "test_passed": False,
                        "message": f"Adapter service not available. Cannot test '{adapter_name}'."
                    }
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Test failed for '{adapter_name}': {str(e)}"
            )

    # =========================================================================
    # File Access Tool Handlers
    # =========================================================================

    async def _list_user_files(self, args: Dict, user_id: str) -> ToolResult:
        """List staged files for the current user."""
        try:
            from sqlalchemy import select
            from models.staged_file import StagedFile
            from datetime import datetime, timezone

            limit = args.get("limit", 20)
            stmt = (
                select(StagedFile)
                .where(StagedFile.user_id == user_id)
                .where(StagedFile.expires_at > datetime.now(timezone.utc))
                .order_by(StagedFile.created_at.desc())
                .limit(limit)
            )
            result = await self.db.execute(stmt)
            files = result.scalars().all()

            file_list = [{
                "file_id": str(f.id),
                "filename": f.filename,
                "content_type": f.content_type,
                "file_size": f.file_size,
                "stage": f.stage,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "has_extracted_data": bool(f.extracted_data)
            } for f in files]

            return ToolResult(
                success=True,
                data={
                    "files": file_list,
                    "count": len(file_list),
                    "message": f"Found {len(file_list)} file(s)" if file_list else "No files found. Upload a file first."
                }
            )
        except Exception as e:
            logger.error(f"[v4] Error listing user files: {e}")
            return ToolResult(success=False, error=f"Failed to list files: {str(e)}")

    async def _access_staged_file(self, args: Dict, user_id: str) -> ToolResult:
        """Read extracted data from a staged file."""
        try:
            from sqlalchemy import select
            from models.staged_file import StagedFile
            from datetime import datetime, timezone

            file_id = args.get("file_id")
            if not file_id:
                return ToolResult(success=False, error="file_id is required")

            stmt = (
                select(StagedFile)
                .where(StagedFile.id == file_id)
                .where(StagedFile.user_id == user_id)
                .where(StagedFile.expires_at > datetime.now(timezone.utc))
            )
            result = await self.db.execute(stmt)
            staged = result.scalar_one_or_none()

            if not staged:
                return ToolResult(
                    success=False,
                    error=f"File '{file_id}' not found, expired, or not owned by you"
                )

            return ToolResult(
                success=True,
                data={
                    "file_id": str(staged.id),
                    "filename": staged.filename,
                    "content_type": staged.content_type,
                    "file_size": staged.file_size,
                    "stage": staged.stage,
                    "extracted_data": staged.extracted_data or {}
                }
            )
        except Exception as e:
            logger.error(f"[v4] Error accessing staged file: {e}")
            return ToolResult(success=False, error=f"Failed to access file: {str(e)}")

    # =========================================================================
    # Browser Automation Tool Handlers
    # =========================================================================

    async def _browser_execute(self, args: Dict, user_id: str) -> ToolResult:
        """Execute a sequence of browser actions via the browser service."""
        import re
        try:
            actions = args.get("actions", [])
            timeout_ms = min(args.get("timeout_ms", 30000), 60000)  # cap at 60s

            if not actions:
                return ToolResult(success=False, error="actions array is required")
            if len(actions) > 10:
                return ToolResult(success=False, error="Maximum 10 actions per call")

            # Validate URL schemes in actions
            blocked_schemes = re.compile(r'^(file|javascript|data):', re.IGNORECASE)
            for action in actions:
                url = action.get("url", "")
                if url and blocked_schemes.match(url):
                    return ToolResult(
                        success=False,
                        error=f"URL scheme not allowed: {url.split(':')[0]}://"
                    )

            # Business+ governance: risk assessment on browser actions
            try:
                from aictrlnet_business.services.ai_governance import RiskAssessmentEngine
                risk_engine = RiskAssessmentEngine()
                risk_result = risk_engine.assess_browser_risk(actions)
                if risk_result.get("risk_level") in ("very_high", "critical"):
                    return ToolResult(
                        success=False,
                        error=f"Browser action blocked: risk level {risk_result.get('risk_level')}"
                    )
                logger.info(f"Browser governance: risk={risk_result.get('risk_level', 'n/a')}")
            except ImportError:
                pass  # Community edition — no governance

            import httpx
            async with httpx.AsyncClient(timeout=timeout_ms / 1000 + 5) as client:
                resp = await client.post(
                    "http://browser-service:8005/browser/execute",
                    json={
                        "actions": actions,
                        "timeout_ms": timeout_ms,
                        "viewport": args.get("viewport", {"width": 1280, "height": 720})
                    }
                )
                if resp.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"Browser service error: {resp.status_code} - {resp.text[:200]}"
                    )
                data = resp.json()
                # Extract screenshots from results array (browser service
                # embeds them as ActionResult entries with type "screenshot")
                results = data.get("results", [])
                screenshots = [
                    r.get("data", {}).get("base64", "")
                    for r in results
                    if r.get("action_type") == "screenshot" and r.get("success")
                ]
                return ToolResult(
                    success=data.get("success", True),
                    data={
                        "results": results,
                        "screenshots": screenshots,
                        "total_duration_ms": data.get("total_duration_ms", 0),
                        "page_url": data.get("page_url"),
                        "page_title": data.get("page_title")
                    }
                )
        except Exception as e:
            logger.error(f"[v4] Browser execute error: {e}")
            return ToolResult(success=False, error=f"Browser automation failed: {str(e)}")

    async def _browser_screenshot(self, args: Dict, user_id: str) -> ToolResult:
        """Navigate to URL and take screenshot."""
        url = args.get("url")
        if not url:
            return ToolResult(success=False, error="url is required")

        return await self._browser_execute({
            "actions": [
                {"type": "navigate", "url": url},
                {"type": "screenshot"}
            ],
            "timeout_ms": 30000
        }, user_id)

    async def _browser_extract(self, args: Dict, user_id: str) -> ToolResult:
        """Navigate to URL and extract text from selectors."""
        url = args.get("url")
        if not url:
            return ToolResult(success=False, error="url is required")

        selectors = args.get("selectors", ["body"])
        actions = [{"type": "navigate", "url": url}]
        for sel in selectors[:5]:  # max 5 selectors
            actions.append({"type": "extract_text", "selector": sel})

        return await self._browser_execute({
            "actions": actions,
            "timeout_ms": 30000
        }, user_id)

    # =========================================================================
    # Tool Chaining Support
    # =========================================================================

    async def execute_tool_chain(
        self,
        tool_calls: List[Dict[str, Any]],
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None
    ) -> ChainResult:
        """Execute multiple tools in sequence with progress reporting.

        Args:
            tool_calls: List of tool calls with name and arguments
            user_id: User ID for authorization
            context: Shared context passed between tools
            progress_callback: Optional callback for progress updates

        Returns:
            ChainResult with all results and final output
        """
        results = []
        chain_context = context.copy() if context else {}

        for idx, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get('name')
            arguments = tool_call.get('arguments', {})

            # Report progress
            if progress_callback:
                await progress_callback({
                    "step": idx + 1,
                    "total": len(tool_calls),
                    "tool": tool_name,
                    "status": "executing"
                })

            # Execute tool with chain context
            result = await self.invoke(
                tool_name,
                {**arguments, **chain_context},
                user_id,
                chain_context
            )

            results.append({
                "tool": tool_name,
                "success": result.success,
                "data": result.data,
                "error": result.error
            })

            # Update context for next tool (pipeline pattern)
            if result.success and result.data:
                chain_context.update(result.data)

            # Handle failure
            if not result.success:
                if result.recovery_strategy == ToolRecoveryStrategy.CLARIFY:
                    return ChainResult(
                        completed=False,
                        results=results,
                        requires_clarification=True,
                        error=result.error
                    )
                elif result.recovery_strategy == ToolRecoveryStrategy.FALLBACK and result.fallback_tool:
                    # Try fallback tool
                    fallback_result = await self.invoke(
                        result.fallback_tool,
                        arguments,
                        user_id,
                        chain_context
                    )
                    if fallback_result.success:
                        results[-1] = {
                            "tool": result.fallback_tool,
                            "success": True,
                            "data": fallback_result.data,
                            "fallback_from": tool_name
                        }
                        if fallback_result.data:
                            chain_context.update(fallback_result.data)
                        continue

                return ChainResult(
                    completed=False,
                    results=results,
                    error=result.error
                )

        return ChainResult(
            completed=True,
            results=results,
            final_context=chain_context
        )
