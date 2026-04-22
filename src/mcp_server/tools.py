"""MCP tool definitions for AICtrlNet.

Each tool has name, description, and inputSchema (JSON Schema).
Edition-aware: Business/Enterprise tools added via accretive try/except.
"""

import logging

logger = logging.getLogger(__name__)

COMMUNITY_TOOLS = [
    {
        "name": "create_workflow",
        "description": (
            "Create a new workflow from a natural language description. "
            "Supports AI processing nodes, human approval steps, data sources, "
            "notifications, and conditional branching."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What the workflow should do, e.g. 'KYC verification with human approval for amounts over $10K'",
                },
                "name": {
                    "type": "string",
                    "description": "Optional workflow name. Auto-generated from description if omitted.",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "list_workflows",
        "description": "List existing workflows with their status, name, and node count.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of workflows to return (default 20)",
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of workflows to skip (default 0)",
                    "default": 0,
                },
            },
        },
    },
    {
        "name": "get_workflow",
        "description": "Get details of a specific workflow including its definition, nodes, and status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The workflow ID to retrieve",
                },
            },
            "required": ["workflow_id"],
        },
    },
    {
        "name": "execute_workflow",
        "description": (
            "Execute an existing workflow with optional input data. Returns the "
            "execution ID for status tracking. Set dry_run=true to simulate "
            "execution without side effects — nodes that would touch external "
            "systems (adapters, notifications, browser, approvals) return "
            "simulated results, and the response includes dry_run_source."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The workflow ID to execute",
                },
                "input_data": {
                    "type": "object",
                    "description": "Optional input data to pass to the workflow",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": (
                        "If true, run in dry-run mode: side-effect nodes are "
                        "simulated. Effective value OR-merged with session / "
                        "agent / pod dry-run settings (any true value wins)."
                    ),
                    "default": False,
                },
                "idempotency_key": {
                    "type": "string",
                    "description": (
                        "Optional client-supplied key for retry safety. If the "
                        "same key is seen within 24h with the same arguments, "
                        "the previous response is returned without re-executing."
                    ),
                },
            },
            "required": ["workflow_id"],
        },
    },
    {
        "name": "get_execution_status",
        "description": "Check the status of a running or completed workflow execution.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "The execution ID returned from execute_workflow",
                },
            },
            "required": ["execution_id"],
        },
    },
    {
        "name": "list_templates",
        "description": "Browse available workflow templates. AICtrlNet provides templates for common patterns like data pipelines, approval flows, and AI processing chains.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category (e.g. 'communication', 'data', 'reliability')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of templates to return (default 20)",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "assess_quality",
        "description": "Assess the quality of content (text, JSON, or code) across dimensions: accuracy, completeness, relevance, clarity, consistency.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to assess",
                },
                "content_type": {
                    "type": "string",
                    "description": "Type of content: 'text', 'json', or 'code'",
                    "enum": ["text", "json", "code"],
                    "default": "text",
                },
                "criteria": {
                    "type": "object",
                    "description": "Optional custom assessment criteria",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to the AICtrlNet intelligent assistant. Automatically manages conversation sessions. Returns the assistant's reply with suggested actions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send",
                },
            },
            "required": ["message"],
        },
    },
]

BUSINESS_TOOLS = [
    {
        "name": "evaluate_policy",
        "description": "Test content against an AI Governance Policy (AGP). Returns whether the content complies and any violations found.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "policy_id": {
                    "type": "string",
                    "description": "The policy ID to evaluate against",
                },
                "content": {
                    "type": "string",
                    "description": "The content to test against the policy",
                },
            },
            "required": ["policy_id", "content"],
        },
    },
    {
        "name": "list_policies",
        "description": "List available AI Governance Policies (AGP) with their status and description.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of policies to return (default 20)",
                    "default": 20,
                },
            },
        },
    },
]

ENTERPRISE_TOOLS = [
    {
        "name": "check_compliance",
        "description": "Run a compliance check against configured frameworks (GDPR, HIPAA, SOC2). Returns compliance status, missing frameworks, and expiring certifications.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "The MCP server ID to check compliance for",
                },
            },
            "required": ["server_id"],
        },
    },
]

TOOL_SCOPES = {
    # Original 11 tools — migrated to per-resource taxonomy.
    # Legacy scopes on existing API keys are expanded at scope-check
    # time via ``scopes.expand_legacy`` (Phase A compatibility).
    "create_workflow": ["write:workflows"],
    "execute_workflow": ["write:workflows"],
    "list_workflows": ["read:workflows"],
    "get_workflow": ["read:workflows"],
    "get_execution_status": ["read:workflows"],
    "list_templates": ["read:templates"],
    "assess_quality": ["read:workflows"],
    "send_message": ["write:conversations"],
    "evaluate_policy": ["read:policies"],
    "list_policies": ["read:policies"],
    "check_compliance": ["read:compliance"],
}


def get_tools_for_edition() -> list:
    """Return tool definitions available for the current edition.

    Uses accretive try/except pattern: Community tools always present,
    Business/Enterprise tools added if their modules are importable.
    """
    tools = list(COMMUNITY_TOOLS)

    try:
        from aictrlnet_business.api.v1.endpoints import agp_evaluation  # noqa: F401
        tools.extend(BUSINESS_TOOLS)
    except ImportError:
        pass

    try:
        from aictrlnet_enterprise.services.mcp_compliance import MCPComplianceService  # noqa: F401
        tools.extend(ENTERPRISE_TOOLS)
    except ImportError:
        pass

    return tools
