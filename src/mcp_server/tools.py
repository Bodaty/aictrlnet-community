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
    # ---- Wave 1: Adapters (Layer 1) ----
    {
        "name": "list_adapters",
        "description": (
            "List platform-adapter definitions available for this edition. "
            "Adapters unlock entire ecosystems (n8n, Zapier, Make, IFTTT, "
            "Power Automate, SaaS APIs) through pre-built, governed integrations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Filter by category"},
                "search": {"type": "string", "description": "Case-insensitive name/description match"},
                "limit": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0},
                "include_unavailable": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include adapters not available to the caller's edition",
                },
            },
        },
    },
    {
        "name": "get_adapter",
        "description": "Get details of a single adapter definition by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"adapter_id": {"type": "string"}},
            "required": ["adapter_id"],
        },
    },
    {
        "name": "list_my_adapter_configs",
        "description": "List the caller's saved adapter configurations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "adapter_type": {"type": "string"},
                "enabled_only": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "test_adapter_config",
        "description": (
            "Dry-run an adapter configuration (no side effects). Verifies "
            "credentials + reachability + quota. Safe to run anytime."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "config_id": {"type": "string"},
                "timeout_seconds": {"type": "integer", "default": 10},
            },
            "required": ["config_id"],
        },
    },
    # ---- Wave 1: NL entry ----
    {
        "name": "nl_to_workflow",
        "description": (
            "Convert a natural-language description into an AICtrlNet workflow "
            "(returns the created workflow's id + node plan). Same service "
            "that powers create_workflow but with the explicit NL->plan shape."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "NL description of the workflow"},
                "context": {"type": "object"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "analyze_intent",
        "description": (
            "Analyze the intent of a piece of text without creating a workflow. "
            "Useful for routing decisions before committing side-effectful calls."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "context": {"type": "object"},
            },
            "required": ["text"],
        },
    },
    # ---- Wave 1: Autonomy read surface (Control Spectrum) ----
    {
        "name": "get_workflow_autonomy",
        "description": (
            "Return the resolved autonomy level + phase for a workflow. "
            "Implements v11 'six autonomy phases' — Observation / Learning / "
            "Recommendation / Cooperation / Supervision / Full Delegation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"workflow_id": {"type": "string"}},
            "required": ["workflow_id"],
        },
    },
    {
        "name": "preview_autonomy",
        "description": (
            "Preview what would auto-approve vs gate at a given autonomy level "
            "for a specific workflow. Read-only; does not change any state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "level": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["workflow_id", "level"],
        },
    },
    # ---- Wave 2: API-key + subscription introspection ----
    {
        "name": "list_api_keys",
        "description": (
            "List the caller's API keys with key identifier (prefix...suffix), "
            "scopes, and expiration. Raw key values are never returned."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_api_key_usage",
        "description": (
            "Get per-tool usage counts for an API key over the last N days. "
            "Defaults to the calling key if no id is supplied."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "api_key_id": {"type": "string"},
                "days": {"type": "integer", "default": 30, "minimum": 1, "maximum": 90},
            },
        },
    },
    {
        "name": "get_subscription",
        "description": (
            "Return the caller's current subscription — plan, status, "
            "billing period, and trial-end date. Use get_upgrade_options "
            "for available upgrades."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_upgrade_options",
        "description": (
            "List subscription plans available to the caller at a higher "
            "edition than the current plan. Includes monthly/annual pricing "
            "and feature/limit diffs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_edition": {
                    "type": "string",
                    "description": "Filter plans by edition: community | business | enterprise",
                },
            },
        },
    },
    # ---- Wave 3: Trial metering surface ----
    {
        "name": "get_trial_status",
        "description": (
            "Return the caller's current usage vs. edition limits — LLM "
            "calls, workflows, adapters, API calls, storage — with "
            "percent-used and upgrade prompts when >=80% of any limit. "
            "This is the canonical claim-validation point for v11.2 "
            "'trial metering works through MCP'."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_usage_report",
        "description": (
            "Aggregate per-resource usage over the current billing period "
            "(or a specific one) with breakdown by resource_type and day. "
            "Backs up v11.2 'usage tracking' claim — CISO-grade introspection."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Look-back window in days (1-90). Default 30.",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 90,
                },
                "resource_type": {
                    "type": "string",
                    "description": "Filter by resource type (e.g. llm_calls, browser_actions, workflows)",
                },
            },
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
    # ---- Wave 1: Autonomy write surface ----
    {
        "name": "set_workflow_autonomy",
        "description": (
            "Set the autonomy level (0-100) and optional lock on a workflow. "
            "Enforces tenant ownership — cross-tenant writes are rejected."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "autonomy_level": {"type": "integer", "minimum": 0, "maximum": 100},
                "autonomy_locked": {"type": "boolean"},
            },
            "required": ["workflow_id"],
        },
    },
    # ---- Wave 1: Self-extending agents (Layer 2 — v11.4 hero) ----
    {
        "name": "research_api",
        "description": (
            "Research an API and return a structured specification (base_url, "
            "auth_type, capabilities, confidence). Uses a known-API database, "
            "OpenAPI extraction, and LLM fallback. Metered."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "api_name": {"type": "string"},
                "documentation_url": {"type": "string"},
                "user_context": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["api_name"],
        },
    },
    {
        "name": "generate_adapter",
        "description": (
            "Generate (research -> codegen -> validate -> risk-assess) an adapter "
            "for a new API. Long-running: returns immediately with the adapter "
            "id + initial status; poll get_generated_adapter_status for "
            "lifecycle transitions. Metered."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "api_name": {"type": "string"},
                "base_url": {"type": "string"},
                "auth_type": {"type": "string"},
                "capabilities": {"type": "array", "items": {"type": "object"}},
                "auth_config": {"type": "object"},
                "description": {"type": "string"},
                "generation_mode": {
                    "type": "string",
                    "enum": ["python_code", "declarative_http"],
                    "default": "python_code",
                },
                "idempotency_key": {"type": "string"},
            },
            "required": ["name", "api_name", "base_url", "auth_type", "capabilities"],
        },
    },
    {
        "name": "self_extend",
        "description": (
            "One-call orchestrator: research an API and kick off adapter "
            "generation. Returns the adapter id + research spec + initial "
            "status. Hero tool for v11.4 'platform that builds its own "
            "integrations'. Metered."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "api_name": {"type": "string"},
                "documentation_url": {"type": "string"},
                "user_context": {"type": "string"},
                "generation_mode": {
                    "type": "string",
                    "enum": ["python_code", "declarative_http"],
                    "default": "python_code",
                },
                "idempotency_key": {"type": "string"},
            },
            "required": ["api_name"],
        },
    },
    {
        "name": "list_generated_adapters",
        "description": "List generated adapters with optional status filter.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "mine_only": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "get_generated_adapter_status",
        "description": (
            "Get the current lifecycle state of a generated adapter — status, "
            "risk score, risk level, risk details, validation errors."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"adapter_id": {"type": "string"}},
            "required": ["adapter_id"],
        },
    },
    {
        "name": "get_generated_adapter_source",
        "description": (
            "Fetch the generated source code or declarative spec for a "
            "generated adapter so Claude (or a human) can review before "
            "approval."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"adapter_id": {"type": "string"}},
            "required": ["adapter_id"],
        },
    },
    {
        "name": "approve_adapter",
        "description": (
            "Approve a generated adapter for activation. Does not yet make the "
            "adapter usable — call activate_adapter after."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string"},
                "comments": {"type": "string"},
            },
            "required": ["adapter_id"],
        },
    },
    {
        "name": "reject_adapter",
        "description": "Reject a generated adapter. Terminal state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["adapter_id"],
        },
    },
    {
        "name": "activate_adapter",
        "description": (
            "Register an approved adapter with the adapter factory so it can "
            "be used in workflows. Refuses adapters not in 'approved' state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"adapter_id": {"type": "string"}},
            "required": ["adapter_id"],
        },
    },
    # ---- Wave 1: Browser (Layer 3) ----
    {
        "name": "browser_execute",
        "description": (
            "Execute a sequence of browser actions (navigate, click, fill, "
            "screenshot, extract_text, wait_for). URLs validated against "
            "RFC1918/link-local/loopback/cloud-metadata deny-lists; capped at "
            "20 actions. run_script/download require a per-tenant feature "
            "flag. Metered per action."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "navigate", "click", "fill", "screenshot",
                                    "extract_text", "wait_for",
                                    "run_script", "download",
                                ],
                            },
                            "url": {"type": "string"},
                            "selector": {"type": "string"},
                            "value": {"type": "string"},
                            "script": {"type": "string"},
                            "full_page": {"type": "boolean"},
                            "timeout_ms": {"type": "integer"},
                        },
                        "required": ["type"],
                    },
                },
                "timeout_ms": {"type": "integer", "default": 30000},
                "viewport": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                },
            },
            "required": ["actions"],
        },
    },
    # ---- Wave 1: Approval queue (Layer 4 — Human Orchestration) ----
    {
        "name": "list_pending_approvals",
        "description": (
            "List approval requests pending human decision. Optional filters "
            "by workflow, resource type, or requester."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "resource_type": {"type": "string"},
                "mine_only": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "get_approval",
        "description": "Get details of a single approval request.",
        "inputSchema": {
            "type": "object",
            "properties": {"request_id": {"type": "string"}},
            "required": ["request_id"],
        },
    },
    {
        "name": "approve_request",
        "description": (
            "Approve a pending approval request. Publishes an "
            "approval.request.approved event (wired to channel notifications)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "comments": {"type": "string"},
            },
            "required": ["request_id"],
        },
    },
    {
        "name": "reject_request",
        "description": (
            "Reject a pending approval request. Reason is recorded and "
            "published on the approval.request.rejected event."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["request_id", "reason"],
        },
    },
    # ---- Wave 2: AI governance visibility ----
    {
        "name": "list_ai_policies",
        "description": (
            "List AI governance policies (content safety, cost, privacy, "
            "custom rules). Direct answer to v11 'Claude gains governance' "
            "claim — Claude can enumerate the policies it's subject to."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "policy_type": {
                    "type": "string",
                    "description": "Filter by policy type (e.g. content_safety, cost_control, data_privacy)",
                },
                "enabled": {"type": "boolean"},
            },
        },
    },
    {
        "name": "create_policy",
        "description": (
            "Create a new AI governance policy. Rule evaluator + severity + "
            "applies_to selectors. Creation itself is audit-logged."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "policy_type": {"type": "string"},
                "rules": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Rule conditions + actions (see AGP rule schema)",
                },
                "applies_to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Resource types / ids this policy applies to",
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "default": "medium",
                },
                "enabled": {"type": "boolean", "default": True},
                "resource_metadata": {"type": "object"},
            },
            "required": ["name", "policy_type", "rules"],
        },
    },
    {
        "name": "get_ai_audit_logs",
        "description": (
            "Query AI governance audit logs — per-action, per-model, "
            "per-user, per-status, with date range. Up to 1000 rows per "
            "call. This is what Claude uses to reason about its own "
            "past actions under governance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "model_id": {"type": "string"},
                "user_id": {"type": "string"},
                "status": {"type": "string"},
                "start_date": {"type": "string", "format": "date-time"},
                "end_date": {"type": "string", "format": "date-time"},
                "limit": {"type": "integer", "default": 100, "maximum": 1000},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "list_violations",
        "description": (
            "List policy violations raised by the governance engine. "
            "Filterable by policy, severity, or resolved/unresolved."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "resolved": {"type": "boolean"},
                "limit": {"type": "integer", "default": 100, "maximum": 1000},
                "offset": {"type": "integer", "default": 0},
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
    # Wave 1: Adapters (Layer 1)
    "list_adapters": ["read:adapters"],
    "get_adapter": ["read:adapters"],
    "list_my_adapter_configs": ["read:adapters"],
    "test_adapter_config": ["write:adapters"],
    # Wave 1: NL entry
    "nl_to_workflow": ["write:workflows"],
    "analyze_intent": ["read:workflows"],
    # Wave 1: Autonomy (Control Spectrum)
    "get_workflow_autonomy": ["read:autonomy"],
    "preview_autonomy": ["read:autonomy"],
    "set_workflow_autonomy": ["write:autonomy"],
    # Wave 1: Self-extending (Layer 2)
    "research_api": ["write:self_extending"],
    "generate_adapter": ["write:self_extending"],
    "self_extend": ["write:self_extending"],
    "list_generated_adapters": ["read:self_extending"],
    "get_generated_adapter_status": ["read:self_extending"],
    "get_generated_adapter_source": ["read:self_extending"],
    "approve_adapter": ["write:self_extending"],
    "reject_adapter": ["write:self_extending"],
    "activate_adapter": ["write:self_extending"],
    # Wave 1: Browser (Layer 3)
    "browser_execute": ["write:browser"],
    # Wave 1: Approval queue (Layer 4)
    "list_pending_approvals": ["read:approvals"],
    "get_approval": ["read:approvals"],
    "approve_request": ["write:approvals"],
    "reject_request": ["write:approvals"],
    # Wave 2: API-key + subscription introspection
    "list_api_keys": ["read:usage"],
    "get_api_key_usage": ["read:usage"],
    "get_subscription": ["read:subscription"],
    "get_upgrade_options": ["read:subscription"],
    # Wave 2: AI governance visibility
    "list_ai_policies": ["read:policies"],
    "create_policy": ["write:policies"],
    "get_ai_audit_logs": ["read:audit"],
    "list_violations": ["read:policies"],
    # Wave 3: Trial metering surface
    "get_trial_status": ["read:usage"],
    "get_usage_report": ["read:usage"],
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
