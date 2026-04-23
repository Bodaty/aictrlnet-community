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
    # ---- Wave 4: Tasks ----
    {
        "name": "create_task",
        "description": (
            "Create a new task. Tasks are status-tracked units of work "
            "that workflows, agents, and humans can operate on. Returns "
            "the new task's id + initial PENDING status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "metadata": {"type": "object"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List tasks, optionally filtered by status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "e.g. pending, in_progress, completed, failed"},
                "limit": {"type": "integer", "default": 100, "maximum": 500},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "get_task",
        "description": "Get details of a single task by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "update_task",
        "description": "Update a task's name, description, status, or metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "status": {"type": "string"},
                "metadata": {"type": "object"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "complete_task",
        "description": (
            "Mark a task as completed. Convenience wrapper around update_task "
            "with status=completed plus completed_at timestamp recorded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "result": {"type": "object", "description": "Optional result payload"},
            },
            "required": ["task_id"],
        },
    },
    # ---- Wave 4: Memory (per-user, session-scoped) ----
    {
        "name": "get_memory",
        "description": (
            "Retrieve a value from the caller's per-user memory store by "
            "key. Community edition: in-memory dict, not persistent. "
            "Returns null if the key doesn't exist."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "set_memory",
        "description": (
            "Store a value under a key in the caller's memory. Max 1000 "
            "keys and 10MB total per user. Overwrites existing values."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"description": "Any JSON-serializable value"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "delete_memory",
        "description": "Delete a key from the caller's memory. Returns true on success.",
        "inputSchema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    # ---- Wave 4: Conversations + Channels (read) ----
    {
        "name": "list_conversations",
        "description": "List the caller's active conversation sessions.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_conversation",
        "description": "Get a single conversation session with its recent messages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "message_limit": {"type": "integer", "default": 50, "maximum": 500},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "list_linked_channels",
        "description": (
            "List the caller's linked external channels (Slack, Discord, "
            "Telegram, WhatsApp, Twilio, Email). v6 channel-agnostic arch."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "request_channel_link_code",
        "description": (
            "Generate a 6-digit code the caller can send via the target "
            "channel to prove ownership and link it to their account. "
            "Code expires in 10 minutes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel_type": {
                    "type": "string",
                    "enum": ["slack", "discord", "telegram", "whatsapp", "twilio", "email"],
                },
            },
            "required": ["channel_type"],
        },
    },
    {
        "name": "unlink_channel",
        "description": "Unlink a previously-linked channel.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel_type": {"type": "string"},
                "channel_user_id": {"type": "string"},
            },
            "required": ["channel_type", "channel_user_id"],
        },
    },
    # ---- Wave 4: Knowledge ----
    {
        "name": "query_knowledge",
        "description": (
            "Query the AICtrlNet knowledge base (capabilities, adapter "
            "docs, feature descriptions). Returns relevant snippets."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "context": {"type": "object"},
                "limit": {"type": "integer", "default": 5, "maximum": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "suggest_next_actions",
        "description": (
            "Given the current action or workflow state, suggest next-step "
            "actions the caller could take. Backed by KnowledgeRetrievalService."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "current_action": {"type": "string"},
                "context": {"type": "object"},
            },
            "required": ["current_action"],
        },
    },
    {
        "name": "get_capabilities_summary",
        "description": (
            "Return a summary of AICtrlNet's capabilities — adapters, "
            "agents, features — as a structured document. Useful for Claude "
            "to self-orient before making bigger decisions."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ---- Wave 4: Templates ----
    {
        "name": "search_templates",
        "description": (
            "Search the 183-template catalog by name, category, or tags. "
            "Returns matching templates with description + complexity."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "string"},
                "limit": {"type": "integer", "default": 20, "maximum": 100},
            },
        },
    },
    {
        "name": "instantiate_template",
        "description": (
            "Create a new workflow from a template with optional parameter "
            "overrides. Returns the new workflow's id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "template_id": {"type": "string"},
                "name": {"type": "string", "description": "Optional name for the instantiated workflow"},
                "parameters": {"type": "object"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["template_id"],
        },
    },
    # ---- Wave 4: Files ----
    {
        "name": "upload_file",
        "description": (
            "Upload a file (base64-encoded content) as a staged file the "
            "caller can reference later. Max 10 MB, MIME allow-list "
            "enforced, magic-byte validated."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content_base64": {"type": "string"},
                "content_type": {"type": "string"},
                "workflow_id": {
                    "type": "string",
                    "description": "Optional — trigger a workflow with this file as input",
                },
            },
            "required": ["filename", "content_base64"],
        },
    },
    {
        "name": "list_staged_files",
        "description": "List files the caller has uploaded.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50, "maximum": 500},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "get_staged_file",
        "description": (
            "Get a staged file's metadata (and optionally its content as "
            "base64). Strictly tenant + user scoped."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "include_content": {"type": "boolean", "default": False},
            },
            "required": ["file_id"],
        },
    },
    # ---- Wave 4: Data Quality ----
    {
        "name": "assess_data_quality",
        "description": (
            "Run a quality assessment on structured data across dimensions "
            "(accuracy, completeness, consistency, timeliness, uniqueness, "
            "validity). Returns scored dimensions + overall score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {"description": "Data to assess (dict, list, or JSON string)"},
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific dimensions to assess (default: all)",
                },
                "rules": {"type": "object", "description": "Optional per-dimension rules"},
            },
            "required": ["data"],
        },
    },
    {
        "name": "list_quality_dimensions",
        "description": "List the supported data-quality dimensions + their descriptions.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ---- Wave 5: Institute (education-led GTM — v11.1) ----
    {
        "name": "list_institute_modules",
        "description": (
            "List AICtrlNet Institute education modules. Institute is "
            "v11.1 education-led GTM surface (curriculum taught by the "
            "patent-holder). Returns modules organized by tier and "
            "audience. Returns feature_pending while the Institute "
            "platform service is under development."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {
                    "type": "string",
                    "description": "1 (foundations), 2 (industry vertical), 3 (advanced)",
                },
                "audience": {
                    "type": "string",
                    "description": "smb | enterprise | developer",
                },
            },
        },
    },
    {
        "name": "enroll_in_module",
        "description": (
            "Enroll the caller in an Institute module. Each module ends "
            "with a running workflow in the attendee's HitLai instance — "
            "learning and using are the same motion (v11.1). "
            "Returns feature_pending until Institute ships."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "module_id": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["module_id"],
        },
    },
    {
        "name": "get_certification_status",
        "description": (
            "Get the caller's Institute certification status — completed "
            "modules, in-progress modules, earned certifications, "
            "next-step recommendations. Returns feature_pending until "
            "Institute ships."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ---- Wave 7 B1.3 Community-tier Personal Agent tools ----
    {
        "name": "get_personal_agent_config",
        "description": (
            "Get the caller's personal agent config (autonomy level, "
            "connected external agents, attached workflows, preferences). "
            "Community-tier; everyone has a personal agent."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_personal_workflow",
        "description": (
            "Attach a workflow to the caller's personal agent. The workflow "
            "runs at personal scope; to share with the org use "
            "promote_personal_workflow (business+)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "role": {"type": "string", "default": "primary"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["workflow_id"],
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
    # ---- Wave 4: Channels + Notifications (business) ----
    {
        "name": "send_channel_message",
        "description": (
            "Send a message through a linked external channel (Slack, "
            "Discord, Telegram, WhatsApp, Twilio, Email). Enforces channel "
            "ownership — cross-tenant sends rejected."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel_type": {"type": "string"},
                "message": {"type": "string"},
                "channel_user_id": {
                    "type": "string",
                    "description": "Optional specific recipient (defaults to caller's linked account on that channel)",
                },
            },
            "required": ["channel_type", "message"],
        },
    },
    {
        "name": "list_notifications",
        "description": "List the caller's notifications.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "unread_only": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 50, "maximum": 200},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "mark_notification_read",
        "description": "Mark a notification as read.",
        "inputSchema": {
            "type": "object",
            "properties": {"notification_id": {"type": "string"}},
            "required": ["notification_id"],
        },
    },
    # ---- Wave 4: Agents ----
    {
        "name": "list_agents",
        "description": (
            "List AI agents in the caller's registry. v11 '43 agents' claim "
            "— addressable via MCP."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_type": {"type": "string"},
                "enabled_only": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "get_agent_capabilities",
        "description": "Get an agent's capability catalog (tools it can call).",
        "inputSchema": {
            "type": "object",
            "properties": {"agent_id": {"type": "string"}},
            "required": ["agent_id"],
        },
    },
    {
        "name": "set_agent_autonomy",
        "description": (
            "Set an agent's autonomy level 0-100. Maps to v11 six autonomy "
            "phases. Same semantics as set_workflow_autonomy, scoped to the "
            "agent."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "autonomy_level": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["agent_id", "autonomy_level"],
        },
    },
    {
        "name": "execute_agent",
        "description": (
            "Execute an agent with a prompt + optional input payload. "
            "Returns execution id; poll via get_execution_status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "prompt": {"type": "string"},
                "input_data": {"type": "object"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["agent_id", "prompt"],
        },
    },
    # ---- Wave 4: LLM Registry ----
    {
        "name": "list_llm_models",
        "description": (
            "List LLM providers + models registered in the platform. v11 "
            "'every AI model' claim — model-independence is now visible to "
            "Claude through the MCP surface."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string"},
                "enabled_only": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "get_llm_recommendation",
        "description": (
            "Recommend an LLM model for a task based on cost, capability, "
            "latency, and policy constraints."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_type": {"type": "string"},
                "constraints": {
                    "type": "object",
                    "description": "Optional constraints (max_cost, max_latency, required_capabilities, etc.)",
                },
            },
            "required": ["task_type"],
        },
    },
    # ---- Wave 4: Living Platform — Patterns ----
    {
        "name": "list_pattern_candidates",
        "description": (
            "List learned operation patterns the platform has detected "
            "that could be promoted to workflow templates. v11 'platform "
            "that learns your operations' — read side."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "candidate|promoted|rejected"},
                "min_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "limit": {"type": "integer", "default": 50, "maximum": 200},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "promote_pattern_to_template",
        "description": (
            "Promote a learned pattern into a reusable workflow template. "
            "v11 'platform that learns your operations' — write side."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern_id": {"type": "string"},
                "template_name": {"type": "string"},
                "category": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["pattern_id"],
        },
    },
    # ---- Wave 4: Living Platform — Org Discovery ----
    {
        "name": "org_discovery_scan",
        "description": (
            "Trigger an org discovery scan — profiles the caller's business "
            "from integrated data sources. Long-running: returns a job id; "
            "poll get_org_landscape for results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Data sources to scan (email, calendar, slack, etc.)",
                },
                "idempotency_key": {"type": "string"},
            },
        },
    },
    {
        "name": "get_org_landscape",
        "description": (
            "Return the latest org-landscape profile — departments, key "
            "people, systems, processes. Result of org_discovery_scan."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_org_recommendations",
        "description": (
            "Given the org landscape, recommend workflows + adapters + "
            "agents the caller should configure. Product surface of v11 "
            "'Always visibility' pillar."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "focus": {"type": "string", "description": "Optional focus area: sales, marketing, ops, etc."},
            },
        },
    },
    # ---- Wave 4: Living Platform — Company Automation (v11 hero) ----
    {
        "name": "automate_company",
        "description": (
            "Generate a complete automation plan for the caller's company. "
            "v11 '24 hours to a running company' magic moment. Takes a "
            "natural-language request and an optional industry override — "
            "the platform auto-detects industry and loads the matching "
            "industry pack (41 available: healthcare, finance, ecommerce, "
            "cybersecurity, etc.) which contributes compliance rules, "
            "recommended workflows + agents, integrations, and KPIs. "
            "Returns the activation plan id + workflow/agent summary; "
            "poll get_company_automation_status for execution progress."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "Natural-language automation request (e.g. 'automate our 50-person marketing SaaS')",
                },
                "industry": {
                    "type": "string",
                    "description": (
                        "Optional industry override — id or alias "
                        "(healthcare, finance, ecommerce, etc.). If omitted, "
                        "the orchestrator detects from the request text. "
                        "Call list_industry_packs to see options."
                    ),
                },
                "organization_id": {
                    "type": "string",
                    "description": "Existing organization UUID, or omit to create a new one",
                },
                "use_bodaty_template": {
                    "type": "boolean",
                    "default": True,
                    "description": "Start from Bodaty's opinionated template",
                },
                "idempotency_key": {"type": "string"},
            },
            "required": ["request"],
        },
    },
    {
        "name": "get_company_automation_status",
        "description": (
            "Poll the status of a company activation plan. Returns status "
            "(generating | approved | executing | completed | failed), "
            "progress, and the full plan payload once ready."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"plan_id": {"type": "string"}},
            "required": ["plan_id"],
        },
    },
    # ---- Wave 4: Industry Packs (horizontal-first, vertical-enabled) ----
    {
        "name": "list_industry_packs",
        "description": (
            "List available industry packs — vertical-specific knowledge "
            "bundles (compliance, workflows, agents, integrations, KPIs, "
            "ROI) that customize company automation. 41 packs shipped "
            "(healthcare, finance, ecommerce, cybersecurity, edtech, "
            "government, etc.). Use the returned id as automate_company's "
            "`industry` param."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "detect_industry",
        "description": (
            "Detect the best-fit industry pack for a piece of text "
            "(e.g. a company description). Returns the pack id or null "
            "if no match. Uses the same keyword/alias detection the "
            "automate_company orchestrator uses internally."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    # ---- Wave 4: Quality Verification ----
    {
        "name": "verify_quality",
        "description": (
            "Run a quality verification against an AI-generated output "
            "(content, workflow plan, or adapter spec). Returns pass/fail "
            "with scored dimensions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "content_type": {
                    "type": "string",
                    "enum": ["text", "code", "workflow", "adapter"],
                    "default": "text",
                },
                "standards": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["content"],
        },
    },
    # ==== Wave 7 B1.1 — MCP Client (connect to external MCP servers) ====
    {
        "name": "register_mcp_server",
        "description": (
            "Register an external MCP server so AICtrlNet can discover + "
            "invoke its tools. Claude becomes the governed gateway to "
            "the broader MCP ecosystem."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "transport": {"type": "string", "enum": ["http", "sse", "stdio"], "default": "http"},
                "url": {"type": "string"},
                "command": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
                "api_key": {"type": "string"},
            },
            "required": ["name", "transport"],
        },
    },
    {
        "name": "discover_mcp_server_tools",
        "description": "List tools advertised by a registered external MCP server.",
        "inputSchema": {
            "type": "object",
            "properties": {"server_id": {"type": "string"}},
            "required": ["server_id"],
        },
    },
    {
        "name": "invoke_external_mcp_tool",
        "description": (
            "Call a tool on a registered external MCP server. The AICtrlNet "
            "governance pipeline still applies — metering, audit log, rate "
            "bucket — so external invocations are governed like native ones."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "tool_name": {"type": "string"},
                "arguments": {"type": "object"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["server_id", "tool_name"],
        },
    },
    {
        "name": "list_registered_mcp_servers",
        "description": "List external MCP servers the caller's tenant has registered.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "unregister_mcp_server",
        "description": "Unregister + disconnect an external MCP server.",
        "inputSchema": {
            "type": "object",
            "properties": {"server_id": {"type": "string"}},
            "required": ["server_id"],
        },
    },
    # ==== Wave 7 B1.2 — Credential Management ====
    {
        "name": "create_credential",
        "description": (
            "Store a credential (API token, OAuth2 refresh token, DB URL, "
            "etc.) for later use by adapters. Value is encrypted at rest "
            "per the backend (vault / database / file) configuration."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "description": "e.g. 'slack', 'stripe', 'custom'"},
                "credential_id": {"type": "string", "description": "Human-friendly id unique per platform"},
                "credentials": {"type": "object", "description": "Credential fields (platform-specific)"},
            },
            "required": ["platform", "credential_id", "credentials"],
        },
    },
    {
        "name": "list_credentials",
        "description": (
            "List credential metadata (platform + credential_id) the "
            "caller's tenant has stored. Secret values are never returned."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string"},
            },
        },
    },
    {
        "name": "get_credential",
        "description": (
            "Return credential metadata + (optionally) the credential "
            "fields. Prefer setting ``reveal=false`` to fetch only metadata "
            "for audit trails."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string"},
                "credential_id": {"type": "string"},
                "reveal": {"type": "boolean", "default": False},
            },
            "required": ["platform", "credential_id"],
        },
    },
    {
        "name": "delete_credential",
        "description": "Delete a stored credential.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string"},
                "credential_id": {"type": "string"},
            },
            "required": ["platform", "credential_id"],
        },
    },
    {
        "name": "rotate_credential",
        "description": (
            "Replace a credential's value while keeping the same id. "
            "Increments ``rotation_count`` on the credential metadata so "
            "audit logs can correlate rotations. Returns feature_pending "
            "if the credential backend doesn't support rotation yet."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string"},
                "credential_id": {"type": "string"},
                "new_credentials": {"type": "object"},
            },
            "required": ["platform", "credential_id", "new_credentials"],
        },
    },
    {
        "name": "validate_credential",
        "description": (
            "Verify a stored credential is still usable (ping the upstream "
            "service). Returns feature_pending if the backend doesn't "
            "support validation yet."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string"},
                "credential_id": {"type": "string"},
            },
            "required": ["platform", "credential_id"],
        },
    },
    # ==== Wave 7 B1.3 — Personal Agent Hub (Business+ tools) ====
    {
        "name": "update_personal_agent_config",
        "description": (
            "Update the caller's personal agent config — autonomy level, "
            "connected external agents (BYOA/OpenClaw), preferences."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "autonomy_level": {"type": "integer", "minimum": 0, "maximum": 100},
                "autonomy_locked": {"type": "boolean"},
                "external_agents": {"type": "array", "items": {"type": "object"}},
                "preferences": {"type": "object"},
            },
        },
    },
    {
        "name": "get_personal_agent_activity",
        "description": (
            "Return the caller's personal agent activity feed — recent "
            "actions, decisions, approvals. Useful for audit trails."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50, "maximum": 500},
                "since": {"type": "string", "format": "date-time"},
            },
        },
    },
    {
        "name": "connect_external_agent",
        "description": (
            "Register an external agent (BYOA — e.g. OpenClaw, custom "
            "A2A agent) with the caller's personal agent config. The "
            "external agent then appears in orchestration."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "agent_type": {"type": "string", "description": "e.g. openclaw, a2a, custom"},
                "endpoint": {"type": "string"},
                "credentials_ref": {
                    "type": "string",
                    "description": "credential_id for this external agent",
                },
            },
            "required": ["agent_id", "agent_type"],
        },
    },
    {
        "name": "promote_personal_workflow",
        "description": (
            "Promote a personal workflow to org scope so others in the "
            "tenant can use it. Business+ (feature_pending — org-promotion "
            "service not yet factored out)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "visibility": {
                    "type": "string",
                    "enum": ["org", "public"],
                    "default": "org",
                },
            },
            "required": ["workflow_id"],
        },
    },
    # ==== Wave 7 B1.4 — Marketplace publishing + composition ====
    # (all feature_pending — MarketplaceService lacks publish/compose/sync methods)
    {
        "name": "list_org_marketplace_items",
        "description": (
            "Browse the caller's org-scoped marketplace (templates + "
            "workflows + adapters published within the tenant)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "limit": {"type": "integer", "default": 50, "maximum": 500},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "publish_to_org_marketplace",
        "description": (
            "Publish a template / workflow / adapter to the org marketplace. "
            "Returns feature_pending — MarketplaceService.publish method is "
            "not yet implemented (Feature #19 P4: Planned)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_type": {
                    "type": "string",
                    "enum": ["template", "workflow", "adapter"],
                },
                "item_id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "visibility": {
                    "type": "string",
                    "enum": ["org", "public"],
                    "default": "org",
                },
            },
            "required": ["item_type", "item_id"],
        },
    },
    {
        "name": "compose_marketplace_items",
        "description": (
            "Compose multiple marketplace items into a new composite "
            "template. feature_pending — composition engine not yet shipped."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_ids": {"type": "array", "items": {"type": "string"}},
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["item_ids", "name"],
        },
    },
    {
        "name": "sync_public_marketplace_updates",
        "description": (
            "Pull in any public marketplace updates. feature_pending — "
            "public registry sync not yet shipped."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ==== Wave 7 B1.5 — External platform native execution ====
    {
        "name": "execute_n8n_workflow",
        "description": (
            "Execute a workflow on a registered n8n instance. Requires a "
            "stored n8n credential. Native execution — cheaper than "
            "browser_execute fallback."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "credential_id": {"type": "string"},
                "workflow_id": {"type": "string"},
                "input_data": {"type": "object"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["credential_id", "workflow_id"],
        },
    },
    {
        "name": "execute_zapier_zap",
        "description": (
            "Trigger a Zapier zap via stored Zapier webhook credential. "
            "Native execution, no browser."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "credential_id": {"type": "string"},
                "zap_id": {"type": "string"},
                "input_data": {"type": "object"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["credential_id", "zap_id"],
        },
    },
    {
        "name": "execute_make_scenario",
        "description": "Run a Make (Integromat) scenario natively.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "credential_id": {"type": "string"},
                "scenario_id": {"type": "string"},
                "input_data": {"type": "object"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["credential_id", "scenario_id"],
        },
    },
    {
        "name": "execute_power_automate_flow",
        "description": "Run a Microsoft Power Automate flow natively.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "credential_id": {"type": "string"},
                "flow_id": {"type": "string"},
                "input_data": {"type": "object"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["credential_id", "flow_id"],
        },
    },
    # ==== Wave 7 B1.6 — OpenClaw + A2A delegation ====
    {
        "name": "evaluate_runtime_action",
        "description": (
            "Run a pre-action evaluation through the OpenClaw runtime "
            "gateway — check if a proposed agent action is allowed "
            "given current policies. feature_pending — RuntimeGatewayService "
            "not factored out yet; fallback attempts via agent_runtime_bridge."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "action": {"type": "string"},
                "context": {"type": "object"},
            },
            "required": ["agent_id", "action"],
        },
    },
    {
        "name": "get_delegation_chain",
        "description": (
            "Return the delegation chain for a given agent invocation "
            "(who delegated what to whom). Useful for audit of A2A flows."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"invocation_id": {"type": "string"}},
            "required": ["invocation_id"],
        },
    },
    {
        "name": "list_a2a_agents",
        "description": "List agents discoverable via A2A protocol.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "capabilities": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    # ==== Wave 7 B1.7 — Pods + Swarm Intelligence ====
    {
        "name": "form_pod",
        "description": (
            "Form a pod — a coordinated group of agents that collaborate "
            "on a shared task. v11 'pod formation' claim."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "agent_ids": {"type": "array", "items": {"type": "string"}},
                "objective": {"type": "string"},
                "collaboration_contract": {"type": "object"},
            },
            "required": ["agent_ids", "objective"],
        },
    },
    {
        "name": "list_pods",
        "description": "List formed pods in the caller's tenant.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "limit": {"type": "integer", "default": 50, "maximum": 200},
            },
        },
    },
    {
        "name": "get_pod_status",
        "description": "Inspect a pod's current state + member agent statuses.",
        "inputSchema": {
            "type": "object",
            "properties": {"pod_id": {"type": "string"}},
            "required": ["pod_id"],
        },
    },
    {
        "name": "dispatch_swarm",
        "description": (
            "Dispatch a swarm pattern — multiple agents working "
            "independently on variations of a task, results combined. "
            "Useful for parallel research / brainstorming."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_ids": {"type": "array", "items": {"type": "string"}},
                "task": {"type": "string"},
                "aggregation_strategy": {
                    "type": "string",
                    "enum": ["best_of_n", "ensemble", "vote"],
                    "default": "best_of_n",
                },
                "idempotency_key": {"type": "string"},
            },
            "required": ["agent_ids", "task"],
        },
    },
    # ==== Wave 7 B1.8 — AI Framework Cascading + Matching ====
    {
        "name": "get_framework_cascade",
        "description": (
            "Show the current priority order for AI framework selection "
            "(LangChain -> AutoGPT -> CrewAI -> AutoGen -> Semantic Kernel "
            "or tenant override). Features-doc flagship claim."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_framework_priority",
        "description": (
            "Override the framework cascade order for this tenant. "
            "Useful when a specific framework handles the workload better."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "priority": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of framework names",
                },
            },
            "required": ["priority"],
        },
    },
    {
        "name": "get_execution_framework_trace",
        "description": (
            "For a given workflow/agent execution, show which framework "
            "actually handled it + any cascade attempts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"execution_id": {"type": "string"}},
            "required": ["execution_id"],
        },
    },
    {
        "name": "match_agents_to_task",
        "description": (
            "Given a task description, return the top-k best-matching "
            "agents + confidence scores (ML-powered cosine similarity "
            "over capability embeddings). Features-doc 'ML-Powered "
            "Matching' claim."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "top_k": {"type": "integer", "default": 3, "maximum": 10},
            },
            "required": ["task"],
        },
    },
    # ==== Wave 7 B1.9 — Activity Timeline + Operations Center ====
    {
        "name": "get_activity_timeline",
        "description": (
            "Unified activity feed — recent workflow executions, agent "
            "actions, approvals, pod events, audit entries. Operations "
            "center view."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "format": "date-time"},
                "limit": {"type": "integer", "default": 100, "maximum": 500},
                "event_types": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "get_operations_status",
        "description": (
            "High-level operations dashboard — pods active, workflows "
            "running, pending approvals, active alerts."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ==== Wave 7 B2.1 — Cost Analysis ====
    {
        "name": "get_cost_analytics",
        "description": "Cost analytics for the caller's tenant — LLM tokens, browser-actions, platform API calls.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "period": {"type": "string", "enum": ["day", "week", "month"], "default": "month"},
                "breakdown_by": {"type": "string", "enum": ["provider", "workflow", "agent"]},
            },
        },
    },
    {
        "name": "get_platform_cost_estimate",
        "description": "Estimate cost of running a workflow on each supported platform — pick the cheapest.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "input_size": {"type": "integer"},
            },
            "required": ["workflow_id"],
        },
    },
    {
        "name": "optimize_workflow_cost",
        "description": "Analyze a workflow and suggest cost-reducing edits (model downgrades, prompt compaction, platform switches).",
        "inputSchema": {
            "type": "object",
            "properties": {"workflow_id": {"type": "string"}},
            "required": ["workflow_id"],
        },
    },
    # ==== Wave 7 B2.2 — SLA Management (Business) ====
    {
        "name": "create_sla",
        "description": "Define an SLA target for a resource (workflow/agent/tool) + metric + window.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "resource_type": {"type": "string"},
                "resource_id": {"type": "string"},
                "metric": {"type": "string", "description": "e.g. latency_p95, success_rate"},
                "target_value": {"type": "number"},
                "window_seconds": {"type": "integer", "default": 3600},
                "severity": {"type": "string", "default": "medium"},
            },
            "required": ["name", "resource_type", "metric", "target_value"],
        },
    },
    {
        "name": "list_slas",
        "description": "List SLAs defined in the caller's tenant.",
        "inputSchema": {
            "type": "object",
            "properties": {"enabled_only": {"type": "boolean", "default": True}},
        },
    },
    {
        "name": "get_sla_status",
        "description": "Current status of an SLA — observed vs target, in-breach flag.",
        "inputSchema": {
            "type": "object",
            "properties": {"sla_id": {"type": "string"}},
            "required": ["sla_id"],
        },
    },
    # ==== Wave 7 B2.3 — Workflow versioning ====
    {
        "name": "list_workflow_versions",
        "description": "List version history of a workflow.",
        "inputSchema": {
            "type": "object",
            "properties": {"workflow_id": {"type": "string"}},
            "required": ["workflow_id"],
        },
    },
    {
        "name": "get_workflow_version",
        "description": "Fetch a specific workflow version (definition snapshot).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "version": {"type": "integer"},
            },
            "required": ["workflow_id", "version"],
        },
    },
    {
        "name": "rollback_workflow",
        "description": "Revert a workflow to a previous version. Current definition becomes a new version.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "target_version": {"type": "integer"},
            },
            "required": ["workflow_id", "target_version"],
        },
    },
    {
        "name": "compare_workflow_versions",
        "description": "Diff two versions of a workflow (node-level + edge-level changes).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "version_a": {"type": "integer"},
                "version_b": {"type": "integer"},
            },
            "required": ["workflow_id", "version_a", "version_b"],
        },
    },
    # ==== Wave 7 B2.5 — Template CRUD ====
    {
        "name": "create_template",
        "description": "Create a new workflow template (from scratch or from an existing workflow).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "category": {"type": "string"},
                "definition": {"type": "object"},
                "parameters_schema": {"type": "object"},
                "source_workflow_id": {"type": "string"},
            },
            "required": ["name", "category"],
        },
    },
    {
        "name": "update_template",
        "description": "Update a template (creates a new version automatically).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "template_id": {"type": "string"},
                "definition": {"type": "object"},
                "parameters_schema": {"type": "object"},
                "change_summary": {"type": "string"},
            },
            "required": ["template_id"],
        },
    },
    {
        "name": "delete_template",
        "description": "Delete a template (soft delete — published copies remain).",
        "inputSchema": {
            "type": "object",
            "properties": {"template_id": {"type": "string"}},
            "required": ["template_id"],
        },
    },
    # ==== Wave 7 B2.6 — MFA read (Business) ====
    {
        "name": "get_mfa_status",
        "description": "Read the caller's MFA enrollment status.",
        "inputSchema": {"type": "object", "properties": {}},
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
    # ---- Wave 6: Analytics ----
    {
        "name": "query_analytics",
        "description": (
            "Execute an analytics query across tasks, workflows, agents, "
            "resource usage, API calls, or errors. Flexible filter/group-by. "
            "Enterprise tier."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "metric_type": {
                    "type": "string",
                    "description": "task | workflow | agent | resource | api | error | cost",
                },
                "filters": {"type": "object"},
                "group_by": {"type": "array", "items": {"type": "string"}},
                "time_range": {"type": "string", "description": "e.g. 24h, 7d, 30d, 90d"},
            },
            "required": ["metric_type"],
        },
    },
    {
        "name": "get_dashboard_metrics",
        "description": (
            "Return the Enterprise dashboard snapshot — high-level "
            "tenant-scoped metrics that power the HitLai Enterprise home."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_metric_trends",
        "description": (
            "Trend data for a given metric type over a time range. Good "
            "for dashboards, anomaly detection, CFO reporting."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "metric_type": {"type": "string"},
                "time_range": {"type": "string", "default": "30d"},
            },
            "required": ["metric_type"],
        },
    },
    # ---- Wave 6: Audit ----
    {
        "name": "get_audit_logs",
        "description": (
            "Query enterprise audit logs with full filter set: resource "
            "type/id, action, severity, success, user, date range, "
            "pagination, ordering. Tenant-scoped."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string"},
                "resource_id": {"type": "string"},
                "action": {"type": "string"},
                "severity": {"type": "string"},
                "success": {"type": "boolean"},
                "user_id": {"type": "string"},
                "start_date": {"type": "string", "format": "date-time"},
                "end_date": {"type": "string", "format": "date-time"},
                "limit": {"type": "integer", "default": 100, "maximum": 1000},
                "offset": {"type": "integer", "default": 0},
                "order_by": {"type": "string", "default": "timestamp"},
                "order_direction": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
            },
        },
    },
    {
        "name": "get_audit_summary",
        "description": (
            "Summary of audit activity over the last N days — by action, "
            "severity, actor. Useful for compliance officers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "default": 7, "minimum": 1, "maximum": 90},
            },
        },
    },
    # ---- Wave 6: Compliance ----
    {
        "name": "run_compliance_check",
        "description": (
            "Run a compliance check across one or more standards (GDPR, "
            "HIPAA, SOC2, CCPA, ISO27001). Returns per-standard status + "
            "failing controls + remediation hints."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "standards": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Standards to check (default: all configured)",
                },
                "scope": {
                    "type": "object",
                    "description": "Optional scope (workflow_id, tenant_id, resource_type)",
                },
            },
        },
    },
    {
        "name": "list_compliance_standards",
        "description": "List supported compliance standards + their controls catalog.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_enterprise_risk_assessment",
        "description": (
            "Enterprise-tier risk assessment summary — per-tenant risk "
            "scores across AI governance dimensions. Aggregates the "
            "business-tier risk data for the CISO view."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ---- Wave 6: Organizations + Tenants ----
    {
        "name": "list_organizations",
        "description": "List organizations the caller can access.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_tenants",
        "description": "List tenants visible to the caller (Enterprise admin).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 100, "maximum": 500},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    # ---- Wave 6: Federated Knowledge + Cross-Tenant ----
    {
        "name": "federated_knowledge_query",
        "description": (
            "Query the federated knowledge base across tenants the caller "
            "has permission to see. Governance-preserving — tenant "
            "boundaries + RLS enforced."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "tenant_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tenant scope; defaults to all accessible",
                },
                "limit": {"type": "integer", "default": 10, "maximum": 100},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_cross_tenant_insights",
        "description": (
            "Return cross-tenant comparison insights — workflow metrics, "
            "permission summaries, shared-resource usage. Fleet-wide "
            "governance view."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "metric_type": {
                    "type": "string",
                    "enum": ["workflows", "permissions", "audit", "summary"],
                    "default": "summary",
                },
                "time_range": {"type": "string", "default": "30d"},
            },
        },
    },
    # ---- Wave 6: Fleet Management ----
    {
        "name": "list_fleet_agents",
        "description": (
            "List fleet runtimes — agents registered across the tenant's "
            "fleet. v11 'fleet-wide autonomy management' read surface."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "limit": {"type": "integer", "default": 100, "maximum": 500},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "get_fleet_autonomy_summary",
        "description": (
            "Fleet-wide autonomy-level distribution + risk summary. Lets "
            "CISOs see 'are my agents operating at the right autonomy "
            "level' at a glance."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ---- Wave 6: License Management ----
    {
        "name": "get_license_status",
        "description": (
            "Return the caller's Enterprise license status — active/"
            "expiring/expired, seat counts, feature entitlements."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_license_entitlements",
        "description": (
            "List the license entitlements (feature flags, limits) "
            "available under the current Enterprise plan."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ==== Wave 7 B1.6 — Runtime gateway (Enterprise) ====
    {
        "name": "register_runtime_webhook",
        "description": (
            "Register an HMAC-signed bidirectional webhook with the "
            "OpenClaw runtime gateway. feature_pending — runtime gateway "
            "service not yet factored out."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "url": {"type": "string"},
                "events": {"type": "array", "items": {"type": "string"}},
                "secret": {"type": "string"},
            },
            "required": ["name", "url"],
        },
    },
    {
        "name": "list_runtime_webhooks",
        "description": "List registered runtime-gateway webhooks.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ==== Wave 7 B2.1 Enterprise cost ====
    {
        "name": "analyze_cost_trends",
        "description": "Long-range cost trend analysis across tenants (Enterprise fleet view).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "default": 90, "maximum": 365},
            },
        },
    },
    # ==== Wave 7 B2.2 Enterprise SLA ====
    {
        "name": "get_sla_violations",
        "description": "List SLA violations (breaches) for the tenant.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "resolved": {"type": "boolean"},
                "severity": {"type": "string"},
                "limit": {"type": "integer", "default": 100, "maximum": 1000},
            },
        },
    },
    {
        "name": "get_sla_metrics",
        "description": "Aggregate SLA-compliance metrics for reporting.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "period": {"type": "string", "enum": ["day", "week", "month"], "default": "month"},
            },
        },
    },
    # ==== Wave 7 B2.4 RBAC ====
    {
        "name": "list_roles",
        "description": "List roles defined in the tenant.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_role",
        "description": "Get a role + its permissions.",
        "inputSchema": {
            "type": "object",
            "properties": {"role_id": {"type": "string"}},
            "required": ["role_id"],
        },
    },
    {
        "name": "create_role",
        "description": "Create a new role with a set of resource+action permissions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "permissions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "resource": {"type": "string"},
                            "action": {"type": "string"},
                        },
                        "required": ["resource", "action"],
                    },
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "grant_role",
        "description": "Grant a role to a user.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "role_id": {"type": "string"},
            },
            "required": ["user_id", "role_id"],
        },
    },
    {
        "name": "revoke_role",
        "description": "Revoke a user's role.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "role_id": {"type": "string"},
            },
            "required": ["user_id", "role_id"],
        },
    },
    {
        "name": "list_permissions",
        "description": "List the permission catalog (all resource+action pairs known to the system).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ==== Wave 7 B2.6 OAuth2 admin (Enterprise) ====
    {
        "name": "list_oauth2_clients",
        "description": "List OAuth2 machine-credential clients in the tenant.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "revoke_oauth2_token",
        "description": "Emergency-revoke an OAuth2 access token (kill a leaked credential).",
        "inputSchema": {
            "type": "object",
            "properties": {"token_id": {"type": "string"}},
            "required": ["token_id"],
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
    # Wave 4: Tasks
    "create_task": ["write:tasks"],
    "list_tasks": ["read:tasks"],
    "get_task": ["read:tasks"],
    "update_task": ["write:tasks"],
    "complete_task": ["write:tasks"],
    # Wave 4: Memory
    "get_memory": ["read:memory"],
    "set_memory": ["write:memory"],
    "delete_memory": ["write:memory"],
    # Wave 4: Conversations + Channels
    "list_conversations": ["read:conversations"],
    "get_conversation": ["read:conversations"],
    "list_linked_channels": ["read:conversations"],
    "request_channel_link_code": ["write:conversations"],
    "unlink_channel": ["write:conversations"],
    "send_channel_message": ["write:messaging"],
    "list_notifications": ["read:notifications"],
    "mark_notification_read": ["write:notifications"],
    # Wave 4: Knowledge
    "query_knowledge": ["read:knowledge"],
    "suggest_next_actions": ["read:knowledge"],
    "get_capabilities_summary": ["read:knowledge"],
    # Wave 4: Templates
    "search_templates": ["read:templates"],
    "instantiate_template": ["write:workflows"],
    # Wave 4: Files
    "upload_file": ["write:files"],
    "list_staged_files": ["read:files"],
    "get_staged_file": ["read:files"],
    # Wave 4: Agents + LLM
    "list_agents": ["read:agents"],
    "get_agent_capabilities": ["read:agents"],
    "set_agent_autonomy": ["write:agents"],
    "execute_agent": ["write:agents"],
    "list_llm_models": ["read:llm"],
    "get_llm_recommendation": ["read:llm"],
    # Wave 4: Living Platform
    "list_pattern_candidates": ["read:patterns"],
    "promote_pattern_to_template": ["write:patterns"],
    "org_discovery_scan": ["write:org"],
    "get_org_landscape": ["read:org"],
    "get_org_recommendations": ["read:org"],
    "automate_company": ["write:company"],
    "get_company_automation_status": ["read:org"],
    "list_industry_packs": ["read:org"],
    "detect_industry": ["read:org"],
    "verify_quality": ["write:quality"],
    # Wave 4: Data Quality
    "assess_data_quality": ["read:workflows"],
    "list_quality_dimensions": ["read:workflows"],
    # Wave 5: Institute
    "list_institute_modules": ["read:institute"],
    "enroll_in_module": ["write:institute"],
    "get_certification_status": ["read:institute"],
    # Wave 6: Enterprise admin — Analytics
    "query_analytics": ["read:analytics"],
    "get_dashboard_metrics": ["read:analytics"],
    "get_metric_trends": ["read:analytics"],
    # Wave 6: Audit
    "get_audit_logs": ["read:audit"],
    "get_audit_summary": ["read:audit"],
    # Wave 6: Compliance
    "run_compliance_check": ["read:compliance"],
    "list_compliance_standards": ["read:compliance"],
    "get_enterprise_risk_assessment": ["read:compliance"],
    # Wave 6: Organizations + Tenants
    "list_organizations": ["read:compliance"],
    "list_tenants": ["read:compliance"],
    # Wave 6: Federated knowledge + cross-tenant
    "federated_knowledge_query": ["read:knowledge"],
    "get_cross_tenant_insights": ["read:analytics"],
    # Wave 6: Fleet management
    "list_fleet_agents": ["read:fleet"],
    "get_fleet_autonomy_summary": ["read:fleet"],
    # Wave 6: License management
    "get_license_status": ["read:license"],
    "list_license_entitlements": ["read:license"],
    # Wave 7 B1.1: MCP Client
    "register_mcp_server": ["write:mcp_client"],
    "discover_mcp_server_tools": ["read:mcp_client"],
    "invoke_external_mcp_tool": ["write:mcp_client"],
    "list_registered_mcp_servers": ["read:mcp_client"],
    "unregister_mcp_server": ["write:mcp_client"],
    # Wave 7 B1.2: Credentials
    "create_credential": ["write:credentials"],
    "list_credentials": ["read:credentials"],
    "get_credential": ["read:credentials"],
    "delete_credential": ["write:credentials"],
    "rotate_credential": ["write:credentials"],
    "validate_credential": ["write:credentials"],
    # Wave 7 B1.3: Personal Agent
    "get_personal_agent_config": ["read:personal_agent"],
    "update_personal_agent_config": ["write:personal_agent"],
    "get_personal_agent_activity": ["read:personal_agent"],
    "connect_external_agent": ["write:personal_agent"],
    "create_personal_workflow": ["write:workflows"],
    "promote_personal_workflow": ["write:workflows"],
    # Wave 7 B1.4: Marketplace
    "list_org_marketplace_items": ["read:marketplace"],
    "publish_to_org_marketplace": ["write:marketplace"],
    "compose_marketplace_items": ["write:marketplace"],
    "sync_public_marketplace_updates": ["write:marketplace"],
    # Wave 7 B1.5: Platform native execution
    "execute_n8n_workflow": ["write:platforms"],
    "execute_zapier_zap": ["write:platforms"],
    "execute_make_scenario": ["write:platforms"],
    "execute_power_automate_flow": ["write:platforms"],
    # Wave 7 B1.6: OpenClaw + A2A
    "evaluate_runtime_action": ["write:runtime_gateway"],
    "get_delegation_chain": ["read:runtime_gateway"],
    "list_a2a_agents": ["read:runtime_gateway"],
    "register_runtime_webhook": ["write:runtime_gateway"],
    "list_runtime_webhooks": ["read:runtime_gateway"],
    # Wave 7 B1.7: Pods + Swarm
    "form_pod": ["write:pods"],
    "list_pods": ["read:pods"],
    "get_pod_status": ["read:pods"],
    "dispatch_swarm": ["write:pods"],
    # Wave 7 B1.8: Framework cascading + ML matching
    "get_framework_cascade": ["read:llm"],
    "set_framework_priority": ["write:agents"],
    "get_execution_framework_trace": ["read:analytics"],
    "match_agents_to_task": ["read:agents"],
    # Wave 7 B1.9: Activity timeline
    "get_activity_timeline": ["read:analytics"],
    "get_operations_status": ["read:analytics"],
    # Wave 7 B2.1: Cost
    "get_cost_analytics": ["read:cost"],
    "get_platform_cost_estimate": ["read:cost"],
    "optimize_workflow_cost": ["write:workflows"],
    "analyze_cost_trends": ["read:cost"],
    # Wave 7 B2.2: SLA
    "create_sla": ["write:sla"],
    "list_slas": ["read:sla"],
    "get_sla_status": ["read:sla"],
    "get_sla_violations": ["read:sla"],
    "get_sla_metrics": ["read:analytics"],
    # Wave 7 B2.3: Workflow versioning
    "list_workflow_versions": ["read:workflows"],
    "get_workflow_version": ["read:workflows"],
    "rollback_workflow": ["write:workflows"],
    "compare_workflow_versions": ["read:workflows"],
    # Wave 7 B2.4: RBAC
    "list_roles": ["read:roles"],
    "get_role": ["read:roles"],
    "create_role": ["write:roles"],
    "grant_role": ["write:roles"],
    "revoke_role": ["write:roles"],
    "list_permissions": ["read:roles"],
    # Wave 7 B2.5: Template CRUD
    "create_template": ["write:templates"],
    "update_template": ["write:templates"],
    "delete_template": ["write:templates"],
    # Wave 7 B2.6: MFA + OAuth2 admin
    "get_mfa_status": ["read:mfa"],
    "list_oauth2_clients": ["read:oauth2"],
    "revoke_oauth2_token": ["write:oauth2"],
}


def get_tools_for_edition() -> list:
    """Return tool definitions available for the current edition.

    Wave 7 A3: Business/Enterprise tools are registered only when BOTH
    their module is importable AND the corresponding env flag is set.
    This is belt-and-suspenders with the plan gate — even if Business
    code leaks into PYTHONPATH on a Community deploy (dev image pushed
    to prod, shared container), the flag must be explicit ``true`` or
    the tools don't register. ``tools/list`` thus doesn't leak tool
    existence across editions.

    Defaults:
    - ``MCP_ENABLE_BUSINESS_TOOLS``: defaults to ``true`` so existing
      Business/Enterprise deployments keep working without config change.
    - ``MCP_ENABLE_ENTERPRISE_TOOLS``: defaults to ``true`` for the same
      reason.

    Set to ``false`` explicitly on pure Community deploys to harden.
    """
    import os

    tools = list(COMMUNITY_TOOLS)

    business_flag = os.environ.get("MCP_ENABLE_BUSINESS_TOOLS", "true").lower() == "true"
    if business_flag:
        try:
            from aictrlnet_business.api.v1.endpoints import agp_evaluation  # noqa: F401
            tools.extend(BUSINESS_TOOLS)
        except ImportError:
            pass
    else:
        try:
            from aictrlnet_business.api.v1.endpoints import agp_evaluation  # noqa: F401
            logger.warning(
                "Business edition code is importable but MCP_ENABLE_BUSINESS_TOOLS=false — "
                "refusing to register Business tools. Set the flag to 'true' to enable."
            )
        except ImportError:
            pass

    enterprise_flag = os.environ.get("MCP_ENABLE_ENTERPRISE_TOOLS", "true").lower() == "true"
    if enterprise_flag:
        try:
            from aictrlnet_enterprise.services.mcp_compliance import MCPComplianceService  # noqa: F401
            tools.extend(ENTERPRISE_TOOLS)
        except ImportError:
            pass
    else:
        try:
            from aictrlnet_enterprise.services.mcp_compliance import MCPComplianceService  # noqa: F401
            logger.warning(
                "Enterprise edition code is importable but MCP_ENABLE_ENTERPRISE_TOOLS=false — "
                "refusing to register Enterprise tools. Set the flag to 'true' to enable."
            )
        except ImportError:
            pass

    return tools
