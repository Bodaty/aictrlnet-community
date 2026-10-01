"""Tool classes and default categories (CONVERSATION_ORCHESTRATION_SPEC.md §7.6).

Every conversational tool declares one class — `discovery`, `read`, `write`,
`long_running` — and the router admits tools by class (R3). This module is the
one place those classes are decided, from what the registry already says about
each tool, so a new tool is classified the moment it is registered:

- `long_running`: can outlive `tool_hard_cap_s`; always a background job (R2).
- `discovery`: describes the platform itself; safe on every non-chat turn.
- `write`: side-effecting in dry-run, destructive, confirmation-gated, or named
  with a mutating verb.
- `read`: everything else. A read tool changes no user object; it may record
  its own result or an audit row (`assess_risk` stores the assessment it
  returns).

It also fills `category` for tools registered without one, from the service
that handles them, so the pruner's category signal reaches every tool.
"""

from typing import Iterable, Mapping, Optional

DISCOVERY = "discovery"
READ = "read"
WRITE = "write"
LONG_RUNNING = "long_running"
TOOL_CLASSES = (DISCOVERY, READ, WRITE, LONG_RUNNING)

LONG_RUNNING_TOOLS = frozenset({"create_workflow", "automate_company", "generate_adapter"})

DISCOVERY_TOOLS = frozenset({
    "search_api_capabilities", "search_documentation", "list_documentation_topics",
    "get_document_content", "get_help", "get_system_status", "list_api_endpoints",
    "get_endpoint_detail", "list_integrations", "get_integration_info",
})

# A tool whose name starts with one of these changes something.
MUTATING_PREFIXES = (
    "add_", "apply_", "approve_", "assign_", "automate_", "cancel_", "change_", "clear_",
    "configure_", "create_", "delegate_", "delete_", "disable_", "downgrade_", "enable_",
    "execute_", "generate_", "grant_", "initiate_", "install_", "instantiate_", "invite_",
    "manage_", "rate_", "refresh_", "reject_", "remove_", "retry_", "revoke_", "rotate_",
    "run_", "schedule_", "send_", "set_", "suspend_", "sync_", "test_", "update_",
    "upgrade_", "warm_",
)

# Writers the name rule cannot see.
EXTRA_WRITE_TOOLS = frozenset({"browser_execute"})

# Mutating-verb names that change nothing (checked handler by handler).
NON_MUTATING_TOOLS = frozenset({
    # Its handler calls AnalyticsService.generate_report, which does not exist
    # (30 Sep 2026; fix queue) — the tool writes nothing because it does nothing.
    "generate_report",
})

# Default category by the handler's service, for tools registered without one.
SERVICE_CATEGORIES = {
    "workflow_service": "workflow",
    "workflow_template_service": "workflow",
    "nlp_service": "workflow",
    "company_automation_orchestrator": "workflow",
    "intelligent_template_discovery": "workflow",
    "agent_service": "agent",
    "agent_execution_service": "agent",
    "pod_service": "agent",
    "intelligent_agent_selector": "agent",
    "task_service": "task_management",
    "ai_governance_service": "governance",
    "analytics_service": "monitoring",
    "system_service": "system",
    "help_service": "system",
    "api_introspection_service": "system",
    "documentation_knowledge_service": "system",
    "adapter_service": "integration",
    "integration_service": "integration",
    "platform_service": "integration",
    "mcp_service": "integration",
    "browser_tool": "browser",
    "file_access": "files",
    "user_memory_service": "personalization",
    "org_discovery_service": "personalization",
    "sso_service": "iam",
    "tenant_service": "iam",
    "federation_service": "iam",
}
# Tools whose handler string names no service.
NAME_CATEGORIES = {
    "list_templates": "workflow",
    "search_templates": "workflow",
    "get_template_detail": "workflow",
    "list_template_categories": "workflow",
    # Dry-run toggles are handled inside the dispatcher itself.
    "set_dry_run_mode": "system",
    "set_agent_dry_run_mode": "agent",
    "set_pod_dry_run_mode": "agent",
}


def classify(tool, side_effect_names: Iterable[str] = ()) -> str:
    name = tool.name
    if name in LONG_RUNNING_TOOLS:
        return LONG_RUNNING
    if (
        name in set(side_effect_names)
        or getattr(tool, "is_destructive", False)
        or getattr(tool, "requires_confirmation", False)
        or (name.startswith(MUTATING_PREFIXES) and name not in NON_MUTATING_TOOLS)
        or name in EXTRA_WRITE_TOOLS
    ):
        return WRITE
    if name in DISCOVERY_TOOLS:
        return DISCOVERY
    return READ


def default_category(tool) -> Optional[str]:
    if getattr(tool, "category", None):
        return tool.category
    if tool.name in NAME_CATEGORIES:
        return NAME_CATEGORIES[tool.name]
    service = (getattr(tool, "handler", "") or "").split(".")[0]
    return SERVICE_CATEGORIES.get(service)


def classify_tools(registry: Mapping[str, object], side_effect_names: Iterable[str] = ()) -> None:
    """Set `tool_class` and fill a missing `category` on every tool in the registry."""
    side_effects = frozenset(side_effect_names)
    for tool in registry.values():
        tool.tool_class = classify(tool, side_effects)
        tool.category = default_category(tool)
