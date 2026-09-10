"""MCP tool titles and behaviour annotations.

Every tool the server lists carries a human ``title`` and the four MCP
annotation hints (``readOnlyHint``, ``destructiveHint``, ``idempotentHint``,
``openWorldHint``). Clients use them to group tools, decide when to ask the
user for confirmation, and (for the Anthropic Connectors Directory) to pass
review: a tool without a title or without the applicable hint is rejected.

Truth lives here, not in the tool dict literals:

- ``TOOL_TITLES`` is explicit per tool. A tool with no title fails
  ``test_mcp_tool_annotations``.
- ``readOnlyHint`` is derived from ``TOOL_SCOPES`` — a tool is read-only
  exactly when every scope it requires is a ``read:`` scope. The scope
  registry is already the enforcement boundary (``tool_executor``), so the
  hint cannot drift from what the server actually allows.
- ``destructiveHint`` is explicit (``DESTRUCTIVE_TOOLS``). The MCP default
  for a write tool is *true*, so additive tools must say ``False`` out loud.
- ``idempotentHint`` is true for read-only tools and for write tools whose
  name prefix is in ``IDEMPOTENT_PREFIXES``; false otherwise (MCP default).
- ``openWorldHint`` is explicit (``OPEN_WORLD_TOOLS``): tools that reach
  systems outside the tenant — the public web, third-party automation
  platforms, external MCP servers, federated peers.

``apply_annotations`` mutates the tool dicts in place so every consumer of
``COMMUNITY_TOOLS`` / ``BUSINESS_TOOLS`` / ``ENTERPRISE_TOOLS`` sees the same
fields.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

TOOL_TITLES: Dict[str, str] = {
    "create_workflow": "Create Workflow",
    "list_workflows": "List Workflows",
    "get_workflow": "Get Workflow",
    "execute_workflow": "Execute Workflow",
    "get_execution_status": "Get Execution Status",
    "list_templates": "List Templates",
    "assess_quality": "Assess Quality",
    "send_message": "Send Message",
    "list_adapters": "List Adapters",
    "get_adapter": "Get Adapter",
    "list_my_adapter_configs": "List My Adapter Configurations",
    "test_adapter_config": "Test Adapter Configuration",
    "nl_to_workflow": "Natural Language to Workflow",
    "analyze_intent": "Analyze Intent",
    "get_workflow_autonomy": "Get Workflow Autonomy",
    "preview_autonomy": "Preview Autonomy",
    "list_api_keys": "List API Keys",
    "get_api_key_usage": "Get API Key Usage",
    "get_subscription": "Get Subscription",
    "get_upgrade_options": "Get Upgrade Options",
    "get_trial_status": "Get Trial Status",
    "get_usage_report": "Get Usage Report",
    "create_task": "Create Task",
    "list_tasks": "List Tasks",
    "get_task": "Get Task",
    "update_task": "Update Task",
    "complete_task": "Complete Task",
    "get_memory": "Get Memory",
    "set_memory": "Set Memory",
    "delete_memory": "Delete Memory",
    "list_conversations": "List Conversations",
    "get_conversation": "Get Conversation",
    "list_linked_channels": "List Linked Channels",
    "request_channel_link_code": "Request Channel Link Code",
    "unlink_channel": "Unlink Channel",
    "query_knowledge": "Query Knowledge Base",
    "suggest_next_actions": "Suggest Next Actions",
    "get_capabilities_summary": "Get Capabilities Summary",
    "search_templates": "Search Templates",
    "instantiate_template": "Instantiate Template",
    "upload_file": "Upload File",
    "list_staged_files": "List Staged Files",
    "get_staged_file": "Get Staged File",
    "assess_data_quality": "Assess Data Quality",
    "list_quality_dimensions": "List Quality Dimensions",
    "list_institute_modules": "List Institute Modules",
    "enroll_in_module": "Enroll in Institute Module",
    "get_certification_status": "Get Certification Status",
    "get_personal_agent_config": "Get Personal Agent Configuration",
    "create_personal_workflow": "Create Personal Workflow",
    "list_block_types": "List Block Types",
    "evaluate_policy": "Evaluate Policy",
    "list_policies": "List Policies",
    "set_workflow_autonomy": "Set Workflow Autonomy",
    "research_api": "Research API",
    "generate_adapter": "Generate Adapter",
    "self_extend": "Self-Extend: Research and Generate Adapter",
    "list_generated_adapters": "List Generated Adapters",
    "get_generated_adapter_status": "Get Generated Adapter Status",
    "get_generated_adapter_source": "Get Generated Adapter Source",
    "approve_adapter": "Approve Adapter",
    "reject_adapter": "Reject Adapter",
    "activate_adapter": "Activate Adapter",
    "browser_execute": "Execute Browser Automation",
    "list_pending_approvals": "List Pending Approvals",
    "get_approval": "Get Approval",
    "approve_request": "Approve Request",
    "reject_request": "Reject Request",
    "list_ai_policies": "List AI Policies",
    "create_policy": "Create Policy",
    "get_ai_audit_logs": "Get AI Audit Logs",
    "list_violations": "List Violations",
    "send_channel_message": "Send Channel Message",
    "list_notifications": "List Notifications",
    "mark_notification_read": "Mark Notification as Read",
    "list_agents": "List Agents",
    "get_agent_capabilities": "Get Agent Capabilities",
    "set_agent_autonomy": "Set Agent Autonomy",
    "execute_agent": "Execute Agent",
    "configure_agent_identity": "Configure Agent Identity",
    "list_llm_models": "List LLM Models",
    "get_llm_recommendation": "Get LLM Recommendation",
    "list_pattern_candidates": "List Pattern Candidates",
    "promote_pattern_to_template": "Promote Pattern to Template",
    "org_discovery_scan": "Scan Organization Landscape",
    "get_org_landscape": "Get Organization Landscape",
    "get_org_recommendations": "Get Organization Recommendations",
    "automate_company": "Automate Company",
    "get_company_automation_status": "Get Company Automation Status",
    "list_industry_packs": "List Industry Packs",
    "detect_industry": "Detect Industry",
    "verify_quality": "Verify Quality",
    "register_mcp_server": "Register MCP Server",
    "discover_mcp_server_tools": "Discover MCP Server Tools",
    "invoke_external_mcp_tool": "Invoke External MCP Tool",
    "list_registered_mcp_servers": "List Registered MCP Servers",
    "unregister_mcp_server": "Unregister MCP Server",
    "create_credential": "Create Credential",
    "list_credentials": "List Credentials",
    "get_credential": "Get Credential",
    "delete_credential": "Delete Credential",
    "rotate_credential": "Rotate Credential",
    "validate_credential": "Validate Credential",
    "update_personal_agent_config": "Update Personal Agent Configuration",
    "get_personal_agent_activity": "Get Personal Agent Activity",
    "connect_external_agent": "Connect External Agent",
    "promote_personal_workflow": "Promote Personal Workflow",
    "list_org_marketplace_items": "List Organization Marketplace Items",
    "publish_to_org_marketplace": "Publish to Organization Marketplace",
    "compose_marketplace_items": "Compose Marketplace Items",
    "sync_public_marketplace_updates": "Sync Public Marketplace Updates",
    "execute_n8n_workflow": "Execute n8n Workflow",
    "execute_zapier_zap": "Execute Zapier Zap",
    "execute_make_scenario": "Execute Make Scenario",
    "execute_power_automate_flow": "Execute Power Automate Flow",
    "evaluate_runtime_action": "Evaluate Runtime Action",
    "get_delegation_chain": "Get Delegation Chain",
    "list_a2a_agents": "List A2A Agents",
    "form_pod": "Form Agent Pod",
    "list_pods": "List Pods",
    "get_pod_status": "Get Pod Status",
    "dispatch_swarm": "Dispatch Agent Swarm",
    "get_framework_cascade": "Get Framework Cascade",
    "set_framework_priority": "Set Framework Priority",
    "get_execution_framework_trace": "Get Execution Framework Trace",
    "match_agents_to_task": "Match Agents to Task",
    "get_activity_timeline": "Get Activity Timeline",
    "get_operations_status": "Get Operations Status",
    "get_cost_analytics": "Get Cost Analytics",
    "get_platform_cost_estimate": "Get Platform Cost Estimate",
    "optimize_workflow_cost": "Optimize Workflow Cost",
    "create_sla": "Create SLA",
    "list_slas": "List SLAs",
    "get_sla_status": "Get SLA Status",
    "list_workflow_versions": "List Workflow Versions",
    "get_workflow_version": "Get Workflow Version",
    "rollback_workflow": "Rollback Workflow",
    "compare_workflow_versions": "Compare Workflow Versions",
    "create_template": "Create Template",
    "update_template": "Update Template",
    "delete_template": "Delete Template",
    "get_mfa_status": "Get MFA Status",
    "list_file_versions": "List File Versions",
    "get_file_version": "Get File Version",
    "generate_signed_file_url": "Generate Signed File URL",
    "delete_file_version": "Delete File Version",
    "get_notification_preferences": "Get Notification Preferences",
    "update_notification_preferences": "Update Notification Preferences",
    "set_channel_notification_rules": "Set Channel Notification Rules",
    "set_notification_frequency": "Set Notification Frequency",
    "list_template_versions": "List Template Versions",
    "configure_update_notifications": "Configure Update Notifications",
    "create_canvas_block": "Create Canvas Block",
    "render_canvas": "Render Canvas",
    "get_org_discovery_status": "Get Organization Discovery Status",
    "get_org_discovery_logs": "Get Organization Discovery Logs",
    "list_discovered_adapters_by_capability": "List Discovered Adapters by Capability",
    "rescan_adapter_registry": "Rescan Adapter Registry",
    "check_compliance": "Check Compliance",
    "query_analytics": "Query Analytics",
    "get_dashboard_metrics": "Get Dashboard Metrics",
    "get_metric_trends": "Get Metric Trends",
    "get_audit_logs": "Get Audit Logs",
    "get_audit_summary": "Get Audit Summary",
    "run_compliance_check": "Run Compliance Check",
    "list_compliance_standards": "List Compliance Standards",
    "get_enterprise_risk_assessment": "Get Enterprise Risk Assessment",
    "list_organizations": "List Organizations",
    "list_tenants": "List Tenants",
    "federated_knowledge_query": "Query Federated Knowledge",
    "get_cross_tenant_insights": "Get Cross-Tenant Insights",
    "list_fleet_agents": "List Fleet Agents",
    "get_fleet_autonomy_summary": "Get Fleet Autonomy Summary",
    "get_license_status": "Get License Status",
    "list_license_entitlements": "List License Entitlements",
    "register_runtime_webhook": "Register Runtime Webhook",
    "list_runtime_webhooks": "List Runtime Webhooks",
    "analyze_cost_trends": "Analyze Cost Trends",
    "get_sla_violations": "Get SLA Violations",
    "get_sla_metrics": "Get SLA Metrics",
    "list_roles": "List Roles",
    "get_role": "Get Role",
    "create_role": "Create Role",
    "grant_role": "Grant Role",
    "revoke_role": "Revoke Role",
    "list_permissions": "List Permissions",
    "list_oauth2_clients": "List OAuth2 Clients",
    "revoke_oauth2_token": "Revoke OAuth2 Token",
    "register_federated_peer": "Register Federated Peer",
    "list_federated_peers": "List Federated Peers",
    "discover_federated_capabilities": "Discover Federated Capabilities",
    "share_resource_with_peer": "Share Resource with Peer",
    "list_resource_pools": "List Resource Pools",
    "allocate_resource": "Allocate Resource from Pool",
}

# Write tools that remove, replace, revoke or otherwise irreversibly change
# state. Everything else that writes is additive or a reversible state
# transition and is annotated destructiveHint=False.
DESTRUCTIVE_TOOLS = frozenset({
    "delete_memory",
    "delete_credential",
    "rotate_credential",
    "delete_template",
    "delete_file_version",
    "revoke_role",
    "revoke_oauth2_token",
    "unlink_channel",
    "unregister_mcp_server",
    "rollback_workflow",
    # Drives arbitrary web UIs; the actions it takes are not additive.
    "browser_execute",
})

# Write tools whose repeat call yields the same end state.
IDEMPOTENT_PREFIXES = (
    "set_", "update_", "configure_",
    "delete_", "revoke_", "unlink_", "unregister_", "rollback_",
    "mark_", "complete_", "approve_", "reject_", "activate_",
    "test_", "validate_", "verify_", "evaluate_",
)

# Tools that interact with entities outside the tenant boundary.
OPEN_WORLD_TOOLS = frozenset({
    "browser_execute",
    "research_api",
    "generate_adapter",
    "self_extend",
    "test_adapter_config",
    "validate_credential",
    "execute_workflow",
    "execute_agent",
    "dispatch_swarm",
    "automate_company",
    "org_discovery_scan",
    "execute_n8n_workflow",
    "execute_zapier_zap",
    "execute_make_scenario",
    "execute_power_automate_flow",
    "register_mcp_server",
    "discover_mcp_server_tools",
    "invoke_external_mcp_tool",
    "send_channel_message",
    "connect_external_agent",
    "register_runtime_webhook",
    "sync_public_marketplace_updates",
    "register_federated_peer",
    "discover_federated_capabilities",
    "federated_knowledge_query",
    "share_resource_with_peer",
})


def is_read_only(name: str, tool_scopes: Mapping[str, List[str]]) -> bool:
    scopes = tool_scopes.get(name) or []
    return bool(scopes) and all(s.startswith("read:") for s in scopes)


def annotations_for(name: str, tool_scopes: Mapping[str, List[str]]) -> Dict[str, bool]:
    read_only = is_read_only(name, tool_scopes)
    if read_only:
        return {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": name in OPEN_WORLD_TOOLS,
        }
    return {
        "readOnlyHint": False,
        "destructiveHint": name in DESTRUCTIVE_TOOLS,
        "idempotentHint": name.startswith(IDEMPOTENT_PREFIXES),
        "openWorldHint": name in OPEN_WORLD_TOOLS,
    }


def apply_annotations(tool_lists: Iterable[List[dict]], tool_scopes: Mapping[str, List[str]]) -> None:
    """Attach ``title`` and ``annotations`` to every tool dict, in place.

    Raises ``KeyError`` at import time if a tool has no title — a new tool
    must be named here before it can ship.
    """
    for tool_list in tool_lists:
        for tool in tool_list:
            name = tool["name"]
            tool["title"] = TOOL_TITLES[name]
            tool["annotations"] = annotations_for(name, tool_scopes)
