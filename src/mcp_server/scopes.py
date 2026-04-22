"""MCP API-key / OAuth2 scope registry.

Per-resource taxonomy introduced by the MCP expansion plan. Replaces the
coarse ``read:all`` / ``write:all`` scopes. Legacy scopes are still
accepted during migration Phase A via ``expand_legacy()``; Phase B drops
that acceptance.

Single source of truth for:
- Which scope strings are valid (``validate_scope``)
- Human descriptions (``describe_scope``) for API-key UI / OAuth2 consent
- Legacy -> new expansion (``LEGACY_SCOPE_MAP``) used by the scope
  migration and by the scope-check compatibility layer
"""

from __future__ import annotations

from typing import Iterable, Set

READ: Set[str] = {
    "read:workflows",
    "read:adapters",
    "read:templates",
    "read:policies",
    "read:approvals",
    "read:tasks",
    "read:conversations",
    "read:knowledge",
    "read:memory",
    "read:files",
    "read:analytics",
    "read:audit",
    "read:usage",
    "read:compliance",
    "read:institute",
    "read:subscription",
    "read:self_extending",
    "read:autonomy",
    "read:patterns",
    "read:org",
    "read:agents",
    "read:notifications",
    "read:llm",
    "read:fleet",
    "read:license",
}

WRITE: Set[str] = {
    "write:workflows",
    "write:adapters",
    "write:self_extending",
    "write:browser",
    "write:approvals",
    "write:tasks",
    "write:conversations",
    "write:knowledge",
    "write:memory",
    "write:files",
    "write:messaging",
    "write:policies",
    "write:autonomy",
    "write:patterns",
    "write:org",
    "write:company",
    "write:agents",
    "write:notifications",
    "write:quality",
    "write:institute",
}

ALL_SCOPES: Set[str] = READ | WRITE

# Legacy scopes expand to a set of new-taxonomy scopes. Used by:
# 1. The one-time migration XXXX_mcp_scope_taxonomy.py
# 2. The Phase-A compatibility layer in tool_executor.expand_legacy
LEGACY_SCOPE_MAP: dict[str, Set[str]] = {
    "read:all": READ,
    "write:all": WRITE,
    # These two already match the new taxonomy; mapping them lets
    # migration + compat code treat ALL inbound scopes uniformly.
    "read:workflows": {"read:workflows", "read:templates"},
    "write:workflows": {"write:workflows"},
}

_DESCRIPTIONS: dict[str, str] = {
    "read:workflows": "List and inspect workflows and executions",
    "read:adapters": "List integration adapters and their configs",
    "read:templates": "Browse workflow templates",
    "read:policies": "List AI governance policies and evaluate content",
    "read:approvals": "List and inspect pending approval requests",
    "read:tasks": "List and inspect tasks",
    "read:conversations": "List conversations and linked channels",
    "read:knowledge": "Query the knowledge base",
    "read:memory": "Read per-user memory values",
    "read:files": "List and retrieve staged files",
    "read:analytics": "Read analytics dashboards and trends (Enterprise)",
    "read:audit": "Read audit logs (Enterprise)",
    "read:usage": "Read API-key usage, trial status, and billing usage",
    "read:compliance": "Read compliance checks and tenant configs (Enterprise)",
    "read:institute": "List AICtrlNet Institute modules and certifications",
    "read:subscription": "Read current subscription and upgrade options",
    "read:self_extending": "Read generated-adapter status, risk, and source",
    "read:autonomy": "Read workflow / agent autonomy levels",
    "read:patterns": "List learned operation patterns",
    "read:org": "Read org discovery landscape and recommendations",
    "read:agents": "List AI agents and capabilities",
    "read:notifications": "List notifications",
    "read:llm": "List available LLM providers / models",
    "read:fleet": "Read fleet-wide agent/autonomy summaries (Enterprise)",
    "read:license": "Read license status and entitlements (Enterprise)",
    "write:workflows": "Create and execute workflows",
    "write:adapters": "Test and configure adapters",
    "write:self_extending": "Research APIs and generate/approve/activate adapters",
    "write:browser": "Execute browser-automation actions",
    "write:approvals": "Approve or reject pending approval requests",
    "write:tasks": "Create, update, or complete tasks",
    "write:conversations": "Send messages and link channels",
    "write:knowledge": "Add knowledge items",
    "write:memory": "Set or delete per-user memory values",
    "write:files": "Upload files",
    "write:messaging": "Send messages into linked external channels",
    "write:policies": "Create or update AI governance policies",
    "write:autonomy": "Set workflow / agent autonomy levels",
    "write:patterns": "Promote learned patterns to templates",
    "write:org": "Trigger org discovery scans",
    "write:company": "Trigger company-wide automation generation",
    "write:agents": "Create, configure, or execute agents",
    "write:notifications": "Mark notifications read",
    "write:quality": "Run data-quality / governance verifications",
    "write:institute": "Enroll in Institute modules",
}


def validate_scope(scope: str) -> bool:
    """Return True if ``scope`` is a known new-taxonomy scope string.

    Legacy scopes (``read:all``, ``write:all``) are NOT valid here — the
    compatibility layer expands them BEFORE calling this.
    """
    return scope in ALL_SCOPES


def describe_scope(scope: str) -> str:
    """Human-readable description for UI/consent screens."""
    return _DESCRIPTIONS.get(scope, scope)


def expand_legacy(scopes: Iterable[str]) -> Set[str]:
    """Expand a mixed list of legacy + new scopes to pure new taxonomy.

    Used at migration time (to rewrite stored scopes) and at request
    time during Phase A (to accept old-taxonomy keys without
    re-issuing). Unknown scopes are dropped.
    """
    expanded: Set[str] = set()
    for s in scopes or ():
        if s in LEGACY_SCOPE_MAP:
            expanded |= LEGACY_SCOPE_MAP[s]
        elif s in ALL_SCOPES:
            expanded.add(s)
        # unknown scope strings silently dropped
    return expanded


def scopes_satisfy(granted: Iterable[str], required: Iterable[str]) -> bool:
    """Return True if ``granted`` (possibly containing legacy scopes) covers
    every ``required`` scope.
    """
    granted_set = expand_legacy(granted)
    return all(r in granted_set for r in required)


__all__ = [
    "ALL_SCOPES",
    "LEGACY_SCOPE_MAP",
    "READ",
    "WRITE",
    "describe_scope",
    "expand_legacy",
    "scopes_satisfy",
    "validate_scope",
]
