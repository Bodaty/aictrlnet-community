"""Runtime upgrade hints for Community Edition responses.

Attaches an X-Upgrade-Hints header to key list endpoints so the frontend
can surface contextual upsell prompts without changing response bodies.
"""

import json
import logging
from typing import Dict, List, Optional

from fastapi import Response

from core.config import get_settings

logger = logging.getLogger(__name__)

# Context-specific upgrade hints shown on list endpoints.
# Keep each context to 2-3 hints max — focused beats exhaustive.
UPGRADE_HINTS: Dict[str, List[Dict[str, str]]] = {
    "workflows": [
        {
            "feature": "Approval Workflows",
            "edition": "business",
            "description": "Add human-in-the-loop approval steps to any workflow",
        },
        {
            "feature": "Workflow Analytics",
            "edition": "business",
            "description": "Track execution metrics, bottlenecks, and optimization suggestions",
        },
        {
            "feature": "SLA Monitoring",
            "edition": "business",
            "description": "Enforce SLA targets and get alerts when workflows fall behind",
        },
    ],
    "agents": [
        {
            "feature": "AI Governance & Risk Assessment",
            "edition": "business",
            "description": "Automated risk scoring and compliance checks for AI agents",
        },
        {
            "feature": "ML-Enhanced Agent Selection",
            "edition": "business",
            "description": "Intelligent agent routing and learning loops that improve over time",
        },
        {
            "feature": "Multi-Agent Collaboration",
            "edition": "enterprise",
            "description": "Coordinate multiple AI agents with shared context and goals",
        },
    ],
    "conversations": [
        {
            "feature": "Semantic Search",
            "edition": "business",
            "description": "Search conversation history by meaning, not just keywords",
        },
        {
            "feature": "Pattern Learning",
            "edition": "business",
            "description": "Automatically learn and suggest optimal conversation flows",
        },
    ],
    "a2a": [
        {
            "feature": "Full Google A2A Protocol",
            "edition": "business",
            "description": "Server hosting, sessions, streaming, and webhook support for A2A",
        },
        {
            "feature": "Cross-Tenant A2A",
            "edition": "enterprise",
            "description": "Federated agent-to-agent communication across organizations",
        },
    ],
    "teams": [
        {
            "feature": "Organization & Team Management",
            "edition": "business",
            "description": "Multi-user collaboration with role-based access control",
        },
        {
            "feature": "OAuth2/OIDC Integration",
            "edition": "business",
            "description": "Connect your existing identity provider for single sign-on",
        },
        {
            "feature": "SAML 2.0 SSO",
            "edition": "enterprise",
            "description": "Enterprise identity federation with SAML 2.0 support",
        },
    ],
}


def get_upgrade_hints(context: str) -> Optional[List[Dict[str, str]]]:
    """Return upgrade hints for the given context, or None if not community edition."""
    settings = get_settings()
    if settings.EDITION.lower() != "community":
        return None
    return UPGRADE_HINTS.get(context)


def attach_upgrade_hints(response: Response, context: str) -> None:
    """Set the X-Upgrade-Hints header on a response if applicable."""
    hints = get_upgrade_hints(context)
    if hints:
        response.headers["X-Upgrade-Hints"] = json.dumps(hints)
