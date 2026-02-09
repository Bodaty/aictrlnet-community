"""System Prompt Assembler — Community Edition.

Composable, edition-aware prompt assembly pipeline that replaces the three
scattered prompt builders (build_tool_system_prompt, _build_knowledge_aware_prompt,
_build_v5_system_prompt) with a single service.

Community layer provides: identity + tool rules + knowledge items + industry
detection + session params + response instructions.
"""

import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class SystemPromptAssembler:
    """Assemble a system prompt from composable sections.

    Each section is built by a private method and may be omitted when its
    input data is ``None``.  The Business sub-class appends additional
    sections (manifest, user context, active state, patterns, Business
    tool rules) via ``super().assemble()``.
    """

    def __init__(self, db):
        self.db = db

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def assemble(
        self,
        edition: str = "community",
        session=None,
        knowledge_items=None,
        conversation_history=None,
        session_context=None,
        user_id=None,
        organization_id=None,
        tool_definitions=None,
    ) -> str:
        """Build the full system prompt.

        Joins: identity -> tool_rules -> knowledge -> industry ->
               session_params -> context -> turn_count -> instructions
        """
        sections: List[str] = []

        sections.append(self._build_identity())
        sections.append(self._build_tool_rules())

        if knowledge_items:
            sections.append(self._build_knowledge_section(knowledge_items))

        # Industry detection — use conversation history when available,
        # otherwise fall back to session_context for single-turn calls.
        industry = None
        if conversation_history:
            industry = self._detect_industry_from_history(conversation_history)
        if industry:
            sections.append(
                f"## Industry Context: {industry['name']}\n\n{industry.get('guidance', '')}"
            )

        # Session parameters (v5)
        if session:
            sections.append(self._build_session_params(session))

        # Session context (intent / extracted params) — used by the
        # lighter v4 path and stream_tool_execution.
        if session_context:
            sections.append(self._build_context_section(session_context))

        # Turn count
        if conversation_history and len(conversation_history) > 1:
            sections.append(
                f"*This is turn {len(conversation_history)} of the conversation.*"
            )

        sections.append(self._build_response_instructions())

        return "\n\n".join(s for s in sections if s)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def _build_identity(self) -> str:
        return (
            "You are an intelligent assistant for AICtrlNet. "
            "You help users with workflows, agents, templates, and integrations."
        )

    # ------------------------------------------------------------------
    # Tool decision rules (Community)
    # ------------------------------------------------------------------

    def _build_tool_rules(self) -> str:
        return """## Tool Selection Rules

**#1 Rule: "create/generate/build [X] workflow" → create_workflow**
- Extract the name from the request (e.g., "sales pipeline workflow" → name="Sales Pipeline")
- Integration tools are ONLY for sending messages, testing adapters, or configuring credentials — never for creating workflows

**Behaviors:**
- LIST/SHOW requests → execute immediately (list_workflows, list_agents, list_templates, list_integrations)
- CREATE requests with a topic → call create_workflow with that topic as the name
- CREATE without a name → ask "What would you like to name it?"
- Multi-turn: check conversation history for context (prior turn may have the name)
- Always use the user's exact name — never use placeholders like "New Workflow" """

    # ------------------------------------------------------------------
    # Knowledge items
    # ------------------------------------------------------------------

    def _build_knowledge_section(self, knowledge_items) -> str:
        # Filter by minimum relevance to avoid noise
        MIN_RELEVANCE = 1.0
        relevant = [k for k in knowledge_items if getattr(k, 'relevance', 0) >= MIN_RELEVANCE]

        templates = [k for k in relevant if k.type == "template"]
        agents = [k for k in relevant if k.type == "agent"]
        adapters = [k for k in relevant if k.type == "adapter"]

        if not (templates or agents or adapters):
            return ""

        parts = ["## Available Resources\n"]

        if templates:
            parts.append(f"**Templates ({len(templates)}):**")
            for t in templates[:3]:
                desc = (t.data.get("description", "") or "")[:60]
                parts.append(f"- **{t.name}**: {desc}")
            parts.append("")

        if agents:
            parts.append(f"**Agents ({len(agents)}):**")
            for a in agents[:3]:
                caps = a.data.get("capabilities", [])[:3]
                parts.append(
                    f"- **{a.name}**: {', '.join(caps) if caps else 'General assistance'}"
                )
            parts.append("")

        if adapters:
            parts.append(f"**Integrations ({len(adapters)}):**")
            for a in adapters[:3]:
                parts.append(
                    f"- **{a.name}**: {(a.data.get('description', '') or '')[:40]}"
                )
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Industry detection
    # ------------------------------------------------------------------

    def _detect_industry_from_history(
        self, conversation_history: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Detect industry context from conversation history."""
        all_text = " ".join(msg["content"].lower() for msg in conversation_history)

        industry_patterns = {
            "legal": {
                "keywords": [
                    "legal", "e-discovery", "ediscovery", "litigation", "attorney",
                    "law firm", "deposition", "privilege", "frcp", "discovery",
                    "matter", "case", "court",
                ],
                "name": "Legal/E-Discovery",
                "guidance": (
                    "Legal industry considerations:\n"
                    "- FRCP compliance may be required for litigation workflows\n"
                    "- Privilege detection and logging should be enabled\n"
                    "- Chain of custody tracking for documents\n"
                    "- Audit logging for all operations\n"
                    "- Consider document hold and preservation requirements"
                ),
            },
            "healthcare": {
                "keywords": [
                    "healthcare", "medical", "patient", "hipaa", "clinical",
                    "hospital", "doctor", "nurse", "diagnosis", "treatment",
                ],
                "name": "Healthcare",
                "guidance": (
                    "Healthcare industry considerations:\n"
                    "- HIPAA compliance is critical\n"
                    "- Patient data must be protected\n"
                    "- Audit trails required for PHI access\n"
                    "- Consider consent management workflows"
                ),
            },
            "finance": {
                "keywords": [
                    "finance", "banking", "investment", "trading", "compliance",
                    "audit", "regulatory", "sox", "kyc", "aml",
                ],
                "name": "Finance/Banking",
                "guidance": (
                    "Financial industry considerations:\n"
                    "- SOX compliance for relevant workflows\n"
                    "- KYC/AML requirements for customer processes\n"
                    "- Regulatory reporting needs\n"
                    "- Strict audit requirements"
                ),
            },
            "retail": {
                "keywords": [
                    "retail", "e-commerce", "ecommerce", "inventory", "customer",
                    "order", "shipping", "product", "store",
                ],
                "name": "Retail/E-Commerce",
                "guidance": (
                    "Retail industry considerations:\n"
                    "- Inventory management integration\n"
                    "- Order processing automation\n"
                    "- Customer communication workflows\n"
                    "- Returns and refunds handling"
                ),
            },
        }

        for _industry, config in industry_patterns.items():
            matches = sum(1 for kw in config["keywords"] if kw in all_text)
            if matches >= 2:
                return {
                    "name": config["name"],
                    "guidance": config["guidance"],
                    "confidence": min(matches / 5, 1.0),
                }

        return None

    # ------------------------------------------------------------------
    # Session parameters (v5)
    # ------------------------------------------------------------------

    def _build_session_params(self, session) -> str:
        v5_params = (
            session.context.get("v5_parameters", {}) if session.context else {}
        )
        if v5_params:
            lines = ["## Parameters Gathered So Far\n"]
            for key, value in v5_params.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("\n*Use these parameters when executing tools.*")
            return "\n".join(lines)

        return (
            "## Parameters Gathered So Far\n\n"
            "*No parameters gathered yet. Ask the user for details as needed.*"
        )

    # ------------------------------------------------------------------
    # Session context (lighter path — v4 / stream_tool_execution)
    # ------------------------------------------------------------------

    def _build_context_section(self, session_context: Dict[str, Any]) -> str:
        parts = []
        if session_context.get("primary_intent"):
            parts.append(
                f"**Current User Intent:** {session_context['primary_intent']}"
            )
        if session_context.get("extracted_params"):
            parts.append(
                f"**Extracted Parameters:** {session_context['extracted_params']}"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Response format instructions
    # ------------------------------------------------------------------

    def _build_response_instructions(self) -> str:
        return """**CRITICAL Response Format Rules:**
- NEVER output JSON in any form - not raw, not in code blocks, not anywhere
- NEVER include code blocks with action objects like ```{"action": ...}```
- Respond ONLY in natural, conversational English prose
- When you want to take an action, call the tool directly - don't show the user what you're calling
- Present template recommendations as a formatted markdown list with descriptions
- Ask questions to gather requirements before taking action
- Be helpful, specific, and leverage your knowledge of available resources

**Persona:**
- Always speak directly TO the user in second person
- Say "I created..." not "The user wants..." — never refer to the user in third person
- Be concise: lead with what you did, then offer next steps

**After creating a workflow, ALWAYS offer these numbered options:**
1. View/Edit this workflow in the editor
2. Add a description or configure this workflow
3. Set up triggers or schedule
4. Connect to an integration (Slack, Email, etc.)
5. Create another workflow"""
