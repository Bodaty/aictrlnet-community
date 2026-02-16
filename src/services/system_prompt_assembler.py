"""System Prompt Assembler — Community Edition.

Composable, edition-aware prompt assembly pipeline that replaces the three
scattered prompt builders (build_tool_system_prompt, _build_knowledge_aware_prompt,
_build_v5_system_prompt) with a single service.

Community layer provides: identity + tool rules + knowledge items + industry
detection + session params + response instructions.  Text content is loaded
from .md files via PromptTemplateLoader; assembly logic stays in Python.
"""

import logging
from typing import Optional, List, Dict, Any

from services.prompt_template_loader import PromptTemplateLoader

logger = logging.getLogger(__name__)

# Module-level singleton — shared across all assembler instances.
_template_loader: Optional[PromptTemplateLoader] = None


def _get_template_loader() -> PromptTemplateLoader:
    """Lazy-init the shared PromptTemplateLoader singleton."""
    global _template_loader
    if _template_loader is None:
        _template_loader = PromptTemplateLoader()
    return _template_loader


class SystemPromptAssembler:
    """Assemble a system prompt from composable sections.

    Each section is built by a private method and may be omitted when its
    input data is ``None``.  The Business sub-class appends additional
    sections (manifest, user context, active state, patterns, Business
    tool rules) via ``super().assemble()``.
    """

    def __init__(self, db):
        self.db = db
        self.template_loader = _get_template_loader()

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
        personal_agent_config=None,
    ) -> str:
        """Build the full system prompt.

        Joins: identity -> personality -> tool_rules -> knowledge -> industry ->
               session_params -> context -> turn_count -> instructions
        """
        sections: List[str] = []

        sections.append(self._build_identity(edition))

        # PersonalAgent personality injection (sandboxed position —
        # after identity, before tool rules / dynamic context)
        personality_section = self._build_personality_section(personal_agent_config)
        if personality_section:
            sections.append(personality_section)

        sections.append(self._build_tool_rules())

        if knowledge_items:
            sections.append(self._build_knowledge_section(knowledge_items))

        # Industry detection — use conversation history when available,
        # otherwise fall back to session_context for single-turn calls.
        industry = None
        if conversation_history:
            industry = self._detect_industry_from_history(conversation_history)
        if industry:
            # Try to load industry guidance from .md file first
            industry_key = self._industry_name_to_key(industry["name"])
            md_guidance = self.template_loader.get_section(f"industries/{industry_key}")
            guidance = md_guidance if md_guidance else industry.get("guidance", "")
            sections.append(
                f"## Industry Context: {industry['name']}\n\n{guidance}"
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

    def _build_identity(self, edition: str = "community") -> str:
        text = self.template_loader.get_section("identity", {"edition": edition})
        if text:
            return text
        # Hardcoded fallback
        return (
            "You are an intelligent assistant for AICtrlNet. "
            "You help users with workflows, agents, templates, and integrations."
        )

    # ------------------------------------------------------------------
    # PersonalAgent personality section
    # ------------------------------------------------------------------

    def _build_personality_section(self, personal_agent_config) -> str:
        """Inject a personality section from PersonalAgentConfig if available.

        The personality data is injected in a sandboxed position — after
        identity/safety but before dynamic context and tool rules output.
        At Community tier, no content validation is applied (self-hosted =
        self-responsible).
        """
        if personal_agent_config is None:
            return ""

        # Accept both dict and ORM object
        if hasattr(personal_agent_config, "personality"):
            personality = personal_agent_config.personality
        elif isinstance(personal_agent_config, dict):
            personality = personal_agent_config.get("personality")
        else:
            return ""

        if not personality or not isinstance(personality, dict):
            return ""

        tone = personality.get("tone", "")
        style = personality.get("style", "")
        expertise_areas = personality.get("expertise_areas", [])

        if not (tone or style or expertise_areas):
            return ""

        lines = ["## Your Personality\n"]
        if tone:
            lines.append(f"- Tone: {tone}")
        if style:
            lines.append(f"- Style: {style}")
        if expertise_areas:
            areas_str = ", ".join(str(a) for a in expertise_areas[:10])
            lines.append(f"- Expertise areas: {areas_str}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool decision rules (Community)
    # ------------------------------------------------------------------

    def _build_tool_rules(self) -> str:
        text = self.template_loader.get_section("tool_rules")
        if text:
            return text
        # Hardcoded fallback
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

    # Maps from display names to .md file keys
    _INDUSTRY_KEY_MAP = {
        "Legal/E-Discovery": "legal",
        "Healthcare": "healthcare",
        "Finance/Banking": "finance",
        "Retail/E-Commerce": "retail",
    }

    def _industry_name_to_key(self, name: str) -> str:
        """Convert an industry display name to its .md file key."""
        return self._INDUSTRY_KEY_MAP.get(name, name.lower().split("/")[0])

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
            },
            "healthcare": {
                "keywords": [
                    "healthcare", "medical", "patient", "hipaa", "clinical",
                    "hospital", "doctor", "nurse", "diagnosis", "treatment",
                ],
                "name": "Healthcare",
            },
            "finance": {
                "keywords": [
                    "finance", "banking", "investment", "trading", "compliance",
                    "audit", "regulatory", "sox", "kyc", "aml",
                ],
                "name": "Finance/Banking",
            },
            "retail": {
                "keywords": [
                    "retail", "e-commerce", "ecommerce", "inventory", "customer",
                    "order", "shipping", "product", "store",
                ],
                "name": "Retail/E-Commerce",
            },
        }

        for _industry, config in industry_patterns.items():
            matches = sum(1 for kw in config["keywords"] if kw in all_text)
            if matches >= 2:
                return {
                    "name": config["name"],
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
        text = self.template_loader.get_section("response_format")
        if text:
            return text
        # Hardcoded fallback
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
