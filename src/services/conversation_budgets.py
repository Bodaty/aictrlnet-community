"""Conversation turn budgets — the single source of truth for §7.2.

docs/architecture/CONVERSATION_ORCHESTRATION_SPEC.md §7.2 holds the same
table; tests/deep-validation/test_conversation_budgets_spec.py parses it (and
the frontend fallback in frontend/src/services/conversationBudgets.js) and
fails when they disagree. Change all three in one commit (§8.4).
"""

from dataclasses import asdict, dataclass
from typing import Dict

LOCAL = "local"
SELF_HOSTED = "self-hosted"
CLOUD = "cloud"
PROVIDER_CLASSES = (LOCAL, SELF_HOSTED, CLOUD)

_LOCAL_PROVIDERS = frozenset({"ollama", "scripted"})
_SELF_HOSTED_PROVIDERS = frozenset({"vllm"})


@dataclass(frozen=True)
class ConversationBudgets:
    provider_class: str
    route_ms: int
    prompt_warm_ms: int
    prompt_cold_ms: int
    knowledge_ms: int
    llm_round_s: int
    tool_max_s: int
    tool_hard_cap_s: int
    job_ack_s: int
    loop_s: int
    turn_s: int
    chat_turn_s: int
    first_event_s: int
    idle_s: int
    client_grace_s: int
    max_tools: int

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _budgets(provider_class: str, llm_round_s: int, loop_s: int, turn_s: int,
             chat_turn_s: int, idle_s: int, max_tools: int) -> ConversationBudgets:
    return ConversationBudgets(
        provider_class=provider_class,
        route_ms=5,
        prompt_warm_ms=150,
        prompt_cold_ms=1500,
        knowledge_ms=2000,
        llm_round_s=llm_round_s,
        tool_max_s=20,
        tool_hard_cap_s=60,
        job_ack_s=2,
        loop_s=loop_s,
        turn_s=turn_s,
        chat_turn_s=chat_turn_s,
        first_event_s=30,
        idle_s=idle_s,
        client_grace_s=10,
        max_tools=max_tools,
    )


BUDGETS: Dict[str, ConversationBudgets] = {
    LOCAL: _budgets(LOCAL, llm_round_s=45, loop_s=120, turn_s=150, chat_turn_s=15, idle_s=60,
                    max_tools=20),
    SELF_HOSTED: _budgets(SELF_HOSTED, llm_round_s=20, loop_s=90, turn_s=100, chat_turn_s=8, idle_s=35,
                          max_tools=64),
    CLOUD: _budgets(CLOUD, llm_round_s=15, loop_s=60, turn_s=75, chat_turn_s=6, idle_s=30,
                    max_tools=48),
}


def provider_class_for(provider: str) -> str:
    """Map a provider name (ModelProvider value or alias) to its §7.1 class.

    Unknown providers are `cloud` (§7.1).
    """
    name = (getattr(provider, "value", provider) or "").lower()
    if name in _LOCAL_PROVIDERS:
        return LOCAL
    if name in _SELF_HOSTED_PROVIDERS:
        return SELF_HOSTED
    return CLOUD


def budgets_for(provider: str) -> ConversationBudgets:
    return BUDGETS[provider_class_for(provider)]


def budgets_payload() -> Dict[str, object]:
    """Body of GET /api/v1/conversation/budgets (spec §7.8, R7)."""
    from llm.tier_resolver import get_environment_default_provider

    return {
        "default_class": provider_class_for(get_environment_default_provider()),
        "classes": {cls: b.as_dict() for cls, b in BUDGETS.items()},
    }


OBSERVE = "observe"
ENFORCE = "enforce"


def enforcement_mode() -> str:
    """`CONVERSATION_BUDGETS=observe|enforce` (spec §7.7). Observe measures and
    reports breaches; enforce applies the deadlines and the §7.4 ladder.
    Defaults to observe so a deployment collects timings before turning it on."""
    import os

    mode = os.environ.get("CONVERSATION_BUDGETS", OBSERVE).strip().lower()
    return ENFORCE if mode == ENFORCE else OBSERVE
