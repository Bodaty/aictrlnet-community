"""Deterministic conversation router (CONVERSATION_ORCHESTRATION_SPEC.md §7.6).

Decides, before any tool is offered, which MODE a turn runs in and therefore
which tool CLASSES may enter the prompt (R3). No LLM call; pure string rules,
so the same message always routes the same way and Tier 0 tests pin it.
English only: a message in another language never reaches `act`.

Ranking (which admitted tool is most relevant) is the pruner's job; admission
(which classes may be in the prompt at all) is this module's. A ranker can
reorder or drop admitted tools but never add one.
"""

import re
from dataclasses import dataclass, field
from typing import Any, FrozenSet, List, Mapping, Optional, Tuple

from services.tool_classes import DISCOVERY, LONG_RUNNING, READ, TOOL_CLASSES, WRITE  # noqa: F401

CONFIRM = "confirm"
CHAT = "chat"
DISCOVER = "discover"
ACT = "act"

_ADMITTED = {
    CHAT: frozenset(),
    DISCOVER: frozenset({DISCOVERY, READ}),
    ACT: frozenset(TOOL_CLASSES),
}

# While the onboarding interview runs, the prompt asks the model to save each
# answer with this tool, and answers ("about 12 people") route to chat or
# discover — so it is admitted in every mode for that session only.
ONBOARDING_TOOLS: FrozenSet[str] = frozenset({"update_onboarding"})

# Whole clauses that are pleasantries or questions about the assistant itself.
PLEASANTRIES: FrozenSet[str] = frozenset({
    "hello", "hi", "hey", "hi there", "hey there", "hello there", "yo",
    "what can you do", "what can you help me with", "what can you help with",
    "what do you do", "who are you", "how are you", "what are you",
    "how are you doing", "how is it going", "whats up", "what s up",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "thank you so much", "thanks a lot", "cheers",
    "bye", "goodbye", "see you", "talk later", "cool", "great", "nice",
})

# Bare consent. Routes to `act` only when the assistant's previous turn asked
# something (it proposed an action); otherwise it is small talk.
AFFIRMATIVES: FrozenSet[str] = frozenset({
    "yes", "y", "yep", "yeah", "sure", "ok", "okay", "confirm", "confirmed",
    "approve", "approved", "correct", "right", "absolutely", "definitely",
})

# Consent that carries its own instruction ("yes please do that").
_CONSENT = re.compile(
    r"^(?:(?:yes|yeah|yep|sure|ok|okay|please|alright)\s+)*"
    r"(?:go ahead|do it|do that|do this|do so|please do|proceed|sounds good|lets do it|let s do it|make it so)\b"
)

# Function words only. Anything else counts toward the 3-content-word floor;
# "more", "tell", "capabilities" are content ("Tell me more about your
# capabilities." must route to discover, not chat).
STOPWORDS: FrozenSet[str] = frozenset({
    "i", "me", "my", "mine", "we", "us", "our", "you", "your", "yours", "it", "its",
    "he", "she", "they", "them", "their", "this", "that", "these", "those",
    "a", "an", "the", "and", "or", "but", "so", "if", "then", "than",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "could", "should",
    "can", "may", "might", "shall", "must",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "about", "into",
    "as", "up", "out", "please", "just", "what", "which", "who", "how", "when",
    "where", "why", "there", "here", "s",
})

# Verbs that ask the platform to change something.
ACTION_VERBS: FrozenSet[str] = frozenset({
    "create", "make", "build", "set", "setup", "add", "generate",
    "update", "edit", "change", "modify", "rename", "configure", "enable", "disable",
    "delete", "remove", "cancel", "revoke", "suspend", "rotate", "reset",
    "run", "rerun", "execute", "start", "stop", "pause", "resume", "trigger", "launch",
    "activate", "deactivate", "turn", "retry", "schedule", "automate",
    "send", "email", "notify", "invite", "assign", "grant", "approve", "reject", "delegate",
    "install", "connect", "integrate", "sync", "upgrade", "downgrade", "deploy", "instantiate",
    "share", "publish", "clone", "duplicate", "export", "import", "upload", "link", "unlink",
    "complete", "mark", "submit", "enroll", "draft", "post", "book", "archive", "restore",
    "remember", "forget",
})

# Verbs that are requests only when they name a platform object ("test my
# quickbooks connection" yes; "Test sidebar conversation" no).
OBJECT_REQUIRED_VERBS: FrozenSet[str] = frozenset({"test"})

# Verbs whose object is always the platform ("automate my accounting").
SELF_OBJECT_VERBS: FrozenSet[str] = frozenset({"automate"})

# Things on the platform an action verb can act upon.
PLATFORM_NOUNS: FrozenSet[str] = frozenset({
    "workflow", "workflows", "automation", "automations", "agent", "agents", "pod", "pods",
    "task", "tasks", "template", "templates", "integration", "integrations",
    "adapter", "adapters", "connector", "connectors", "webhook", "webhooks",
    "approval", "approvals", "request", "requests", "policy", "policies",
    "schedule", "schedules", "job", "jobs", "key", "keys", "token", "tokens",
    "role", "roles", "permission", "permissions", "user", "users", "team", "teams",
    "sla", "slas", "alert", "alerts", "rule", "rules", "report", "reports",
    "subscription", "plan", "plans", "invoice", "invoices", "payment", "billing",
    "mfa", "2fa", "password", "session", "sessions", "profile", "company", "business",
    "process", "processes", "pipeline", "pipelines", "server", "servers", "mcp",
    "account", "accounts", "connection", "connections", "file", "files", "module", "modules",
    "channel", "channels", "ticket", "tickets", "reminder", "reminders", "mode",
    "onboarding", "data", "appointment", "appointments",
})

_PRONOUN_OBJECTS: FrozenSet[str] = frozenset({"it", "that", "this", "them", "those"})
_AUXILIARIES: FrozenSet[str] = frozenset({"is", "are", "was", "were", "am", "be", "been", "being"})
_DETERMINERS: FrozenSet[str] = frozenset({"any", "the", "my", "your", "our", "some", "no", "all", "its", "their"})

# A clause opening like this asks about something; it never reaches `act`.
_QUESTION = re.compile(
    r"^(?:how|what|whats|why|when|where|who|whose|which|is|are|was|were|does|did|"
    r"do (?:i|we|you)|can (?:i|we)|could (?:i|we)|should (?:i|we)|explain|tell me)\b"
)
# Politeness and intent wrappers stripped before looking for an imperative verb.
_LEAD = re.compile(
    r"^(?:(?:please|can you|could you|would you|will you|i want to|i d like to|i would like to|"
    r"i need to|let s|lets|help me|go ahead and|yes|sure|ok|okay|now|also|then|and)\s+)+"
)
_NEED_A = re.compile(r"\b(?:need|needs|want|wants)\s+(?:a|an|new|some|another)\b")
_NEGATED = re.compile(r"^(?:don t|do not|dont|never|stop me from)\b")

_CLAUSE_SPLIT = re.compile(r"[.,;:!?\n]+| - ")
_NON_WORD = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class RouteDecision:
    mode: str
    reasons: List[str] = field(default_factory=list)
    allowed_classes: FrozenSet[str] = frozenset()
    proposed_tool: Optional[str] = None
    # Named tools admitted regardless of class (session-scoped, e.g. onboarding).
    extra_tools: FrozenSet[str] = frozenset()


def _normalise(text: str) -> str:
    return _WS.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()


def _clauses(message: str) -> List[str]:
    collapsed = _WS.sub(" ", message or "")  # before splitting: no backtracking on long runs
    parts = (_normalise(p) for p in _CLAUSE_SPLIT.split(collapsed))
    return [p for p in parts if p]


def _content_words(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]


def _forms(token: str) -> Tuple[str, ...]:
    """The token plus its likely base forms (emails→email, setting→set, created→create)."""
    forms = [token]
    for suffix, restore in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ing", "e"),
                            ("ed", ""), ("ed", "e"), ("d", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)] + restore
            forms.append(stem)
            if suffix in ("ing", "ed") and len(stem) >= 4 and stem[-1] == stem[-2]:
                forms.append(stem[:-1])  # running→run, setting→set
    return tuple(forms)


def _is_noun(token: str) -> bool:
    return any(f in PLATFORM_NOUNS for f in _forms(token))


def _verb_base(token: str, verbs: FrozenSet[str]) -> Optional[str]:
    return next((f for f in _forms(token) if f in verbs), None)


def _is_request_verb(tokens: List[str], i: int, verbs: FrozenSet[str] = ACTION_VERBS) -> bool:
    """tokens[i] is an action verb used as a request, not a state or a noun."""
    token = tokens[i]
    if _verb_base(token, verbs) is None:
        return False
    before = tokens[max(0, i - 3):i]
    after = tokens[i + 1] if i + 1 < len(tokens) else None
    if token.endswith("ing") and any(t in _AUXILIARIES for t in before):
        return False  # "is my workflow running"
    if token.endswith(("ing", "ed")) and token not in verbs and after and _is_noun(after):
        return False  # "running workflows", "scheduled tasks": adjectives
    if token.endswith("s") and token not in verbs and i > 0 and tokens[i - 1] in _DETERMINERS:
        return False  # "any updates", "my reports": nouns
    return True


def _is_chat_clause(clause: str) -> bool:
    if clause in PLEASANTRIES:
        return True
    tokens = clause.split()
    if len(_content_words(tokens)) >= 3:
        return False
    # A short clause is still a request when it names an action or a platform
    # object ("list my workflows", "automate my accounting").
    return not any(
        _is_request_verb(tokens, i) or _is_request_verb(tokens, i, OBJECT_REQUIRED_VERBS)
        or _is_noun(t)
        for i, t in enumerate(tokens)
    )


def _decision(mode: str, reasons: List[str], extra: FrozenSet[str]) -> RouteDecision:
    return RouteDecision(mode, reasons, _ADMITTED[mode], extra_tools=extra)


def route(message: str, session_context: Optional[Mapping[str, Any]] = None) -> RouteDecision:
    """Route one turn.

    `session_context` keys read: `pending_proposal` ({"tool": name}, set by the
    confirmation gate), `assistant_asked` (the assistant's previous turn ended
    with a question), `onboarding_active` (the onboarding interview is running).
    """
    context = session_context if isinstance(session_context, Mapping) else {}
    extra = ONBOARDING_TOOLS if context.get("onboarding_active") else frozenset()
    normalised = _normalise(message or "")
    tokens = normalised.split()

    pending = context.get("pending_proposal")
    if (isinstance(pending, Mapping) and pending.get("tool")
            and (normalised in AFFIRMATIVES or _CONSENT.match(normalised))):
        # Every class, narrowed to the one proposed tool by the caller (only_tools).
        return RouteDecision(
            CONFIRM, [f"affirms pending proposal {pending['tool']}"], frozenset(TOOL_CLASSES),
            pending["tool"], extra,
        )

    clauses = _clauses(message)
    if not clauses:
        return _decision(CHAT, ["empty"], extra)

    if normalised in AFFIRMATIVES:
        if context.get("assistant_asked"):
            return _decision(ACT, ["consents to the assistant's proposal"], extra)
        return _decision(CHAT, ["bare consent with nothing proposed"], extra)
    if _CONSENT.match(normalised):
        return _decision(ACT, ["consent with an instruction"], extra)

    if all(_is_chat_clause(c) for c in clauses):
        reason = (
            "every clause is a pleasantry"
            if all(c in PLEASANTRIES for c in clauses)
            else "no clause has 3 content words, an action verb or a platform noun"
        )
        return _decision(CHAT, [reason], extra)

    if _NEGATED.match(normalised):
        return _decision(DISCOVER, ["negated request"], extra)
    if _QUESTION.match(clauses[0]):
        return _decision(DISCOVER, ["question form"], extra)

    # Imperative: an action verb first, once politeness/intent wrappers go.
    rest = _LEAD.sub("", normalised).split()
    if rest and _is_request_verb(rest, 0):
        base = _verb_base(rest[0], ACTION_VERBS)
        if base in SELF_OBJECT_VERBS:
            return _decision(ACT, [f"action verb {base} implies its platform object"], extra)
        return _decision(ACT, [f"imperative {base}"], extra)

    verb_at = [i for i in range(len(tokens)) if _is_request_verb(tokens, i)]
    test_at = [i for i in range(len(tokens)) if _is_request_verb(tokens, i, OBJECT_REQUIRED_VERBS)]
    implied = [i for i in verb_at if _verb_base(tokens[i], ACTION_VERBS) in SELF_OBJECT_VERBS]
    if implied:
        return _decision(ACT, [f"action verb {tokens[implied[0]]} implies its platform object"], extra)
    noun_at = [j for j, t in enumerate(tokens) if _is_noun(t)]
    # The same token cannot be both the verb and its object ("show my schedule").
    for i in verb_at + test_at:
        for j in noun_at:
            if i != j:
                return _decision(ACT, [f"action verb {tokens[i]}", f"platform noun {tokens[j]}"], extra)
    for i in verb_at:
        if i + 1 < len(tokens) and tokens[i + 1] in _PRONOUN_OBJECTS:
            return _decision(ACT, [f"action verb {tokens[i]} on {tokens[i + 1]}"], extra)
    if _NEED_A.search(normalised) and noun_at:
        return _decision(ACT, [f"asks for a {tokens[noun_at[0]]}"], extra)
    if verb_at:
        return _decision(DISCOVER, [f"action verb {tokens[verb_at[0]]} without a platform object"], extra)
    return _decision(DISCOVER, ["no action verb"], extra)
