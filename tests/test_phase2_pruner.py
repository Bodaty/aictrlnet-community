"""Phase 2: adapter-aware tool pruner + clarification short-circuit.

Covers the class-level subsystem described in
docs/architecture/CONVERSATION_ORCHESTRATION_SPEC.md §2: `_prune_tools`,
`_last_prune_meta`, `_prune_is_ambiguous`, `_build_clarification_block`,
and `_max_tools_for_adapter`.

The service's DB/LLM plumbing is not touched — we construct a bare service
via `__new__` and only exercise the pruner methods.
"""

import pytest
from types import SimpleNamespace

from services.enhanced_conversation_manager import EnhancedConversationService


def _svc():
    """Bypass __init__ (knowledge svc, prompt assembler, etc.)."""
    return EnhancedConversationService.__new__(EnhancedConversationService)


def _tool(name, description="", category=None):
    """Build a ToolDefinition-like stand-in."""
    return SimpleNamespace(name=name, description=description, category=category)


# ---------------------------------------------------------------------------
# _max_tools_for_adapter
# ---------------------------------------------------------------------------

def test_adapter_cap_known_providers():
    svc = _svc()
    assert svc._max_tools_for_adapter("ollama") == 32
    assert svc._max_tools_for_adapter("OpenAI") == 64  # case-insensitive
    assert svc._max_tools_for_adapter("anthropic") == 64
    assert svc._max_tools_for_adapter("bedrock") == 48


def test_adapter_cap_unknown_falls_back_to_default():
    svc = _svc()
    assert svc._max_tools_for_adapter("some-new-provider") == svc._DEFAULT_TOOL_CAP
    assert svc._max_tools_for_adapter(None) == svc._DEFAULT_TOOL_CAP
    assert svc._max_tools_for_adapter("") == svc._DEFAULT_TOOL_CAP


# ---------------------------------------------------------------------------
# _prune_tools — selection + telemetry
# ---------------------------------------------------------------------------

def test_prune_respects_adapter_cap():
    svc = _svc()
    tools = [_tool(f"tool_{i}", description="") for i in range(50)]
    kept = svc._prune_tools(tools, user_message="do a workflow thing", adapter="ollama")
    assert len(kept) <= 32


def test_prune_keeps_always_include_even_when_no_keyword_match():
    svc = _svc()
    # User query is unrelated — only ALWAYS_INCLUDE sentinels survive.
    tools = [_tool(f"noop_{i}", description="") for i in range(5)]
    tools.append(_tool("get_help", description="help"))
    tools.append(_tool("search_api_capabilities", description=""))
    kept = svc._prune_tools(tools, user_message="weather in chicago", adapter="ollama")
    kept_names = {t.name for t in kept}
    assert "get_help" in kept_names
    assert "search_api_capabilities" in kept_names


def test_prune_ranks_keyword_matches_above_unrelated():
    svc = _svc()
    tools = [
        _tool("delete_agent", description="delete an agent"),
        _tool("create_workflow", description="create a new workflow"),
        _tool("list_tenants", description="list all tenants"),
    ]
    svc._prune_tools(tools, user_message="create a workflow for invoices",
                      adapter="openai")
    # When under the cap, _prune_tools returns tools unsorted. Ranking is
    # exposed via top_keyword_scores instead, which the clarification path
    # reads.
    keyword_scores = svc._last_prune_meta["top_keyword_scores"]
    assert keyword_scores[0][0] == "create_workflow"


def test_prune_populates_last_prune_meta():
    svc = _svc()
    tools = [
        _tool("create_workflow", description="create a workflow"),
        _tool("list_agents", description="list agents"),
    ]
    svc._prune_tools(tools, user_message="create a workflow", adapter="openai")
    meta = svc._last_prune_meta
    assert meta is not None
    assert meta["adapter"] == "openai"
    assert meta["total"] == 2
    assert meta["kept"] <= 2
    assert "top_scores" in meta
    assert "top_keyword_scores" in meta
    assert isinstance(meta["took_ms"], int)


def test_top_keyword_scores_excludes_always_include_sentinels():
    """Load-bearing: ambiguity detection reads top_keyword_scores, not
    top_scores. ALWAYS_INCLUDE sentinels (score 200) would otherwise
    dominate the top-5 and mask real keyword signal."""
    svc = _svc()
    tools = [
        _tool("get_help"),                           # ALWAYS_INCLUDE → score 200
        _tool("search_api_capabilities"),            # ALWAYS_INCLUDE → score 200
        _tool("create_workflow", description="create workflow"),
        _tool("list_agents", description="list agents"),
    ]
    svc._prune_tools(tools, user_message="create a workflow",
                      adapter="openai")
    keyword_names = {n for (n, _s) in svc._last_prune_meta["top_keyword_scores"]}
    assert "get_help" not in keyword_names
    assert "search_api_capabilities" not in keyword_names
    assert "create_workflow" in keyword_names


# ---------------------------------------------------------------------------
# _prune_is_ambiguous
# ---------------------------------------------------------------------------

def test_ambiguous_when_close_low_scores():
    svc = _svc()
    svc._last_prune_meta = {
        "top_keyword_scores": [("tool_a", 3), ("tool_b", 2)],
    }
    assert svc._prune_is_ambiguous() is True


def test_not_ambiguous_when_clear_winner():
    svc = _svc()
    svc._last_prune_meta = {
        "top_keyword_scores": [("tool_a", 50), ("tool_b", 2)],
    }
    assert svc._prune_is_ambiguous() is False


def test_not_ambiguous_when_top_score_is_zero():
    """No keyword hits at all → the user is off-topic (small talk slipped
    past _needs_tools). Clarification would confuse, not help."""
    svc = _svc()
    svc._last_prune_meta = {
        "top_keyword_scores": [("tool_a", 0), ("tool_b", 0)],
    }
    assert svc._prune_is_ambiguous() is False


def test_not_ambiguous_when_fewer_than_two_candidates():
    svc = _svc()
    svc._last_prune_meta = {"top_keyword_scores": [("tool_a", 3)]}
    assert svc._prune_is_ambiguous() is False


def test_not_ambiguous_when_no_meta():
    svc = _svc()
    # No _last_prune_meta attribute at all.
    assert svc._prune_is_ambiguous() is False


# ---------------------------------------------------------------------------
# _build_clarification_block
# ---------------------------------------------------------------------------

def test_build_clarification_block_emits_text_with_candidates():
    svc = _svc()
    svc._last_prune_meta = {
        "top_keyword_scores": [
            ("create_workflow", 4),
            ("update_workflow", 3),
            ("list_workflows", 2),
        ],
    }
    block = svc._build_clarification_block()
    assert block is not None
    assert block["type"] == "text"
    candidates = block["data"]["candidates"]
    assert len(candidates) == 3
    assert candidates[0]["tool"] == "create_workflow"
    # One action per candidate, all `open` verb (idempotent-only rule).
    assert all(a["verb"] == "open" for a in block["actions"])
    # First candidate's button is primary.
    assert block["actions"][0]["primary"] is True
    assert block["actions"][1]["primary"] is False


def test_build_clarification_block_returns_none_when_not_enough():
    svc = _svc()
    svc._last_prune_meta = {"top_keyword_scores": [("create_workflow", 4)]}
    assert svc._build_clarification_block() is None
