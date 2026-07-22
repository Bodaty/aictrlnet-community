"""Canonical provider keys (Phase 1): one alias table + one adapter map.

User/org/tenant configs say things like "claude" or "google"; adapters register
under canonical names. normalize_provider() maps input to the canonical key and
PROVIDER_TO_ADAPTER is the single provider->adapter-candidates table (the two
near-duplicate maps in ai_process_node collapse into it).
Plan: .claude/plans/llm-generate-fallback-and-model-selection.md
"""
from llm.tier_resolver import PROVIDER_ALIASES, PROVIDER_TO_ADAPTER, normalize_provider


def test_known_aliases_resolve_to_canonical():
    assert normalize_provider("claude") == "anthropic"
    assert normalize_provider("google") == "gemini"
    assert normalize_provider("vertex-ai") == "vertex_ai"


def test_normalization_is_case_and_whitespace_insensitive():
    assert normalize_provider("Claude") == "anthropic"
    assert normalize_provider(" GOOGLE ") == "gemini"


def test_canonical_names_pass_through():
    for canonical in ("anthropic", "gemini", "openai", "ollama", "vertex_ai", "deepseek", "dashscope", "vllm"):
        assert normalize_provider(canonical) == canonical


def test_unknown_provider_lowercases_but_survives():
    assert normalize_provider("some-future-provider") == "some-future-provider"
    assert normalize_provider("SomeFuture") == "somefuture"


def test_empty_and_none_pass_through():
    assert normalize_provider("") == ""
    assert normalize_provider(None) is None


def test_every_alias_target_has_adapter_candidates():
    missing = set(PROVIDER_ALIASES.values()) - set(PROVIDER_TO_ADAPTER)
    assert not missing, f"alias targets without adapter candidates: {missing}"


def test_anthropic_candidates_cover_both_registry_names():
    assert "claude" in PROVIDER_TO_ADAPTER["anthropic"]
    assert "anthropic" in PROVIDER_TO_ADAPTER["anthropic"]


def test_org_llm_settings_normalizes_legacy_blobs():
    from llm.org_llm_settings import OrgLLMSettings

    s = OrgLLMSettings(
        preferred_provider="claude",
        fallback_provider="Google",
        allowed_providers=["Claude", "openai"],
        api_key_refs={"claude": "cred-1"},
    )
    assert s.preferred_provider == "anthropic"
    assert s.fallback_provider == "gemini"
    assert s.allowed_providers == ["anthropic", "openai"]
    assert s.api_key_refs == {"anthropic": "cred-1"}
    assert s.has_own_key("claude") is True


def test_org_llm_settings_normalizes_on_assignment():
    from llm.org_llm_settings import OrgLLMSettings

    s = OrgLLMSettings()
    s.preferred_provider = "claude"
    assert s.preferred_provider == "anthropic"
