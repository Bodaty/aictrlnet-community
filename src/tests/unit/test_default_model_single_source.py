"""Guard: DEFAULT_LLM_MODEL must have ONE source of truth (Phase 1 target).

Today four divergent literals exist (community core/config.py, tier_resolver.py
module constant + tier-fallback default, enterprise config.py), so with the env
var unset, settings.DEFAULT_LLM_MODEL and get_environment_default_model() can
resolve DIFFERENT models in the same deployment. Strict xfail: flips green when
Phase 1 routes everything through the settings accessor.
Plan: .claude/plans/llm-generate-fallback-and-model-selection.md
"""
import pytest


@pytest.mark.xfail(
    strict=True,
    reason="Phase 1: four divergent DEFAULT_LLM_MODEL literals (config, tier_resolver x2, enterprise)",
)
def test_default_model_has_single_source_when_env_unset(monkeypatch):
    monkeypatch.delenv("DEFAULT_LLM_MODEL", raising=False)
    from core.config import Settings
    from llm.tier_resolver import get_environment_default_model

    assert get_environment_default_model() == Settings().DEFAULT_LLM_MODEL
