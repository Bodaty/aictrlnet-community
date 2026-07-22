"""Guard: DEFAULT_LLM_MODEL has ONE source of truth (Phase 1, shipped).

The only literal lives in community core/config.py Settings.DEFAULT_LLM_MODEL;
tier_resolver's accessor reads env-then-Settings-default, the tier-fallback
param resolves through the accessor, and enterprise inherits the Community
field. This guard keeps divergent literals from creeping back.
Plan: .claude/plans/llm-generate-fallback-and-model-selection.md
"""


def test_default_model_has_single_source_when_env_unset(monkeypatch):
    monkeypatch.delenv("DEFAULT_LLM_MODEL", raising=False)
    from core.config import Settings
    from llm.tier_resolver import get_environment_default_model

    assert get_environment_default_model() == Settings().DEFAULT_LLM_MODEL
