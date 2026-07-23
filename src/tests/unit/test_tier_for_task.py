import logging

import pytest

from llm.generation import LLMGenerationEngine
from llm.models import LLMRequest, ModelTier


@pytest.fixture
def engine():
    return LLMGenerationEngine()


@pytest.mark.parametrize("task_type,expected", [
    ("structured_generation", ModelTier.BALANCED),
    ("planning", ModelTier.BALANCED),
    ("basic_agent", ModelTier.BALANCED),
    ("general", ModelTier.BALANCED),
    ("conversation", ModelTier.BALANCED),
    ("extraction", ModelTier.BALANCED),
    ("data_extraction", ModelTier.BALANCED),
    ("ai_task", ModelTier.BALANCED),
])
def test_explicit_mappings(engine, task_type, expected, caplog):
    with caplog.at_level(logging.WARNING, logger="llm.generation"):
        tier = engine._determine_tier_for_task(LLMRequest(prompt="x", task_type=task_type))
    assert tier == expected
    assert not caplog.records  # mapped tasks must not warn


def test_unknown_task_type_warns(engine, caplog):
    with caplog.at_level(logging.WARNING, logger="llm.generation"):
        tier = engine._determine_tier_for_task(LLMRequest(prompt="x", task_type="totally_new_thing"))
    assert tier == ModelTier.BALANCED
    assert any("totally_new_thing" in r.message for r in caplog.records)
