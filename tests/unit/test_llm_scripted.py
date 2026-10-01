"""Tier 0 tests for scripted LLM mode (CONVERSATION_ORCHESTRATION_SPEC.md §8.1).

The default gate script (tests/e2e/fixtures/llm-script.default.json) is loaded
from the repo when it is mounted; the rule-matching tests use their own script.
"""

import json
import os
from types import SimpleNamespace

import pytest

from llm import scripted
from llm.models import ModelProvider
from llm.model_selection import get_provider_from_model
from llm.tier_resolver import is_ollama_model, provider_for_model_name

SCRIPT = {
    "rules": [
        {"match": {"task_type": "workflow_generation"}, "respond": {"text": "steps"}},
        {"match": {"message_regex": "^capabilities", "round": 0, "tools_offered": True},
         "respond": {"tool_calls": [{"name": "get_help", "arguments": {"topic": "all"}}]}},
        {"match": {"message_regex": "^capabilities"}, "respond": {"text": "after the tool round"}},
        {"match": {"prompt_regex": "extract key information"}, "respond": {"text": "[]"}},
        {"match": {"message_regex": "^boom"}, "respond": {"error": "scripted failure"}},
    ],
    "default": {"text": "default answer"},
}


@pytest.fixture
def script_file(tmp_path, monkeypatch):
    path = tmp_path / "script.json"
    path.write_text(json.dumps(SCRIPT))
    monkeypatch.setenv("LLM_SCRIPT_FILE", str(path))
    monkeypatch.setenv("LLM_SCRIPTED_MODE", "1")
    return path


def _user(text):
    return {"role": "user", "content": text}


def test_tool_round_then_text_counts_rounds_from_the_last_real_user_message(script_file):
    first = scripted.answer(messages=[_user("capabilities please")], tools_offered=True)
    assert first["tool_calls"][0]["name"] == "get_help"

    after_round = [
        _user("capabilities please"),
        {"role": "assistant", "content": "", "tool_calls": [{"id": "scripted-0"}]},
        {"role": "tool", "content": "{}"},
        _user("Based on the tool results above, answer the user."),
    ]
    # The synthesis prompt is not a user turn: the message still matches, round is 1.
    assert scripted.answer(messages=after_round, tools_offered=True)["text"] == "after the tool round"


def test_tools_offered_false_skips_the_tool_rule(script_file):
    assert scripted.answer(messages=[_user("capabilities")], tools_offered=False)["text"] == "after the tool round"


def test_prompt_task_type_and_default(script_file):
    assert scripted.answer(prompt="anything", task_type="workflow_generation")["text"] == "steps"
    assert scripted.answer(prompt="Please extract key information about the user")["text"] == "[]"
    assert scripted.answer(messages=[_user("unmatched")]) == SCRIPT["default"]


def test_script_edits_are_picked_up_without_a_restart(script_file):
    edited = dict(SCRIPT, default={"text": "edited"})
    script_file.write_text(json.dumps(edited))
    os.utime(script_file, (1, 1))  # a different mtime even within one clock tick
    assert scripted.answer(messages=[_user("unmatched")])["text"] == "edited"


async def test_tool_stream_streams_text_and_completes_with_tool_calls(script_file):
    events = [e async for e in scripted.tool_stream(
        messages=[_user("capabilities")], prompt="", tools=[{"name": "get_help"}], task_type=None)]
    assert events[-1]["type"] == "complete"
    response = events[-1]["response"]
    assert response.provider == ModelProvider.SCRIPTED
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [("scripted-0", "get_help", {"topic": "all"})]

    text_events = [e async for e in scripted.tool_stream(
        messages=[_user("unmatched")], prompt="", tools=[], task_type=None)]
    assert "".join(e["text"] for e in text_events if e["type"] == "text_delta") == "default answer"
    assert text_events[-1]["response"].tool_calls is None


async def test_scripted_error_raises(script_file):
    with pytest.raises(scripted.ScriptedError, match="scripted failure"):
        await scripted.generate_response(SimpleNamespace(prompt="", messages=[_user("boom")], task_type=None))


async def test_generate_response_is_a_scripted_llm_response(script_file):
    response = await scripted.generate_response(
        SimpleNamespace(prompt="x", messages=None, task_type="workflow_generation"))
    assert (response.text, response.model_used, response.provider) == ("steps", scripted.SCRIPTED_MODEL, ModelProvider.SCRIPTED)


def test_scripted_model_name_resolves_to_the_scripted_provider():
    assert get_provider_from_model(scripted.SCRIPTED_MODEL) == ModelProvider.SCRIPTED
    assert provider_for_model_name(scripted.SCRIPTED_MODEL) == "scripted"
    assert is_ollama_model(scripted.SCRIPTED_MODEL) is False


# --- boot guard -----------------------------------------------------------------

def _settings(environment="development", phi="false"):
    return SimpleNamespace(ENVIRONMENT=environment, AICTRLNET_PHI_MODE=phi)


def test_guard_is_a_no_op_when_scripted_mode_is_off(monkeypatch):
    monkeypatch.delenv("LLM_SCRIPTED_MODE", raising=False)
    scripted.assert_scripted_mode_allowed(_settings(environment="production"))


@pytest.mark.parametrize("environment", ["production", "staging", ""])
def test_guard_refuses_outside_development_and_test(script_file, environment):
    with pytest.raises(RuntimeError, match="LLM_SCRIPTED_MODE=1"):
        scripted.assert_scripted_mode_allowed(_settings(environment=environment))


def test_guard_refuses_with_phi_mode(script_file):
    with pytest.raises(RuntimeError, match="PHI"):
        scripted.assert_scripted_mode_allowed(_settings(phi="true"))


def test_guard_fails_at_boot_on_a_missing_or_broken_script(script_file, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_SCRIPT_FILE", str(tmp_path / "missing.json"))
    with pytest.raises(scripted.ScriptedError, match="does not exist"):
        scripted.assert_scripted_mode_allowed(_settings())

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"rules": []}))
    monkeypatch.setenv("LLM_SCRIPT_FILE", str(broken))
    with pytest.raises(scripted.ScriptedError, match="'rules' and 'default'"):
        scripted.assert_scripted_mode_allowed(_settings())


def test_guard_accepts_development_and_test(script_file):
    for environment in ("development", "test", " Development "):
        scripted.assert_scripted_mode_allowed(_settings(environment=environment))


def test_default_gate_script_is_well_formed():
    """The committed gate script parses and no text answer looks like a tool call."""
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "..", "..", "..", "..", "tests", "e2e", "fixtures", "llm-script.default.json"),
        "/opt/llm-scripts/llm-script.default.json",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        pytest.skip("gate script not mounted in this container")
    with open(path) as handle:
        script = json.load(handle)
    assert script["rules"] and script["default"].get("text")
    import re
    for rule in script["rules"] + [{"respond": script["default"]}]:
        text = rule["respond"].get("text") or ""
        assert not re.search(r"\b[a-z_]+\(.*\)", text), f"text shaped like a tool call: {text[:60]}"
