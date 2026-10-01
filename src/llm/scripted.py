"""Scripted LLM mode for the local gates (CONVERSATION_ORCHESTRATION_SPEC.md §8.1 Tier 1).

With `LLM_SCRIPTED_MODE=1` every model call in the stack is answered from a JSON
script (`LLM_SCRIPT_FILE`) instead of a model, so the gates test what the
pipeline does with model output — routing, budgets, terminal events, UI — and
never a model's choices or speed (§8.2).

Refuses to start outside development/test or together with PHI mode. A scripted
call that fails raises `ScriptedError`, which every fallback ladder re-raises:
scripted mode never quietly falls back to a real model.

Script: `{"rules": [{"match": {...}, "respond": {...}}, ...], "default": {...}}`
  match:   message_regex (last real user message), prompt_regex (non-chat calls),
           round (tool rounds since that message), tools_offered (bool),
           task_type
  respond: text, tool_calls [{name, arguments}], delay_ms, error
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

_SYNTHESIS_PREFIX = "Based on the tool results above"
_cache: Dict[str, Any] = {"path": None, "mtime": None, "script": None}

SCRIPTED_MODEL = "scripted:canned"
ALLOWED_ENVIRONMENTS = frozenset({"development", "test"})


class ScriptedError(RuntimeError):
    """A scripted answer that is an error by design, or a broken script."""


def scripted_mode() -> bool:
    return os.environ.get("LLM_SCRIPTED_MODE", "").strip() == "1"


def assert_scripted_mode_allowed(settings) -> None:
    """Boot guard: scripted answers are for local gates, never a deployment."""
    if not scripted_mode():
        return
    environment = str(getattr(settings, "ENVIRONMENT", "production")).strip().lower()
    if environment not in ALLOWED_ENVIRONMENTS:
        raise RuntimeError(
            f"LLM_SCRIPTED_MODE=1 is set with ENVIRONMENT={environment!r}. Scripted model answers are "
            f"for local test gates only ({', '.join(sorted(ALLOWED_ENVIRONMENTS))}). Unset LLM_SCRIPTED_MODE."
        )
    if str(getattr(settings, "AICTRLNET_PHI_MODE", "")).strip().lower() in ("1", "true", "on"):
        raise RuntimeError("LLM_SCRIPTED_MODE=1 cannot run with AICTRLNET_PHI_MODE on.")
    _load()  # a missing or broken script fails at boot, not on the first turn


def _load() -> Dict[str, Any]:
    path = os.environ.get("LLM_SCRIPT_FILE", "")
    if not path or not os.path.exists(path):
        raise ScriptedError(f"LLM_SCRIPT_FILE {path!r} does not exist")
    mtime = os.path.getmtime(path)
    if _cache["path"] != path or _cache["mtime"] != mtime:
        with open(path) as handle:
            script = json.load(handle)
        if "rules" not in script or "default" not in script:
            raise ScriptedError(f"{path}: a script needs 'rules' and 'default'")
        _cache.update(path=path, mtime=mtime, script=script)
    return _cache["script"]


def _last_real_user_index(messages: List[Dict[str, Any]]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "user" and not str(m.get("content") or "").startswith(_SYNTHESIS_PREFIX):
            return i
    return -1


def answer(*, messages: Optional[List[Dict[str, Any]]] = None, prompt: Optional[str] = None,
           tools_offered: bool = False, task_type: Optional[str] = None) -> Dict[str, Any]:
    """The scripted response for one model call."""
    script = _load()
    messages = messages or []
    user_index = _last_real_user_index(messages)
    user_message = str(messages[user_index].get("content") or "") if user_index >= 0 else ""
    round_number = sum(
        1 for m in messages[user_index + 1:] if m.get("role") == "assistant" and m.get("tool_calls")
    )
    for rule in script["rules"]:
        match = rule.get("match") or {}
        if "message_regex" in match and not re.search(match["message_regex"], user_message, re.I):
            continue
        if "prompt_regex" in match and not re.search(match["prompt_regex"], prompt or "", re.I | re.S):
            continue
        if "round" in match and match["round"] != round_number:
            continue
        if "tools_offered" in match and bool(match["tools_offered"]) != bool(tools_offered):
            continue
        if "task_type" in match and match["task_type"] != task_type:
            continue
        return rule["respond"]
    return script["default"]


async def _respond(respond: Dict[str, Any]) -> Dict[str, Any]:
    import asyncio

    if respond.get("delay_ms"):
        await asyncio.sleep(respond["delay_ms"] / 1000)
    if respond.get("error"):
        raise ScriptedError(respond["error"])
    return respond


async def generate_response(request):
    """An `LLMResponse` for a plain generation request (workflow generation, memory
    extraction, template matching, AI text) — answered from the script."""
    from llm.models import LLMResponse, ModelProvider, ModelTier

    respond = await _respond(answer(
        messages=getattr(request, "messages", None), prompt=request.prompt,
        task_type=getattr(request, "task_type", None),
    ))
    return LLMResponse(
        text=respond.get("text", ""), model_used=SCRIPTED_MODEL, provider=ModelProvider.SCRIPTED,
        tier=ModelTier.QUALITY, metadata={"resolution_source": "scripted"},
    )


async def tool_stream(*, messages, prompt, tools, task_type):
    """The `generate_with_tools_stream` events for one scripted round."""
    from llm.models import LLMToolResponse, ModelProvider, ModelTier, ToolCall

    respond = await _respond(answer(
        messages=messages, prompt=prompt, tools_offered=bool(tools), task_type=task_type,
    ))
    text = respond.get("text")
    if text:
        for start in range(0, len(text), 24):  # stream like a model does
            yield {"type": "text_delta", "text": text[start:start + 24]}
    calls = [
        ToolCall(id=f"scripted-{i}", name=call["name"], arguments=call.get("arguments") or {})
        for i, call in enumerate(respond.get("tool_calls") or [])
    ]
    yield {"type": "complete", "response": LLMToolResponse(
        text=text, model_used=SCRIPTED_MODEL, provider=ModelProvider.SCRIPTED, tier=ModelTier.QUALITY,
        tool_calls=calls or None, metadata={"resolution_source": "scripted"},
    )}
