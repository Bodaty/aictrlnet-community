"""Scripted AI adapter for workflow AI nodes in the local test gates (llm/scripted.py).

Registered into the adapter registry only while `LLM_SCRIPTED_MODE=1` — never in
the adapter catalog — and forced by `ai_process_node` in that mode, so a gate's
AI nodes are answered from the script instead of a model.
"""
from typing import List

from adapters.base_adapter import BaseAdapter
from adapters.models import AdapterCapability, AdapterRequest, AdapterResponse


class ScriptedAdapter(BaseAdapter):
    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def get_capabilities(self) -> List[AdapterCapability]:
        return [
            AdapterCapability(name="generate", description="Scripted text generation"),
            AdapterCapability(name="chat", description="Scripted chat"),
        ]

    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        from llm.scripted import _respond, answer

        params = request.parameters or {}
        respond = await _respond(answer(
            messages=params.get("messages"), prompt=params.get("prompt") or params.get("system") or "",
            task_type=params.get("task_type"),
        ))
        return AdapterResponse(
            request_id=request.id, capability=request.capability, status="success",
            data={"text": respond.get("text", ""), "model": "scripted:canned"}, duration_ms=0.0,
        )
