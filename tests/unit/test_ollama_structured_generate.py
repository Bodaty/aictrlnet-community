"""`LLMService.generate` reaches Ollama with the schema, `think` and temperature it was given.

The schema used to be dropped (LLMRequest's field is `output_schema`; it was passed as `schema`),
`think` was never sent (qwen3 then reasons before every answer), and temperature went top-level
where Ollama does not read it.
"""
import json

import httpx
import pytest

from llm import generation, ollama_options
from llm.models import LLMRequest

SCHEMA = {"type": "object", "properties": {"answer": {"type": "boolean"}}, "required": ["answer"]}


@pytest.fixture
def sent(monkeypatch):
    captured = {}

    async def none_loaded(base_url):
        return {}

    def handler(request):
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"response": '{"answer": true}', "eval_count": 3})

    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ollama_options, "_loaded_contexts", none_loaded)
    monkeypatch.setattr(generation.httpx, "AsyncClient", client)
    return captured


async def _generate(**kw):
    engine = generation.LLMGenerationEngine()
    return await engine._generate_with_ollama(LLMRequest(prompt="p", system_prompt="s", **kw), "qwen3:14b")


async def test_schema_think_and_temperature_reach_ollama(sent):
    response = await _generate(output_schema=SCHEMA, think=False, temperature=0)
    body = sent["json"]
    assert sent["path"].endswith("/api/generate")
    assert body["format"] == SCHEMA and body["think"] is False
    assert body["options"]["temperature"] == 0 and "temperature" not in body
    assert json.loads(response.text) == {"answer": True}


async def test_defaults_send_no_format_or_think(sent):
    await _generate()
    body = sent["json"]
    assert "format" not in body and "think" not in body
    assert body["options"]["temperature"] == 0.7


async def test_plain_json_format(sent):
    await _generate(response_format="json")
    assert sent["json"]["format"] == "json"


async def test_service_passes_schema_through(monkeypatch):
    from llm.service import LLMService

    seen = {}

    async def fake_generate(request):
        seen["request"] = request
        raise RuntimeError("stop here")

    s = LLMService()
    monkeypatch.setattr(s.generation_engine, "generate", fake_generate)
    with pytest.raises(RuntimeError):
        await s.generate("p", schema=SCHEMA, think=False)
    assert seen["request"].output_schema == SCHEMA and seen["request"].think is False
