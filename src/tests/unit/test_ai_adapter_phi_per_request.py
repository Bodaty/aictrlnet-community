"""G10: the PHI egress check runs on every request, PHI pipelines carry explicit
sampling, and the Ollama handlers pass `format` and `images`.

The payload assertions go over real HTTP to a server on 127.0.0.1 in this
process, so what is asserted is what an Ollama or vLLM server would receive.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from adapters.models import AdapterCategory, AdapterConfig, AdapterRequest
from core import phi_egress
from core.phi_egress import PHI_REFUSAL_MARKER, PHIEgressRefused


SCHEMA = {
    "type": "object",
    "properties": {"is_referral": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["is_referral"],
}
PAGE_B64 = "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZ0FBQUFF"


class CannedLLMServer:
    """Answers Ollama's /api/chat, /api/generate, /api/ps and vLLM's
    /v1/chat/completions, and records every request it sees."""

    def __init__(self):
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, body, status=200):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                server.requests.append(("GET", self.path, None))
                if self.path == "/api/ps":
                    return self._reply({"models": []})
                return self._reply({"models": []})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append(("POST", self.path, body))
                if self.path == "/api/chat":
                    return self._reply({"model": body.get("model"), "message": {"role": "assistant", "content": "{}"}})
                if self.path == "/api/generate":
                    return self._reply({"model": body.get("model"), "response": "{}"})
                if self.path == "/v1/chat/completions":
                    return self._reply({
                        "id": "c1",
                        "model": body.get("model"),
                        "choices": [{"message": {"role": "assistant", "content": "{}"}}],
                        "usage": {"total_tokens": 3},
                    })
                return self._reply({"error": "not found"}, status=404)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()

    def posted(self, path):
        return [body for method, p, body in self.requests if method == "POST" and p == path]


@pytest.fixture
def server():
    with CannedLLMServer() as srv:
        yield srv


@pytest.fixture
def phi_mode(monkeypatch):
    monkeypatch.setenv("AICTRLNET_PHI_MODE", "true")
    monkeypatch.setenv("AICTRLNET_PHI_LLM_PROVIDERS", "ollama,vllm")
    monkeypatch.delenv("AICTRLNET_PHI_BAA_PROVIDERS", raising=False)
    phi_egress.reset_endpoint_cache()
    yield
    phi_egress.reset_endpoint_cache()


def _cfg(name, base_url):
    return AdapterConfig(name=name, version="1.0.0", category=AdapterCategory.AI, base_url=base_url)


async def _ollama(base_url):
    from adapters.implementations.ai.ollama_adapter import OllamaAdapter

    adapter = OllamaAdapter(_cfg("ollama", base_url))
    await adapter.initialize()
    return adapter


async def _vllm(base_url):
    from adapters.implementations.ai.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter(_cfg("vllm", base_url))
    await adapter.initialize()
    return adapter


def _chat(**kw):
    return AdapterRequest(
        capability="chat",
        parameters={"model": "qwen3.8:27b", "messages": [{"role": "user", "content": "Read this page."}]},
        **kw,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("build,public_url", [
    (_ollama, "http://ollama.example.com:11434"),
    (_vllm, "http://vllm.example.com:8000/v1"),
])
async def test_execute_rechecks_phi_after_endpoint_changes(phi_mode, server, monkeypatch, build, public_url):
    adapter = await build(server.url)
    try:
        monkeypatch.setattr(
            socket, "getaddrinfo",
            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.0.1", 0))],
        )
        adapter.base_url = public_url

        with pytest.raises(PHIEgressRefused) as excinfo:
            await adapter.execute(_chat(temperature=0))
        assert str(excinfo.value).startswith(PHI_REFUSAL_MARKER)
        assert not [r for r in server.requests if r[0] == "POST"]
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("build", [_ollama, _vllm])
async def test_execute_checks_the_url_the_live_client_sends_to(phi_mode, server, monkeypatch, build):
    """The client built at initialize keeps its own base URL; that is the URL
    that actually receives the request, so that is the one checked."""
    import httpx

    adapter = await build(server.url)
    try:
        monkeypatch.setattr(
            socket, "getaddrinfo",
            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.0.1", 0))],
        )
        await adapter.client.aclose()
        adapter.client = httpx.AsyncClient(base_url="http://llm.example.com:11434")

        with pytest.raises(PHIEgressRefused):
            await adapter.execute(_chat(temperature=0))
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("build", [_ollama, _vllm])
async def test_phi_pipeline_without_temperature_refused(phi_mode, server, build):
    adapter = await build(server.url)
    try:
        server.requests.clear()
        with pytest.raises(PHIEgressRefused) as excinfo:
            await adapter.execute(_chat(phi_pipeline=True))
        message = str(excinfo.value)
        assert message.startswith(PHI_REFUSAL_MARKER)
        assert "temperature" in message
        assert server.requests == []
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_phi_pipeline_without_temperature_allowed_when_phi_mode_off(monkeypatch, server):
    monkeypatch.delenv("AICTRLNET_PHI_MODE", raising=False)
    adapter = await _ollama(server.url)
    try:
        response = await adapter.execute(_chat(phi_pipeline=True))
        assert response.status == "success", response.error
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_non_pipeline_request_unchanged(phi_mode, server):
    """Existing callers (care-gap classification, the monthly narrative, voice)
    send no explicit temperature and are not PHI pipelines: under PHI mode they
    are not refused and their payload is exactly what it was."""
    adapter = await _ollama(server.url)
    try:
        chat_params = {
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.2},
        }
        response = await adapter.execute(AdapterRequest(capability="chat", parameters=chat_params))
        assert response.status == "success", response.error

        gen_params = {"model": "llama3.2:3b", "prompt": "hi", "options": {"top_p": 0.9}}
        response = await adapter.execute(AdapterRequest(capability="generate", parameters=gen_params))
        assert response.status == "success", response.error

        [chat] = server.posted("/api/chat")
        assert set(chat) == {"model", "messages", "options", "stream"}
        assert chat["messages"] == [{"role": "user", "content": "hi"}]
        assert {k: v for k, v in chat["options"].items() if k != "num_ctx"} == {"temperature": 0.2}
        assert chat["stream"] is False

        [gen] = server.posted("/api/generate")
        assert set(gen) == {"model", "prompt", "options", "stream"}
        assert {k: v for k, v in gen["options"].items() if k != "num_ctx"} == {"top_p": 0.9}
    finally:
        await adapter.shutdown()

    vllm = await _vllm(server.url)
    try:
        response = await vllm.execute(AdapterRequest(capability="chat", parameters={
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
        }))
        assert response.status == "success", response.error
        [body] = server.posted("/v1/chat/completions")
        assert body == {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 1.0, "stream": False}
    finally:
        await vllm.shutdown()


@pytest.mark.asyncio
async def test_ollama_chat_passes_format_and_images(phi_mode, server):
    adapter = await _ollama(server.url)
    try:
        response = await adapter.execute(_chat(
            phi_pipeline=True, temperature=0, response_format=SCHEMA, images=[PAGE_B64],
        ))
        assert response.status == "success", response.error

        [body] = server.posted("/api/chat")
        assert body["format"] == SCHEMA
        assert body["messages"][0]["images"] == [PAGE_B64]
        assert body["messages"][0]["content"] == "Read this page."
        assert body["options"]["temperature"] == 0
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", ["chat", "generate"])
@pytest.mark.parametrize("think", [False, True, None])
async def test_ollama_sends_think_only_when_set(server, capability, think):
    """`think` reaches Ollama's request body as given; unset, the body has no
    `think` and the model's default applies."""
    adapter = await _ollama(server.url)
    try:
        parameters = ({"model": "qwen3.8:27b", "messages": [{"role": "user", "content": "Read this page."}]}
                      if capability == "chat" else {"model": "qwen3.8:27b", "prompt": "Read this page."})
        response = await adapter.execute(AdapterRequest(
            capability=capability, parameters=parameters, temperature=0, think=think))
        assert response.status == "success", response.error

        [body] = server.posted(f"/api/{capability}")
        if think is None:
            assert "think" not in body
        else:
            assert body["think"] is think
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_ollama_generate_passes_format_images_and_temperature(phi_mode, server):
    adapter = await _ollama(server.url)
    try:
        response = await adapter.execute(AdapterRequest(
            capability="generate",
            parameters={"model": "qwen3.8:27b", "prompt": "Read this page.", "options": {"temperature": 0.9}},
            phi_pipeline=True, temperature=0, response_format=SCHEMA, images=[PAGE_B64],
        ))
        assert response.status == "success", response.error

        [body] = server.posted("/api/generate")
        assert body["format"] == SCHEMA
        assert body["images"] == [PAGE_B64]
        assert body["options"]["temperature"] == 0
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_vllm_uses_the_explicit_temperature_and_schema(phi_mode, server):
    adapter = await _vllm(server.url)
    try:
        response = await adapter.execute(_chat(phi_pipeline=True, temperature=0, response_format=SCHEMA))
        assert response.status == "success", response.error
        [body] = server.posted("/v1/chat/completions")
        assert body["temperature"] == 0
        assert body["response_format"] == {
            "type": "json_schema", "json_schema": {"name": "response", "schema": SCHEMA},
        }

        response = await adapter.execute(_chat(phi_pipeline=True, temperature=0, images=[PAGE_B64]))
        assert response.status == "error"
        assert "images" in response.error
        assert len(server.posted("/v1/chat/completions")) == 1
    finally:
        await adapter.shutdown()


# --- aiProcess sets the flags from the run's context --------------------------


def _recording_adapter_class():
    from adapters.base_adapter import BaseAdapter
    from adapters.models import AdapterCapability, AdapterResponse

    recorded = []

    class _RecordingAdapter(BaseAdapter):
        async def initialize(self):
            return None

        async def shutdown(self):
            return None

        def get_capabilities(self):
            return [AdapterCapability(name=n, description=n) for n in ("generate", "extract_fields")]

        async def execute(self, request):
            recorded.append(request)
            return AdapterResponse(
                request_id=request.id, capability=request.capability, status="success",
                data={"text": "{}"}, duration_ms=0.0,
            )

    return _RecordingAdapter, recorded


@pytest.fixture
def recording_adapter(monkeypatch):
    from adapters.registry import adapter_registry

    monkeypatch.delenv("LLM_SCRIPTED_MODE", raising=False)
    cls, recorded = _recording_adapter_class()
    adapter_registry.register_adapter_class("recording_a4", cls, AdapterCategory.AI)
    yield recorded
    adapter_registry._adapter_classes.pop("recording_a4", None)


def _ai_node(**parameters):
    from nodes.implementations.ai_process_node import AIProcessNode
    from nodes.models import NodeConfig, NodeType

    return AIProcessNode(NodeConfig(
        id="a4", name="a4", type=NodeType.TASK,
        parameters={"adapter_id": "recording_a4", "model": "m", **parameters},
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("parameters", [
    {"ai_task": "generate", "prompt": "Read this page."},
    {"ai_task": "custom", "capability": "extract_fields", "request_parameters": {"model": "m"}},
])
async def test_ai_step_in_phi_workflow_sends_a_pipeline_request(recording_adapter, parameters):
    node = _ai_node(temperature=0, response_format=SCHEMA, **parameters)
    await node.execute({}, {"workflow_phi": True})

    [request] = recording_adapter
    assert request.phi_pipeline is True
    assert request.temperature == 0
    assert request.response_format == SCHEMA


@pytest.mark.asyncio
async def test_ai_step_in_phi_workflow_without_temperature_carries_none(recording_adapter):
    """No default stands in for an explicit temperature: the adapter refuses it."""
    await _ai_node(ai_task="generate", prompt="x").execute({}, {"workflow_phi": True})

    [request] = recording_adapter
    assert request.phi_pipeline is True
    assert request.temperature is None


@pytest.mark.asyncio
@pytest.mark.parametrize("context", [{"workflow_phi": True}, {}])
async def test_ai_step_think_reaches_the_request(recording_adapter, context):
    """document-read's extraction step sets think: false; a step that sets
    none sends None (the provider's default)."""
    await _ai_node(ai_task="generate", prompt="x", temperature=0, think=False).execute({}, context)
    await _ai_node(ai_task="generate", prompt="x", temperature=0).execute({}, context)

    first, second = recording_adapter
    assert first.think is False
    assert second.think is None


@pytest.mark.asyncio
async def test_ai_step_outside_phi_workflow_is_unchanged(recording_adapter):
    """care-gap-refresh.json's classifier sets temperature 0.2 and
    response_format "json"; outside a phi workflow neither reaches the request."""
    await _ai_node(ai_task="generate", prompt="x", temperature=0.2, response_format="json").execute({}, {})

    [request] = recording_adapter
    assert request.phi_pipeline is False
    assert request.temperature is None
    assert request.response_format is None
    assert request.images is None


# --- The execution service puts the workflow's phi flag on every node ---------


@pytest.mark.asyncio
@pytest.mark.parametrize("definition_phi,expected", [(True, True), (None, False)])
@pytest.mark.parametrize("status_name", ["RUNNING", "RESUMING"])
async def test_execution_context_carries_workflow_phi(definition_phi, expected, status_name):
    import uuid
    from datetime import datetime

    from sqlalchemy import delete

    from core.database import bind_session_tenant, get_session_maker
    from models.community import WorkflowDefinition
    from models.workflow_execution import WorkflowExecution, WorkflowExecutionStatus
    from services.workflow_execution import WorkflowExecutionService

    tenant = "default-tenant"
    definition = {
        "nodes": [{"id": "s", "type": "start"}, {"id": "e", "type": "end"}],
        "edges": [{"source": "s", "target": "e"}],
    }
    if definition_phi is not None:
        definition["phi"] = definition_phi

    workflow_id = str(uuid.uuid4())
    maker = get_session_maker()
    async with maker() as db:
        bind_session_tenant(db, tenant)
        db.add(WorkflowDefinition(
            id=workflow_id, name=f"a4-phi-{workflow_id[:8]}", tenant_id=tenant, definition=definition,
        ))
        await db.flush()
        execution = WorkflowExecution(
            workflow_id=workflow_id, tenant_id=tenant, context={}, input_data={},
            status=getattr(WorkflowExecutionStatus, status_name), started_at=datetime.utcnow(),
        )
        db.add(execution)
        await db.commit()
        execution_id = execution.id

    seen = {}

    class _CapturingExecutor:
        async def execute_workflow(self, workflow_instance, is_cancelled=None, between_batches=None):
            for node_id, ni in workflow_instance.node_instances.items():
                seen[node_id] = ni.context.get("workflow_phi")

    try:
        async with maker() as db:
            service = WorkflowExecutionService(db)
            service.node_executor = _CapturingExecutor()
            await service._execute_workflow_locally(execution_id, tenant_id=tenant)
        assert seen == {"s": expected, "e": expected}
    finally:
        async with maker() as db:
            bind_session_tenant(db, tenant)
            await db.execute(delete(WorkflowExecution).where(WorkflowExecution.id == execution_id))
            await db.execute(delete(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id))
            await db.commit()
