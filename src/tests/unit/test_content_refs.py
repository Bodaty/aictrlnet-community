"""Content by reference in the AI step (G15).

A PHI pipeline's page text, images and extracted values must never enter run
history: every step's inputs and outputs are persisted, and a failed step's
error text is stored too. aiProcess takes its content from a registered,
tenant-scoped resolver (`input_ref`) and hands the model's output to a
registered sink (`output_sink`); the step's own output is ids, counts, model,
usage and duration. A failed image or by-reference call is never retried as a
text-only chat (R-09).
"""

import json

import pytest

from adapters.models import AdapterCategory, AdapterResponse
from core import phi_egress
from core.phi_egress import PHIEgressRefused
from nodes import content_refs
from nodes.content_refs import scrub_error

PAGE_TEXT = "SECRET-PAGE"
MODEL_TEXT = "SECRET-OUTPUT"
PROMPT_TEXT = "SECRET-PROMPT"


@pytest.fixture
def phi_mode(monkeypatch):
    monkeypatch.setenv("AICTRLNET_PHI_MODE", "true")
    monkeypatch.setenv("AICTRLNET_PHI_LLM_PROVIDERS", "ollama,llamacpp")
    monkeypatch.delenv("AICTRLNET_PHI_BAA_PROVIDERS", raising=False)
    phi_egress.reset_endpoint_cache()
    yield
    phi_egress.reset_endpoint_cache()


def _adapter_class(behaviour):
    from adapters.base_adapter import BaseAdapter
    from adapters.models import AdapterCapability

    recorded = []

    class _Adapter(BaseAdapter):
        async def initialize(self):
            return None

        async def shutdown(self):
            return None

        def get_capabilities(self):
            return [AdapterCapability(name=n, description=n) for n in ("generate", "chat", "ocr_page")]

        async def execute(self, request):
            recorded.append(request)
            return behaviour(request)

    return _Adapter, recorded


def _answer(request):
    return AdapterResponse(
        request_id=request.id, capability=request.capability, status="success",
        data={"text": MODEL_TEXT, "model": "reader-1", "usage": {"total_tokens": 7}},
        duration_ms=12.0,
    )


def _sent_text(request):
    messages = request.parameters.get("messages") or [{}]
    return request.parameters.get("prompt") or messages[-1].get("content")


def _echo_prompt_and_raise(request):
    raise RuntimeError(f"model rejected the request: {_sent_text(request)}")


def _echo_prompt_in_error_status(request):
    return AdapterResponse(
        request_id=request.id, capability=request.capability, status="error",
        error=f"bad request: {_sent_text(request)}", duration_ms=1.0,
    )


@pytest.fixture
def adapter(monkeypatch):
    """Registers a recording adapter under a test id; `use(behaviour)` picks
    how it answers. Yields the list of requests it received."""
    from adapters.registry import adapter_registry

    monkeypatch.delenv("LLM_SCRIPTED_MODE", raising=False)
    state = {}

    def use(behaviour):
        cls, recorded = _adapter_class(behaviour)
        adapter_registry.register_adapter_class("recording_a6", cls, AdapterCategory.AI)
        state["recorded"] = recorded
        return recorded

    yield use
    adapter_registry._adapter_classes.pop("recording_a6", None)


@pytest.fixture
def refs():
    """A test resolver and sink, registered for this test only."""
    calls = {"resolve": [], "write": []}

    async def resolve(tenant_id, ref_id, context):
        calls["resolve"].append((tenant_id, ref_id, context.get("workflow_id")))
        return {"text": PAGE_TEXT, "images": ["aW1hZ2U="]}

    async def write(tenant_id, sink_id, context, data):
        calls["write"].append((tenant_id, sink_id, data))
        return {"page_id": sink_id, "chars": len(data.get("text") or "")}

    content_refs.register_resolver("test_page_a6", resolve)
    content_refs.register_sink("test_text_a6", write)
    yield calls
    content_refs.unregister_resolver("test_page_a6")
    content_refs.unregister_sink("test_text_a6")


def _ai_node(**parameters):
    from nodes.implementations.ai_process_node import AIProcessNode
    from nodes.models import NodeConfig, NodeType

    return AIProcessNode(NodeConfig(
        id="a6", name="a6", type=NodeType.TASK,
        parameters={"adapter_id": "recording_a6", **parameters},
    ))


def _by_ref_node(**extra):
    return _ai_node(
        ai_task="custom",
        capability="ocr_page",
        temperature=0,
        request_parameters={"prompt": "Read this page.", "max_tokens": 512},
        input_ref={"kind": "test_page_a6", "id": "{{page_id}}"},
        output_sink={"kind": "test_text_a6", "id": "{{page_id}}"},
        **extra,
    )


# --- content by reference -----------------------------------------------------


@pytest.mark.asyncio
async def test_input_ref_resolved_per_tenant_and_output_has_no_text(adapter, refs):
    recorded = adapter(_answer)
    context = {"tenant_id": "tenant-a6", "workflow_phi": True, "workflow_id": "wf-1"}

    output = await _by_ref_node().execute({"page_id": "page-1", "text": "INLINE"}, context)

    assert refs["resolve"] == [("tenant-a6", "page-1", "wf-1")]
    [request] = recorded
    assert PAGE_TEXT in request.parameters["prompt"]
    assert request.images == ["aW1hZ2U="]
    assert "text" not in request.parameters  # step input is not merged in by-reference mode
    assert request.phi_pipeline is True and request.temperature == 0
    [(tenant, sink_id, data)] = refs["write"]
    assert (tenant, sink_id, data["text"]) == ("tenant-a6", "page-1", MODEL_TEXT)

    assert set(output) == {"ref", "sink", "model", "usage", "duration_ms"}
    assert output["ref"] == {"kind": "test_page_a6", "id": "page-1"}
    assert output["sink"] == {"page_id": "page-1", "chars": len(MODEL_TEXT)}
    assert output["model"] == "reader-1"
    assert output["usage"] == {"total_tokens": 7}
    dumped = json.dumps(output)
    assert PAGE_TEXT not in dumped
    assert MODEL_TEXT not in dumped


@pytest.mark.asyncio
async def test_by_reference_request_sends_no_default_model(adapter, refs):
    """llama-server serves the model it was started with: a by-reference request
    must not carry aiProcess's default (Ollama) model name."""
    recorded = adapter(_answer)

    await _by_ref_node().execute({"page_id": "page-1"}, {"tenant_id": "tenant-a6"})

    [request] = recorded
    assert "model" not in request.parameters


@pytest.mark.asyncio
async def test_by_reference_request_carries_the_nodes_model(adapter, refs):
    recorded = adapter(_answer)

    node = _by_ref_node()
    node.config.parameters["request_parameters"]["model"] = "qwen3.8:27b"
    await node.execute({"page_id": "page-1"}, {"tenant_id": "tenant-a6"})

    [request] = recorded
    assert request.parameters["model"] == "qwen3.8:27b"


@pytest.mark.asyncio
async def test_input_ref_appends_text_to_chat_messages(adapter, refs):
    recorded = adapter(_answer)
    node = _ai_node(
        ai_task="custom", capability="chat", temperature=0,
        request_parameters={"model": "m", "messages": [{"role": "system", "content": "Extract."}]},
        input_ref={"kind": "test_page_a6", "id": "doc-1"},
        output_sink={"kind": "test_text_a6", "id": "doc-1"},
    )

    await node.execute({}, {"tenant_id": "tenant-a6"})

    [request] = recorded
    assert request.parameters["messages"] == [
        {"role": "system", "content": "Extract."},
        {"role": "user", "content": PAGE_TEXT},
    ]
    assert node.config.parameters["request_parameters"]["messages"] == [
        {"role": "system", "content": "Extract."}
    ]


@pytest.mark.asyncio
async def test_inline_content_refused_in_phi_workflow(adapter, phi_mode):
    recorded = adapter(_answer)
    node = _ai_node(ai_task="generate", prompt="{{text}}", temperature=0, model="m")

    with pytest.raises(PHIEgressRefused, match="input_ref"):
        await node.execute({"text": PAGE_TEXT}, {"tenant_id": "tenant-a6", "workflow_phi": True})

    assert recorded == []


@pytest.mark.asyncio
async def test_inline_content_allowed_outside_phi_workflow(adapter, phi_mode):
    recorded = adapter(_answer)
    node = _ai_node(ai_task="generate", prompt="{{text}}", model="m")

    await node.execute({"text": "hello"}, {"tenant_id": "tenant-a6"})

    assert len(recorded) == 1


@pytest.mark.asyncio
async def test_by_reference_step_runs_under_phi_mode(adapter, refs, phi_mode):
    recorded = adapter(_answer)

    output = await _by_ref_node().execute(
        {"page_id": "page-1"}, {"tenant_id": "tenant-a6", "workflow_phi": True}
    )

    assert len(recorded) == 1
    assert output["sink"] == {"page_id": "page-1", "chars": len(MODEL_TEXT)}


@pytest.mark.asyncio
@pytest.mark.parametrize("field,kind", [("input_ref", "nope_input"), ("output_sink", "nope_sink")])
async def test_unknown_kind_fails_clearly(adapter, refs, field, kind):
    recorded = adapter(_answer)
    node = _by_ref_node()
    node.config.parameters[field] = {"kind": kind, "id": "x"}

    with pytest.raises(content_refs.ContentRefError, match=f"'{kind}'"):
        await node.execute({"page_id": "page-1"}, {"tenant_id": "tenant-a6"})

    assert recorded == []


@pytest.mark.asyncio
@pytest.mark.parametrize("parameters,match", [
    ({"output_sink": None}, "output_sink"),
    ({"ai_task": "generate"}, "custom"),
    ({"input_ref": {"kind": "test_page_a6"}}, "'kind' and 'id'"),
])
async def test_malformed_reference_fails_clearly(adapter, refs, parameters, match):
    recorded = adapter(_answer)
    node = _by_ref_node()
    for key, value in parameters.items():
        if value is None:
            node.config.parameters.pop(key)
        else:
            node.config.parameters[key] = value

    with pytest.raises(content_refs.ContentRefError, match=match):
        await node.execute({"page_id": "page-1"}, {"tenant_id": "tenant-a6"})

    assert recorded == []


@pytest.mark.asyncio
async def test_by_reference_needs_a_tenant(adapter, refs, monkeypatch):
    import core.tenant_context as tenant_context

    monkeypatch.setattr(tenant_context, "get_current_tenant_id", lambda: None)
    recorded = adapter(_answer)

    with pytest.raises(content_refs.ContentRefError, match="tenant"):
        await _by_ref_node().execute({"page_id": "page-1"}, {})

    assert refs["resolve"] == [] and recorded == []


# --- errors are scrubbed in phi workflows -------------------------------------


def test_scrub_error_redacts_only_in_phi_workflows():
    assert scrub_error(f"boom {PAGE_TEXT}", True, "ValueError") == "<redacted: ValueError>"
    assert scrub_error(f"boom {PAGE_TEXT}", False, "ValueError") == f"boom {PAGE_TEXT}"


def _instance(node, phi):
    from nodes.models import NodeInstance

    return NodeInstance(
        node_config=node.config, workflow_instance_id="wf-a6",
        input_data={"page_id": "page-1"},
        context={"tenant_id": "tenant-a6", "workflow_phi": phi},
    )


@pytest.mark.asyncio
async def test_error_text_scrubbed_in_phi_workflow(adapter):
    """The adapter echoes the prompt in its error; neither the step's stored
    error nor the error handler's error_details may carry it."""
    from nodes.error_handler import NodeErrorHandler
    from nodes.models import NodeStatus, WorkflowInstance

    adapter(_echo_prompt_and_raise)
    node = _ai_node(ai_task="generate", prompt=PROMPT_TEXT, temperature=0, model="m")
    instance = _instance(node, phi=True)

    result = await node.run(instance, {})

    assert result.status == NodeStatus.FAILED
    assert result.error == "<redacted: RuntimeError>"
    assert PROMPT_TEXT not in (instance.last_error or "")

    saved = []

    class _StateManager:
        async def save_node_state(self, node_instance, state, workflow_id):
            saved.append(state)

    handler = NodeErrorHandler(state_manager=_StateManager())
    workflow = WorkflowInstance(id="wf-a6", template_id="wf-a6", name="a6")
    try:
        raise RuntimeError(f"model rejected the request: {PROMPT_TEXT}")
    except RuntimeError as err:
        handled = await handler.handle_node_error(err, instance, workflow)

    dumped = json.dumps(saved, default=str) + str(handled.error)
    assert PROMPT_TEXT not in dumped
    assert "<redacted: RuntimeError>" in dumped


@pytest.mark.asyncio
async def test_error_text_kept_outside_phi_workflow(adapter):
    adapter(_echo_prompt_and_raise)
    node = _ai_node(ai_task="generate", prompt=PROMPT_TEXT, model="m")

    result = await node.run(_instance(node, phi=False), {})

    assert PROMPT_TEXT in result.error


# --- R13: no text-only chat retry for image, by-reference or phi calls --------


@pytest.mark.asyncio
@pytest.mark.parametrize("behaviour", [_echo_prompt_and_raise, _echo_prompt_in_error_status])
async def test_failed_call_in_phi_workflow_is_not_retried_as_chat(adapter, behaviour):
    from nodes.models import NodeStatus

    recorded = adapter(behaviour)
    node = _ai_node(ai_task="generate", prompt=PROMPT_TEXT, temperature=0, model="m")

    result = await node.run(_instance(node, phi=True), {})

    assert result.status == NodeStatus.FAILED
    assert len(recorded) == 1
    assert recorded[0].capability == "generate"


@pytest.mark.asyncio
async def test_failed_by_reference_call_is_not_retried(adapter, refs):
    from nodes.models import NodeStatus

    recorded = adapter(_echo_prompt_in_error_status)
    node = _by_ref_node()

    result = await node.run(_instance(node, phi=False), {})

    assert result.status == NodeStatus.FAILED
    assert len(recorded) == 1
    assert PAGE_TEXT not in result.error
    assert refs["write"] == []


@pytest.mark.asyncio
async def test_failed_image_call_is_not_retried_as_chat(adapter):
    recorded = adapter(_echo_prompt_and_raise)
    node = _ai_node(ai_task="generate", model="m")

    with pytest.raises(RuntimeError):
        await node._call_adapter(
            _adapter_instance(), "generate", {"prompt": "x", "model": "m"}, {"images": ["aW1hZ2U="]},
        )
    assert len(recorded) == 1


@pytest.mark.asyncio
async def test_failed_plain_call_still_retries_as_chat(adapter):
    recorded = adapter(_echo_prompt_and_raise)
    node = _ai_node(ai_task="generate", model="m")

    with pytest.raises(RuntimeError):
        await node._call_adapter(_adapter_instance(), "generate", {"prompt": "x", "model": "m"}, {})
    assert [r.capability for r in recorded] == ["generate", "chat"]


def _adapter_instance():
    from adapters.models import AdapterConfig
    from adapters.registry import adapter_registry

    cls = adapter_registry.get_adapter_class("recording_a6")
    return cls(AdapterConfig(name="recording_a6", version="1.0.0", category=AdapterCategory.AI))


# --- previews -----------------------------------------------------------------
# A preview writes through a sink only when the sink says it handles previews;
# any other sink is not called, so a sink added later can't act in a preview.

@pytest.fixture
def two_sinks():
    calls = []

    async def plain(tenant_id, sink_id, context, data):
        calls.append(("plain", sink_id))
        return {"saved": True}

    async def aware(tenant_id, sink_id, context, data):
        calls.append(("aware", sink_id, bool(context.get("preview"))))
        return {"saved": False, "chars": 3}

    content_refs.register_sink("plain-sink", plain)
    content_refs.register_sink("aware-sink", aware, previews=True)
    yield calls
    content_refs.unregister_sink("plain-sink")
    content_refs.unregister_sink("aware-sink")


@pytest.mark.asyncio
async def test_a_preview_does_not_call_a_sink_that_does_not_handle_previews(two_sinks):
    result = await content_refs.write_output({"kind": "plain-sink", "id": "x"}, "t1", {"preview": True}, {"text": "hi"})
    assert two_sinks == []
    assert result == {"saved": False, "preview": True}


@pytest.mark.asyncio
async def test_a_preview_calls_a_sink_that_handles_previews(two_sinks):
    result = await content_refs.write_output({"kind": "aware-sink", "id": "x"}, "t1", {"preview": True}, {"text": "hi"})
    assert two_sinks == [("aware", "x", True)]
    assert result == {"saved": False, "chars": 3}


@pytest.mark.asyncio
async def test_a_live_run_calls_every_sink(two_sinks):
    await content_refs.write_output({"kind": "plain-sink", "id": "a"}, "t1", {}, {"text": "hi"})
    await content_refs.write_output({"kind": "aware-sink", "id": "b"}, "t1", {"preview": False}, {"text": "hi"})
    assert two_sinks == [("plain", "a"), ("aware", "b", False)]
