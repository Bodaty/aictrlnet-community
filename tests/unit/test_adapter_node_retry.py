"""An adapter step retries a transient upstream error when it asks to, and only then.

The GEO audit lost one to three Gemini answers on every run from Sep 4 to Nov 1
to HTTP 503 "high demand", and nothing retried them. A step can now set
retry_attempts / retry_backoff_seconds / retry_on_error_codes; the adapter's own
error_code (HTTP_503, ...) decides whether a failure is retried.
"""
import pytest

from nodes import base_node
from nodes.base_node import AdapterCallError
from nodes.implementations import adapter_node as adapter_node_module
from nodes.implementations.adapter_node import AdapterNode
from nodes.models import NodeConfig, NodeType


def _node(**params):
    base = {"adapter_id": "gemini", "capability": "answer", "fail_on_error": False}
    base.update(params)
    return AdapterNode(NodeConfig(name="ask", type=NodeType.ADAPTER, parameters=base))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(adapter_node_module.asyncio, "sleep", fake_sleep)
    return waits


def _script(node, monkeypatch, outcomes):
    calls = []

    async def fake_call_adapter(**kwargs):
        calls.append(kwargs)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(node, "call_adapter", fake_call_adapter)
    return calls


def _busy():
    return AdapterCallError("Adapter error: Gemini HTTP 503: high demand", "HTTP_503")


async def test_a_503_is_retried_and_the_answer_used(monkeypatch, _no_sleep):
    node = _node(retry_attempts=2, retry_backoff_seconds=15)
    calls = _script(node, monkeypatch, [_busy(), {"content": "answer"}])
    out = await node.execute({}, {})
    assert out["content"] == "answer"
    assert out["_attempts"] == 2
    assert len(calls) == 2
    assert _no_sleep == [15]


async def test_backoff_doubles_and_the_last_error_is_reported(monkeypatch, _no_sleep):
    node = _node(retry_attempts=2, retry_backoff_seconds=15)
    calls = _script(node, monkeypatch, [_busy(), _busy(), _busy()])
    out = await node.execute({}, {})
    assert out["_failed"] is True and "HTTP 503" in out["error"]
    assert len(calls) == 3
    assert _no_sleep == [15, 30]


async def test_no_retry_unless_the_step_asks(monkeypatch, _no_sleep):
    node = _node()
    calls = _script(node, monkeypatch, [_busy(), {"content": "never reached"}])
    out = await node.execute({}, {})
    assert out["_failed"] is True
    assert len(calls) == 1 and _no_sleep == []


async def test_a_bad_request_is_never_retried(monkeypatch, _no_sleep):
    node = _node(retry_attempts=2)
    calls = _script(node, monkeypatch, [
        AdapterCallError("Adapter error: Gemini HTTP 400: bad key", "HTTP_400"),
        {"content": "never reached"},
    ])
    out = await node.execute({}, {})
    assert out["_failed"] is True and len(calls) == 1


async def test_retry_codes_can_be_narrowed(monkeypatch, _no_sleep):
    node = _node(retry_attempts=2, retry_on_error_codes=["HTTP_503"])
    calls = _script(node, monkeypatch, [
        AdapterCallError("Adapter error: OpenAI HTTP 429: no credits", "HTTP_429"),
        {"content": "never reached"},
    ])
    out = await node.execute({}, {})
    assert out["_failed"] is True and len(calls) == 1


async def test_fail_on_error_still_raises_after_retries(monkeypatch, _no_sleep):
    node = _node(retry_attempts=1, fail_on_error=True)
    _script(node, monkeypatch, [_busy(), _busy()])
    with pytest.raises(AdapterCallError):
        await node.execute({}, {})


async def test_call_adapter_raises_the_typed_error_with_its_code(monkeypatch):
    # The error the base helper raises must still read "Adapter error: ..." (the
    # GEO scorecard and logs match on it) and carry the adapter's code.
    from adapters.models import AdapterResponse, AdapterStatus

    class _BusyAdapter:
        def __init__(self, config):
            self.status = None

        async def start(self):
            self.status = AdapterStatus.READY

        async def shutdown(self):
            pass

        async def handle_request(self, request):
            return AdapterResponse(
                request_id="r", capability=request.capability, status="error",
                error="Gemini HTTP 503: high demand", error_code="HTTP_503", duration_ms=0.0,
            )

    async def no_creds(*args, **kwargs):
        return {}

    monkeypatch.setattr(base_node.adapter_registry, "get_adapter_class", lambda _id: _BusyAdapter)
    import nodes.template_utils as template_utils
    monkeypatch.setattr(template_utils, "resolve_adapter_credentials", no_creds)

    with pytest.raises(AdapterCallError) as exc:
        await _node().call_adapter(adapter_id="gemini", capability="answer", parameters={"query": "q"})
    assert isinstance(exc.value, RuntimeError)
    assert str(exc.value) == "Adapter error: Gemini HTTP 503: high demand"
    assert exc.value.error_code == "HTTP_503"


async def test_a_retried_call_reuses_the_adapters_registration(monkeypatch, caplog):
    # Each attempt goes back through call_adapter, which builds and starts a new
    # adapter. The retry must reuse the first attempt's control-plane
    # registration, not add a second one or log "already registered".
    import logging
    import uuid

    from adapters.base_adapter import BaseAdapter
    from adapters.models import AdapterRequest, AdapterResponse
    from control_plane.registry import component_registry

    name = f"busy-{uuid.uuid4().hex[:8]}"
    answers = iter([
        AdapterResponse(request_id="r", capability="answer", status="error",
                        error="Gemini HTTP 503: high demand", error_code="HTTP_503", duration_ms=0.0),
        AdapterResponse(request_id="r", capability="answer", status="success",
                        data={"content": "answer"}, duration_ms=0.0),
    ])

    class _Busy(BaseAdapter):
        def __init__(self, config):
            config.name = name
            super().__init__(config)

        async def initialize(self): pass
        async def shutdown(self): pass
        def get_capabilities(self): return []

        async def execute(self, request: AdapterRequest) -> AdapterResponse:
            return next(answers)

    async def no_creds(*args, **kwargs):
        return {}

    monkeypatch.setattr(base_node.adapter_registry, "get_adapter_class", lambda _id: _Busy)
    import nodes.template_utils as template_utils
    monkeypatch.setattr(template_utils, "resolve_adapter_credentials", no_creds)

    caplog.set_level(logging.INFO)
    out = await _node(retry_attempts=2, retry_backoff_seconds=15).execute({}, {})

    assert out["content"] == "answer"
    assert out["_attempts"] == 2
    assert len([c for c in component_registry.components.values() if c.name == name]) == 1
    assert [r.getMessage() for r in caplog.records
            if r.levelno >= logging.ERROR and "already registered" in r.getMessage()] == []
