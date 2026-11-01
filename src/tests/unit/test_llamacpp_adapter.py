"""G12: the llama.cpp adapter reads a page image with a local llama-server.

Every payload assertion goes over real HTTP to a server on 127.0.0.1 in this
process, so what is asserted is what llama-server would receive. No real
document and no real model endpoint are involved.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from adapters.models import AdapterCategory, AdapterConfig, AdapterRequest
from core import phi_egress
from core.phi_egress import PHI_REFUSAL_MARKER, PHIEgressRefused, phi_boot_problems


PAGE_B64 = "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZ0FBQUFF"
MARKDOWN = "# Referral\n\nPatient: Test Person"


class CannedLlamaServer:
    """Answers llama-server's /v1/chat/completions and /health, recording requests."""

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
                if self.path == "/health":
                    return self._reply({"status": "ok"})
                return self._reply({"error": "not found"}, status=404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append(("POST", self.path, body))
                if self.path == "/v1/chat/completions":
                    return self._reply({
                        "id": "chatcmpl-1",
                        "model": "HunyuanOCR-1.5",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": MARKDOWN}}],
                        "usage": {"prompt_tokens": 812, "completion_tokens": 14, "total_tokens": 826},
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
    with CannedLlamaServer() as srv:
        yield srv


@pytest.fixture
def phi_mode(monkeypatch):
    monkeypatch.setenv("AICTRLNET_PHI_MODE", "true")
    monkeypatch.setenv("AICTRLNET_PHI_LLM_PROVIDERS", "llamacpp")
    monkeypatch.delenv("AICTRLNET_PHI_BAA_PROVIDERS", raising=False)
    phi_egress.reset_endpoint_cache()
    yield
    phi_egress.reset_endpoint_cache()


async def _llamacpp(base_url):
    from adapters.implementations.ai.llamacpp_adapter import LlamaCppAdapter

    adapter = LlamaCppAdapter(AdapterConfig(
        name="llamacpp", version="1.0.0", category=AdapterCategory.AI, base_url=base_url,
    ))
    await adapter.initialize()
    return adapter


@pytest.mark.asyncio
async def test_ocr_page_sends_image_part_and_returns_text(phi_mode, server):
    adapter = await _llamacpp(server.url)
    try:
        response = await adapter.execute(AdapterRequest(
            capability="ocr_page",
            parameters={"image_b64": PAGE_B64},
            phi_pipeline=True,
            temperature=0,
        ))
        assert response.status == "success", response.error
        assert response.data["text"] == MARKDOWN
        assert response.data["model"] == "HunyuanOCR-1.5"
        assert response.data["usage"]["total_tokens"] == 826

        [body] = server.posted("/v1/chat/completions")
        assert body["temperature"] == 0
        assert body["max_tokens"] == 4096
        [message] = body["messages"]
        assert message["role"] == "user"
        image_part, text_part = message["content"]
        assert image_part == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{PAGE_B64}"},
        }
        assert text_part == {"type": "text", "text": "Parse this document into Markdown."}
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_chat_passes_messages_temperature_and_schema(phi_mode, server):
    schema = {"type": "object", "properties": {"is_referral": {"type": "boolean"}}}
    adapter = await _llamacpp(server.url)
    try:
        response = await adapter.execute(AdapterRequest(
            capability="chat",
            parameters={"messages": [{"role": "user", "content": "Is this a referral?"}]},
            phi_pipeline=True,
            temperature=0,
            response_format=schema,
        ))
        assert response.status == "success", response.error
        assert response.data["text"] == MARKDOWN

        [body] = server.posted("/v1/chat/completions")
        assert body["messages"] == [{"role": "user", "content": "Is this a referral?"}]
        assert body["temperature"] == 0
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_refuses_public_endpoint_under_phi_mode(phi_mode, monkeypatch):
    from adapters.implementations.ai.llamacpp_adapter import LlamaCppAdapter

    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.0.1", 0))],
    )
    with pytest.raises(PHIEgressRefused) as excinfo:
        LlamaCppAdapter(AdapterConfig(
            name="llamacpp", category=AdapterCategory.AI, base_url="http://ocr.example.com:8089",
        ))
    message = str(excinfo.value)
    assert message.startswith(PHI_REFUSAL_MARKER)
    assert "llamacpp" in message and "ocr.example.com" in message


@pytest.mark.asyncio
async def test_rechecks_endpoint_per_request(phi_mode, server, monkeypatch):
    adapter = await _llamacpp(server.url)
    try:
        monkeypatch.setattr(
            socket, "getaddrinfo",
            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.0.1", 0))],
        )
        adapter.base_url = "http://ocr.example.com:8089"
        with pytest.raises(PHIEgressRefused):
            await adapter.execute(AdapterRequest(
                capability="ocr_page", parameters={"image_b64": PAGE_B64}, temperature=0,
            ))
        assert not server.posted("/v1/chat/completions")
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_phi_pipeline_without_temperature_refused(phi_mode, server):
    adapter = await _llamacpp(server.url)
    try:
        with pytest.raises(PHIEgressRefused) as excinfo:
            await adapter.execute(AdapterRequest(
                capability="ocr_page", parameters={"image_b64": PAGE_B64}, phi_pipeline=True,
            ))
        assert "temperature" in str(excinfo.value)
        assert server.requests == []
    finally:
        await adapter.shutdown()


def test_boot_problems_reports_unset_llamacpp_url_when_allowed():
    settings = SimpleNamespace(
        AICTRLNET_PHI_LLM_PROVIDERS="llamacpp",
        AICTRLNET_PHI_BAA_PROVIDERS="",
        DEFAULT_LLM_MODEL=None,
        LLAMACPP_URL=None,
    )
    problems = phi_boot_problems(settings)
    assert any("LLAMACPP_URL" in p and "llamacpp" in p for p in problems), problems

    settings.LLAMACPP_URL = "http://host.docker.internal:8089"
    assert phi_boot_problems(settings) == []


def test_boot_problems_reports_public_llamacpp_url(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.0.1", 0))],
    )
    phi_egress.reset_endpoint_cache()
    settings = SimpleNamespace(
        AICTRLNET_PHI_LLM_PROVIDERS="llamacpp",
        AICTRLNET_PHI_BAA_PROVIDERS="",
        DEFAULT_LLM_MODEL=None,
        LLAMACPP_URL="https://ocr.example.com",
    )
    problems = phi_boot_problems(settings)
    phi_egress.reset_endpoint_cache()
    assert any("LLAMACPP_URL" in p and "non-local" in p for p in problems), problems


@pytest.mark.asyncio
async def test_health_check_reports_down_when_unreachable():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()

    adapter = await _llamacpp(f"http://127.0.0.1:{closed_port}")
    try:
        response = await adapter.execute(AdapterRequest(capability="health_check", parameters={}))
        assert response.status == "success", response.error
        assert response.data["healthy"] is False
        assert response.data["detail"]
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_health_check_reports_up(server):
    adapter = await _llamacpp(server.url)
    try:
        response = await adapter.execute(AdapterRequest(capability="health_check", parameters={}))
        assert response.status == "success", response.error
        assert response.data["healthy"] is True
        assert ("GET", "/health", None) in server.requests
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_unconfigured_refuses_every_capability(monkeypatch):
    monkeypatch.delenv("LLAMACPP_URL", raising=False)
    adapter = await _llamacpp(None)
    try:
        for capability, parameters in (
            ("ocr_page", {"image_b64": PAGE_B64}),
            ("chat", {"messages": [{"role": "user", "content": "hi"}]}),
            ("health_check", {}),
        ):
            response = await adapter.execute(AdapterRequest(capability=capability, parameters=parameters))
            assert response.status == "error", capability
            assert "LLAMACPP_URL" in response.error, capability
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", [
    "http://images.example.com/page.png",
    "https://images.example.com/page.png",
    {"url": "https://images.example.com/page.png"},
    "file:///etc/passwd",
])
async def test_chat_refuses_non_inline_image_before_any_http(server, monkeypatch, image_url):
    monkeypatch.delenv("AICTRLNET_PHI_MODE", raising=False)
    adapter = await _llamacpp(server.url)
    try:
        response = await adapter.execute(AdapterRequest(
            capability="chat",
            parameters={"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": image_url},
                {"type": "text", "text": "Read this page."},
            ]}]},
            temperature=0,
        ))
        assert response.status == "error"
        assert "data:" in response.error
        assert response.error_code == "REMOTE_IMAGE_REFUSED"
        assert server.requests == []
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_chat_accepts_inline_data_uri_image(phi_mode, server):
    part = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{PAGE_B64}"}}
    adapter = await _llamacpp(server.url)
    try:
        response = await adapter.execute(AdapterRequest(
            capability="chat",
            parameters={"messages": [{"role": "user", "content": [part, {"type": "text", "text": "Read."}]}]},
            temperature=0,
        ))
        assert response.status == "success", response.error
        [body] = server.posted("/v1/chat/completions")
        assert body["messages"][0]["content"][0] == part
    finally:
        await adapter.shutdown()
