"""A client that disconnects before the response makes Starlette's
BaseHTTPMiddleware raise RuntimeError("No response returned."). That is not a
server error: it was logged as an ERROR traceback on every Cloud Run cold start
(startup probes that timed out while queued) and whenever a user closed a tab.
"""
import pytest

from core.client_disconnect import call_next_unless_disconnected


class _Request:
    def __init__(self, disconnected):
        self._disconnected = disconnected

    async def is_disconnected(self):
        return self._disconnected


async def _no_response(request):
    raise RuntimeError("No response returned.")


@pytest.mark.asyncio
async def test_disconnected_client_gets_a_quiet_499():
    response = await call_next_unless_disconnected(_Request(True), _no_response)
    assert response.status_code == 499


@pytest.mark.asyncio
async def test_connected_client_still_raises():
    with pytest.raises(RuntimeError):
        await call_next_unless_disconnected(_Request(False), _no_response)


@pytest.mark.asyncio
async def test_other_errors_still_raise():
    async def _boom(request):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await call_next_unless_disconnected(_Request(True), _boom)


@pytest.mark.asyncio
async def test_normal_response_passes_through():
    sentinel = object()

    async def _ok(request):
        return sentinel

    assert await call_next_unless_disconnected(_Request(False), _ok) is sentinel
