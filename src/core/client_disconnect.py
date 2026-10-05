"""Treat a client that left before the response as a client event, not a server error.

Starlette's BaseHTTPMiddleware raises RuntimeError("No response returned.") from
call_next when the downstream app returns without sending, which happens when
the client disconnected. Logged as an ERROR traceback, it buried real errors:
every Cloud Run cold start produced several (startup probes that timed out while
queued), and so did any user closing a tab mid-request.
"""
from starlette.responses import Response

CLIENT_CLOSED_REQUEST = 499


async def call_next_unless_disconnected(request, call_next):
    """Run call_next; if it failed only because the client went away, return 499."""
    try:
        return await call_next(request)
    except RuntimeError as exc:
        if str(exc) == "No response returned." and await request.is_disconnected():
            return Response(status_code=CLIENT_CLOSED_REQUEST)
        raise
