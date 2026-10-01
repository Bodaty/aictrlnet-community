"""Turn an unhandled exception into a 500 that still passes through CORS.

Starlette's ServerErrorMiddleware sits outside every user middleware, so the
500 it renders for an escaped exception never reaches CORSMiddleware. The
browser then sees a CORS failure, axios reports "no response", and the SPA
shows "Can't reach the server" for one broken endpoint. Registered directly
inside CORSMiddleware, this boundary answers the request itself.
"""

import logging

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class ErrorBoundaryMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "Unhandled error method=%s path=%s", scope.get("method"), scope.get("path")
            )
            if response_started:
                # Headers already went out (e.g. a stream failed midway);
                # nothing valid can be sent now.
                raise
            response = JSONResponse({"detail": "Internal server error"}, status_code=500)
            await response(scope, receive, send)
