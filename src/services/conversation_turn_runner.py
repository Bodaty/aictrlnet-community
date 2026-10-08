"""Run one v5 turn without streaming (spec §7 "Turn paths": non-streaming collector).

Every entry point that is not the SSE endpoint — the `/messages` endpoints, MCP
`send_message`, the channel webhook — runs its turn here, so each gets the
registered edition's `process_message_v2` with its budgets, terminal-event
contract and confirmation gate. Raises `TurnFailed` instead of an HTTP error so
non-HTTP callers can use it.
"""

from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import select

from models.conversation import ConversationMessage, ConversationSession


class TurnFailed(Exception):
    """The turn ended in an `error` event, or produced no response. `code` is the
    error event's code when it has one ("session_not_found")."""

    def __init__(self, message: str, turn: Optional[Dict[str, Any]] = None, code: Optional[str] = None):
        super().__init__(message)
        self.turn = turn
        self.code = code

    @property
    def http_status(self) -> int:
        return 404 if self.code == "session_not_found" else 500


async def run_turn(
    service,
    session_id: UUID,
    content: str,
    user_id: str,
    user_preferences: Optional[dict] = None,
    channel_type: str = "web",
    external_message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The turn's `response` data, with the `complete` event's data under `turn`."""
    response_data: Dict[str, Any] = {}
    turn = None
    async for event in service.process_message_v2(
        session_id, content, user_id, stream=False, user_preferences=user_preferences,
        channel_type=channel_type, external_message_id=external_message_id,
    ):
        kind = event.get("event")
        if kind == "response":
            response_data = event.get("data") or {}
        elif kind == "complete":
            turn = event.get("data")
        elif kind == "error":
            data = event.get("data") or {}
            raise TurnFailed(data.get("message", "Error"), turn=data, code=data.get("code"))
    if not response_data:
        raise TurnFailed("No response from conversation service", turn=turn)
    return {**response_data, "turn": turn}


async def collect_response(
    service,
    db,
    session_id: UUID,
    content: str,
    user_id: str,
    user_preferences: Optional[dict] = None,
) -> Dict[str, Any]:
    """A ConversationResponse-shaped dict for the `/messages` endpoints."""
    response_data = await run_turn(service, session_id, content, user_id, user_preferences)

    message_id = UUID(response_data["message_id"])
    assistant_msg = (await db.execute(
        select(ConversationMessage).where(ConversationMessage.id == message_id)
    )).scalar_one_or_none()
    session = (await db.execute(
        select(ConversationSession).where(ConversationSession.id == session_id)
    )).scalar_one_or_none()
    if not assistant_msg or not session:
        raise TurnFailed("Failed to retrieve persisted message or session", turn=response_data["turn"])

    return {
        "session_id": session_id,
        "message": assistant_msg,
        "state": session.state,
        "context": response_data.get("session_context", {}),
        "quick_actions": [],
        "automation_result": response_data.get("automation_result") or response_data.get("created_workflow"),
        "turn": response_data["turn"],
    }
