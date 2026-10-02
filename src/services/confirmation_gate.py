"""The in-chat confirmation gate (spec §7.3 R4, T4b).

A tool call the user's autonomy level says must be confirmed
(`confirmation_policy.needs_confirmation`) does not run in the turn that asked
for it. It becomes ONE pending proposal on the session; the next turn runs it
only on consent (router `confirm`), drops it on a "no" (router `cancel`), and
any other turn dismisses it. Both turn loops (Community, Business) call into
this module; plan: .claude/plans/conversation-reliability-t4b-confirmation.md.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from llm.models import ToolResult
from services.confirmation_policy import needs_confirmation

logger = logging.getLogger(__name__)

PROPOSAL_KEY = "pending_proposal"
PROPOSAL_TTL_S = 15 * 60
NEEDS_CONFIRMATION = "needs_confirmation"
CANCELLED_ANSWER = "Cancelled — nothing was run."


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def resolve_turn_level(db, user_id: str, tenant_id: Optional[str]) -> int:
    """The user's effective autonomy level for this turn (AI Control Spectrum).

    Business/Enterprise walk the full cascade; Community has workflow → user →
    system default. Any failure falls back to the system default, never to
    "confirm nothing".
    """
    from services.autonomy_taxonomy import SYSTEM_DEFAULT_LEVEL

    try:
        try:
            from aictrlnet_business.services.autonomy_resolver import get_resolver
            resolver = get_resolver(db)
        except ImportError:
            from services.autonomy_resolver import CommunityAutonomyResolver
            resolver = CommunityAutonomyResolver(db)
        resolved = await resolver.resolve(tenant_id=tenant_id, user_id=str(user_id))
        return int(resolved.level)
    except Exception as e:
        logger.warning(f"[confirmation] autonomy level unavailable, using the system default: {e}")
        return SYSTEM_DEFAULT_LEVEL


def missing_required(tool, arguments: Optional[Dict[str, Any]]) -> List[str]:
    """Required parameters the call leaves empty (review finding 12): asking for
    them after "yes" would run a different call than the one confirmed."""
    params = getattr(tool, "parameters", None) or {}
    required = params.get("required") or []
    arguments = arguments or {}
    return [name for name in required if arguments.get(name) in (None, "", [], {})]


def check(tool, arguments: Optional[Dict[str, Any]], level: int) -> Optional[ToolResult]:
    """None when the call may run now; otherwise the result the loop records
    instead of running it: a missing-arguments error, or a proposal."""
    if tool is None or not needs_confirmation(tool, level):
        return None
    missing = missing_required(tool, arguments)
    if missing:
        return ToolResult(
            success=False,
            error=f"{tool.name} needs {', '.join(missing)} before it can be proposed; ask the user for it.",
            error_type="validation",
            data={"tool_name": tool.name, "missing": missing},
        )
    return ToolResult(success=False, error_type=NEEDS_CONFIRMATION,
                      data={"tool_name": tool.name, "arguments": dict(arguments or {})})


def make_proposal(tool, arguments: Dict[str, Any], *, user_id: str, tenant_id: Optional[str],
                  user_request: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    proposal = {
        "id": str(uuid.uuid4()),
        "tool": tool.name,
        "arguments": dict(arguments or {}),
        "destructive": bool(getattr(tool, "is_destructive", False)),
        "user_id": str(user_id),
        "tenant_id": tenant_id,
        "user_request": user_request,
        "created_at": _now().isoformat(),
        "status": "pending",
    }
    # Only the turn-specific keys that turn actually built (review finding 9).
    for key in ("templates", "user_query"):
        if context and context.get(key):
            proposal[key] = context[key]
    return proposal


def _plain(name: str) -> str:
    return name.replace("_", " ")


def _describe_arguments(arguments: Dict[str, Any]) -> str:
    shown = [(k, v) for k, v in arguments.items() if v not in (None, "", [], {})][:5]
    if not shown:
        return ""
    parts = []
    for key, value in shown:
        text = value if isinstance(value, str) else str(value)
        parts.append(f"{_plain(key)} **{text[:80]}**")
    return " with " + ", ".join(parts)


def proposal_answer(proposal: Dict[str, Any], ran: Optional[List[ToolResult]] = None) -> str:
    """Deterministic answer for a proposal turn (no synthesis round: a model told
    "awaiting confirmation" tends to claim it was done)."""
    lead = ""
    done = [r for r in (ran or []) if r.data and r.data.get("tool_name")]
    if done:
        lead = "Done so far: " + ", ".join(_plain(r.data["tool_name"]) for r in done) + ".\n\n"
    verb = _plain(proposal["tool"])
    return (f"{lead}I can {verb}{_describe_arguments(proposal.get('arguments') or {})}. "
            f"Reply **yes** to go ahead, or tell me what to change.")


def proposal_block(proposal: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "execution_preview",
        "data": {
            "status": "awaiting_confirmation",
            "proposal_id": proposal["id"],
            "tools": [{"name": proposal["tool"], "status": "pending",
                       "arguments": proposal.get("arguments") or {}}],
        },
        "actions": [
            {"label": "Confirm", "verb": "run_now", "primary": True,
             "destructive": bool(proposal.get("destructive")), "target": None},
            {"label": "Cancel", "verb": "cancel", "primary": False, "destructive": False, "target": None},
        ],
        "entity_ref": None,
    }


async def claim(db, session_id, *, user_id: str, tenant_id: Optional[str], turn_id: str,
                offered: set, proposal_id: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """Take the pending proposal for this confirm turn, or say why not.

    Compare-and-set under SELECT … FOR UPDATE, committed before the tool runs
    (review finding 3): a second "yes" racing this one finds `executing:…`.
    An invalid proposal is deleted.
    """
    from sqlalchemy import select
    from sqlalchemy.orm.attributes import flag_modified
    from models.conversation import ConversationSession

    # populate_existing: the turn's session already holds this row from turn start;
    # without it the locked read returns that stale copy and a racing second
    # "yes" still sees "pending" (review 2 Oct).
    row = (await db.execute(
        select(ConversationSession).where(ConversationSession.id == session_id)
        .with_for_update().execution_options(populate_existing=True)
    )).scalar_one_or_none()
    if row is None:
        await db.rollback()
        return None, "missing"
    context = dict(row.context or {})
    proposal = context.get(PROPOSAL_KEY)
    reason = None
    if not isinstance(proposal, dict) or proposal.get("status") != "pending":
        reason = "missing"
    elif proposal_id and proposal_id != proposal.get("id"):
        reason = "mismatch"
    elif str(proposal.get("user_id")) != str(user_id):
        reason = "other_user"
    elif (proposal.get("tenant_id") or None) != (tenant_id or None):
        reason = "other_tenant"
    elif proposal.get("tool") not in offered:
        reason = "not_offered"
    else:
        try:
            created = datetime.fromisoformat(proposal["created_at"])
            if (_now() - created).total_seconds() > PROPOSAL_TTL_S:
                reason = "expired"
        except (KeyError, ValueError):
            reason = "expired"
    if reason:
        # Keep a proposal another turn is running, and the current proposal when
        # a stale button named an older one; drop anything invalid.
        keep = reason in ("missing", "mismatch")
        if not keep and PROPOSAL_KEY in context:
            context.pop(PROPOSAL_KEY, None)
            row.context = context
            flag_modified(row, "context")
        await db.commit()
        return None, reason
    proposal = {**proposal, "status": f"executing:{turn_id}"}
    context[PROPOSAL_KEY] = proposal
    row.context = context
    flag_modified(row, "context")
    await db.commit()
    return proposal, "ok"


INVALID_ANSWERS = {
    "missing": "There's nothing waiting for a yes right now — tell me what you'd like me to do.",
    "mismatch": "That confirmation was for an older proposal, so nothing was run.",
    "other_user": "That proposal belongs to someone else, so nothing was run.",
    "other_tenant": "That proposal belongs to another workspace, so nothing was run.",
    "not_offered": "That action isn't available here any more, so nothing was run.",
    "expired": "That proposal expired (after 15 minutes), so nothing was run. Ask again and I'll set it up.",
}


def is_pending(session) -> bool:
    proposal = (getattr(session, "context", None) or {}).get(PROPOSAL_KEY)
    return isinstance(proposal, dict) and proposal.get("status") == "pending"


async def clear(session_id, only_id: Optional[str] = None) -> None:
    """Delete the proposal on a fresh session (after a confirm turn, whatever happened)."""
    from sqlalchemy import select
    from sqlalchemy.orm.attributes import flag_modified
    from core.database import get_session_maker
    from models.conversation import ConversationSession

    try:
        async with get_session_maker()() as db:
            row = (await db.execute(
                select(ConversationSession).where(ConversationSession.id == session_id)
                .with_for_update().execution_options(populate_existing=True)
            )).scalar_one_or_none()
            if row is None:
                return
            context = dict(row.context or {})
            proposal = context.get(PROPOSAL_KEY)
            if proposal is None or (only_id and proposal.get("id") != only_id):
                return
            context.pop(PROPOSAL_KEY, None)
            row.context = context
            flag_modified(row, "context")
            await db.commit()
    except Exception as e:
        logger.warning(f"[confirmation] could not clear the proposal on session {session_id}: {e}")


def persist(live_session, proposal: Optional[Dict[str, Any]], *, keep_existing: bool = False) -> None:
    """Write (or delete) the proposal on the live session row inside the turn's
    commit. `keep_existing` leaves a proposal another path owns untouched."""
    from sqlalchemy.orm.attributes import flag_modified

    context = dict(live_session.context or {})
    if proposal is not None:
        context[PROPOSAL_KEY] = proposal
    elif keep_existing or PROPOSAL_KEY not in context:
        return
    else:
        context.pop(PROPOSAL_KEY, None)
    live_session.context = context
    flag_modified(live_session, "context")
