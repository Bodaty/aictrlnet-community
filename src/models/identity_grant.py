"""The server-stamped grant of a workflow's protected identity.

A workflow definition is recognised as a PHI or document workflow only when it
carries ``identityGrant``: who granted that identity and when. The grant is
written by ``services.workflow_identity.grant_identity`` when an administrator
creates, duplicates, instantiates or updates such a workflow, and by nothing
else.

This module is the one place that enforces it, at the model layer, for every
ORM write of ``workflow_definitions``:

- On insert, a grant is kept only if ``grant_identity`` issued it in this
  process and it hasn't been used (a one-time nonce). Anything else a client
  put under ``identityGrant`` is removed.
- On update, the stored grant is kept exactly as it is, unless a fresh grant
  was issued. A client can't add, change or erase one.

A route that forgets its admin gate can therefore create only an inert row.
"""

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

GRANT_KEY = "identityGrant"

logger = logging.getLogger(__name__)

# Issued, unspent nonces -> when issued (monotonic). A stamp whose request
# fails before its flush is never spent, so the set is bounded: a nonce lapses
# after NONCE_TTL_SECONDS (a stamp is flushed within its request), and beyond
# MAX_UNSPENT the oldest lapse first. Within those bounds each is single-use.
NONCE_TTL_SECONDS = 3600.0
MAX_UNSPENT = 10000
_issued: "OrderedDict[str, float]" = OrderedDict()
_issued_lock = threading.Lock()


def _evict(now: float) -> None:
    """Drop lapsed nonces, then the oldest past MAX_UNSPENT (hold the lock)."""
    while _issued:
        oldest, issued_at = next(iter(_issued.items()))
        lapsed = now - issued_at >= NONCE_TTL_SECONDS
        if not lapsed and len(_issued) <= MAX_UNSPENT:
            break
        del _issued[oldest]
        if not lapsed:
            # An administrator's grant still inside its TTL is dropped unspent:
            # its write, if it comes, stores no grant (fail closed). Say so.
            logger.warning("identity grant nonce cap (%d) reached: an unspent grant "
                           "issued %.0f s ago was dropped", MAX_UNSPENT, now - issued_at)


def stamp(definition: Dict[str, Any], granted_by: str, fields: List[str]) -> Dict[str, Any]:
    """``definition`` with a fresh grant (call through services.workflow_identity)."""
    nonce = uuid.uuid4().hex
    with _issued_lock:
        now = time.monotonic()
        _issued[nonce] = now
        _evict(now)
    stamped = dict(definition)
    stamped[GRANT_KEY] = {"by": str(granted_by), "at": datetime.now(timezone.utc).isoformat(),
                          "fields": list(fields), "nonce": nonce}
    return stamped


def is_granted(definition: Any) -> bool:
    """A stored definition carries a grant (the model layer let it through)."""
    definition = _as_dict(definition)
    grant = definition.get(GRANT_KEY) if definition else None
    return isinstance(grant, dict) and bool(grant.get("by")) and "nonce" not in grant


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _take_fresh(grant: Any) -> Optional[Dict[str, Any]]:
    """The grant without its nonce if ``grant_identity`` issued it and it's unused."""
    if not isinstance(grant, dict):
        return None
    nonce = grant.get("nonce")
    with _issued_lock:
        _evict(time.monotonic())
        if not isinstance(nonce, str) or _issued.pop(nonce, None) is None:
            return None
    return {k: v for k, v in grant.items() if k != "nonce"}


def _stored_grant(connection, workflow_id) -> Optional[Dict[str, Any]]:
    from sqlalchemy import text

    value = connection.execute(text(
        "SELECT definition::jsonb -> 'identityGrant' FROM workflow_definitions WHERE id = :id"
    ), {"id": str(workflow_id)}).scalar()
    value = _as_dict(value) if value is not None else None
    return value


def _guard(connection, target, is_update: bool) -> None:
    definition = _as_dict(target.definition)
    if definition is None:
        return
    if is_update:
        from sqlalchemy import inspect

        if not inspect(target).attrs.definition.history.has_changes():
            return
    incoming = definition.get(GRANT_KEY, None)
    fresh = _take_fresh(incoming)
    if fresh is not None:
        wanted = fresh
    elif is_update:
        wanted = _stored_grant(connection, target.id)
    else:
        wanted = None
    if wanted == incoming and (wanted is not None or GRANT_KEY not in definition):
        return
    guarded = dict(definition)
    if wanted is None:
        guarded.pop(GRANT_KEY, None)
    else:
        guarded[GRANT_KEY] = wanted
    target.definition = guarded


def before_insert(mapper, connection, target) -> None:
    _guard(connection, target, is_update=False)


def before_update(mapper, connection, target) -> None:
    _guard(connection, target, is_update=True)
