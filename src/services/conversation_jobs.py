"""Long-running conversation tools as background jobs (CONVERSATION_ORCHESTRATION_SPEC.md §7.3 R2).

`create_workflow`, `automate_company` and `generate_adapter` take 70–300 s. With
`CONVERSATION_JOBS=on` a turn starts them here and answers at once; the job runs
on its own tenant-bound session and reports through a `job_card`.

Rules that keep it honest:
- the job never writes the conversation session — the next turn claims a
  finished job (`claim_finished`) and records its outcome in its own transaction;
- liveness is a lease the running task renews by compare-and-set on its owner
  token; a row whose lease lapsed reads as `stale`, and the same owner may still
  finish it (a slow job is not a dead one);
- one running job per tool per session; at most `CONVERSATION_JOB_CONCURRENCY`
  per worker; the work is bounded by `CONVERSATION_JOB_MAX_S`.

Modes (`CONVERSATION_JOBS`): `off` (default: the tool runs inside the turn),
`on` (an in-process task; needs CPU between requests — local, Beast), and
`cloud_tasks` (Cloud Run, which throttles CPU between requests: each job is a
Cloud Task that calls back into the service, so the job runs inside a request
with CPU allocated — see `run_dispatched` and the 1 Oct ruling in
`.claude/plans/conversation-reliability-t5-jobs.md`).
"""

import asyncio
import logging
import os
import uuid
import weakref
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, text, update

logger = logging.getLogger(__name__)

TERMINAL = ("succeeded", "failed", "needs_input")
_tasks: set = set()
_owned: Dict[str, Tuple[str, str]] = {}  # job_id -> (owner_token, tenant_id), this worker only
_limits: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()

_LABELS = {
    "create_workflow": "building your workflow",
    "automate_company": "setting up your company automation",
    "generate_adapter": "generating the integration",
}


CLOUD_TASKS = "cloud_tasks"
# Only these dispatchers may be named in a Cloud Task payload.
_DISPATCHERS = {
    "services.tool_dispatcher.ToolDispatcher",
    "aictrlnet_business.services.tool_dispatcher.ToolDispatcher",
}


def mode() -> str:
    value = os.environ.get("CONVERSATION_JOBS", "off").strip().lower()
    return value if value in ("on", CLOUD_TASKS) else "off"


def enabled() -> bool:
    return mode() != "off"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def lease_s() -> float:
    return _env_float("CONVERSATION_JOB_LEASE_S", 15)


def heartbeat_s() -> float:
    return _env_float("CONVERSATION_JOB_HEARTBEAT_S", 5)


def max_s() -> float:
    return _env_float("CONVERSATION_JOB_MAX_S", 900)


def _limit() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _limits.get(loop)
    if sem is None:
        sem = _limits[loop] = asyncio.Semaphore(int(_env_float("CONVERSATION_JOB_CONCURRENCY", 2)))
    return sem


def outcome(tool_name: str, success: bool, data: Optional[Dict[str, Any]], error: Optional[str]) -> Dict[str, Any]:
    """What a finished tool run means for the user: status, summary, link, entity."""
    data = data or {}
    if not success:
        return {"status": "failed", "summary": error or data.get("message") or "It did not complete."}
    if data.get("needs_clarification") or data.get("industry_clarification") or data.get("status") == "needs_clarification":
        return {"status": "needs_input",
                "summary": data.get("message") or data.get("question") or "I need a bit more information."}
    if str(data.get("status", "")).lower() in ("failed", "error"):
        return {"status": "failed", "summary": data.get("error") or data.get("message") or "It did not complete."}
    result: Dict[str, Any] = {"status": "succeeded", "summary": data.get("message") or "Done."}
    if data.get("workflow_id"):
        name = data.get("workflow_name") or "Workflow"
        result.update(summary=f"Created workflow '{name}'.", target=f"/workflows/{data['workflow_id']}/edit",
                      entity={"type": "workflow", "id": str(data["workflow_id"]), "label": name})
    elif tool_name == "automate_company":
        result.update(target="/ai-company", summary=data.get("message") or "Your company automation is ready.")
    return result


async def _bound_session(tenant_id: str):
    from core.database import bind_session_tenant, get_session_maker

    session = get_session_maker()()
    bind_session_tenant(session, tenant_id)
    return session


async def start(dispatcher, tool_name: str, arguments: Dict[str, Any], user_id: str,
                context: Dict[str, Any], session_id: Any):
    """Record a running job and start it; returns the ToolResult the turn reports."""
    from core.tenant_context import get_current_tenant_id
    from llm.models import ToolResult
    from models.conversation import ConversationJob

    tenant_id = (getattr(dispatcher.db, "info", None) or {}).get("tenant_id") or get_current_tenant_id()
    now = datetime.utcnow()
    owner = uuid.uuid4().hex
    # A queued Cloud Task may take a little while to start: its first lease
    # covers the wait, then the running job's heartbeat takes over.
    first_lease = lease_s() + (_env_float("CONVERSATION_JOB_QUEUE_GRACE_S", 120) if mode() == CLOUD_TASKS else 0)
    db = await _bound_session(tenant_id)
    try:
        # Serialise starts of this tool in this session (double-submit, two tabs).
        await db.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
                         {"key": f"conversation-job:{session_id}:{tool_name}"})
        # A job counts as still going while it is running — or stale but young
        # enough that its owner may yet finish it (a slow job is not a dead one).
        running = (await db.execute(select(ConversationJob.id).where(
            ConversationJob.session_id == session_id, ConversationJob.tool_name == tool_name,
            ConversationJob.status.in_(("running", "stale")),
            ConversationJob.created_at > now - timedelta(seconds=max_s()),
        ))).scalars().first()
        if running:
            return ToolResult(success=True, data={
                "job_id": str(running), "status": "running", "tool_name": tool_name,
                "message": f"That is already running ({_LABELS.get(tool_name, tool_name)}).",
            })
        job = ConversationJob(
            tenant_id=tenant_id, session_id=session_id, user_id=str(user_id), tool_name=tool_name,
            arguments=arguments or {}, status="running", progress=_LABELS.get(tool_name, "working"),
            owner_token=owner, lease_expires_at=now + timedelta(seconds=first_lease),
        )
        db.add(job)
        await db.commit()
        job_id = str(job.id)
    finally:
        await db.close()

    if mode() == CLOUD_TASKS:
        from fastapi.encoders import jsonable_encoder

        cls = type(dispatcher)
        payload = jsonable_encoder({
            "job_id": job_id, "owner": owner, "tenant_id": tenant_id, "tool_name": tool_name,
            "arguments": arguments or {}, "user_id": str(user_id), "context": dict(context or {}),
            "dispatcher": f"{cls.__module__}.{cls.__name__}",
            "edition": getattr(dispatcher.edition, "value", dispatcher.edition),
        })
        try:
            await enqueue_cloud_task(payload)
        except Exception as exc:
            logger.error("[jobs] could not queue %s: %s", tool_name, exc)
            await _cas(tenant_id, job_id, owner, status="failed", error=str(exc),
                       progress="It could not be started — please try again.", finished_at=datetime.utcnow())
            return ToolResult(success=False, error="It could not be started — please try again.",
                              data={"job_id": job_id, "tool_name": tool_name})
        return ToolResult(success=True, data={
            "job_id": job_id, "status": "running", "tool_name": tool_name,
            "message": f"Started {_LABELS.get(tool_name, tool_name)} in the background.",
        })

    _owned[job_id] = (owner, tenant_id)
    task = asyncio.get_running_loop().create_task(
        _run(type(dispatcher), dispatcher.edition, tenant_id, job_id, owner, tool_name,
             arguments or {}, str(user_id), dict(context or {})),
        name=f"conversation-job:{tool_name}",
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return ToolResult(success=True, data={
        "job_id": job_id, "status": "running", "tool_name": tool_name,
        "message": f"Started {_LABELS.get(tool_name, tool_name)} in the background.",
    })


async def _cas(tenant_id: str, job_id: str, owner: str, **values) -> bool:
    """Update the job only while this owner still holds it (running, or lapsed to stale)."""
    from models.conversation import ConversationJob

    db = await _bound_session(tenant_id)
    try:
        result = await db.execute(update(ConversationJob).where(
            ConversationJob.id == job_id, ConversationJob.owner_token == owner,
            ConversationJob.status.in_(("running", "stale")),
        ).values(updated_at=datetime.utcnow(), **values))
        await db.commit()
        return result.rowcount == 1
    finally:
        await db.close()


async def _heartbeat(tenant_id: str, job_id: str, owner: str) -> None:
    while True:
        await asyncio.sleep(heartbeat_s())
        try:
            held = await _cas(tenant_id, job_id, owner, status="running",
                              lease_expires_at=datetime.utcnow() + timedelta(seconds=lease_s()))
        except Exception as exc:  # a missed beat is recoverable; the next one retries
            logger.warning("[jobs] heartbeat for %s failed: %s", job_id, exc)
            continue
        if not held:  # the row was finished elsewhere (e.g. shutdown): stop claiming it
            return


async def _run(dispatcher_cls, edition, tenant_id: str, job_id: str, owner: str, tool_name: str,
               arguments: Dict[str, Any], user_id: str, context: Dict[str, Any]) -> None:
    from fastapi.encoders import jsonable_encoder

    beat = asyncio.get_running_loop().create_task(_heartbeat(tenant_id, job_id, owner))
    success, data, error = False, None, None
    try:
        async with _limit():
            db = await _bound_session(tenant_id)
            try:
                result = await asyncio.wait_for(
                    dispatcher_cls(db, edition).invoke(tool_name, arguments, user_id, context), timeout=max_s()
                )
                data, error = jsonable_encoder(result.data or {}), result.error
                if result.success:
                    await db.commit()  # a commit that fails is a failed job, never "succeeded"
                    success = True
                else:
                    await db.rollback()
            finally:
                await db.close()
    except asyncio.TimeoutError:
        error = f"It took longer than {max_s():.0f}s and was stopped."
    except asyncio.CancelledError:
        error = "The server stopped while this was running — please try again."
        raise
    except Exception as exc:
        logger.exception("[jobs] %s failed", tool_name)
        error = str(exc)
    finally:
        beat.cancel()
        meaning = outcome(tool_name, success, data, error)
        try:
            await _cas(tenant_id, job_id, owner, status=meaning["status"], progress=meaning["summary"],
                       result={"data": data, "outcome": meaning}, error=error,
                       finished_at=datetime.utcnow())
        except Exception as exc:
            logger.error("[jobs] could not record the end of %s: %s", job_id, exc)
        _owned.pop(job_id, None)


async def status_for(db, session_id: Any, user_id: str, job_id: str) -> Dict[str, Any]:
    """The job as the card shows it, scoped to the caller's session; a lapsed lease
    reads as `stale` (detected here, at read time). Unknown → status None."""
    from models.conversation import ConversationJob

    try:
        job_uuid = uuid.UUID(str(job_id))
    except ValueError:
        return {"status": None}
    scope = (ConversationJob.id == job_uuid, ConversationJob.session_id == session_id,
             ConversationJob.user_id == str(user_id))
    await db.execute(update(ConversationJob).where(
        *scope, ConversationJob.status == "running", ConversationJob.lease_expires_at < datetime.utcnow(),
    ).values(status="stale", progress="It stopped reporting progress — it may have been interrupted."))
    await db.commit()
    job = (await db.execute(select(ConversationJob).where(*scope))).scalar_one_or_none()
    if job is None:
        return {"status": None}
    meaning = (job.result or {}).get("outcome") or {}
    return {
        "status": job.status,
        "summary": job.progress,
        "label": _LABELS.get(job.tool_name, job.tool_name).capitalize(),
        "tool_name": job.tool_name,
        "target": meaning.get("target"),
        "error": job.error,
    }


async def claim_finished(db, session_id: Any) -> List[Dict[str, Any]]:
    """Finished jobs of this session not yet folded into the conversation, stamped
    as harvested in the caller's transaction (the caller commits)."""
    from models.conversation import ConversationJob

    jobs = (await db.execute(select(ConversationJob).where(
        ConversationJob.session_id == session_id, ConversationJob.status.in_(TERMINAL),
        ConversationJob.harvested_at.is_(None),
    ).with_for_update(skip_locked=True))).scalars().all()
    claimed = []
    for job in jobs:
        job.harvested_at = datetime.utcnow()
        claimed.append({"job_id": str(job.id), "tool_name": job.tool_name, "status": job.status,
                        "outcome": (job.result or {}).get("outcome") or {}})
    return claimed


async def fail_owned_on_shutdown() -> None:
    """At worker shutdown, mark this worker's running jobs failed (no waiting for the lease)."""
    for job_id, (owner, tenant_id) in list(_owned.items()):
        try:
            await _cas(tenant_id, job_id, owner, status="failed",
                       error="The server restarted while this was running — please try again.",
                       progress="The server restarted while this was running — please try again.",
                       finished_at=datetime.utcnow())
        except Exception as exc:
            logger.warning("[jobs] could not mark %s failed at shutdown: %s", job_id, exc)
    _owned.clear()


def _metadata_access_token() -> str:
    import httpx

    resp = httpx.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"}, timeout=5.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


async def enqueue_cloud_task(payload: Dict[str, Any]) -> str:
    """Queue one job as a Cloud Task that POSTs `payload` to the run endpoint with
    an OIDC token minted for `CONVERSATION_JOBS_TASK_SA` (REST; no client library)."""
    import base64
    import json

    import httpx

    queue = os.environ["CONVERSATION_JOBS_QUEUE"]          # projects/P/locations/L/queues/Q
    url = os.environ["CONVERSATION_JOBS_TASK_URL"]         # …/api/v1/conversation/internal/jobs/run
    service_account = os.environ["CONVERSATION_JOBS_TASK_SA"]
    token = await asyncio.to_thread(_metadata_access_token)
    deadline = int(min(1800, max(15, max_s() + 30)))
    body = {"task": {
        "dispatchDeadline": f"{deadline}s",
        "httpRequest": {
            "httpMethod": "POST", "url": url,
            "headers": {"Content-Type": "application/json"},
            "body": base64.b64encode(json.dumps(payload).encode()).decode(),
            "oidcToken": {"serviceAccountEmail": service_account, "audience": url},
        },
    }}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"https://cloudtasks.googleapis.com/v2/{queue}/tasks", json=body,
                                 headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    return resp.json().get("name", "")


def verify_task_request(authorization: Optional[str]) -> bool:
    """True when the request carries a Google OIDC token for the configured
    service account and audience (what Cloud Tasks sends)."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return False
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
    except ImportError:  # google-auth ships with the Enterprise (Cloud Run) image only
        logger.error("[jobs] google-auth is not installed; refusing the job callback")
        return False
    try:
        claims = id_token.verify_oauth2_token(
            authorization.split(" ", 1)[1], google_requests.Request(),
            audience=os.environ["CONVERSATION_JOBS_TASK_URL"],
        )
    except Exception as exc:
        logger.warning("[jobs] rejected a job callback: %s", exc)
        return False
    return (claims.get("email") == os.environ.get("CONVERSATION_JOBS_TASK_SA")
            and bool(claims.get("email_verified")))


async def run_dispatched(payload: Dict[str, Any]) -> str:
    """Run a queued job inside the calling request (Cloud Tasks callback).

    The job is taken over by a fresh owner token in one compare-and-set, so a
    duplicate delivery finds it already taken and runs nothing. Returns
    `ran` or `skipped`.
    """
    import importlib

    name = payload.get("dispatcher", "")
    if name not in _DISPATCHERS:
        raise ValueError(f"unknown dispatcher {name!r}")
    module_name, cls_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    dispatcher_cls = getattr(module, cls_name)
    edition = getattr(module, "Edition")(payload["edition"])

    tenant_id, job_id = payload["tenant_id"], payload["job_id"]
    owner = uuid.uuid4().hex
    if not await _cas(tenant_id, job_id, payload["owner"], owner_token=owner, status="running",
                      lease_expires_at=datetime.utcnow() + timedelta(seconds=lease_s())):
        logger.info("[jobs] %s was already taken or finished; skipping a repeat delivery", job_id)
        return "skipped"
    _owned[job_id] = (owner, tenant_id)
    await _run(dispatcher_cls, edition, tenant_id, job_id, owner, payload["tool_name"],
               payload.get("arguments") or {}, payload["user_id"], payload.get("context") or {})
    return "ran"
