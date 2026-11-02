"""Who may give a workflow a protected identity.

A workflow whose definition is marked ``phi``, or which carries a template id
an edition has registered as protected (its curated, PHI-handling templates),
is recognised by other parts of the platform by that identity. Only an
administrator (or the system acting on an administrator's behalf) creates,
duplicates or instantiates such a workflow, the same rule as for the autonomy
dial. ``services.workflow.check_definition_change`` applies the same rule to
updates.

Each of those paths, once it has checked the caller, stamps the definition with
the administrator's grant (``grant_identity``). The Documents area drives only
granted workflows, and the model layer (``models.identity_grant``) removes any
grant the server didn't issue, so a path that forgets its gate creates only an
inert row.
"""

from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from core.exceptions import ValidationError
from models.identity_grant import GRANT_KEY, is_granted, stamp  # noqa: F401

SYSTEM_ACTOR = "system"

_protected_template_ids: set = set()
_reserved_names: set = set()

# Trigger types that bind a workflow to something only the server sets up (the
# Documents inbox). The generic trigger route never inserts one itself: it hands
# an administrator's request to the creator the owning edition registered, which
# applies that feature's own checks, and refuses everyone else.
SERVER_MANAGED_TRIGGER_TYPES = frozenset({"document_inbox"})
_trigger_creators: Dict[str, Callable[..., Awaitable[Any]]] = {}


class DefinitionChangeForbidden(PermissionError):
    """A non-admin would give a workflow a protected identity (phi, or a template id)."""

    def __init__(self, fields: List[str]):
        super().__init__(f"Only an administrator can set {', '.join(fields)} on a workflow")
        self.fields = fields


class ReservedWorkflowName(ValidationError):
    """A non-admin would give a workflow a name an edition reserved (422)."""

    def __init__(self, name: str):
        super().__init__(f"The workflow name {name!r} is reserved; only an administrator can use it")
        self.name = name


class GrantedWorkflowChangeForbidden(PermissionError):
    """A non-admin would change, delete, reschedule or re-trigger a workflow
    whose identity an administrator granted (403)."""

    def __init__(self):
        super().__init__("Only an administrator can change this workflow, its schedules or its triggers")


def check_granted_workflow_change(workflow: Any, is_admin: bool) -> None:
    """The one rule for every mutation of a granted workflow (the document
    workflows and the nightly purge): updating or deleting it, its runs'
    pause/resume/cancel, and creating or deleting its schedules and triggers
    are an administrator's acts (GrantedWorkflowChangeForbidden otherwise).
    A workflow without a grant is unaffected."""
    if not is_admin and is_granted(getattr(workflow, "definition", None)):
        raise GrantedWorkflowChangeForbidden()


class TriggerRefused(Exception):
    """A server-managed trigger the generic route won't create (an HTTP status)."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def register_protected_template_ids(template_ids: Iterable[str]) -> None:
    """Editions register the template ids of their curated PHI templates."""
    _protected_template_ids.update(str(t) for t in template_ids)


def register_reserved_workflow_names(names: Iterable[str]) -> None:
    """Editions register workflow names other parts of the platform look up
    (the Documents area's stage workflows)."""
    _reserved_names.update(_name_key(n) for n in names)


def _name_key(name: Any) -> str:
    """Reserved names match whatever the case or surrounding whitespace."""
    return str(name).strip().casefold()


def check_workflow_name(name: Any, is_admin: bool, current_name: Any = None) -> None:
    """Raise ReservedWorkflowName when a non-admin would create a workflow with
    a reserved name, or rename one to it. Keeping a name it already has is fine."""
    if is_admin or name is None or name == current_name:
        return
    if _name_key(name) in _reserved_names:
        raise ReservedWorkflowName(str(name))


def register_trigger_creator(trigger_type: str, creator: Callable[..., Awaitable[Any]]) -> None:
    """An edition registers how a server-managed trigger type is created:
    ``await creator(workflow=<WorkflowDefinition>, trigger_data=<WorkflowTriggerCreate>)``,
    returning the trigger or raising TriggerRefused."""
    _trigger_creators[trigger_type] = creator


async def create_server_managed_trigger(workflow: Any, trigger_data: Any, is_admin: bool) -> Any:
    """Create a SERVER_MANAGED_TRIGGER_TYPES trigger for the generic route:
    an administrator only (403), through the owning edition's creator (422 when
    this edition has none, or when the creator refuses)."""
    if not is_admin:
        raise TriggerRefused(403, f"Only an administrator can create a {trigger_data.trigger_type} trigger")
    creator = _trigger_creators.get(trigger_data.trigger_type)
    if creator is None:
        raise TriggerRefused(422, f"{trigger_data.trigger_type} triggers are not available in this edition")
    return await creator(workflow=workflow, trigger_data=trigger_data)


def protected_identity(definition: Any) -> List[str]:
    """The parts of ``definition`` that only an administrator may give a workflow."""
    if isinstance(definition, str):
        import json

        try:
            definition = json.loads(definition)
        except ValueError:
            return []
    if not isinstance(definition, dict):
        return []
    fields = []
    if bool(definition.get("phi")):
        fields.append("phi")
    template_id = (definition.get("metadata") or {}).get("templateId")
    if template_id is not None and str(template_id) in _protected_template_ids:
        fields.append("templateId")
    return fields


def without_protected_identity(definition: Any) -> Any:
    """``definition`` with its protected identity (phi, a protected template id,
    any grant) removed, for paths that never give one (generated workflows)."""
    if not isinstance(definition, dict) or not (protected_identity(definition) or GRANT_KEY in definition):
        return definition
    cleaned = {k: v for k, v in definition.items() if k not in ("phi", GRANT_KEY)}
    metadata = dict(cleaned.get("metadata") or {})
    if str(metadata.get("templateId")) in _protected_template_ids:
        metadata.pop("templateId", None)
    cleaned["metadata"] = metadata
    return cleaned


def grant_identity(definition: Dict[str, Any], granted_by: str) -> Dict[str, Any]:
    """``definition`` stamped with an administrator's grant of its protected
    identity. Call only after the caller is known to be an administrator: the
    model layer keeps this stamp (once) and no other."""
    return stamp(definition, granted_by, protected_identity(definition))


def grant_if_protected(definition: Any, granted_by: Optional[str]) -> Any:
    """Stamp a grant when ``definition`` has a protected identity; else as is."""
    if isinstance(definition, dict) and protected_identity(definition) and granted_by:
        return grant_identity(definition, granted_by)
    return definition


def check_identity_grant(definition: Any, is_admin: bool) -> None:
    """Raise DefinitionChangeForbidden when a non-admin would create a workflow
    with a protected identity."""
    fields = protected_identity(definition)
    if fields and not is_admin:
        raise DefinitionChangeForbidden(fields)


async def actor_is_admin(db: Any, user_id: Optional[str]) -> bool:
    """The system actor, or an active administrator of the tenant explicitly
    bound to the session, else set for the request. With neither, no one is an
    administrator: the ``default-tenant`` fallback is not a tenant anyone chose."""
    if user_id == SYSTEM_ACTOR:
        return True
    if not user_id:
        return False
    from sqlalchemy import select
    from core.tenant_context import _current_tenant_id
    from models.user import User

    # The tenant is stated, not left to row-level security: a session with the
    # admin bypass would otherwise see every tenant's administrators.
    info = getattr(db, "info", None) or {}
    tenant_id = info.get("tenant_id") or _current_tenant_id.get()
    if not tenant_id:
        return False
    row = (await db.execute(
        select(User.is_superuser, User.is_active).where(
            User.id == str(user_id), User.tenant_id == str(tenant_id))
    )).first()
    return bool(row and row[0] and row[1])


def definition_dict(definition: Any) -> Dict[str, Any]:
    if hasattr(definition, "model_dump"):
        return definition.model_dump()
    return definition if isinstance(definition, dict) else {}
