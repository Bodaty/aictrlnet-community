"""Content by reference for workflow steps.

Every step's inputs and outputs are persisted to run history, so a step that
handles a PHI pipeline's page images, page text or extracted values must not
carry them inline. It names them instead: `input_ref = {"kind", "id"}` is read
through a resolver registered for that kind, and `output_sink = {"kind", "id"}`
is written through a registered sink. Both are called with the run's tenant.
A sink returns ids and counts only; that is what the step keeps.

Editions register their resolvers and sinks at startup.
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypedDict


class ResolvedContent(TypedDict, total=False):
    text: Optional[str]
    images: Optional[List[str]]  # base64, no data-URI prefix


Resolver = Callable[[str, str, dict], Awaitable[ResolvedContent]]
Sink = Callable[[str, str, dict, dict], Awaitable[dict]]


class ContentRefError(ValueError):
    """A content reference that is malformed, unregistered or unresolvable."""


_resolvers: Dict[str, Resolver] = {}
_sinks: Dict[str, Sink] = {}


def register_resolver(kind: str, resolve: Resolver) -> None:
    _resolvers[kind] = resolve


def unregister_resolver(kind: str) -> None:
    _resolvers.pop(kind, None)


def register_sink(kind: str, write: Sink) -> None:
    _sinks[kind] = write


def unregister_sink(kind: str) -> None:
    _sinks.pop(kind, None)


def parse_ref(ref: Any, field: str) -> Tuple[str, str]:
    if not isinstance(ref, dict) or not ref.get("kind") or not ref.get("id"):
        raise ContentRefError(f"'{field}' must be an object with 'kind' and 'id'; got {type(ref).__name__}")
    return str(ref["kind"]), str(ref["id"])


def check_registered(input_ref: Any, output_sink: Any) -> None:
    """Fail before any model call when either side cannot be handled."""
    in_kind, _ = parse_ref(input_ref, "input_ref")
    out_kind, _ = parse_ref(output_sink, "output_sink")
    if in_kind not in _resolvers:
        raise ContentRefError(f"No content resolver is registered for input_ref kind '{in_kind}'")
    if out_kind not in _sinks:
        raise ContentRefError(f"No content sink is registered for output_sink kind '{out_kind}'")


def _require_tenant(tenant_id: Optional[str]) -> str:
    if not tenant_id:
        raise ContentRefError("Content by reference needs the run's tenant; none is set")
    return tenant_id


async def resolve_input(input_ref: Any, tenant_id: Optional[str], context: dict) -> ResolvedContent:
    kind, ref_id = parse_ref(input_ref, "input_ref")
    resolve = _resolvers.get(kind)
    if resolve is None:
        raise ContentRefError(f"No content resolver is registered for input_ref kind '{kind}'")
    resolved = await resolve(_require_tenant(tenant_id), ref_id, context)
    if not isinstance(resolved, dict) or not (resolved.get("text") or resolved.get("images")):
        raise ContentRefError(f"The '{kind}' resolver returned no content for id '{ref_id}'")
    return resolved


async def write_output(output_sink: Any, tenant_id: Optional[str], context: dict, data: dict) -> dict:
    kind, sink_id = parse_ref(output_sink, "output_sink")
    write = _sinks.get(kind)
    if write is None:
        raise ContentRefError(f"No content sink is registered for output_sink kind '{kind}'")
    result = await write(_require_tenant(tenant_id), sink_id, context, data)
    if not isinstance(result, dict):
        raise ContentRefError(f"The '{kind}' sink must return a dict of ids and counts")
    return result


def scrub_error(msg: str, workflow_phi: bool, exc_type: str = "Error") -> str:
    """An error text safe to store: in a `phi: true` workflow any message may
    quote document content, so only the exception type is kept."""
    if workflow_phi:
        return f"<redacted: {exc_type}>"
    return msg
