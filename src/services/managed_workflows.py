from typing import Optional

from core.exceptions import ConflictError

MANAGED_KEY = "managed"


def managed_by(workflow) -> Optional[dict]:
    marker = (getattr(workflow, "workflow_metadata", None) or {}).get(MANAGED_KEY)
    if not isinstance(marker, dict) or not marker.get("by"):
        return None
    return marker


def managed_steps_detail(marker: dict) -> dict:
    title = marker.get("title") or marker.get("by")
    return {
        "code": "managed_steps",
        "detail": (
            f"This workflow's steps are kept up to date by {title} and can't be changed here. "
            "Make your own copy to change them."
        ),
    }


def copy_tags(source_tags, marker: Optional[dict]) -> list:
    strip = set((marker or {}).get("strip_tags") or [])
    return [t for t in (source_tags or []) if t not in strip]


def refuse_managed_change(workflow, *, definition_change: bool = False, new_metadata: Optional[dict] = None) -> None:
    marker = managed_by(workflow)
    if not marker:
        return
    altered = new_metadata is not None and MANAGED_KEY in new_metadata and new_metadata[MANAGED_KEY] != marker
    if definition_change or altered:
        detail = managed_steps_detail(marker)
        raise ConflictError(detail["detail"], {"code": detail["code"]})


def keep_marker(workflow, new_metadata: dict) -> dict:
    marker = managed_by(workflow)
    if marker and MANAGED_KEY not in new_metadata:
        return {**new_metadata, MANAGED_KEY: workflow.workflow_metadata[MANAGED_KEY]}
    return new_metadata
