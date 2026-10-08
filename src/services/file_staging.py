"""Stage an uploaded file: one validation and storage path for every entry point.

POST /file-upload/upload and the MCP `upload_file` tool both stage files here,
so the type allow-list, size cap, magic-byte check, governance risk check
(Business+), audit (Enterprise) and optional workflow trigger apply to both.
"""

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from core.config import get_settings
from core.tenant_context import get_current_tenant_id
from models.staged_file import StagedFile

logger = logging.getLogger(__name__)


class UploadRejected(Exception):
    """The upload is refused; `status` is the HTTP status the endpoint returns."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
    "application/vnd.ms-excel",  # xls
    "text/csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
    "text/plain",
    "image/png",
    "image/jpeg",
    "application/json",
}

# Magic byte signatures for content verification
MAGIC_BYTES = {
    "application/pdf": [b"%PDF"],
    "image/png": [b"\x89PNG"],
    "image/jpeg": [b"\xff\xd8\xff"],
    "application/json": [b"{", b"["],  # JSON starts with { or [
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [b"PK"],  # ZIP-based
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [b"PK"],
    "application/vnd.ms-excel": [b"\xd0\xcf\x11\xe0"],  # OLE2
}

MAX_FILENAME_LENGTH = 255


def _sanitize_filename(filename: str) -> str:
    """Strip path components and dangerous characters from filename."""
    # Take only the basename (strip directory components)
    name = os.path.basename(filename)
    # Remove any null bytes or control characters
    name = re.sub(r'[\x00-\x1f]', '', name)
    # Limit length
    if len(name) > MAX_FILENAME_LENGTH:
        base, ext = os.path.splitext(name)
        name = base[:MAX_FILENAME_LENGTH - len(ext)] + ext
    return name or "unnamed"


def _validate_magic_bytes(content: bytes, declared_type: str) -> bool:
    """Verify file content matches declared MIME type via magic bytes."""
    signatures = MAGIC_BYTES.get(declared_type)
    if not signatures:
        return True  # No signature to check (e.g., text/csv, text/plain)
    return any(content[:len(sig)] == sig for sig in signatures)


def check_size(size: int) -> None:
    if size > MAX_FILE_SIZE:
        raise UploadRejected(413, f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)} MB")


async def stage_upload(
    db,
    user_id: str,
    filename: str,
    content_type: Optional[str],
    contents: bytes,
    workflow_id: Optional[str] = None,
    source: str = "web_upload",
) -> Dict[str, Any]:
    """Validate, store and record the file; optionally start a workflow with it."""
    safe_filename = _sanitize_filename(filename or "unnamed")
    if content_type not in ALLOWED_TYPES:
        logger.warning(f"File upload rejected: unsupported type {content_type} from user {user_id}")
        raise UploadRejected(400, f"Unsupported file type: {content_type}. Allowed: {', '.join(sorted(ALLOWED_TYPES))}")
    check_size(len(contents))
    if not _validate_magic_bytes(contents, content_type):
        logger.warning(
            f"File upload rejected: magic bytes mismatch for {safe_filename} "
            f"(declared={content_type}) from user {user_id}"
        )
        raise UploadRejected(400, "File content does not match declared type. The file may be corrupted or mislabeled.")

    logger.info(f"File upload accepted: {safe_filename} ({content_type}, {len(contents)} bytes) from user {user_id}")

    # Business+ governance: risk assessment on file uploads
    try:
        from aictrlnet_business.services.ai_governance import RiskAssessmentEngine
        risk_result = RiskAssessmentEngine(db).assess_file_risk({
            "filename": safe_filename,
            "content_type": content_type,
            "file_size": len(contents),
            "source": source,
        })
        if risk_result.get("risk_level") in ("very_high", "critical"):
            logger.warning(f"File upload blocked by governance: {risk_result}")
            raise UploadRejected(
                403,
                f"File upload blocked: risk level {risk_result.get('risk_level')} — "
                f"{risk_result.get('summary', 'policy violation')}",
            )
        logger.info(f"File governance: risk={risk_result.get('risk_level', 'n/a')}, score={risk_result.get('risk_score', 0):.2f}")
    except ImportError:
        pass  # Community edition — no governance

    # Enterprise audit logging
    try:
        from aictrlnet_enterprise.services.audit_service import AuditService
        await AuditService.audit_file_upload(
            db=db,
            user_id=str(user_id),
            tenant_id=get_current_tenant_id(),
            file_metadata={"filename": safe_filename, "content_type": content_type, "file_size": len(contents)},
        )
    except ImportError:
        pass  # Community/Business edition — no audit service

    file_id = uuid.uuid4()
    # Read at call time, not import time: PHI deployments point this at the
    # encrypted volume, and the startup guard has already refused to boot if
    # it still resolves under /tmp.
    upload_dir = get_settings().STAGED_FILES_DIR
    storage_path = os.path.join(upload_dir, str(file_id))

    def _write_upload():
        os.makedirs(upload_dir, exist_ok=True)
        with open(storage_path, "wb") as f:
            f.write(contents)

    await asyncio.to_thread(_write_upload)

    staged = StagedFile(
        id=file_id,
        user_id=str(user_id),
        filename=safe_filename,
        content_type=content_type or "application/octet-stream",
        file_size=len(contents),
        storage_path=storage_path,
        stage="uploaded",
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )
    db.add(staged)
    await db.commit()

    execution_id = None
    if workflow_id:
        try:
            from services.workflow_execution import WorkflowExecutionService
            execution = await WorkflowExecutionService(db).create_execution(
                workflow_id=workflow_id,
                input_data={
                    "file_id": str(file_id),
                    "filename": staged.filename,
                    "content_type": staged.content_type,
                    "storage_path": storage_path,
                },
                triggered_by="file_upload",
                trigger_metadata={"file_id": str(file_id), "user_id": str(user_id)},
                tenant_id=get_current_tenant_id(),
                user_id=str(user_id),
            )
            execution_id = str(execution.id)
            logger.info(f"File upload triggered workflow {workflow_id}, execution {execution_id}")
        except Exception as e:
            logger.error(f"Failed to trigger workflow from file upload: {e}")

    return {
        "file_id": file_id,
        "filename": staged.filename,
        "content_type": staged.content_type,
        "file_size": staged.file_size,
        "execution_id": execution_id,
    }
