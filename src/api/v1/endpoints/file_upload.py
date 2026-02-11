"""File upload endpoint for staging files for workflow processing."""

import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from core.security import get_current_user
from models.staged_file import StagedFile
from schemas.file_upload import FileUploadResponse, StagedFileResponse

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "/tmp/aictrlnet/staged_files"
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


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    workflow_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Upload a file for processing in workflows.

    Optionally pass ``workflow_id`` to trigger a workflow execution with the
    uploaded file as input (``triggered_by="file_upload"``).
    """
    user_id = current_user.id if hasattr(current_user, "id") else current_user.get("id", "unknown")

    # Sanitize filename
    safe_filename = _sanitize_filename(file.filename or "unnamed")

    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        logger.warning(f"File upload rejected: unsupported type {file.content_type} from user {user_id}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        logger.warning(f"File upload rejected: too large ({len(contents)} bytes) from user {user_id}")
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)} MB")

    # Magic-byte validation: verify content matches declared MIME type
    if not _validate_magic_bytes(contents, file.content_type):
        logger.warning(
            f"File upload rejected: magic bytes mismatch for {safe_filename} "
            f"(declared={file.content_type}) from user {user_id}"
        )
        raise HTTPException(
            status_code=400,
            detail="File content does not match declared type. The file may be corrupted or mislabeled.",
        )

    logger.info(
        f"File upload accepted: {safe_filename} ({file.content_type}, {len(contents)} bytes) "
        f"from user {user_id}"
    )

    # Store file
    file_id = uuid.uuid4()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    storage_path = os.path.join(UPLOAD_DIR, str(file_id))
    with open(storage_path, "wb") as f:
        f.write(contents)

    # Create DB record
    staged = StagedFile(
        id=file_id,
        user_id=str(user_id),
        filename=safe_filename,
        content_type=file.content_type or "application/octet-stream",
        file_size=len(contents),
        storage_path=storage_path,
        stage="uploaded",
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )
    db.add(staged)
    await db.commit()

    # Optionally trigger a workflow with this file
    execution_id = None
    if workflow_id:
        try:
            from services.workflow_execution import WorkflowExecutionService
            wf_service = WorkflowExecutionService(db)
            execution = await wf_service.create_execution(
                workflow_id=workflow_id,
                input_data={
                    "file_id": str(file_id),
                    "filename": staged.filename,
                    "content_type": staged.content_type,
                    "storage_path": storage_path,
                },
                triggered_by="file_upload",
                trigger_metadata={
                    "file_id": str(file_id),
                    "user_id": str(user_id),
                },
            )
            execution_id = str(execution.id)
            logger.info(f"File upload triggered workflow {workflow_id}, execution {execution_id}")
        except Exception as e:
            logger.error(f"Failed to trigger workflow from file upload: {e}")

    return FileUploadResponse(
        file_id=file_id,
        filename=staged.filename,
        content_type=staged.content_type,
        file_size=staged.file_size,
        execution_id=execution_id,
    )


@router.get("/{file_id}", response_model=StagedFileResponse)
async def get_staged_file(
    file_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get details of a staged file."""
    user_id = current_user.id if hasattr(current_user, "id") else current_user.get("id", "unknown")

    result = await db.execute(
        select(StagedFile).filter(StagedFile.id == file_id, StagedFile.user_id == str(user_id))
    )
    staged = result.scalar_one_or_none()
    if not staged:
        raise HTTPException(status_code=404, detail="File not found")

    return staged
