"""File upload endpoint for staging files for workflow processing."""

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.config import get_settings
from core.database import get_db
from core.security import get_current_user
from core.tenant_context import get_current_tenant_id
from models.staged_file import StagedFile
from services.file_staging import ALLOWED_TYPES, UploadRejected, check_size, stage_upload
from schemas.file_upload import FileUploadResponse, StagedFileResponse
from api.v1.endpoints._auth_helpers import get_safe_user_id

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    workflow_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Upload a file for processing in workflows.

    Optionally pass ``workflow_id`` to trigger a workflow execution with the
    uploaded file as input (``triggered_by="file_upload"``).
    """
    user_id = get_safe_user_id(current_user) or "unknown"

    # Reject oversized uploads by Content-Length BEFORE reading the body — avoids
    # buffering a huge (or spooled-to-disk) payload just to measure it (DoS).
    declared_len = request.headers.get("content-length")
    try:
        if declared_len and declared_len.isdigit():
            check_size(int(declared_len))
        if file.content_type not in ALLOWED_TYPES:
            raise UploadRejected(
                400, f"Unsupported file type: {file.content_type}. Allowed: {', '.join(sorted(ALLOWED_TYPES))}")

        # Read in bounded chunks so a lying/absent Content-Length can't OOM the
        # process: abort as soon as the running total exceeds the cap.
        chunks = []
        total = 0
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            total += len(chunk)
            check_size(total)
            chunks.append(chunk)

        staged = await stage_upload(
            db, str(user_id), file.filename or "unnamed", file.content_type, b"".join(chunks),
            workflow_id=workflow_id,
        )
    except UploadRejected as e:
        raise HTTPException(status_code=e.status, detail=e.detail)
    return FileUploadResponse(**staged)


@router.get("/{file_id}", response_model=StagedFileResponse)
async def get_staged_file(
    file_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get details of a staged file."""
    user_id = get_safe_user_id(current_user) or "unknown"

    result = await db.execute(
        select(StagedFile).filter(StagedFile.id == file_id, StagedFile.user_id == str(user_id))
    )
    staged = result.scalar_one_or_none()
    if not staged:
        raise HTTPException(status_code=404, detail="File not found")

    return staged
