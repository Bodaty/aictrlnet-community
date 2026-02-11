"""Schemas for file upload and staged file processing."""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FileUploadResponse(BaseModel):
    """Response after uploading a file."""
    file_id: UUID
    filename: str
    content_type: str
    file_size: int
    stage: str = "uploaded"
    execution_id: Optional[str] = Field(None, description="Workflow execution ID if a workflow was triggered")


class StagedFileResponse(BaseModel):
    """Full staged file details."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: str
    filename: str
    content_type: str
    file_size: int
    stage: str
    extracted_data: Optional[Dict[str, Any]] = None
    created_at: datetime
    expires_at: datetime
