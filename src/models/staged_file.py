"""StagedFile model for file upload and processing pipeline."""

from sqlalchemy import Column, String, Text, JSON, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timedelta
import uuid

from .base import Base


class StagedFile(Base):
    """Represents an uploaded file staged for processing in workflows."""
    __tablename__ = "staged_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)

    # File metadata
    filename = Column(String(500), nullable=False)
    content_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    storage_path = Column(Text, nullable=False)

    # Processing state
    stage = Column(String(50), default="uploaded", nullable=False)  # uploaded, processing, processed, expired
    extracted_data = Column(JSON, nullable=True)  # Structured data extracted from file

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(hours=24), nullable=False)
