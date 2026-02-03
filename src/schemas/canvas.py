"""A2UI Canvas schemas — defines block types and rendering requests.

Community edition supports: chart, table, text, metric, status blocks.
Business extends with: form, diagram, log, composite blocks.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    """Community canvas block types."""
    CHART = "chart"
    TABLE = "table"
    TEXT = "text"
    METRIC = "metric"
    STATUS = "status"


class CanvasBlock(BaseModel):
    """A single renderable block in the canvas."""
    id: Optional[str] = None
    block_type: str = Field(..., description="Block type: chart, table, text, metric, status")
    title: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict, description="Block data payload")
    layout: Dict[str, Any] = Field(
        default_factory=dict,
        description="Layout hints: width, height, position"
    )
    style: Dict[str, Any] = Field(default_factory=dict, description="Visual styling hints")


class CanvasRenderRequest(BaseModel):
    """Request to render a canvas with specified blocks."""
    blocks: List[CanvasBlock] = Field(..., description="Blocks to render")
    layout: str = Field(default="auto", description="Layout mode: auto, grid, flow, stack")
    title: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class CanvasRenderResponse(BaseModel):
    """Rendered canvas output."""
    canvas_id: str
    blocks: List[CanvasBlock]
    layout: str
    title: Optional[str] = None
    render_hints: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class AutoDetectRequest(BaseModel):
    """Request to auto-detect the best block type for given data."""
    data: Any = Field(..., description="Raw data to analyze")
    hints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional hints: preferred_type, max_rows, etc."
    )


class AutoDetectResponse(BaseModel):
    """Result of auto-detecting block type."""
    detected_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    block: CanvasBlock
    alternatives: List[str] = Field(default_factory=list)
    reasoning: str = ""


class CanvasTemplate(BaseModel):
    """A reusable canvas template."""
    id: str
    name: str
    description: str
    block_types: List[str]
    layout: str
    preview_data: Dict[str, Any] = Field(default_factory=dict)
    category: str = "general"


class CanvasTemplateListResponse(BaseModel):
    """List of available canvas templates."""
    templates: List[CanvasTemplate]
    total: int
