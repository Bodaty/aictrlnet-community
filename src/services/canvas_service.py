"""A2UI Canvas Render Service — auto-detects data shapes and renders blocks.

Community edition supports chart, table, text, metric, and status blocks.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from schemas.canvas import (
    AutoDetectRequest,
    AutoDetectResponse,
    BlockType,
    CanvasBlock,
    CanvasRenderRequest,
    CanvasRenderResponse,
    CanvasTemplate,
    CanvasTemplateListResponse,
)

logger = logging.getLogger(__name__)

# Community block types
COMMUNITY_BLOCK_TYPES = {
    BlockType.CHART.value,
    BlockType.TABLE.value,
    BlockType.TEXT.value,
    BlockType.METRIC.value,
    BlockType.STATUS.value,
}

# Built-in templates
BUILT_IN_TEMPLATES = [
    CanvasTemplate(
        id="metric-dashboard",
        name="Metric Dashboard",
        description="Display key metrics with charts and status indicators",
        block_types=["metric", "chart", "status"],
        layout="grid",
        preview_data={"metrics": 4, "charts": 2},
        category="dashboard",
    ),
    CanvasTemplate(
        id="data-table-view",
        name="Data Table View",
        description="Tabular data display with optional summary metrics",
        block_types=["table", "metric"],
        layout="stack",
        preview_data={"rows": 10, "columns": 5},
        category="data",
    ),
    CanvasTemplate(
        id="status-overview",
        name="Status Overview",
        description="System status dashboard with health indicators",
        block_types=["status", "text", "metric"],
        layout="grid",
        preview_data={"services": 6},
        category="monitoring",
    ),
    CanvasTemplate(
        id="report-layout",
        name="Report Layout",
        description="Text-heavy layout for reports with embedded charts",
        block_types=["text", "chart", "table"],
        layout="flow",
        preview_data={"sections": 3},
        category="reports",
    ),
]


class CanvasRenderService:
    """Renders canvas blocks from data hints and auto-detects data shapes."""

    def render(self, request: CanvasRenderRequest) -> CanvasRenderResponse:
        """Render a canvas from the provided blocks and layout."""
        rendered_blocks = []
        for block in request.blocks:
            if block.block_type not in COMMUNITY_BLOCK_TYPES:
                logger.warning(
                    f"Block type '{block.block_type}' not available in Community. "
                    f"Falling back to 'text'."
                )
                block.block_type = "text"

            if not block.id:
                block.id = str(uuid.uuid4())[:8]

            rendered_blocks.append(block)

        return CanvasRenderResponse(
            canvas_id=str(uuid.uuid4()),
            blocks=rendered_blocks,
            layout=request.layout,
            title=request.title,
            render_hints={
                "block_count": len(rendered_blocks),
                "supported_types": list(COMMUNITY_BLOCK_TYPES),
                "edition": "community",
            },
            created_at=datetime.utcnow(),
        )

    def auto_detect(self, request: AutoDetectRequest) -> AutoDetectResponse:
        """Auto-detect the best block type for given data."""
        data = request.data
        hints = request.hints
        preferred = hints.get("preferred_type")

        detected_type = "text"
        confidence = 0.5
        reasoning = ""
        alternatives = []

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # List of objects → table or chart
                keys = set()
                for item in data[:10]:
                    if isinstance(item, dict):
                        keys.update(item.keys())

                has_numeric = any(
                    isinstance(data[0].get(k), (int, float))
                    for k in keys
                    if k in data[0]
                )

                if has_numeric and len(data) > 2:
                    detected_type = "chart"
                    confidence = 0.8
                    reasoning = f"List of {len(data)} objects with numeric values → chart"
                    alternatives = ["table"]
                else:
                    detected_type = "table"
                    confidence = 0.85
                    reasoning = f"List of {len(data)} objects → table"
                    alternatives = ["chart"]
            elif len(data) > 0 and isinstance(data[0], (int, float)):
                detected_type = "chart"
                confidence = 0.9
                reasoning = f"List of {len(data)} numbers → chart"
                alternatives = ["table", "metric"]
            else:
                detected_type = "table"
                confidence = 0.6
                reasoning = f"List of {len(data)} items → table"
                alternatives = ["text"]

        elif isinstance(data, dict):
            if "status" in data or "health" in data or "state" in data:
                detected_type = "status"
                confidence = 0.85
                reasoning = "Dict with status/health/state key → status block"
                alternatives = ["metric", "text"]
            elif len(data) <= 5 and all(
                isinstance(v, (int, float, str)) for v in data.values()
            ):
                detected_type = "metric"
                confidence = 0.8
                reasoning = f"Small dict with {len(data)} scalar values → metric"
                alternatives = ["table", "text"]
            else:
                detected_type = "table"
                confidence = 0.6
                reasoning = "Dict with mixed values → table"
                alternatives = ["text"]

        elif isinstance(data, str):
            detected_type = "text"
            confidence = 0.95
            reasoning = "String data → text block"
            alternatives = []

        elif isinstance(data, (int, float)):
            detected_type = "metric"
            confidence = 0.9
            reasoning = "Single numeric value → metric"
            alternatives = ["text"]

        # Apply preferred type if valid
        if preferred and preferred in COMMUNITY_BLOCK_TYPES:
            detected_type = preferred
            confidence = max(confidence, 0.7)
            reasoning = f"User preferred '{preferred}' applied"

        # Build the block
        block_data = {"raw": data}
        if detected_type == "chart" and isinstance(data, list):
            block_data = {"series": data}
        elif detected_type == "table" and isinstance(data, list):
            block_data = {"rows": data}
        elif detected_type == "metric":
            if isinstance(data, dict):
                block_data = data
            else:
                block_data = {"value": data}
        elif detected_type == "status":
            if isinstance(data, dict):
                block_data = data
            else:
                block_data = {"status": str(data)}
        elif detected_type == "text":
            block_data = {"content": str(data) if not isinstance(data, dict) else str(data)}

        block = CanvasBlock(
            id=str(uuid.uuid4())[:8],
            block_type=detected_type,
            data=block_data,
        )

        return AutoDetectResponse(
            detected_type=detected_type,
            confidence=confidence,
            block=block,
            alternatives=alternatives,
            reasoning=reasoning,
        )

    def get_templates(self) -> CanvasTemplateListResponse:
        """Return built-in canvas templates."""
        return CanvasTemplateListResponse(
            templates=BUILT_IN_TEMPLATES,
            total=len(BUILT_IN_TEMPLATES),
        )
