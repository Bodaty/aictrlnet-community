"""Tests for Canvas Render service and endpoints (OPENCLAW Priority 3)."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
import uuid

from services.canvas_service import (
    CanvasRenderService,
    COMMUNITY_BLOCK_TYPES,
    BUILT_IN_TEMPLATES,
)
from schemas.canvas import (
    CanvasBlock,
    CanvasRenderRequest,
    AutoDetectRequest,
    BlockType,
)


@pytest.fixture
def canvas_service():
    """Create CanvasRenderService instance."""
    return CanvasRenderService()


class TestCanvasRenderServiceRender:
    """Tests for canvas rendering operations."""

    def test_render_single_text_block(self, canvas_service):
        """Test rendering a single text block."""
        request = CanvasRenderRequest(
            blocks=[
                CanvasBlock(block_type="text", data={"content": "Hello World"}),
            ],
            layout="stack",
            title="Test Canvas",
        )

        result = canvas_service.render(request)

        assert result.canvas_id is not None
        assert len(result.blocks) == 1
        assert result.blocks[0].block_type == "text"
        assert result.title == "Test Canvas"
        assert result.layout == "stack"

    def test_render_multiple_blocks(self, canvas_service):
        """Test rendering multiple blocks."""
        request = CanvasRenderRequest(
            blocks=[
                CanvasBlock(block_type="metric", data={"value": 42, "label": "Count"}),
                CanvasBlock(block_type="chart", data={"series": [1, 2, 3, 4, 5]}),
                CanvasBlock(block_type="table", data={"rows": [{"a": 1}, {"a": 2}]}),
            ],
            layout="grid",
            title="Dashboard",
        )

        result = canvas_service.render(request)

        assert len(result.blocks) == 3
        assert result.render_hints["block_count"] == 3
        assert result.render_hints["edition"] == "community"

    def test_render_assigns_ids_to_blocks(self, canvas_service):
        """Test that blocks without IDs get assigned one."""
        request = CanvasRenderRequest(
            blocks=[
                CanvasBlock(block_type="text", data={"content": "No ID"}),
            ],
            layout="stack",
        )

        result = canvas_service.render(request)

        assert result.blocks[0].id is not None
        assert len(result.blocks[0].id) == 8

    def test_render_preserves_existing_ids(self, canvas_service):
        """Test that blocks with existing IDs are preserved."""
        request = CanvasRenderRequest(
            blocks=[
                CanvasBlock(id="my-id-123", block_type="text", data={"content": "Has ID"}),
            ],
            layout="stack",
        )

        result = canvas_service.render(request)

        assert result.blocks[0].id == "my-id-123"

    def test_render_unsupported_type_falls_back_to_text(self, canvas_service):
        """Test that unsupported block types fall back to text."""
        request = CanvasRenderRequest(
            blocks=[
                CanvasBlock(block_type="diagram", data={"nodes": []}),  # diagram is Business-only
            ],
            layout="stack",
        )

        result = canvas_service.render(request)

        assert result.blocks[0].block_type == "text"

    def test_render_all_community_block_types(self, canvas_service):
        """Test rendering all Community-supported block types."""
        blocks = [
            CanvasBlock(block_type="chart", data={"series": [1, 2, 3]}),
            CanvasBlock(block_type="table", data={"rows": []}),
            CanvasBlock(block_type="text", data={"content": "text"}),
            CanvasBlock(block_type="metric", data={"value": 100}),
            CanvasBlock(block_type="status", data={"status": "ok"}),
        ]
        request = CanvasRenderRequest(blocks=blocks, layout="grid")

        result = canvas_service.render(request)

        assert len(result.blocks) == 5
        block_types = [b.block_type for b in result.blocks]
        assert "chart" in block_types
        assert "table" in block_types
        assert "text" in block_types
        assert "metric" in block_types
        assert "status" in block_types


class TestCanvasRenderServiceAutoDetect:
    """Tests for auto-detection of block types."""

    def test_auto_detect_list_of_objects_as_table(self, canvas_service):
        """Test auto-detecting list of objects as table."""
        request = AutoDetectRequest(
            data=[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.detected_type == "table"
        assert result.confidence >= 0.6

    def test_auto_detect_list_of_objects_with_numbers_as_chart(self, canvas_service):
        """Test auto-detecting list of objects with numbers as chart."""
        request = AutoDetectRequest(
            data=[
                {"month": "Jan", "sales": 100},
                {"month": "Feb", "sales": 150},
                {"month": "Mar", "sales": 200},
            ],
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.detected_type == "chart"
        assert result.confidence >= 0.7

    def test_auto_detect_list_of_numbers_as_chart(self, canvas_service):
        """Test auto-detecting list of numbers as chart."""
        request = AutoDetectRequest(
            data=[10, 20, 30, 40, 50],
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.detected_type == "chart"
        assert result.confidence >= 0.9

    def test_auto_detect_dict_with_status_key(self, canvas_service):
        """Test auto-detecting dict with status key as status block."""
        request = AutoDetectRequest(
            data={"status": "healthy", "uptime": "99.9%"},
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.detected_type == "status"
        assert result.confidence >= 0.8

    def test_auto_detect_dict_with_health_key(self, canvas_service):
        """Test auto-detecting dict with health key as status block."""
        request = AutoDetectRequest(
            data={"health": "ok", "latency_ms": 50},
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.detected_type == "status"

    def test_auto_detect_small_dict_as_metric(self, canvas_service):
        """Test auto-detecting small dict with scalars as metric."""
        request = AutoDetectRequest(
            data={"count": 42, "avg": 3.14},
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.detected_type == "metric"
        assert result.confidence >= 0.7

    def test_auto_detect_string_as_text(self, canvas_service):
        """Test auto-detecting string as text block."""
        request = AutoDetectRequest(
            data="This is a paragraph of text content.",
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.detected_type == "text"
        assert result.confidence >= 0.9

    def test_auto_detect_single_number_as_metric(self, canvas_service):
        """Test auto-detecting single number as metric."""
        request = AutoDetectRequest(
            data=42,
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.detected_type == "metric"
        assert result.confidence >= 0.9

    def test_auto_detect_float_as_metric(self, canvas_service):
        """Test auto-detecting float as metric."""
        request = AutoDetectRequest(
            data=3.14159,
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.detected_type == "metric"

    def test_auto_detect_respects_preferred_type(self, canvas_service):
        """Test that preferred type hint is respected."""
        request = AutoDetectRequest(
            data=[1, 2, 3, 4, 5],  # Would normally be chart
            hints={"preferred_type": "table"},
        )

        result = canvas_service.auto_detect(request)

        assert result.detected_type == "table"

    def test_auto_detect_ignores_invalid_preferred_type(self, canvas_service):
        """Test that invalid preferred type is ignored."""
        request = AutoDetectRequest(
            data="text content",
            hints={"preferred_type": "diagram"},  # Not in Community
        )

        result = canvas_service.auto_detect(request)

        # Should still detect as text since diagram is invalid
        assert result.detected_type == "text"

    def test_auto_detect_returns_block_object(self, canvas_service):
        """Test that auto_detect returns a valid block object."""
        request = AutoDetectRequest(
            data={"metric": 100},
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.block is not None
        assert result.block.id is not None
        assert result.block.block_type == result.detected_type
        assert result.block.data is not None

    def test_auto_detect_provides_alternatives(self, canvas_service):
        """Test that auto_detect provides alternative suggestions."""
        request = AutoDetectRequest(
            data=[{"x": 1}, {"x": 2}],
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert isinstance(result.alternatives, list)

    def test_auto_detect_provides_reasoning(self, canvas_service):
        """Test that auto_detect provides reasoning."""
        request = AutoDetectRequest(
            data=[1, 2, 3],
            hints={},
        )

        result = canvas_service.auto_detect(request)

        assert result.reasoning is not None
        assert len(result.reasoning) > 0


class TestCanvasRenderServiceTemplates:
    """Tests for canvas template operations."""

    def test_get_templates(self, canvas_service):
        """Test getting built-in templates."""
        result = canvas_service.get_templates()

        assert result.total == len(BUILT_IN_TEMPLATES)
        assert len(result.templates) > 0

    def test_templates_have_required_fields(self, canvas_service):
        """Test that all templates have required fields."""
        result = canvas_service.get_templates()

        for template in result.templates:
            assert template.id is not None
            assert template.name is not None
            assert template.description is not None
            assert template.block_types is not None
            assert template.layout is not None

    def test_metric_dashboard_template_exists(self, canvas_service):
        """Test that metric dashboard template exists."""
        result = canvas_service.get_templates()

        template_ids = [t.id for t in result.templates]
        assert "metric-dashboard" in template_ids

    def test_data_table_view_template_exists(self, canvas_service):
        """Test that data table view template exists."""
        result = canvas_service.get_templates()

        template_ids = [t.id for t in result.templates]
        assert "data-table-view" in template_ids

    def test_status_overview_template_exists(self, canvas_service):
        """Test that status overview template exists."""
        result = canvas_service.get_templates()

        template_ids = [t.id for t in result.templates]
        assert "status-overview" in template_ids


class TestCommunityBlockTypes:
    """Tests for Community block type constants."""

    def test_all_community_types_exist(self):
        """Test all expected Community block types are defined."""
        expected = ["chart", "table", "text", "metric", "status"]
        for block_type in expected:
            assert block_type in COMMUNITY_BLOCK_TYPES

    def test_community_types_count(self):
        """Test correct number of Community block types."""
        assert len(COMMUNITY_BLOCK_TYPES) == 5

    def test_diagram_not_in_community(self):
        """Test that diagram is not in Community types (Business-only)."""
        assert "diagram" not in COMMUNITY_BLOCK_TYPES

    def test_form_not_in_community(self):
        """Test that form is not in Community types (Business-only)."""
        assert "form" not in COMMUNITY_BLOCK_TYPES


class TestBuiltInTemplates:
    """Tests for built-in template constants."""

    def test_templates_list_not_empty(self):
        """Test that built-in templates list is not empty."""
        assert len(BUILT_IN_TEMPLATES) > 0

    def test_templates_have_unique_ids(self):
        """Test that all templates have unique IDs."""
        ids = [t.id for t in BUILT_IN_TEMPLATES]
        assert len(ids) == len(set(ids))

    def test_template_block_types_are_valid(self):
        """Test that template block types are valid Community types."""
        for template in BUILT_IN_TEMPLATES:
            for block_type in template.block_types:
                assert block_type in COMMUNITY_BLOCK_TYPES, f"Template {template.id} has invalid block type {block_type}"

    def test_templates_have_categories(self):
        """Test that templates have categories."""
        for template in BUILT_IN_TEMPLATES:
            assert template.category is not None

    def test_templates_have_layouts(self):
        """Test that templates have valid layouts."""
        valid_layouts = {"grid", "stack", "flow"}
        for template in BUILT_IN_TEMPLATES:
            assert template.layout in valid_layouts, f"Template {template.id} has invalid layout {template.layout}"
