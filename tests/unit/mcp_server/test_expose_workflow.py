"""Exposing a workflow as an MCP endpoint reads the mapped metadata column.

`workflow.metadata` on a SQLAlchemy model is the declarative MetaData object,
not the `metadata` column (mapped as `workflow_metadata`). Reading it made
every expose call fail with a TypeError (a 500 from the route).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_server.services.workflow import MCPWorkflowService
from models.community_complete import WorkflowDefinition


def _service(workflow):
    svc = MCPWorkflowService.__new__(MCPWorkflowService)
    svc.workflow_service = MagicMock()
    svc.workflow_service.get_workflow = AsyncMock(return_value=workflow)
    svc.workflow_service.update_workflow = AsyncMock(return_value=workflow)
    return svc


@pytest.mark.asyncio
async def test_expose_stores_the_endpoint_in_the_workflow_metadata_column():
    wf = WorkflowDefinition(name="Daily Report", definition={}, workflow_metadata={"keep": 1})
    svc = _service(wf)

    result = await svc.expose_workflow_as_mcp_endpoint(
        str(wf.id), {"endpoint_name": "daily", "auth_required": False})

    assert result["endpoint_name"] == "daily"
    assert result["endpoint_url"] == "/mcp/v1/workflows/daily"
    saved = svc.workflow_service.update_workflow.await_args.kwargs["metadata"]
    assert saved["keep"] == 1
    assert saved["mcp_endpoint"]["enabled"] is True
    assert saved["mcp_endpoint"]["auth_required"] is False


@pytest.mark.asyncio
async def test_expose_works_for_a_workflow_with_no_metadata():
    wf = WorkflowDefinition(name="No Meta", definition={}, workflow_metadata=None)
    svc = _service(wf)

    result = await svc.expose_workflow_as_mcp_endpoint(str(wf.id), {})

    assert result["endpoint_name"] == "no-meta"
    saved = svc.workflow_service.update_workflow.await_args.kwargs["metadata"]
    assert saved["mcp_endpoint"]["endpoint_name"] == "no-meta"
