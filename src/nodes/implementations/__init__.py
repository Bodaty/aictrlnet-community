"""Node implementations."""

from .task_node import TaskNode
from .start_node import StartNode
from .end_node import EndNode
from .decision_node import DecisionNode
from .api_call_node import APICallNode
from .data_source_node import DataSourceNode
from .notification_node import NotificationNode
from .transform_node import TransformNode
from .adapter_node import AdapterNode
from .mcp_node import MCPNode
from .ai_process_node import AIProcessNode
from .iam_node import IAMNode
from .mcp_client_node import MCPClientNode
from .mcp_server_node import MCPServerNode
try:
    from .platform_integration_node import PlatformIntegrationNode
except ImportError as _e:
    # platform_integration_node has unresolved deps in some edition contexts
    # (Business/Enterprise shadow `schemas` and `services` packages). The
    # live app avoids eager-loading this via the edition_nodes.py plugin
    # path; tests can still import sibling nodes without this one.
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "PlatformIntegrationNode unavailable in this context: %s", _e
    )
    PlatformIntegrationNode = None
from .file_process_node import FileProcessNode
from .doc_generation_node import DocGenerationNode
from .browser_automation_node import BrowserAutomationNode
from .approval_stub_node import ApprovalNode

__all__ = [
    "TaskNode",
    "StartNode",
    "EndNode",
    "DecisionNode",
    "APICallNode",
    "DataSourceNode",
    "NotificationNode",
    "TransformNode",
    "AdapterNode",
    "MCPNode",
    "AIProcessNode",
    "IAMNode",
    "MCPClientNode",
    "MCPServerNode",
    "PlatformIntegrationNode",
    "FileProcessNode",
    "DocGenerationNode",
    "BrowserAutomationNode",
    "ApprovalNode",
]