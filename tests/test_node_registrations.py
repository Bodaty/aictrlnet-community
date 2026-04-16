"""Regression tests for node registry contract bugs.

Covers:
- DataSourceNode.validate_config() defaults source_type to "static"
  (matches execute() runtime default — was inconsistent before).
- Community ApprovalNode stub raises a clear edition-required error
  instead of falling back to TaskNode or "no implementation".
"""

import pytest

from nodes.models import NodeConfig, NodeType
from nodes.registry import get_node_registry
from nodes.implementations.data_source_node import DataSourceNode
from nodes.implementations.approval_stub_node import ApprovalNode as CommunityApprovalStub


class TestDataSourceValidateDefault:
    def test_validate_config_no_source_type_defaults_to_static(self):
        config = NodeConfig(
            name="ds-no-source-type",
            type=NodeType.TASK,
            parameters={"data": [{"k": "v"}]},
        )
        node = DataSourceNode(config)
        assert node.validate_config() is True

    def test_validate_config_explicit_static_still_works(self):
        config = NodeConfig(
            name="ds-explicit-static",
            type=NodeType.TASK,
            parameters={"source_type": "static", "data": [1, 2, 3]},
        )
        node = DataSourceNode(config)
        assert node.validate_config() is True

    def test_validate_config_file_still_requires_file_path(self):
        config = NodeConfig(
            name="ds-file-missing-path",
            type=NodeType.TASK,
            parameters={"source_type": "file"},
        )
        node = DataSourceNode(config)
        with pytest.raises(ValueError, match="file_path"):
            node.validate_config()


class TestCommunityApprovalStub:
    def test_registry_returns_stub_for_approval_enum(self):
        registry = get_node_registry()
        config = NodeConfig(
            name="approval-test",
            type=NodeType.APPROVAL,
            parameters={},
        )
        node = registry.create_node(config)
        assert isinstance(node, CommunityApprovalStub)

    def test_registry_returns_stub_for_approval_custom_alias(self):
        registry = get_node_registry()
        config = NodeConfig(
            name="approval-custom-test",
            type=NodeType.TASK,
            parameters={"custom_node_type": "approval"},
        )
        node = registry.create_node(config)
        assert isinstance(node, CommunityApprovalStub)

    def test_registry_returns_stub_for_approval_request_alias(self):
        registry = get_node_registry()
        config = NodeConfig(
            name="approval-request-test",
            type=NodeType.TASK,
            parameters={"custom_node_type": "approvalRequest"},
        )
        node = registry.create_node(config)
        assert isinstance(node, CommunityApprovalStub)

    @pytest.mark.asyncio
    async def test_stub_execute_raises_edition_required_error(self):
        config = NodeConfig(
            name="approval-stub-exec",
            type=NodeType.APPROVAL,
            parameters={},
        )
        stub = CommunityApprovalStub(config)
        with pytest.raises(RuntimeError, match="Business Edition"):
            await stub.execute({}, {})
