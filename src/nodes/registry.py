"""Node registry for managing node types."""

import logging
from typing import Dict, Type, Optional, List, Any

from .base_node import BaseNode
from .models import NodeType, NodeConfig
from .implementations import (
    TaskNode, AdapterNode, DecisionNode, TransformNode,
    StartNode, EndNode, DataSourceNode,
    AIProcessNode, APICallNode, MCPNode, IAMNode,
    NotificationNode, MCPClientNode, MCPServerNode,
    PlatformIntegrationNode,
    FileProcessNode, DocGenerationNode, BrowserAutomationNode,
)


logger = logging.getLogger(__name__)


class NodeRegistry:
    """Registry for node types and implementations."""
    
    def __init__(self):
        self._node_classes: Dict[NodeType, Type[BaseNode]] = {}
        self._custom_nodes: Dict[str, Type[BaseNode]] = {}
        self._register_default_nodes()
    
    def _register_default_nodes(self):
        """Register default node implementations."""
        # Control flow nodes
        self.register_node_class(NodeType.START, StartNode)
        self.register_node_class(NodeType.END, EndNode)
        self.register_node_class(NodeType.DECISION, DecisionNode)
        
        # Processing nodes
        self.register_node_class(NodeType.TASK, TaskNode)
        self.register_node_class(NodeType.ADAPTER, AdapterNode)
        self.register_node_class(NodeType.TRANSFORM, TransformNode)
        
        # Community Edition nodes
        self.register_custom_node("dataSource", DataSourceNode)
        self.register_custom_node("aiProcess", AIProcessNode)
        self.register_custom_node("apiCall", APICallNode)
        self.register_custom_node("mcp", MCPNode)
        self.register_custom_node("iam", IAMNode)
        self.register_custom_node("notification", NotificationNode)
        
        # MCP Integration nodes
        self.register_custom_node("mcpClient", MCPClientNode)
        self.register_custom_node("mcpServer", MCPServerNode)
        
        # Platform Integration node
        self.register_custom_node("platform", PlatformIntegrationNode)

        # File processing & document generation nodes
        self.register_custom_node("fileProcess", FileProcessNode)
        self.register_custom_node("docGeneration", DocGenerationNode)

        # Browser automation node
        self.register_custom_node("browserAutomation", BrowserAutomationNode)

        # Additional node types can be registered here
        logger.info("Default node types registered")
    
    def register_node_class(
        self,
        node_type: NodeType,
        node_class: Type[BaseNode]
    ):
        """Register a node class for a node type."""
        if not issubclass(node_class, BaseNode):
            raise ValueError(f"{node_class} must be a subclass of BaseNode")
        
        self._node_classes[node_type] = node_class
        logger.info(f"Registered node class {node_class.__name__} for type {node_type}")
    
    def register_custom_node(
        self,
        name: str,
        node_class: Type[BaseNode]
    ):
        """Register a custom node implementation."""
        if not issubclass(node_class, BaseNode):
            raise ValueError(f"{node_class} must be a subclass of BaseNode")
        
        self._custom_nodes[name] = node_class
        logger.info(f"Registered custom node: {name}")
    
    def create_node(self, config: NodeConfig) -> BaseNode:
        """Create a node instance from configuration."""
        # Check for custom node first
        if config.parameters.get("custom_node_type"):
            custom_type = config.parameters["custom_node_type"]
            if custom_type in self._custom_nodes:
                node_class = self._custom_nodes[custom_type]
                return node_class(config)
        
        # Get node class for type
        node_class = self._node_classes.get(config.type)
        if not node_class:
            raise ValueError(f"No implementation for node type: {config.type}")
        
        return node_class(config)
    
    def get_available_node_types(self) -> List[str]:
        """Get list of available node types."""
        types = [nt.value for nt in self._node_classes.keys()]
        types.extend(self._custom_nodes.keys())
        return types
    
    def get_node_info(self, node_type: str) -> Dict[str, Any]:
        """Get information about a node type."""
        # Check if it's a standard type
        try:
            node_type_enum = NodeType(node_type)
            if node_type_enum in self._node_classes:
                node_class = self._node_classes[node_type_enum]
                return {
                    "type": node_type,
                    "class": node_class.__name__,
                    "description": node_class.__doc__
                }
        except ValueError:
            pass
        
        # Check custom nodes
        if node_type in self._custom_nodes:
            node_class = self._custom_nodes[node_type]
            return {
                "type": node_type,
                "class": node_class.__name__,
                "description": node_class.__doc__,
                "custom": True
            }
        
        return None
    
    def list_node_types(self) -> List[Dict[str, Any]]:
        """List all available node types with details."""
        node_types = []
        
        # Add standard node types
        for node_type, node_class in self._node_classes.items():
            node_types.append({
                "id": node_type.value,
                "name": node_type.value.replace("_", " ").title(),
                "category": self._get_node_category(node_type),
                "description": node_class.__doc__ or "",
                "edition": "community"
            })
        
        # Add custom nodes
        for name, node_class in self._custom_nodes.items():
            node_types.append({
                "id": name,
                "name": name.replace("_", " ").title(),
                "category": self._get_custom_node_category(name),
                "description": node_class.__doc__ or "",
                "edition": self._get_node_edition(name)
            })
        
        return node_types
    
    def get_catalog(self) -> Dict[str, Any]:
        """Get full node catalog organized by category."""
        catalog = {
            "control_flow": [],
            "data_processing": [],
            "ai_ml": [],
            "integration": [],
            "quality": [],
            "governance": [],
            "mcp": []
        }
        
        # Categorize standard nodes
        for node_type, node_class in self._node_classes.items():
            category = self._get_node_category(node_type)
            if category in catalog:
                catalog[category].append({
                    "id": node_type.value,
                    "name": node_type.value.replace("_", " ").title(),
                    "description": node_class.__doc__ or "",
                    "edition": "community"
                })
        
        # Categorize custom nodes
        for name, node_class in self._custom_nodes.items():
            category = self._get_custom_node_category(name)
            if category in catalog:
                catalog[category].append({
                    "id": name,
                    "name": name.replace("_", " ").title(),
                    "description": node_class.__doc__ or "",
                    "edition": self._get_node_edition(name)
                })
        
        return catalog
    
    def _get_node_category(self, node_type: NodeType) -> str:
        """Get category for a standard node type."""
        if node_type in [NodeType.START, NodeType.END, NodeType.DECISION]:
            return "control_flow"
        elif node_type in [NodeType.TASK, NodeType.TRANSFORM]:
            return "data_processing"
        elif node_type == NodeType.ADAPTER:
            return "integration"
        return "data_processing"
    
    def _get_custom_node_category(self, name: str) -> str:
        """Get category for a custom node."""
        if name in ["dataSource", "transform", "fileProcess", "docGeneration"]:
            return "data_processing"
        elif name in ["aiProcess"]:
            return "ai_ml"
        elif name in ["apiCall", "iam", "adapter"]:
            return "integration"
        elif name in ["mcp", "mcpClient", "mcpServer"]:
            return "mcp"
        elif name in ["notification"]:
            return "integration"
        elif name in ["browserAutomation"]:
            return "integration"
        return "data_processing"
    
    def _get_node_edition(self, name: str) -> str:
        """Get edition requirement for a node."""
        if name in ["aiProcess", "iam"]:
            return "business"
        elif name in ["mcp"]:
            return "enterprise"
        return "community"


# Global node registry
node_registry = NodeRegistry()