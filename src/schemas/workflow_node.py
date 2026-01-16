"""ReactFlow-compatible node schemas for workflows."""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class PortType(str, Enum):
    """Port types for node connections."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    ANY = "any"


class NodeCategory(str, Enum):
    """Node categories for organization."""
    CONTROL_FLOW = "control_flow"
    DATA_PROCESSING = "data_processing"
    AI_ML = "ai_ml"
    INTEGRATION = "integration"
    QUALITY = "quality"
    GOVERNANCE = "governance"
    HUMAN_INTERACTION = "human_interaction"
    MCP = "mcp"
    INTERNAL_AGENT = "internal_agent"
    EXTERNAL_AGENT = "external_agent"


class InputPort(BaseModel):
    """Input port definition for a node."""
    id: str = Field(..., description="Unique port identifier")
    name: str = Field(..., description="Display name")
    type: PortType = Field(..., description="Data type expected")
    description: Optional[str] = Field(None, description="Port description")
    required: bool = Field(True, description="Whether this input is required")
    default_value: Optional[Any] = Field(None, description="Default value if not connected")
    validation: Optional[Dict[str, Any]] = Field(None, description="Validation rules")


class OutputPort(BaseModel):
    """Output port definition for a node."""
    id: str = Field(..., description="Unique port identifier")
    name: str = Field(..., description="Display name")
    type: PortType = Field(..., description="Data type produced")
    description: Optional[str] = Field(None, description="Port description")
    multiple: bool = Field(False, description="Can connect to multiple inputs")


class NodePosition(BaseModel):
    """Position of a node in the visual editor."""
    x: float
    y: float


class NodeData(BaseModel):
    """Data associated with a workflow node."""
    label: str = Field(..., description="Display label")
    description: Optional[str] = Field(None, description="Node description")
    icon: Optional[str] = Field(None, description="Icon identifier")
    color: Optional[str] = Field(None, description="Node color")
    config: Dict[str, Any] = Field(default_factory=dict, description="Node configuration")
    inputs: List[InputPort] = Field(default_factory=list, description="Input ports")
    outputs: List[OutputPort] = Field(default_factory=list, description="Output ports")
    edition: str = Field("community", description="Minimum edition required")
    custom_node_type: Optional[str] = Field(None, description="Custom node type identifier")


class WorkflowNode(BaseModel):
    """ReactFlow-compatible workflow node."""
    id: str = Field(..., description="Unique node identifier")
    type: str = Field(..., description="Node type")
    position: NodePosition = Field(..., description="Node position")
    data: NodeData = Field(..., description="Node data")
    draggable: bool = Field(True, description="Can be dragged")
    selectable: bool = Field(True, description="Can be selected")
    connectable: bool = Field(True, description="Can be connected")
    deletable: bool = Field(True, description="Can be deleted")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "node_1",
                "type": "ai_model",
                "position": {"x": 100, "y": 100},
                "data": {
                    "label": "GPT-4 Analysis",
                    "description": "Analyze text using GPT-4",
                    "icon": "brain",
                    "color": "#9C27B0",
                    "config": {
                        "model_id": "gpt-4",
                        "temperature": 0.7
                    },
                    "inputs": [
                        {
                            "id": "text",
                            "name": "Input Text",
                            "type": "string",
                            "required": True
                        }
                    ],
                    "outputs": [
                        {
                            "id": "result",
                            "name": "Analysis Result",
                            "type": "object"
                        }
                    ],
                    "edition": "business"
                }
            }
        }


class EdgeData(BaseModel):
    """Data associated with a workflow edge."""
    label: Optional[str] = Field(None, description="Edge label")
    condition: Optional[str] = Field(None, description="Condition for conditional edges")
    transform: Optional[str] = Field(None, description="Data transformation")
    style: Optional[Dict[str, Any]] = Field(None, description="Edge styling")


class WorkflowEdge(BaseModel):
    """ReactFlow-compatible workflow edge."""
    id: str = Field(..., description="Unique edge identifier")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    sourceHandle: Optional[str] = Field(None, description="Source port ID")
    targetHandle: Optional[str] = Field(None, description="Target port ID")
    type: str = Field("default", description="Edge type")
    animated: bool = Field(False, description="Animated edge")
    data: Optional[EdgeData] = Field(None, description="Edge data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "edge_1",
                "source": "node_1",
                "target": "node_2",
                "sourceHandle": "result",
                "targetHandle": "input",
                "type": "default",
                "animated": True,
                "data": {
                    "label": "On Success",
                    "condition": "result.score > 0.8"
                }
            }
        }


class NodeMetadata(BaseModel):
    """Metadata for a node type in the catalog."""
    type: str = Field(..., description="Node type identifier")
    category: NodeCategory = Field(..., description="Node category")
    label: str = Field(..., description="Display name")
    description: str = Field(..., description="Detailed description")
    icon: Optional[str] = Field(None, description="Icon identifier")
    color: Optional[str] = Field(None, description="Default color")
    edition: str = Field("community", description="Minimum edition required")
    inputs: List[InputPort] = Field(default_factory=list, description="Input port definitions")
    outputs: List[OutputPort] = Field(default_factory=list, description="Output port definitions")
    config_schema: Optional[Dict[str, Any]] = Field(None, description="Configuration schema")
    capabilities: List[str] = Field(default_factory=list, description="Node capabilities")
    resource_requirements: Optional[Dict[str, Any]] = Field(None, description="Resource requirements")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "quality_check",
                "category": "quality",
                "label": "Data Quality Check",
                "description": "Validate data quality using ISO 25012 standards",
                "icon": "check-circle",
                "color": "#4CAF50",
                "edition": "business",
                "inputs": [
                    {
                        "id": "data",
                        "name": "Data",
                        "type": "any",
                        "required": True
                    }
                ],
                "outputs": [
                    {
                        "id": "validated_data",
                        "name": "Validated Data",
                        "type": "any"
                    },
                    {
                        "id": "quality_report",
                        "name": "Quality Report",
                        "type": "object"
                    }
                ],
                "capabilities": ["iso_25012", "validation", "reporting"]
            }
        }


class WorkflowCatalog(BaseModel):
    """Complete workflow node catalog."""
    categories: Dict[str, List[NodeMetadata]] = Field(..., description="Nodes organized by category")
    total_nodes: int = Field(..., description="Total number of available nodes")
    edition_summary: Dict[str, int] = Field(..., description="Node count by edition")
    generated_at: str = Field(..., description="Catalog generation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "categories": {
                    "ai_ml": [
                        {
                            "type": "ai_model_gpt4",
                            "category": "ai_ml",
                            "label": "GPT-4",
                            "description": "OpenAI GPT-4 model",
                            "edition": "business"
                        }
                    ]
                },
                "total_nodes": 45,
                "edition_summary": {
                    "community": 20,
                    "business": 15,
                    "enterprise": 10
                },
                "generated_at": "2025-01-12T10:00:00Z"
            }
        }


class NodeExecutionUpdate(BaseModel):
    """Update for node execution status."""
    node_id: str
    status: str
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    message: Optional[str] = None


class WorkflowExecutionUpdate(BaseModel):
    """Real-time workflow execution update."""
    execution_id: str
    workflow_id: str
    type: str = Field(..., description="Update type: started, node_update, completed, failed")
    timestamp: str
    data: Union[NodeExecutionUpdate, Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "execution_id": "123e4567-e89b-12d3-a456-426614174000",
                "workflow_id": "workflow_1",
                "type": "node_update",
                "timestamp": "2025-01-12T10:00:00Z",
                "data": {
                    "node_id": "node_1",
                    "status": "completed",
                    "outputs": {"result": "success"}
                }
            }
        }


class WorkflowValidationResult(BaseModel):
    """Result of workflow validation."""
    is_valid: bool
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    node_count: int
    edge_count: int
    estimated_execution_time: Optional[float] = None
    resource_requirements: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "errors": [],
                "warnings": [
                    {
                        "node_id": "node_3",
                        "message": "This node has no error handling configured"
                    }
                ],
                "node_count": 5,
                "edge_count": 4,
                "estimated_execution_time": 30.5,
                "resource_requirements": {
                    "compute": "2 vCPU",
                    "memory": "4GB",
                    "ml_models": ["gpt-4"]
                }
            }
        }