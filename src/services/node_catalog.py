"""Dynamic node catalog service for workflows."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging

from models.community_complete import Adapter, MCPTool, MCPServer
from models.iam import IAMAgent
from schemas.workflow_node import (
    NodeMetadata, NodeCategory, InputPort, OutputPort,
    PortType, WorkflowCatalog
)
from services.adapter import AdapterService
from services.iam import IAMService
from services.mcp import MCPService
from adapters.implementations.ai.ml_service_adapter import MLServiceAdapter
from adapters.models import AdapterConfig, AdapterCategory, Edition

logger = logging.getLogger(__name__)


class DynamicNodeCatalogService:
    """Service for generating dynamic workflow node catalog."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.adapter_service = AdapterService(db)
        self.iam_service = IAMService(db)
        self.mcp_service = MCPService(db)
        
        # Initialize ML Service adapter
        self.ml_adapter = MLServiceAdapter(
            AdapterConfig(
                name="node-catalog-ml",
                category=AdapterCategory.AI,
                version="1.0.0",
                description="ML Service adapter for node catalog",
                required_edition=Edition.COMMUNITY,  # Available in Community with limited features
                base_url="http://ml-service:8003",
                timeout_seconds=30,
                custom_config={"discovery_only": True}  # Community uses discovery mode
            )
        )
    
    async def get_catalog(
        self,
        tenant_id: str,
        edition: str = "community"
    ) -> WorkflowCatalog:
        """Get complete node catalog for tenant and edition."""
        catalog = {
            "control_flow": await self._get_control_flow_nodes(edition),
            "data_processing": await self._get_data_processing_nodes(edition),
            "ai_ml": await self._get_ai_ml_nodes(tenant_id, edition),
            "integration": await self._get_integration_nodes(tenant_id, edition),
            "quality": await self._get_quality_nodes(edition),
            "governance": await self._get_governance_nodes(edition),
            "human_interaction": await self._get_human_interaction_nodes(edition),
            "mcp": await self._get_mcp_nodes(tenant_id, edition),
            "internal_agent": await self._get_internal_agent_nodes(tenant_id, edition),
            "external_agent": await self._get_external_agent_nodes(tenant_id, edition)
        }
        
        # Calculate statistics
        total_nodes = sum(len(nodes) for nodes in catalog.values())
        edition_summary = self._calculate_edition_summary(catalog)
        
        return WorkflowCatalog(
            categories=catalog,
            total_nodes=total_nodes,
            edition_summary=edition_summary,
            generated_at=datetime.utcnow().isoformat() + "Z"
        )
    
    async def _get_control_flow_nodes(self, edition: str) -> List[NodeMetadata]:
        """Get control flow nodes."""
        nodes = [
            NodeMetadata(
                type="start",
                category=NodeCategory.CONTROL_FLOW,
                label="Start",
                description="Workflow entry point",
                icon="play-circle",
                color="#4CAF50",
                edition="community",
                outputs=[
                    OutputPort(id="output", name="Start Signal", type=PortType.ANY)
                ]
            ),
            NodeMetadata(
                type="end",
                category=NodeCategory.CONTROL_FLOW,
                label="End",
                description="Workflow completion",
                icon="stop-circle",
                color="#F44336",
                edition="community",
                inputs=[
                    InputPort(id="input", name="Final Data", type=PortType.ANY)
                ]
            ),
            NodeMetadata(
                type="condition",
                category=NodeCategory.CONTROL_FLOW,
                label="Condition",
                description="Conditional branching",
                icon="git-branch",
                color="#FF9800",
                edition="community",
                inputs=[
                    InputPort(id="data", name="Data", type=PortType.ANY),
                    InputPort(id="condition", name="Condition", type=PortType.STRING)
                ],
                outputs=[
                    OutputPort(id="true", name="True Branch", type=PortType.ANY),
                    OutputPort(id="false", name="False Branch", type=PortType.ANY)
                ]
            )
        ]
        
        if edition in ["business", "enterprise"]:
            nodes.extend([
                NodeMetadata(
                    type="parallel",
                    category=NodeCategory.CONTROL_FLOW,
                    label="Parallel",
                    description="Execute branches in parallel",
                    icon="git-merge",
                    color="#3F51B5",
                    edition="business",
                    inputs=[
                        InputPort(id="input", name="Input Data", type=PortType.ANY)
                    ],
                    outputs=[
                        OutputPort(id="branch1", name="Branch 1", type=PortType.ANY),
                        OutputPort(id="branch2", name="Branch 2", type=PortType.ANY),
                        OutputPort(id="branch3", name="Branch 3", type=PortType.ANY)
                    ]
                ),
                NodeMetadata(
                    type="loop",
                    category=NodeCategory.CONTROL_FLOW,
                    label="Loop",
                    description="Iterate over items",
                    icon="repeat",
                    color="#9C27B0",
                    edition="business",
                    inputs=[
                        InputPort(id="items", name="Items", type=PortType.ARRAY),
                        InputPort(id="parallel", name="Parallel", type=PortType.BOOLEAN, required=False)
                    ],
                    outputs=[
                        OutputPort(id="results", name="Results", type=PortType.ARRAY)
                    ]
                )
            ])
        
        return nodes
    
    async def _get_data_processing_nodes(self, edition: str) -> List[NodeMetadata]:
        """Get data processing nodes."""
        nodes = [
            NodeMetadata(
                type="data_input",
                category=NodeCategory.DATA_PROCESSING,
                label="Data Input",
                description="Input data from various sources",
                icon="database",
                color="#2196F3",
                edition="community",
                inputs=[
                    InputPort(id="source", name="Source", type=PortType.STRING),
                    InputPort(id="config", name="Config", type=PortType.OBJECT, required=False)
                ],
                outputs=[
                    OutputPort(id="data", name="Data", type=PortType.ANY),
                    OutputPort(id="metadata", name="Metadata", type=PortType.OBJECT)
                ]
            ),
            NodeMetadata(
                type="transform",
                category=NodeCategory.DATA_PROCESSING,
                label="Transform",
                description="Transform data using custom logic",
                icon="transform",
                color="#00BCD4",
                edition="community",
                inputs=[
                    InputPort(id="data", name="Input Data", type=PortType.ANY),
                    InputPort(id="transform", name="Transform Logic", type=PortType.STRING)
                ],
                outputs=[
                    OutputPort(id="data", name="Transformed Data", type=PortType.ANY)
                ]
            )
        ]
        
        if edition in ["business", "enterprise"]:
            nodes.append(
                NodeMetadata(
                    type="aggregate",
                    category=NodeCategory.DATA_PROCESSING,
                    label="Aggregate",
                    description="Aggregate data from multiple sources",
                    icon="layers",
                    color="#795548",
                    edition="business",
                    inputs=[
                        InputPort(id="data1", name="Data 1", type=PortType.ANY),
                        InputPort(id="data2", name="Data 2", type=PortType.ANY),
                        InputPort(id="data3", name="Data 3", type=PortType.ANY, required=False)
                    ],
                    outputs=[
                        OutputPort(id="aggregated", name="Aggregated Data", type=PortType.ANY)
                    ]
                )
            )
        
        return nodes
    
    async def _get_ai_ml_nodes(self, tenant_id: str, edition: str) -> List[NodeMetadata]:
        """Get AI/ML nodes including available models."""
        nodes = []
        
        if edition in ["business", "enterprise"]:
            # Static AI nodes
            nodes.extend([
                NodeMetadata(
                    type="ai_process",
                    category=NodeCategory.AI_ML,
                    label="AI Process",
                    description="Generic AI processing node",
                    icon="brain",
                    color="#9C27B0",
                    edition="business",
                    inputs=[
                        InputPort(id="input", name="Input", type=PortType.ANY),
                        InputPort(id="model", name="Model", type=PortType.STRING),
                        InputPort(id="parameters", name="Parameters", type=PortType.OBJECT, required=False)
                    ],
                    outputs=[
                        OutputPort(id="result", name="Result", type=PortType.ANY),
                        OutputPort(id="confidence", name="Confidence", type=PortType.NUMBER)
                    ]
                ),
                NodeMetadata(
                    type="agent_task",
                    category=NodeCategory.AI_ML,
                    label="Agent Task",
                    description="Assign task to an AI agent",
                    icon="robot",
                    color="#3F51B5",
                    edition="business",
                    inputs=[
                        InputPort(id="agent_id", name="Agent ID", type=PortType.STRING),
                        InputPort(id="task", name="Task", type=PortType.STRING),
                        InputPort(id="context", name="Context", type=PortType.OBJECT, required=False)
                    ],
                    outputs=[
                        OutputPort(id="result", name="Result", type=PortType.ANY),
                        OutputPort(id="agent_response", name="Agent Response", type=PortType.OBJECT)
                    ]
                )
            ])
            
            # Dynamic AI model nodes
            try:
                # Use ML adapter to get models (Community has limited access)
                result = await self.ml_adapter.execute({
                    "operation": "get_models",
                    "data": {}
                })
                models = result.data.get("models", []) if result.status == "success" else []
                for model in models:
                    nodes.append(
                        NodeMetadata(
                            type=f"ai_model_{model['id']}",
                            category=NodeCategory.AI_ML,
                            label=f"AI: {model['name']}",
                            description=model.get('description', f"Use {model['name']} for processing"),
                            icon="ai-model",
                            color="#9C27B0",
                            edition="business",
                            inputs=[
                                InputPort(id="input", name="Input", type=PortType.ANY),
                                InputPort(id="prompt", name="Prompt", type=PortType.STRING, required=False),
                                InputPort(id="config", name="Config", type=PortType.OBJECT, required=False)
                            ],
                            outputs=[
                                OutputPort(id="output", name="Output", type=PortType.ANY),
                                OutputPort(id="metadata", name="Metadata", type=PortType.OBJECT)
                            ],
                            capabilities=model.get('capabilities', [])
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not fetch ML models: {e}")
        
        return nodes
    
    async def _get_integration_nodes(self, tenant_id: str, edition: str) -> List[NodeMetadata]:
        """Get integration nodes including adapters."""
        nodes = [
            NodeMetadata(
                type="api_call",
                category=NodeCategory.INTEGRATION,
                label="API Call",
                description="Make HTTP API calls",
                icon="globe",
                color="#607D8B",
                edition="community",
                inputs=[
                    InputPort(id="url", name="URL", type=PortType.STRING),
                    InputPort(id="method", name="Method", type=PortType.STRING),
                    InputPort(id="headers", name="Headers", type=PortType.OBJECT, required=False),
                    InputPort(id="body", name="Body", type=PortType.ANY, required=False)
                ],
                outputs=[
                    OutputPort(id="response", name="Response", type=PortType.ANY),
                    OutputPort(id="status", name="Status Code", type=PortType.NUMBER)
                ]
            )
        ]
        
        # Dynamic adapter nodes
        try:
            adapters_result = await self.db.execute(
                select(Adapter).where(Adapter.enabled == True)
            )
            adapters = adapters_result.scalars().all()
            
            for adapter in adapters:
                nodes.append(
                    NodeMetadata(
                        type=f"adapter_{adapter.id}",
                        category=NodeCategory.INTEGRATION,
                        label=f"Adapter: {adapter.name}",
                        description=adapter.description or f"Connect to {adapter.name}",
                        icon="plug",
                        color="#FF9800",
                        edition="community",
                        inputs=[
                            InputPort(id="action", name="Action", type=PortType.STRING),
                            InputPort(id="payload", name="Payload", type=PortType.OBJECT)
                        ],
                        outputs=[
                            OutputPort(id="result", name="Result", type=PortType.ANY),
                            OutputPort(id="status", name="Status", type=PortType.STRING)
                        ],
                        capabilities=adapter.config.get('capabilities', []) if adapter.config else []
                    )
                )
        except Exception as e:
            logger.warning(f"Could not fetch adapters: {e}")
        
        return nodes
    
    async def _get_quality_nodes(self, edition: str) -> List[NodeMetadata]:
        """Get quality control nodes."""
        nodes = []
        
        if edition in ["business", "enterprise"]:
            nodes.extend([
                NodeMetadata(
                    type="quality_check",
                    category=NodeCategory.QUALITY,
                    label="Quality Check",
                    description="Automated quality validation",
                    icon="check-circle",
                    color="#4CAF50",
                    edition="business",
                    inputs=[
                        InputPort(id="data", name="Data", type=PortType.ANY),
                        InputPort(id="checks", name="Checks", type=PortType.ARRAY)
                    ],
                    outputs=[
                        OutputPort(id="data", name="Validated Data", type=PortType.ANY),
                        OutputPort(id="report", name="Quality Report", type=PortType.OBJECT),
                        OutputPort(id="passed", name="Passed", type=PortType.BOOLEAN)
                    ]
                ),
                NodeMetadata(
                    type="manual_review",
                    category=NodeCategory.QUALITY,
                    label="Manual Review",
                    description="Human review checkpoint",
                    icon="user-check",
                    color="#FF5722",
                    edition="business",
                    inputs=[
                        InputPort(id="data", name="Data", type=PortType.ANY),
                        InputPort(id="instructions", name="Instructions", type=PortType.STRING),
                        InputPort(id="reviewer", name="Reviewer", type=PortType.STRING, required=False)
                    ],
                    outputs=[
                        OutputPort(id="approved", name="Approved", type=PortType.BOOLEAN),
                        OutputPort(id="feedback", name="Feedback", type=PortType.STRING),
                        OutputPort(id="reviewer_id", name="Reviewer ID", type=PortType.STRING)
                    ]
                ),
                NodeMetadata(
                    type="data_quality_iso",
                    category=NodeCategory.QUALITY,
                    label="ISO 25012 Quality",
                    description="Data quality validation using ISO 25012 standards",
                    icon="shield-check",
                    color="#2196F3",
                    edition="business",
                    inputs=[
                        InputPort(id="data", name="Data", type=PortType.ANY),
                        InputPort(id="dimensions", name="Dimensions", type=PortType.ARRAY),
                        InputPort(id="rules", name="Validation Rules", type=PortType.ARRAY, required=False)
                    ],
                    outputs=[
                        OutputPort(id="validated_data", name="Validated Data", type=PortType.ANY),
                        OutputPort(id="quality_score", name="Quality Score", type=PortType.NUMBER),
                        OutputPort(id="issues", name="Issues", type=PortType.ARRAY)
                    ],
                    capabilities=["iso_25012", "ml_enhanced"]
                )
            ])
        
        return nodes
    
    async def _get_governance_nodes(self, edition: str) -> List[NodeMetadata]:
        """Get governance nodes."""
        nodes = []
        
        if edition == "enterprise":
            nodes.extend([
                NodeMetadata(
                    type="policy_check",
                    category=NodeCategory.GOVERNANCE,
                    label="Policy Check",
                    description="Verify compliance with policies",
                    icon="shield",
                    color="#F44336",
                    edition="enterprise",
                    inputs=[
                        InputPort(id="data", name="Data", type=PortType.ANY),
                        InputPort(id="policy_id", name="Policy ID", type=PortType.STRING),
                        InputPort(id="enforcement_level", name="Enforcement Level", type=PortType.STRING)
                    ],
                    outputs=[
                        OutputPort(id="compliant", name="Compliant", type=PortType.BOOLEAN),
                        OutputPort(id="violations", name="Violations", type=PortType.ARRAY),
                        OutputPort(id="recommendations", name="Recommendations", type=PortType.ARRAY)
                    ]
                ),
                NodeMetadata(
                    type="risk_assessment",
                    category=NodeCategory.GOVERNANCE,
                    label="Risk Assessment",
                    description="ML-based risk scoring",
                    icon="warning",
                    color="#FF9800",
                    edition="enterprise",
                    inputs=[
                        InputPort(id="task_data", name="Task Data", type=PortType.OBJECT),
                        InputPort(id="context", name="Context", type=PortType.OBJECT)
                    ],
                    outputs=[
                        OutputPort(id="risk_score", name="Risk Score", type=PortType.NUMBER),
                        OutputPort(id="risk_factors", name="Risk Factors", type=PortType.ARRAY),
                        OutputPort(id="mitigation", name="Mitigation", type=PortType.OBJECT)
                    ],
                    capabilities=["ml_powered", "real_time"]
                ),
                NodeMetadata(
                    type="audit_log",
                    category=NodeCategory.GOVERNANCE,
                    label="Audit Log",
                    description="Create audit trail entry",
                    icon="file-text",
                    color="#607D8B",
                    edition="enterprise",
                    inputs=[
                        InputPort(id="action", name="Action", type=PortType.STRING),
                        InputPort(id="details", name="Details", type=PortType.OBJECT),
                        InputPort(id="metadata", name="Metadata", type=PortType.OBJECT, required=False)
                    ],
                    outputs=[
                        OutputPort(id="audit_id", name="Audit ID", type=PortType.STRING),
                        OutputPort(id="timestamp", name="Timestamp", type=PortType.STRING)
                    ]
                )
            ])
        
        return nodes
    
    async def _get_human_interaction_nodes(self, edition: str) -> List[NodeMetadata]:
        """Get human interaction nodes."""
        nodes = []
        
        if edition in ["business", "enterprise"]:
            nodes.extend([
                NodeMetadata(
                    type="human_task",
                    category=NodeCategory.HUMAN_INTERACTION,
                    label="Human Task",
                    description="Create task for human completion",
                    icon="user",
                    color="#9E9E9E",
                    edition="business",
                    inputs=[
                        InputPort(id="title", name="Title", type=PortType.STRING),
                        InputPort(id="description", name="Description", type=PortType.STRING),
                        InputPort(id="data", name="Task Data", type=PortType.ANY),
                        InputPort(id="assignee", name="Assignee", type=PortType.STRING, required=False)
                    ],
                    outputs=[
                        OutputPort(id="result", name="Result", type=PortType.ANY),
                        OutputPort(id="completed_by", name="Completed By", type=PortType.STRING),
                        OutputPort(id="completion_time", name="Completion Time", type=PortType.STRING)
                    ]
                ),
                NodeMetadata(
                    type="approval",
                    category=NodeCategory.HUMAN_INTERACTION,
                    label="Approval",
                    description="Request approval from authorized user",
                    icon="check-square",
                    color="#4CAF50",
                    edition="business",
                    inputs=[
                        InputPort(id="request", name="Request", type=PortType.OBJECT),
                        InputPort(id="approver_role", name="Approver Role", type=PortType.STRING),
                        InputPort(id="timeout_hours", name="Timeout Hours", type=PortType.NUMBER, required=False)
                    ],
                    outputs=[
                        OutputPort(id="approved", name="Approved", type=PortType.BOOLEAN),
                        OutputPort(id="approver", name="Approver", type=PortType.STRING),
                        OutputPort(id="comments", name="Comments", type=PortType.STRING)
                    ]
                )
            ])
        
        if edition == "enterprise":
            nodes.append(
                NodeMetadata(
                    type="parallel_human_ai",
                    category=NodeCategory.HUMAN_INTERACTION,
                    label="Parallel Human-AI",
                    description="Run human and AI tasks in parallel",
                    icon="users",
                    color="#3F51B5",
                    edition="enterprise",
                    inputs=[
                        InputPort(id="task", name="Task", type=PortType.OBJECT),
                        InputPort(id="ai_model", name="AI Model", type=PortType.STRING),
                        InputPort(id="human_assignee", name="Human Assignee", type=PortType.STRING)
                    ],
                    outputs=[
                        OutputPort(id="human_result", name="Human Result", type=PortType.ANY),
                        OutputPort(id="ai_result", name="AI Result", type=PortType.ANY),
                        OutputPort(id="agreement_score", name="Agreement Score", type=PortType.NUMBER)
                    ]
                )
            )
        
        return nodes
    
    async def _get_mcp_nodes(self, tenant_id: str, edition: str) -> List[NodeMetadata]:
        """Get MCP (Model Context Protocol) nodes."""
        nodes = []
        
        # MCP Integration nodes (Phase 3)
        nodes.extend([
            NodeMetadata(
                type="mcpClient",
                category=NodeCategory.MCP,
                label="MCP Client",
                description="Connect to external MCP servers and consume their services",
                icon="cloud-download",
                color="#1976D2",
                edition="community",
                inputs=[
                    InputPort(id="messages", name="Messages", type=PortType.ARRAY, required=False),
                    InputPort(id="content", name="Content", type=PortType.STRING, required=False),
                    InputPort(id="parameters", name="Parameters", type=PortType.OBJECT, required=False)
                ],
                outputs=[
                    OutputPort(id="result", name="Result", type=PortType.ANY),
                    OutputPort(id="error", name="Error", type=PortType.STRING)
                ],
                parameters={
                    "mcp_server_url": {"type": "string", "required": True, "description": "URL of external MCP server"},
                    "api_key": {"type": "string", "required": False, "description": "API key for authentication"},
                    "server_name": {"type": "string", "default": "external_mcp", "description": "Name to identify server"},
                    "operation": {"type": "select", "default": "message", "options": ["message", "quality", "workflow", "tool", "custom"]},
                    "timeout": {"type": "number", "default": 30, "description": "Request timeout in seconds"}
                }
            ),
            NodeMetadata(
                type="mcpServer",
                category=NodeCategory.MCP,
                label="MCP Server",
                description="Expose workflow as an MCP endpoint for external clients",
                icon="cloud-upload",
                color="#388E3C",
                edition="community",
                inputs=[
                    InputPort(id="trigger", name="Trigger", type=PortType.ANY, required=False)
                ],
                outputs=[
                    OutputPort(id="request", name="Request Data", type=PortType.OBJECT),
                    OutputPort(id="status", name="Status", type=PortType.STRING)
                ],
                parameters={
                    "endpoint_name": {"type": "string", "required": True, "description": "Name of MCP endpoint to expose"},
                    "allowed_operations": {"type": "array", "default": ["execute"], "description": "Allowed operations"},
                    "auth_required": {"type": "boolean", "default": True, "description": "Require authentication"},
                    "mode": {"type": "select", "default": "single", "options": ["single", "continuous", "webhook"]},
                    "timeout": {"type": "number", "default": 300, "description": "Timeout in seconds"},
                    "max_requests": {"type": "number", "default": 10, "description": "Max requests (continuous mode)"}
                }
            )
        ])
        
        # Static MCP operation nodes
        nodes.extend([
            NodeMetadata(
                type="mcp_tool_execute",
                category=NodeCategory.MCP,
                label="Execute MCP Tool",
                description="Execute a tool from registered MCP servers",
                icon="tool",
                color="#00BCD4",
                edition="community",
                inputs=[
                    InputPort(id="server_name", name="Server Name", type=PortType.STRING, required=False),
                    InputPort(id="tool_name", name="Tool Name", type=PortType.STRING),
                    InputPort(id="tool_args", name="Arguments", type=PortType.OBJECT)
                ],
                outputs=[
                    OutputPort(id="result", name="Result", type=PortType.ANY),
                    OutputPort(id="server", name="Server Used", type=PortType.STRING)
                ]
            ),
            NodeMetadata(
                type="mcp_discover",
                category=NodeCategory.MCP,
                label="Discover MCP Tools",
                description="Find available tools across MCP servers",
                icon="search",
                color="#00BCD4",
                edition="community",
                inputs=[
                    InputPort(id="filter", name="Filter", type=PortType.OBJECT, required=False)
                ],
                outputs=[
                    OutputPort(id="tools", name="Available Tools", type=PortType.ARRAY),
                    OutputPort(id="count", name="Tool Count", type=PortType.NUMBER)
                ]
            )
        ])
        
        # Dynamic MCP tool nodes
        try:
            # Get MCP servers and tools
            servers_result = await self.db.execute(
                select(MCPServer).where(MCPServer.status == "active")
            )
            servers = servers_result.scalars().all()
            
            for server in servers:
                # Get tools for this server
                tools_result = await self.db.execute(
                    select(MCPTool).where(
                        and_(
                            MCPTool.server_id == server.id,
                            MCPTool.is_available == True
                        )
                    )
                )
                tools = tools_result.scalars().all()
                
                for tool in tools:
                    nodes.append(
                        NodeMetadata(
                            type=f"mcp_{server.name}_{tool.name}",
                            category=NodeCategory.MCP,
                            label=f"{server.name}: {tool.name}",
                            description=tool.description or f"Execute {tool.name} via MCP",
                            icon="mcp-tool",
                            color="#00BCD4",
                            edition="community",
                            inputs=self._parse_mcp_tool_inputs(tool),
                            outputs=[
                                OutputPort(id="result", name="Result", type=PortType.ANY),
                                OutputPort(id="execution_time", name="Execution Time", type=PortType.NUMBER)
                            ],
                            config_schema={
                                "server_name": server.name,
                                "tool_name": tool.name
                            }
                        )
                    )
        except Exception as e:
            logger.warning(f"Could not fetch MCP tools: {e}")
        
        return nodes
    
    async def _get_internal_agent_nodes(self, tenant_id: str, edition: str) -> List[NodeMetadata]:
        """Get internal agent (IAM) nodes."""
        nodes = []
        
        try:
            # Get IAM agents
            agents = await self.iam_service.get_agents(
                tenant_id=tenant_id,
                include_inactive=False
            )
            
            for agent in agents:
                nodes.append(
                    NodeMetadata(
                        type=f"iam_agent_{agent.id}",
                        category=NodeCategory.INTERNAL_AGENT,
                        label=f"IAM: {agent.name}",
                        description=agent.description or f"Send message to {agent.name}",
                        icon="message-circle",
                        color="#9C27B0",
                        edition="community",
                        inputs=[
                            InputPort(id="message", name="Message", type=PortType.ANY),
                            InputPort(id="priority", name="Priority", type=PortType.STRING, 
                                     required=False, default_value="normal")
                        ],
                        outputs=[
                            OutputPort(id="response", name="Response", type=PortType.ANY),
                            OutputPort(id="message_id", name="Message ID", type=PortType.STRING)
                        ],
                        capabilities=agent.capabilities
                    )
                )
        except Exception as e:
            logger.warning(f"Could not fetch IAM agents: {e}")
        
        return nodes
    
    async def _get_external_agent_nodes(self, tenant_id: str, edition: str) -> List[NodeMetadata]:
        """Get external agent nodes (e.g., Google A2A)."""
        nodes = []
        
        if edition in ["business", "enterprise"]:
            # This would integrate with Google A2A when migrated from Flask
            nodes.append(
                NodeMetadata(
                    type="google_a2a",
                    category=NodeCategory.EXTERNAL_AGENT,
                    label="Google A2A Agent",
                    description="Execute task via Google Agent-to-Agent protocol",
                    icon="google",
                    color="#4285F4",
                    edition="business",
                    inputs=[
                        InputPort(id="task", name="Task", type=PortType.OBJECT),
                        InputPort(id="agent_id", name="Agent ID", type=PortType.STRING, required=False),
                        InputPort(id="timeout", name="Timeout", type=PortType.NUMBER, required=False)
                    ],
                    outputs=[
                        OutputPort(id="result", name="Result", type=PortType.ANY),
                        OutputPort(id="execution_id", name="Execution ID", type=PortType.STRING)
                    ]
                )
            )
        
        return nodes
    
    def _parse_mcp_tool_inputs(self, tool: MCPTool) -> List[InputPort]:
        """Parse MCP tool input schema to input ports."""
        inputs = []
        
        if tool.input_schema:
            schema = tool.input_schema
            if isinstance(schema, dict) and "properties" in schema:
                for prop_name, prop_def in schema["properties"].items():
                    port_type = self._json_schema_to_port_type(prop_def.get("type", "any"))
                    required = prop_name in schema.get("required", [])
                    
                    inputs.append(
                        InputPort(
                            id=prop_name,
                            name=prop_name.replace("_", " ").title(),
                            type=port_type,
                            description=prop_def.get("description"),
                            required=required
                        )
                    )
        
        # Default input if no schema
        if not inputs:
            inputs.append(
                InputPort(id="input", name="Input", type=PortType.ANY)
            )
        
        return inputs
    
    def _json_schema_to_port_type(self, json_type: str) -> PortType:
        """Convert JSON schema type to port type."""
        mapping = {
            "string": PortType.STRING,
            "number": PortType.NUMBER,
            "integer": PortType.NUMBER,
            "boolean": PortType.BOOLEAN,
            "object": PortType.OBJECT,
            "array": PortType.ARRAY
        }
        return mapping.get(json_type, PortType.ANY)
    
    def _calculate_edition_summary(self, catalog: Dict[str, List[NodeMetadata]]) -> Dict[str, int]:
        """Calculate node count by edition."""
        summary = {
            "community": 0,
            "business": 0,
            "enterprise": 0
        }
        
        for nodes in catalog.values():
            for node in nodes:
                if node.edition == "community":
                    summary["community"] += 1
                elif node.edition == "business":
                    summary["business"] += 1
                elif node.edition == "enterprise":
                    summary["enterprise"] += 1
        
        return summary
    
    async def validate_workflow_definition(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        edition: str
    ) -> Dict[str, Any]:
        """Validate a workflow definition."""
        errors = []
        warnings = []
        
        # Check node availability for edition
        for node in nodes:
            node_type = node.get("type")
            if node_type:
                # Check if node is available in catalog
                catalog = await self.get_catalog("default", edition)
                found = False
                
                for category_nodes in catalog.categories.values():
                    if any(n.type == node_type for n in category_nodes):
                        found = True
                        break
                
                if not found:
                    errors.append({
                        "node_id": node.get("id"),
                        "message": f"Node type '{node_type}' not available in {edition} edition"
                    })
        
        # Check edge connections
        node_ids = {node.get("id") for node in nodes}
        for edge in edges:
            if edge.get("source") not in node_ids:
                errors.append({
                    "edge_id": edge.get("id"),
                    "message": f"Edge source '{edge.get('source')}' not found"
                })
            if edge.get("target") not in node_ids:
                errors.append({
                    "edge_id": edge.get("id"),
                    "message": f"Edge target '{edge.get('target')}' not found"
                })
        
        # Check for start node
        has_start = any(node.get("type") == "start" for node in nodes)
        if not has_start:
            warnings.append({
                "message": "Workflow has no start node"
            })
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }