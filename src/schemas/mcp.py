"""MCP (Model Context Protocol) schemas."""

from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum


# Transport type for MCP servers
class MCPTransportType(str, Enum):
    """MCP transport types per specification."""
    stdio = "stdio"
    http_sse = "http_sse"


# Server schemas

class MCPServerCreate(BaseModel):
    """MCP server creation request.

    MCP servers can be configured in two ways:
    1. stdio transport: Uses command + args + env_vars to spawn subprocess
    2. HTTP/SSE transport: Uses url + api_key for remote servers

    See: https://modelcontextprotocol.io/specification/2025-03-26
    """
    name: str = Field(..., description="Server name")
    transport_type: MCPTransportType = Field(MCPTransportType.stdio, description="Transport type: stdio or http_sse")

    # stdio transport configuration (required if transport_type is stdio)
    command: Optional[str] = Field(None, description="Executable command (e.g., 'npx', 'python', 'node')")
    args: Optional[List[str]] = Field(None, description="Command arguments")
    env_vars: Optional[Dict[str, str]] = Field(None, description="Environment variables for the subprocess")

    # HTTP/SSE transport configuration (required if transport_type is http_sse)
    url: Optional[str] = Field(None, description="Server URL (for HTTP/SSE transport)")
    api_key: Optional[str] = Field(None, description="API key for the server")

    service_type: str = Field("general", description="Service type category")
    server_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional server metadata")

    # OAuth2 integration for MCP server authentication (SEP-991)
    oauth2_provider_id: Optional[str] = Field(None, description="ID of OAuth2 provider for server authentication (Business/Enterprise)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "filesystem-mcp",
                "transport_type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"],
                "env_vars": {"DEBUG": "true"},
                "service_type": "filesystem"
            }
        }


class MCPServerResponse(BaseModel):
    """MCP server response."""
    id: str
    name: str
    transport_type: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env_vars: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    service_type: str
    status: str
    protocol_version: Optional[str] = None
    server_capabilities: Optional[Dict[str, Any]] = None
    last_checked: Optional[datetime] = None
    server_info: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    server_metadata: Optional[Dict[str, Any]] = None
    # OAuth2 integration for MCP server authentication (SEP-991)
    oauth2_provider_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class MCPServerUpdate(BaseModel):
    """MCP server update request."""
    name: Optional[str] = None
    transport_type: Optional[MCPTransportType] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env_vars: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    status: Optional[str] = None
    server_metadata: Optional[Dict[str, Any]] = None
    # OAuth2 integration for MCP server authentication (SEP-991)
    oauth2_provider_id: Optional[str] = None


# Context schemas

class MCPContextCreate(BaseModel):
    """MCP context creation request."""
    server_id: str = Field(..., description="MCP server ID")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Initial messages")
    max_tokens: int = Field(4096, description="Maximum context tokens")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MCPContextResponse(BaseModel):
    """MCP context response."""
    id: str
    server_id: str
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None
    messages: List[Dict[str, Any]]
    token_count: int
    max_tokens: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Capability schemas

class MCPCapability(BaseModel):
    """MCP capability."""
    name: str
    enabled: bool = True
    description: Optional[str] = None


# Execution schemas

class MCPExecuteRequest(BaseModel):
    """MCP execution request."""
    server_id: str = Field(..., description="MCP server ID")
    method: str = Field(..., description="Method to execute")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Messages to send")
    context_id: Optional[str] = Field(None, description="Context ID to use")
    update_context: bool = Field(True, description="Update context with response")
    model: Optional[str] = Field(None, description="Specific model to use")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class MCPExecuteResponse(BaseModel):
    """MCP execution response."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


# Discovery schemas

class MCPServerDiscoveryResponse(BaseModel):
    """MCP server discovery response."""
    servers: List[MCPServerResponse]
    total: int
    providers: List[str]
    capabilities: List[str]


# Info schemas

class MCPInfo(BaseModel):
    """MCP system information."""
    version: str = "1.0.0"
    supported_providers: List[str] = ["openai", "anthropic", "google", "cohere", "mistral"]
    features: List[str] = ["context-management", "token-optimization", "multi-provider", "streaming"]
    status: str = "active"


# List schemas

class MCPServerList(BaseModel):
    """MCP server list response."""
    servers: List[MCPServerResponse]
    total: int
    page: int = 1
    per_page: int = 20


# Health check schemas

class MCPHealthCheck(BaseModel):
    """MCP health check response."""
    server_id: str
    server_name: str
    transport_type: str
    url: Optional[str] = None  # Only for HTTP/SSE transport
    command: Optional[str] = None  # Only for stdio transport
    status: str
    protocol_version: Optional[str] = None
    error: Optional[str] = None
    checked_at: datetime


class MCPHealthCheckList(BaseModel):
    """MCP health check list response."""
    health_checks: List[MCPHealthCheck]
    total: int


# Test schemas

class MCPTestRequest(BaseModel):
    """MCP test request."""
    server_id: str
    test_message: str = "Hello from AICtrlNet!"


class MCPTestResponse(BaseModel):
    """MCP test response."""
    server_id: str
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    tested_at: datetime


# Task schemas

class MCPTaskCreate(BaseModel):
    """MCP task creation request."""
    api_type: str = Field("message", description="API type: message, embedding, tool")
    messages: Optional[List[Dict[str, str]]] = None
    server_name: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1000, ge=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MCPTaskResponse(BaseModel):
    """MCP task response."""
    task_id: str
    source_id: str
    destination: str
    status: str
    result: Optional[Dict[str, Any]] = None
    mcp_server_used: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# MCP Server endpoint schemas

class MCPRole(str, Enum):
    """MCP message roles."""
    system = "system"
    user = "user"
    assistant = "assistant"


class MCPMessage(BaseModel):
    """MCP message format."""
    role: MCPRole
    content: Union[str, Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class MCPMessageRequest(BaseModel):
    """Request for MCP message processing."""
    messages: List[MCPMessage] = Field(..., min_items=1)
    execute_immediately: bool = True
    routing_preferences: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "system",
                        "content": "Task: Analyze customer feedback sentiment"
                    },
                    {
                        "role": "user",
                        "content": "Please analyze the sentiment of our latest product reviews"
                    }
                ],
                "execute_immediately": True,
                "routing_preferences": {
                    "preferred_providers": ["openai", "anthropic"]
                }
            }
        }


class MCPMessageResponse(BaseModel):
    """Response from MCP message processing."""
    task_id: Optional[str] = None
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPQualityRequest(BaseModel):
    """Request for quality assessment via MCP."""
    content: Optional[str] = None
    content_type: str = "text"
    criteria: Optional[Dict[str, Any]] = None
    batch_mode: bool = False
    items: Optional[List[Dict[str, Any]]] = None
    
    @validator('items')
    def validate_batch_mode(cls, v, values):
        if values.get('batch_mode') and not v:
            raise ValueError("items required when batch_mode is True")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "This is a sample text for quality assessment",
                "content_type": "text",
                "criteria": {
                    "accuracy": {"weight": 0.3, "threshold": 0.8},
                    "completeness": {"weight": 0.25, "threshold": 0.9}
                }
            }
        }


class MCPQualityDimension(BaseModel):
    """Quality dimension assessment."""
    score: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)


class MCPQualityResponse(BaseModel):
    """Response from quality assessment."""
    quality_score: float = Field(..., ge=0.0, le=1.0)
    dimensions: Dict[str, MCPQualityDimension] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    content_type: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # For batch mode
    batch_size: Optional[int] = None
    average_quality_score: Optional[float] = None
    results: Optional[List[Dict[str, Any]]] = None


class MCPWorkflowStep(BaseModel):
    """Workflow step definition."""
    id: Optional[str] = None
    name: str
    type: str = "task"
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class MCPWorkflowConnection(BaseModel):
    """Workflow connection definition."""
    from_step: str = Field(..., alias="from")
    to_step: str = Field(..., alias="to")
    type: str = "default"
    data: Dict[str, Any] = Field(default_factory=dict)


class MCPWorkflowRequest(BaseModel):
    """Request to create workflow via MCP."""
    workflow_definition: Dict[str, Any]
    execute_immediately: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "workflow_definition": {
                    "name": "Customer Feedback Analysis",
                    "description": "Analyze and categorize customer feedback",
                    "steps": [
                        {
                            "name": "Extract Feedback",
                            "type": "data_extraction",
                            "parameters": {"source": "reviews_api"}
                        },
                        {
                            "name": "Sentiment Analysis",
                            "type": "ai_analysis",
                            "parameters": {"model": "sentiment-v2"}
                        }
                    ]
                },
                "execute_immediately": False
            }
        }


class MCPWorkflowResponse(BaseModel):
    """Response from workflow creation."""
    workflow_id: Optional[str] = None
    name: Optional[str] = None
    status: str
    error: Optional[str] = None
    created_at: Optional[str] = None
    execution_id: Optional[str] = None
    execution_status: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Enterprise MCP schemas

class MCPComplianceRequest(BaseModel):
    """Request for compliance checking."""
    content: str
    compliance_frameworks: List[str] = Field(..., min_items=1)
    check_pii: bool = True
    suggest_remediation: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Customer data including name John Doe and email john@example.com",
                "compliance_frameworks": ["GDPR", "HIPAA"],
                "check_pii": True,
                "suggest_remediation": True
            }
        }


class MCPComplianceResponse(BaseModel):
    """Response from compliance check."""
    compliant: bool
    frameworks_checked: List[str]
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    pii_detected: bool = False
    pii_locations: List[Dict[str, Any]] = Field(default_factory=list)
    remediation_suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPSLAConfigRequest(BaseModel):
    """Request to configure SLA monitoring."""
    server_id: str
    sla_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "response_time_ms": 1000,
            "availability_percent": 99.9,
            "success_rate_percent": 99.5
        }
    )
    alert_enabled: bool = True
    alert_channels: List[str] = Field(default_factory=lambda: ["email", "slack"])
    
    class Config:
        json_schema_extra = {
            "example": {
                "server_id": "mcp-server-001",
                "sla_thresholds": {
                    "response_time_ms": 500,
                    "availability_percent": 99.95,
                    "success_rate_percent": 99.8
                },
                "alert_enabled": True,
                "alert_channels": ["email", "slack", "webhook"]
            }
        }


class MCPSLAConfigResponse(BaseModel):
    """Response from SLA configuration."""
    id: str
    server_id: str
    sla_thresholds: Dict[str, float]
    alert_enabled: bool
    alert_channels: List[str]
    created_at: datetime
    updated_at: Optional[datetime] = None


class MCPDiscoveryService(BaseModel):
    """Discovered MCP service."""
    name: str
    endpoint: str
    description: str
    capabilities: List[str]


class MCPDiscoveryResponse(BaseModel):
    """Response from service discovery."""
    services: Dict[str, MCPDiscoveryService]
    version: str
    server_name: str


class MCPServiceRegionCreate(BaseModel):
    """Schema for creating MCP service region (Enterprise)."""
    server_id: str
    region: str
    url: str
    priority: int = 1
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPServiceRegionResponse(BaseModel):
    """Response schema for MCP service region (Enterprise)."""
    id: str
    server_id: str
    tenant_id: Optional[str] = None
    region: str
    url: str
    status: str
    priority: int
    latency_ms: Optional[float] = None
    last_checked: Optional[datetime] = None
    metadata: Dict[str, Any]
    created_at: float
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# MCP LLM Service schemas (for AI agent frameworks)

class MCPLLMMessage(BaseModel):
    """Message format for LLM service via MCP."""
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Explain quantum computing"
            }
        }


class MCPLLMRequest(BaseModel):
    """Request for LLM service via MCP protocol."""
    messages: List[MCPLLMMessage]
    model: Optional[str] = None
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 100
            }
        }


class MCPLLMChoice(BaseModel):
    """Single choice in LLM response."""
    index: int
    message: MCPLLMMessage
    finish_reason: Optional[str] = None


class MCPLLMUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MCPLLMResponse(BaseModel):
    """Response from LLM service via MCP protocol."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[MCPLLMChoice]
    usage: Optional[MCPLLMUsage] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The capital of France is Paris."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18
                }
            }
        }


class MCPEmbeddingRequest(BaseModel):
    """Request for embedding generation via MCP."""
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: str = "float"
    
    class Config:
        json_schema_extra = {
            "example": {
                "input": "The quick brown fox jumps over the lazy dog",
                "model": "text-embedding-ada-002"
            }
        }


class MCPEmbeddingData(BaseModel):
    """Single embedding in response."""
    index: int
    embedding: List[float]
    object: str = "embedding"


class MCPEmbeddingResponse(BaseModel):
    """Response from embedding service via MCP."""
    object: str = "list"
    data: List[MCPEmbeddingData]
    model: str
    usage: MCPLLMUsage
    
    class Config:
        json_schema_extra = {
            "example": {
                "object": "list",
                "data": [
                    {
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3],
                        "object": "embedding"
                    }
                ],
                "model": "text-embedding-ada-002",
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 0,
                    "total_tokens": 8
                }
            }
        }


class MCPModelInfo(BaseModel):
    """Information about an available model."""
    model_config = ConfigDict(protected_namespaces=())

    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = Field(default_factory=list)
    root: str
    parent: Optional[str] = None
    model_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MCPModelsResponse(BaseModel):
    """Response listing available models."""
    object: str = "list"
    data: List[MCPModelInfo]


# MCP Agent Tool Provider schemas (Phase 5)

class MCPAgentToolResponse(BaseModel):
    """Response for a single MCP agent tool."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    server_name: str = Field(..., description="MCP server providing this tool")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON schema for parameters")
    agent_tool_name: str = Field(..., description="Name when used in agent (mcp_server_tool)")


class MCPAgentToolListResponse(BaseModel):
    """Response listing available MCP tools for agents."""
    tools: List[MCPAgentToolResponse]
    total: int
    servers: int
    cache_valid: bool = True


class MCPAgentExecutionRequest(BaseModel):
    """Request to execute an agent with MCP tools."""
    agent_id: str = Field(..., description="Agent ID to execute")
    task: Dict[str, Any] = Field(..., description="Task for the agent")
    mcp_servers: Optional[List[str]] = Field(None, description="Specific MCP servers to use tools from")
    auto_inject_tools: bool = Field(True, description="Auto-inject available MCP tools")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "agent-123",
                "task": {
                    "type": "analysis",
                    "content": "Analyze this customer feedback"
                },
                "mcp_servers": ["openai-mcp", "custom-analytics"],
                "auto_inject_tools": True
            }
        }


class MCPAgentExecutionResponse(BaseModel):
    """Response from agent execution with MCP tools."""
    status: str
    agent_id: str
    task_id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    mcp_tools_used: List[str] = Field(default_factory=list)
    execution_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Structured Tool Output Schemas (MCP 2025-11-25)
# Tools can declare outputSchema for typed returns, results include structuredContent
# =============================================================================

class MCPToolOutputSchema(BaseModel):
    """JSON Schema for tool output validation.

    Per MCP spec, tools can declare an outputSchema to specify the structure
    of their return values. This enables type-safe tool interactions.
    """
    type: str = Field("object", description="JSON Schema type")
    properties: Optional[Dict[str, Any]] = Field(None, description="Schema properties")
    required: Optional[List[str]] = Field(None, description="Required properties")
    description: Optional[str] = Field(None, description="Schema description")
    additional_properties: Optional[bool] = Field(None, alias="additionalProperties")

    model_config = ConfigDict(populate_by_name=True)


class MCPToolDefinition(BaseModel):
    """Complete MCP tool definition with optional outputSchema.

    This extends the basic tool definition with structured output support.
    """
    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    input_schema: Dict[str, Any] = Field(..., alias="inputSchema", description="JSON Schema for tool input")
    output_schema: Optional[MCPToolOutputSchema] = Field(None, alias="outputSchema", description="JSON Schema for structured output")
    annotations: Optional["MCPToolAnnotations"] = Field(None, description="Tool behavior annotations")

    model_config = ConfigDict(populate_by_name=True)


class MCPStructuredContent(BaseModel):
    """Structured content returned by a tool.

    When a tool has an outputSchema, results should include structuredContent
    that conforms to that schema.
    """
    data: Dict[str, Any] = Field(..., description="Structured data matching outputSchema")
    schema_version: Optional[str] = Field(None, description="Version of the schema used")
    validated: bool = Field(True, description="Whether data was validated against schema")


class MCPToolResultContent(BaseModel):
    """Content item in a tool result.

    Per MCP spec, tool results contain an array of content items.
    """
    type: str = Field(..., description="Content type: text, image, resource, etc.")
    text: Optional[str] = Field(None, description="Text content")
    data: Optional[str] = Field(None, description="Base64-encoded data for images")
    mime_type: Optional[str] = Field(None, alias="mimeType", description="MIME type for binary data")
    resource: Optional[Dict[str, Any]] = Field(None, description="Embedded resource")

    model_config = ConfigDict(populate_by_name=True)


class MCPToolResult(BaseModel):
    """Complete tool execution result with optional structured content.

    Per MCP 2025-11-25, results can include both traditional content array
    and new structuredContent field for typed data.
    """
    content: List[MCPToolResultContent] = Field(default_factory=list, description="Result content array")
    structured_content: Optional[MCPStructuredContent] = Field(None, alias="structuredContent", description="Typed structured data")
    is_error: bool = Field(False, alias="isError", description="Whether result is an error")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# Tool Annotations (MCP 2025-11-25)
# Metadata about tool behavior for informed decision-making
# =============================================================================

class MCPToolAnnotations(BaseModel):
    """Annotations describing tool behavior and characteristics.

    These help clients and LLMs make informed decisions about tool usage.
    """
    # Execution characteristics
    title: Optional[str] = Field(None, description="Human-readable title for the tool")
    read_only_hint: Optional[bool] = Field(None, alias="readOnlyHint", description="True if tool only reads data, doesn't modify")
    destructive_hint: Optional[bool] = Field(None, alias="destructiveHint", description="True if tool may cause irreversible changes")
    idempotent_hint: Optional[bool] = Field(None, alias="idempotentHint", description="True if repeated calls have same effect")
    open_world_hint: Optional[bool] = Field(None, alias="openWorldHint", description="True if tool interacts with external world")

    model_config = ConfigDict(populate_by_name=True)


class MCPToolCallRequest(BaseModel):
    """Request to call a tool with structured output support.

    Enhanced tool call request that can request structured output
    if the tool supports it.
    """
    name: str = Field(..., description="Tool name to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    request_structured: bool = Field(False, alias="requestStructured", description="Request structured output if available")

    model_config = ConfigDict(populate_by_name=True)


class MCPToolCallResponse(BaseModel):
    """Response from a tool call with structured output.

    Includes both traditional content and optional structured content.
    """
    tool_name: str = Field(..., description="Name of the tool called")
    success: bool = Field(..., description="Whether the call succeeded")
    content: List[MCPToolResultContent] = Field(default_factory=list, description="Result content")
    structured_content: Optional[MCPStructuredContent] = Field(None, alias="structuredContent")
    is_error: bool = Field(False, alias="isError")
    latency_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    output_schema_used: Optional[str] = Field(None, description="Name of output schema that was applied")

    model_config = ConfigDict(populate_by_name=True)


class MCPToolInjectionRequest(BaseModel):
    """Request to inject MCP tools into an agent."""
    servers: Optional[List[str]] = Field(None, description="Specific servers to inject tools from")

    class Config:
        json_schema_extra = {
            "example": {
                "servers": ["openai-mcp"]
            }
        }


class MCPToolInjectionResponse(BaseModel):
    """Response from tool injection."""
    agent_id: str
    tools_injected: int
    servers_used: List[str]
    message: str


class MCPToolsSummaryResponse(BaseModel):
    """Summary of available MCP tools."""
    total_tools: int
    servers: int
    tools_by_server: Dict[str, int]
    cache_valid: bool
    last_refresh: Optional[str] = None


# =============================================================================
# MCP Async Task Schemas (SEP-1686 - Tasks)
# Per MCP specification 2025-11-25
# See: https://spec.modelcontextprotocol.io/specification/2025-11-25/server/tasks/
# =============================================================================

class MCPTaskState(str, Enum):
    """MCP Task state per SEP-1686 specification."""
    working = "working"      # Task is actively processing
    completed = "completed"  # Task finished successfully
    failed = "failed"        # Task encountered an error
    cancelled = "cancelled"  # Task was cancelled by client


class MCPAsyncTaskCreate(BaseModel):
    """Request to create an async MCP task.

    Used when a tool call or resource operation returns a task token
    instead of immediate results.
    """
    server_id: str = Field(..., description="MCP server ID handling this task")
    tool_id: Optional[str] = Field(None, description="Tool that initiated this task (if applicable)")
    method: str = Field(..., description="MCP method that created the task (e.g., 'tools/call')")
    request_params: Optional[Dict[str, Any]] = Field(None, description="Original request parameters")
    timeout_seconds: Optional[int] = Field(300, description="Task timeout in seconds")
    client_id: Optional[str] = Field(None, description="Client that initiated the task")
    task_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional task metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "server_id": "mcp-server-123",
                "method": "tools/call",
                "request_params": {"name": "analyze_data", "arguments": {"query": "sales Q4"}},
                "timeout_seconds": 300
            }
        }


class MCPAsyncTaskResponse(BaseModel):
    """Response schema for an MCP async task."""
    id: str
    task_token: str
    server_id: str
    tool_id: Optional[str] = None
    method: str
    state: MCPTaskState
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage 0-100")
    progress_message: Optional[str] = None
    result_content: Optional[Dict[str, Any]] = None
    structured_result: Optional[Dict[str, Any]] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    error_data: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    timeout_seconds: Optional[int] = None
    client_id: Optional[str] = None
    task_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class MCPAsyncTaskUpdate(BaseModel):
    """Update schema for MCP async task state.

    Used to update task progress or mark as completed/failed.
    """
    state: Optional[MCPTaskState] = None
    progress: Optional[int] = Field(None, ge=0, le=100)
    progress_message: Optional[str] = None
    result_content: Optional[Dict[str, Any]] = None
    structured_result: Optional[Dict[str, Any]] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    error_data: Optional[Dict[str, Any]] = None


class MCPAsyncTaskList(BaseModel):
    """List of MCP async tasks."""
    tasks: List[MCPAsyncTaskResponse]
    total: int
    page: int = 1
    per_page: int = 20


# JSON-RPC 2.0 Request/Response schemas for MCP Tasks protocol

class MCPTasksGetRequest(BaseModel):
    """JSON-RPC request for tasks/get method.

    Per MCP specification, clients use this to poll task status.
    """
    task_token: str = Field(..., description="The task token returned when task was created")


class MCPTasksCancelRequest(BaseModel):
    """JSON-RPC request for tasks/cancel method.

    Per MCP specification, clients use this to request task cancellation.
    """
    task_token: str = Field(..., description="The task token to cancel")


class MCPTaskProgressNotification(BaseModel):
    """Notification schema for task progress updates.

    Servers send these notifications to update clients on task progress.
    """
    task_token: str
    progress: int = Field(..., ge=0, le=100)
    message: Optional[str] = None


class MCPTaskCompletedNotification(BaseModel):
    """Notification schema for task completion.

    Servers send this when a task completes successfully.
    """
    task_token: str
    result: Dict[str, Any]
    structured_result: Optional[Dict[str, Any]] = None


class MCPTaskFailedNotification(BaseModel):
    """Notification schema for task failure.

    Servers send this when a task fails.
    """
    task_token: str
    error_code: int
    error_message: str
    error_data: Optional[Dict[str, Any]] = None


# =============================================================================
# Sampling with Tools Schemas (SEP-1577 - MCP 2025-11-25)
# Enables agentic workflows where server requests LLM sampling from client
# See: https://spec.modelcontextprotocol.io/specification/2025-11-25/client/sampling/
# =============================================================================

class MCPSamplingMessage(BaseModel):
    """A message in an MCP sampling request/response.

    Messages can be from the user or assistant, with text or image content.
    """
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: Union[str, Dict[str, Any]] = Field(..., description="Message content (text or structured)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "What is the capital of France?"
            }
        }
    )


class MCPSamplingModelHint(BaseModel):
    """Hint for model selection in sampling requests.

    Servers can suggest models for the client to use.
    """
    name: Optional[str] = Field(None, description="Model name hint (e.g., 'claude-3-opus')")


class MCPSamplingModelPreferences(BaseModel):
    """Model preferences for sampling requests.

    Allows servers to specify desired model characteristics.
    """
    hints: Optional[List[MCPSamplingModelHint]] = Field(None, description="Model name hints")
    cost_priority: Optional[float] = Field(None, alias="costPriority", ge=0, le=1, description="Priority for cost (0=don't care, 1=prioritize low cost)")
    speed_priority: Optional[float] = Field(None, alias="speedPriority", ge=0, le=1, description="Priority for speed (0=don't care, 1=prioritize fast)")
    intelligence_priority: Optional[float] = Field(None, alias="intelligencePriority", ge=0, le=1, description="Priority for capability (0=don't care, 1=prioritize smart)")

    model_config = ConfigDict(populate_by_name=True)


class MCPCreateSamplingRequest(BaseModel):
    """Request from server to client for LLM sampling.

    Per MCP SEP-1577, servers can request the client to perform sampling
    to enable agentic workflows with multi-step reasoning.
    """
    messages: List[MCPSamplingMessage] = Field(..., description="Conversation messages for sampling")
    model_preferences: Optional[MCPSamplingModelPreferences] = Field(None, alias="modelPreferences")
    system_prompt: Optional[str] = Field(None, alias="systemPrompt", description="System prompt for the sampling")
    include_context: Optional[Literal["none", "thisServer", "allServers"]] = Field(
        "none", alias="includeContext",
        description="Whether to include MCP context in sampling"
    )
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Sampling temperature")
    max_tokens: int = Field(..., alias="maxTokens", description="Maximum tokens to generate")
    stop_sequences: Optional[List[str]] = Field(None, alias="stopSequences", description="Stop sequences")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(
        populate_by_name=True,
        protected_namespaces=(),  # Allow model_ prefix for MCP fields
        json_schema_extra={
            "example": {
                "messages": [{"role": "user", "content": "Analyze this data and suggest improvements"}],
                "maxTokens": 1000,
                "temperature": 0.7,
                "includeContext": "thisServer"
            }
        }
    )


class MCPSamplingStopReason(str, Enum):
    """Reason why sampling stopped."""
    end_turn = "endTurn"
    stop_sequence = "stopSequence"
    max_tokens = "maxTokens"


class MCPCreateSamplingResponse(BaseModel):
    """Response from client with sampling result.

    Contains the generated text from the LLM sampling.
    """
    role: Literal["assistant"] = Field("assistant", description="Role is always assistant")
    content: Union[str, Dict[str, Any]] = Field(..., description="Generated content")
    model: Optional[str] = Field(None, description="Model that was used")
    stop_reason: Optional[MCPSamplingStopReason] = Field(None, alias="stopReason")

    model_config = ConfigDict(populate_by_name=True)


class MCPSamplingCapability(BaseModel):
    """Client capability declaration for sampling support.

    Clients that support sampling should advertise this capability.
    """
    supported: bool = Field(True, description="Whether sampling is supported")
    models: Optional[List[str]] = Field(None, description="Available model names")
    max_context_tokens: Optional[int] = Field(None, alias="maxContextTokens", description="Maximum context size")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# URL Mode Elicitation Schemas (SEP-1036 - MCP 2025-11-25)
# Secure out-of-band credential flows for sensitive data
# =============================================================================

class MCPElicitationMode(str, Enum):
    """Mode for elicitation request."""
    url = "url"  # Open URL for user interaction


class MCPElicitationRequest(BaseModel):
    """Request to elicit information from user via URL.

    Per SEP-1036, servers can request the client to open a URL
    for secure credential input or OAuth flows.
    """
    mode: MCPElicitationMode = Field(MCPElicitationMode.url, description="Elicitation mode")
    url: str = Field(..., description="URL to open for user interaction")
    message: Optional[str] = Field(None, description="Message to display to user")
    timeout_seconds: Optional[int] = Field(300, alias="timeoutSeconds", description="Timeout for user action")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class MCPElicitationStatus(str, Enum):
    """Status of an elicitation request."""
    pending = "pending"
    completed = "completed"
    cancelled = "cancelled"
    timeout = "timeout"


class MCPElicitationResponse(BaseModel):
    """Response from elicitation request.

    Contains the result of the user's action at the URL.
    """
    status: MCPElicitationStatus = Field(..., description="Elicitation status")
    result: Optional[Dict[str, Any]] = Field(None, description="Result data from elicitation")
    error: Optional[str] = Field(None, description="Error message if failed")

    model_config = ConfigDict(populate_by_name=True)