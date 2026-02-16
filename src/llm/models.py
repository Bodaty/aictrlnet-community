"""LLM module data models."""

from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class ModelProvider(str, Enum):
    """LLM model providers."""
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    BEDROCK = "bedrock"
    AZURE_OPENAI = "azure_openai"
    VERTEX_AI = "vertex_ai"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    DASHSCOPE = "dashscope"
    CUSTOM = "custom"


class ModelTier(str, Enum):
    """Model performance tiers."""
    FAST = "fast"          # < 1s, simple tasks
    BALANCED = "balanced"  # 2-5s, moderate complexity
    QUALITY = "quality"    # 10-20s, complex tasks
    PREMIUM = "premium"    # Claude/GPT-4, highest quality


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str
    provider: ModelProvider
    tier: ModelTier
    cost_per_1k_tokens: float = 0.0
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_system_prompt: bool = True
    supports_json_mode: bool = False
    average_response_time: float = 1.0
    description: str
    local: bool = False
    parameter_size: Optional[str] = None  # e.g., "8B", "70B"


class UserLLMSettings(BaseModel):
    """User's LLM preferences from UI settings."""
    user_id: str
    selected_model: str  # e.g., "llama3.2:1b", "claude-3-haiku" (legacy, set to preferredQualityModel)
    provider: ModelProvider
    api_keys: Dict[str, str] = {}  # Provider-specific API keys
    temperature: float = 0.7
    max_tokens: int = 1000
    stream_responses: bool = False
    fallback_model: Optional[str] = "llama3.2:3b"

    # Tier-based model preferences (new)
    preferredFastModel: Optional[str] = None      # Fast tier (~1-2s)
    preferredBalancedModel: Optional[str] = None  # Balanced tier (~3-5s)
    preferredQualityModel: Optional[str] = None   # Quality tier (~20-25s)


class LLMRequest(BaseModel):
    """Request for LLM generation."""
    model_config = ConfigDict(
        protected_namespaces=(),  # Allow model_ field names
        arbitrary_types_allowed=True
    )
    
    prompt: str
    user_settings: Optional[UserLLMSettings] = None  # From UI settings
    model_override: Optional[str] = None              # Direct override
    task_type: str = "general"                        # For auto model selection
    complexity: Optional[float] = None                # Override complexity detection
    temperature: Optional[float] = None               # Override user preference
    max_tokens: Optional[int] = None                  # Override user preference
    stream: Optional[bool] = None                     # Override user preference
    system_prompt: Optional[str] = None
    response_format: str = "text"                     # "text", "json", "structured"
    output_schema: Optional[Dict[str, Any]] = None   # For structured output (renamed from schema)
    cache_key: Optional[str] = None                   # For caching
    context: Optional[Dict[str, Any]] = None         # MCP or other context


class LLMResponse(BaseModel):
    """Response from LLM generation."""
    model_config = ConfigDict(protected_namespaces=())  # Allow model_used field name
    
    text: str
    model_used: str
    provider: ModelProvider
    tier: ModelTier
    tokens_used: int = 0
    cost: float = 0.0
    response_time: float = 0.0
    cache_hit: bool = False
    metadata: Dict[str, Any] = {}


class CostEstimate(BaseModel):
    """Cost estimation for a request."""
    estimated_tokens: int
    estimated_cost: float
    model: str
    provider: ModelProvider
    confidence: float  # 0-1, how confident in estimate


class UsageStats(BaseModel):
    """Usage statistics."""
    total_requests: int
    total_tokens: int
    total_cost: float
    by_model: Dict[str, Dict[str, float]]
    by_provider: Dict[str, Dict[str, float]]
    cache_hit_rate: float


class WorkflowStep(BaseModel):
    """Workflow step model (migrated from model_adapters)."""
    label: str
    description: Optional[str] = None
    action: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    agent: Optional[str] = None
    template: Optional[str] = None
    # Branching support fields - critical for non-linear workflows
    node_type: Optional[str] = None  # decision, parallel, ai_agent, process, etc.
    branches: Optional[List[Dict[str, Any]]] = None  # Branch definitions with condition/target
    condition: Optional[str] = None  # Condition expression for decision nodes
    capability: Optional[str] = None  # What the step does (e.g., "notification", "ai_processing")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "label": self.label,
            "description": self.description,
            "action": self.action,
            "input_data": self.input_data,
            "output_schema": self.output_schema,
            "dependencies": self.dependencies,
            "agent": self.agent,
            "template": self.template,
            "node_type": self.node_type,
            "capability": self.capability,
        }
        # Only include branches and condition if present (for decision/parallel nodes)
        if self.branches:
            result["branches"] = self.branches
        if self.condition:
            result["condition"] = self.condition
        return result


# =============================================================================
# Tool Calling Models (AICtrlNet Intelligent Assistant v4)
# =============================================================================

class ToolRecoveryStrategy(str, Enum):
    """Recovery strategies for tool execution failures."""
    RETRY = "retry"          # Exponential backoff retry
    FALLBACK = "fallback"    # Try alternative tool
    CLARIFY = "clarify"      # Ask user for more info
    FAIL = "fail"            # Return error to user


class ToolCall(BaseModel):
    """A tool call from the LLM.

    Represents a single tool invocation request from the LLM,
    including the tool name and its arguments.
    """
    model_config = ConfigDict(protected_namespaces=())

    name: str
    arguments: Dict[str, Any] = {}
    id: Optional[str] = None  # Unique ID for tracking

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "arguments": self.arguments,
            "id": self.id
        }


class ToolResult(BaseModel):
    """Result of tool execution with error handling.

    Contains the outcome of a tool execution including success/failure status,
    the returned data, and information for error recovery.
    """
    model_config = ConfigDict(protected_namespaces=())

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None  # "validation", "timeout", "service_error", "not_found"
    recovery_strategy: Optional[ToolRecoveryStrategy] = None
    fallback_tool: Optional[str] = None  # Alternative tool to try
    retry_after: Optional[int] = None  # Seconds to wait before retry
    attempts: int = 1
    execution_time_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_type": self.error_type,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
            "fallback_tool": self.fallback_tool,
            "retry_after": self.retry_after,
            "attempts": self.attempts,
            "execution_time_ms": self.execution_time_ms
        }


class LLMToolResponse(BaseModel):
    """Response from LLM with tool calling support.

    Extends the standard LLM response to include tool calls and
    support for multi-tool chains.
    """
    model_config = ConfigDict(protected_namespaces=())

    # Standard response fields
    text: Optional[str] = None  # Direct text response (if no tool call)
    model_used: str = ""
    provider: ModelProvider = ModelProvider.OLLAMA
    tier: ModelTier = ModelTier.QUALITY
    tokens_used: int = 0
    cost: float = 0.0
    response_time: float = 0.0
    cache_hit: bool = False
    metadata: Dict[str, Any] = {}

    # Tool calling fields
    tool_calls: Optional[List[ToolCall]] = None  # Tools to execute
    execution_plan: Optional[str] = None  # "I'll first X, then Y, then Z"
    requires_user_input: bool = False  # Pause for user input between steps
    intermediate_results: List[Dict[str, Any]] = []  # Results so far in chain

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "model_used": self.model_used,
            "provider": self.provider.value,
            "tier": self.tier.value,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "response_time": self.response_time,
            "cache_hit": self.cache_hit,
            "metadata": self.metadata,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls] if self.tool_calls else None,
            "execution_plan": self.execution_plan,
            "requires_user_input": self.requires_user_input,
            "intermediate_results": self.intermediate_results
        }


class ChainResult(BaseModel):
    """Result of executing a tool chain.

    Contains the outcome of executing multiple tools in sequence,
    including intermediate results and error handling.
    """
    model_config = ConfigDict(protected_namespaces=())

    completed: bool
    results: List[ToolResult] = []
    final_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None
    execution_time_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "completed": self.completed,
            "results": [r.to_dict() for r in self.results],
            "final_output": self.final_output,
            "error": self.error,
            "requires_clarification": self.requires_clarification,
            "clarification_prompt": self.clarification_prompt,
            "execution_time_ms": self.execution_time_ms
        }


class ToolDefinition(BaseModel):
    """Definition of a tool available to the LLM.

    Used to register tools in the tool registry and to format
    them for the LLM's tool calling prompt.
    """
    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: str
    parameters: Dict[str, Any] = {}  # JSON Schema for parameters
    editions: List[str] = ["community", "business", "enterprise"]
    handler: str = ""  # Service method to invoke (e.g. "api_key_service.list_keys")
    requires_confirmation: bool = False  # Require user confirmation before execution
    is_destructive: bool = False  # Marks destructive operations
    timeout_seconds: int = 60
    retry_count: int = 3
    fallback_tool: Optional[str] = None
    # Category-aware pruning fields (for 170+ tool environments)
    category: Optional[str] = None  # Primary category for pruning (e.g. "workflow", "access_control")
    subcategory: Optional[str] = None  # Subcategory (e.g. "api_keys", "rbac")
    tags: List[str] = []  # Searchable tags for keyword matching

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }

    def to_prompt_format(self) -> str:
        """Convert to a text format suitable for prompt injection."""
        params_desc = []
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        for name, schema in props.items():
            req_marker = " (required)" if name in required else " (optional)"
            param_type = schema.get("type", "any")
            desc = schema.get("description", "")
            params_desc.append(f"  - {name}: {param_type}{req_marker} - {desc}")

        params_str = "\n".join(params_desc) if params_desc else "  (no parameters)"

        return f"""Tool: {self.name}
Description: {self.description}
Parameters:
{params_str}"""