"""Tool-calling contract for AI adapters.

Defines the ToolCallingMixin that AI adapters can implement alongside BaseAdapter
to support native tool calling. Not all adapters need this — only AI/LLM adapters
that support structured tool invocation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolCallingRequest(BaseModel):
    """Request for tool-augmented generation."""
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]  # JSON Schema tool definitions (OpenAI format)
    model: str
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tool_choice: str = "auto"  # "auto", "required", or specific tool name
    cache_system_prefix: bool = True  # Enable prompt/context caching for system prompt


class ToolCallingResponse(BaseModel):
    """Response from tool-augmented generation."""
    text: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []  # [{name, arguments, id}]
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    cached_tokens: int = 0
    stop_reason: Optional[str] = None


class ToolCallingStreamEvent(BaseModel):
    """Event yielded during streaming tool-augmented generation."""
    type: str  # "text_delta", "tool_calls", "done", "error"
    text: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None


class ToolCallingMixin(ABC):
    """Mixin for adapters that support native tool calling.

    AI adapters (Ollama, Claude, OpenAI, etc.) can implement this alongside
    BaseAdapter to provide structured tool calling through the adapter framework.
    Non-AI adapters (Slack, Stripe, etc.) don't need this.

    Usage:
        class OllamaAdapter(BaseAdapter, ToolCallingMixin):
            async def chat_with_tools(self, request):
                ...
    """

    @abstractmethod
    async def chat_with_tools(self, request: ToolCallingRequest) -> ToolCallingResponse:
        """Execute a chat completion with tool calling support.

        Args:
            request: ToolCallingRequest with messages, tools, and model config.

        Returns:
            ToolCallingResponse with text and/or tool_calls.
        """
        ...

    async def chat_with_tools_stream(self, request: ToolCallingRequest):
        """Stream tool-calling response. Default: falls back to non-streaming."""
        result = await self.chat_with_tools(request)
        if result.text:
            yield ToolCallingStreamEvent(type="text_delta", text=result.text)
        if result.tool_calls:
            yield ToolCallingStreamEvent(type="tool_calls", tool_calls=result.tool_calls)
        yield ToolCallingStreamEvent(
            type="done",
            tokens_used=result.tokens_used,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            stop_reason=result.stop_reason,
        )
