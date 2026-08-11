"""MCP endpoints for LLM service integration.

These call the LLM provider directly, so every request spends paid inference
quota. They previously declared `authorization: Optional[str] = Header(None)`
and never read it — the header looked like auth in the signature and the
OpenAPI schema, but anyone could call them anonymously and bill us for it.
`get_current_user` was already imported here and simply unused, so the
intent was clearly to require a token.

The dependency is resolved before the handler body, so the 401 it raises is
never caught by the broad `except Exception` blocks below.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
import uuid
import logging
from datetime import datetime

from llm.client import LLMClient
from llm.models import LLMRequest
from core.security import get_current_user
from schemas.mcp import (
    MCPLLMRequest, MCPLLMResponse, MCPLLMMessage, MCPLLMChoice, MCPLLMUsage,
    MCPEmbeddingRequest, MCPEmbeddingResponse, MCPEmbeddingData,
    MCPModelsResponse, MCPModelInfo
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp/v1", tags=["MCP LLM"])

# Used when the caller does not name a model. Matches the value already
# advertised by get_mcp_models() and openai_adapter's embeddings capability.
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"


@router.post("/messages", response_model=MCPLLMResponse)
async def handle_mcp_messages(
    request: MCPLLMRequest,
    current_user=Depends(get_current_user),
):
    """
    Handle LLM generation via MCP protocol.
    
    This endpoint provides a standardized interface for AI frameworks
    (LangChain, AutoGPT, CrewAI) to access our unified LLM service.
    """
    try:
        # Initialize LLM client
        llm_client = LLMClient()
        
        # Extract the last message as the prompt
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        prompt = request.messages[-1].content
        
        # Prepare context from conversation history
        context = {
            "conversation_history": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages[:-1]
            ] if len(request.messages) > 1 else []
        }
        
        # Generate response using LLM client
        response = await llm_client.generate(
            prompt=prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            task_type=request.metadata.get("task_type", "general") if request.metadata else "general",
            context=context
        )
        
        # Format MCP response
        return MCPLLMResponse(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(datetime.utcnow().timestamp()),
            model=response.model_used,
            choices=[
                MCPLLMChoice(
                    index=0,
                    message=MCPLLMMessage(
                        role="assistant",
                        content=response.text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=MCPLLMUsage(
                prompt_tokens=response.tokens_used // 2,  # Rough estimate
                completion_tokens=response.tokens_used // 2,
                total_tokens=response.tokens_used
            ) if response.tokens_used else None
        )
        
    except Exception as e:
        logger.error(f"MCP message handling failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings", response_model=MCPEmbeddingResponse)
async def handle_mcp_embeddings(
    request: MCPEmbeddingRequest,
    current_user=Depends(get_current_user),
):
    """Generate embeddings via MCP protocol.

    Routed through the adapter registry rather than LLMClient: embeddings are a
    distinct adapter capability (see openai_adapter._handle_embeddings), and
    LLMClient only exposes text generation. When no registered adapter advertises
    the capability this returns 503 — it never synthesises vectors, because a
    fabricated embedding is worse than an outage for anything that indexes it.
    """
    texts = request.input if isinstance(request.input, list) else [request.input]
    model = request.model or DEFAULT_EMBEDDING_MODEL

    try:
        from adapters.registry import adapter_registry
        from adapters.models import AdapterRequest

        adapters = await adapter_registry.get_adapters_by_capability("embeddings")
        if not adapters:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No embeddings-capable adapter is registered. Configure an "
                    "adapter that provides the 'embeddings' capability."
                ),
            )

        response = await adapters[0].execute(
            AdapterRequest(
                capability="embeddings",
                parameters={"model": model, "input": texts},
                user_id=str(getattr(current_user, "id", "")) or None,
            )
        )

        if response.status != "success":
            raise HTTPException(
                status_code=502,
                detail=f"Embeddings provider error: {response.error or 'unknown'}",
            )

        payload = response.data or {}
        # OpenAI-shaped: [{"object": "embedding", "index": i, "embedding": [...]}]
        rows = payload.get("embeddings") or []
        usage = payload.get("usage") or {}
        total_tokens = usage.get("total_tokens") or response.tokens_used or 0

        return MCPEmbeddingResponse(
            object="list",
            data=[
                MCPEmbeddingData(
                    object="embedding",
                    index=row.get("index", i) if isinstance(row, dict) else i,
                    embedding=row.get("embedding", []) if isinstance(row, dict) else row,
                )
                for i, row in enumerate(rows)
            ],
            model=model,
            usage=MCPLLMUsage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=MCPModelsResponse)
async def get_mcp_models():
    """
    List available models in MCP format.
    
    Note: The LLM service automatically selects the best model based on
    task complexity and user settings. This endpoint shows the models
    that the system knows about across all tiers.
    """
    try:
        from llm.model_selection import MODEL_CONFIGS
        
        # Build a list of all models from the configuration
        models = []
        for tier, config in MODEL_CONFIGS.items():
            for model_name in config["models"]:
                # Determine provider from model name
                if "gpt" in model_name.lower():
                    provider = "openai"
                elif "claude" in model_name.lower():
                    provider = "anthropic"
                elif "llama" in model_name.lower() or "mistral" in model_name.lower() or "phi" in model_name.lower() or "gemma" in model_name.lower():
                    provider = "ollama"
                else:
                    provider = "unknown"
                
                # Estimate cost based on tier
                cost_map = {
                    "fast": 0.0,  # Local models
                    "balanced": 0.001,
                    "quality": 0.01,
                    "premium": 0.03
                }
                
                models.append(
                    MCPModelInfo(
                        id=model_name,
                        object="model",
                        created=int(datetime.utcnow().timestamp()),
                        owned_by=provider,
                        permission=[],
                        root=model_name,
                        parent=None,
                        model_metadata={
                            "tier": tier.value,
                            "capabilities": ["chat", "completion"],
                            "cost_per_1k_tokens": cost_map.get(tier.value, 0.001),
                            "auto_selected": True  # Indicates LLM service will auto-select
                        }
                    )
                )
        
        return MCPModelsResponse(
            object="list",
            data=models
        )
        
    except Exception as e:
        logger.error(f"Failed to list MCP models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/execute")
async def handle_mcp_tool_execution(
    request: Dict[str, Any],
    current_user=Depends(get_current_user),
):
    """
    Execute tools/functions via MCP protocol.
    
    This can be used to execute adapters, workflows, or other tools
    through the MCP interface.
    """
    try:
        tool_name = request.get("tool")
        tool_input = request.get("input", {})

        if not tool_name:
            raise HTTPException(
                status_code=422,
                detail="Missing required field 'tool' in request body"
            )

        # Route to appropriate tool handler
        if tool_name == "adapter":
            from services.adapter import AdapterService
            adapter_service = AdapterService()
            result = await adapter_service.execute_adapter(
                adapter_id=tool_input.get("adapter_id"),
                params=tool_input.get("params", {})
            )
        elif tool_name == "workflow":
            from services.workflow import WorkflowService
            workflow_service = WorkflowService()
            result = await workflow_service.execute_workflow(
                workflow_id=tool_input.get("workflow_id"),
                inputs=tool_input.get("inputs", {})
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
        
        return {
            "result": result,
            "tool": tool_name,
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP tool execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))