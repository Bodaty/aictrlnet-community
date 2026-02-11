"""API endpoints for LLM module."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from core.security import get_current_user
from core.config import get_settings
from models import User
from ..service import llm_service
from ..models import (
    LLMRequest, LLMResponse, ModelInfo, CostEstimate,
    UsageStats, WorkflowStep, UserLLMSettings
)

router = APIRouter(tags=["llm"])


class GenerateRequest(BaseModel):
    """Request for text generation."""
    prompt: str
    model: Optional[str] = None
    task_type: str = "general"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    stream: bool = False


class WorkflowGenerationRequest(BaseModel):
    """Request for workflow generation."""
    description: str
    context: Optional[Dict[str, Any]] = None


class StructuredGenerationRequest(BaseModel):
    """Request for structured generation."""
    model_config = ConfigDict(protected_namespaces=())  # Allow schema field name
    
    prompt: str
    schema: Dict[str, Any]
    model: Optional[str] = None
    examples: Optional[List[Dict]] = None


class CostEstimateRequest(BaseModel):
    """Request for cost estimation."""
    prompt: str
    model: str


@router.post("/generate", response_model=LLMResponse)
async def generate_text(
    request: GenerateRequest,
    current_user: User = Depends(get_current_user)
) -> LLMResponse:
    """
    Generate text using the best available LLM.
    
    This endpoint:
    - Respects user's model preferences from settings
    - Automatically selects the best model if not specified
    - Supports caching for repeated requests
    - Tracks usage and costs
    """
    try:
        settings = get_settings()
        # Get user's LLM settings if available
        user_settings = UserLLMSettings(
            user_id=str(current_user.id),
            selected_model=request.model or settings.DEFAULT_LLM_MODEL,
            provider="ollama",
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 1000,
            stream_responses=request.stream
        )
        
        response = await llm_service.generate(
            prompt=request.prompt,
            user_settings=user_settings,
            model_override=request.model,
            task_type=request.task_type,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
            stream=request.stream
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@router.get("/models", response_model=List[ModelInfo])
async def get_available_models(
    current_user: User = Depends(get_current_user)
) -> List[ModelInfo]:
    """
    Get all available models across all providers.
    
    Returns information about:
    - Local Ollama models
    - API models (if configured)
    - Model capabilities and pricing
    """
    try:
        return await llm_service.get_available_models()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models: {str(e)}"
        )


@router.post("/estimate-cost", response_model=CostEstimate)
async def estimate_cost(
    request: CostEstimateRequest,
    current_user: User = Depends(get_current_user)
) -> CostEstimate:
    """
    Estimate the cost of a generation before making the request.
    
    Useful for:
    - Budget planning
    - Model selection based on cost
    - Warning users about expensive operations
    """
    try:
        return await llm_service.estimate_cost(
            prompt=request.prompt,
            model=request.model
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cost estimation failed: {str(e)}"
        )


@router.post("/workflow/generate", response_model=List[WorkflowStep])
async def generate_workflow_steps(
    request: WorkflowGenerationRequest,
    current_user: User = Depends(get_current_user)
) -> List[WorkflowStep]:
    """
    Generate workflow steps from natural language description.
    
    This is a specialized endpoint for workflow generation that:
    - Parses natural language into structured workflow steps
    - Identifies agents and templates to use
    - Maintains logical flow and dependencies
    """
    try:
        settings = get_settings()
        user_settings = UserLLMSettings(
            user_id=str(current_user.id),
            selected_model=settings.DEFAULT_LLM_MODEL,
            provider="ollama"
        )
        
        steps = await llm_service.generate_workflow_steps(
            prompt=request.description,
            user_settings=user_settings,
            context=request.context
        )
        
        return steps
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow generation failed: {str(e)}"
        )


@router.post("/structured/generate")
async def generate_structured(
    request: StructuredGenerationRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate structured output matching a JSON schema.
    
    This endpoint:
    - Ensures output matches the provided schema
    - Supports few-shot learning with examples
    - Useful for generating configuration files, specs, etc.
    """
    try:
        result = await llm_service.generate_structured(
            prompt=request.prompt,
            schema=request.schema,
            model=request.model,
            examples=request.examples
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Structured generation failed: {str(e)}"
        )


@router.get("/usage/stats", response_model=UsageStats)
async def get_usage_stats(
    current_user: User = Depends(get_current_user)
) -> UsageStats:
    """
    Get usage statistics for the current user.
    
    Returns:
    - Total tokens used
    - Total cost incurred
    - Breakdown by model and provider
    - Cache hit rate
    """
    try:
        stats = await llm_service.get_usage_stats(
            user_id=str(current_user.id)
        )
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage stats: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Check LLM service health.
    
    Returns:
    - Service status
    - Available models count
    - Cache status
    """
    try:
        models = await llm_service.get_available_models()
        
        return {
            "status": "healthy",
            "available_models": len(models),
            "cache_enabled": True,
            "providers": list(set(m.provider.value for m in models))
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """
    Get LLM service status (alias for health check).

    Used by the LLM Service adapter for connection verification.
    """
    return await health_check()


@router.get("/models/{model_name}/provider-status")
async def get_model_provider_status(
    model_name: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get provider configuration status for a specific model.

    Returns:
    - provider: Provider name (e.g., "Google Vertex AI", "Anthropic", "OpenAI")
    - adapter_type: Adapter type identifier (e.g., "google-vertex-ai", "claude", "openai")
    - configured: Whether the required adapter is configured
    - configuration_url: URL to configure the adapter in Integration Hub (if not configured)
    - status_message: Human-readable status message
    """
    try:
        # Map model names to provider information
        model_provider_map = {
            # Gemini models - Direct API (Google AI Studio)
            "gemini-2.0-flash": {
                "provider": "Google Gemini",
                "adapter_type": "google-gemini",
                "local": False
            },
            "gemini-2.5-flash": {
                "provider": "Google Gemini",
                "adapter_type": "google-gemini",
                "local": False
            },
            "gemini-1.5-pro": {
                "provider": "Google Gemini",
                "adapter_type": "google-gemini",
                "local": False
            },
            "gemini-pro": {
                "provider": "Google Gemini",
                "adapter_type": "google-gemini",
                "local": False
            },
            # Gemini models - via GCP Vertex AI
            "gemini-2.0-flash-vertex": {
                "provider": "Google Vertex AI",
                "adapter_type": "google-vertex-ai",
                "local": False
            },
            "gemini-2.5-flash-vertex": {
                "provider": "Google Vertex AI",
                "adapter_type": "google-vertex-ai",
                "local": False
            },
            "gemini-1.5-pro-vertex": {
                "provider": "Google Vertex AI",
                "adapter_type": "google-vertex-ai",
                "local": False
            },
            # Claude models
            "claude-3-haiku": {
                "provider": "Anthropic",
                "adapter_type": "claude",
                "local": False
            },
            "claude-3-sonnet": {
                "provider": "Anthropic",
                "adapter_type": "claude",
                "local": False
            },
            "claude-3-opus": {
                "provider": "Anthropic",
                "adapter_type": "claude",
                "local": False
            },
            # OpenAI models
            "gpt-3.5-turbo": {
                "provider": "OpenAI",
                "adapter_type": "openai",
                "local": False
            },
            "gpt-4": {
                "provider": "OpenAI",
                "adapter_type": "openai",
                "local": False
            },
            "gpt-4-turbo": {
                "provider": "OpenAI",
                "adapter_type": "openai",
                "local": False
            },
            # Cohere models
            "command": {
                "provider": "Cohere",
                "adapter_type": "cohere",
                "local": False
            },
            "command-light": {
                "provider": "Cohere",
                "adapter_type": "cohere",
                "local": False
            },
            # DeepSeek models
            "deepseek-chat": {
                "provider": "DeepSeek",
                "adapter_type": "deepseek",
                "local": False
            },
            "deepseek-reasoner": {
                "provider": "DeepSeek",
                "adapter_type": "deepseek",
                "local": False
            },
            # DashScope/Qwen models
            "qwen-turbo": {
                "provider": "DashScope (Alibaba)",
                "adapter_type": "dashscope",
                "local": False
            },
            "qwen-plus": {
                "provider": "DashScope (Alibaba)",
                "adapter_type": "dashscope",
                "local": False
            },
            "qwen-max": {
                "provider": "DashScope (Alibaba)",
                "adapter_type": "dashscope",
                "local": False
            },
            "qwen-long": {
                "provider": "DashScope (Alibaba)",
                "adapter_type": "dashscope",
                "local": False
            },
            # Ollama models (local)
            "llama3.1:8b-instruct-q4_K_M": {
                "provider": "Ollama (Local)",
                "adapter_type": "ollama",
                "local": True
            },
            "llama3.2:1b": {
                "provider": "Ollama (Local)",
                "adapter_type": "ollama",
                "local": True
            },
            "llama3.2:3b": {
                "provider": "Ollama (Local)",
                "adapter_type": "ollama",
                "local": True
            },
        }

        # Check if model exists in our map
        if model_name not in model_provider_map:
            # Unknown model - return generic info
            return {
                "provider": "Unknown",
                "adapter_type": None,
                "configured": False,
                "local": False,
                "status_message": f"Unknown model: {model_name}",
                "configuration_url": None
            }

        provider_info = model_provider_map[model_name]

        # For local Ollama models, check if Ollama is accessible
        if provider_info["local"]:
            try:
                # Check if Ollama is running
                models = await llm_service.generation_engine._get_ollama_models()
                configured = model_name in models or len(models) > 0

                return {
                    "provider": provider_info["provider"],
                    "adapter_type": provider_info["adapter_type"],
                    "configured": configured,
                    "local": True,
                    "status_message": "✅ Ollama is running locally" if configured else "⚠️ Ollama not detected",
                    "configuration_url": None
                }
            except Exception:
                return {
                    "provider": provider_info["provider"],
                    "adapter_type": provider_info["adapter_type"],
                    "configured": False,
                    "local": True,
                    "status_message": "⚠️ Ollama not running",
                    "configuration_url": None
                }

        # For API models, check if adapter is configured
        # In Community Edition, we can't check adapter registry, so assume not configured
        # Business/Enterprise editions will override this endpoint with adapter checking
        configured = False
        status_message = f"⚠️ Not configured - requires {provider_info['provider']} API key"
        configuration_url = f"/integrations?adapter={provider_info['adapter_type']}"

        return {
            "provider": provider_info["provider"],
            "adapter_type": provider_info["adapter_type"],
            "configured": configured,
            "local": False,
            "status_message": status_message,
            "configuration_url": configuration_url
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider status: {str(e)}"
        )