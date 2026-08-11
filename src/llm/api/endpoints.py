"""API endpoints for LLM module."""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.security import get_current_user
from core.database import get_db
from core.tenant_context import get_current_tenant_id
from core.exceptions import UpstreamResponseError
from llm.org_llm_settings import get_org_llm_settings
from services.llm_helpers import get_user_llm_settings
from models import User
from ..service import llm_service
from ..models import (
    LLMRequest, LLMResponse, ModelInfo, CostEstimate,
    UsageStats, WorkflowStep
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["llm"])


async def _load_llm_context(db: AsyncSession, user_id: str):
    """User prefs + org settings for canonical resolution.

    Org settings load is wrapped here and never fails the request (falls back
    to None on any exception). User prefs load (get_user_llm_settings) is not
    wrapped here — it doesn't need to be, since its own DB fetch is internally
    guarded and degrades to unset preferences rather than raising.
    """
    user_settings = await get_user_llm_settings(db=db, user_id=user_id)
    org_settings = None
    try:
        org_settings = await get_org_llm_settings(get_current_tenant_id(), db)
    except Exception as e:
        logger.debug(f"Org LLM settings unavailable: {e}")
    return user_settings, org_settings


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
    model: Optional[str] = None


class StructuredGenerationRequest(BaseModel):
    """Request for structured generation."""
    # `populate_by_name` lets Python code use either field name or alias;
    # `protected_namespaces=()` allows `model` as a field.
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    prompt: str
    # Pydantic v2 emits a UserWarning if a field shadows `BaseModel.schema`.
    # Use a trailing underscore with `alias='schema'` so the JSON API is
    # unchanged but the class attribute doesn't shadow the base class.
    schema_: Dict[str, Any] = Field(alias='schema')
    model: Optional[str] = None
    examples: Optional[List[Dict]] = None


class CostEstimateRequest(BaseModel):
    """Request for cost estimation."""
    prompt: str
    model: str


@router.post("/generate", response_model=LLMResponse)
async def generate_text(
    request: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
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
        user_settings, org_settings = await _load_llm_context(db, str(current_user.id))
        user_settings.temperature = request.temperature or user_settings.temperature
        user_settings.max_tokens = request.max_tokens or user_settings.max_tokens
        user_settings.stream_responses = request.stream

        response = await llm_service.generate(
            prompt=request.prompt,
            user_settings=user_settings,
            model_override=request.model,
            task_type=request.task_type,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
            stream=request.stream,
            org_settings=org_settings
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@router.get("/models/system-defaults")
async def get_system_defaults(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get the system default model for each tier."""
    return await llm_service.generation_engine.get_system_tier_defaults()


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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[WorkflowStep]:
    """
    Generate workflow steps from natural language description.

    This is a specialized endpoint for workflow generation that:
    - Parses natural language into structured workflow steps
    - Identifies agents and templates to use
    - Maintains logical flow and dependencies
    """
    try:
        user_settings, org_settings = await _load_llm_context(db, str(current_user.id))

        steps = await llm_service.generate_workflow_steps(
            prompt=request.description,
            user_settings=user_settings,
            context=request.context,
            model=request.model,
            org_settings=org_settings
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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Generate structured output matching a JSON schema.

    This endpoint:
    - Ensures output matches the provided schema
    - Supports few-shot learning with examples
    - Useful for generating configuration files, specs, etc.
    """
    try:
        user_settings, org_settings = await _load_llm_context(db, str(current_user.id))

        result = await llm_service.generate_structured(
            prompt=request.prompt,
            schema=request.schema_,
            model=request.model,
            examples=request.examples,
            user_settings=user_settings,
            org_settings=org_settings
        )

        return result

    except UpstreamResponseError as e:
        # 502: the model returned something unparseable. Previously this path
        # answered 200 with {} — success reported, nothing delivered.
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=e.message)
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


@router.get("/models/{model_name:path}/provider-status")
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
        # vLLM is routed via the vllm: prefix; any model name with that prefix
        # is served by a self-hosted vLLM instance regardless of the underlying model.
        if model_name.startswith("vllm:"):
            return {
                "provider": "vLLM (Self-Hosted)",
                "adapter_type": "vllm",
                "configured": True,
                "local": True,
                "status_message": "✅ vLLM model (self-hosted)",
                "configuration_url": None,
            }

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
            # DashScope/Qwen models (adapter is Business-only; info shown for model listing)
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
        if model_name in model_provider_map:
            provider_info = model_provider_map[model_name]
        else:
            # Not in static map — check if it's a dynamically discovered Ollama model
            try:
                ollama_models = await llm_service.generation_engine._get_ollama_models()
                if model_name in ollama_models:
                    provider_info = {
                        "provider": "Ollama (Local)",
                        "adapter_type": "ollama",
                        "local": True,
                    }
                else:
                    # Truly unknown model
                    return {
                        "provider": "Unknown",
                        "adapter_type": None,
                        "configured": False,
                        "local": False,
                        "status_message": f"Model not found: {model_name}",
                        "configuration_url": None,
                    }
            except Exception:
                return {
                    "provider": "Unknown",
                    "adapter_type": None,
                    "configured": False,
                    "local": False,
                    "status_message": f"Unknown model: {model_name}",
                    "configuration_url": None,
                }

        # For local Ollama models, check if Ollama is accessible
        if provider_info["local"]:
            try:
                # Check if Ollama is running
                models = await llm_service.generation_engine._get_ollama_models()
                configured = model_name in models

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
        configuration_url = f"/adapters/configure/{provider_info['adapter_type']}"

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