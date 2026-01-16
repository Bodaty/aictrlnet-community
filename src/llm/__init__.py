"""
LLM module for unified text generation.

This module provides a centralized interface for all LLM operations,
using the existing AI adapters from Community/Business/Enterprise editions.
"""

from .service import LLMService, llm_service
from .client import LLMClient, llm_client
from .models import (
    ModelProvider,
    ModelTier,
    ModelInfo,
    UserLLMSettings,
    LLMRequest,
    LLMResponse,
    WorkflowStep,
    CostEstimate,
    UsageStats
)
from .model_selection import (
    select_model_for_task,
    estimate_complexity,
    estimate_complexity_hybrid,
    get_model_config
)

__all__ = [
    # Service and client
    'LLMService',
    'llm_service',
    'LLMClient',
    'llm_client',
    
    # Models
    'ModelProvider',
    'ModelTier',
    'ModelInfo',
    'UserLLMSettings',
    'LLMRequest',
    'LLMResponse',
    'WorkflowStep',
    'CostEstimate',
    'UsageStats',
    
    # Utilities
    'select_model_for_task',
    'estimate_complexity',
    'estimate_complexity_hybrid',
    'get_model_config'
]