"""Internal client for LLM service usage."""

from typing import Optional, List, Dict, Any
from .service import llm_service
from .models import (
    LLMRequest, LLMResponse, UserLLMSettings,
    WorkflowStep, CostEstimate, UsageStats
)


class LLMClient:
    """
    Internal client for using the LLM service.
    
    This provides a simple interface for other modules to use LLM functionality
    without needing to know about the internal implementation details.
    """
    
    def __init__(self):
        """Initialize the client."""
        self.service = llm_service  # Use singleton instance
    
    async def generate(
        self,
        prompt: str,
        task_type: str = "general",
        model: Optional[str] = None,
        user_settings: Optional[UserLLMSettings] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using LLM.
        
        Args:
            prompt: The prompt to generate from
            task_type: Type of task (workflow_generation, spec_generation, etc.)
            model: Optional model override
            user_settings: Optional user settings
            **kwargs: Additional parameters passed to service
            
        Returns:
            LLM response with generated text
        """
        return await self.service.generate(
            prompt=prompt,
            task_type=task_type,
            model_override=model,
            user_settings=user_settings,
            **kwargs
        )
    
    async def generate_workflow_steps(
        self,
        prompt: str,
        user_settings: Optional[UserLLMSettings] = None
    ) -> List[WorkflowStep]:
        """
        Generate workflow steps from natural language.
        
        Args:
            prompt: Natural language description
            user_settings: Optional user settings
            
        Returns:
            List of workflow steps
        """
        return await self.service.generate_workflow_steps(
            prompt=prompt,
            user_settings=user_settings
        )
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured output matching a schema.
        
        Args:
            prompt: The prompt
            schema: Target schema
            model: Optional model override
            
        Returns:
            Structured data
        """
        return await self.service.generate_structured(
            prompt=prompt,
            schema=schema,
            model=model
        )
    
    async def estimate_cost(
        self,
        prompt: str,
        model: str
    ) -> CostEstimate:
        """
        Estimate cost for a generation.
        
        Args:
            prompt: The prompt
            model: The model to use
            
        Returns:
            Cost estimate
        """
        return await self.service.estimate_cost(prompt, model)
    
    async def get_usage_stats(
        self,
        user_id: Optional[str] = None
    ) -> UsageStats:
        """
        Get usage statistics.
        
        Args:
            user_id: Optional user ID filter
            
        Returns:
            Usage statistics
        """
        return await self.service.get_usage_stats(user_id)


# Global client instance for convenience
llm_client = LLMClient()