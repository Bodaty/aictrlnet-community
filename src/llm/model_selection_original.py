"""Model selection logic for LLM module."""

from typing import Dict, Any, Optional, List, Tuple
import logging
import httpx

from .models import ModelTier, ModelProvider

logger = logging.getLogger(__name__)


# Model configurations by tier
MODEL_CONFIGS = {
    ModelTier.FAST: {
        "models": ["llama3.2:1b", "phi3:mini", "gemma:2b"],
        "temperature": 0.1,
        "max_tokens": 500,
        "timeout": 30
    },
    ModelTier.BALANCED: {
        "models": ["llama3.2:3b", "mistral:7b", "llama3.1:8b-instruct-q4_K_M"],
        "temperature": 0.3,
        "max_tokens": 1000,
        "timeout": 60
    },
    ModelTier.QUALITY: {
        "models": ["llama3.1:8b-instruct-q4_K_M", "llama3.1:70b", "mixtral:8x7b"],
        "temperature": 0.5,
        "max_tokens": 2000,
        "timeout": 120
    },
    ModelTier.PREMIUM: {
        "models": ["claude-3-sonnet", "claude-3-opus", "gpt-4", "gpt-4-turbo"],
        "temperature": 0.7,
        "max_tokens": 4000,
        "timeout": 180
    }
}


def estimate_complexity(prompt: str) -> float:
    """
    Estimate the complexity of a prompt (0-1).
    
    Args:
        prompt: The user's prompt
        
    Returns:
        Complexity score between 0 and 1
    """
    complexity = 0.3  # Base complexity
    
    # Length factor
    word_count = len(prompt.split())
    if word_count > 50:
        complexity += 0.2
    elif word_count > 20:
        complexity += 0.1
    
    # Complexity indicators
    complex_indicators = [
        'multiple', 'complex', 'sophisticated', 'advanced',
        'integrate', 'coordinate', 'orchestrate', 'analyze',
        'machine learning', 'ai', 'predictive', 'optimize'
    ]
    
    prompt_lower = prompt.lower()
    for indicator in complex_indicators:
        if indicator in prompt_lower:
            complexity += 0.05
    
    # Workflow indicators
    if any(word in prompt_lower for word in ['workflow', 'process', 'pipeline', 'automation']):
        complexity += 0.1
    
    # Multi-step indicators
    if any(word in prompt_lower for word in ['then', 'after', 'before', 'finally', 'steps']):
        complexity += 0.1
    
    return min(complexity, 1.0)


def select_model_for_task(
    task_type: str, 
    complexity: float = 0.5, 
    available_models: List[str] = None
) -> Tuple[str, ModelTier]:
    """
    Select appropriate model based on task type and complexity.
    
    Args:
        task_type: Type of task (intent_detection, workflow_generation, etc.)
        complexity: Estimated complexity (0-1)
        available_models: Optional list of available models to check against
        
    Returns:
        Tuple of (model_name, tier)
    """
    # Task-specific model selection
    if task_type == "intent_detection":
        # Intent detection can use fast models
        tier = ModelTier.FAST
    elif task_type == "template_matching":
        # Template matching needs balanced performance
        tier = ModelTier.BALANCED
    elif task_type == "workflow_generation":
        # Workflow generation needs quality for complex requests
        if complexity > 0.7:
            tier = ModelTier.QUALITY
        elif complexity > 0.4:
            tier = ModelTier.BALANCED
        else:
            tier = ModelTier.FAST
    elif task_type == "spec_generation":
        # Spec generation needs higher quality
        if complexity > 0.5:
            tier = ModelTier.QUALITY
        else:
            tier = ModelTier.BALANCED
    elif task_type == "code_generation":
        # Code generation often needs premium models
        tier = ModelTier.PREMIUM if complexity > 0.7 else ModelTier.QUALITY
    else:
        # Default to balanced
        tier = ModelTier.BALANCED
    
    # Get models for this tier
    models = MODEL_CONFIGS[tier]["models"]
    
    # If we have a list of available models, find the best match
    if available_models:
        for model in models:
            if model in available_models:
                logger.info(f"Selected {model} ({tier.value} tier) for {task_type} with complexity {complexity:.2f}")
                return model, tier
    
    # Return the first model in the tier as fallback
    model = models[0]
    logger.info(f"Using default {model} ({tier.value} tier) for {task_type} with complexity {complexity:.2f}")
    return model, tier


def get_model_config(tier: ModelTier) -> Dict[str, Any]:
    """Get configuration for a model tier."""
    return MODEL_CONFIGS.get(tier, MODEL_CONFIGS[ModelTier.BALANCED])


async def get_ai_complexity_assessment(
    prompt: str, 
    ollama_url: str = "http://host.docker.internal:11434"
) -> str:
    """
    Use AI to assess the complexity of a prompt.
    
    Args:
        prompt: The user's prompt
        ollama_url: URL of the Ollama service
        
    Returns:
        Complexity classification: "SIMPLE", "MEDIUM", or "COMPLEX"
    """
    try:
        classification_prompt = f'Classify the complexity of this request as SIMPLE, MEDIUM, or COMPLEX (respond with just one word): "{prompt}"'
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": classification_prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "options": {"num_predict": 10}  # We only need one word
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                classification = result.get("response", "").strip().upper()
                
                # Clean up response (sometimes includes punctuation)
                classification = classification.rstrip('.,!? ')
                
                if classification in ["SIMPLE", "MEDIUM", "COMPLEX"]:
                    logger.debug(f"AI complexity assessment for '{prompt[:50]}...': {classification}")
                    return classification
                else:
                    logger.warning(f"AI returned unexpected classification: {classification}")
                    return "MEDIUM"  # Safe default
            else:
                logger.warning(f"AI complexity assessment failed with status {response.status_code}")
                return "MEDIUM"
                
    except Exception as e:
        logger.debug(f"AI complexity assessment failed: {e}")
        return "MEDIUM"  # Safe default on error


async def estimate_complexity_hybrid(
    prompt: str, 
    ollama_url: str = "http://host.docker.internal:11434"
) -> float:
    """
    Hybrid complexity estimation using both keyword analysis and AI assessment.
    
    Args:
        prompt: The user's prompt
        ollama_url: URL of the Ollama service
        
    Returns:
        Complexity score between 0 and 1
    """
    # First, do keyword-based calculation
    keyword_complexity = estimate_complexity(prompt)
    
    # Then get AI assessment
    ai_assessment = await get_ai_complexity_assessment(prompt, ollama_url)
    
    # Convert AI assessment to numeric
    ai_complexity = {
        "SIMPLE": 0.2,
        "MEDIUM": 0.5,
        "COMPLEX": 0.8
    }.get(ai_assessment, 0.5)
    
    # Combine both assessments with weighted average
    # AI gets slightly more weight as it understands context better
    final_complexity = (keyword_complexity * 0.4) + (ai_complexity * 0.6)
    
    logger.debug(f"Hybrid complexity: keyword={keyword_complexity:.2f}, AI={ai_complexity:.2f}, final={final_complexity:.2f}")
    
    return final_complexity


def classify_model_tier(model: str) -> ModelTier:
    """
    Classify a model into a tier based on its name.
    
    Args:
        model: Model name
        
    Returns:
        ModelTier enum value
    """
    model_lower = model.lower()
    
    if any(x in model_lower for x in ["1b", "2b", "mini", "tiny"]):
        return ModelTier.FAST
    elif any(x in model_lower for x in ["3b", "7b", "mistral"]):
        return ModelTier.BALANCED
    elif any(x in model_lower for x in ["8b", "13b", "mixtral"]):
        return ModelTier.QUALITY
    elif any(x in model_lower for x in ["claude", "gpt", "70b"]):
        return ModelTier.PREMIUM
    else:
        return ModelTier.BALANCED


def get_provider_from_model(model: str) -> ModelProvider:
    """
    Determine provider from model name.
    
    Args:
        model: Model name
        
    Returns:
        ModelProvider enum value
    """
    model_lower = model.lower()
    
    if "claude" in model_lower:
        return ModelProvider.ANTHROPIC
    elif "gpt" in model_lower:
        return ModelProvider.OPENAI
    elif "gemini" in model_lower:
        return ModelProvider.GEMINI
    elif "cohere" in model_lower or "command" in model_lower:
        return ModelProvider.COHERE
    elif "titan" in model_lower or "bedrock" in model_lower:
        return ModelProvider.BEDROCK
    elif "bison" in model_lower or "vertex" in model_lower:
        return ModelProvider.VERTEX_AI
    elif any(x in model_lower for x in ["llama", "mistral", "phi", "mixtral"]):
        return ModelProvider.OLLAMA
    else:
        return ModelProvider.OLLAMA