"""
Performance optimization utilities for NLP workflow generation.

This module provides model selection, caching, and other optimizations
to improve workflow generation performance.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import httpx
import asyncio

logger = logging.getLogger(__name__)


class ModelTier:
    """Model tier definitions for performance optimization."""
    FAST = "fast"          # < 1s response time
    BALANCED = "balanced"  # 2-5s response time  
    QUALITY = "quality"    # 10-20s response time


# Model configurations by tier
# Note: If only one model is available, it will be used for all tiers with different parameters
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


def select_model_for_task(task_type: str, complexity: float = 0.5, available_models: List[str] = None) -> Tuple[str, str]:
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
    else:
        # Default to balanced
        tier = ModelTier.BALANCED
    
    # Get models for this tier
    models = MODEL_CONFIGS[tier]["models"]
    
    # If we have a list of available models, find the best match
    if available_models:
        for model in models:
            if model in available_models:
                logger.info(f"Selected {model} ({tier} tier) for {task_type} with complexity {complexity:.2f}")
                return model, tier
    
    # Return the first model in the tier as fallback
    model = models[0]
    logger.info(f"Using default {model} ({tier} tier) for {task_type} with complexity {complexity:.2f}")
    return model, tier


def get_model_config(tier: str) -> Dict[str, Any]:
    """Get configuration for a model tier."""
    return MODEL_CONFIGS.get(tier, MODEL_CONFIGS[ModelTier.BALANCED])


async def get_ai_complexity_assessment(prompt: str, ollama_url: str = "http://host.docker.internal:11434") -> str:
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


async def estimate_complexity_hybrid(prompt: str, ollama_url: str = "http://host.docker.internal:11434") -> float:
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
    
    # Fast path: If already high complexity, skip AI
    if keyword_complexity >= 0.7:
        logger.info(f"High keyword complexity ({keyword_complexity:.2f}), skipping AI assessment")
        return keyword_complexity
    
    # Fast path: If very simple and short, skip AI
    word_count = len(prompt.split())
    if word_count < 5 and keyword_complexity < 0.4:
        logger.info(f"Short simple prompt ({word_count} words, {keyword_complexity:.2f}), skipping AI assessment")
        return keyword_complexity
    
    # For ambiguous cases, consult AI
    try:
        ai_classification = await get_ai_complexity_assessment(prompt, ollama_url)
        
        # Combine both signals
        if ai_classification == "COMPLEX" and keyword_complexity < 0.5:
            # AI says complex but keywords say simple - boost it
            final_complexity = 0.8
            logger.info(f"AI override: '{prompt[:30]}...' keyword={keyword_complexity:.2f}, AI={ai_classification}, final=0.8")
        elif ai_classification == "SIMPLE" and keyword_complexity > 0.5:
            # AI says simple but keywords say complex - lower it
            final_complexity = 0.3
            logger.info(f"AI override: '{prompt[:30]}...' keyword={keyword_complexity:.2f}, AI={ai_classification}, final=0.3")
        elif ai_classification == "MEDIUM":
            # AI says medium - use balanced tier threshold
            final_complexity = 0.5
            logger.info(f"AI suggests medium: '{prompt[:30]}...' keyword={keyword_complexity:.2f}, final=0.5")
        else:
            # Agreement or AI failed - trust keywords
            final_complexity = keyword_complexity
            logger.info(f"Using keyword complexity: '{prompt[:30]}...' keyword={keyword_complexity:.2f}, AI={ai_classification}")
            
        return final_complexity
        
    except Exception as e:
        logger.debug(f"Hybrid complexity failed, using keyword only: {e}")
        return keyword_complexity