"""Enhanced model selection logic for LLM module with sophisticated scoring.

This module combines the original model_selection.py with the advanced
scoring logic from Business Edition's LLM Router.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import re
import httpx
from datetime import datetime

from .models import ModelTier, ModelProvider

logger = logging.getLogger(__name__)


# Model configurations by tier (existing)
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


class EnhancedModelSelector:
    """Enhanced model selector with sophisticated scoring and dynamic discovery."""
    
    def __init__(self):
        """Initialize the enhanced selector."""
        self._models_cache = {}
        self._performance_metrics = {}
        self._routing_history = []
        self._last_registry_check = None
        
        # Quality scores for different adapters/providers
        self.quality_scores = {
            "openai": 1.0,
            "claude": 0.95,
            "gemini": 0.9,
            "llm-service": 0.85,
            "huggingface": 0.8,
            "ollama": 0.75,
            "mcp-client": 0.7,
            "custom": 0.6
        }
        
        # Cascade order for failover
        self.cascade_order = [
            "llm-service",      # Priority 1: Internal (cheapest, fastest)
            "ollama",           # Priority 2: Local models (free, moderate speed)
            "huggingface",      # Priority 3: Model registry (varied cost/speed)
            "openai",           # Priority 4: Cloud providers (expensive, high quality)
            "claude",
            "gemini",
            "mcp-client"        # Priority 5: External MCP (last resort)
        ]
    
    async def discover_models(self) -> Dict[str, Any]:
        """Discover available models from adapter registry.
        
        Dynamically discovers models from all available adapters.
        """
        models = {}
        
        # Try to use adapter registry if available
        try:
            from adapters.registry import adapter_registry
            from adapters.models import AdapterCategory
            
            # Get all AI adapters
            for adapter_name in self.cascade_order:
                try:
                    # Get adapter class from registry
                    adapter_class = adapter_registry.get_adapter_class(adapter_name)
                    if adapter_class:
                        # Create adapter instance
                        adapter = adapter_class({})
                        # Check health
                        health = await adapter.health_check()
                        if health.get("status") == "healthy":
                            # Add models from this adapter
                            models.update(self._get_adapter_models(adapter_name))
                except:
                    continue
                    
        except ImportError:
            # Adapter registry not available, use static list
            pass
        
        # If no models discovered, return static fallback
        if not models:
            models = {
                "ollama-llama3.2:3b": {
                    "name": "Llama 3.2 3B (Local)",
                    "provider": "ollama",
                    "capabilities": ["text-generation", "chat"],
                    "max_tokens": 4096,
                    "cost_per_1k_tokens": 0.0,
                    "adapter": "ollama"
                },
                "openai-gpt4": {
                    "name": "GPT-4",
                    "provider": "openai",
                    "capabilities": ["text-generation", "chat", "code-generation", "function-calling"],
                    "max_tokens": 8192,
                    "cost_per_1k_tokens": 0.03,
                    "adapter": "openai"
                },
                "claude-3": {
                    "name": "Claude 3",
                    "provider": "claude",
                    "capabilities": ["text-generation", "chat", "analysis", "code-generation"],
                    "max_tokens": 100000,
                    "cost_per_1k_tokens": 0.025,
                    "adapter": "claude"
                }
            }
        
        return models
    
    def _get_adapter_models(self, adapter_name: str) -> Dict[str, Any]:
        """Get model definitions for a specific adapter."""
        models = {}
        
        if adapter_name == "ollama":
            models[f"{adapter_name}-llama2"] = {
                "name": "Llama 2",
                "provider": adapter_name,
                "capabilities": ["text-generation", "chat"],
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.0,
                "adapter": adapter_name
            }
            models[f"{adapter_name}-codellama"] = {
                "name": "Code Llama",
                "provider": adapter_name,
                "capabilities": ["code-generation"],
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.0,
                "adapter": adapter_name
            }
        
        elif adapter_name == "openai":
            models[f"{adapter_name}-gpt4"] = {
                "name": "GPT-4",
                "provider": adapter_name,
                "capabilities": ["text-generation", "chat", "code-generation", "function-calling"],
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.03,
                "adapter": adapter_name
            }
            models[f"{adapter_name}-gpt35"] = {
                "name": "GPT-3.5 Turbo",
                "provider": adapter_name,
                "capabilities": ["text-generation", "chat", "code-generation"],
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.002,
                "adapter": adapter_name
            }
        
        elif adapter_name == "claude":
            models[f"{adapter_name}-claude3"] = {
                "name": "Claude 3",
                "provider": adapter_name,
                "capabilities": ["text-generation", "chat", "analysis", "code-generation"],
                "max_tokens": 100000,
                "cost_per_1k_tokens": 0.025,
                "adapter": adapter_name
            }
        
        # Add more adapters as needed
        
        return models
    
    def score_model(
        self,
        model: Dict[str, Any],
        task_type: str,
        complexity: float,
        requirements: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score a model based on task requirements and preferences.
        
        Scoring factors:
        - Capability match (40%)
        - Cost efficiency (25%)
        - Performance/Quality (20%)
        - Availability (10%)
        - Locality preference (5%)
        
        Args:
            model: Model information dict
            task_type: Type of task
            complexity: Task complexity (0-1)
            requirements: Optional requirements dict with:
                - capabilities: List of required capabilities
                - max_cost: Maximum cost per 1k tokens
                - min_tokens: Minimum token requirement
                - prefer_local: Prefer local models
                - quality_threshold: Minimum quality threshold
                
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        requirements = requirements or {}
        
        # Extract requirements
        required_capabilities = requirements.get("capabilities", [])
        max_cost = requirements.get("max_cost_per_1k_tokens", float("inf"))
        min_tokens = requirements.get("min_tokens", 1024)
        prefer_local = requirements.get("prefer_local", False)
        quality_threshold = requirements.get("quality_threshold", 0.7)
        
        # 1. Capability match (40%)
        model_capabilities = model.get("capabilities", [])
        
        # Add task-specific capability requirements
        if task_type == "code_generation":
            required_capabilities.append("code-generation")
        elif task_type == "workflow_generation":
            required_capabilities.extend(["text-generation", "chat"])
        
        if required_capabilities:
            capability_match = sum(
                1 for cap in required_capabilities
                if cap in model_capabilities
            ) / len(required_capabilities)
            score += capability_match * 0.4
        else:
            score += 0.4  # No specific requirements
        
        # 2. Cost efficiency (25%)
        model_cost = model.get("cost_per_1k_tokens", 0.01)
        if model_cost <= max_cost:
            if model_cost == 0:  # Free models get full score
                score += 0.25
            else:
                # Lower cost = higher score
                cost_score = (max_cost - model_cost) / max_cost
                score += cost_score * 0.25
        else:
            return 0  # Exceeds budget
        
        # 3. Token capacity check
        if model.get("max_tokens", 0) < min_tokens:
            return 0  # Doesn't meet minimum requirement
        
        # 4. Performance/Quality (20%)
        adapter = model.get("adapter", model.get("provider", "unknown"))
        quality = self.quality_scores.get(adapter, 0.5)
        
        # Adjust quality based on complexity
        if complexity > 0.7:
            # High complexity tasks need high quality models
            if quality < 0.8:
                quality *= 0.5  # Penalize low quality for complex tasks
        
        if quality >= quality_threshold:
            score += quality * 0.2
        
        # 5. Availability (10%)
        # For now, assume all discovered models are available
        score += 0.1
        
        # 6. Locality preference (5%)
        if prefer_local and adapter in ["ollama", "llm-service"]:
            score += 0.05
        
        # 7. Performance history bonus (up to 10% extra)
        model_id = f"{adapter}-{model.get('name', 'unknown')}"
        if model_id in self._performance_metrics:
            perf = self._performance_metrics[model_id]
            success_rate = perf.get("success_rate", 0.5)
            score += success_rate * 0.1
        
        return min(score, 1.0)
    
    async def select_model(
        self,
        task_type: str,
        prompt: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Select the best model for a task using enhanced scoring.
        
        Args:
            task_type: Type of task
            prompt: The prompt (for complexity estimation)
            requirements: Optional requirements dict
            
        Returns:
            Tuple of (model_id, routing_metadata)
        """
        # Estimate complexity
        complexity = estimate_complexity_enhanced(prompt, task_type)
        
        # Discover available models
        models = await self.discover_models()
        
        # Score each model
        scored_models = []
        for model_id, model_info in models.items():
            score = self.score_model(model_info, task_type, complexity, requirements)
            if score > 0:
                scored_models.append((model_id, model_info, score))
        
        if not scored_models:
            # Fallback to first available model
            if models:
                first_id = list(models.keys())[0]
                return first_id, {
                    "fallback": True,
                    "reason": "No models met requirements"
                }
            raise RuntimeError("No models available")
        
        # Sort by score and select best
        scored_models.sort(key=lambda x: x[2], reverse=True)
        selected_id, selected_info, score = scored_models[0]
        
        # Record routing decision
        routing_metadata = {
            "model_id": selected_id,
            "model_name": selected_info.get("name"),
            "score": score,
            "complexity": complexity,
            "task_type": task_type,
            "alternatives": [m[0] for m in scored_models[1:5]],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._routing_history.append(routing_metadata)
        if len(self._routing_history) > 100:
            self._routing_history.pop(0)
        
        return selected_id, routing_metadata
    
    def update_performance_metrics(
        self,
        model_id: str,
        success: bool,
        latency_ms: float,
        tokens_used: int
    ):
        """Update performance metrics for a model based on actual usage."""
        if model_id not in self._performance_metrics:
            self._performance_metrics[model_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_latency_ms": 0,
                "total_tokens": 0
            }
        
        metrics = self._performance_metrics[model_id]
        metrics["total_requests"] += 1
        if success:
            metrics["successful_requests"] += 1
        metrics["total_latency_ms"] += latency_ms
        metrics["total_tokens"] += tokens_used
        
        # Calculate derived metrics
        metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
        metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["total_requests"]
        metrics["avg_tokens_per_request"] = metrics["total_tokens"] / metrics["total_requests"]


def estimate_complexity_enhanced(prompt: str, task_type: str = "general") -> float:
    """Enhanced complexity estimation with task-type awareness.
    
    Args:
        prompt: The user's prompt
        task_type: Type of task for context
        
    Returns:
        Complexity score between 0 and 1
    """
    complexity = 0.3  # Base complexity
    
    # Task type base complexity
    task_complexities = {
        "workflow_generation": 0.5,
        "code_generation": 0.6,
        "analysis": 0.7,
        "intent_detection": 0.2,
        "chat": 0.3,
        "general": 0.4
    }
    complexity = task_complexities.get(task_type, 0.4)
    
    # Length factor
    word_count = len(prompt.split())
    if word_count > 100:
        complexity += 0.3
    elif word_count > 50:
        complexity += 0.2
    elif word_count > 20:
        complexity += 0.1
    
    # Complexity indicators
    complex_indicators = [
        'multiple', 'complex', 'sophisticated', 'advanced',
        'integrate', 'coordinate', 'orchestrate', 'analyze',
        'machine learning', 'ai', 'predictive', 'optimize',
        'architecture', 'design', 'implement', 'refactor'
    ]
    
    prompt_lower = prompt.lower()
    for indicator in complex_indicators:
        if indicator in prompt_lower:
            complexity += 0.05
    
    # Multi-step indicators
    if any(word in prompt_lower for word in ['then', 'after', 'before', 'finally', 'steps', 'phase']):
        complexity += 0.15
    
    # Code-related complexity
    if task_type == "code_generation" or "code" in prompt_lower:
        if any(word in prompt_lower for word in ['class', 'function', 'async', 'await', 'import']):
            complexity += 0.1
    
    return min(complexity, 1.0)


# Backward compatibility functions
def estimate_complexity(prompt: str) -> float:
    """Original complexity estimation for backward compatibility."""
    return estimate_complexity_enhanced(prompt, "general")


def select_model_for_task(
    task_type: str,
    complexity: float = 0.5,
    available_models: List[str] = None
) -> Tuple[str, ModelTier]:
    """Original model selection for backward compatibility.

    This maintains the original interface but uses simplified logic.
    Falls back to adjacent tiers and then ANY available model before
    resorting to a hardcoded default.
    """
    # Map complexity to tier
    if complexity < 0.3:
        tier = ModelTier.FAST
    elif complexity < 0.5:
        tier = ModelTier.BALANCED
    elif complexity < 0.7:
        tier = ModelTier.QUALITY
    else:
        tier = ModelTier.PREMIUM

    config = MODEL_CONFIGS[tier]
    models = config["models"]

    # Filter by available models if provided (use `is not None` to distinguish
    # "no filter" from "empty list means Ollama is down")
    if available_models is not None:
        models = [m for m in models if m in available_models]

    if models:
        return models[0], tier

    # Try adjacent tiers before giving up
    tier_order = [ModelTier.QUALITY, ModelTier.BALANCED, ModelTier.FAST, ModelTier.PREMIUM]
    if available_models is not None and len(available_models) > 0:
        for fallback_tier in tier_order:
            fallback_models = [m for m in MODEL_CONFIGS[fallback_tier]["models"]
                              if m in available_models]
            if fallback_models:
                return fallback_models[0], fallback_tier

        # Still nothing in MODEL_CONFIGS — pick ANY available model
        best = available_models[0]
        return best, classify_model_tier(best)

    # Absolute last resort (no Ollama models at all)
    return "llama3.2:1b", ModelTier.FAST


def _estimate_model_size_billions(model_name: str) -> Optional[float]:
    """Extract approximate parameter count (billions) from model name.

    Ollama patterns: llama3.2:1b, mistral:7b, mixtral:8x7b, qwen2.5:0.5b, deepseek-coder:6.7b
    Returns None if size cannot be determined.
    """
    name_lower = model_name.lower()

    # MoE pattern: 8x7b → 56.0
    moe_match = re.search(r'(\d+)x(\d+\.?\d*)b', name_lower)
    if moe_match:
        return float(moe_match.group(1)) * float(moe_match.group(2))

    # Standard pattern: 1b, 3b, 7b, 0.5b, 6.7b
    size_match = re.search(r'(\d+\.?\d*)b', name_lower)
    if size_match:
        return float(size_match.group(1))

    return None  # Can't determine (e.g., "phi3:mini")


def classify_model_tier(model_name: str) -> ModelTier:
    """Classify a model into a tier.

    Priority:
    1. Exact match in MODEL_CONFIGS → that tier
    2. Cloud/API model patterns → PREMIUM or QUALITY
    3. Size-based (parse parameter count from name):
       ≤2B → FAST, 3-8B → BALANCED, 9B+ → QUALITY
    4. Default → BALANCED (safest middle ground)
    """
    # 1. Exact match
    for tier, config in MODEL_CONFIGS.items():
        if model_name in config["models"]:
            return tier

    model_lower = model_name.lower()

    # 2. Cloud/API model patterns
    premium_patterns = ["claude-3-opus", "gpt-4", "gemini-1.5-pro", "gemini-2"]
    quality_patterns = [
        "claude-3-sonnet", "claude-3-haiku", "gpt-3.5", "command",
        "deepseek-chat", "deepseek-reasoner", "qwen-max", "qwen-plus",
    ]
    if any(p in model_lower for p in premium_patterns):
        return ModelTier.PREMIUM
    if any(p in model_lower for p in quality_patterns):
        return ModelTier.QUALITY

    # 3. Size-based for Ollama models
    size = _estimate_model_size_billions(model_name)
    if size is not None:
        if size <= 2.0:
            return ModelTier.FAST
        elif size <= 8.0:
            return ModelTier.BALANCED
        else:
            return ModelTier.QUALITY

    # 4. Default: BALANCED (middle ground)
    return ModelTier.BALANCED


def get_model_config(model: str) -> Dict[str, Any]:
    """Get configuration for a model."""
    tier = classify_model_tier(model)
    return MODEL_CONFIGS[tier]


def get_provider_from_model(model: str) -> ModelProvider:
    """Get provider from model name."""
    model_lower = model.lower()

    # Check for Vertex AI models FIRST - models with "-vertex" suffix use GCP Vertex AI API
    # (These require GCP credentials and use different API than Google AI Studio)
    if "-vertex" in model_lower or model_lower.endswith("vertex"):
        return ModelProvider.VERTEX_AI
    elif "gemini" in model_lower:
        # Standard Gemini models use Google AI Studio API
        return ModelProvider.GEMINI
    elif "claude" in model_lower:
        return ModelProvider.ANTHROPIC  # Use ANTHROPIC for Claude models
    elif "gpt" in model_lower:
        return ModelProvider.OPENAI
    elif model_lower in ("deepseek-chat", "deepseek-reasoner"):
        return ModelProvider.DEEPSEEK
    elif any(model_lower.startswith(p) for p in ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long"]):
        return ModelProvider.DASHSCOPE
    elif any(x in model_lower for x in ["llama", "mistral", "phi", "mixtral", "qwen", "deepseek"]):
        return ModelProvider.OLLAMA
    else:
        return ModelProvider.HUGGINGFACE


def estimate_complexity_hybrid(prompt: str, user_complexity: Optional[float] = None) -> float:
    """Hybrid complexity estimation combining automatic and user input."""
    auto_complexity = estimate_complexity(prompt)
    
    if user_complexity is not None:
        # Weighted average: 60% user, 40% automatic
        return (user_complexity * 0.6) + (auto_complexity * 0.4)
    
    return auto_complexity


# Global selector instance
_global_selector = None


def get_enhanced_selector() -> EnhancedModelSelector:
    """Get the global enhanced selector instance."""
    global _global_selector
    if _global_selector is None:
        _global_selector = EnhancedModelSelector()
    return _global_selector