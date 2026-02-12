"""LLM generation engine that uses existing adapters."""

import logging
import httpx
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

from .models import (
    ModelProvider, ModelTier, ModelInfo, UserLLMSettings,
    LLMRequest, LLMResponse, WorkflowStep, CostEstimate
)
from .model_selection import (
    select_model_for_task,
    estimate_complexity_hybrid,
    classify_model_tier,
    get_provider_from_model,
    get_enhanced_selector,
    EnhancedModelSelector
)
from .tier_resolver import (
    get_environment_default_model,
    is_ollama_model
)

logger = logging.getLogger(__name__)


def normalize_model_name(ui_model_name: str) -> str:
    """
    Normalize UI model names to actual Ollama model names.

    UI presents user-friendly names, but Ollama expects exact model names.
    """
    MODEL_NAME_MAP = {
        # UI name -> Ollama name
        "llama3.1-local": "llama3.1:8b-instruct-q4_K_M",
        "llama3.2-1b": "llama3.2:1b",
        "llama3.2-3b": "llama3.2:3b",
        "llama3.1:8b": "llama3.1:8b-instruct-q4_K_M",  # Also accept colon format
    }

    return MODEL_NAME_MAP.get(ui_model_name, ui_model_name)


class LLMGenerationEngine:
    """
    Core generation engine that interfaces with existing adapters.
    
    This engine directly uses the adapters already present in the
    Community/Business/Enterprise editions, avoiding duplication.
    """
    
    def __init__(self):
        """Initialize the generation engine."""
        self.ollama_url = "http://host.docker.internal:11434"
        self._available_models_cache = None
        self._models_cache_time = 0
        self._adapters = {}
        # Use enhanced selector for sophisticated routing
        self.enhanced_selector = get_enhanced_selector()
        
        # Model pricing (per 1K tokens)
        self.pricing = {
            # Ollama models (free - local)
            "llama3.2:1b": 0.0,
            "llama3.2:3b": 0.0,
            "llama3.1:8b-instruct-q4_K_M": 0.0,
            "llama3.1:70b": 0.0,
            "phi3:mini": 0.0,
            "mistral:7b": 0.0,
            "mixtral:8x7b": 0.0,
            
            # Claude models
            "claude-3-haiku": 0.00025,
            "claude-3-sonnet": 0.003,
            "claude-3-opus": 0.015,
            
            # OpenAI models  
            "gpt-3.5-turbo": 0.0015,
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            
            # Gemini models (Direct API pricing - same as Vertex)
            "gemini-2.0-flash": 0.000035,   # $0.035 per 1M tokens
            "gemini-2.5-flash": 0.000035,   # $0.035 per 1M tokens
            "gemini-1.5-pro": 0.00125,       # $1.25 per 1M tokens (input)
            "gemini-pro": 0.00025,
            "gemini-ultra": 0.007,

            # Gemini via Vertex AI (same pricing)
            "gemini-2.0-flash-vertex": 0.000035,
            "gemini-2.5-flash-vertex": 0.000035,
            "gemini-1.5-pro-vertex": 0.00125,
            
            # Cohere models
            "command": 0.001,
            "command-light": 0.0004,

            # DeepSeek models
            "deepseek-chat": 0.00014,       # $0.14/1M input tokens
            "deepseek-reasoner": 0.00055,   # $0.55/1M input tokens

            # DashScope/Qwen models
            "qwen-turbo": 0.0001,
            "qwen-plus": 0.0004,
            "qwen-max": 0.002,
            "qwen-long": 0.0002,
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text using the appropriate adapter.
        
        Args:
            request: LLM generation request
            
        Returns:
            LLM response with generated text
        """
        start_time = datetime.utcnow()
        
        # Select the best model
        model, tier = await self._select_model(request)
        provider = get_provider_from_model(model)
        
        logger.info(f"Generating with {model} ({tier.value}) via {provider.value} for task: {request.task_type}")

        # Route to appropriate adapter
        try:
            # FIX: Check for workflow generation BEFORE provider routing
            # This ensures ALL providers (not just Ollama) get workflow-specific prompts
            if request.task_type == "workflow_generation":
                logger.info(f"Using workflow-specific generation for {provider.value}")
                response = await self._generate_workflow_with_any_provider(request, model, provider)
            elif provider == ModelProvider.OLLAMA:
                response = await self._generate_with_ollama(request, model)
            else:
                response = await self._generate_with_adapter(request, model, provider)
        except Exception as e:
            logger.error(f"Generation failed with {provider.value}: {e}")
            # On Cloud Run, Ollama is NOT available (no localhost:11434)
            # Only fallback to Ollama if we're NOT on GCP and NOT already using Ollama
            if provider != ModelProvider.OLLAMA and not self._is_cloud_environment():
                logger.info("Falling back to Ollama (local development)")
                try:
                    response = await self._generate_with_ollama(request, "llama3.2:3b")
                except Exception as ollama_error:
                    logger.error(f"Ollama fallback also failed: {ollama_error}")
                    raise e  # Re-raise the original error
            else:
                # On Cloud Run, don't try Ollama - just propagate the error
                logger.error(f"No fallback available in cloud environment. Original error: {e}")
                raise
        
        # Add metadata
        response.model_used = model
        response.provider = provider
        response.tier = tier
        response.response_time = (datetime.utcnow() - start_time).total_seconds()
        response.cost = self._calculate_cost(model, response.tokens_used)
        
        return response
    
    async def _select_model(self, request: LLMRequest) -> Tuple[str, ModelTier]:
        """
        Select the best model for the request using tier-based selection and enhanced selector.

        Priority order:
        1. model_override (explicit override from request)
        2. Tier-based user preferences (preferredFastModel/Balanced/Quality)
        3. user_settings.selected_model (legacy single model preference)
        4. context['preferred_model']
        5. Enhanced auto-selection with scoring
        """
        # Import tier resolver
        from llm.tier_resolver import get_model_for_tier, get_system_default_for_tier

        # Handle explicit overrides first
        if request.model_override:
            return request.model_override, classify_model_tier(request.model_override)

        # NEW: Tier-based model selection for task complexity
        # This allows users to configure different models for fast/balanced/quality tiers
        if request.user_settings:
            # Determine the appropriate tier for this request
            # We'll use task_type and context to determine tier
            tier = self._determine_tier_for_task(request)

            # Try to get user's preferred model for this tier
            user_preferences = request.user_settings.__dict__ if hasattr(request.user_settings, '__dict__') else {}
            preferred_model = get_model_for_tier(tier, user_preferences)

            if preferred_model:
                # Normalize the model name (UI names -> actual names)
                model = normalize_model_name(preferred_model)

                # Verify model is available
                available_models = await self._get_ollama_models()
                if model in available_models or self._is_api_model(model):
                    logger.info(f"Using tier-based preference: {tier.value} tier -> {model}")
                    return model, tier

        # Legacy: Single model preference (for backward compatibility)
        if request.user_settings and request.user_settings.selected_model:
            model = normalize_model_name(request.user_settings.selected_model)
            # Check if model is available
            available_models = await self._get_ollama_models()
            if model in available_models or self._is_api_model(model):
                return model, classify_model_tier(model)

        # System-default tier selection (no user preferences — use task-type mapping)
        # This ensures tool_use, workflow_generation, etc. get the right model size
        if not request.user_settings:
            from llm.tier_resolver import get_dynamic_system_default_for_tier
            tier = self._determine_tier_for_task(request)
            try:
                available_models = await self._get_ollama_models()
                # Try dynamic default (picks best available model for tier)
                system_model = get_dynamic_system_default_for_tier(tier, available_models)
                if system_model:
                    logger.info(f"Using system tier default: {tier.value} tier -> {system_model} (task_type={request.task_type})")
                    return system_model, tier

                # No model available for exact tier — try adjacent tiers
                for fallback_tier in [ModelTier.QUALITY, ModelTier.BALANCED, ModelTier.FAST]:
                    fallback_model = get_dynamic_system_default_for_tier(fallback_tier, available_models)
                    if fallback_model:
                        logger.info(f"Using fallback tier: {fallback_tier.value} -> {fallback_model}")
                        return fallback_model, fallback_tier
            except Exception:
                pass  # Fall through to enhanced selector

        # Use enhanced selector for sophisticated routing
        requirements = {
            "capabilities": request.context.get("required_capabilities", []) if request.context else [],
            "max_cost_per_1k_tokens": getattr(request.user_settings, 'max_cost', float("inf")) if request.user_settings else float("inf"),
            "prefer_local": getattr(request.user_settings, 'prefer_local', False) if request.user_settings else False,
            "quality_threshold": 0.7
        }
        
        try:
            model_id, routing_metadata = await self.enhanced_selector.select_model(
                task_type=request.task_type,
                prompt=request.prompt,
                requirements=requirements
            )
            
            # Log routing decision
            logger.info(f"Enhanced selector chose {model_id} with score {routing_metadata.get('score', 0)}")
            
            # Extract model name from model_id (format: adapter-modelname)
            if "-" in model_id:
                model = model_id.split("-", 1)[1]
            else:
                model = model_id

            # Check if selected model is available - if it's an Ollama model but Ollama isn't running,
            # fall back to DEFAULT_LLM_MODEL (e.g., for Cloud Run deployments)
            if is_ollama_model(model):
                # Model is an Ollama model - check if Ollama is available
                try:
                    available_models = await self._get_ollama_models()
                    if model not in available_models:
                        default_model = get_environment_default_model()
                        logger.info(f"Enhanced selector chose {model} but it's not available (Ollama not running?). Using DEFAULT_LLM_MODEL: {default_model}")
                        return default_model, classify_model_tier(default_model)
                except Exception as ollama_check_error:
                    # Ollama check failed - likely not available (Cloud Run), use default
                    default_model = get_environment_default_model()
                    logger.info(f"Ollama availability check failed ({ollama_check_error}). Using DEFAULT_LLM_MODEL: {default_model}")
                    return default_model, classify_model_tier(default_model)

            return model, classify_model_tier(model)
            
        except Exception as e:
            logger.warning(f"Enhanced selector failed: {e}, falling back to original logic")
            # Fallback to original selection logic
            available_models = await self._get_ollama_models()
        
        # 1. Explicit override
        if request.model_override:
            return request.model_override, classify_model_tier(request.model_override)
        
        # 2. User's UI selection
        if request.user_settings and request.user_settings.selected_model:
            model = normalize_model_name(request.user_settings.selected_model)
            # Check if model is available or is an API model
            if model in available_models or self._is_api_model(model):
                return model, classify_model_tier(model)
            elif request.user_settings.fallback_model:
                logger.warning(f"User's model {model} unavailable, using fallback")
                fallback = normalize_model_name(request.user_settings.fallback_model)
                return fallback, classify_model_tier(fallback)
        
        # 3. Context preference (e.g., from MCP)
        if request.context and request.context.get('preferred_model'):
            model = request.context['preferred_model']
            if model in available_models or self._is_api_model(model):
                return model, classify_model_tier(model)
        
        # 4. Auto-selection based on task and complexity
        complexity = request.complexity
        if complexity is None:
            # Estimate complexity
            try:
                complexity = estimate_complexity_hybrid(request.prompt)
            except Exception:
                complexity = len(request.prompt.split()) / 100  # Simple fallback

        model, tier = select_model_for_task(
            request.task_type,
            complexity,
            available_models
        )

        # 5. Final fallback: If selected model is not available (e.g., Ollama model on Cloud Run),
        # use DEFAULT_LLM_MODEL from environment. This ensures cloud deployments work with Vertex AI
        # while local dev can use Ollama.
        if model not in available_models and not self._is_api_model(model):
            default_model = get_environment_default_model()
            if default_model:
                logger.info(f"Model {model} not available, using DEFAULT_LLM_MODEL: {default_model}")
                return default_model, classify_model_tier(default_model)

        return model, tier

    def _determine_tier_for_task(self, request: LLMRequest) -> ModelTier:
        """
        Determine the appropriate model tier for a given task.

        Maps task types to tiers based on complexity and performance requirements:
        - FAST tier (~1-2s): Quick classification and intent detection tasks
        - BALANCED tier (~3-5s): Semantic matching, analysis, and validation
        - QUALITY tier (~20-25s): Complex generation and workflow creation

        Args:
            request: The LLM request containing task_type and context

        Returns:
            ModelTier enum (FAST, BALANCED, or QUALITY)
        """
        task_type = request.task_type.lower() if request.task_type else ""

        # FAST tier tasks: Simple classification and quick decisions (~1-2 seconds)
        fast_tasks = {
            "intent_classification",
            "domain_classification",
            "intent",
            "classification",
            "quick_analysis",
            "validation",
        }

        # BALANCED tier tasks: Semantic analysis and matching (~3-5 seconds)
        balanced_tasks = {
            "semantic_matching",
            "semantic_analysis",
            "analysis",
            "response_formatting",
            "spec_generation",
            "template_matching",
            "domain_matching",
        }

        # QUALITY tier tasks: Complex generation and creation (~20-25 seconds)
        quality_tasks = {
            "workflow_generation",
            "generation",
            "complex_generation",
            "workflow_creation",
            "detailed_analysis",
            "tool_use",
        }

        # Check for exact matches first
        if task_type in fast_tasks:
            return ModelTier.FAST
        elif task_type in balanced_tasks:
            return ModelTier.BALANCED
        elif task_type in quality_tasks:
            return ModelTier.QUALITY

        # Fallback: Check for partial matches (contains keyword)
        if any(keyword in task_type for keyword in ["classify", "intent", "quick"]):
            return ModelTier.FAST
        elif any(keyword in task_type for keyword in ["semantic", "match", "analysis", "format"]):
            return ModelTier.BALANCED
        elif any(keyword in task_type for keyword in ["generate", "workflow", "create", "complex"]):
            return ModelTier.QUALITY

        # Default to BALANCED for unknown tasks (middle ground)
        logger.debug(f"Unknown task type '{task_type}', defaulting to BALANCED tier")
        return ModelTier.BALANCED

    async def _generate_with_ollama(self, request: LLMRequest, model: str) -> LLMResponse:
        """Generate using Ollama directly."""
        try:
            # For workflow generation, use the model adapter
            if request.task_type == "workflow_generation":
                from services.model_adapters import get_model_adapter
                
                adapter = get_model_adapter(model, self.ollama_url)
                steps = await adapter.generate_workflow_steps(
                    prompt=request.prompt,
                    model_name=model,
                    temperature=request.temperature or 0.7,
                    timeout=60.0
                )
                
                # Convert steps to text
                text = self._steps_to_text(steps) if steps else "Failed to generate workflow steps"
                
                return LLMResponse(
                    text=text,
                    model_used=model,
                    provider=ModelProvider.OLLAMA,
                    tier=classify_model_tier(model),
                    tokens_used=int(len(text.split()) * 1.3),  # Rough estimate
                    metadata={"steps": [s.to_dict() for s in steps] if steps else []}
                )
            else:
                # General text generation
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": request.prompt,
                            "system": request.system_prompt,
                            "stream": False,
                            "temperature": request.temperature or 0.7,
                            "options": {
                                "num_predict": request.max_tokens or 1000
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return LLMResponse(
                            text=result.get("response", ""),
                            model_used=model,
                            provider=ModelProvider.OLLAMA,
                            tier=classify_model_tier(model),
                            tokens_used=result.get("eval_count", 0) + result.get("prompt_eval_count", 0)
                        )
                    else:
                        raise Exception(f"Ollama error: {response.status_code}")
                        
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def _generate_workflow_with_any_provider(
        self, request: LLMRequest, model: str, provider: ModelProvider
    ) -> LLMResponse:
        """
        Generate workflow steps using ANY provider with workflow-specific prompts.

        This method ensures ALL LLM providers (Gemini, Claude, OpenAI, Cohere, etc.)
        receive the same sophisticated workflow-specific prompting that Ollama gets,
        producing comprehensive workflows with 10-35 nodes instead of minimal 3-5 nodes.

        This fixes the LLM routing bug documented in WORKFLOW_CREATION_IMPLEMENTATION_SPEC.md
        """
        import json
        import re

        # Enhanced workflow generation prompt - same quality as Ollama adapter
        workflow_system_prompt = """You are an enterprise workflow architect. Generate comprehensive workflow steps.

Return a JSON array where each step has:
- action: short identifier (e.g., "validate_input", "process_data")
- node_type: one of [process, aiProcess, decision, approval, adapter, dataSource, transformer, humanAgent, start, end]
- capability: what it does (e.g., "data_validation", "ai_processing", "notification")
- label: human-readable name
- description: detailed description of what this step does

IMPORTANT:
- Generate 10-30 comprehensive steps for enterprise-grade workflows
- Include proper start and end nodes
- Add decision nodes for branching logic
- Include error handling and validation steps
- Add AI processing nodes where intelligent analysis is needed
- Include human approval nodes for critical decisions
- Add notification steps for status updates

Example output format:
[
  {"action": "start", "node_type": "start", "capability": "workflow_entry", "label": "Start Workflow", "description": "Initialize the workflow"},
  {"action": "validate_input", "node_type": "process", "capability": "data_validation", "label": "Validate Input Data", "description": "Ensure all required fields are present and valid"},
  {"action": "ai_analyze", "node_type": "aiProcess", "capability": "ai_processing", "label": "AI Analysis", "description": "Use AI to analyze and extract insights"},
  {"action": "decision_check", "node_type": "decision", "capability": "routing", "label": "Check Results", "description": "Route based on analysis outcome"},
  {"action": "human_review", "node_type": "humanAgent", "capability": "manual_review", "label": "Human Review", "description": "Manual review for edge cases"},
  {"action": "notify_complete", "node_type": "adapter", "capability": "notification", "label": "Send Notification", "description": "Notify stakeholders of completion"},
  {"action": "end", "node_type": "end", "capability": "workflow_exit", "label": "End Workflow", "description": "Complete the workflow"}
]

Return ONLY the JSON array, no other text or explanation."""

        # Build the full prompt with workflow context
        full_prompt = f"{workflow_system_prompt}\n\nUser Request: {request.prompt}\n\nGenerate a comprehensive workflow:"

        try:
            if provider == ModelProvider.OLLAMA:
                # Use Ollama directly for workflow generation
                from services.model_adapters import get_model_adapter

                adapter = get_model_adapter(model, self.ollama_url)
                steps = await adapter.generate_workflow_steps(
                    prompt=request.prompt,
                    model_name=model,
                    temperature=request.temperature or 0.7,
                    timeout=60.0
                )

                text = self._steps_to_text(steps) if steps else "Failed to generate workflow steps"

                return LLMResponse(
                    text=text,
                    model_used=model,
                    provider=provider,
                    tier=classify_model_tier(model),
                    tokens_used=int(len(text.split()) * 1.3),
                    metadata={"steps": [s.to_dict() for s in steps] if steps else []}
                )
            else:
                # Use adapter with workflow-specific prompt for non-Ollama providers
                adapter = None
                try:
                    # Import the appropriate adapter based on provider
                    if provider == ModelProvider.ANTHROPIC:
                        from adapters.implementations.ai.claude_adapter import ClaudeAdapter
                        adapter = ClaudeAdapter()
                    elif provider == ModelProvider.OPENAI:
                        from adapters.implementations.ai.openai_adapter import OpenAIAdapter
                        adapter = OpenAIAdapter()
                    elif provider == ModelProvider.GEMINI:
                        from adapters.implementations.ai.gemini_adapter import GeminiAdapter
                        adapter = GeminiAdapter()
                    elif provider == ModelProvider.VERTEX_AI:
                        from business_adapters.implementations.ai.vertex_ai_adapter import VertexAIAdapter
                        adapter = VertexAIAdapter(system_mode=True)
                    elif provider == ModelProvider.COHERE:
                        from adapters.implementations.ai.cohere_adapter import CohereAdapter
                        adapter = CohereAdapter()
                    elif provider == ModelProvider.HUGGINGFACE:
                        from adapters.implementations.ai.huggingface_adapter import HuggingFaceAdapter
                        adapter = HuggingFaceAdapter()
                    elif provider == ModelProvider.BEDROCK:
                        from adapters.implementations.ai.bedrock_adapter import BedrockAdapter
                        adapter = BedrockAdapter()
                    elif provider == ModelProvider.AZURE_OPENAI:
                        from adapters.implementations.ai.azure_openai_adapter import AzureOpenAIAdapter
                        adapter = AzureOpenAIAdapter()
                    else:
                        raise ValueError(f"Unknown provider: {provider}")

                    # Initialize the adapter
                    logger.info(f"Initializing {provider.value} adapter for workflow generation...")
                    await adapter.initialize()

                    # Build adapter request with workflow-specific prompt
                    from adapters.models import AdapterRequest

                    messages = [
                        {"role": "system", "content": workflow_system_prompt},
                        {"role": "user", "content": f"Generate a comprehensive workflow for: {request.prompt}"}
                    ]

                    adapter_request = AdapterRequest(
                        capability="chat",
                        parameters={
                            "model": model,
                            "messages": messages,
                            "max_tokens": request.max_tokens or 4096,  # Higher for workflows
                            "temperature": request.temperature or 0.7,
                            "stream": False
                        }
                    )

                    # Execute request
                    response = await adapter.execute(adapter_request)

                    # Extract text from response - handle different adapter formats
                    text = ""
                    if response.data:
                        if "choices" in response.data:
                            # OpenAI format
                            text = response.data["choices"][0]["message"]["content"]
                        elif "response" in response.data:
                            # Vertex AI/Gemini format - returns data["response"]
                            text = response.data["response"]
                        elif "text" in response.data:
                            # Generic format
                            text = response.data["text"]
                        elif "content" in response.data:
                            # Anthropic/Claude format
                            text = response.data["content"]

                    # Log the response for debugging
                    logger.info(f"Workflow generation response from {provider.value}: {len(text)} chars")
                    if not text:
                        logger.warning(f"Empty response from {provider.value}. Response data keys: {list(response.data.keys()) if response.data else 'None'}")

                    # Try to parse and structure the workflow steps
                    steps = self._parse_workflow_response(text)

                    if steps:
                        logger.info(f"Successfully generated {len(steps)} workflow steps via {provider.value}")
                        structured_text = self._steps_to_text(steps)
                        return LLMResponse(
                            text=structured_text,
                            model_used=model,
                            provider=provider,
                            tier=classify_model_tier(model),
                            tokens_used=response.tokens_used or 0,
                            cost=response.cost or 0.0,
                            metadata={
                                "steps": steps,
                                "raw_response": text,
                                "workflow_generation": True
                            }
                        )
                    else:
                        # Return raw response if parsing failed
                        logger.warning(f"Could not parse workflow steps from {provider.value} response, returning raw")
                        return LLMResponse(
                            text=text,
                            model_used=model,
                            provider=provider,
                            tier=classify_model_tier(model),
                            tokens_used=response.tokens_used or 0,
                            cost=response.cost or 0.0,
                            metadata={"raw_response": True}
                        )

                except ImportError as e:
                    logger.error(f"Could not import adapter for {provider.value}: {e}")
                    raise ValueError(f"Adapter for {provider.value} not available: {e}")
                finally:
                    if adapter is not None:
                        try:
                            await adapter.shutdown()
                        except Exception as cleanup_error:
                            logger.warning(f"Error cleaning up {provider.value} adapter: {cleanup_error}")

        except Exception as e:
            logger.error(f"Workflow generation failed with {provider.value}: {e}")
            raise

    def _parse_workflow_response(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse workflow steps from LLM response text with robust JSON and text parsing."""
        import json
        import re

        logger.info(f"Parsing workflow response, first 800 chars: {text[:800] if len(text) > 0 else 'EMPTY'}")

        # Try direct JSON parsing first
        try:
            result = json.loads(text.strip())
            if isinstance(result, list) and len(result) > 0:
                logger.info(f"Direct JSON parsing succeeded: {len(result)} steps")
                return result
        except Exception as e:
            logger.debug(f"Direct JSON parsing failed: {e}")

        # Try json_repair library early (comprehensive LLM JSON repair)
        try:
            from json_repair import repair_json
            repaired = repair_json(text.strip(), return_objects=True)
            if isinstance(repaired, list) and len(repaired) > 0:
                logger.info(f"json_repair library succeeded: {len(repaired)} steps")
                return repaired
            elif isinstance(repaired, dict):
                # Maybe it's a dict with a 'steps' or 'nodes' key
                for key in ['steps', 'nodes', 'workflow_steps', 'items']:
                    if key in repaired and isinstance(repaired[key], list):
                        logger.info(f"json_repair found {len(repaired[key])} steps in '{key}' key")
                        return repaired[key]
        except ImportError:
            logger.debug("json_repair library not available")
        except Exception as e:
            logger.debug(f"json_repair direct parsing failed: {e}")

        # Try to extract from markdown code blocks (including incomplete ones without closing ```)
        code_block_patterns = [
            r'```json\s*([\s\S]*?)\s*```',   # Complete code block with json tag
            r'```\s*([\s\S]*?)\s*```',       # Complete code block without json tag
            r'```json\s*([\s\S]+)',           # Incomplete code block (no closing ```) with json tag
            r'```\s*([\s\S]+)',               # Incomplete code block (no closing ```) without json tag
        ]

        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            logger.info(f"Code block pattern found {len(matches)} matches")
            for match in matches:
                try:
                    json_str = match if isinstance(match, str) else match
                    # Try direct parsing first
                    try:
                        result = json.loads(json_str.strip())
                        if isinstance(result, list) and len(result) > 0:
                            logger.info(f"Code block JSON parsing succeeded: {len(result)} steps")
                            return result
                    except json.JSONDecodeError:
                        # Try with cleanup
                        cleaned = self._cleanup_llm_json(json_str.strip())
                        result = json.loads(cleaned)
                        if isinstance(result, list) and len(result) > 0:
                            logger.info(f"Code block JSON parsing succeeded after cleanup: {len(result)} steps")
                            return result
                except Exception as e:
                    logger.info(f"Code block parsing failed: {e}")
                    continue

        # Try to find the LONGEST JSON array in the text using bracket matching
        result = self._find_json_array_bracket_matching(text)
        if result:
            logger.info(f"Bracket matching parsing succeeded: {len(result)} steps")
            return result

        # Fallback: Parse numbered/bulleted text format
        result = self._parse_text_format_steps(text)
        if result:
            logger.info(f"Text format parsing succeeded: {len(result)} steps")
            return result

        logger.warning(f"All parsing methods failed for text of length {len(text)}")
        return None

    def _cleanup_llm_json(self, json_str: str) -> str:
        """Clean up common LLM JSON mistakes before parsing using json_repair library.

        This handles:
        - Unterminated strings (unescaped newlines/quotes in strings)
        - Trailing commas
        - Missing quotes around keys
        - Single quotes instead of double quotes
        - Control characters
        - And many other LLM JSON mistakes
        """
        import re

        # Try json_repair library first (most comprehensive solution)
        try:
            from json_repair import repair_json
            repaired = repair_json(json_str, return_objects=False)
            logger.debug(f"json_repair successfully repaired JSON")
            return repaired
        except ImportError:
            logger.warning("json_repair library not available, using manual cleanup")
        except Exception as e:
            logger.debug(f"json_repair failed: {e}, falling back to manual cleanup")

        # Fallback: Manual cleanup for common issues

        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        # Replace single quotes with double quotes (but not in strings)
        # This is a simplistic approach - json_repair does this better
        json_str = re.sub(r"(?<![\\])'", '"', json_str)

        # Handle unescaped newlines in strings by replacing with \n
        # Match strings and escape newlines inside them
        def escape_newlines_in_string(match):
            content = match.group(1)
            # Replace literal newlines with escaped versions
            content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return f'"{content}"'

        # This regex matches JSON strings and escapes newlines within them
        # Note: This is imperfect but handles common cases
        json_str = re.sub(r'"((?:[^"\\]|\\.)*)(?:\n|\r\n?)([^"]*)"',
                         lambda m: f'"{m.group(1)}\\n{m.group(2)}"',
                         json_str, flags=re.MULTILINE)

        # Remove control characters that break JSON
        json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_str)

        # Fix unquoted keys: word: -> "word":
        json_str = re.sub(r'(?<=[{,\s])(\w+)\s*:', r'"\1":', json_str)

        return json_str

    def _find_json_array_bracket_matching(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Find the longest valid JSON array using bracket matching."""
        import json

        # Find all positions of '[' in the text
        candidates = []
        last_error = None

        for i, char in enumerate(text):
            if char == '[':
                # Try to find matching bracket
                depth = 0
                end_pos = -1
                for j in range(i, len(text)):
                    if text[j] == '[':
                        depth += 1
                    elif text[j] == ']':
                        depth -= 1
                        if depth == 0:
                            end_pos = j
                            break

                if end_pos > i:
                    potential_json = text[i:end_pos + 1]

                    # Try direct parsing first
                    try:
                        result = json.loads(potential_json)
                        if isinstance(result, list) and len(result) > 0:
                            # Check if items look like workflow steps
                            if isinstance(result[0], dict):
                                candidates.append(result)
                                continue
                    except json.JSONDecodeError as e:
                        last_error = str(e)

                    # Try with cleanup for common LLM mistakes
                    try:
                        cleaned_json = self._cleanup_llm_json(potential_json)
                        result = json.loads(cleaned_json)
                        if isinstance(result, list) and len(result) > 0:
                            if isinstance(result[0], dict):
                                logger.info(f"Bracket matching succeeded after JSON cleanup")
                                candidates.append(result)
                    except json.JSONDecodeError as e:
                        last_error = str(e)
                        continue

        # Return the longest valid array
        if candidates:
            return max(candidates, key=len)

        # Log the last error for debugging
        if last_error:
            logger.warning(f"Bracket matching failed with last error: {last_error}")
        return None

    def _parse_text_format_steps(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse workflow steps from numbered or bulleted text format."""
        import re

        steps = []
        lines = text.strip().split('\n')

        # Pattern for numbered steps: "1. Step Name: Description" or "Step 1: Name - Description"
        step_patterns = [
            r'^\s*(\d+)[\.\)]\s*(.+?)(?::\s*(.+))?$',  # "1. Name: Description" or "1) Name: Description"
            r'^\s*Step\s*(\d+)[:\s]+(.+?)(?:\s*[-:]\s*(.+))?$',  # "Step 1: Name - Description"
            r'^\s*[-*•]\s*(.+?)(?::\s*(.+))?$',  # "- Name: Description" or "* Name: Description"
        ]

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            for pattern in step_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        # Handle different patterns
                        if groups[0] and groups[0].isdigit():
                            # Numbered pattern
                            label = groups[1].strip() if groups[1] else f"Step {groups[0]}"
                            description = groups[2].strip() if len(groups) > 2 and groups[2] else ""
                        else:
                            # Bulleted pattern
                            label = groups[0].strip() if groups[0] else "Step"
                            description = groups[1].strip() if len(groups) > 1 and groups[1] else ""

                        if label:
                            steps.append({
                                "label": label,
                                "description": description,
                                "type": "task"
                            })
                    break

        # Only return if we found meaningful steps (at least 3)
        if len(steps) >= 3:
            return steps
        return None

    async def _generate_with_adapter(self, request: LLMRequest, model: str, provider: ModelProvider) -> LLMResponse:
        """
        Generate using existing adapters from Community/Business/Enterprise editions.

        This method directly imports and uses the adapters that are already
        implemented in the various editions, avoiding code duplication.
        """
        adapter = None
        try:
            # Import the appropriate adapter based on provider
            if provider == ModelProvider.ANTHROPIC:
                from adapters.implementations.ai.claude_adapter import ClaudeAdapter
                adapter = ClaudeAdapter()
            elif provider == ModelProvider.OPENAI:
                from adapters.implementations.ai.openai_adapter import OpenAIAdapter
                adapter = OpenAIAdapter()
            elif provider == ModelProvider.GEMINI:
                # Business edition adapter
                from adapters.implementations.ai.gemini_adapter import GeminiAdapter
                adapter = GeminiAdapter()
            elif provider == ModelProvider.HUGGINGFACE:
                from adapters.implementations.ai.huggingface_adapter import HuggingFaceAdapter
                adapter = HuggingFaceAdapter()
            elif provider == ModelProvider.BEDROCK:
                # Business edition adapter
                from adapters.implementations.ai.bedrock_adapter import BedrockAdapter
                adapter = BedrockAdapter()
            elif provider == ModelProvider.AZURE_OPENAI:
                # Business edition adapter
                from adapters.implementations.ai.azure_openai_adapter import AzureOpenAIAdapter
                adapter = AzureOpenAIAdapter()
            elif provider == ModelProvider.VERTEX_AI:
                # Business edition adapter - use system_mode for platform operations
                # This uses GCP default credentials automatically (hidden from customers)
                # Customers must provide their own credentials for workflow AI nodes
                from business_adapters.implementations.ai.vertex_ai_adapter import VertexAIAdapter
                adapter = VertexAIAdapter(system_mode=True)
            elif provider == ModelProvider.COHERE:
                # Enterprise edition adapter
                from adapters.implementations.ai.cohere_adapter import CohereAdapter
                adapter = CohereAdapter()
            else:
                raise ValueError(f"Unknown provider: {provider}")

            # CRITICAL: Initialize the adapter before use
            # This sets up credentials (for Vertex AI), creates HTTP clients, etc.
            logger.info(f"Initializing {provider.value} adapter...")
            await adapter.initialize()
            logger.info(f"{provider.value} adapter initialized successfully")
            
            # Build adapter request
            from adapters.models import AdapterRequest
            
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            adapter_request = AdapterRequest(
                capability="chat",  # Use "chat" - matches adapter capability
                parameters={
                    "model": model,
                    "messages": messages,
                    "max_tokens": request.max_tokens or 1024,
                    "temperature": request.temperature or 0.7,
                    "stream": request.stream or False
                }
            )
            
            # Execute request
            response = await adapter.execute(adapter_request)
            
            # Extract text from response - handle different adapter formats
            text = ""
            if response.data:
                if "choices" in response.data:
                    # OpenAI/Claude format
                    text = response.data["choices"][0]["message"]["content"]
                elif "response" in response.data:
                    # Vertex AI/Gemini format - returns data["response"]
                    text = response.data["response"]
                elif "text" in response.data:
                    # Generic format
                    text = response.data["text"]
                elif "content" in response.data:
                    # Anthropic format
                    text = response.data["content"]

            logger.info(f"Extracted {len(text)} chars from {provider.value} response")
            
            return LLMResponse(
                text=text,
                model_used=model,
                provider=provider,
                tier=classify_model_tier(model),
                tokens_used=response.tokens_used or 0,
                cost=response.cost or 0.0,
                metadata=response.metadata or {}
            )
            
        except ImportError as e:
            logger.error(f"Could not import adapter for {provider.value}: {e}")
            # Don't fallback to Ollama on import errors - it won't be available on Cloud Run
            raise ValueError(f"Adapter for {provider.value} not available: {e}")
        except Exception as e:
            logger.error(f"{provider.value} generation failed: {e}")
            raise
        finally:
            # Clean up adapter if it was initialized
            if adapter is not None:
                try:
                    await adapter.shutdown()
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up {provider.value} adapter: {cleanup_error}")
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get all available models across all providers."""
        from llm.model_selection import _estimate_model_size_billions

        models = []

        # Get Ollama models
        ollama_models = await self._get_ollama_models()
        for model_name in ollama_models:
            size = _estimate_model_size_billions(model_name)
            models.append(ModelInfo(
                name=model_name,
                provider=ModelProvider.OLLAMA,
                tier=classify_model_tier(model_name),
                cost_per_1k_tokens=0.0,
                supports_streaming=True,
                description=f"Local Ollama model: {model_name}",
                local=True,
                parameter_size=f"{size}B" if size else None,
            ))

        # Add known API models (shown in dropdown for all editions)
        # Users will be prompted for API keys when they select these models
        api_models = [
            # Anthropic Claude models
            ("claude-3-haiku", ModelProvider.ANTHROPIC),
            ("claude-3-sonnet", ModelProvider.ANTHROPIC),
            ("claude-3-opus", ModelProvider.ANTHROPIC),

            # OpenAI models
            ("gpt-3.5-turbo", ModelProvider.OPENAI),
            ("gpt-4", ModelProvider.OPENAI),
            ("gpt-4-turbo", ModelProvider.OPENAI),

            # Google Gemini models (Direct API via Google AI Studio)
            ("gemini-2.0-flash", ModelProvider.GEMINI),
            ("gemini-2.5-flash", ModelProvider.GEMINI),
            ("gemini-1.5-pro", ModelProvider.GEMINI),
            ("gemini-pro", ModelProvider.GEMINI),

            # Google Gemini models (via GCP Vertex AI)
            ("gemini-2.0-flash-vertex", ModelProvider.VERTEX_AI),
            ("gemini-2.5-flash-vertex", ModelProvider.VERTEX_AI),
            ("gemini-1.5-pro-vertex", ModelProvider.VERTEX_AI),

            # Cohere models
            ("command", ModelProvider.COHERE),
            ("command-light", ModelProvider.COHERE),

            # DeepSeek models
            ("deepseek-chat", ModelProvider.DEEPSEEK),
            ("deepseek-reasoner", ModelProvider.DEEPSEEK),

            # DashScope (Qwen) models
            ("qwen-turbo", ModelProvider.DASHSCOPE),
            ("qwen-plus", ModelProvider.DASHSCOPE),
            ("qwen-max", ModelProvider.DASHSCOPE),
            ("qwen-long", ModelProvider.DASHSCOPE),
        ]

        for model_name, provider in api_models:
            if self._is_provider_configured(provider):
                models.append(ModelInfo(
                    name=model_name,
                    provider=provider,
                    tier=classify_model_tier(model_name),
                    cost_per_1k_tokens=self.pricing.get(model_name, 0.0),
                    supports_streaming=True,
                    supports_json_mode=provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.DEEPSEEK, ModelProvider.DASHSCOPE],
                    description=f"{provider.value} model: {model_name}",
                    local=False,
                ))

        return models
    
    async def estimate_cost(self, prompt: str, model: str) -> CostEstimate:
        """Estimate cost before generation."""
        # Rough token estimation (1 token H 4 chars)
        estimated_tokens = len(prompt) // 4 + 500  # Input + expected output
        
        provider = get_provider_from_model(model)
        cost_per_1k = self.pricing.get(model, 0.0)
        estimated_cost = (estimated_tokens / 1000) * cost_per_1k
        
        return CostEstimate(
            estimated_tokens=estimated_tokens,
            estimated_cost=estimated_cost,
            model=model,
            provider=provider,
            confidence=0.7
        )
    
    async def _get_ollama_models(self) -> List[str]:
        """Get available Ollama models (5-minute TTL cache)."""
        import time
        now = time.time()
        if self._available_models_cache is not None and (now - self._models_cache_time) < 300:
            return self._available_models_cache

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    self._available_models_cache = [model["name"] for model in models]
                    self._models_cache_time = now
                    logger.info(f"Available Ollama models: {self._available_models_cache}")
                    return self._available_models_cache
        except Exception as e:
            logger.warning(f"Failed to get Ollama models: {e}")

        self._available_models_cache = []
        # Use short TTL (30s) for failures so recovery is fast
        self._models_cache_time = now - 270
        return self._available_models_cache
    
    def _is_api_model(self, model: str) -> bool:
        """Check if model requires API access."""
        return any(x in model.lower() for x in [
            "claude", "gpt", "gemini", "command",
            "deepseek-chat", "deepseek-reasoner",
            "qwen-turbo", "qwen-plus", "qwen-max", "qwen-long",
        ])

    def _is_cloud_environment(self) -> bool:
        """Check if running in a cloud environment (GCP Cloud Run, GCE, etc.)

        On Cloud Run, Ollama is NOT available because there's no localhost:11434.
        This helps us avoid pointless fallback attempts.
        """
        import os
        # Cloud Run environment variables
        if os.environ.get('K_SERVICE'):
            return True
        if os.environ.get('CLOUD_RUN_JOB'):
            return True
        # GCE metadata indicator
        if os.environ.get('GCE_METADATA_HOST'):
            return True
        # Generic cloud indicator (can be set in deployment)
        if os.environ.get('CLOUD_ENVIRONMENT'):
            return True
        # AWS indicators
        if os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
            return True
        if os.environ.get('AWS_EXECUTION_ENV'):
            return True
        return False
    
    def _is_provider_configured(self, provider: ModelProvider) -> bool:
        """
        Check if a provider is configured with API keys.

        Returns True to show all models in the dropdown.
        Users will be prompted for API keys when they try to use them.
        """
        # Show all models in dropdown regardless of configuration status
        # This allows users to see what's available and configure as needed
        return True
    
    def _steps_to_text(self, steps: Union[List[WorkflowStep], List[Dict[str, Any]]]) -> str:
        """Convert workflow steps to text.

        Handles both WorkflowStep objects and dictionaries (from JSON parsing).
        """
        if not steps:
            return ""

        lines = ["Generated workflow steps:"]
        for i, step in enumerate(steps, 1):
            # Handle both dictionary and object formats
            if isinstance(step, dict):
                label = step.get("label", step.get("name", f"Step {i}"))
                description = step.get("description", step.get("action", ""))
            else:
                label = getattr(step, "label", getattr(step, "name", f"Step {i}"))
                description = getattr(step, "description", None) or getattr(step, "action", "")
            lines.append(f"{i}. {label}: {description}")

        return "\n".join(lines)
    
    def _calculate_cost(self, model: str, tokens: int) -> float:
        """Calculate cost for token usage."""
        rate = self.pricing.get(model, 0.0)
        return (tokens / 1000) * rate