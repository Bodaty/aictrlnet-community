"""
Model adapters for handling different AI model response formats.

This module provides a unified interface for working with different AI models
that may return workflow steps in various formats.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import httpx

# Import WorkflowStep from LLM module for consistency
try:
    from llm import WorkflowStep
except ImportError:
    # Fallback to local definition if LLM module not available
    from pydantic import BaseModel
    
    class WorkflowStep(BaseModel):
        """Fallback workflow step representation."""
        label: str
        description: Optional[str] = None
        action: Optional[str] = None
        input_data: Optional[Dict[str, Any]] = None
        output_schema: Optional[Dict[str, Any]] = None
        dependencies: Optional[List[str]] = None
        agent: Optional[str] = None
        template: Optional[str] = None
        
        # Additional fields for backward compatibility
        node_type: Optional[str] = None
        capability: Optional[str] = None
        parameters: Optional[Dict[str, Any]] = None
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to the format expected by NLP service."""
            return {
                "action": self.action or self.label,
                "node_type": self.node_type or "process",
                "capability": self.capability,
                "intent": {
                    "label": self.label,
                    "originalText": self.label,
                    "description": self.description or self.label
                },
                "parameters": self.parameters or self.input_data or {},
                "dependencies": self.dependencies or []
            }

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Base class for model-specific adapters."""
    
    def __init__(self, ollama_url: str = "http://host.docker.internal:11434"):
        self.ollama_url = ollama_url
    
    @abstractmethod
    async def generate_workflow_steps(
        self, 
        prompt: str,
        model_name: str,
        temperature: float = 0.3,
        timeout: float = 60.0
    ) -> Optional[List[WorkflowStep]]:
        """Generate workflow steps from a prompt."""
        pass
    
    def _extract_json_from_text(self, text: str) -> Optional[Any]:
        """Extract JSON from text that might contain markdown or other formatting."""
        # Try direct JSON parsing first
        try:
            return json.loads(text.strip())
        except:
            pass
        
        # Try to extract from markdown code blocks
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\[[\s\S]*\]',  # Just the array
            r'\{[\s\S]*\}'   # Just an object
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str.strip())
                except:
                    continue
        
        return None
    
    def _infer_node_type(self, text: str) -> str:
        """Infer node type from action text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['ai', 'analyze', 'predict', 'classify', 'detect']):
            return "aiProcess"
        elif any(word in text_lower for word in ['approve', 'approval', 'review', 'sign']):
            return "approval"
        elif any(word in text_lower for word in ['decide', 'check', 'if', 'route', 'branch']):
            return "decision"
        elif any(word in text_lower for word in ['transform', 'convert', 'map', 'format']):
            return "transformer"
        elif any(word in text_lower for word in ['send', 'notify', 'email', 'sms', 'alert']):
            return "adapter"
        elif any(word in text_lower for word in ['extract', 'fetch', 'retrieve', 'query', 'get']):
            return "dataSource"
        elif any(word in text_lower for word in ['human', 'manual', 'person', 'user']):
            return "humanAgent"
        else:
            return "process"
    
    def _infer_capability(self, text: str) -> str:
        """Infer capability from action text."""
        text_lower = text.lower()
        
        if 'ai' in text_lower or 'analyze' in text_lower:
            return "ai_processing"
        elif 'approve' in text_lower:
            return "approval_workflow"
        elif 'notify' in text_lower or 'send' in text_lower:
            return "notification"
        elif 'transform' in text_lower:
            return "data_transformation"
        elif 'validate' in text_lower:
            return "data_validation"
        elif 'extract' in text_lower or 'fetch' in text_lower:
            return "data_extraction"
        else:
            return "data_processing"


class LlamaAdapter(ModelAdapter):
    """Adapter for Llama models that handles various response formats."""
    
    async def generate_workflow_steps(
        self, 
        prompt: str,
        model_name: str,
        temperature: float = 0.3,
        timeout: float = 60.0
    ) -> Optional[List[WorkflowStep]]:
        """Generate workflow steps using Llama models."""
        
        # Create a more explicit prompt based on model size
        if "1b" in model_name:
            # Simpler prompt for smaller model
            system_prompt = """List workflow steps as JSON array:
[{"action": "step_name", "description": "what it does"}]
Only return the JSON array."""
        else:
            # More detailed prompt for larger models
            system_prompt = """Extract workflow steps from the user's request.

Return a JSON array where each step has:
- action: short identifier (e.g., "send_email")
- node_type: one of [process, aiProcess, decision, approval, adapter, dataSource, transformer, humanAgent]
- capability: what it does (e.g., "notification", "ai_processing")
- label: human-readable name
- description: detailed description

Example:
[
  {
    "action": "analyze_data",
    "node_type": "aiProcess",
    "capability": "ai_processing",
    "label": "Analyze customer data",
    "description": "Use AI to analyze customer behavior patterns"
  }
]

Return ONLY the JSON array, no other text."""
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": f"{system_prompt}\n\nUser request: {prompt}",
                        "stream": False,
                        "temperature": temperature,
                        "options": {
                            "num_predict": 2000
                        }
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama request failed with status {response.status_code}")
                    return None
                
                result = response.json()
                response_text = result.get("response", "")
                
                # Try to parse the response
                steps = self._parse_llama_response(response_text, model_name)
                
                if steps:
                    logger.info(f"Successfully parsed {len(steps)} steps from {model_name}")
                else:
                    logger.warning(f"Failed to parse steps from {model_name} response")
                
                return steps
                
        except Exception as e:
            logger.error(f"Error generating workflow steps with {model_name}: {e}")
            return None
    
    def _parse_llama_response(self, response_text: str, model_name: str) -> Optional[List[WorkflowStep]]:
        """Parse Llama response which might be in various formats."""
        
        # Try to extract JSON
        json_data = self._extract_json_from_text(response_text)
        
        if not json_data:
            logger.debug(f"No JSON found in response: {response_text[:200]}...")
            return None
        
        # Handle different JSON structures
        if isinstance(json_data, list):
            return self._parse_step_array(json_data)
        elif isinstance(json_data, dict):
            # Might be wrapped in an object
            if "steps" in json_data:
                return self._parse_step_array(json_data["steps"])
            elif "workflow" in json_data:
                return self._parse_step_array(json_data["workflow"])
        
        return None
    
    def _parse_step_array(self, steps: List[Dict[str, Any]]) -> List[WorkflowStep]:
        """Parse an array of steps in various formats."""
        parsed_steps = []
        
        for i, step in enumerate(steps):
            # Handle expected format
            if "action" in step and "node_type" in step:
                parsed_steps.append(WorkflowStep(
                    label=step.get("label") or step.get("intent", {}).get("label") or step["action"],
                    description=step.get("description") or step.get("intent", {}).get("description"),
                    action=step["action"],
                    node_type=step["node_type"],
                    capability=step.get("capability", self._infer_capability(step["action"])),
                    parameters=step.get("parameters", {}),
                    dependencies=step.get("dependencies", [])
                ))
            
            # Handle {"Step": "...", "Description": "..."} format
            elif "Step" in step:
                action = f"step_{i+1}"
                label = step["Step"]
                parsed_steps.append(WorkflowStep(
                    label=label,
                    description=step.get("Description", label),
                    action=action,
                    node_type=self._infer_node_type(label),
                    capability=self._infer_capability(label)
                ))
            
            # Handle {"name": "...", "type": "..."} format
            elif "name" in step:
                action = step.get("id", f"step_{i+1}")
                label = step["name"]
                parsed_steps.append(WorkflowStep(
                    label=label,
                    description=step.get("description", label),
                    action=action,
                    node_type=step.get("type", self._infer_node_type(label)),
                    capability=self._infer_capability(label)
                ))
            
            # Handle simple string items
            elif isinstance(step, str):
                parsed_steps.append(WorkflowStep(
                    label=step,
                    description=step,
                    action=f"step_{i+1}",
                    node_type=self._infer_node_type(step),
                    capability=self._infer_capability(step)
                ))
        
        return parsed_steps


class MultiModelAdapter(ModelAdapter):
    """Adapter that tries multiple strategies to get the best result."""
    
    def __init__(self, ollama_url: str = "http://host.docker.internal:11434"):
        super().__init__(ollama_url)
        self.llama_adapter = LlamaAdapter(ollama_url)
    
    async def generate_workflow_steps(
        self, 
        prompt: str,
        model_name: str,
        temperature: float = 0.3,
        timeout: float = 60.0
    ) -> Optional[List[WorkflowStep]]:
        """Try multiple strategies to generate workflow steps."""
        
        # Strategy 1: Try model-specific adapter
        if "llama" in model_name.lower():
            steps = await self.llama_adapter.generate_workflow_steps(
                prompt, model_name, temperature, timeout
            )
            if steps and len(steps) > 2:  # More than just start/end
                return steps
        
        # Strategy 2: Two-stage generation (understand then structure)
        steps = await self._two_stage_generation(prompt, model_name, temperature, timeout)
        if steps and len(steps) > 2:
            return steps
        
        # Strategy 3: Fallback to simple extraction
        return await self._simple_extraction(prompt)
    
    async def _two_stage_generation(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        timeout: float
    ) -> Optional[List[WorkflowStep]]:
        """First understand the steps, then structure them."""
        
        try:
            # Stage 1: Get natural language steps
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": f"List the specific steps needed for this request: {prompt}\n\nList each step on a new line.",
                        "stream": False,
                        "temperature": temperature,
                        "options": {"num_predict": 1000}
                    }
                )
                
                if response.status_code != 200:
                    return None
                
                nl_steps = response.json().get("response", "").strip().split('\n')
                nl_steps = [s.strip() for s in nl_steps if s.strip() and not s.strip().startswith('#')]
                
                if not nl_steps:
                    return None
                
                # Stage 2: Convert to structured format (can use fast model)
                steps = []
                for i, step_text in enumerate(nl_steps):
                    # Remove numbering if present
                    step_text = re.sub(r'^\d+[\.\)]\s*', '', step_text)
                    
                    if step_text:
                        steps.append(WorkflowStep(
                            label=step_text,
                            description=step_text,
                            action=f"step_{i+1}",
                            node_type=self._infer_node_type(step_text),
                            capability=self._infer_capability(step_text)
                        ))
                
                return steps if len(steps) > 2 else None
                
        except Exception as e:
            logger.error(f"Two-stage generation failed: {e}")
            return None
    
    async def _simple_extraction(self, prompt: str) -> List[WorkflowStep]:
        """Simple keyword-based extraction as last resort."""
        steps = []
        
        # Extract key actions from the prompt
        prompt_lower = prompt.lower()
        
        # Common workflow patterns
        if "marketing campaign" in prompt_lower:
            steps = [
                WorkflowStep(label="Define target audience", action="define_audience", node_type="process", capability="planning"),
                WorkflowStep(label="Create campaign content", action="create_content", node_type="process", capability="content_creation"),
                WorkflowStep(label="Set up marketing channels", action="setup_channels", node_type="adapter", capability="channel_setup"),
                WorkflowStep(label="Launch the campaign", action="launch_campaign", node_type="process", capability="execution"),
                WorkflowStep(label="Track campaign metrics", action="track_metrics", node_type="dataSource", capability="analytics")
            ]
        elif "approval" in prompt_lower:
            steps = [
                WorkflowStep(label="Submit for approval", action="submit_request", node_type="process", capability="submission"),
                WorkflowStep(label="Review the request", action="review_request", node_type="humanAgent", capability="review"),
                WorkflowStep(label="Approve or reject", action="make_decision", node_type="decision", capability="decision_making"),
                WorkflowStep(label="Notify of decision", action="notify_result", node_type="adapter", capability="notification")
            ]
        elif "data" in prompt_lower and ("process" in prompt_lower or "pipeline" in prompt_lower):
            steps = [
                WorkflowStep(label="Extract data from source", action="extract_data", node_type="dataSource", capability="data_extraction"),
                WorkflowStep(label="Validate data quality", action="validate_data", node_type="process", capability="data_validation"),
                WorkflowStep(label="Transform data", action="transform_data", node_type="transformer", capability="data_transformation"),
                WorkflowStep(label="Load to destination", action="load_data", node_type="dataSource", capability="data_loading")
            ]
        
        # If we found pattern-based steps, return them
        if steps:
            return steps
        
        # Otherwise, create a single generic step
        return [
            WorkflowStep(
                label=prompt[:50] + ("..." if len(prompt) > 50 else ""),
                description=prompt,
                action="process_request",
                node_type="process",
                capability="processing"
            )
        ]


# Factory function
def get_model_adapter(model_name: str, ollama_url: str = "http://host.docker.internal:11434") -> ModelAdapter:
    """Get the appropriate adapter for a model."""
    # For now, use the multi-model adapter for everything
    # Can add specific adapters for other models later
    return MultiModelAdapter(ollama_url)