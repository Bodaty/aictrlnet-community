"""AI Process node implementation for AI/ML processing tasks."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..base_node import BaseNode
from ..models import NodeConfig, NodeExecutionResult, NodeStatus
from events.event_bus import event_bus
from adapters.registry import adapter_registry
from adapters.models import AdapterRequest


logger = logging.getLogger(__name__)


class AIProcessNode(BaseNode):
    """Node for AI/ML processing tasks.
    
    Supports various AI operations:
    - Text generation
    - Sentiment analysis
    - Classification
    - Summarization
    - Entity extraction
    - Embeddings generation
    - Translation
    - Q&A
    """
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> NodeExecutionResult:
        """Execute the AI process node."""
        start_time = datetime.utcnow()
        
        try:
            # Get AI processing parameters
            ai_task = self.config.parameters.get("ai_task", "generate")
            adapter_id = self.config.parameters.get("adapter_id")
            
            if not adapter_id:
                # Try to auto-select adapter based on task
                adapter_id = await self._auto_select_adapter(ai_task)
            
            # Get AI adapter class from registry
            adapter_class = adapter_registry.get_adapter_class(adapter_id)
            if not adapter_class:
                raise ValueError(f"AI adapter {adapter_id} not found")
            
            # Create adapter instance
            adapter = adapter_class({})
            
            # Process based on task type
            if ai_task == "generate":
                output_data = await self._process_generation(adapter, input_data)
            elif ai_task == "sentiment":
                output_data = await self._process_sentiment(adapter, input_data)
            elif ai_task == "classify":
                output_data = await self._process_classification(adapter, input_data)
            elif ai_task == "summarize":
                output_data = await self._process_summarization(adapter, input_data)
            elif ai_task == "extract_entities":
                output_data = await self._process_entity_extraction(adapter, input_data)
            elif ai_task == "embeddings":
                output_data = await self._process_embeddings(adapter, input_data)
            elif ai_task == "translate":
                output_data = await self._process_translation(adapter, input_data)
            elif ai_task == "qa":
                output_data = await self._process_qa(adapter, input_data)
            elif ai_task == "custom":
                output_data = await self._process_custom(adapter, input_data)
            else:
                raise ValueError(f"Unsupported AI task: {ai_task}")
            
            # Calculate duration
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Publish completion event
            await event_bus.publish(
                "node.executed",
                {
                    "node_id": self.config.id,
                    "node_type": "aiProcess",
                    "ai_task": ai_task,
                    "adapter_id": adapter_id,
                    "duration_ms": duration_ms
                }
            )
            
            return NodeExecutionResult(
                node_instance_id=self.config.id,
                status=NodeStatus.COMPLETED,
                output_data=output_data,
                duration_ms=duration_ms,
                events_published=1,
                adapters_called=[adapter_id]
            )
            
        except Exception as e:
            logger.error(f"AI Process node {self.config.id} failed: {str(e)}")
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return NodeExecutionResult(
                node_instance_id=self.config.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms
            )
    
    async def _auto_select_adapter(self, ai_task: str) -> str:
        """Auto-select appropriate adapter based on task."""
        # Get available AI adapters from registry
        available_adapters = [
            adapter_type for adapter_type, adapter_class 
            in adapter_registry._adapter_classes.items()
            if adapter_registry._adapter_metadata.get(adapter_type, {}).get('category') == 'ai'
        ]
        
        if not available_adapters:
            raise ValueError("No AI adapters available")
        
        # Prefer certain adapters for specific tasks
        task_preferences = {
            "generate": ["openai", "claude", "gemini", "ollama"],
            "sentiment": ["openai", "claude", "huggingface"],
            "classify": ["openai", "claude", "huggingface"],
            "summarize": ["openai", "claude", "gemini"],
            "extract_entities": ["openai", "claude", "huggingface"],
            "embeddings": ["openai", "cohere", "huggingface"],
            "translate": ["openai", "gemini", "huggingface"],
            "qa": ["openai", "claude", "gemini"]
        }
        
        preferences = task_preferences.get(ai_task, ["openai", "claude"])
        
        # Find first available preferred adapter
        for pref in preferences:
            if pref in available_adapters:
                return pref
        
        # Return first available adapter
        return list(available_adapters.keys())[0]
    
    async def _process_generation(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text generation task."""
        # Get prompt from input or parameters
        prompt = input_data.get("prompt") or self.config.parameters.get("prompt")
        if not prompt:
            raise ValueError("Prompt is required for generation task")
        
        # Build request
        request = AdapterRequest(
            capability="generate",
            parameters={
                "prompt": prompt,
                "max_tokens": self.config.parameters.get("max_tokens", 1000),
                "temperature": self.config.parameters.get("temperature", 0.7),
                "top_p": self.config.parameters.get("top_p", 1.0),
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Generation failed: {response.error}")
        
        return {
            "generated_text": response.data.get("text", ""),
            "usage": response.data.get("usage", {}),
            "model": response.data.get("model")
        }
    
    async def _process_sentiment(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sentiment analysis task."""
        # Get text to analyze
        text = input_data.get("text") or self.config.parameters.get("text")
        if not text:
            raise ValueError("Text is required for sentiment analysis")
        
        # Build request - use classification or custom sentiment
        request = AdapterRequest(
            capability="classify",
            parameters={
                "text": text,
                "categories": ["positive", "negative", "neutral"],
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            # Fallback to generation-based sentiment
            request = AdapterRequest(
                capability="generate",
                parameters={
                    "prompt": f"Analyze the sentiment of the following text and respond with only 'positive', 'negative', or 'neutral':\n\n{text}",
                    "max_tokens": 10,
                    "temperature": 0,
                    "model": self.config.parameters.get("model")
                }
            )
            response = await adapter.execute(request)
            
            if response.status == "error":
                raise Exception(f"Sentiment analysis failed: {response.error}")
            
            sentiment = response.data.get("text", "").strip().lower()
            return {
                "sentiment": sentiment,
                "confidence": 0.8  # Default confidence for generation-based
            }
        
        # Extract classification result
        classifications = response.data.get("classifications", [])
        if classifications:
            top_class = max(classifications, key=lambda x: x.get("confidence", 0))
            return {
                "sentiment": top_class.get("label"),
                "confidence": top_class.get("confidence"),
                "scores": {c["label"]: c["confidence"] for c in classifications}
            }
        
        return {"sentiment": "neutral", "confidence": 0.5}
    
    async def _process_classification(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text classification task."""
        # Get text and categories
        text = input_data.get("text") or self.config.parameters.get("text")
        categories = input_data.get("categories") or self.config.parameters.get("categories")
        
        if not text:
            raise ValueError("Text is required for classification")
        if not categories:
            raise ValueError("Categories are required for classification")
        
        # Build request
        request = AdapterRequest(
            capability="classify",
            parameters={
                "text": text,
                "categories": categories,
                "multi_label": self.config.parameters.get("multi_label", False),
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            # Fallback to generation-based classification
            categories_str = ", ".join(categories)
            request = AdapterRequest(
                capability="generate",
                parameters={
                    "prompt": f"Classify the following text into one of these categories [{categories_str}]. Respond with only the category name:\n\n{text}",
                    "max_tokens": 50,
                    "temperature": 0,
                    "model": self.config.parameters.get("model")
                }
            )
            response = await adapter.execute(request)
            
            if response.status == "error":
                raise Exception(f"Classification failed: {response.error}")
            
            category = response.data.get("text", "").strip()
            return {
                "category": category,
                "confidence": 0.8,
                "multi_label": False
            }
        
        # Extract classification results
        classifications = response.data.get("classifications", [])
        if self.config.parameters.get("multi_label", False):
            # Return all classifications above threshold
            threshold = self.config.parameters.get("threshold", 0.5)
            selected = [c for c in classifications if c.get("confidence", 0) >= threshold]
            return {
                "categories": [c["label"] for c in selected],
                "scores": {c["label"]: c["confidence"] for c in classifications},
                "multi_label": True
            }
        else:
            # Return top classification
            if classifications:
                top_class = max(classifications, key=lambda x: x.get("confidence", 0))
                return {
                    "category": top_class.get("label"),
                    "confidence": top_class.get("confidence"),
                    "scores": {c["label"]: c["confidence"] for c in classifications},
                    "multi_label": False
                }
        
        return {"category": None, "confidence": 0, "multi_label": False}
    
    async def _process_summarization(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text summarization task."""
        # Get text to summarize
        text = input_data.get("text") or self.config.parameters.get("text")
        if not text:
            raise ValueError("Text is required for summarization")
        
        # Get summarization parameters
        max_length = self.config.parameters.get("max_length", 200)
        style = self.config.parameters.get("style", "concise")  # concise, detailed, bullet_points
        
        # Build prompt based on style
        if style == "bullet_points":
            prompt = f"Summarize the following text in bullet points:\n\n{text}"
        elif style == "detailed":
            prompt = f"Provide a detailed summary of the following text:\n\n{text}"
        else:
            prompt = f"Provide a concise summary of the following text in no more than {max_length} words:\n\n{text}"
        
        # Build request
        request = AdapterRequest(
            capability="summarize",
            parameters={
                "text": text,
                "max_length": max_length,
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            # Fallback to generation-based summarization
            request = AdapterRequest(
                capability="generate",
                parameters={
                    "prompt": prompt,
                    "max_tokens": max_length * 2,  # Rough estimate
                    "temperature": 0.3,
                    "model": self.config.parameters.get("model")
                }
            )
            response = await adapter.execute(request)
            
            if response.status == "error":
                raise Exception(f"Summarization failed: {response.error}")
            
            return {
                "summary": response.data.get("text", ""),
                "method": "generation"
            }
        
        return {
            "summary": response.data.get("summary", ""),
            "method": "native"
        }
    
    async def _process_entity_extraction(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process entity extraction task."""
        # Get text to analyze
        text = input_data.get("text") or self.config.parameters.get("text")
        if not text:
            raise ValueError("Text is required for entity extraction")
        
        # Get entity types to extract
        entity_types = self.config.parameters.get("entity_types", 
            ["person", "organization", "location", "date", "email", "phone", "url"])
        
        # Build prompt for extraction
        prompt = f"""Extract the following types of entities from the text: {', '.join(entity_types)}.
        
Format the response as JSON with entity type as keys and lists of found entities as values.
        
Text: {text}"""
        
        # Build request
        request = AdapterRequest(
            capability="generate",
            parameters={
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0,
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Entity extraction failed: {response.error}")
        
        # Try to parse JSON response
        import json
        result_text = response.data.get("text", "")
        try:
            # Extract JSON from response
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                entities = json.loads(result_text[start:end])
            else:
                entities = {}
        except:
            # Fallback to empty entities
            entities = {entity_type: [] for entity_type in entity_types}
        
        return {
            "entities": entities,
            "entity_count": sum(len(v) for v in entities.values())
        }
    
    async def _process_embeddings(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process embeddings generation task."""
        # Get text(s) to embed
        text = input_data.get("text") or self.config.parameters.get("text")
        texts = input_data.get("texts") or self.config.parameters.get("texts")
        
        if not text and not texts:
            raise ValueError("Text or texts are required for embeddings generation")
        
        # Prepare input list
        if texts:
            input_texts = texts if isinstance(texts, list) else [texts]
        else:
            input_texts = [text]
        
        # Build request
        request = AdapterRequest(
            capability="embeddings",
            parameters={
                "texts": input_texts,
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Embeddings generation failed: {response.error}")
        
        embeddings = response.data.get("embeddings", [])
        
        return {
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings)
        }
    
    async def _process_translation(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text translation task."""
        # Get text to translate
        text = input_data.get("text") or self.config.parameters.get("text")
        if not text:
            raise ValueError("Text is required for translation")
        
        # Get languages
        source_lang = self.config.parameters.get("source_language", "auto")
        target_lang = self.config.parameters.get("target_language", "en")
        
        # Build prompt for translation
        if source_lang == "auto":
            prompt = f"Translate the following text to {target_lang}:\n\n{text}"
        else:
            prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}"
        
        # Build request
        request = AdapterRequest(
            capability="generate",
            parameters={
                "prompt": prompt,
                "max_tokens": len(text) * 2,  # Rough estimate
                "temperature": 0.3,
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Translation failed: {response.error}")
        
        return {
            "translated_text": response.data.get("text", ""),
            "source_language": source_lang,
            "target_language": target_lang
        }
    
    async def _process_qa(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process question answering task."""
        # Get question and context
        question = input_data.get("question") or self.config.parameters.get("question")
        context = input_data.get("context") or self.config.parameters.get("context")
        
        if not question:
            raise ValueError("Question is required for Q&A task")
        
        # Build prompt
        if context:
            prompt = f"""Based on the following context, answer the question.
            
Context: {context}
            
Question: {question}
            
Answer:"""
        else:
            prompt = question
        
        # Build request
        request = AdapterRequest(
            capability="generate",
            parameters={
                "prompt": prompt,
                "max_tokens": self.config.parameters.get("max_answer_length", 500),
                "temperature": 0.3,
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Q&A failed: {response.error}")
        
        return {
            "answer": response.data.get("text", ""),
            "question": question,
            "has_context": context is not None
        }
    
    async def _process_custom(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process custom AI task."""
        # Get custom capability and parameters
        capability = self.config.parameters.get("capability", "generate")
        request_params = self.config.parameters.get("request_parameters", {})
        
        # Merge with input data
        for key, value in input_data.items():
            if key not in request_params:
                request_params[key] = value
        
        # Build request
        request = AdapterRequest(
            capability=capability,
            parameters=request_params
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Custom AI task failed: {response.error}")
        
        return response.data
    
    def validate_config(self) -> bool:
        """Validate node configuration."""
        ai_task = self.config.parameters.get("ai_task")
        if not ai_task:
            raise ValueError("ai_task parameter is required")
        
        # Validate task-specific requirements
        if ai_task in ["generate", "sentiment", "classify", "summarize", "extract_entities", "translate", "qa"]:
            # These tasks need either input text or configured text
            pass  # Will be validated at runtime
        elif ai_task == "embeddings":
            # Needs text or texts
            pass  # Will be validated at runtime
        elif ai_task == "custom":
            if not self.config.parameters.get("capability"):
                raise ValueError("capability parameter is required for custom AI task")
        
        return True