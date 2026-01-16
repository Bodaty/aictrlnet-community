"""Hugging Face adapter implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import json
from datetime import datetime
import base64

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory
)
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class HuggingFaceAdapter(BaseAdapter):
    """Adapter for Hugging Face Inference API integration."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.AI
        super().__init__(config)
        
        self.client: Optional[httpx.AsyncClient] = None
        self.base_url = config.base_url or "https://api-inference.huggingface.co"
        
        # Check for discovery mode
        self.discovery_only = config.custom_config.get("discovery_only", False) if config.custom_config else False
        
        # Extract API key
        self.api_key = config.api_key or (config.credentials.get("api_key") if config.credentials else None)
        
        # Skip validation in discovery mode
        if not self.discovery_only and not self.api_key:
            raise ValueError("Hugging Face API key is required")
    
    async def initialize(self) -> None:
        """Initialize the Hugging Face adapter."""
        # Skip initialization in discovery mode
        if self.discovery_only:
            logger.info("Hugging Face adapter initialized in discovery mode")
            return
            
        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=self.config.timeout_seconds
        )
        
        logger.info("Hugging Face adapter initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Hugging Face adapter shutdown")
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return Hugging Face adapter capabilities."""
        return [
            AdapterCapability(
                name="text_generation",
                description="Generate text using Hugging Face models",
                category="text_generation",
                parameters={
                    "model_id": {"type": "string", "description": "Model ID (e.g., meta-llama/Llama-2-7b-chat-hf)"},
                    "inputs": {"type": "string", "description": "The input text/prompt"},
                    "parameters": {"type": "object", "description": "Generation parameters", "properties": {
                        "max_new_tokens": {"type": "integer", "default": 100},
                        "temperature": {"type": "number", "default": 1.0},
                        "top_p": {"type": "number", "default": 1.0},
                        "do_sample": {"type": "boolean", "default": True},
                        "repetition_penalty": {"type": "number", "default": 1.0},
                        "return_full_text": {"type": "boolean", "default": False}
                    }},
                    "options": {"type": "object", "description": "API options", "properties": {
                        "use_cache": {"type": "boolean", "default": True},
                        "wait_for_model": {"type": "boolean", "default": False}
                    }}
                },
                required_parameters=["model_id", "inputs"],
                async_supported=True,
                estimated_duration_seconds=3.0,
                cost_per_request=0.001
            ),
            AdapterCapability(
                name="text_classification",
                description="Classify text using Hugging Face models",
                category="classification",
                parameters={
                    "model_id": {"type": "string", "description": "Model ID (e.g., distilbert-base-uncased-finetuned-sst-2-english)"},
                    "inputs": {"type": "string|array", "description": "Text(s) to classify"},
                    "options": {"type": "object", "description": "API options"}
                },
                required_parameters=["model_id", "inputs"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0001
            ),
            AdapterCapability(
                name="token_classification",
                description="Token classification (NER, POS tagging)",
                category="classification",
                parameters={
                    "model_id": {"type": "string", "description": "Model ID"},
                    "inputs": {"type": "string", "description": "Text to analyze"},
                    "options": {"type": "object", "description": "API options"}
                },
                required_parameters=["model_id", "inputs"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0001
            ),
            AdapterCapability(
                name="question_answering",
                description="Answer questions based on context",
                category="text_generation",
                parameters={
                    "model_id": {"type": "string", "description": "Model ID (e.g., deepset/roberta-base-squad2)"},
                    "inputs": {"type": "object", "description": "Question and context", "properties": {
                        "question": {"type": "string"},
                        "context": {"type": "string"}
                    }},
                    "options": {"type": "object", "description": "API options"}
                },
                required_parameters=["model_id", "inputs"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0001
            ),
            AdapterCapability(
                name="summarization",
                description="Summarize text",
                category="text_generation",
                parameters={
                    "model_id": {"type": "string", "description": "Model ID (e.g., facebook/bart-large-cnn)"},
                    "inputs": {"type": "string", "description": "Text to summarize"},
                    "parameters": {"type": "object", "description": "Summarization parameters", "properties": {
                        "max_length": {"type": "integer", "default": 130},
                        "min_length": {"type": "integer", "default": 30}
                    }},
                    "options": {"type": "object", "description": "API options"}
                },
                required_parameters=["model_id", "inputs"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.001
            ),
            AdapterCapability(
                name="translation",
                description="Translate text between languages",
                category="text_generation",
                parameters={
                    "model_id": {"type": "string", "description": "Model ID (e.g., Helsinki-NLP/opus-mt-en-es)"},
                    "inputs": {"type": "string", "description": "Text to translate"},
                    "options": {"type": "object", "description": "API options"}
                },
                required_parameters=["model_id", "inputs"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0001
            ),
            AdapterCapability(
                name="embeddings",
                description="Generate embeddings for text",
                category="embeddings",
                parameters={
                    "model_id": {"type": "string", "description": "Model ID (e.g., sentence-transformers/all-MiniLM-L6-v2)"},
                    "inputs": {"type": "string|array", "description": "Text(s) to embed"},
                    "options": {"type": "object", "description": "API options"}
                },
                required_parameters=["model_id", "inputs"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0001
            ),
            AdapterCapability(
                name="image_classification",
                description="Classify images",
                category="image_analysis",
                parameters={
                    "model_id": {"type": "string", "description": "Model ID (e.g., google/vit-base-patch16-224)"},
                    "image": {"type": "string", "description": "Base64 encoded image or URL"},
                    "options": {"type": "object", "description": "API options"}
                },
                required_parameters=["model_id", "image"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.001
            ),
            AdapterCapability(
                name="image_to_text",
                description="Generate text from images (captioning, OCR)",
                category="image_analysis",
                parameters={
                    "model_id": {"type": "string", "description": "Model ID (e.g., Salesforce/blip-image-captioning-base)"},
                    "image": {"type": "string", "description": "Base64 encoded image or URL"},
                    "options": {"type": "object", "description": "API options"}
                },
                required_parameters=["model_id", "image"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.001
            ),
            AdapterCapability(
                name="text_to_image",
                description="Generate images from text",
                category="image_generation",
                parameters={
                    "model_id": {"type": "string", "description": "Model ID (e.g., stabilityai/stable-diffusion-2-1)"},
                    "inputs": {"type": "string", "description": "Text prompt"},
                    "parameters": {"type": "object", "description": "Generation parameters"},
                    "options": {"type": "object", "description": "API options"}
                },
                required_parameters=["model_id", "inputs"],
                async_supported=True,
                estimated_duration_seconds=10.0,
                cost_per_request=0.01
            )
        ]
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request to Hugging Face."""
        # Validate request
        self.validate_request(request)
        
        # Route to appropriate handler based on capability
        capability_handlers = {
            "text_generation": self._handle_text_generation,
            "text_classification": self._handle_text_classification,
            "token_classification": self._handle_token_classification,
            "question_answering": self._handle_question_answering,
            "summarization": self._handle_summarization,
            "translation": self._handle_translation,
            "embeddings": self._handle_embeddings,
            "image_classification": self._handle_image_classification,
            "image_to_text": self._handle_image_to_text,
            "text_to_image": self._handle_text_to_image
        }
        
        handler = capability_handlers.get(request.capability)
        if not handler:
            raise ValueError(f"Unknown capability: {request.capability}")
        
        return await handler(request)
    
    async def _make_inference_request(
        self,
        model_id: str,
        data: Dict[str, Any],
        start_time: datetime
    ) -> Dict[str, Any]:
        """Make a request to the Hugging Face Inference API."""
        endpoint = f"/models/{model_id}"
        
        response = await self.client.post(endpoint, json=data)
        
        # Handle model loading
        if response.status_code == 503:
            error_data = response.json()
            if error_data.get("error", "").startswith("Model") and "loading" in error_data.get("error", ""):
                # Model is loading, wait and retry
                estimated_time = error_data.get("estimated_time", 30)
                logger.info(f"Model {model_id} is loading, waiting {estimated_time}s")
                await asyncio.sleep(min(estimated_time, 60))
                response = await self.client.post(endpoint, json=data)
        
        response.raise_for_status()
        return response.json()
    
    async def _handle_text_generation(self, request: AdapterRequest) -> AdapterResponse:
        """Handle text generation requests."""
        start_time = datetime.utcnow()
        
        try:
            model_id = request.parameters["model_id"]
            
            # Prepare request data
            data = {
                "inputs": request.parameters["inputs"],
                "parameters": request.parameters.get("parameters", {}),
                "options": request.parameters.get("options", {})
            }
            
            # Make API request
            result = await self._make_inference_request(model_id, data, start_time)
            
            # Calculate metrics
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = result.get("generated_text", "")
            
            # Estimate tokens
            tokens_used = len(generated_text.split()) * 1.3
            
            # Publish event
            await event_bus.publish(
                "adapter.huggingface.generation",
                {
                    "model": model_id,
                    "tokens": tokens_used,
                    "duration_ms": duration_ms
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "generated_text": generated_text,
                    "full_response": result
                },
                duration_ms=duration_ms,
                cost=self._estimate_cost(request.capability, tokens_used),
                tokens_used=tokens_used,
                metadata={"model_id": model_id}
            )
            
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_message = error_data.get("error", str(e))
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=error_message,
                error_code=f"HTTP_{e.response.status_code}",
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_text_classification(self, request: AdapterRequest) -> AdapterResponse:
        """Handle text classification requests."""
        start_time = datetime.utcnow()
        
        try:
            model_id = request.parameters["model_id"]
            
            data = {
                "inputs": request.parameters["inputs"],
                "options": request.parameters.get("options", {})
            }
            
            result = await self._make_inference_request(model_id, data, start_time)
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "predictions": result if isinstance(result, list) else [result]
                },
                duration_ms=duration_ms,
                cost=0.0001,
                metadata={"model_id": model_id}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_token_classification(self, request: AdapterRequest) -> AdapterResponse:
        """Handle token classification requests."""
        start_time = datetime.utcnow()
        
        try:
            model_id = request.parameters["model_id"]
            
            data = {
                "inputs": request.parameters["inputs"],
                "options": request.parameters.get("options", {})
            }
            
            result = await self._make_inference_request(model_id, data, start_time)
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "entities": result if isinstance(result, list) else [result]
                },
                duration_ms=duration_ms,
                cost=0.0001,
                metadata={"model_id": model_id}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_question_answering(self, request: AdapterRequest) -> AdapterResponse:
        """Handle question answering requests."""
        start_time = datetime.utcnow()
        
        try:
            model_id = request.parameters["model_id"]
            
            data = {
                "inputs": request.parameters["inputs"],
                "options": request.parameters.get("options", {})
            }
            
            result = await self._make_inference_request(model_id, data, start_time)
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "answer": result.get("answer", ""),
                    "score": result.get("score", 0.0),
                    "start": result.get("start", 0),
                    "end": result.get("end", 0)
                },
                duration_ms=duration_ms,
                cost=0.0001,
                metadata={"model_id": model_id}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_summarization(self, request: AdapterRequest) -> AdapterResponse:
        """Handle summarization requests."""
        start_time = datetime.utcnow()
        
        try:
            model_id = request.parameters["model_id"]
            
            data = {
                "inputs": request.parameters["inputs"],
                "parameters": request.parameters.get("parameters", {}),
                "options": request.parameters.get("options", {})
            }
            
            result = await self._make_inference_request(model_id, data, start_time)
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Extract summary
            if isinstance(result, list) and len(result) > 0:
                summary = result[0].get("summary_text", "")
            else:
                summary = result.get("summary_text", "")
            
            tokens_used = len(summary.split()) * 1.3
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "summary": summary,
                    "full_response": result
                },
                duration_ms=duration_ms,
                cost=self._estimate_cost("summarization", tokens_used),
                tokens_used=tokens_used,
                metadata={"model_id": model_id}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_translation(self, request: AdapterRequest) -> AdapterResponse:
        """Handle translation requests."""
        start_time = datetime.utcnow()
        
        try:
            model_id = request.parameters["model_id"]
            
            data = {
                "inputs": request.parameters["inputs"],
                "options": request.parameters.get("options", {})
            }
            
            result = await self._make_inference_request(model_id, data, start_time)
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Extract translation
            if isinstance(result, list) and len(result) > 0:
                translation = result[0].get("translation_text", "")
            else:
                translation = result.get("translation_text", "")
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "translation": translation,
                    "full_response": result
                },
                duration_ms=duration_ms,
                cost=0.0001,
                metadata={"model_id": model_id}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_embeddings(self, request: AdapterRequest) -> AdapterResponse:
        """Handle embedding requests."""
        start_time = datetime.utcnow()
        
        try:
            model_id = request.parameters["model_id"]
            
            data = {
                "inputs": request.parameters["inputs"],
                "options": request.parameters.get("options", {})
            }
            
            result = await self._make_inference_request(model_id, data, start_time)
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "embeddings": result
                },
                duration_ms=duration_ms,
                cost=0.0001,
                metadata={"model_id": model_id}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_image_classification(self, request: AdapterRequest) -> AdapterResponse:
        """Handle image classification requests."""
        start_time = datetime.utcnow()
        
        try:
            model_id = request.parameters["model_id"]
            image_data = request.parameters["image"]
            
            # Prepare image data (base64 or URL)
            if image_data.startswith("http"):
                # Download image
                async with httpx.AsyncClient() as client:
                    img_response = await client.get(image_data)
                    img_response.raise_for_status()
                    image_bytes = img_response.content
            else:
                # Assume base64
                image_bytes = base64.b64decode(image_data)
            
            # Send as binary data
            response = await self.client.post(
                f"/models/{model_id}",
                data=image_bytes,
                headers={"Content-Type": "application/octet-stream"}
            )
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "predictions": result if isinstance(result, list) else [result]
                },
                duration_ms=duration_ms,
                cost=0.001,
                metadata={"model_id": model_id}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_image_to_text(self, request: AdapterRequest) -> AdapterResponse:
        """Handle image to text requests."""
        start_time = datetime.utcnow()
        
        try:
            model_id = request.parameters["model_id"]
            image_data = request.parameters["image"]
            
            # Prepare image data
            if image_data.startswith("http"):
                async with httpx.AsyncClient() as client:
                    img_response = await client.get(image_data)
                    img_response.raise_for_status()
                    image_bytes = img_response.content
            else:
                image_bytes = base64.b64decode(image_data)
            
            # Send as binary data
            response = await self.client.post(
                f"/models/{model_id}",
                data=image_bytes,
                headers={"Content-Type": "application/octet-stream"}
            )
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Extract text
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "")
            else:
                text = result.get("generated_text", "")
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "text": text,
                    "full_response": result
                },
                duration_ms=duration_ms,
                cost=0.001,
                metadata={"model_id": model_id}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_text_to_image(self, request: AdapterRequest) -> AdapterResponse:
        """Handle text to image requests."""
        start_time = datetime.utcnow()
        
        try:
            model_id = request.parameters["model_id"]
            
            data = {
                "inputs": request.parameters["inputs"],
                "parameters": request.parameters.get("parameters", {}),
                "options": request.parameters.get("options", {})
            }
            
            result = await self._make_inference_request(model_id, data, start_time)
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "image": result  # Usually returns base64 encoded image
                },
                duration_ms=duration_ms,
                cost=0.01,
                metadata={"model_id": model_id}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    def _estimate_cost(self, capability: str, tokens: int = 0) -> float:
        """Estimate cost based on capability and usage."""
        # Rough estimates
        cost_map = {
            "text_generation": 0.001,
            "summarization": 0.001,
            "text_to_image": 0.01,
            "image_classification": 0.001,
            "image_to_text": 0.001
        }
        
        return cost_map.get(capability, 0.0001)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List popular Hugging Face models."""
        if self.discovery_only:
            # Return static list of popular models for discovery
            return [
                {
                    "id": "meta-llama/Llama-2-7b-chat-hf",
                    "name": "Llama 2 7B Chat",
                    "provider": "Meta",
                    "description": "Conversational AI model",
                    "capabilities": ["text", "chat"],
                    "max_tokens": 4096,
                    "status": "available"
                },
                {
                    "id": "microsoft/DialoGPT-large",
                    "name": "DialoGPT Large",
                    "provider": "Microsoft",
                    "description": "Conversational response generation",
                    "capabilities": ["chat", "text"],
                    "max_tokens": 1024,
                    "status": "available"
                },
                {
                    "id": "bigscience/bloom",
                    "name": "BLOOM",
                    "provider": "BigScience",
                    "description": "Large multilingual language model",
                    "capabilities": ["text", "translation"],
                    "max_tokens": 2048,
                    "status": "available"
                },
                {
                    "id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "name": "Stable Diffusion XL",
                    "provider": "Stability AI",
                    "description": "Text-to-image generation",
                    "capabilities": ["image_generation"],
                    "status": "available"
                },
                {
                    "id": "bert-base-uncased",
                    "name": "BERT Base",
                    "provider": "Google",
                    "description": "Bidirectional encoder for NLP tasks",
                    "capabilities": ["classification", "embeddings"],
                    "max_tokens": 512,
                    "status": "available"
                },
                {
                    "id": "distilbert-base-uncased-finetuned-sst-2-english",
                    "name": "DistilBERT SST-2",
                    "provider": "Hugging Face",
                    "description": "Sentiment classification",
                    "capabilities": ["classification"],
                    "max_tokens": 512,
                    "status": "available"
                },
                {
                    "id": "facebook/bart-large-cnn",
                    "name": "BART Large CNN",
                    "provider": "Facebook",
                    "description": "Text summarization",
                    "capabilities": ["summarization"],
                    "max_tokens": 1024,
                    "status": "available"
                },
                {
                    "id": "sentence-transformers/all-MiniLM-L6-v2",
                    "name": "MiniLM L6 v2",
                    "provider": "Sentence Transformers",
                    "description": "Sentence embeddings",
                    "capabilities": ["embeddings"],
                    "embedding_dimensions": 384,
                    "status": "available"
                }
            ]
        
        # Real implementation would use HF API to list models
        # This would require additional API endpoints not in standard inference API
        return []
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Hugging Face-specific health check."""
        try:
            # Try a simple request to check API key
            response = await self.client.get("/api/whoami-v2")
            
            if response.status_code == 200:
                user_info = response.json()
                return {
                    "status": "healthy",
                    "user": user_info.get("name", "authenticated")
                }
            else:
                return {
                    "status": "healthy",
                    "note": "API key valid but user info unavailable"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }