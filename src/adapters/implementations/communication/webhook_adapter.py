"""Webhook adapter implementation for generic HTTP integrations."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import json
from datetime import datetime
import hashlib
import hmac
from urllib.parse import urlencode

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory
)
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class WebhookAdapter(BaseAdapter):
    """Adapter for generic webhook/HTTP integrations."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.COMMUNICATION
        super().__init__(config)
        
        self.client: Optional[httpx.AsyncClient] = None
        
        # Default timeout for webhook calls
        self.default_timeout = config.credentials.get("default_timeout", 30.0)
        
        # Security settings
        self.verify_ssl = config.credentials.get("verify_ssl", True)
        self.follow_redirects = config.credentials.get("follow_redirects", True)
    
    async def initialize(self) -> None:
        """Initialize the webhook adapter."""
        # Create HTTP client with configurable settings
        self.client = httpx.AsyncClient(
            timeout=self.default_timeout,
            verify=self.verify_ssl,
            follow_redirects=self.follow_redirects,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0
            )
        )
        
        logger.info("Webhook adapter initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Webhook adapter shutdown")
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return webhook adapter capabilities."""
        return [
            AdapterCapability(
                name="send_webhook",
                description="Send a webhook request to a URL",
                category="http_request",
                parameters={
                    "url": {"type": "string", "description": "Target URL for the webhook"},
                    "method": {"type": "string", "description": "HTTP method", "default": "POST"},
                    "headers": {"type": "object", "description": "HTTP headers"},
                    "body": {"type": "any", "description": "Request body (will be JSON encoded if object)"},
                    "params": {"type": "object", "description": "URL query parameters"},
                    "timeout": {"type": "number", "description": "Request timeout in seconds"},
                    "auth": {"type": "object", "description": "Authentication settings"},
                    "retry_count": {"type": "integer", "description": "Number of retries on failure", "default": 0},
                    "retry_delay": {"type": "number", "description": "Delay between retries in seconds", "default": 1.0}
                },
                required_parameters=["url"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="send_json_webhook",
                description="Send a JSON webhook with proper content-type",
                category="http_request",
                parameters={
                    "url": {"type": "string", "description": "Target URL"},
                    "data": {"type": "object", "description": "JSON data to send"},
                    "headers": {"type": "object", "description": "Additional headers"},
                    "method": {"type": "string", "description": "HTTP method", "default": "POST"},
                    "timeout": {"type": "number", "description": "Request timeout"}
                },
                required_parameters=["url", "data"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="send_form_webhook",
                description="Send form-encoded data webhook",
                category="http_request",
                parameters={
                    "url": {"type": "string", "description": "Target URL"},
                    "form_data": {"type": "object", "description": "Form data to send"},
                    "headers": {"type": "object", "description": "Additional headers"},
                    "method": {"type": "string", "description": "HTTP method", "default": "POST"},
                    "timeout": {"type": "number", "description": "Request timeout"}
                },
                required_parameters=["url", "form_data"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="send_signed_webhook",
                description="Send a webhook with HMAC signature",
                category="http_request",
                parameters={
                    "url": {"type": "string", "description": "Target URL"},
                    "data": {"type": "object", "description": "Data to send"},
                    "secret": {"type": "string", "description": "Secret key for HMAC signature"},
                    "signature_header": {"type": "string", "description": "Header name for signature", "default": "X-Webhook-Signature"},
                    "signature_algorithm": {"type": "string", "description": "HMAC algorithm", "default": "sha256"},
                    "headers": {"type": "object", "description": "Additional headers"},
                    "method": {"type": "string", "description": "HTTP method", "default": "POST"}
                },
                required_parameters=["url", "data", "secret"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="poll_endpoint",
                description="Poll an endpoint at intervals",
                category="polling",
                parameters={
                    "url": {"type": "string", "description": "URL to poll"},
                    "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                    "interval": {"type": "number", "description": "Polling interval in seconds", "default": 60},
                    "max_polls": {"type": "integer", "description": "Maximum number of polls", "default": 10},
                    "stop_condition": {"type": "object", "description": "Condition to stop polling"},
                    "headers": {"type": "object", "description": "HTTP headers"}
                },
                required_parameters=["url"],
                async_supported=True,
                estimated_duration_seconds=60.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="batch_webhooks",
                description="Send multiple webhooks in parallel",
                category="batch",
                parameters={
                    "webhooks": {"type": "array", "description": "Array of webhook configurations"},
                    "concurrency": {"type": "integer", "description": "Max concurrent requests", "default": 5},
                    "continue_on_error": {"type": "boolean", "description": "Continue if a webhook fails", "default": True}
                },
                required_parameters=["webhooks"],
                async_supported=True,
                estimated_duration_seconds=5.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="download_file",
                description="Download a file from a URL",
                category="file_transfer",
                parameters={
                    "url": {"type": "string", "description": "File URL"},
                    "headers": {"type": "object", "description": "HTTP headers"},
                    "chunk_size": {"type": "integer", "description": "Download chunk size", "default": 8192},
                    "max_size": {"type": "integer", "description": "Maximum file size in bytes"}
                },
                required_parameters=["url"],
                async_supported=True,
                estimated_duration_seconds=10.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="upload_file",
                description="Upload a file to a URL",
                category="file_transfer",
                parameters={
                    "url": {"type": "string", "description": "Upload URL"},
                    "file_content": {"type": "string", "description": "File content (base64 encoded)"},
                    "file_name": {"type": "string", "description": "File name"},
                    "field_name": {"type": "string", "description": "Form field name", "default": "file"},
                    "additional_fields": {"type": "object", "description": "Additional form fields"},
                    "headers": {"type": "object", "description": "HTTP headers"}
                },
                required_parameters=["url", "file_content", "file_name"],
                async_supported=True,
                estimated_duration_seconds=10.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="verify_webhook_signature",
                description="Verify an incoming webhook signature",
                category="security",
                parameters={
                    "payload": {"type": "string", "description": "Webhook payload"},
                    "signature": {"type": "string", "description": "Provided signature"},
                    "secret": {"type": "string", "description": "Secret key"},
                    "algorithm": {"type": "string", "description": "HMAC algorithm", "default": "sha256"}
                },
                required_parameters=["payload", "signature", "secret"],
                async_supported=True,
                estimated_duration_seconds=0.1,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="health_check",
                description="Check if an endpoint is healthy",
                category="monitoring",
                parameters={
                    "url": {"type": "string", "description": "URL to check"},
                    "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                    "expected_status": {"type": "integer", "description": "Expected status code", "default": 200},
                    "timeout": {"type": "number", "description": "Request timeout", "default": 10.0}
                },
                required_parameters=["url"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            )
        ]
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a webhook request."""
        # Validate request
        self.validate_request(request)
        
        # Route to appropriate handler
        capability_handlers = {
            "send_webhook": self._handle_send_webhook,
            "send_json_webhook": self._handle_send_json_webhook,
            "send_form_webhook": self._handle_send_form_webhook,
            "send_signed_webhook": self._handle_send_signed_webhook,
            "poll_endpoint": self._handle_poll_endpoint,
            "batch_webhooks": self._handle_batch_webhooks,
            "download_file": self._handle_download_file,
            "upload_file": self._handle_upload_file,
            "verify_webhook_signature": self._handle_verify_signature,
            "health_check": self._handle_health_check
        }
        
        handler = capability_handlers.get(request.capability)
        if not handler:
            raise ValueError(f"Unknown capability: {request.capability}")
        
        return await handler(request)
    
    async def _handle_send_webhook(self, request: AdapterRequest) -> AdapterResponse:
        """Handle generic webhook sending."""
        start_time = datetime.utcnow()
        
        try:
            url = request.parameters["url"]
            method = request.parameters.get("method", "POST").upper()
            headers = request.parameters.get("headers", {})
            body = request.parameters.get("body")
            params = request.parameters.get("params")
            timeout = request.parameters.get("timeout", self.default_timeout)
            auth = request.parameters.get("auth")
            retry_count = request.parameters.get("retry_count", 0)
            retry_delay = request.parameters.get("retry_delay", 1.0)
            
            # Prepare request kwargs
            request_kwargs = {
                "method": method,
                "url": url,
                "headers": headers,
                "timeout": timeout
            }
            
            # Add query parameters
            if params:
                request_kwargs["params"] = params
            
            # Handle body
            if body is not None:
                if isinstance(body, (dict, list)):
                    request_kwargs["json"] = body
                    if "Content-Type" not in headers:
                        headers["Content-Type"] = "application/json"
                else:
                    request_kwargs["content"] = str(body)
            
            # Handle authentication
            if auth:
                auth_type = auth.get("type", "basic")
                if auth_type == "basic":
                    request_kwargs["auth"] = (auth.get("username"), auth.get("password"))
                elif auth_type == "bearer":
                    headers["Authorization"] = f"Bearer {auth.get('token')}"
                elif auth_type == "api_key":
                    key_location = auth.get("key_location", "header")
                    key_name = auth.get("key_name", "X-API-Key")
                    key_value = auth.get("key_value")
                    
                    if key_location == "header":
                        headers[key_name] = key_value
                    elif key_location == "query":
                        if "params" not in request_kwargs:
                            request_kwargs["params"] = {}
                        request_kwargs["params"][key_name] = key_value
            
            # Execute with retries
            last_error = None
            for attempt in range(retry_count + 1):
                try:
                    response = await self.client.request(**request_kwargs)
                    response.raise_for_status()
                    
                    # Parse response
                    response_data = {
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "body": None
                    }
                    
                    # Try to parse response body
                    if response.content:
                        content_type = response.headers.get("content-type", "")
                        if "application/json" in content_type:
                            response_data["body"] = response.json()
                        else:
                            response_data["body"] = response.text
                    
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    # Publish event
                    await event_bus.publish(
                        "adapter.webhook.sent",
                        {
                            "url": url,
                            "method": method,
                            "status_code": response.status_code,
                            "duration_ms": duration_ms
                        },
                        source_id=self.id,
                        source_type="adapter"
                    )
                    
                    return AdapterResponse(
                        request_id=request.id,
                        capability=request.capability,
                        status="success",
                        data=response_data,
                        duration_ms=duration_ms,
                        cost=0.0,
                        metadata={
                            "attempts": attempt + 1,
                            "url": url,
                            "method": method
                        }
                    )
                    
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    last_error = e
                    if attempt < retry_count:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise
            
            # If we get here, all retries failed
            raise last_error
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_send_json_webhook(self, request: AdapterRequest) -> AdapterResponse:
        """Handle JSON webhook sending."""
        # Convert to generic webhook request
        webhook_request = AdapterRequest(
            id=request.id,
            capability="send_webhook",
            parameters={
                "url": request.parameters["url"],
                "method": request.parameters.get("method", "POST"),
                "headers": {
                    "Content-Type": "application/json",
                    **request.parameters.get("headers", {})
                },
                "body": request.parameters["data"],
                "timeout": request.parameters.get("timeout", self.default_timeout)
            },
            metadata=request.metadata
        )
        
        return await self._handle_send_webhook(webhook_request)
    
    async def _handle_send_form_webhook(self, request: AdapterRequest) -> AdapterResponse:
        """Handle form-encoded webhook sending."""
        start_time = datetime.utcnow()
        
        try:
            url = request.parameters["url"]
            form_data = request.parameters["form_data"]
            method = request.parameters.get("method", "POST").upper()
            headers = request.parameters.get("headers", {})
            timeout = request.parameters.get("timeout", self.default_timeout)
            
            # Send form-encoded request
            response = await self.client.request(
                method=method,
                url=url,
                data=form_data,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            
            # Parse response
            response_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text if response.content else None
            }
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data=response_data,
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_send_signed_webhook(self, request: AdapterRequest) -> AdapterResponse:
        """Handle signed webhook sending."""
        start_time = datetime.utcnow()
        
        try:
            url = request.parameters["url"]
            data = request.parameters["data"]
            secret = request.parameters["secret"]
            signature_header = request.parameters.get("signature_header", "X-Webhook-Signature")
            algorithm = request.parameters.get("signature_algorithm", "sha256")
            headers = request.parameters.get("headers", {})
            method = request.parameters.get("method", "POST").upper()
            
            # Serialize data to JSON
            payload = json.dumps(data, separators=(',', ':'), sort_keys=True)
            
            # Generate signature
            if algorithm == "sha256":
                signature = hmac.new(
                    secret.encode('utf-8'),
                    payload.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
            elif algorithm == "sha1":
                signature = hmac.new(
                    secret.encode('utf-8'),
                    payload.encode('utf-8'),
                    hashlib.sha1
                ).hexdigest()
            else:
                raise ValueError(f"Unsupported signature algorithm: {algorithm}")
            
            # Add signature to headers
            headers[signature_header] = signature
            headers["Content-Type"] = "application/json"
            
            # Send request
            response = await self.client.request(
                method=method,
                url=url,
                content=payload,
                headers=headers,
                timeout=self.default_timeout
            )
            response.raise_for_status()
            
            # Parse response
            response_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.json() if response.content else None
            }
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data=response_data,
                duration_ms=duration_ms,
                cost=0.0,
                metadata={
                    "signature_algorithm": algorithm,
                    "signature_header": signature_header
                }
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_poll_endpoint(self, request: AdapterRequest) -> AdapterResponse:
        """Handle endpoint polling."""
        start_time = datetime.utcnow()
        
        try:
            url = request.parameters["url"]
            method = request.parameters.get("method", "GET").upper()
            interval = request.parameters.get("interval", 60)
            max_polls = request.parameters.get("max_polls", 10)
            stop_condition = request.parameters.get("stop_condition", {})
            headers = request.parameters.get("headers", {})
            
            results = []
            
            for poll_count in range(max_polls):
                # Make request
                response = await self.client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=self.default_timeout
                )
                
                # Parse response
                poll_result = {
                    "poll_number": poll_count + 1,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status_code": response.status_code,
                    "body": None
                }
                
                if response.content:
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        poll_result["body"] = response.json()
                    else:
                        poll_result["body"] = response.text
                
                results.append(poll_result)
                
                # Check stop condition
                if stop_condition:
                    field = stop_condition.get("field")
                    value = stop_condition.get("value")
                    operator = stop_condition.get("operator", "equals")
                    
                    if field and self._check_stop_condition(poll_result, field, value, operator):
                        break
                
                # Wait for next poll (except on last iteration)
                if poll_count < max_polls - 1:
                    await asyncio.sleep(interval)
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "polls": results,
                    "total_polls": len(results),
                    "stopped_early": len(results) < max_polls
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_batch_webhooks(self, request: AdapterRequest) -> AdapterResponse:
        """Handle batch webhook sending."""
        start_time = datetime.utcnow()
        
        try:
            webhooks = request.parameters["webhooks"]
            concurrency = request.parameters.get("concurrency", 5)
            continue_on_error = request.parameters.get("continue_on_error", True)
            
            results = []
            semaphore = asyncio.Semaphore(concurrency)
            
            async def send_single_webhook(idx: int, webhook_config: Dict[str, Any]):
                async with semaphore:
                    try:
                        # Create request for single webhook
                        single_request = AdapterRequest(
                            id=f"{request.id}-{idx}",
                            capability="send_webhook",
                            parameters=webhook_config
                        )
                        
                        response = await self._handle_send_webhook(single_request)
                        
                        return {
                            "index": idx,
                            "url": webhook_config.get("url"),
                            "success": response.status == "success",
                            "response": response.data if response.status == "success" else None,
                            "error": response.error if response.status == "error" else None
                        }
                    except Exception as e:
                        if not continue_on_error:
                            raise
                        return {
                            "index": idx,
                            "url": webhook_config.get("url"),
                            "success": False,
                            "error": str(e)
                        }
            
            # Send all webhooks concurrently
            tasks = [
                send_single_webhook(i, webhook)
                for i, webhook in enumerate(webhooks)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=continue_on_error)
            
            # Filter out exceptions if continue_on_error is True
            if continue_on_error:
                results = [
                    r if not isinstance(r, Exception) else {
                        "index": i,
                        "success": False,
                        "error": str(r)
                    }
                    for i, r in enumerate(results)
                ]
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            successful = sum(1 for r in results if r.get("success", False))
            failed = len(results) - successful
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "results": results,
                    "total": len(results),
                    "successful": successful,
                    "failed": failed
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_download_file(self, request: AdapterRequest) -> AdapterResponse:
        """Handle file download."""
        start_time = datetime.utcnow()
        
        try:
            url = request.parameters["url"]
            headers = request.parameters.get("headers", {})
            chunk_size = request.parameters.get("chunk_size", 8192)
            max_size = request.parameters.get("max_size")
            
            # Stream download
            content = bytearray()
            
            async with self.client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                
                # Get content length
                content_length = response.headers.get("content-length")
                if content_length and max_size and int(content_length) > max_size:
                    raise ValueError(f"File too large: {content_length} bytes (max: {max_size})")
                
                async for chunk in response.aiter_bytes(chunk_size):
                    content.extend(chunk)
                    
                    # Check size limit
                    if max_size and len(content) > max_size:
                        raise ValueError(f"File too large: exceeds {max_size} bytes")
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Convert to base64 for transport
            import base64
            content_b64 = base64.b64encode(content).decode('utf-8')
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "content": content_b64,
                    "size": len(content),
                    "content_type": response.headers.get("content-type"),
                    "filename": self._extract_filename(response.headers)
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_upload_file(self, request: AdapterRequest) -> AdapterResponse:
        """Handle file upload."""
        start_time = datetime.utcnow()
        
        try:
            url = request.parameters["url"]
            file_content = request.parameters["file_content"]
            file_name = request.parameters["file_name"]
            field_name = request.parameters.get("field_name", "file")
            additional_fields = request.parameters.get("additional_fields", {})
            headers = request.parameters.get("headers", {})
            
            # Decode base64 content
            import base64
            file_bytes = base64.b64decode(file_content)
            
            # Prepare multipart form data
            files = {field_name: (file_name, file_bytes)}
            
            # Send request
            response = await self.client.post(
                url,
                files=files,
                data=additional_fields,
                headers=headers,
                timeout=self.default_timeout
            )
            response.raise_for_status()
            
            # Parse response
            response_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": None
            }
            
            if response.content:
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    response_data["body"] = response.json()
                else:
                    response_data["body"] = response.text
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data=response_data,
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_verify_signature(self, request: AdapterRequest) -> AdapterResponse:
        """Handle webhook signature verification."""
        start_time = datetime.utcnow()
        
        try:
            payload = request.parameters["payload"]
            signature = request.parameters["signature"]
            secret = request.parameters["secret"]
            algorithm = request.parameters.get("algorithm", "sha256")
            
            # Calculate expected signature
            if algorithm == "sha256":
                expected_signature = hmac.new(
                    secret.encode('utf-8'),
                    payload.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
            elif algorithm == "sha1":
                expected_signature = hmac.new(
                    secret.encode('utf-8'),
                    payload.encode('utf-8'),
                    hashlib.sha1
                ).hexdigest()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Compare signatures
            is_valid = hmac.compare_digest(signature, expected_signature)
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "valid": is_valid,
                    "algorithm": algorithm
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_health_check(self, request: AdapterRequest) -> AdapterResponse:
        """Handle endpoint health check."""
        start_time = datetime.utcnow()
        
        try:
            url = request.parameters["url"]
            method = request.parameters.get("method", "GET").upper()
            expected_status = request.parameters.get("expected_status", 200)
            timeout = request.parameters.get("timeout", 10.0)
            
            # Make request
            response = await self.client.request(
                method=method,
                url=url,
                timeout=timeout
            )
            
            # Check health
            is_healthy = response.status_code == expected_status
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "healthy": is_healthy,
                    "status_code": response.status_code,
                    "expected_status": expected_status,
                    "response_time_ms": duration_ms
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",  # Still success, just unhealthy
                data={
                    "healthy": False,
                    "error": str(e)
                },
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                cost=0.0
            )
    
    def _check_stop_condition(
        self,
        result: Dict[str, Any],
        field: str,
        value: Any,
        operator: str
    ) -> bool:
        """Check if stop condition is met."""
        # Navigate nested fields
        current = result
        for part in field.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        
        # Check condition
        if operator == "equals":
            return current == value
        elif operator == "not_equals":
            return current != value
        elif operator == "contains":
            return value in str(current)
        elif operator == "greater_than":
            return current > value
        elif operator == "less_than":
            return current < value
        else:
            return False
    
    def _extract_filename(self, headers: Dict[str, str]) -> Optional[str]:
        """Extract filename from Content-Disposition header."""
        content_disposition = headers.get("content-disposition", "")
        if "filename=" in content_disposition:
            # Extract filename from header
            parts = content_disposition.split("filename=")
            if len(parts) > 1:
                filename = parts[1].strip('"').strip("'")
                return filename
        return None
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform webhook adapter health check."""
        try:
            # Test with a simple request to a known endpoint
            response = await self.client.get(
                "https://httpbin.org/status/200",
                timeout=5.0
            )
            
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "http_client": "active",
                "ssl_verify": self.verify_ssl,
                "follow_redirects": self.follow_redirects
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }