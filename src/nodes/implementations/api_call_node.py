"""API Call node implementation for external API integration."""

import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime
import httpx
from urllib.parse import urljoin

from ..base_node import BaseNode
from ..models import NodeConfig
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class APICallNode(BaseNode):
    """Node for making external API calls.
    
    Supports:
    - RESTful API calls (GET, POST, PUT, DELETE, PATCH)
    - Authentication (Bearer, API Key, Basic)
    - Request/response transformations
    - Error handling and retries
    - Response validation
    """
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the API call node. Returns output dict for BaseNode.run() to wrap."""
        # Get API configuration
        url = self._build_url(input_data)
        method = self.config.parameters.get("method", "GET").upper()
        headers = self._build_headers(input_data)
        body = self._build_body(input_data)
        params = self._build_params(input_data)

        # Make API call
        response_data = await self._make_request(
            method=method,
            url=url,
            headers=headers,
            body=body,
            params=params
        )

        # Transform response if needed
        if self.config.parameters.get("response_transform"):
            response_data = await self._transform_response(response_data)

        # Validate response if schema provided
        if self.config.parameters.get("response_schema"):
            await self._validate_response(response_data)

        # Publish completion event
        await event_bus.publish(
            "node.executed",
            {
                "node_id": self.config.id,
                "node_type": "apiCall",
                "method": method,
                "url": url,
                "status_code": response_data.get("status_code", 200)
            }
        )

        return response_data
    
    def _build_url(self, input_data: Dict[str, Any]) -> str:
        """Build the request URL."""
        base_url = self.config.parameters.get("url")
        if not base_url:
            raise ValueError("url parameter is required")
        
        # Handle URL templates
        if "{" in base_url:
            # Replace placeholders with values from input_data
            template_data = {**self.config.parameters, **input_data}
            try:
                base_url = base_url.format(**template_data)
            except KeyError as e:
                raise ValueError(f"Missing template variable: {e}")
        
        # Add path if provided
        path = input_data.get("path") or self.config.parameters.get("path")
        if path:
            base_url = urljoin(base_url, path)
        
        return base_url
    
    def _build_headers(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Build request headers."""
        headers = self.config.parameters.get("headers", {}).copy()
        
        # Add authentication headers
        auth_type = self.config.parameters.get("auth_type")
        
        if auth_type == "bearer":
            token = self.config.parameters.get("auth_token") or input_data.get("auth_token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif auth_type == "api_key":
            api_key = self.config.parameters.get("api_key") or input_data.get("api_key")
            api_key_header = self.config.parameters.get("api_key_header", "X-API-Key")
            if api_key:
                headers[api_key_header] = api_key
        
        elif auth_type == "basic":
            username = self.config.parameters.get("username") or input_data.get("username")
            password = self.config.parameters.get("password") or input_data.get("password")
            if username and password:
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        # Add custom headers from input
        if "headers" in input_data:
            headers.update(input_data["headers"])
        
        # Set content type if not set
        if "Content-Type" not in headers and self.config.parameters.get("method", "GET") in ["POST", "PUT", "PATCH"]:
            headers["Content-Type"] = "application/json"
        
        return headers
    
    def _build_body(self, input_data: Dict[str, Any]) -> Optional[Any]:
        """Build request body."""
        method = self.config.parameters.get("method", "GET").upper()
        
        # Only build body for methods that support it
        if method not in ["POST", "PUT", "PATCH", "DELETE"]:
            return None
        
        # Get body from configuration or input
        body = self.config.parameters.get("body") or input_data.get("body")
        
        # Handle body template
        body_template = self.config.parameters.get("body_template")
        if body_template:
            template_data = {**self.config.parameters, **input_data}
            if isinstance(body_template, str):
                # JSON string template
                try:
                    body_str = body_template.format(**template_data)
                    body = json.loads(body_str)
                except Exception as e:
                    raise ValueError(f"Invalid body template: {e}")
            elif isinstance(body_template, dict):
                # Dictionary template
                body = self._replace_template_values(body_template, template_data)
        
        # Apply request transformation if specified
        if self.config.parameters.get("request_transform"):
            body = self._transform_request(body, input_data)
        
        return body
    
    def _build_params(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build query parameters."""
        params = self.config.parameters.get("params", {}).copy()
        
        # Add params from input
        if "params" in input_data:
            params.update(input_data["params"])
        
        # Handle param templates
        for key, value in params.items():
            if isinstance(value, str) and "{" in value:
                template_data = {**self.config.parameters, **input_data}
                try:
                    params[key] = value.format(**template_data)
                except KeyError:
                    pass  # Keep original value if template fails
        
        return params
    
    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make the HTTP request."""
        timeout = self.config.parameters.get("timeout", 30)
        follow_redirects = self.config.parameters.get("follow_redirects", True)
        max_retries = self.config.parameters.get("max_retries", 0)
        retry_delay = self.config.parameters.get("retry_delay", 1)
        
        async with httpx.AsyncClient(
            follow_redirects=follow_redirects,
            timeout=timeout
        ) as client:
            
            # Prepare request kwargs
            request_kwargs = {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params
            }
            
            # Add body
            if body is not None:
                content_type = headers.get("Content-Type", "application/json")
                if "application/json" in content_type:
                    request_kwargs["json"] = body
                elif "application/x-www-form-urlencoded" in content_type:
                    request_kwargs["data"] = body
                else:
                    request_kwargs["content"] = body
            
            # Make request with retries
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    response = await client.request(**request_kwargs)
                    
                    # Check if we should retry on this status code
                    retry_on_status = self.config.parameters.get("retry_on_status", [500, 502, 503, 504])
                    if response.status_code in retry_on_status and attempt < max_retries:
                        last_error = f"Status code {response.status_code}"
                        await self._sleep(retry_delay * (attempt + 1))
                        continue
                    
                    # Parse response
                    response_data = await self._parse_response(response)
                    
                    # Add metadata
                    response_data["status_code"] = response.status_code
                    response_data["headers"] = dict(response.headers)
                    
                    # Check for errors
                    if not self.config.parameters.get("ignore_errors", False):
                        response.raise_for_status()
                    
                    return response_data
                    
                except httpx.HTTPStatusError as e:
                    if attempt < max_retries:
                        last_error = str(e)
                        await self._sleep(retry_delay * (attempt + 1))
                        continue
                    raise
                except Exception as e:
                    if attempt < max_retries:
                        last_error = str(e)
                        await self._sleep(retry_delay * (attempt + 1))
                        continue
                    raise
            
            # All retries failed
            raise Exception(f"Request failed after {max_retries + 1} attempts: {last_error}")
    
    async def _parse_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse HTTP response."""
        content_type = response.headers.get("content-type", "")
        
        if "application/json" in content_type:
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = response.text
        elif "text/" in content_type:
            data = response.text
        else:
            # Binary data
            data = response.content
        
        return {"data": data}
    
    def _replace_template_values(self, template: Any, data: Dict[str, Any]) -> Any:
        """Recursively replace template values."""
        if isinstance(template, str):
            if "{" in template:
                try:
                    return template.format(**data)
                except KeyError:
                    return template
            return template
        elif isinstance(template, dict):
            return {
                key: self._replace_template_values(value, data)
                for key, value in template.items()
            }
        elif isinstance(template, list):
            return [
                self._replace_template_values(item, data)
                for item in template
            ]
        else:
            return template
    
    def _transform_request(self, body: Any, input_data: Dict[str, Any]) -> Any:
        """Transform request body."""
        transform = self.config.parameters.get("request_transform")
        if not transform:
            return body
        
        if isinstance(transform, dict):
            # Simple field mapping
            if "field_mapping" in transform:
                if isinstance(body, dict):
                    new_body = {}
                    for old_key, new_key in transform["field_mapping"].items():
                        if old_key in body:
                            new_body[new_key] = body[old_key]
                    return new_body
            
            # Add fields
            if "add_fields" in transform and isinstance(body, dict):
                body.update(transform["add_fields"])
            
            # Remove fields
            if "remove_fields" in transform and isinstance(body, dict):
                for field in transform["remove_fields"]:
                    body.pop(field, None)
        
        return body
    
    async def _transform_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform response data."""
        transform = self.config.parameters.get("response_transform")
        if not transform:
            return response_data
        
        data = response_data.get("data")
        
        if isinstance(transform, dict):
            # Extract specific field
            if "extract_field" in transform:
                field_path = transform["extract_field"].split(".")
                extracted = data
                for field in field_path:
                    if isinstance(extracted, dict) and field in extracted:
                        extracted = extracted[field]
                    else:
                        extracted = None
                        break
                response_data["data"] = extracted
            
            # Field mapping
            if "field_mapping" in transform and isinstance(data, dict):
                new_data = {}
                for old_key, new_key in transform["field_mapping"].items():
                    if old_key in data:
                        new_data[new_key] = data[old_key]
                response_data["data"] = new_data
            
            # Array processing
            if "array_processing" in transform and isinstance(data, list):
                processing = transform["array_processing"]
                if processing == "first":
                    response_data["data"] = data[0] if data else None
                elif processing == "last":
                    response_data["data"] = data[-1] if data else None
                elif processing == "count":
                    response_data["data"] = len(data)
                elif "filter" in processing:
                    # Simple filtering
                    filter_field = processing["filter"].get("field")
                    filter_value = processing["filter"].get("value")
                    if filter_field:
                        filtered = [
                            item for item in data
                            if isinstance(item, dict) and item.get(filter_field) == filter_value
                        ]
                        response_data["data"] = filtered
        
        return response_data
    
    async def _validate_response(self, response_data: Dict[str, Any]) -> None:
        """Validate response against schema."""
        schema = self.config.parameters.get("response_schema")
        if not schema:
            return
        
        data = response_data.get("data")
        
        # Simple validation
        if isinstance(schema, dict):
            # Check required fields
            if "required_fields" in schema:
                if not isinstance(data, dict):
                    raise ValueError("Response data must be a dictionary for field validation")
                
                for field in schema["required_fields"]:
                    if field not in data:
                        raise ValueError(f"Required field '{field}' not found in response")
            
            # Check field types
            if "field_types" in schema and isinstance(data, dict):
                for field, expected_type in schema["field_types"].items():
                    if field in data:
                        value = data[field]
                        if expected_type == "string" and not isinstance(value, str):
                            raise ValueError(f"Field '{field}' must be a string")
                        elif expected_type == "number" and not isinstance(value, (int, float)):
                            raise ValueError(f"Field '{field}' must be a number")
                        elif expected_type == "boolean" and not isinstance(value, bool):
                            raise ValueError(f"Field '{field}' must be a boolean")
                        elif expected_type == "array" and not isinstance(value, list):
                            raise ValueError(f"Field '{field}' must be an array")
                        elif expected_type == "object" and not isinstance(value, dict):
                            raise ValueError(f"Field '{field}' must be an object")
            
            # Check status code
            if "expected_status" in schema:
                actual_status = response_data.get("status_code", 200)
                expected = schema["expected_status"]
                if isinstance(expected, list):
                    if actual_status not in expected:
                        raise ValueError(f"Expected status code in {expected}, got {actual_status}")
                else:
                    if actual_status != expected:
                        raise ValueError(f"Expected status code {expected}, got {actual_status}")
    
    async def _sleep(self, seconds: float) -> None:
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(seconds)
    
    def validate_config(self) -> bool:
        """Validate node configuration."""
        if not self.config.parameters.get("url"):
            raise ValueError("url parameter is required")
        
        method = self.config.parameters.get("method", "GET").upper()
        if method not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
            raise ValueError(f"Invalid HTTP method: {method}")
        
        return True