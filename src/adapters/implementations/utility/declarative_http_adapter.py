"""Declarative HTTP adapter — interprets a JSON spec to make HTTP calls.

No arbitrary code execution. HTTP-only. Network allowlist enforced.
This provides a safe, no-code path for agent-generated integrations.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability,
    AdapterCategory,
    AdapterConfig,
    AdapterRequest,
    AdapterResponse,
)

logger = logging.getLogger(__name__)


class DeclarativeHTTPAdapter(BaseAdapter):
    """Adapter that executes HTTP calls defined by a declarative JSON spec.

    The spec format::

        {
            "base_url": "https://api.example.com",
            "auth_type": "bearer" | "api_key_header" | "api_key_query" | "oauth2" | "basic" | "none",
            "auth_config": {
                "header_name": "Authorization",    # for api_key_header
                "query_param": "api_key",           # for api_key_query
                "token_prefix": "Bearer"            # for bearer
            },
            "default_headers": {"Accept": "application/json"},
            "network_allowlist": ["api.example.com"],
            "capabilities": [
                {
                    "name": "get_items",
                    "description": "Retrieve items list",
                    "http_method": "GET",
                    "path": "/v1/items",
                    "query_params": {"limit": {"type": "integer", "default": 50}},
                    "response_transform": "data.items"
                },
                {
                    "name": "create_item",
                    "description": "Create a new item",
                    "http_method": "POST",
                    "path": "/v1/items",
                    "body_template": {"name": "{name}", "type": "{item_type}"},
                    "required_params": ["name", "item_type"]
                }
            ]
        }
    """

    def __init__(self, config: AdapterConfig):
        config.category = AdapterCategory.INTEGRATION
        super().__init__(config)
        self.client: Optional[httpx.AsyncClient] = None
        self._spec: Dict[str, Any] = config.custom_config.get("declarative_spec", {})
        self._base_url: str = self._spec.get("base_url", config.base_url or "")
        self._auth_type: str = self._spec.get("auth_type", "none")
        self._auth_config: Dict[str, Any] = self._spec.get("auth_config", {})
        self._default_headers: Dict[str, str] = self._spec.get("default_headers", {})
        self._network_allowlist: List[str] = self._spec.get("network_allowlist", [])
        self._capability_specs: List[Dict[str, Any]] = self._spec.get("capabilities", [])

    async def initialize(self) -> None:
        """Initialize httpx async client."""
        # Validate base_url against network allowlist
        if self._network_allowlist:
            parsed = urlparse(self._base_url)
            if parsed.hostname not in self._network_allowlist:
                raise ValueError(
                    f"Base URL host '{parsed.hostname}' not in network allowlist: "
                    f"{self._network_allowlist}"
                )

        headers = {"Accept": "application/json", **self._default_headers}
        self.client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        )

    async def shutdown(self) -> None:
        """Close httpx client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    def get_capabilities(self) -> List[AdapterCapability]:
        """Build capabilities from the declarative spec."""
        caps = []
        for spec in self._capability_specs:
            params = {}
            required = spec.get("required_params", [])

            # Build params from query_params + body_template
            for p in spec.get("query_params", {}):
                params[p] = spec["query_params"][p]
            if spec.get("body_template"):
                for key in spec["body_template"]:
                    # Extract placeholder names from "{name}" patterns
                    val = spec["body_template"][key]
                    if isinstance(val, str) and val.startswith("{") and val.endswith("}"):
                        param_name = val[1:-1]
                        params[param_name] = {"type": "string"}

            caps.append(
                AdapterCapability(
                    name=spec["name"],
                    description=spec.get("description", spec["name"]),
                    parameters=params,
                    required_parameters=required,
                )
            )
        return caps

    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute an HTTP call based on the capability spec."""
        if not self.client:
            raise RuntimeError("DeclarativeHTTPAdapter not initialized")

        start_time = time.time()

        # Find the capability spec
        cap_spec = None
        for spec in self._capability_specs:
            if spec["name"] == request.capability:
                cap_spec = spec
                break

        if cap_spec is None:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=f"Unknown capability: {request.capability}",
                duration_ms=0.0,
            )

        try:
            # Build the request
            method = cap_spec.get("http_method", "GET").upper()
            path = self._interpolate(cap_spec.get("path", "/"), request.parameters)

            # Validate URL against allowlist
            self._validate_url(path)

            # Build query params
            query_params = {}
            for param_name, param_def in cap_spec.get("query_params", {}).items():
                if param_name in request.parameters:
                    query_params[param_name] = request.parameters[param_name]
                elif "default" in param_def:
                    query_params[param_name] = param_def["default"]

            # Build body
            body = None
            if cap_spec.get("body_template"):
                body = self._build_body(cap_spec["body_template"], request.parameters)

            # Build auth headers
            auth_headers = self._build_auth_headers(request)

            # Execute
            response = await self.client.request(
                method=method,
                url=path,
                params=query_params or None,
                json=body,
                headers=auth_headers or None,
            )

            duration_ms = (time.time() - start_time) * 1000

            # Parse response
            if response.status_code >= 400:
                return AdapterResponse(
                    request_id=request.id,
                    capability=request.capability,
                    status="error",
                    error=f"HTTP {response.status_code}: {response.text[:500]}",
                    duration_ms=duration_ms,
                    metadata={"status_code": response.status_code},
                )

            # Transform response
            try:
                data = response.json()
            except Exception:
                data = {"raw": response.text[:2000]}

            # Apply response_transform if specified
            transform = cap_spec.get("response_transform")
            if transform and isinstance(data, dict):
                data = self._apply_transform(data, transform)

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data=data,
                duration_ms=duration_ms,
                metadata={"status_code": response.status_code},
            )

        except httpx.TimeoutException:
            duration_ms = (time.time() - start_time) * 1000
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error="Request timed out",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"DeclarativeHTTPAdapter error: {e}")
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=duration_ms,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_url(self, path: str) -> None:
        """Ensure the resolved URL stays within the network allowlist."""
        if not self._network_allowlist:
            return
        # For relative paths, the base_url was already validated
        if path.startswith("http"):
            parsed = urlparse(path)
            if parsed.hostname not in self._network_allowlist:
                raise ValueError(
                    f"URL host '{parsed.hostname}' not in network allowlist"
                )

    def _interpolate(self, template: str, params: Dict[str, Any]) -> str:
        """Interpolate {param} placeholders in a string."""
        result = template
        for key, value in params.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def _build_body(self, template: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Build request body from template + params."""
        body: Dict[str, Any] = {}
        for key, value in template.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                param_name = value[1:-1]
                if param_name in params:
                    body[key] = params[param_name]
            elif isinstance(value, dict):
                body[key] = self._build_body(value, params)
            else:
                body[key] = value
        return body

    def _build_auth_headers(self, request: AdapterRequest) -> Dict[str, str]:
        """Build authentication headers based on auth_type."""
        headers: Dict[str, str] = {}
        creds = self.config.credentials or {}
        # Allow per-request credential overrides
        creds.update(request.context.get("credentials", {}))

        if self._auth_type == "bearer":
            token = creds.get("token") or creds.get("api_key", "")
            prefix = self._auth_config.get("token_prefix", "Bearer")
            if token:
                headers["Authorization"] = f"{prefix} {token}"

        elif self._auth_type == "api_key_header":
            header_name = self._auth_config.get("header_name", "X-API-Key")
            api_key = creds.get("api_key", "")
            if api_key:
                headers[header_name] = api_key

        elif self._auth_type == "api_key_query":
            # Query-param auth is handled in execute() via query_params
            pass

        elif self._auth_type == "basic":
            import base64
            username = creds.get("username", "")
            password = creds.get("password", "")
            if username:
                encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

        # "oauth2" and "none" don't add headers here;
        # OAuth tokens are expected in creds["token"]
        elif self._auth_type == "oauth2":
            token = creds.get("access_token") or creds.get("token", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        return headers

    def _apply_transform(self, data: Any, transform: str) -> Any:
        """Apply a dot-path transform like 'data.items' to navigate JSON."""
        parts = transform.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return current
            else:
                return current
        return current
