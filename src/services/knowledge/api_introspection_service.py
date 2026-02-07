"""API Introspection Service — Phase E.

Reads the FastAPI OpenAPI spec, parses it into a searchable in-memory index,
and exposes three tool methods: list_endpoints, get_endpoint_detail, search_capabilities.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Cache TTL in seconds (30 minutes)
_CACHE_TTL = 30 * 60


class ApiIntrospectionService:
    """Provides searchable access to the platform's OpenAPI specification."""

    def __init__(self):
        self._endpoints: List[Dict[str, Any]] = []
        self._raw_spec: Optional[Dict[str, Any]] = None
        self._last_loaded: float = 0.0
        self._initialized = False
        port = os.environ.get("PORT", "8000")
        self._openapi_url = f"http://localhost:{port}/api/v1/openapi.json"

    async def _ensure_initialized(self) -> None:
        """Lazy-load and cache the OpenAPI spec."""
        now = time.time()
        if self._initialized and (now - self._last_loaded) < _CACHE_TTL:
            return
        await self._load_spec()

    async def _load_spec(self) -> None:
        """Fetch and parse the OpenAPI spec into a flat endpoint index."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(self._openapi_url)
                resp.raise_for_status()
                spec = resp.json()
        except Exception as e:
            logger.warning(f"[ApiIntrospection] Failed to fetch OpenAPI spec: {e}")
            # If we already have cached data, keep it
            if self._endpoints:
                return
            self._endpoints = []
            self._initialized = True
            self._last_loaded = time.time()
            return

        self._raw_spec = spec
        self._endpoints = []

        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, operation in methods.items():
                if method in ("get", "post", "put", "patch", "delete"):
                    endpoint = self._parse_operation(path, method.upper(), operation, spec)
                    self._endpoints.append(endpoint)

        self._initialized = True
        self._last_loaded = time.time()
        logger.info(f"[ApiIntrospection] Indexed {len(self._endpoints)} endpoints")

    def _parse_operation(
        self, path: str, method: str, operation: Dict, spec: Dict
    ) -> Dict[str, Any]:
        """Parse a single OpenAPI operation into a searchable record."""
        summary = operation.get("summary", "")
        description = operation.get("description", "")
        tags = operation.get("tags", [])

        # Extract parameters
        parameters = []
        for param in operation.get("parameters", []):
            parameters.append({
                "name": param.get("name"),
                "in": param.get("in"),
                "required": param.get("required", False),
                "type": param.get("schema", {}).get("type", "string"),
                "description": param.get("description", ""),
            })

        # Extract request body schema (resolve $ref one level)
        request_body = None
        rb = operation.get("requestBody")
        if rb:
            content = rb.get("content", {})
            json_schema = content.get("application/json", {}).get("schema", {})
            request_body = self._resolve_ref(json_schema, spec)

        # Extract response schema
        response_schema = None
        responses = operation.get("responses", {})
        success = responses.get("200") or responses.get("201") or responses.get("204")
        if success:
            resp_content = success.get("content", {})
            resp_json = resp_content.get("application/json", {}).get("schema", {})
            if resp_json:
                response_schema = self._resolve_ref(resp_json, spec)

        # Build searchable text for keyword scoring
        search_text = f"{path} {method} {summary} {description} {' '.join(tags)}".lower()

        return {
            "path": path,
            "method": method,
            "summary": summary,
            "description": description,
            "tags": tags,
            "parameters": parameters,
            "request_body": request_body,
            "response_schema": response_schema,
            "search_text": search_text,
        }

    def _resolve_ref(self, schema: Dict, spec: Dict, depth: int = 0) -> Dict:
        """Resolve a $ref in the OpenAPI spec (one level deep)."""
        if depth > 2:
            return schema
        ref = schema.get("$ref")
        if ref and ref.startswith("#/"):
            parts = ref.lstrip("#/").split("/")
            resolved = spec
            for part in parts:
                resolved = resolved.get(part, {})
                if not resolved:
                    return {"$ref": ref}
            return resolved
        return schema

    # =========================================================================
    # Tool Methods
    # =========================================================================

    async def list_endpoints(
        self,
        method: Optional[str] = None,
        path_prefix: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """List API endpoints with optional filtering."""
        await self._ensure_initialized()

        results = self._endpoints
        if method:
            results = [e for e in results if e["method"] == method.upper()]
        if path_prefix:
            results = [e for e in results if e["path"].startswith(path_prefix)]
        if tag:
            tag_lower = tag.lower()
            results = [e for e in results if any(t.lower() == tag_lower for t in e["tags"])]

        total = len(results)
        results = results[:limit]

        return {
            "endpoints": [
                {
                    "path": e["path"],
                    "method": e["method"],
                    "summary": e["summary"],
                    "tags": e["tags"],
                }
                for e in results
            ],
            "count": len(results),
            "total": total,
            "message": f"Found {total} endpoints" + (f" (showing first {limit})" if total > limit else ""),
        }

    async def get_endpoint_detail(
        self,
        path: str,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """Get full detail for a specific API endpoint."""
        await self._ensure_initialized()

        method_upper = method.upper()
        for ep in self._endpoints:
            if ep["path"] == path and ep["method"] == method_upper:
                return {
                    "path": ep["path"],
                    "method": ep["method"],
                    "summary": ep["summary"],
                    "description": ep["description"],
                    "tags": ep["tags"],
                    "parameters": ep["parameters"],
                    "request_body": ep["request_body"],
                    "response_schema": ep["response_schema"],
                }

        return {"error": f"Endpoint {method_upper} {path} not found"}

    async def search_capabilities(
        self,
        query: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search API endpoints by natural-language query using word-overlap scoring."""
        await self._ensure_initialized()

        if not query or not self._endpoints:
            return {"results": [], "count": 0, "query": query}

        # Tokenize query into words
        query_words = set(re.findall(r'\w+', query.lower()))
        if not query_words:
            return {"results": [], "count": 0, "query": query}

        scored = []
        for ep in self._endpoints:
            ep_words = set(re.findall(r'\w+', ep["search_text"]))
            overlap = query_words & ep_words
            if overlap:
                score = len(overlap) / len(query_words)
                scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:limit]

        return {
            "results": [
                {
                    "path": ep["path"],
                    "method": ep["method"],
                    "summary": ep["summary"],
                    "tags": ep["tags"],
                    "relevance": round(score, 2),
                }
                for score, ep in top
            ],
            "count": len(top),
            "query": query,
            "message": f"Found {len(scored)} matching endpoints for '{query}'" + (
                f" (showing top {limit})" if len(scored) > limit else ""
            ),
        }
