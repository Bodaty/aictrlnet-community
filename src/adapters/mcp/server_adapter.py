"""MCP Server Adapter for connecting to external MCP-compliant servers."""

from typing import Dict, Any, Optional, List
import httpx

from core.ssrf import SSRFError, pin_outbound_client, validate_outbound_url
import time
import uuid
import json
from datetime import datetime
from sqlalchemy import select

from models import MCPServer, MCPServerCapability
from core.database import get_session_maker
from core.exceptions import AdapterError
import logging

logger = logging.getLogger(__name__)


class MCPServerAdapter:
    """
    Adapter for connecting to external MCP-compliant servers.
    Allows AICtrlNet to use capabilities from third-party MCP servers.
    """
    
    def __init__(
        self,
        control_plane_url: str,
        mcp_server_url: str,
        api_key: Optional[str] = None,
        server_name: Optional[str] = None,
        **kwargs
    ):
        self.control_plane_url = control_plane_url
        self.mcp_server_url = mcp_server_url.rstrip('/')
        self.api_key = api_key
        self.server_name = server_name or f"mcp-server-{int(time.time())}"
        self.client = pin_outbound_client(httpx.AsyncClient(timeout=30.0))
        self.capabilities = {}
        self.server_info = {}
        self.server_id = None
        
    async def _fetch_server_info(self) -> Dict[str, Any]:
        """Fetch server information from /mcp/v1/info endpoint"""
        headers = self._get_headers()
        
        try:
            response = await self.client.get(
                await self._validated_url("/mcp/v1/info"),
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except SSRFError:
            # A refused destination is not "the info endpoint is missing".
            raise
        except httpx.HTTPError as e:
            if type(e).__name__ == "SSRFProtectionError":
                raise
            logger.error(f"Failed to fetch server info: {str(e)}")
            # Return basic info if endpoint doesn't exist
            return {
                "protocol_version": "1.0",
                "server_name": self.server_name,
                "capabilities": ["message", "completion"],
                "status": "active"
            }
    
    async def _validated_url(self, path: str) -> str:
        """Check the destination before each send: a stored url can change."""
        import asyncio as _asyncio

        await _asyncio.to_thread(validate_outbound_url, self.mcp_server_url)
        return f"{self.mcp_server_url}{path}"

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for MCP server requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-MCP-Version": "1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _extract_capabilities(self, server_info: Dict[str, Any]) -> Dict[str, bool]:
        """Extract capabilities from server info"""
        capabilities = {}
        
        # Check for explicit capabilities
        if "capabilities" in server_info:
            for cap in server_info["capabilities"]:
                capabilities[cap] = True
        
        # Check for endpoints that indicate capabilities
        if "endpoints" in server_info:
            for endpoint in server_info["endpoints"]:
                if "messages" in endpoint.get("path", ""):
                    capabilities["message"] = True
                if "embeddings" in endpoint.get("path", ""):
                    capabilities["embedding"] = True
                if "tools" in endpoint.get("path", ""):
                    capabilities["tool"] = True
        
        # Default capabilities if none specified
        if not capabilities:
            capabilities = {"message": True, "completion": True}
        
        return capabilities
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task using the external MCP server"""
        logger.debug(f"process_task received: {task}")
        task_payload = task.get("payload", {})
        api_type = task_payload.get("api_type", "message")
        logger.debug(f"Task payload: {task_payload}, api_type: {api_type}")
        
        try:
            if api_type == "message":
                return await self._handle_message_completion(task_payload)
            elif api_type == "embedding":
                return await self._handle_embedding(task_payload)
            elif api_type == "tool":
                return await self._handle_tool_execution(task_payload)
            else:
                raise ValueError(f"Unsupported MCP API type: {api_type}")
        except Exception as e:
            logger.error(f"Failed to process task: {str(e)}")
            raise AdapterError(f"Task processing failed: {str(e)}")
    
    async def _handle_message_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle message/completion requests"""
        logger.debug(f"_handle_message_completion received payload: {payload}")
        messages = payload.get("messages", [])
        model = payload.get("model")
        temperature = payload.get("temperature", 0.7)
        max_tokens = payload.get("max_tokens", 1000)
        logger.debug(f"Extracted messages: {messages}")
        
        headers = self._get_headers()
        
        request_body = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if model:
            request_body["model"] = model
        
        try:
            response = await self.client.post(
                await self._validated_url("/mcp/v1/messages"),
                headers=headers,
                json=request_body
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "response": result.get("content", result.get("response", "")),
                "usage": result.get("usage", {}),
                "model": result.get("model", model),
                "mcp_metadata": {
                    "server_id": self.server_id,
                    "server_name": self.server_name,
                    "api_type": "message"
                }
            }
        except httpx.HTTPError as e:
            logger.error(f"MCP message request failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mcp_metadata": {
                    "server_id": self.server_id,
                    "server_name": self.server_name,
                    "api_type": "message"
                }
            }
    
    async def _handle_embedding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle embedding requests"""
        input_text = payload.get("input", payload.get("text", ""))
        model = payload.get("model")
        
        headers = self._get_headers()
        
        request_body = {
            "input": input_text
        }
        
        if model:
            request_body["model"] = model
        
        try:
            response = await self.client.post(
                await self._validated_url("/mcp/v1/embeddings"),
                headers=headers,
                json=request_body
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "embedding": result.get("embedding", result.get("data", [])),
                "usage": result.get("usage", {}),
                "model": result.get("model", model),
                "mcp_metadata": {
                    "server_id": self.server_id,
                    "server_name": self.server_name,
                    "api_type": "embedding"
                }
            }
        except httpx.HTTPError as e:
            logger.error(f"MCP embedding request failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mcp_metadata": {
                    "server_id": self.server_id,
                    "server_name": self.server_name,
                    "api_type": "embedding"
                }
            }
    
    async def _handle_tool_execution(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution requests"""
        tool_name = payload.get("tool_name")
        tool_input = payload.get("tool_input", {})
        
        headers = self._get_headers()
        
        request_body = {
            "tool": tool_name,
            "input": tool_input
        }
        
        try:
            response = await self.client.post(
                await self._validated_url("/mcp/v1/tools/execute"),
                headers=headers,
                json=request_body
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "result": result.get("result", result.get("output", {})),
                "usage": result.get("usage", {}),
                "mcp_metadata": {
                    "server_id": self.server_id,
                    "server_name": self.server_name,
                    "api_type": "tool",
                    "tool_name": tool_name
                }
            }
        except httpx.HTTPError as e:
            logger.error(f"MCP tool execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mcp_metadata": {
                    "server_id": self.server_id,
                    "server_name": self.server_name,
                    "api_type": "tool",
                    "tool_name": tool_name
                }
            }
    
    async def check_server_health(self) -> Dict[str, Any]:
        """Check if the MCP server is healthy"""
        try:
            server_info = await self._fetch_server_info()
            
            # Update last checked time in database
            SessionLocal = get_session_maker()
            async with SessionLocal() as db:
                result = await db.execute(
                    select(MCPServer).filter_by(id=self.server_id)
                )
                server = result.scalar_one_or_none()
                if server:
                    server.last_checked = datetime.utcnow().timestamp()
                    server.status = "active"
                    await db.commit()
            
            return {
                "status": "healthy",
                "server_id": self.server_id,
                "server_name": self.server_name,
                "url": self.mcp_server_url,
                "protocol_version": server_info.get("protocol_version", "unknown"),
                "last_checked": datetime.utcnow().isoformat()
            }
        except Exception as e:
            # Update status to error in database
            SessionLocal = get_session_maker()
            async with SessionLocal() as db:
                result = await db.execute(
                    select(MCPServer).filter_by(id=self.server_id)
                )
                server = result.scalar_one_or_none()
                if server:
                    server.status = "error"
                    server.last_checked = datetime.utcnow().timestamp()
                    await db.commit()
            
            return {
                "status": "unhealthy",
                "server_id": self.server_id,
                "server_name": self.server_name,
                "url": self.mcp_server_url,
                "error": str(e),
                "last_checked": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()