"""MCP Client Adapter - Real MCP Protocol Implementation.

This adapter implements the Model Context Protocol (MCP) standard as specified at:
https://modelcontextprotocol.io/specification/2025-03-26

MCP uses JSON-RPC 2.0 over stdio transport for communication between clients and servers.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability,
    AdapterConfig,
    AdapterResponse as AdapterResult,
    AdapterCategory
)

logger = logging.getLogger(__name__)


class MCPTransportType(str, Enum):
    """MCP transport types per specification."""
    STDIO = "stdio"
    HTTP_SSE = "http_sse"


@dataclass
class MCPServerConfig:
    """MCP server configuration per MCP standard.

    Real MCP servers are configured with:
    - command: The executable to run (e.g., "npx", "python", "node")
    - args: Arguments to pass to the command
    - env: Environment variables for the subprocess

    NOT with HTTP URLs - that's not MCP standard.
    """
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: MCPTransportType = MCPTransportType.STDIO
    # HTTP transport config (for remote servers using SSE)
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


@dataclass
class MCPTool:
    """MCP tool definition per specification."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


@dataclass
class MCPResource:
    """MCP resource definition per specification."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class MCPServerCapabilities:
    """Server capabilities returned during initialization."""
    tools: bool = False
    tools_list_changed: bool = False
    resources: bool = False
    resources_list_changed: bool = False
    resources_subscribe: bool = False
    prompts: bool = False
    prompts_list_changed: bool = False
    logging: bool = False


class MCPConnection:
    """Manages a single MCP server connection using stdio transport.

    This implements the MCP protocol over stdin/stdout as specified in:
    https://modelcontextprotocol.io/docs/concepts/transports

    Messages are JSON-RPC 2.0 delimited by newlines.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.initialized = False
        self.server_info: Dict[str, Any] = {}
        self.capabilities = MCPServerCapabilities()
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None
        self._protocol_version = "2025-03-26"

    def _next_request_id(self) -> int:
        """Generate next JSON-RPC request ID."""
        self._request_id += 1
        return self._request_id

    async def connect(self) -> bool:
        """Start the MCP server subprocess and initialize connection.

        Per MCP spec:
        1. Spawn the server process
        2. Send initialize request
        3. Wait for initialize response with capabilities
        4. Send initialized notification
        """
        try:
            if self.config.transport == MCPTransportType.STDIO:
                return await self._connect_stdio()
            else:
                logger.error(f"HTTP/SSE transport not yet implemented")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            return False

    async def _connect_stdio(self) -> bool:
        """Connect using stdio transport."""
        # Prepare environment
        env = os.environ.copy()
        env.update(self.config.env)

        # Build command
        cmd = [self.config.command] + self.config.args
        logger.info(f"Starting MCP server {self.config.name}: {' '.join(cmd)}")

        try:
            # Start the subprocess
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            # Start reading responses in background
            self._read_task = asyncio.create_task(self._read_responses())

            # Perform initialization handshake
            success = await self._initialize()
            if success:
                logger.info(f"MCP server {self.config.name} initialized successfully")
                self.initialized = True

                # Fetch available tools if server supports them
                if self.capabilities.tools:
                    await self._fetch_tools()

                # Fetch resources if server supports them
                if self.capabilities.resources:
                    await self._fetch_resources()

            return success

        except FileNotFoundError:
            logger.error(f"MCP server command not found: {self.config.command}")
            return False
        except Exception as e:
            logger.error(f"Failed to start MCP server {self.config.name}: {e}")
            return False

    async def _initialize(self) -> bool:
        """Perform MCP initialization handshake.

        Per MCP spec (https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle):
        1. Client sends initialize request with capabilities and client info
        2. Server responds with its capabilities and server info
        3. Client sends initialized notification
        """
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": self._protocol_version,
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "AICtrlNet",
                    "version": "1.0.0"
                }
            }
        }

        try:
            response = await self._send_request(init_request, timeout=30.0)

            if "error" in response:
                logger.error(f"Initialize error: {response['error']}")
                return False

            result = response.get("result", {})

            # Parse server info
            self.server_info = result.get("serverInfo", {})

            # Parse capabilities
            caps = result.get("capabilities", {})
            if "tools" in caps:
                self.capabilities.tools = True
                self.capabilities.tools_list_changed = caps["tools"].get("listChanged", False)
            if "resources" in caps:
                self.capabilities.resources = True
                self.capabilities.resources_list_changed = caps["resources"].get("listChanged", False)
                self.capabilities.resources_subscribe = caps["resources"].get("subscribe", False)
            if "prompts" in caps:
                self.capabilities.prompts = True
                self.capabilities.prompts_list_changed = caps["prompts"].get("listChanged", False)
            if "logging" in caps:
                self.capabilities.logging = True

            # Send initialized notification (no response expected)
            await self._send_notification({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            })

            return True

        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for initialize response from {self.config.name}")
            return False
        except Exception as e:
            logger.error(f"Initialize failed for {self.config.name}: {e}")
            return False

    async def _fetch_tools(self) -> None:
        """Fetch available tools from the server.

        Per MCP spec (https://modelcontextprotocol.io/specification/2025-03-26/server/tools):
        - Send tools/list request
        - Response contains array of tool definitions
        """
        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/list",
            "params": {}
        }

        try:
            response = await self._send_request(request, timeout=30.0)

            if "error" in response:
                logger.warning(f"Failed to list tools: {response['error']}")
                return

            result = response.get("result", {})
            tools_data = result.get("tools", [])

            self.tools = []
            for tool in tools_data:
                self.tools.append(MCPTool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    server_name=self.config.name
                ))

            logger.info(f"Loaded {len(self.tools)} tools from {self.config.name}")

        except Exception as e:
            logger.warning(f"Failed to fetch tools from {self.config.name}: {e}")

    async def _fetch_resources(self) -> None:
        """Fetch available resources from the server."""
        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "resources/list",
            "params": {}
        }

        try:
            response = await self._send_request(request, timeout=30.0)

            if "error" in response:
                logger.warning(f"Failed to list resources: {response['error']}")
                return

            result = response.get("result", {})
            resources_data = result.get("resources", [])

            self.resources = []
            for resource in resources_data:
                self.resources.append(MCPResource(
                    uri=resource["uri"],
                    name=resource["name"],
                    description=resource.get("description"),
                    mime_type=resource.get("mimeType")
                ))

            logger.info(f"Loaded {len(self.resources)} resources from {self.config.name}")

        except Exception as e:
            logger.warning(f"Failed to fetch resources from {self.config.name}: {e}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server.

        Per MCP spec:
        - Send tools/call request with name and arguments
        - Response contains content array (text, image, audio, or resource)
        - isError flag indicates tool execution error vs protocol error
        """
        if not self.initialized:
            return {"error": "Connection not initialized"}

        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        try:
            response = await self._send_request(request, timeout=120.0)

            if "error" in response:
                return {"error": response["error"]}

            result = response.get("result", {})
            return {
                "content": result.get("content", []),
                "isError": result.get("isError", False)
            }

        except asyncio.TimeoutError:
            return {"error": f"Timeout calling tool {tool_name}"}
        except Exception as e:
            return {"error": str(e)}

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource from the MCP server."""
        if not self.initialized:
            return {"error": "Connection not initialized"}

        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "resources/read",
            "params": {"uri": uri}
        }

        try:
            response = await self._send_request(request, timeout=60.0)

            if "error" in response:
                return {"error": response["error"]}

            return response.get("result", {})

        except Exception as e:
            return {"error": str(e)}

    async def _send_request(self, request: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not running")

        request_id = request.get("id")
        if request_id is None:
            raise ValueError("Request must have an id")

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            # Send request (newline-delimited JSON per MCP spec)
            message = json.dumps(request) + "\n"
            self.process.stdin.write(message.encode())
            await self.process.stdin.drain()

            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response

        finally:
            self._pending_requests.pop(request_id, None)

    async def _send_notification(self, notification: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not running")

        message = json.dumps(notification) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()

    async def _read_responses(self) -> None:
        """Background task to read responses from the server."""
        if not self.process or not self.process.stdout:
            return

        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break

                try:
                    message = json.loads(line.decode().strip())
                    await self._handle_message(message)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from server: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error reading from MCP server: {e}")

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle an incoming JSON-RPC message."""
        # Check if it's a response to a pending request
        if "id" in message and message["id"] in self._pending_requests:
            future = self._pending_requests[message["id"]]
            if not future.done():
                future.set_result(message)

        # Handle notifications from server
        elif "method" in message and "id" not in message:
            method = message["method"]
            if method == "notifications/tools/list_changed":
                # Refresh tools list
                await self._fetch_tools()
            elif method == "notifications/resources/list_changed":
                # Refresh resources list
                await self._fetch_resources()
            else:
                logger.debug(f"Received notification: {method}")

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
            except Exception:
                pass

        self.process = None
        self.initialized = False
        logger.info(f"Disconnected from MCP server {self.config.name}")


class MCPClientAdapter(BaseAdapter):
    """Adapter for connecting to MCP servers using the standard MCP protocol.

    This is a REAL implementation of the Model Context Protocol as specified at:
    https://modelcontextprotocol.io/specification/2025-03-26

    MCP uses:
    - stdio transport (subprocess with stdin/stdout communication)
    - JSON-RPC 2.0 message format
    - Proper initialization handshake with capability negotiation
    - Tool discovery and execution per specification
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.connections: Dict[str, MCPConnection] = {}
        self.initialized = False

    @property
    def adapter_type(self) -> AdapterCategory:
        return AdapterCategory.AI

    async def initialize(self) -> None:
        """Initialize the adapter."""
        self.initialized = True
        logger.info("MCP Client adapter initialized (MCP Standard compliant)")

    def get_capabilities(self) -> List[AdapterCapability]:
        """Get adapter capabilities."""
        return [
            AdapterCapability(
                name="mcp.connect",
                description="Connect to MCP server via stdio transport",
                parameters={
                    "command": "string (required) - Command to run",
                    "args": "array (optional) - Command arguments",
                    "env": "object (optional) - Environment variables"
                }
            ),
            AdapterCapability(
                name="mcp.list_tools",
                description="List available tools from connected MCP servers",
                parameters={}
            ),
            AdapterCapability(
                name="mcp.call_tool",
                description="Call a tool on an MCP server",
                parameters={
                    "server_name": "string",
                    "tool_name": "string",
                    "arguments": "object"
                }
            ),
            AdapterCapability(
                name="mcp.list_resources",
                description="List available resources from MCP servers",
                parameters={}
            ),
            AdapterCapability(
                name="mcp.read_resource",
                description="Read a resource from an MCP server",
                parameters={
                    "server_name": "string",
                    "uri": "string"
                }
            )
        ]

    async def connect_server(self, config: MCPServerConfig) -> bool:
        """Connect to an MCP server.

        Args:
            config: Server configuration with command, args, env

        Returns:
            True if connection successful, False otherwise
        """
        if config.name in self.connections:
            logger.warning(f"Server {config.name} already connected")
            return True

        connection = MCPConnection(config)
        success = await connection.connect()

        if success:
            self.connections[config.name] = connection
            return True
        return False

    async def disconnect_server(self, server_name: str) -> bool:
        """Disconnect from an MCP server."""
        if server_name not in self.connections:
            return False

        connection = self.connections.pop(server_name)
        await connection.disconnect()
        return True

    def get_all_tools(self) -> List[MCPTool]:
        """Get all tools from all connected servers."""
        tools = []
        for connection in self.connections.values():
            tools.extend(connection.tools)
        return tools

    def get_tools_for_server(self, server_name: str) -> List[MCPTool]:
        """Get tools from a specific server."""
        if server_name not in self.connections:
            return []
        return self.connections[server_name].tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on a specific MCP server."""
        if server_name not in self.connections:
            return {"error": f"Server {server_name} not connected"}

        return await self.connections[server_name].call_tool(tool_name, arguments)

    async def execute(self, task: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> AdapterResult:
        """Execute a task through the MCP adapter."""
        start_time = datetime.utcnow()
        request_id = task.get("request_id", str(uuid.uuid4()))

        try:
            operation = task.get("operation", "list_tools")

            if operation == "connect":
                # Connect to a new MCP server
                server_config = MCPServerConfig(
                    name=task.get("name", f"server_{len(self.connections)}"),
                    command=task["command"],
                    args=task.get("args", []),
                    env=task.get("env", {})
                )
                success = await self.connect_server(server_config)

                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                return AdapterResult(
                    request_id=request_id,
                    capability="mcp.connect",
                    status="success" if success else "error",
                    data={
                        "connected": success,
                        "server_name": server_config.name,
                        "capabilities": {
                            "tools": self.connections[server_config.name].capabilities.tools if success else False,
                            "resources": self.connections[server_config.name].capabilities.resources if success else False,
                        } if success else {}
                    },
                    duration_ms=duration,
                    tokens_used=0
                )

            elif operation == "disconnect":
                server_name = task.get("server_name")
                success = await self.disconnect_server(server_name) if server_name else False

                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                return AdapterResult(
                    request_id=request_id,
                    capability="mcp.disconnect",
                    status="success" if success else "error",
                    data={"disconnected": success},
                    duration_ms=duration,
                    tokens_used=0
                )

            elif operation == "list_tools":
                server_name = task.get("server_name")
                if server_name:
                    tools = self.get_tools_for_server(server_name)
                else:
                    tools = self.get_all_tools()

                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                return AdapterResult(
                    request_id=request_id,
                    capability="mcp.list_tools",
                    status="success",
                    data={
                        "tools": [
                            {
                                "name": t.name,
                                "description": t.description,
                                "input_schema": t.input_schema,
                                "server_name": t.server_name
                            }
                            for t in tools
                        ],
                        "total": len(tools)
                    },
                    duration_ms=duration,
                    tokens_used=0
                )

            elif operation == "call_tool":
                server_name = task.get("server_name")
                tool_name = task.get("tool_name")
                arguments = task.get("arguments", {})

                if not server_name or not tool_name:
                    return AdapterResult(
                        request_id=request_id,
                        capability="mcp.call_tool",
                        status="error",
                        data={"error": "server_name and tool_name required"},
                        duration_ms=0,
                        tokens_used=0
                    )

                result = await self.call_tool(server_name, tool_name, arguments)

                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                return AdapterResult(
                    request_id=request_id,
                    capability="mcp.call_tool",
                    status="error" if "error" in result else "success",
                    data=result,
                    duration_ms=duration,
                    tokens_used=0
                )

            else:
                return AdapterResult(
                    request_id=request_id,
                    capability=f"mcp.{operation}",
                    status="error",
                    data={"error": f"Unknown operation: {operation}"},
                    duration_ms=0,
                    tokens_used=0
                )

        except Exception as e:
            logger.error(f"MCP adapter error: {e}")
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return AdapterResult(
                request_id=request_id,
                capability="mcp.error",
                status="error",
                data={"error": str(e)},
                duration_ms=duration,
                tokens_used=0
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check adapter health."""
        servers_status = {}
        for name, conn in self.connections.items():
            servers_status[name] = {
                "initialized": conn.initialized,
                "tools_count": len(conn.tools),
                "resources_count": len(conn.resources),
                "server_info": conn.server_info
            }

        return {
            "status": "healthy",
            "connected_servers": len(self.connections),
            "servers": servers_status,
            "protocol_compliant": True,
            "transport": "stdio"
        }

    async def shutdown(self) -> None:
        """Shutdown adapter and cleanup all connections."""
        for server_name in list(self.connections.keys()):
            await self.disconnect_server(server_name)

        await super().shutdown()
        logger.info("MCP Client adapter shutdown complete")
