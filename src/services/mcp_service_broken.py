"""MCP (Model Context Protocol) service implementation."""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.exc import IntegrityError

from models.community import MCPServer, MCPTool, MCPInvocation
from schemas.mcp import (
    MCPServerCreate, MCPServerResponse, MCPServerUpdate,
    MCPContextCreate, MCPContextResponse,
    MCPExecuteRequest, MCPExecuteResponse,
    MCPCapability
)

logger = logging.getLogger(__name__)


class MCPDispatcher:
    """Dispatches MCP requests to appropriate servers."""
    
    def __init__(self):
        self.servers = {}
        
    async def register_server(self, server_id: str, server_info: Dict[str, Any]):
        """Register an MCP server."""
        self.servers[server_id] = server_info
        
    async def dispatch(self, server_id: str, method: str, params: Dict[str, Any]) -> Any:
        """Dispatch a request to an MCP server."""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")
            
        # In production, this would make actual HTTP/gRPC calls
        # For now, return mock response
        return {
            "result": f"Executed {method} on server {server_id}",
            "params": params
        }


class MCPTokenizer:
    """Handles token counting and optimization for MCP contexts."""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text (simplified - uses character count / 4)."""
        return len(text) // 4
    
    @staticmethod
    def optimize_context(messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """Optimize context to fit within token limit."""
        total_tokens = 0
        optimized = []
        
        # Start from most recent messages
        for msg in reversed(messages):
            msg_tokens = MCPTokenizer.count_tokens(str(msg))
            if total_tokens + msg_tokens <= max_tokens:
                optimized.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
                
        return optimized


class MCPService:
    """Service for MCP (Model Context Protocol) management."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.dispatcher = MCPDispatcher()
        self.tokenizer = MCPTokenizer()
        
    # Server Management
    
    async def register_server(self, server_data: MCPServerCreate) -> MCPServerResponse:
        """Register a new MCP server."""
        try:
            # Create server
            server = MCPServer(
                id=str(uuid.uuid4()),
                name=server_data.name,
                url=server_data.url,
                provider=server_data.provider,
                model_family=server_data.model_family,
                status="active",
                capabilities=server_data.capabilities,
                config=json.dumps(server_data.config) if server_data.config else None,
                metadata_=json.dumps(server_data.metadata) if server_data.metadata else None,
                created_at=datetime.utcnow()
            )
            
            self.db.add(server)
            
            # Add capabilities
            for cap_name in server_data.capabilities:
                capability = MCPServerCapability(
                    id=str(uuid.uuid4()),
                    server_id=server.id,
                    capability=cap_name,
                    enabled=True,
                    created_at=datetime.utcnow()
                )
                self.db.add(capability)
            
            await self.db.commit()
            await self.db.refresh(server)
            
            # Register with dispatcher
            await self.dispatcher.register_server(server.id, {
                "url": server.url,
                "provider": server.provider,
                "capabilities": server.capabilities
            })
            
            return self._server_to_response(server)
            
        except IntegrityError:
            await self.db.rollback()
            raise ValueError(f"Server with name {server_data.name} already exists")
    
    async def get_server(self, server_id: str) -> Optional[MCPServerResponse]:
        """Get MCP server by ID."""
        result = await self.db.execute(
            select(MCPServer).where(MCPServer.id == server_id)
        )
        server = result.scalar_one_or_none()
        
        if server:
            return self._server_to_response(server)
        return None
    
    async def list_servers(
        self, 
        provider: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[MCPServerResponse]:
        """List all MCP servers with optional filters."""
        query = select(MCPServer)
        
        if provider:
            query = query.where(MCPServer.provider == provider)
        if status:
            query = query.where(MCPServer.status == status)
        
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        servers = result.scalars().all()
        
        return [self._server_to_response(server) for server in servers]
    
    async def update_server(self, server_id: str, update_data: MCPServerUpdate) -> Optional[MCPServerResponse]:
        """Update MCP server."""
        result = await self.db.execute(
            select(MCPServer).where(MCPServer.id == server_id)
        )
        server = result.scalar_one_or_none()
        
        if server:
            if update_data.name is not None:
                server.name = update_data.name
            if update_data.url is not None:
                server.url = update_data.url
            if update_data.status is not None:
                server.status = update_data.status
            if update_data.config is not None:
                server.config = json.dumps(update_data.config)
            if update_data.capabilities is not None:
                server.capabilities = update_data.capabilities
                
                # Update capabilities in separate table
                await self.db.execute(
                    select(MCPServerCapability).where(
                        MCPServerCapability.server_id == server_id
                    ).delete()
                )
                
                for cap_name in update_data.capabilities:
                    capability = MCPServerCapability(
                        id=str(uuid.uuid4()),
                        server_id=server.id,
                        capability=cap_name,
                        enabled=True,
                        created_at=datetime.utcnow()
                    )
                    self.db.add(capability)
            
            server.updated_at = datetime.utcnow()
            await self.db.commit()
            await self.db.refresh(server)
            
            return self._server_to_response(server)
        
        return None
    
    async def get_server_capabilities(self, server_id: str) -> List[MCPCapability]:
        """Get capabilities for a specific server."""
        result = await self.db.execute(
            select(MCPServerCapability).where(MCPServerCapability.server_id == server_id)
        )
        capabilities = result.scalars().all()
        
        return [
            MCPCapability(
                name=cap.capability,
                enabled=cap.enabled,
                description=f"Capability: {cap.capability}"
            )
            for cap in capabilities
        ]
    
    # Context Management
    # TODO: Implement context management when models are available
    
    async def create_context(self, context_data: MCPContextCreate) -> MCPContextResponse:
        """Create a new MCP context."""
        # Validate server exists
        server = await self.get_server(context_data.server_id)
        if not server:
            raise ValueError(f"Server {context_data.server_id} not found")
        
        # Create context
        context = MCPContext(
            id=str(uuid.uuid4()),
            server_id=context_data.server_id,
            workflow_id=context_data.workflow_id,
            task_id=context_data.task_id,
            messages=json.dumps(context_data.messages),
            token_count=self.tokenizer.count_tokens(str(context_data.messages)),
            max_tokens=context_data.max_tokens,
            metadata_=json.dumps(context_data.metadata) if context_data.metadata else None,
            created_at=datetime.utcnow()
        )
        
        self.db.add(context)
        
        # Add to history
        history_entry = MCPContextHistory(
            id=str(uuid.uuid4()),
            context_id=context.id,
            messages=context.messages,
            token_count=context.token_count,
            operation="create",
            timestamp=datetime.utcnow()
        )
        self.db.add(history_entry)
        
        await self.db.commit()
        await self.db.refresh(context)
        
        return self._context_to_response(context)
    
    async def get_context(self, context_id: str) -> Optional[MCPContextResponse]:
        """Get MCP context by ID."""
        result = await self.db.execute(
            select(MCPContext).where(MCPContext.id == context_id)
        )
        context = result.scalar_one_or_none()
        
        if context:
            return self._context_to_response(context)
        return None
    
    async def update_context(self, context_id: str, messages: List[Dict[str, Any]]) -> Optional[MCPContextResponse]:
        """Update context with new messages."""
        result = await self.db.execute(
            select(MCPContext).where(MCPContext.id == context_id)
        )
        context = result.scalar_one_or_none()
        
        if context:
            # Optimize messages to fit token limit
            all_messages = json.loads(context.messages) + messages
            optimized_messages = self.tokenizer.optimize_context(all_messages, context.max_tokens)
            
            # Update context
            context.messages = json.dumps(optimized_messages)
            context.token_count = self.tokenizer.count_tokens(str(optimized_messages))
            context.updated_at = datetime.utcnow()
            
            # Add to history
            history_entry = MCPContextHistory(
                id=str(uuid.uuid4()),
                context_id=context.id,
                messages=context.messages,
                token_count=context.token_count,
                operation="update",
                timestamp=datetime.utcnow()
            )
            self.db.add(history_entry)
            
            await self.db.commit()
            await self.db.refresh(context)
            
            return self._context_to_response(context)
        
        return None
    
    async def get_context_history(self, context_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get context history."""
        result = await self.db.execute(
            select(MCPContextHistory)
            .where(MCPContextHistory.context_id == context_id)
            .order_by(MCPContextHistory.timestamp.desc())
            .limit(limit)
        )
        history = result.scalars().all()
        
        return [
            {
                "id": entry.id,
                "messages": json.loads(entry.messages),
                "token_count": entry.token_count,
                "operation": entry.operation,
                "timestamp": entry.timestamp.isoformat()
            }
            for entry in history
        ]
    
    # Execution
    
    async def execute(self, request: MCPExecuteRequest) -> MCPExecuteResponse:
        """Execute an MCP request."""
        # Get server
        server = await self.get_server(request.server_id)
        if not server:
            raise ValueError(f"Server {request.server_id} not found")
        
        # Get or create context
        context = None
        if request.context_id:
            context = await self.get_context(request.context_id)
        
        # Prepare messages
        messages = []
        if context:
            messages = json.loads(context.messages)
        if request.messages:
            messages.extend(request.messages)
        
        # Dispatch to server
        try:
            result = await self.dispatcher.dispatch(
                server_id=request.server_id,
                method=request.method,
                params={
                    "messages": messages,
                    "model": request.model,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    **request.parameters
                }
            )
            
            # Update context if provided
            if context and request.update_context:
                response_message = {
                    "role": "assistant",
                    "content": result.get("result", "")
                }
                await self.update_context(request.context_id, [response_message])
            
            return MCPExecuteResponse(
                success=True,
                result=result,
                usage={
                    "input_tokens": self.tokenizer.count_tokens(str(messages)),
                    "output_tokens": self.tokenizer.count_tokens(str(result)),
                    "total_tokens": self.tokenizer.count_tokens(str(messages)) + self.tokenizer.count_tokens(str(result))
                },
                metadata={
                    "server_id": request.server_id,
                    "model": request.model or server.model_family,
                    "provider": server.provider
                }
            )
            
        except Exception as e:
            logger.error(f"MCP execution error: {e}")
            return MCPExecuteResponse(
                success=False,
                error=str(e)
            )
    
    # Helper methods
    
    def _server_to_response(self, server: MCPServer) -> MCPServerResponse:
        """Convert server model to response."""
        config = json.loads(server.config) if server.config else None
        metadata = json.loads(server.metadata_) if server.metadata_ else None
        
        return MCPServerResponse(
            id=server.id,
            name=server.name,
            url=server.url,
            provider=server.provider,
            model_family=server.model_family,
            status=server.status,
            capabilities=server.capabilities or [],
            config=config,
            created_at=server.created_at,
            updated_at=server.updated_at,
            metadata=metadata
        )
    
    def _context_to_response(self, context: MCPContextStorage) -> MCPContextResponse:
        """Convert context model to response."""
        messages = json.loads(context.messages) if context.messages else []
        metadata = json.loads(context.metadata_) if context.metadata_ else None
        
        return MCPContextResponse(
            id=context.id,
            server_id=context.server_id,
            workflow_id=context.workflow_id,
            task_id=context.task_id,
            messages=messages,
            token_count=context.token_count,
            max_tokens=context.max_tokens,
            created_at=context.created_at,
            updated_at=context.updated_at,
            metadata=metadata
        )