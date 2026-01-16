"""Base adapter class for all AI agent framework adapters."""

import asyncio
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

from .base_adapter import BaseAdapter
from .models import (
    AdapterConfig, AdapterCapability, AdapterRequest,
    AdapterResponse, AdapterStatus
)
from core.tenant_context import get_current_tenant_id
# These would be imported in a full implementation
# For now, we'll handle licensing at the adapter level
# from core.database import get_db
# from sqlalchemy.ext.asyncio import AsyncSession
# from services.usage_tracker import usage_tracker
# from services.license_enforcer import license_enforcer, LimitType


logger = logging.getLogger(__name__)


class AgenticAIAdapter(BaseAdapter):
    """Base class for AI agent framework adapters (LangChain, AutoGPT, etc)."""
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.agent_type = "base"  # Override in subclasses
        self.supports_memory = True  # Most agent frameworks support memory
        self.supports_tools = True   # Most can use external tools
        self.max_iterations = 10     # Default max reasoning iterations
        
    @abstractmethod
    async def create_agent(self, config: Dict[str, Any]) -> Any:
        """Create an agent instance with given configuration."""
        pass
    
    @abstractmethod
    async def execute_agent(self, agent: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with given request."""
        pass
    
    @abstractmethod
    async def get_agent_state(self, agent: Any) -> Dict[str, Any]:
        """Get current state of the agent including memory."""
        pass
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request using this AI agent adapter."""
        start_time = datetime.utcnow()
        
        try:
            # Validate request
            self.validate_request(request)
            
            # License checking would happen here in full implementation
            # For now, individual adapters handle their own rate limiting
            user_id = request.metadata.get("user_id", "anonymous") if request.metadata else "anonymous"
            tenant_id = request.metadata.get("tenant_id") or get_current_tenant_id() if request.metadata else get_current_tenant_id()
            
            # Handle different capabilities
            if request.capability == "create_agent":
                result = await self._handle_create_agent(request)
            elif request.capability == "execute_agent":
                result = await self._handle_execute_agent(request)
            elif request.capability == "get_agent_state":
                result = await self._handle_get_agent_state(request)
            elif request.capability == "configure_memory":
                result = await self._handle_configure_memory(request)
            elif request.capability == "add_tools":
                result = await self._handle_add_tools(request)
            else:
                raise ValueError(f"Unknown capability: {request.capability}")
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data=result,
                duration_ms=duration_ms,
                cost=self._calculate_cost(request, result)
            )
            
        except Exception as e:
            logger.error(f"Error in {self.config.name} adapter: {str(e)}")
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
    
    async def _handle_create_agent(self, request: AdapterRequest) -> Dict[str, Any]:
        """Handle agent creation request."""
        config = request.parameters.get("config", {})
        
        # Set defaults
        config.setdefault("max_iterations", self.max_iterations)
        config.setdefault("temperature", 0.7)
        config.setdefault("agent_type", self.agent_type)
        
        # Create the agent
        agent = await self.create_agent(config)
        
        # Store agent reference (in real implementation, this would go to a registry)
        agent_id = f"{self.agent_type}-{datetime.utcnow().isoformat()}"
        
        return {
            "agent_id": agent_id,
            "agent_type": self.agent_type,
            "config": config,
            "capabilities": {
                "memory": self.supports_memory,
                "tools": self.supports_tools,
                "max_iterations": config["max_iterations"]
            }
        }
    
    async def _handle_execute_agent(self, request: AdapterRequest) -> Dict[str, Any]:
        """Handle agent execution request."""
        agent_id = request.parameters.get("agent_id")
        if not agent_id:
            raise ValueError("agent_id is required")
        
        # Get agent (in real implementation, from registry)
        agent_config = request.parameters.get("agent_config", {})
        agent = await self.create_agent(agent_config)
        
        # Prepare execution request
        exec_request = {
            "input": request.parameters.get("input"),
            "context": request.parameters.get("context", {}),
            "tools": request.parameters.get("tools", []),
            "max_iterations": request.parameters.get("max_iterations", self.max_iterations)
        }
        
        # Execute the agent
        result = await self.execute_agent(agent, exec_request)
        
        return {
            "agent_id": agent_id,
            "result": result,
            "execution_metadata": {
                "duration_ms": result.get("duration_ms", 0),
                "iterations": result.get("iterations", 0),
                "tools_used": result.get("tools_used", [])
            }
        }
    
    async def _handle_get_agent_state(self, request: AdapterRequest) -> Dict[str, Any]:
        """Handle get agent state request."""
        agent_id = request.parameters.get("agent_id")
        if not agent_id:
            raise ValueError("agent_id is required")
        
        # Get agent (in real implementation, from registry)
        agent_config = request.parameters.get("agent_config", {})
        agent = await self.create_agent(agent_config)
        
        # Get agent state
        state = await self.get_agent_state(agent)
        
        return {
            "agent_id": agent_id,
            "state": state,
            "metadata": {
                "agent_type": self.agent_type,
                "supports_memory": self.supports_memory,
                "supports_tools": self.supports_tools
            }
        }
    
    async def _handle_configure_memory(self, request: AdapterRequest) -> Dict[str, Any]:
        """Handle memory configuration request."""
        if not self.supports_memory:
            raise ValueError(f"{self.agent_type} does not support memory configuration")
        
        memory_config = request.parameters.get("memory_config", {})
        
        # Default implementation - override in subclasses for specific frameworks
        return {
            "status": "configured",
            "memory_type": memory_config.get("type", "conversation"),
            "max_tokens": memory_config.get("max_tokens", 2000),
            "persistence": memory_config.get("persistence", False)
        }
    
    async def _handle_add_tools(self, request: AdapterRequest) -> Dict[str, Any]:
        """Handle add tools request."""
        if not self.supports_tools:
            raise ValueError(f"{self.agent_type} does not support external tools")
        
        tools = request.parameters.get("tools", [])
        
        # Validate tools format
        validated_tools = []
        for tool in tools:
            if not isinstance(tool, dict) or "name" not in tool:
                continue
            validated_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
                "adapter_id": tool.get("adapter_id")  # Reference to another adapter
            })
        
        return {
            "status": "tools_added",
            "tools_count": len(validated_tools),
            "tools": validated_tools
        }
    
    def _calculate_cost(self, request: AdapterRequest, result: Dict[str, Any]) -> float:
        """Calculate cost for the agent operation."""
        # Base cost calculation - override in subclasses for specific pricing
        base_cost = 0.01  # $0.01 per agent operation
        
        # Add cost based on iterations
        iterations = result.get("execution_metadata", {}).get("iterations", 1)
        iteration_cost = iterations * 0.005  # $0.005 per iteration
        
        # Add cost based on tools used
        tools_used = result.get("execution_metadata", {}).get("tools_used", [])
        tools_cost = len(tools_used) * 0.002  # $0.002 per tool use
        
        return base_cost + iteration_cost + tools_cost
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return list of capabilities this AI agent adapter provides."""
        capabilities = [
            AdapterCapability(
                name="create_agent",
                description=f"Create a new {self.agent_type} agent instance",
                required_parameters=["config"],
                optional_parameters=["agent_type", "max_iterations", "temperature"]
            ),
            AdapterCapability(
                name="execute_agent",
                description=f"Execute {self.agent_type} agent with given input",
                required_parameters=["agent_id", "input"],
                optional_parameters=["context", "tools", "max_iterations", "agent_config"]
            ),
            AdapterCapability(
                name="get_agent_state",
                description="Get current state of the agent including memory",
                required_parameters=["agent_id"],
                optional_parameters=["agent_config"]
            )
        ]
        
        if self.supports_memory:
            capabilities.append(
                AdapterCapability(
                    name="configure_memory",
                    description="Configure agent memory settings",
                    required_parameters=["memory_config"],
                    optional_parameters=[]
                )
            )
        
        if self.supports_tools:
            capabilities.append(
                AdapterCapability(
                    name="add_tools",
                    description="Add external tools/adapters for agent to use",
                    required_parameters=["tools"],
                    optional_parameters=[]
                )
            )
        
        return capabilities
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform adapter-specific health check."""
        try:
            # Try to create a simple agent to verify framework is accessible
            test_config = {
                "agent_type": self.agent_type,
                "max_iterations": 1,
                "temperature": 0.0
            }
            agent = await self.create_agent(test_config)
            
            return {
                "status": "healthy",
                "framework": self.agent_type,
                "supports_memory": self.supports_memory,
                "supports_tools": self.supports_tools
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }