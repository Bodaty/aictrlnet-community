"""Factory for creating MCP dispatchers."""

from typing import Optional
import os

from .dispatcher import MCPDispatcher

# Global dispatcher instance
_dispatcher_instance: Optional[MCPDispatcher] = None


def create_mcp_dispatcher(control_plane_url: Optional[str] = None) -> MCPDispatcher:
    """
    Create or get the global MCP dispatcher instance.
    
    Args:
        control_plane_url: URL of the control plane. If not provided, uses environment variable.
    
    Returns:
        MCPDispatcher instance
    """
    global _dispatcher_instance
    
    if _dispatcher_instance is None:
        if control_plane_url is None:
            control_plane_url = os.getenv("CONTROL_PLANE_URL", "http://localhost:8000")
        
        _dispatcher_instance = MCPDispatcher(control_plane_url)
    
    return _dispatcher_instance


def get_dispatcher() -> Optional[MCPDispatcher]:
    """Get the current dispatcher instance if it exists."""
    return _dispatcher_instance


def reset_dispatcher():
    """Reset the global dispatcher instance (useful for testing)."""
    global _dispatcher_instance
    if _dispatcher_instance:
        # Note: Should await close_all() in an async context
        _dispatcher_instance = None