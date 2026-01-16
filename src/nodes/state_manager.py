"""Simple in-memory state management for Community Edition node execution."""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time
from enum import Enum

from .models import NodeInstance, NodeStatus, WorkflowInstance
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class StateScope(Enum):
    """State scope for Community Edition."""
    NODE = "node"
    WORKFLOW = "workflow"
    GLOBAL = "global"


class StateStorageBackend(Enum):
    """State storage backend types."""
    MEMORY = "memory"


class SimpleStateManager:
    """Simple in-memory state manager for Community Edition.
    
    This provides basic state management without database persistence.
    State is lost when the process restarts.
    """
    
    def __init__(self):
        """Initialize the simple state manager."""
        self._state: Dict[str, Dict[str, Any]] = {
            StateScope.NODE.value: {},
            StateScope.WORKFLOW.value: {},
            StateScope.GLOBAL.value: {}
        }
        self._locks: Dict[str, asyncio.Lock] = {}
        self._checkpoints: Dict[str, List[Dict[str, Any]]] = {}
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for a key."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]
    
    def _make_key(self, scope: StateScope, scope_id: Optional[str], key: str) -> str:
        """Make a composite key for state storage."""
        if scope == StateScope.GLOBAL:
            return f"global:{key}"
        elif scope_id:
            return f"{scope.value}:{scope_id}:{key}"
        else:
            return f"{scope.value}::{key}"
    
    async def get_state(
        self,
        key: str,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """Get state value."""
        composite_key = self._make_key(scope, scope_id, key)
        return self._state.get(scope.value, {}).get(composite_key, default)
    
    async def set_state(
        self,
        key: str,
        value: Any,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> None:
        """Set state value."""
        composite_key = self._make_key(scope, scope_id, key)
        
        async with self._get_lock(composite_key):
            if scope.value not in self._state:
                self._state[scope.value] = {}
            
            self._state[scope.value][composite_key] = {
                "value": value,
                "updated_at": datetime.utcnow().isoformat(),
                "ttl": ttl
            }
            
            # Emit state change event
            await event_bus.emit("state.changed", {
                "key": key,
                "scope": scope.value,
                "scope_id": scope_id,
                "timestamp": time.time()
            })
    
    async def delete_state(
        self,
        key: str,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None
    ) -> bool:
        """Delete state value."""
        composite_key = self._make_key(scope, scope_id, key)
        
        async with self._get_lock(composite_key):
            if scope.value in self._state and composite_key in self._state[scope.value]:
                del self._state[scope.value][composite_key]
                return True
            return False
    
    async def list_state(
        self,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None,
        prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """List all state entries for a scope."""
        result = {}
        scope_data = self._state.get(scope.value, {})
        
        for composite_key, entry in scope_data.items():
            # Check if this entry matches our scope_id
            parts = composite_key.split(":", 2)
            if len(parts) >= 3 and (not scope_id or parts[1] == scope_id):
                key = parts[2]
                if not prefix or key.startswith(prefix):
                    result[key] = entry.get("value")
        
        return result
    
    async def clear_state(
        self,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None
    ) -> int:
        """Clear all state for a scope."""
        count = 0
        scope_data = self._state.get(scope.value, {})
        keys_to_delete = []
        
        for composite_key in scope_data:
            parts = composite_key.split(":", 2)
            if len(parts) >= 2 and (not scope_id or parts[1] == scope_id):
                keys_to_delete.append(composite_key)
        
        for key in keys_to_delete:
            del scope_data[key]
            count += 1
        
        return count
    
    async def create_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: Optional[str] = None
    ) -> str:
        """Create a state checkpoint."""
        if not checkpoint_id:
            checkpoint_id = f"checkpoint_{int(time.time() * 1000)}"
        
        # Copy current workflow state
        workflow_state = {}
        scope_data = self._state.get(StateScope.WORKFLOW.value, {})
        
        for composite_key, entry in scope_data.items():
            if f"workflow:{workflow_id}:" in composite_key:
                workflow_state[composite_key] = entry.copy()
        
        if workflow_id not in self._checkpoints:
            self._checkpoints[workflow_id] = []
        
        self._checkpoints[workflow_id].append({
            "id": checkpoint_id,
            "created_at": datetime.utcnow().isoformat(),
            "state": workflow_state
        })
        
        # Keep only last 10 checkpoints
        if len(self._checkpoints[workflow_id]) > 10:
            self._checkpoints[workflow_id] = self._checkpoints[workflow_id][-10:]
        
        return checkpoint_id
    
    async def restore_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: str
    ) -> bool:
        """Restore from a checkpoint."""
        if workflow_id not in self._checkpoints:
            return False
        
        checkpoint = None
        for cp in self._checkpoints[workflow_id]:
            if cp["id"] == checkpoint_id:
                checkpoint = cp
                break
        
        if not checkpoint:
            return False
        
        # Restore state
        scope_data = self._state.get(StateScope.WORKFLOW.value, {})
        
        # Clear current workflow state
        keys_to_delete = [k for k in scope_data if f"workflow:{workflow_id}:" in k]
        for key in keys_to_delete:
            del scope_data[key]
        
        # Restore checkpoint state
        for composite_key, entry in checkpoint["state"].items():
            scope_data[composite_key] = entry
        
        return True
    
    async def get_state_stats(self) -> Dict[str, Any]:
        """Get state statistics."""
        stats = {
            "total_entries": 0,
            "by_scope": {}
        }
        
        for scope, data in self._state.items():
            count = len(data)
            stats["total_entries"] += count
            stats["by_scope"][scope] = count
        
        stats["checkpoints"] = sum(len(cps) for cps in self._checkpoints.values())
        
        return stats


# Create singleton instance
state_manager = SimpleStateManager()


class NodeStateManager:
    """Node-specific state management."""
    
    def __init__(self, node_instance: NodeInstance):
        self.node_instance = node_instance
        self.workflow_id = node_instance.workflow_instance_id
        self.node_id = node_instance.id
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get node state."""
        return await state_manager.get_state(
            key=key,
            scope=StateScope.NODE,
            scope_id=self.node_id,
            default=default
        )
    
    async def set(self, key: str, value: Any) -> None:
        """Set node state."""
        await state_manager.set_state(
            key=key,
            value=value,
            scope=StateScope.NODE,
            scope_id=self.node_id
        )
    
    async def get_workflow_state(self, key: str, default: Any = None) -> Any:
        """Get workflow state."""
        return await state_manager.get_state(
            key=key,
            scope=StateScope.WORKFLOW,
            scope_id=self.workflow_id,
            default=default
        )
    
    async def set_workflow_state(self, key: str, value: Any) -> None:
        """Set workflow state."""
        await state_manager.set_state(
            key=key,
            value=value,
            scope=StateScope.WORKFLOW,
            scope_id=self.workflow_id
        )
    
    async def get_global_state(self, key: str, default: Any = None) -> Any:
        """Get global state."""
        return await state_manager.get_state(
            key=key,
            scope=StateScope.GLOBAL,
            default=default
        )
    
    async def set_global_state(self, key: str, value: Any) -> None:
        """Set global state."""
        await state_manager.set_state(
            key=key,
            value=value,
            scope=StateScope.GLOBAL
        )
    
    async def clear_node_state(self) -> int:
        """Clear all state for this node."""
        return await state_manager.clear_state(
            scope=StateScope.NODE,
            scope_id=self.node_id
        )
    
    async def list_node_state(self) -> Dict[str, Any]:
        """List all state for this node."""
        return await state_manager.list_state(
            scope=StateScope.NODE,
            scope_id=self.node_id
        )