"""Basic State service for Community Edition.

Provides in-memory state management without database persistence.
Advanced features like distributed state and persistence are
available in Business Edition.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from pathlib import Path
import asyncio
from enum import Enum

from core.config import settings


logger = logging.getLogger(__name__)


class StateScope(Enum):
    """State scope levels."""
    GLOBAL = "global"
    TENANT = "tenant" 
    USER = "user"
    SESSION = "session"


class StateService:
    """Basic state service for Community Edition.
    
    Provides:
    - In-memory state management
    - Scoped state (global, user, session)
    - State snapshots
    
    Advanced features available in Business Edition:
    - Persistent state storage
    - Distributed state synchronization
    - State versioning and history
    - State replication
    - Advanced querying
    """
    
    def __init__(self):
        """Initialize the basic state service."""
        # In-memory state storage by scope
        self._state: Dict[StateScope, Dict[str, Dict[str, Any]]] = {
            StateScope.GLOBAL: {},
            StateScope.TENANT: {},
            StateScope.USER: {},
            StateScope.SESSION: {}
        }
        self._locks: Dict[str, asyncio.Lock] = {}
        
        # File-based snapshots
        self.snapshot_dir = Path(settings.DATA_PATH) / "state" / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for a key."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]
    
    def _make_key(self, scope: StateScope, scope_id: str, key: str) -> str:
        """Make a composite key for locking."""
        return f"{scope.value}:{scope_id}:{key}"
    
    async def set_state(
        self,
        key: str,
        value: Any,
        scope: StateScope = StateScope.GLOBAL,
        scope_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set state value."""
        lock_key = self._make_key(scope, scope_id, key)
        
        async with self._get_lock(lock_key):
            if scope_id not in self._state[scope]:
                self._state[scope][scope_id] = {}
            
            self._state[scope][scope_id][key] = {
                "value": value,
                "metadata": metadata or {},
                "updated_at": datetime.utcnow().isoformat(),
                "version": self._state[scope][scope_id].get(key, {}).get("version", 0) + 1
            }
            
            return True
    
    async def get_state(
        self,
        key: str,
        scope: StateScope = StateScope.GLOBAL,
        scope_id: str = "default",
        default: Any = None
    ) -> Any:
        """Get state value."""
        lock_key = self._make_key(scope, scope_id, key)
        
        async with self._get_lock(lock_key):
            if scope_id not in self._state[scope]:
                return default
            
            if key not in self._state[scope][scope_id]:
                return default
            
            return self._state[scope][scope_id][key]["value"]
    
    async def delete_state(
        self,
        key: str,
        scope: StateScope = StateScope.GLOBAL,
        scope_id: str = "default"
    ) -> bool:
        """Delete state value."""
        lock_key = self._make_key(scope, scope_id, key)
        
        async with self._get_lock(lock_key):
            if scope_id in self._state[scope] and key in self._state[scope][scope_id]:
                del self._state[scope][scope_id][key]
                return True
            return False
    
    async def list_keys(
        self,
        scope: StateScope = StateScope.GLOBAL,
        scope_id: str = "default",
        prefix: Optional[str] = None
    ) -> List[str]:
        """List keys in a scope."""
        if scope_id not in self._state[scope]:
            return []
        
        keys = list(self._state[scope][scope_id].keys())
        
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]
        
        return sorted(keys)
    
    async def get_all_state(
        self,
        scope: StateScope = StateScope.GLOBAL,
        scope_id: str = "default"
    ) -> Dict[str, Any]:
        """Get all state in a scope."""
        if scope_id not in self._state[scope]:
            return {}
        
        return {
            key: entry["value"]
            for key, entry in self._state[scope][scope_id].items()
        }
    
    async def clear_scope(
        self,
        scope: StateScope = StateScope.GLOBAL,
        scope_id: str = "default"
    ) -> int:
        """Clear all state in a scope."""
        if scope_id in self._state[scope]:
            count = len(self._state[scope][scope_id])
            self._state[scope][scope_id] = {}
            return count
        return 0
    
    async def merge_state(
        self,
        updates: Dict[str, Any],
        scope: StateScope = StateScope.GLOBAL,
        scope_id: str = "default"
    ) -> int:
        """Merge multiple state updates."""
        count = 0
        for key, value in updates.items():
            if await self.set_state(key, value, scope, scope_id):
                count += 1
        return count
    
    async def create_snapshot(
        self,
        name: str,
        scope: Optional[StateScope] = None,
        scope_id: Optional[str] = None
    ) -> str:
        """Create a state snapshot."""
        snapshot_id = f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        snapshot_file = self.snapshot_dir / f"{snapshot_id}.json"
        
        # Collect state to snapshot
        snapshot_data = {
            "id": snapshot_id,
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "state": {}
        }
        
        if scope and scope_id:
            # Snapshot specific scope
            if scope_id in self._state[scope]:
                snapshot_data["state"][scope.value] = {
                    scope_id: self._state[scope][scope_id].copy()
                }
        else:
            # Snapshot all state
            for s in StateScope:
                snapshot_data["state"][s.value] = self._state[s].copy()
        
        # Save to file
        snapshot_file.write_text(json.dumps(snapshot_data, indent=2))
        
        logger.info(f"Created state snapshot: {snapshot_id}")
        return snapshot_id
    
    async def restore_snapshot(
        self,
        snapshot_id: str,
        scope: Optional[StateScope] = None,
        scope_id: Optional[str] = None
    ) -> bool:
        """Restore from a state snapshot."""
        snapshot_file = self.snapshot_dir / f"{snapshot_id}.json"
        
        if not snapshot_file.exists():
            logger.warning(f"Snapshot not found: {snapshot_id}")
            return False
        
        try:
            snapshot_data = json.loads(snapshot_file.read_text())
            
            if scope and scope_id:
                # Restore specific scope
                scope_data = snapshot_data["state"].get(scope.value, {}).get(scope_id)
                if scope_data:
                    self._state[scope][scope_id] = scope_data
            else:
                # Restore all state
                for scope_name, scope_data in snapshot_data["state"].items():
                    scope_enum = StateScope(scope_name)
                    self._state[scope_enum] = scope_data
            
            logger.info(f"Restored state from snapshot: {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore snapshot {snapshot_id}: {e}")
            return False
    
    async def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        snapshots = []
        
        for snapshot_file in self.snapshot_dir.glob("*.json"):
            try:
                data = json.loads(snapshot_file.read_text())
                snapshots.append({
                    "id": data["id"],
                    "name": data["name"],
                    "created_at": data["created_at"],
                    "size": snapshot_file.stat().st_size
                })
            except Exception as e:
                logger.warning(f"Failed to read snapshot {snapshot_file}: {e}")
        
        return sorted(snapshots, key=lambda x: x["created_at"], reverse=True)
    
    async def get_state_stats(self) -> Dict[str, Any]:
        """Get state statistics."""
        stats = {
            "scopes": {}
        }
        
        for scope in StateScope:
            scope_stats = {
                "total_keys": 0,
                "scope_ids": len(self._state[scope])
            }
            
            for scope_id, state_data in self._state[scope].items():
                scope_stats["total_keys"] += len(state_data)
            
            stats["scopes"][scope.value] = scope_stats
        
        stats["total_keys"] = sum(s["total_keys"] for s in stats["scopes"].values())
        stats["snapshots"] = len(list(self.snapshot_dir.glob("*.json")))
        
        stats["features"] = {
            "in_memory": True,
            "persistent": False,
            "distributed": False,
            "versioning": "basic",
            "snapshots": True
        }
        
        stats["upgrade_available"] = True
        stats["upgrade_benefits"] = [
            "Persistent state storage",
            "Distributed state synchronization",
            "Full state versioning and history",
            "State replication",
            "Advanced state querying"
        ]
        
        return stats