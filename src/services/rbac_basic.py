"""Basic RBAC service for Community Edition.

This provides simple role-based access control without advanced features
like groups, hierarchical permissions, or database persistence.
"""

from typing import List, Optional, Dict, Any, Set
from datetime import datetime
import json
from pathlib import Path

from core.config import settings


class BasicRBACService:
    """Basic RBAC service for Community Edition.
    
    Uses file-based configuration for roles and permissions.
    Advanced features like groups, dynamic permissions, and 
    database persistence are available in Business Edition.
    """
    
    # Default roles for Community Edition
    DEFAULT_ROLES = {
        "admin": {
            "description": "Full system access",
            "permissions": ["*"]  # All permissions
        },
        "user": {
            "description": "Standard user access",
            "permissions": [
                "tasks:read",
                "tasks:create",
                "tasks:update",
                "workflows:read",
                "workflows:create",
                "workflows:execute",
                "templates:read",
                "adapters:read"
            ]
        },
        "viewer": {
            "description": "Read-only access",
            "permissions": [
                "tasks:read",
                "workflows:read",
                "templates:read",
                "adapters:read"
            ]
        }
    }
    
    def __init__(self):
        """Initialize the basic RBAC service."""
        self.roles_file = Path(settings.DATA_PATH) / "rbac" / "roles.json"
        self.user_roles_file = Path(settings.DATA_PATH) / "rbac" / "user_roles.json"
        self._ensure_files()
        self._load_data()
    
    def _ensure_files(self):
        """Ensure RBAC files exist."""
        self.roles_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.roles_file.exists():
            self.roles_file.write_text(json.dumps(self.DEFAULT_ROLES, indent=2))
        
        if not self.user_roles_file.exists():
            # Default: first user is admin
            default_user_roles = {
                "default-user": ["admin"]
            }
            self.user_roles_file.write_text(json.dumps(default_user_roles, indent=2))
    
    def _load_data(self):
        """Load roles and user assignments from files."""
        self.roles = json.loads(self.roles_file.read_text())
        self.user_roles = json.loads(self.user_roles_file.read_text())
    
    def _save_data(self):
        """Save roles and user assignments to files."""
        self.roles_file.write_text(json.dumps(self.roles, indent=2))
        self.user_roles_file.write_text(json.dumps(self.user_roles, indent=2))
    
    async def list_roles(self) -> List[Dict[str, Any]]:
        """List all available roles."""
        return [
            {
                "name": name,
                "description": role.get("description", ""),
                "permissions": role.get("permissions", []),
                "is_system": name in self.DEFAULT_ROLES
            }
            for name, role in self.roles.items()
        ]
    
    async def get_role(self, role_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific role."""
        if role_name in self.roles:
            return {
                "name": role_name,
                "description": self.roles[role_name].get("description", ""),
                "permissions": self.roles[role_name].get("permissions", []),
                "is_system": role_name in self.DEFAULT_ROLES
            }
        return None
    
    async def create_role(
        self,
        name: str,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a custom role (Community Edition limitation: file-based)."""
        if name in self.roles:
            raise ValueError(f"Role '{name}' already exists")
        
        self.roles[name] = {
            "description": description or "",
            "permissions": permissions or []
        }
        self._save_data()
        
        return {
            "name": name,
            "description": description or "",
            "permissions": permissions or [],
            "is_system": False
        }
    
    async def delete_role(self, role_name: str) -> bool:
        """Delete a custom role."""
        if role_name in self.DEFAULT_ROLES:
            raise ValueError("Cannot delete system roles")
        
        if role_name in self.roles:
            del self.roles[role_name]
            
            # Remove role from all users
            for user_id in self.user_roles:
                if role_name in self.user_roles[user_id]:
                    self.user_roles[user_id].remove(role_name)
            
            self._save_data()
            return True
        return False
    
    async def get_user_roles(self, user_id: str) -> List[str]:
        """Get roles assigned to a user."""
        return self.user_roles.get(user_id, ["user"])  # Default to user role
    
    async def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' does not exist")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        
        if role_name not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role_name)
            self._save_data()
            return True
        return False
    
    async def remove_role(self, user_id: str, role_name: str) -> bool:
        """Remove a role from a user."""
        if user_id in self.user_roles and role_name in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role_name)
            if not self.user_roles[user_id]:
                del self.user_roles[user_id]
            self._save_data()
            return True
        return False
    
    async def check_permission(
        self,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """Check if a user has permission for a resource:action."""
        user_roles = await self.get_user_roles(user_id)
        required_permission = f"{resource}:{action}"
        
        for role_name in user_roles:
            if role_name not in self.roles:
                continue
            
            permissions = self.roles[role_name].get("permissions", [])
            
            # Check for wildcard permission
            if "*" in permissions:
                return True
            
            # Check for specific permission
            if required_permission in permissions:
                return True
            
            # Check for resource wildcard
            if f"{resource}:*" in permissions:
                return True
        
        return False
    
    async def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user."""
        user_roles = await self.get_user_roles(user_id)
        permissions = set()
        
        for role_name in user_roles:
            if role_name not in self.roles:
                continue
            
            role_permissions = self.roles[role_name].get("permissions", [])
            
            # Handle wildcard
            if "*" in role_permissions:
                # Return a comprehensive set of permissions
                return {"*"}
            
            permissions.update(role_permissions)
        
        return permissions
    
    # Simplified interface methods that match Business Edition
    
    async def list_users(self) -> List[Dict[str, Any]]:
        """List users with roles (simplified for Community)."""
        # In Community Edition, we just return user IDs with their roles
        return [
            {
                "id": user_id,
                "username": user_id,
                "roles": roles,
                "is_active": True
            }
            for user_id, roles in self.user_roles.items()
        ]
    
    async def list_permissions(self) -> List[Dict[str, Any]]:
        """List all defined permissions."""
        # Extract unique permissions from all roles
        all_permissions = set()
        for role_data in self.roles.values():
            all_permissions.update(role_data.get("permissions", []))
        
        # Parse permissions into resource:action format
        permissions = []
        for perm in all_permissions:
            if perm == "*":
                permissions.append({
                    "resource": "*",
                    "action": "*",
                    "description": "All permissions"
                })
            elif ":" in perm:
                resource, action = perm.split(":", 1)
                permissions.append({
                    "resource": resource,
                    "action": action,
                    "description": f"Permission to {action} {resource}"
                })
        
        return sorted(permissions, key=lambda x: (x["resource"], x["action"]))