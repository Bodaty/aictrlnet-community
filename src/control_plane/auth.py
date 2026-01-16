"""Component authentication for the control plane."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.config import get_settings
from .models import Component, ComponentStatus


class ComponentBearer(HTTPBearer):
    """Custom Bearer authentication for components."""
    
    def __init__(self, auto_error: bool = True):
        super(ComponentBearer, self).__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        credentials: HTTPAuthorizationCredentials = await super(ComponentBearer, self).__call__(request)
        if credentials:
            if not credentials.credentials:
                if self.auto_error:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid authentication credentials"
                    )
                else:
                    return None
            return credentials
        return None


component_bearer = ComponentBearer()


class ComponentAuth:
    """Handle component JWT authentication."""
    
    def __init__(self):
        self.settings = get_settings()
        self.algorithm = "HS256"
        self.token_expire_hours = 24
    
    def create_component_token(self, component: Component) -> tuple[str, datetime]:
        """Create a JWT token for a component."""
        expire = datetime.utcnow() + timedelta(hours=self.token_expire_hours)
        
        import uuid
        payload = {
            "sub": component.id,
            "component_name": component.name,
            "component_type": component.type.value,
            "edition": component.edition,
            "capabilities": [cap.name for cap in component.capabilities],
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),  # JWT ID for uniqueness
            "type": "component"
        }
        
        token = jwt.encode(payload, self.settings.SECRET_KEY, algorithm=self.algorithm)
        return token, expire
    
    def verify_component_token(self, token: str) -> Dict[str, Any]:
        """Verify a component token and return the payload."""
        try:
            payload = jwt.decode(
                token, 
                self.settings.SECRET_KEY, 
                algorithms=[self.algorithm]
            )
            
            # Check if it's a component token
            if payload.get("type") != "component":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid token type"
                )
            
            return payload
            
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid token: {str(e)}"
            )
    
    async def get_current_component(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(component_bearer)
    ) -> Dict[str, Any]:
        """Get the current component from the JWT token."""
        token = credentials.credentials
        payload = self.verify_component_token(token)
        
        # Return component info from token
        return {
            "id": payload["sub"],
            "name": payload.get("component_name"),
            "type": payload.get("component_type"),
            "edition": payload.get("edition"),
            "capabilities": payload.get("capabilities", [])
        }
    
    def refresh_component_token(self, old_token: str) -> tuple[str, datetime]:
        """Refresh a component token if it's still valid."""
        try:
            # Decode without verification to get payload
            unverified_payload = jwt.decode(
                old_token, 
                options={"verify_signature": False}
            )
            
            # Verify the old token
            payload = self.verify_component_token(old_token)
            
            # Create new token with same data
            expire = datetime.utcnow() + timedelta(hours=self.token_expire_hours)
            
            new_payload = {
                "sub": payload["sub"],
                "component_name": payload.get("component_name"),
                "component_type": payload.get("component_type"),
                "edition": payload.get("edition"),
                "capabilities": payload.get("capabilities", []),
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "component"
            }
            
            new_token = jwt.encode(
                new_payload, 
                self.settings.SECRET_KEY, 
                algorithm=self.algorithm
            )
            
            return new_token, expire
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Failed to refresh token: {str(e)}"
            )
    
    def validate_component_access(
        self, 
        component_info: Dict[str, Any], 
        required_edition: str = "community",
        required_capabilities: list[str] = None
    ) -> bool:
        """Validate if a component has access to a resource."""
        # Check edition
        edition_hierarchy = ["community", "business", "enterprise"]
        component_edition_idx = edition_hierarchy.index(component_info.get("edition", "community"))
        required_edition_idx = edition_hierarchy.index(required_edition)
        
        if component_edition_idx < required_edition_idx:
            return False
        
        # Check capabilities
        if required_capabilities:
            component_caps = set(component_info.get("capabilities", []))
            if not all(cap in component_caps for cap in required_capabilities):
                return False
        
        return True


# Global instance
component_auth = ComponentAuth()