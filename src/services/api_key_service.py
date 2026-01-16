"""Service for managing API keys with secure generation and validation."""

import secrets
import hashlib
import string
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging

from models import APIKey, APIKeyLog, User
from schemas import (
    APIKeyCreate, APIKeyResponse, APIKeyCreateResponse,
    APIKeyUpdate, APIKeyListResponse
)

logger = logging.getLogger(__name__)


class APIKeyService:
    """Service for managing API keys."""
    
    # API key configuration
    KEY_PREFIX = "aictrl"
    KEY_LENGTH = 32  # Length of random part
    SALT_LENGTH = 32
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_api_key(
        self,
        user_id: str,
        data: APIKeyCreate
    ) -> APIKeyCreateResponse:
        """Create a new API key for a user."""
        # Generate the API key components
        environment = "live"  # Could be "test" for test keys
        random_part = self._generate_random_string(self.KEY_LENGTH)
        full_key = f"{self.KEY_PREFIX}_{environment}_{random_part}"
        
        # Extract prefix and suffix for storage
        key_prefix = full_key[:14]  # "aictrl_live_ab"
        key_suffix = full_key[-4:]   # Last 4 chars
        
        # Generate salt and hash the key
        salt = secrets.token_bytes(self.SALT_LENGTH)
        key_hash = self._hash_key(full_key, salt)
        
        # Calculate expiration if specified
        expires_at = None
        if data.expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=data.expires_in_days)
        
        # Create the API key record
        api_key = APIKey(
            user_id=user_id,
            name=data.name,
            description=data.description,
            key_prefix=key_prefix,
            key_suffix=key_suffix,
            key_hash=key_hash,
            key_salt=salt.hex(),  # Store as hex string
            scopes=data.scopes,
            allowed_ips=data.allowed_ips,
            expires_at=expires_at
        )
        
        self.db.add(api_key)
        await self.db.commit()
        await self.db.refresh(api_key)
        
        # Return response with full key (only time it's shown)
        response_data = api_key.to_dict()
        response_data['api_key'] = full_key
        
        return APIKeyCreateResponse(**response_data)
    
    async def list_user_api_keys(
        self,
        user_id: str,
        include_inactive: bool = False
    ) -> APIKeyListResponse:
        """List all API keys for a user."""
        query = select(APIKey).where(APIKey.user_id == user_id)
        
        if not include_inactive:
            query = query.where(APIKey.is_active == True)
        
        query = query.order_by(APIKey.created_at.desc())
        
        result = await self.db.execute(query)
        api_keys = result.scalars().all()
        
        return APIKeyListResponse(
            keys=[APIKeyResponse(**key.to_dict()) for key in api_keys],
            total=len(api_keys)
        )
    
    async def get_api_key(
        self,
        user_id: str,
        key_id: str
    ) -> Optional[APIKeyResponse]:
        """Get a specific API key."""
        result = await self.db.execute(
            select(APIKey).where(
                and_(
                    APIKey.id == key_id,
                    APIKey.user_id == user_id
                )
            )
        )
        api_key = result.scalar_one_or_none()
        
        if api_key:
            return APIKeyResponse(**api_key.to_dict())
        return None
    
    async def update_api_key(
        self,
        user_id: str,
        key_id: str,
        data: APIKeyUpdate
    ) -> Optional[APIKeyResponse]:
        """Update an API key."""
        result = await self.db.execute(
            select(APIKey).where(
                and_(
                    APIKey.id == key_id,
                    APIKey.user_id == user_id
                )
            )
        )
        api_key = result.scalar_one_or_none()
        
        if not api_key:
            return None
        
        # Update fields
        for field, value in data.dict(exclude_unset=True).items():
            setattr(api_key, field, value)
        
        api_key.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(api_key)
        
        return APIKeyResponse(**api_key.to_dict())
    
    async def revoke_api_key(
        self,
        user_id: str,
        key_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """Revoke an API key."""
        result = await self.db.execute(
            select(APIKey).where(
                and_(
                    APIKey.id == key_id,
                    APIKey.user_id == user_id
                )
            )
        )
        api_key = result.scalar_one_or_none()
        
        if not api_key:
            return False
        
        api_key.is_active = False
        api_key.revoked_at = datetime.utcnow()
        api_key.revoked_reason = reason
        
        await self.db.commit()
        return True
    
    async def verify_api_key(
        self,
        provided_key: str,
        required_scopes: Optional[List[str]] = None,
        ip_address: Optional[str] = None
    ) -> Tuple[Optional[User], Optional[APIKey]]:
        """
        Verify an API key and return the associated user.
        
        Returns:
            Tuple of (User, APIKey) if valid, (None, None) if invalid
        """
        # Extract prefix from provided key
        if not provided_key.startswith(f"{self.KEY_PREFIX}_"):
            return None, None
        
        try:
            parts = provided_key.split('_')
            if len(parts) < 3:
                return None, None
            
            key_prefix = f"{parts[0]}_{parts[1]}_{parts[2][:2]}"
        except:
            return None, None
        
        # Find potential matches by prefix
        result = await self.db.execute(
            select(APIKey).where(
                and_(
                    APIKey.key_prefix == key_prefix,
                    APIKey.is_active == True
                )
            )
        )
        potential_keys = result.scalars().all()
        
        # Check each potential key
        for api_key in potential_keys:
            # Check expiration
            if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                continue
            
            # Verify the key hash
            salt = bytes.fromhex(api_key.key_salt)
            if self._verify_key_hash(provided_key, api_key.key_hash, salt):
                # Check IP whitelist if configured
                if api_key.allowed_ips and ip_address:
                    if not self._check_ip_allowed(ip_address, api_key.allowed_ips):
                        await self._log_access(api_key.id, False, ip_address, "IP not allowed")
                        continue
                
                # Check required scopes
                if required_scopes:
                    if not all(scope in api_key.scopes for scope in required_scopes):
                        await self._log_access(api_key.id, False, ip_address, "Insufficient scopes")
                        continue
                
                # Get the user
                user_result = await self.db.execute(
                    select(User).where(User.id == api_key.user_id)
                )
                user = user_result.scalar_one_or_none()
                
                if user and user.is_active:
                    # Update usage tracking
                    api_key.last_used_at = datetime.utcnow()
                    api_key.last_used_ip = ip_address
                    api_key.usage_count += 1
                    
                    await self.db.commit()
                    await self._log_access(api_key.id, True, ip_address)
                    
                    return user, api_key
        
        return None, None
    
    def _generate_random_string(self, length: int) -> str:
        """Generate a cryptographically secure random string."""
        alphabet = string.ascii_lowercase + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def _hash_key(self, key: str, salt: bytes) -> str:
        """Hash an API key with salt using PBKDF2."""
        key_hash = hashlib.pbkdf2_hmac(
            'sha256',
            key.encode('utf-8'),
            salt,
            100000  # iterations
        )
        return key_hash.hex()
    
    def _verify_key_hash(self, provided_key: str, stored_hash: str, salt: bytes) -> bool:
        """Verify a provided key against stored hash."""
        provided_hash = self._hash_key(provided_key, salt)
        return secrets.compare_digest(provided_hash, stored_hash)
    
    def _check_ip_allowed(self, ip_address: str, allowed_ips: List[str]) -> bool:
        """Check if IP address is in the allowed list."""
        import ipaddress
        
        try:
            request_ip = ipaddress.ip_address(ip_address)
            
            for allowed in allowed_ips:
                try:
                    if '/' in allowed:
                        # It's a network
                        network = ipaddress.ip_network(allowed)
                        if request_ip in network:
                            return True
                    else:
                        # It's a single IP
                        if request_ip == ipaddress.ip_address(allowed):
                            return True
                except:
                    continue
            
            return False
        except:
            return False
    
    async def _log_access(
        self,
        api_key_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Log API key access attempt."""
        # This is a simplified version - you might want more details
        log = APIKeyLog(
            api_key_id=api_key_id,
            ip_address=ip_address,
            status_code=200 if success else 401,
            error_message=error_message
        )
        
        self.db.add(log)
        # Don't wait for commit to not slow down the request
        try:
            await self.db.commit()
        except:
            # Log but don't fail the request
            logger.error("Failed to log API key access", exc_info=True)