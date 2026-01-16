"""Platform Credential Service for secure credential management"""
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from models.platform_integration import PlatformCredential, PlatformType, AuthMethod
from schemas.platform_integration import (
    PlatformCredentialCreate,
    PlatformCredentialUpdate,
    PlatformCredentialResponse
)

logger = logging.getLogger(__name__)


class CredentialBackend(ABC):
    """Abstract base class for credential storage backends"""
    
    @abstractmethod
    async def get_credential(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve credential by key"""
        pass
    
    @abstractmethod
    async def store_credential(self, key: str, credential: Dict[str, Any]) -> bool:
        """Store credential with key"""
        pass
    
    @abstractmethod
    async def delete_credential(self, key: str) -> bool:
        """Delete credential by key"""
        pass
    
    @abstractmethod
    async def list_credentials(self) -> List[str]:
        """List all credential keys"""
        pass


class EnvironmentBackend(CredentialBackend):
    """Environment variable backend for credentials"""
    
    async def get_credential(self, key: str) -> Optional[Dict[str, Any]]:
        """Get credential from environment variables"""
        value = os.environ.get(f"PLATFORM_CRED_{key.upper()}")
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in environment variable for {key}")
        return None
    
    async def store_credential(self, key: str, credential: Dict[str, Any]) -> bool:
        """Cannot store to environment variables at runtime"""
        logger.warning("Cannot store credentials to environment variables at runtime")
        return False
    
    async def delete_credential(self, key: str) -> bool:
        """Cannot delete environment variables at runtime"""
        logger.warning("Cannot delete environment variables at runtime")
        return False
    
    async def list_credentials(self) -> List[str]:
        """List credential keys from environment"""
        prefix = "PLATFORM_CRED_"
        keys = []
        for env_key in os.environ:
            if env_key.startswith(prefix):
                keys.append(env_key[len(prefix):].lower())
        return keys


class FileBackend(CredentialBackend):
    """File-based backend for credentials"""
    
    def __init__(self, file_path: str = "/app/data/credentials.json"):
        self.file_path = file_path
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure credentials file exists"""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump({}, f)
    
    async def get_credential(self, key: str) -> Optional[Dict[str, Any]]:
        """Get credential from file"""
        try:
            with open(self.file_path, 'r') as f:
                credentials = json.load(f)
                return credentials.get(key)
        except Exception as e:
            logger.error(f"Error reading credentials file: {e}")
            return None
    
    async def store_credential(self, key: str, credential: Dict[str, Any]) -> bool:
        """Store credential to file"""
        try:
            with open(self.file_path, 'r') as f:
                credentials = json.load(f)
            
            credentials[key] = credential
            
            with open(self.file_path, 'w') as f:
                json.dump(credentials, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error storing credential to file: {e}")
            return False
    
    async def delete_credential(self, key: str) -> bool:
        """Delete credential from file"""
        try:
            with open(self.file_path, 'r') as f:
                credentials = json.load(f)
            
            if key in credentials:
                del credentials[key]
                
                with open(self.file_path, 'w') as f:
                    json.dump(credentials, f, indent=2)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting credential from file: {e}")
            return False
    
    async def list_credentials(self) -> List[str]:
        """List credential keys from file"""
        try:
            with open(self.file_path, 'r') as f:
                credentials = json.load(f)
                return list(credentials.keys())
        except Exception as e:
            logger.error(f"Error listing credentials from file: {e}")
            return []


class DatabaseBackend(CredentialBackend):
    """Database backend for encrypted credentials"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._init_encryption()
    
    def _init_encryption(self):
        """Initialize encryption key"""
        # In production, this should come from a secure key management service
        encryption_key = os.environ.get("PLATFORM_CREDENTIAL_KEY")
        if not encryption_key:
            # Generate a key for development - DO NOT use in production
            logger.warning("No encryption key found, generating one for development")
            encryption_key = Fernet.generate_key().decode()
            os.environ["PLATFORM_CREDENTIAL_KEY"] = encryption_key
        
        self.cipher = Fernet(encryption_key.encode())
    
    def _encrypt_data(self, data: Dict[str, Any]) -> str:
        """Encrypt credential data"""
        json_data = json.dumps(data)
        return self.cipher.encrypt(json_data.encode()).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt credential data"""
        try:
            decrypted = self.cipher.decrypt(encrypted_data.encode())
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Error decrypting credential data: {e}")
            return {}
    
    async def get_credential(self, key: str) -> Optional[Dict[str, Any]]:
        """Get credential from database"""
        # Key format: platform:user_id:credential_id
        parts = key.split(":")
        if len(parts) != 3:
            return None
        
        platform, user_id, credential_id = parts
        
        result = await self.db.execute(
            select(PlatformCredential).where(
                PlatformCredential.id == int(credential_id),
                PlatformCredential.user_id == user_id,
                PlatformCredential.platform == platform
            )
        )
        credential = result.scalar_one_or_none()
        
        if credential:
            return self._decrypt_data(credential.encrypted_data)
        return None
    
    async def store_credential(self, key: str, credential: Dict[str, Any]) -> bool:
        """Store credential to database (handled by service layer)"""
        # This backend doesn't directly store - it's handled by the service
        logger.warning("Direct storage not supported - use PlatformCredentialService")
        return False
    
    async def delete_credential(self, key: str) -> bool:
        """Delete credential from database (handled by service layer)"""
        # This backend doesn't directly delete - it's handled by the service
        logger.warning("Direct deletion not supported - use PlatformCredentialService")
        return False
    
    async def list_credentials(self) -> List[str]:
        """List credential keys from database"""
        result = await self.db.execute(select(PlatformCredential))
        credentials = result.scalars().all()
        keys = []
        for cred in credentials:
            key = f"{cred.platform}:{cred.user_id}:{cred.id}"
            keys.append(key)
        return keys


class PlatformCredentialService:
    """Service for managing platform credentials"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.backend = self._select_backend()
    
    def _select_backend(self) -> CredentialBackend:
        """Select credential backend based on configuration"""
        backend_type = os.environ.get("CREDENTIAL_BACKEND", "database").lower()
        
        if backend_type == "environment":
            return EnvironmentBackend()
        elif backend_type == "file":
            file_path = os.environ.get("CREDENTIAL_FILE_PATH", "/app/data/credentials.json")
            return FileBackend(file_path)
        elif backend_type == "vault":
            # Import here to avoid circular dependency and optional dependency
            try:
                from services.vault_credential_backend import VaultCredentialBackend
                return VaultCredentialBackend()
            except ImportError:
                logger.error("Vault backend requested but hvac not installed")
                raise ValueError("Vault backend requires 'hvac' package to be installed")
            except Exception as e:
                logger.error(f"Failed to initialize Vault backend: {e}")
                raise
        else:  # database
            return DatabaseBackend(self.db)
    
    async def create_credential(
        self,
        user_id: str,
        credential_data: PlatformCredentialCreate
    ) -> PlatformCredentialResponse:
        """Create a new platform credential"""
        # Encrypt the credential data
        if isinstance(self.backend, DatabaseBackend):
            encrypted_data = self.backend._encrypt_data(credential_data.credentials)
        else:
            # For non-database backends, store as-is
            encrypted_data = json.dumps(credential_data.credentials)
        
        # Create database record
        db_credential = PlatformCredential(
            name=credential_data.name,
            platform=credential_data.platform.value,
            auth_method=credential_data.auth_method.value,
            encrypted_data=encrypted_data,
            user_id=user_id,
            is_shared=credential_data.is_shared,
            config_metadata=credential_data.config_metadata
        )
        
        self.db.add(db_credential)
        await self.db.commit()
        await self.db.refresh(db_credential)
        
        # Store in backend if not database
        if not isinstance(self.backend, DatabaseBackend):
            key = f"{db_credential.platform}:{user_id}:{db_credential.id}"
            await self.backend.store_credential(key, credential_data.credentials)
        
        return PlatformCredentialResponse.model_validate(db_credential)
    
    async def get_credential(
        self,
        credential_id: int,
        user_id: str
    ) -> Optional[PlatformCredentialResponse]:
        """Get a platform credential"""
        result = await self.db.execute(
            select(PlatformCredential).where(
                PlatformCredential.id == credential_id,
                PlatformCredential.user_id == user_id
            )
        )
        credential = result.scalar_one_or_none()
        
        if credential:
            return PlatformCredentialResponse.model_validate(credential)
        return None
    
    async def get_credential_data(
        self,
        credential_id: int,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get decrypted credential data"""
        result = await self.db.execute(
            select(PlatformCredential).where(
                PlatformCredential.id == credential_id,
                PlatformCredential.user_id == user_id
            )
        )
        credential = result.scalar_one_or_none()
        
        if not credential:
            return None
        
        # Update last used timestamp
        credential.last_used_at = datetime.utcnow()
        await self.db.commit()
        
        # Get credential data from backend
        key = f"{credential.platform}:{user_id}:{credential_id}"
        cred_data = await self.backend.get_credential(key)
        
        if cred_data:
            return cred_data
        
        # Fallback to database if backend doesn't have it
        if isinstance(self.backend, DatabaseBackend):
            return self.backend._decrypt_data(credential.encrypted_data)
        else:
            try:
                return json.loads(credential.encrypted_data)
            except:
                return {}
    
    async def update_credential(
        self,
        credential_id: int,
        user_id: str,
        update_data: PlatformCredentialUpdate
    ) -> Optional[PlatformCredentialResponse]:
        """Update a platform credential"""
        result = await self.db.execute(
            select(PlatformCredential).where(
                PlatformCredential.id == credential_id,
                PlatformCredential.user_id == user_id
            )
        )
        credential = result.scalar_one_or_none()
        
        if not credential:
            return None
        
        # Update fields
        if update_data.name is not None:
            credential.name = update_data.name
        if update_data.is_shared is not None:
            credential.is_shared = update_data.is_shared
        if update_data.config_metadata is not None:
            credential.config_metadata = update_data.config_metadata
        
        # Update credentials if provided
        if update_data.credentials is not None:
            if isinstance(self.backend, DatabaseBackend):
                credential.encrypted_data = self.backend._encrypt_data(update_data.credentials)
            else:
                credential.encrypted_data = json.dumps(update_data.credentials)
                # Update in backend
                key = f"{credential.platform}:{user_id}:{credential_id}"
                await self.backend.store_credential(key, update_data.credentials)
        
        credential.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(credential)
        
        return PlatformCredentialResponse.model_validate(credential)
    
    async def delete_credential(
        self,
        credential_id: int,
        user_id: str
    ) -> bool:
        """Delete a platform credential"""
        result = await self.db.execute(
            select(PlatformCredential).where(
                PlatformCredential.id == credential_id,
                PlatformCredential.user_id == user_id
            )
        )
        credential = result.scalar_one_or_none()
        
        if not credential:
            return False
        
        # Delete from backend if not database
        if not isinstance(self.backend, DatabaseBackend):
            key = f"{credential.platform}:{user_id}:{credential_id}"
            await self.backend.delete_credential(key)
        
        # Delete from database
        await self.db.delete(credential)
        await self.db.commit()
        
        return True
    
    async def list_credentials(
        self,
        user_id: str,
        platform: Optional[str] = None
    ) -> List[PlatformCredentialResponse]:
        """List platform credentials for a user"""
        stmt = select(PlatformCredential).where(
            PlatformCredential.user_id == user_id
        )
        
        if platform:
            stmt = stmt.where(PlatformCredential.platform == platform)
        
        result = await self.db.execute(stmt)
        credentials = result.scalars().all()
        return [PlatformCredentialResponse.model_validate(c) for c in credentials]
    
    async def record_error(
        self,
        credential_id: int,
        error_message: str
    ):
        """Record an error for a credential"""
        result = await self.db.execute(
            select(PlatformCredential).where(
                PlatformCredential.id == credential_id
            )
        )
        credential = result.scalar_one_or_none()
        
        if credential:
            credential.last_error = error_message
            credential.updated_at = datetime.utcnow()
            await self.db.commit()