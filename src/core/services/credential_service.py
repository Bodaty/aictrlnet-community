"""
Configurable credential management for platform integrations.

This service supports multiple backend configurations for storing credentials,
from simple environment variables to encrypted database storage and external vaults.
"""

import os
import json
import base64
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from sqlalchemy.orm import Session
from abc import ABC, abstractmethod

from core.database import get_db
from models.platform_integration import PlatformCredential
from core.exceptions import CredentialNotFoundError, CredentialDecryptionError


class CredentialBackend(ABC):
    """Abstract base for credential storage backends"""
    
    @abstractmethod
    async def get_credentials(self, platform: str, credential_id: str) -> Dict[str, Any]:
        """Retrieve credentials for a platform"""
        pass
    
    @abstractmethod
    async def store_credentials(self, platform: str, credential_id: str, credentials: Dict[str, Any]) -> bool:
        """Store credentials for a platform"""
        pass
    
    @abstractmethod
    async def delete_credentials(self, platform: str, credential_id: str) -> bool:
        """Delete credentials for a platform"""
        pass


class EnvironmentCredentialBackend(CredentialBackend):
    """Environment variable based credential storage (default for self-hosted)"""
    
    async def get_credentials(self, platform: str, credential_id: str) -> Dict[str, Any]:
        """Get credentials from environment variables"""
        platform_upper = platform.upper()
        
        # Try different patterns
        api_key = os.environ.get(f'{platform_upper}_API_KEY')
        api_secret = os.environ.get(f'{platform_upper}_API_SECRET')
        oauth_token = os.environ.get(f'{platform_upper}_OAUTH_TOKEN')
        oauth_refresh = os.environ.get(f'{platform_upper}_OAUTH_REFRESH_TOKEN')
        webhook_url = os.environ.get(f'{platform_upper}_WEBHOOK_URL')
        
        credentials = {}
        if api_key:
            credentials['api_key'] = api_key
        if api_secret:
            credentials['api_secret'] = api_secret
        if oauth_token:
            credentials['oauth_token'] = oauth_token
        if oauth_refresh:
            credentials['oauth_refresh_token'] = oauth_refresh
        if webhook_url:
            credentials['webhook_url'] = webhook_url
            
        if not credentials:
            raise CredentialNotFoundError(f"No credentials found for platform {platform} in environment")
            
        return credentials
    
    async def store_credentials(self, platform: str, credential_id: str, credentials: Dict[str, Any]) -> bool:
        """Environment backend doesn't support dynamic storage"""
        raise NotImplementedError("Environment backend is read-only. Set environment variables directly.")
    
    async def delete_credentials(self, platform: str, credential_id: str) -> bool:
        """Environment backend doesn't support deletion"""
        raise NotImplementedError("Environment backend is read-only. Unset environment variables directly.")


class FileCredentialBackend(CredentialBackend):
    """Local encrypted file storage"""
    
    def __init__(self, file_path: str, encryption_key: Optional[str] = None):
        self.file_path = file_path
        self.fernet = None
        
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        else:
            # Generate a key if none provided (for development)
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
            print(f"Generated encryption key: {key.decode()}")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Initialize file if it doesn't exist
        if not os.path.exists(file_path):
            self._save_data({})
    
    def _load_data(self) -> Dict[str, Any]:
        """Load and decrypt credential data"""
        try:
            with open(self.file_path, 'rb') as f:
                encrypted_data = f.read()
                if encrypted_data:
                    decrypted = self.fernet.decrypt(encrypted_data)
                    return json.loads(decrypted)
        except Exception as e:
            raise CredentialDecryptionError(f"Failed to load credentials: {str(e)}")
        return {}
    
    def _save_data(self, data: Dict[str, Any]):
        """Encrypt and save credential data"""
        try:
            encrypted = self.fernet.encrypt(json.dumps(data).encode())
            with open(self.file_path, 'wb') as f:
                f.write(encrypted)
        except Exception as e:
            raise CredentialDecryptionError(f"Failed to save credentials: {str(e)}")
    
    async def get_credentials(self, platform: str, credential_id: str) -> Dict[str, Any]:
        """Get credentials from encrypted file"""
        data = self._load_data()
        key = f"{platform}:{credential_id}"
        
        if key not in data:
            raise CredentialNotFoundError(f"Credentials not found for {platform}:{credential_id}")
            
        return data[key]
    
    async def store_credentials(self, platform: str, credential_id: str, credentials: Dict[str, Any]) -> bool:
        """Store credentials in encrypted file"""
        data = self._load_data()
        key = f"{platform}:{credential_id}"
        data[key] = credentials
        self._save_data(data)
        return True
    
    async def delete_credentials(self, platform: str, credential_id: str) -> bool:
        """Delete credentials from file"""
        data = self._load_data()
        key = f"{platform}:{credential_id}"
        
        if key in data:
            del data[key]
            self._save_data(data)
            return True
        return False


class DatabaseCredentialBackend(CredentialBackend):
    """Database storage with encryption (for cloud deployments)"""
    
    def __init__(self, encryption_key: str, db_session_factory=None):
        self.fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        self.db_session_factory = db_session_factory or get_db
    
    async def get_credentials(self, platform: str, credential_id: str) -> Dict[str, Any]:
        """Get credentials from database"""
        db = next(self.db_session_factory())
        try:
            credential = db.query(PlatformCredential).filter(
                PlatformCredential.platform == platform,
                PlatformCredential.credential_id == credential_id,
                PlatformCredential.is_active == True
            ).first()
            
            if not credential:
                raise CredentialNotFoundError(f"Credentials not found for {platform}:{credential_id}")
            
            # Decrypt the credentials
            decrypted = self.fernet.decrypt(credential.encrypted_credentials.encode())
            return json.loads(decrypted)
            
        finally:
            db.close()
    
    async def store_credentials(self, platform: str, credential_id: str, credentials: Dict[str, Any]) -> bool:
        """Store credentials in database"""
        db = next(self.db_session_factory())
        try:
            # Encrypt credentials
            encrypted = self.fernet.encrypt(json.dumps(credentials).encode()).decode()
            
            # Check if credential exists
            existing = db.query(PlatformCredential).filter(
                PlatformCredential.platform == platform,
                PlatformCredential.credential_id == credential_id
            ).first()
            
            if existing:
                existing.encrypted_credentials = encrypted
                existing.is_active = True
            else:
                credential = PlatformCredential(
                    platform=platform,
                    credential_id=credential_id,
                    encrypted_credentials=encrypted,
                    is_active=True
                )
                db.add(credential)
            
            db.commit()
            return True
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    async def delete_credentials(self, platform: str, credential_id: str) -> bool:
        """Soft delete credentials from database"""
        db = next(self.db_session_factory())
        try:
            credential = db.query(PlatformCredential).filter(
                PlatformCredential.platform == platform,
                PlatformCredential.credential_id == credential_id
            ).first()
            
            if credential:
                credential.is_active = False
                db.commit()
                return True
            return False
            
        finally:
            db.close()


class VaultCredentialBackend(CredentialBackend):
    """External vault integration (e.g., HashiCorp Vault) for enterprise"""
    
    def __init__(self, vault_url: str, vault_token: str, mount_point: str = "secret"):
        self.vault_url = vault_url
        self.vault_token = vault_token
        self.mount_point = mount_point
        
        # In production, use hvac library for HashiCorp Vault
        # import hvac
        # self.client = hvac.Client(url=vault_url, token=vault_token)
    
    async def get_credentials(self, platform: str, credential_id: str) -> Dict[str, Any]:
        """Get credentials from external vault"""
        # Simplified implementation - in production use proper vault client
        path = f"{self.mount_point}/data/platforms/{platform}/{credential_id}"
        
        # Mock implementation for now
        raise NotImplementedError("Vault backend requires hvac library and vault setup")
    
    async def store_credentials(self, platform: str, credential_id: str, credentials: Dict[str, Any]) -> bool:
        """Store credentials in vault"""
        raise NotImplementedError("Vault backend requires hvac library and vault setup")
    
    async def delete_credentials(self, platform: str, credential_id: str) -> bool:
        """Delete credentials from vault"""
        raise NotImplementedError("Vault backend requires hvac library and vault setup")


class CredentialService:
    """
    Configurable credential management for platform integrations.
    
    Supports multiple backends:
    - environment: Environment variables (default for self-hosted)
    - file: Local encrypted file storage
    - database: Encrypted database storage (for cloud)
    - vault: External vault (e.g., HashiCorp Vault) for enterprise
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend_type = config.get('credential_backend', 'environment')
        self.backend = self._initialize_backend()
    
    def _initialize_backend(self) -> CredentialBackend:
        """Initialize the appropriate backend based on configuration"""
        if self.backend_type == 'environment':
            return EnvironmentCredentialBackend()
            
        elif self.backend_type == 'file':
            file_path = self.config.get('credential_file_path', 
                                       os.path.expanduser('~/.aictrlnet/credentials.enc'))
            encryption_key = self.config.get('credential_encryption_key')
            return FileCredentialBackend(file_path, encryption_key)
            
        elif self.backend_type == 'database':
            encryption_key = self.config.get('credential_encryption_key')
            if not encryption_key:
                raise ValueError("credential_encryption_key required for database backend")
            return DatabaseCredentialBackend(encryption_key)
            
        elif self.backend_type == 'vault':
            vault_url = self.config.get('vault_url')
            vault_token = self.config.get('vault_token')
            if not vault_url or not vault_token:
                raise ValueError("vault_url and vault_token required for vault backend")
            return VaultCredentialBackend(vault_url, vault_token)
            
        else:
            raise ValueError(f"Unknown credential backend: {self.backend_type}")
    
    async def get_credentials(self, platform: str, credential_id: str) -> Dict[str, Any]:
        """
        Retrieve credentials for a platform.
        
        Args:
            platform: Platform name (e.g., 'n8n', 'zapier')
            credential_id: Credential identifier
            
        Returns:
            Dictionary containing credential data
        """
        return await self.backend.get_credentials(platform, credential_id)
    
    async def store_credentials(self, platform: str, credential_id: str, credentials: Dict[str, Any]) -> bool:
        """
        Store credentials for a platform (not available for environment backend).
        
        Args:
            platform: Platform name
            credential_id: Credential identifier
            credentials: Credential data to store
            
        Returns:
            True if successful
        """
        return await self.backend.store_credentials(platform, credential_id, credentials)
    
    async def delete_credentials(self, platform: str, credential_id: str) -> bool:
        """
        Delete credentials for a platform (not available for environment backend).
        
        Args:
            platform: Platform name
            credential_id: Credential identifier
            
        Returns:
            True if successful
        """
        return await self.backend.delete_credentials(platform, credential_id)
    
    async def validate_credentials(self, platform: str, credential_id: str) -> bool:
        """
        Validate that credentials exist and are accessible.
        
        Args:
            platform: Platform name
            credential_id: Credential identifier
            
        Returns:
            True if credentials exist and are accessible
        """
        try:
            await self.get_credentials(platform, credential_id)
            return True
        except CredentialNotFoundError:
            return False
    
    def get_backend_type(self) -> str:
        """Get the current backend type"""
        return self.backend_type
    
    def supports_dynamic_storage(self) -> bool:
        """Check if the backend supports dynamic credential storage"""
        return self.backend_type != 'environment'


# Global instance configured from environment/settings
_credential_service = None

def get_credential_service() -> CredentialService:
    """Get the configured credential service instance"""
    global _credential_service
    
    if _credential_service is None:
        # Load configuration from environment
        config = {
            'credential_backend': os.getenv('CREDENTIAL_BACKEND', 'environment'),
            'credential_encryption_key': os.getenv('CREDENTIAL_ENCRYPTION_KEY'),
            'credential_file_path': os.getenv('CREDENTIAL_FILE_PATH'),
            'vault_url': os.getenv('VAULT_URL'),
            'vault_token': os.getenv('VAULT_TOKEN'),
        }
        
        _credential_service = CredentialService(config)
    
    return _credential_service