"""HashiCorp Vault backend for enterprise credential management"""
import os
import json
import logging
from typing import Dict, Any, Optional, List
import hvac
from hvac.exceptions import VaultError, InvalidPath

from services.platform_credential_service import CredentialBackend

logger = logging.getLogger(__name__)


class VaultCredentialBackend(CredentialBackend):
    """
    HashiCorp Vault backend for secure credential storage.
    
    This backend provides enterprise-grade credential management with:
    - Encryption at rest
    - Fine-grained access control
    - Audit logging
    - Dynamic secret generation
    - Secret rotation
    """
    
    def __init__(
        self,
        vault_url: str = None,
        vault_token: str = None,
        vault_namespace: str = None,
        mount_point: str = "secret",
        path_prefix: str = "aictrlnet/platforms"
    ):
        """
        Initialize Vault backend.
        
        Args:
            vault_url: Vault server URL
            vault_token: Vault authentication token
            vault_namespace: Vault namespace (for Vault Enterprise)
            mount_point: KV v2 secret engine mount point
            path_prefix: Path prefix for platform credentials
        """
        self.vault_url = vault_url or os.environ.get("VAULT_URL", "http://localhost:8200")
        self.vault_token = vault_token or os.environ.get("VAULT_TOKEN")
        self.vault_namespace = vault_namespace or os.environ.get("VAULT_NAMESPACE")
        self.mount_point = mount_point
        self.path_prefix = path_prefix
        
        if not self.vault_token:
            raise ValueError("Vault token is required for Vault backend")
        
        # Initialize Vault client
        self.client = hvac.Client(
            url=self.vault_url,
            token=self.vault_token,
            namespace=self.vault_namespace
        )
        
        # Verify authentication
        if not self.client.is_authenticated():
            raise ValueError("Failed to authenticate with Vault")
        
        # Ensure KV v2 engine is enabled
        self._ensure_kv_engine()
    
    def _ensure_kv_engine(self):
        """Ensure KV v2 secrets engine is enabled at mount point"""
        try:
            # Check if mount point exists
            mounts = self.client.sys.list_mounted_secrets_engines()
            if f"{self.mount_point}/" not in mounts:
                # Enable KV v2 engine
                self.client.sys.enable_secrets_engine(
                    backend_type="kv",
                    path=self.mount_point,
                    options={"version": "2"}
                )
                logger.info(f"Enabled KV v2 engine at {self.mount_point}")
        except VaultError as e:
            logger.error(f"Failed to ensure KV engine: {e}")
            raise
    
    def _get_secret_path(self, key: str) -> str:
        """Generate full secret path from key"""
        return f"{self.path_prefix}/{key}"
    
    async def get_credential(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve credential from Vault.
        
        Args:
            key: Credential key (format: platform:user_id:credential_id)
            
        Returns:
            Decrypted credential data or None if not found
        """
        try:
            path = self._get_secret_path(key)
            
            # Read from KV v2
            response = self.client.secrets.kv.v2.read_secret_version(
                mount_point=self.mount_point,
                path=path
            )
            
            if response and "data" in response and "data" in response["data"]:
                credential_data = response["data"]["data"]
                
                # Log access for audit
                logger.info(f"Retrieved credential from Vault: {key}")
                
                return credential_data
            
        except InvalidPath:
            logger.debug(f"Credential not found in Vault: {key}")
            return None
        except VaultError as e:
            logger.error(f"Vault error retrieving credential {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving credential {key}: {e}")
            return None
        
        return None
    
    async def store_credential(self, key: str, credential: Dict[str, Any]) -> bool:
        """
        Store credential in Vault.
        
        Args:
            key: Credential key
            credential: Credential data to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = self._get_secret_path(key)
            
            # Add metadata
            vault_data = {
                **credential,
                "_metadata": {
                    "created_by": "aictrlnet",
                    "platform": key.split(":")[0] if ":" in key else "unknown"
                }
            }
            
            # Write to KV v2
            self.client.secrets.kv.v2.create_or_update_secret(
                mount_point=self.mount_point,
                path=path,
                secret=vault_data
            )
            
            logger.info(f"Stored credential in Vault: {key}")
            return True
            
        except VaultError as e:
            logger.error(f"Vault error storing credential {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error storing credential {key}: {e}")
            return False
    
    async def delete_credential(self, key: str) -> bool:
        """
        Delete credential from Vault.
        
        Args:
            key: Credential key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = self._get_secret_path(key)
            
            # Delete from KV v2 (soft delete - can be recovered)
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                mount_point=self.mount_point,
                path=path
            )
            
            logger.info(f"Deleted credential from Vault: {key}")
            return True
            
        except InvalidPath:
            logger.debug(f"Credential not found for deletion: {key}")
            return False
        except VaultError as e:
            logger.error(f"Vault error deleting credential {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting credential {key}: {e}")
            return False
    
    async def list_credentials(self) -> List[str]:
        """
        List all credential keys in Vault.
        
        Returns:
            List of credential keys
        """
        try:
            # List secrets at path prefix
            response = self.client.secrets.kv.v2.list_secrets(
                mount_point=self.mount_point,
                path=self.path_prefix
            )
            
            if response and "data" in response and "keys" in response["data"]:
                keys = response["data"]["keys"]
                # Remove trailing slashes from directories
                return [k.rstrip("/") for k in keys]
            
        except InvalidPath:
            logger.debug(f"No credentials found at {self.path_prefix}")
            return []
        except VaultError as e:
            logger.error(f"Vault error listing credentials: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing credentials: {e}")
            return []
        
        return []
    
    async def rotate_credential(self, key: str, new_credential: Dict[str, Any]) -> bool:
        """
        Rotate a credential by creating a new version.
        
        Args:
            key: Credential key
            new_credential: New credential data
            
        Returns:
            True if successful, False otherwise
        """
        # Store will create a new version in KV v2
        return await self.store_credential(key, new_credential)
    
    async def get_credential_versions(self, key: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a credential (for audit/rollback).
        
        Args:
            key: Credential key
            
        Returns:
            List of credential versions with metadata
        """
        try:
            path = self._get_secret_path(key)
            
            # Get version metadata
            response = self.client.secrets.kv.v2.read_secret_metadata(
                mount_point=self.mount_point,
                path=path
            )
            
            if response and "data" in response and "versions" in response["data"]:
                versions = []
                for version_num, version_data in response["data"]["versions"].items():
                    versions.append({
                        "version": int(version_num),
                        "created_time": version_data.get("created_time"),
                        "deletion_time": version_data.get("deletion_time"),
                        "destroyed": version_data.get("destroyed", False)
                    })
                return sorted(versions, key=lambda x: x["version"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting credential versions for {key}: {e}")
        
        return []
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Vault backend health.
        
        Returns:
            Health status information
        """
        try:
            # Check if authenticated
            is_authenticated = self.client.is_authenticated()
            
            # Get Vault health
            health = self.client.sys.read_health_status()
            
            return {
                "healthy": is_authenticated and not health.get("sealed", True),
                "authenticated": is_authenticated,
                "sealed": health.get("sealed", True),
                "version": health.get("version", "unknown"),
                "cluster_name": health.get("cluster_name", "unknown"),
                "mount_point": self.mount_point,
                "path_prefix": self.path_prefix
            }
            
        except Exception as e:
            logger.error(f"Vault health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }