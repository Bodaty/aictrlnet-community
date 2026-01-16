"""Test credential backends"""
import os
import json
import tempfile
import pytest
from unittest.mock import Mock, patch
from cryptography.fernet import Fernet

from services.platform_credential_service import (
    EnvironmentBackend,
    FileBackend,
    DatabaseBackend,
    PlatformCredentialService
)


class TestEnvironmentBackend:
    """Test environment variable credential backend"""
    
    @pytest.mark.asyncio
    async def test_get_credential_success(self):
        """Test successful credential retrieval from environment"""
        backend = EnvironmentBackend()
        
        # Mock environment variable
        test_creds = {"api_key": "test-key", "base_url": "https://test.com"}
        with patch.dict(os.environ, {"PLATFORM_CRED_TEST": json.dumps(test_creds)}):
            result = await backend.get_credential("test")
            assert result == test_creds
    
    @pytest.mark.asyncio
    async def test_get_credential_not_found(self):
        """Test credential not found in environment"""
        backend = EnvironmentBackend()
        
        with patch.dict(os.environ, {}, clear=True):
            result = await backend.get_credential("nonexistent")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_store_credential_not_supported(self):
        """Test that store is not supported for environment backend"""
        backend = EnvironmentBackend()
        
        result = await backend.store_credential("test", {"key": "value"})
        assert result is False
    
    @pytest.mark.asyncio
    async def test_list_credentials(self):
        """Test listing credentials from environment"""
        backend = EnvironmentBackend()
        
        with patch.dict(os.environ, {
            "PLATFORM_CRED_N8N": "{}",
            "PLATFORM_CRED_ZAPIER": "{}",
            "OTHER_VAR": "ignored"
        }):
            keys = await backend.list_credentials()
            assert sorted(keys) == ["n8n", "zapier"]


class TestFileBackend:
    """Test file-based credential backend"""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("{}")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_credential(self, temp_file):
        """Test storing and retrieving credentials from file"""
        backend = FileBackend(temp_file)
        
        test_creds = {"api_key": "test-key", "secret": "test-secret"}
        
        # Store credential
        result = await backend.store_credential("test:user:1", test_creds)
        assert result is True
        
        # Retrieve credential
        retrieved = await backend.get_credential("test:user:1")
        assert retrieved == test_creds
    
    @pytest.mark.asyncio
    async def test_delete_credential(self, temp_file):
        """Test deleting credentials from file"""
        backend = FileBackend(temp_file)
        
        # Store credential
        await backend.store_credential("test:user:1", {"key": "value"})
        
        # Delete credential
        result = await backend.delete_credential("test:user:1")
        assert result is True
        
        # Verify deleted
        retrieved = await backend.get_credential("test:user:1")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_list_credentials(self, temp_file):
        """Test listing all credentials from file"""
        backend = FileBackend(temp_file)
        
        # Store multiple credentials
        await backend.store_credential("n8n:user:1", {"key": "value1"})
        await backend.store_credential("zapier:user:2", {"key": "value2"})
        
        keys = await backend.list_credentials()
        assert sorted(keys) == ["n8n:user:1", "zapier:user:2"]
    
    @pytest.mark.asyncio
    async def test_encryption(self, temp_file):
        """Test that credentials are encrypted in file"""
        backend = FileBackend(temp_file)
        
        test_creds = {"api_key": "super-secret-key"}
        await backend.store_credential("test:user:1", test_creds)
        
        # Read raw file content
        with open(temp_file, 'r') as f:
            content = f.read()
        
        # Should not contain the plain text secret
        assert "super-secret-key" not in content
        assert "api_key" not in content


class TestDatabaseBackend:
    """Test database credential backend"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = Mock()
        return db
    
    @pytest.fixture
    def mock_credential(self):
        """Create a mock credential object"""
        credential = Mock()
        credential.id = 1
        credential.platform = "n8n"
        credential.encrypted_data = Fernet.generate_key()
        return credential
    
    @pytest.mark.asyncio
    async def test_get_credential_success(self, mock_db, mock_credential):
        """Test successful credential retrieval from database"""
        backend = DatabaseBackend(mock_db)
        
        # Mock query result
        mock_db.query.return_value.filter.return_value.first.return_value = mock_credential
        
        # Mock decryption
        test_data = {"api_key": "test-key"}
        with patch.object(backend, '_decrypt_data', return_value=test_data):
            result = await backend.get_credential("n8n:user:1")
            assert result == test_data
    
    @pytest.mark.asyncio
    async def test_get_credential_not_found(self, mock_db):
        """Test credential not found in database"""
        backend = DatabaseBackend(mock_db)
        
        # Mock query result
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = await backend.get_credential("n8n:user:1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_encryption_decryption(self, mock_db):
        """Test encryption and decryption of credentials"""
        backend = DatabaseBackend(mock_db)
        
        test_data = {"api_key": "secret-key", "url": "https://example.com"}
        
        # Encrypt
        encrypted = backend._encrypt_data(test_data)
        assert isinstance(encrypted, str)
        assert "secret-key" not in encrypted
        
        # Decrypt
        decrypted = backend._decrypt_data(encrypted)
        assert decrypted == test_data


class TestPlatformCredentialService:
    """Test the main credential service"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        return Mock()
    
    @pytest.mark.asyncio
    async def test_backend_selection_environment(self, mock_db):
        """Test selecting environment backend"""
        with patch.dict(os.environ, {"CREDENTIAL_BACKEND": "environment"}):
            service = PlatformCredentialService(mock_db)
            assert isinstance(service.backend, EnvironmentBackend)
    
    @pytest.mark.asyncio
    async def test_backend_selection_file(self, mock_db):
        """Test selecting file backend"""
        with tempfile.NamedTemporaryFile() as tmp:
            with patch.dict(os.environ, {
                "CREDENTIAL_BACKEND": "file",
                "CREDENTIAL_FILE_PATH": tmp.name
            }):
                service = PlatformCredentialService(mock_db)
                assert isinstance(service.backend, FileBackend)
    
    @pytest.mark.asyncio
    async def test_backend_selection_database(self, mock_db):
        """Test selecting database backend (default)"""
        with patch.dict(os.environ, {"CREDENTIAL_BACKEND": "database"}):
            service = PlatformCredentialService(mock_db)
            assert isinstance(service.backend, DatabaseBackend)
    
    @pytest.mark.asyncio
    async def test_backend_selection_vault(self, mock_db):
        """Test selecting vault backend"""
        with patch.dict(os.environ, {
            "CREDENTIAL_BACKEND": "vault",
            "VAULT_URL": "http://vault:8200",
            "VAULT_TOKEN": "test-token"
        }):
            # Mock the vault backend import
            with patch('services.platform_credential_service.VaultCredentialBackend') as mock_vault:
                service = PlatformCredentialService(mock_db)
                # Should have tried to import vault backend
                assert service.backend is not None


@pytest.mark.integration
class TestVaultBackendIntegration:
    """Integration tests for Vault backend (requires running Vault)"""
    
    @pytest.fixture
    def vault_backend(self):
        """Create Vault backend if Vault is available"""
        if not os.environ.get("VAULT_TOKEN"):
            pytest.skip("Vault not configured, skipping integration test")
        
        from services.vault_credential_backend import VaultCredentialBackend
        return VaultCredentialBackend()
    
    @pytest.mark.asyncio
    async def test_vault_health_check(self, vault_backend):
        """Test Vault health check"""
        health = vault_backend.health_check()
        assert "healthy" in health
        assert "authenticated" in health
    
    @pytest.mark.asyncio
    async def test_vault_store_retrieve_delete(self, vault_backend):
        """Test full credential lifecycle in Vault"""
        test_key = "test:integration:123"
        test_creds = {
            "api_key": "vault-test-key",
            "secret": "vault-test-secret"
        }
        
        # Store
        result = await vault_backend.store_credential(test_key, test_creds)
        assert result is True
        
        # Retrieve
        retrieved = await vault_backend.get_credential(test_key)
        assert retrieved["api_key"] == test_creds["api_key"]
        assert retrieved["secret"] == test_creds["secret"]
        
        # List
        keys = await vault_backend.list_credentials()
        assert test_key.split("/")[-1] in keys
        
        # Delete
        result = await vault_backend.delete_credential(test_key)
        assert result is True
        
        # Verify deleted
        retrieved = await vault_backend.get_credential(test_key)
        assert retrieved is None