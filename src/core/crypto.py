"""Cryptographic utilities for encrypting and decrypting sensitive data."""

import json
import base64
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import os
import logging
from functools import lru_cache

from core.config import _DEPLOY_ENVIRONMENTS


# Get encryption key from environment or generate a default one
# In production, this should ALWAYS come from environment variable
ENCRYPTION_KEY = os.getenv("AICTRLNET_ENCRYPTION_KEY")

if not ENCRYPTION_KEY:
    # Generate a key from a password for development
    # WARNING: In production, use a proper key management system
    password = b"dev-encryption-key-change-in-production"
    salt = b"aictrlnet-salt-v1"  
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    ENCRYPTION_KEY = key.decode('utf-8')

_logger = logging.getLogger(__name__)

# Purposes that have already announced a derived key, so the warning is emitted
# once per process rather than on every backend construction.
_announced_derivations = set()


@lru_cache(maxsize=8)
def _derive(purpose: str, secret: str) -> str:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=f"aictrlnet-{purpose}-v1".encode(),
        iterations=100000,
        backend=default_backend(),
    )
    return base64.urlsafe_b64encode(kdf.derive(secret.encode())).decode()


def derive_fernet_key(purpose: str) -> str:
    """Deterministic per-deployment Fernet key for `purpose`.

    A fallback for credential backends whose explicit key variable is unset, and
    only that — an explicitly configured key always wins.

    The alternative it replaces was `Fernet.generate_key()` at process start,
    which meant the key died with the process while the ciphertext it produced
    lived on in the database. After a restart those credentials could not be
    decrypted; the platform backend caught the failure and returned an empty
    dict, so a credential came back blank rather than erroring. Since no compose
    file, `.env`, or deploy script sets either credential key variable, that was
    every deployment.

    Deriving from SECRET_KEY keeps the key stable for the life of a deployment
    without requiring key provisioning that does not exist today, and keeps it
    distinct between deployments. `purpose` is mixed into the salt so separate
    credential stores do not share a key.

    Rotating SECRET_KEY changes this key and makes existing ciphertext
    unreadable, which is why a deployment holding credentials it cannot afford
    to re-enter should set the explicit variable instead. Result is cached, since
    backends are constructed per request in places.
    """
    from core.config import get_settings

    settings = get_settings()
    key = _derive(purpose, settings.SECRET_KEY or "")

    if purpose not in _announced_derivations:
        _announced_derivations.add(purpose)
        deploy = (settings.ENVIRONMENT or "").strip().lower() in _DEPLOY_ENVIRONMENTS
        _logger.log(
            logging.CRITICAL if deploy else logging.INFO,
            "Credential key for '%s' is derived from SECRET_KEY because its "
            "explicit key variable is unset. Credentials survive restarts, but "
            "rotating SECRET_KEY will make them unreadable. Provision a dedicated "
            "key for this deployment.",
            purpose,
        )
    return key


# Initialize Fernet cipher
_cipher = None

def get_cipher() -> Fernet:
    """Get or create the Fernet cipher instance."""
    global _cipher
    if _cipher is None:
        key = ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY
        _cipher = Fernet(key)
    return _cipher


def encrypt_data(data: Any) -> str:
    """Encrypt data to a string.
    
    Args:
        data: Any JSON-serializable data to encrypt
        
    Returns:
        Base64-encoded encrypted string
    """
    try:
        # Convert to JSON string
        json_str = json.dumps(data)
        
        # Encrypt
        cipher = get_cipher()
        encrypted = cipher.encrypt(json_str.encode())
        
        # Return as base64 string
        return base64.b64encode(encrypted).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encrypt data: {str(e)}")


def decrypt_data(encrypted_data: str) -> Any:
    """Decrypt data from an encrypted string.
    
    Args:
        encrypted_data: Base64-encoded encrypted string
        
    Returns:
        The original decrypted data
    """
    try:
        # Decode from base64
        encrypted = base64.b64decode(encrypted_data.encode())
        
        # Decrypt
        cipher = get_cipher()
        decrypted = cipher.decrypt(encrypted)
        
        # Parse JSON
        return json.loads(decrypted.decode())
    except Exception as e:
        raise ValueError(f"Failed to decrypt data: {str(e)}")


def encrypt_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Encrypt sensitive fields in a dictionary.
    
    Only encrypts fields that are commonly sensitive:
    - password, api_key, secret, token, credential
    
    Args:
        data: Dictionary potentially containing sensitive data
        
    Returns:
        Dictionary with sensitive fields encrypted
    """
    sensitive_fields = {
        'password', 'api_key', 'secret', 'token', 'credential',
        'access_token', 'refresh_token', 'client_secret', 'private_key'
    }
    
    encrypted_data = data.copy()
    
    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_fields):
            if value is not None:
                encrypted_data[key] = encrypt_data(value)
    
    return encrypted_data


def decrypt_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Decrypt sensitive fields in a dictionary.
    
    Attempts to decrypt fields that appear to be encrypted (base64 format).
    
    Args:
        data: Dictionary potentially containing encrypted data
        
    Returns:
        Dictionary with sensitive fields decrypted
    """
    decrypted_data = data.copy()
    
    for key, value in data.items():
        if isinstance(value, str) and value:
            # Check if it looks like encrypted data (base64)
            try:
                # Try to decode as base64
                base64.b64decode(value)
                # If successful, try to decrypt
                decrypted_data[key] = decrypt_data(value)
            except:
                # Not encrypted or failed to decrypt, keep original
                pass
    
    return decrypted_data


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.
    
    This is for password hashing, not encryption.
    Use this for storing user passwords.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password to compare against
        
    Returns:
        True if password matches, False otherwise
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.verify(plain_password, hashed_password)