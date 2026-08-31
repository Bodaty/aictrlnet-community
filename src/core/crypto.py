"""Cryptographic utilities for encrypting and decrypting sensitive data."""

import json
import base64
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import os
import logging
from functools import lru_cache

from core.config import _DEPLOY_ENVIRONMENTS


_logger = logging.getLogger(__name__)

# The key this module used to encrypt with, unconditionally: PBKDF2 over a
# password and salt that are BOTH literals in this file. Deterministic, so
# anyone with repo access can reproduce it exactly. Kept here as a
# DECRYPT-ONLY key so ciphertext already written under it stays readable —
# without it, configuring a real key would orphan existing rows, which is why
# this finding sat open across three filings as "needs a re-encryption path".
# It is never used to encrypt anything new.
_LEGACY_PASSWORD = b"dev-encryption-key-change-in-production"
_LEGACY_SALT = b"aictrlnet-salt-v1"


def _pbkdf2_key(password: bytes, salt: bytes) -> str:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    return base64.urlsafe_b64encode(kdf.derive(password)).decode()


def coerce_fernet_key(material: str) -> str:
    """Accept either a canonical Fernet key or arbitrary secret material.

    Two readers of `AICTRLNET_ENCRYPTION_KEY` disagreed on its format: this
    module passed the value straight to Fernet, while the Business
    EncryptionService base64-DECODED it first. So provisioning the variable
    with a normal 44-char Fernet key — what an operator actually mints, and the
    shape of the platform credential key already deployed on Beast — made
    Fernet(...) raise "must be 32 url-safe base64-encoded bytes" and took the
    Business service down at construction. The documented remediation
    ("map ENCRYPTION_KEY to AICTRLNET_ENCRYPTION_KEY") would have triggered it.

    Both call sites now go through here, so any reasonable value works: a
    canonical key is used as-is, anything else is stretched into one.
    """
    material = (material or "").strip()
    try:
        Fernet(material.encode())
        return material
    except Exception:
        return _pbkdf2_key(material.encode(), b"aictrlnet-encryption-key-v1")


def _resolve_encryption_key() -> tuple[str, str]:
    """(key used for encryption, where it came from).

    `AICTRLNET_ENCRYPTION_KEY` is the documented name and wins. `ENCRYPTION_KEY`
    is accepted because it is the name the GCP deploy actually binds Secret
    Manager to (`deploy-gcp.sh:518,622`, `config/deployment.yaml:54`) while
    nothing read it — so honouring it closes the gap with no deploy change and
    no key rotation. Failing those, derive per-deployment from SECRET_KEY, the
    same approach derive_fernet_key() already uses for credential backends:
    stable across restarts, distinct between deployments, and not printed in
    this repository.
    """
    for var in ("AICTRLNET_ENCRYPTION_KEY", "ENCRYPTION_KEY"):
        raw = os.getenv(var)
        if raw and raw.strip():
            return coerce_fernet_key(raw), var

    from core.config import get_settings

    secret = (get_settings().SECRET_KEY or "").strip()
    if secret:
        return _pbkdf2_key(secret.encode(), b"aictrlnet-encryption-v1"), "SECRET_KEY"

    return _pbkdf2_key(_LEGACY_PASSWORD, _LEGACY_SALT), "hardcoded-fallback"


ENCRYPTION_KEY, _ENCRYPTION_KEY_SOURCE = _resolve_encryption_key()

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
_announced_encryption_source = False


def get_cipher() -> MultiFernet:
    """Cipher for encrypt_data/decrypt_data.

    A MultiFernet, not a Fernet: it encrypts with the FIRST key and decrypts
    with ANY of them. That is what lets a deployment start using a real key
    without re-encrypting — rows written under the old source-visible key still
    decrypt via the legacy entry, while everything new is written under the
    configured one.
    """
    global _cipher, _announced_encryption_source
    if _cipher is None:
        legacy = _pbkdf2_key(_LEGACY_PASSWORD, _LEGACY_SALT)
        keys = [Fernet(ENCRYPTION_KEY.encode())]
        if ENCRYPTION_KEY != legacy:
            keys.append(Fernet(legacy.encode()))
        _cipher = MultiFernet(keys)

        if not _announced_encryption_source:
            _announced_encryption_source = True
            from core.config import get_settings

            deploy = (get_settings().ENVIRONMENT or "").strip().lower() in _DEPLOY_ENVIRONMENTS
            if _ENCRYPTION_KEY_SOURCE == "hardcoded-fallback":
                _logger.log(
                    logging.CRITICAL if deploy else logging.WARNING,
                    "Adapter/federation data is being encrypted with a key derived "
                    "from a password committed to this repository. Anyone with repo "
                    "access can decrypt it. Set AICTRLNET_ENCRYPTION_KEY.",
                )
            elif _ENCRYPTION_KEY_SOURCE == "SECRET_KEY":
                _logger.log(
                    logging.CRITICAL if deploy else logging.INFO,
                    "Encryption key derived from SECRET_KEY because neither "
                    "AICTRLNET_ENCRYPTION_KEY nor ENCRYPTION_KEY is set. Stable "
                    "across restarts, but rotating SECRET_KEY makes existing data "
                    "unreadable. Provision a dedicated key.",
                )
            else:
                _logger.info("Encryption key loaded from %s.", _ENCRYPTION_KEY_SOURCE)
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