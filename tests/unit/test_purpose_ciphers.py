"""MFA and OAuth2 ciphers stop encrypting with keys committed to this repository.

Deployments (GCP, Beast) ran on the committed dev defaults for MFA_ENCRYPTION_KEY
and OAUTH2_ENCRYPTION_KEY, so stored OAuth tokens and provider client secrets were
decryptable by anyone with the public Community source. Same remedy as NEW-S1:
when the configured key is the dev default, encrypt under a key derived per
purpose from the deployment's ENCRYPTION_KEY, and keep the dev default as a
decrypt-only fallback so existing rows still read until they are rotated.
"""
import base64
import hashlib

import pytest
from cryptography.fernet import Fernet, InvalidToken

from core import crypto
from core.config import _ENCRYPTION_KEY_DEFAULTS

MFA_DEV = _ENCRYPTION_KEY_DEFAULTS["MFA_ENCRYPTION_KEY"]
OAUTH2_DEV = _ENCRYPTION_KEY_DEFAULTS["OAUTH2_ENCRYPTION_KEY"]


def _mfa_dev_fernet():
    # mfa_service's historical reading of a non-Fernet key: SHA-256, base64.
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(MFA_DEV.encode()).digest()))


def test_oauth2_dev_default_encrypts_under_a_derived_key():
    cipher = crypto.oauth2_cipher(OAUTH2_DEV)
    token = cipher.encrypt(b"refresh-token")
    with pytest.raises(InvalidToken):
        Fernet(OAUTH2_DEV.encode()).decrypt(token)
    assert cipher.decrypt(token) == b"refresh-token"


def test_oauth2_dev_default_still_reads_existing_rows():
    old = Fernet(OAUTH2_DEV.encode()).encrypt(b"client-secret")
    assert crypto.oauth2_cipher(OAUTH2_DEV).decrypt(old) == b"client-secret"


def test_oauth2_explicit_key_is_used_as_is_and_accepts_bytes():
    explicit = Fernet.generate_key()
    token = crypto.oauth2_cipher(explicit).encrypt(b"x")
    assert Fernet(explicit).decrypt(token) == b"x"
    old = Fernet(OAUTH2_DEV.encode()).encrypt(b"y")
    assert crypto.oauth2_cipher(explicit.decode()).decrypt(old) == b"y"


def test_mfa_dev_default_encrypts_under_a_derived_key_and_reads_old_rows():
    cipher = crypto.mfa_cipher(MFA_DEV)
    token = cipher.encrypt(b"totp-secret")
    with pytest.raises(InvalidToken):
        _mfa_dev_fernet().decrypt(token)
    old = _mfa_dev_fernet().encrypt(b"old-secret")
    assert cipher.decrypt(old) == b"old-secret"


def test_mfa_explicit_non_fernet_key_keeps_its_sha256_reading():
    explicit = "an-operator-chosen-passphrase-0123456789"
    legacy_reader = Fernet(base64.urlsafe_b64encode(hashlib.sha256(explicit.encode()).digest()))
    token = crypto.mfa_cipher(explicit).encrypt(b"s")
    assert legacy_reader.decrypt(token) == b"s"


def test_mfa_and_oauth2_derived_keys_differ():
    token = crypto.mfa_cipher(MFA_DEV).encrypt(b"s")
    with pytest.raises(InvalidToken):
        crypto.oauth2_cipher(OAUTH2_DEV).decrypt(token)


def test_rotate_moves_old_rows_to_the_primary_key():
    old = Fernet(OAUTH2_DEV.encode()).encrypt(b"access-token")
    cipher = crypto.oauth2_cipher(OAUTH2_DEV)
    rotated = cipher.rotate(old)
    with pytest.raises(InvalidToken):
        Fernet(OAUTH2_DEV.encode()).decrypt(rotated)
    assert cipher.decrypt(rotated) == b"access-token"
