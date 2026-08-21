"""Regression: credential encryption keys must survive a restart.

Both credential services minted a fresh Fernet key with `Fernet.generate_key()`
whenever their key environment variable was unset, then cached it on the process:

- `PlatformCredentialService`'s file AND database backends, on PLATFORM_CREDENTIAL_KEY
- `CredentialService`'s file backend, on CREDENTIAL_ENCRYPTION_KEY, logged at INFO

No compose file, no `.env`, and no GCP deploy script sets either variable — the GA
deploy binds JWT_SECRET, POSTGRES_PASSWORD and an ENCRYPTION_KEY that nothing reads,
but neither credential key. So every deployment encrypted stored credentials with a
key generated at boot and discarded at exit. The ciphertext outlived the key: after
the next restart the platform decrypt path caught the failure and returned an empty
dict, so a credential did not error, it simply came back blank.

The fix is a deterministic key derived from the deployment's own SECRET_KEY when no
explicit key is set, so the key is stable across restarts without requiring new
provisioning on GA or Beast. An explicit key still wins, and remains what a real
deployment should set — rotating SECRET_KEY changes the derived key.

Verified here:
- a key survives a simulated restart, and old ciphertext still decrypts
- an explicitly configured key takes precedence
- two deployments with different SECRET_KEYs do not share a key
- the backends no longer mutate os.environ as a side effect of construction

This lives in the community unit suite rather than tests/integration/regressions/
because it needs `cryptography`, which is installed in the edition images but not on
the host where `make test-security` runs. `make test` executes this suite inside the
container.

covers: credential-key-persistence
"""

import os

import pytest
from cryptography.fernet import Fernet

from core.crypto import derive_fernet_key


A_REAL_KEY = Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def _clean_key_env(monkeypatch):
    """These are unset in every real deployment; make that the test baseline."""
    for name in ("PLATFORM_CREDENTIAL_KEY", "CREDENTIAL_ENCRYPTION_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SECRET_KEY", "a-deployment-secret-of-at-least-32-characters")


def _platform_file_backend(tmp_path):
    from services.platform_credential_service import FileBackend

    return FileBackend(str(tmp_path / "creds.json"))


# --- The actual defect: ciphertext outliving its key ------------------------

def test_derived_key_is_stable_across_restarts():
    """Two processes with the same SECRET_KEY must derive the same key, or
    anything written before a restart is unreadable after it."""
    assert derive_fernet_key("platform-credentials") == derive_fernet_key(
        "platform-credentials"
    )


def test_ciphertext_survives_a_simulated_restart(tmp_path):
    """Construct a backend, encrypt, throw it away, construct a fresh one — the
    shape of a container restart — and the credential must still be readable."""
    before = _platform_file_backend(tmp_path)
    token = before._encrypt_data({"username": "ehr-service-account"})

    after = _platform_file_backend(tmp_path)
    assert after._decrypt_data(token) == {"username": "ehr-service-account"}


def test_database_backend_shares_the_same_derived_key(tmp_path):
    """The database backend is the platform service's default, so it has the
    same exposure as the file backend and must resolve the same key."""
    from services.platform_credential_service import DatabaseBackend

    token = _platform_file_backend(tmp_path)._encrypt_data({"token": "abc"})
    assert DatabaseBackend(db=None)._decrypt_data(token) == {"token": "abc"}


def test_core_file_backend_key_also_survives_restart(tmp_path):
    """CredentialService's file backend had the same flaw, logged at INFO."""
    from core.services.credential_service import FileCredentialBackend

    path = str(tmp_path / "credentials.enc")
    token = FileCredentialBackend(path).fernet.encrypt(b"secret-value")
    assert FileCredentialBackend(path).fernet.decrypt(token) == b"secret-value"


# --- An explicit key must still win -----------------------------------------

def test_explicit_key_takes_precedence(tmp_path, monkeypatch):
    """Deriving is the fallback, never an override of real provisioning."""
    monkeypatch.setenv("PLATFORM_CREDENTIAL_KEY", A_REAL_KEY)
    token = _platform_file_backend(tmp_path)._encrypt_data({"k": "v"})
    assert Fernet(A_REAL_KEY.encode()).decrypt(token.encode())


def test_derived_key_is_deployment_specific(monkeypatch):
    """Two deployments must not share a credential key just because neither set
    one explicitly."""
    monkeypatch.setenv("SECRET_KEY", "first-deployment-secret-key-32-chars-long")
    first = derive_fernet_key("platform-credentials")
    monkeypatch.setenv("SECRET_KEY", "second-deployment-secret-key-32-chars-ok")
    assert derive_fernet_key("platform-credentials") != first


def test_purposes_do_not_share_a_key():
    """Key separation: the platform store and the core store are distinct."""
    assert derive_fernet_key("platform-credentials") != derive_fernet_key(
        "credential-service"
    )


def test_derived_key_is_a_valid_fernet_key():
    Fernet(derive_fernet_key("platform-credentials").encode())


# --- No hidden side effects --------------------------------------------------

def test_constructing_a_backend_does_not_mutate_the_environment(tmp_path):
    """The old code assigned its generated key back into os.environ, so merely
    reading a credential rewrote process state that later readers depended on."""
    _platform_file_backend(tmp_path)
    assert "PLATFORM_CREDENTIAL_KEY" not in os.environ
