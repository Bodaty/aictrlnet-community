"""Configuration management using Pydantic settings."""

from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field, PostgresDsn, computed_field
from functools import lru_cache
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AICtrlNet"
    VERSION: str = "1.0.0"
    EDITION: str = Field(default="community", env="AICTRLNET_EDITION")
    
    # Security
    # Accept the env var names the infra already sets. docker-compose uses
    # JWT_SECRET_KEY and the GCP deploy passes JWT_SECRET; before this the code
    # only read SECRET_KEY, so prod silently fell back to the dev default below.
    SECRET_KEY: str = Field(
        default="dev-secret-key-change-in-production",
        validation_alias=AliasChoices("SECRET_KEY", "JWT_SECRET_KEY", "JWT_SECRET"),
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # MFA Encryption
    MFA_ENCRYPTION_KEY: str = Field(
        default="dev-mfa-encryption-key-32-chars!",  # Must be 32 chars for Fernet
        description="Key for encrypting MFA secrets (must be 32 bytes when encoded)"
    )
    
    # OAuth2 Configuration
    OAUTH2_ENCRYPTION_KEY: str = Field(
        default="W3sKDBj0Wqrq-Fu9cVMd0cKCC0iF9SiNjHVKcIjRjko=",  # Valid Fernet key for dev
        description="Key for encrypting OAuth2 secrets (base64-encoded)"
    )
    OAUTH2_REDIRECT_URI: str = Field(
        default="http://localhost:3000/auth/callback",
        description="OAuth2 callback URI"
    )
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = Field(
        default=[
            "http://localhost:3000", 
            "http://localhost:8000", 
            "http://localhost:8001",
            "http://localhost:8002",
            "http://localhost:8080", 
            "http://127.0.0.1:8080",
            "http://127.0.0.1:3000"
        ]
    )
    
    # Database
    POSTGRES_SERVER: str = Field(default="localhost")
    POSTGRES_USER: str = Field(default="app")
    POSTGRES_PASSWORD: str = Field(default="local_dev_password")
    POSTGRES_DB: str = Field(default="aictrlnet_community")
    POSTGRES_PORT: int = Field(default=5432)
    
    @computed_field
    @property
    def DATABASE_URL(self) -> PostgresDsn | str:
        """Construct database URL from components."""
        # Check for SQLALCHEMY_DATABASE_URI env var (for Cloud SQL Unix sockets)
        # This bypasses Pydantic validation which fails on Unix socket paths
        sqlalchemy_uri = os.getenv("SQLALCHEMY_DATABASE_URI")
        if sqlalchemy_uri:
            return sqlalchemy_uri

        # Standard URL construction with Pydantic validation
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_SERVER,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB,
        )
    
    # Redis (for caching)
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_DB: int = Field(default=0)
    
    @computed_field
    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL from components."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    CACHE_TTL: int = Field(default=300)  # Default cache TTL in seconds
    
    # Ollama (for AI features)
    OLLAMA_URL: str = Field(default="http://localhost:11434")

    # Default LLM Model (can be overridden by environment for cloud deployments)
    # Aligned with UI default (llama3.1-local maps to this)
    DEFAULT_LLM_MODEL: str = Field(default="llama3.1:8b-instruct-q4_K_M", env="DEFAULT_LLM_MODEL")

    # LLM Service URL (if using external LLM service adapter)
    # Set to None to disable the LLM service adapter
    LLM_SERVICE_URL: Optional[str] = Field(default=None, env="LLM_SERVICE_URL")
    
    # Chinese AI Provider API Keys
    DEEPSEEK_API_KEY: str = Field(default="", description="DeepSeek Platform API key")

    # Channel Webhook Secrets
    DISCORD_PUBLIC_KEY: str = Field(default="", description="Hex-encoded Ed25519 public key from Discord Developer Portal")
    EMAIL_WEBHOOK_SECRET: str = Field(default="", description="Shared secret for email inbound webhook validation")

    # Stripe Payment Processing
    STRIPE_SECRET_KEY: str = Field(default="sk_test_dummy")
    STRIPE_WEBHOOK_SECRET: str = Field(default="whsec_dummy")
    # Pre-created Stripe Price IDs (create these in Stripe Dashboard)
    STRIPE_PRICE_BUSINESS_STARTER: str = Field(default="")
    STRIPE_PRICE_BUSINESS_GROWTH: str = Field(default="")
    STRIPE_PRICE_BUSINESS_SCALE: str = Field(default="")
    STRIPE_PRICE_ENTERPRISE: str = Field(default="")
    STRIPE_PRICE_BUSINESS_STARTER_ANNUAL: str = Field(default="")
    STRIPE_PRICE_BUSINESS_GROWTH_ANNUAL: str = Field(default="")
    STRIPE_PRICE_BUSINESS_SCALE_ANNUAL: str = Field(default="")
    STRIPE_PRICE_ENTERPRISE_ANNUAL: str = Field(default="")
    FRONTEND_URL: str = Field(default="http://localhost:3000")
    TRIAL_DAYS: int = Field(default=14)
    # Trial redemption codes: "CODE:days:max_redemptions[:expiry_iso]" comma-separated,
    # e.g. "INSTITUTE90:90:40:2026-10-31,LIVE30:30:100". Empty = no codes valid.
    TRIAL_CODES: str = Field(default="")
    
    # Feature Flags
    FEATURES: Dict[str, Any] = Field(default_factory=lambda: {
        "ai_enabled": True,
        "websocket_enabled": True,
        "audit_logging": False,
        "multi_tenant": False,
    })
    
    # Edition Features
    EDITION_FEATURES: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "community": {
            "max_workflows": 10,
            "max_adapters": 5,
            "max_users": 1,
            "ai_enabled": True,
            "custom_branding": False,
        },
        "business": {
            "max_workflows": 100,
            "max_adapters": 20,
            "max_users": 50,
            "ai_enabled": True,
            "custom_branding": True,
            "approval_workflows": True,
            "rbac": True,
        },
        "enterprise": {
            "max_workflows": -1,  # Unlimited
            "max_adapters": -1,
            "max_users": -1,
            "ai_enabled": True,
            "custom_branding": True,
            "approval_workflows": True,
            "rbac": True,
            "multi_tenant": True,
            "federation": True,
            "audit_logging": True,
        }
    })
    
    def get_edition_features(self) -> Dict[str, Any]:
        """Get features for current edition."""
        return self.EDITION_FEATURES.get(self.EDITION.lower(), self.EDITION_FEATURES["community"])
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Environment settings
    # Fail-safe default: unless a deployment *explicitly* declares itself as
    # development, it is treated as production (secret guard armed, dev token
    # refused). The dev docker-compose sets ENVIRONMENT=development explicitly.
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    # Explicit opt-in for the static dev bearer token. Never set outside local;
    # prod/staging leave it false. Enforced together with ENVIRONMENT=development.
    ALLOW_DEV_TOKENS: bool = Field(default=False, env="ALLOW_DEV_TOKENS")
    USE_CREATE_ALL: bool = Field(default=False, env="USE_CREATE_ALL")  # Default to migrations even in dev
    DATA_PATH: str = Field(default="/tmp/aictrlnet", env="DATA_PATH")
    # Explicit opt-in for a deployment that handles PHI. When true, the startup
    # guard below refuses to boot unless documents can only land on the encrypted
    # volume and credentials are held by an encrypted backend. Default false, so
    # deployments that handle no PHI are unaffected.
    AICTRLNET_PHI_MODE: bool = Field(default=False, env="AICTRLNET_PHI_MODE")
    # Where uploaded and generated documents are staged. Promoted from a bare
    # os.environ read so the PHI guard can see it — a guard cannot enforce what
    # it cannot see. The default must stay this literal rather than deriving from
    # DATA_PATH: the staged-file read path rejects anything outside it, so
    # deriving it would break legitimate reads whenever DATA_PATH is set (see the
    # comment in nodes/implementations/file_process_node.py).
    STAGED_FILES_DIR: str = Field(
        default="/tmp/aictrlnet/staged_files", env="STAGED_FILES_DIR"
    )
    # Credential storage backend: 'environment' (plaintext env vars), 'file' and
    # 'database' (both Fernet-encrypted), or 'vault'. The default matches the one
    # get_credential_service() has always used; PHI mode refuses 'environment'.
    CREDENTIAL_BACKEND: str = Field(default="environment", env="CREDENTIAL_BACKEND")
    
    # Performance
    # PER-WORKER pool size. With uvicorn `--workers 2` x 3 editions = 6 worker
    # processes, each with pool_size=20 + max_overflow=20 = 40 connections max,
    # the steady-state ceiling is 240 connections (well under the Postgres
    # max_connections=300 that docker-compose.yml now sets).
    # The previous 50/10 sizing assumed single-worker per edition (3 procs ×
    # 70 conns = 210, already over the default Postgres 100 cap).
    # See docs/architecture/APPROVALS_SPEC.md §13.3 for the original derivation.
    # Per-worker DB pool sizing. Defaults assume 4 uvicorn workers per
    # edition × 3 editions sharing one Postgres (max_connections=300).
    # Math: 3 × 4 × (MAX_CONNECTIONS_COUNT + MAX_OVERFLOW_COUNT) ≤ 280.
    # Override via env var if you change worker count.
    MAX_CONNECTIONS_COUNT: int = Field(default=10)
    MAX_OVERFLOW_COUNT: int = Field(default=10)
    MIN_CONNECTIONS_COUNT: int = Field(default=5)

    # Approvals feature flags (PR 1 of approvals workstream).
    # Defaulted to true post-PR-2: pre-customer state means "bake in
    # production" yields no signal, so we make the strict path the
    # default and exercise it on every dev/CI run. Flag-checking
    # branches survive for one more PR cycle (cheap insurance) and
    # are removed in the stabilization PR after PR 4 lands.
    APPROVALS_STRICT_LOCKING: bool = Field(default=True, env="APPROVALS_STRICT_LOCKING")
    APPROVALS_FAIL_CLOSED: bool = Field(default=True, env="APPROVALS_FAIL_CLOSED")


import logging as _logging

_config_logger = _logging.getLogger(__name__)

DEV_DEFAULT_SECRET_KEY = "dev-secret-key-change-in-production"
# Environments that represent a real deployment and must not run on the dev key.
# Intentionally an allow-list: "test"/"ci"/"development" keep the built-in default.
_DEPLOY_ENVIRONMENTS = {"production", "prod", "staging", "stage"}


# Environments where the interactive API docs and the OpenAPI schema are served.
# An allow-list in the other direction from _DEPLOY_ENVIRONMENTS above, and
# deliberately so: this follows the same fail-safe rule as the ENVIRONMENT field
# ("unless a deployment explicitly declares itself as development, it is treated
# as production"). An unset, empty, or unrecognised value therefore hides the
# docs rather than publishing them.
_DOCS_ENVIRONMENTS = {"development", "dev", "local", "test", "ci"}


def exposes_interactive_docs(settings) -> bool:
    """True only where the environment explicitly declares itself non-deployed.

    Used by the app factories to decide whether to serve /docs, /redoc and
    /openapi.json at all.
    """
    return (getattr(settings, "ENVIRONMENT", "") or "").strip().lower() in _DOCS_ENVIRONMENTS

# Minimum acceptable length for a signing/encryption secret in a deployment.
_MIN_SECRET_LEN = 32

# Committed / well-known weak literals that must never sign or encrypt in a
# real deployment. Exact-match alone is not enough (staging shipped its own
# predictable literal), so the guard rejects any of these AND anything shorter
# than _MIN_SECRET_LEN.
_DENYLISTED_SECRETS = {
    DEV_DEFAULT_SECRET_KEY,
    "staging-jwt-secret-key-change-in-production",
    "staging-secret-key-change-in-production",
    "your-secret-key-here",
    "changeme",
    "secret",
}

# Encryption keys whose committed dev defaults are still in the codebase. These
# are WARN-ONLY for now (prod has historically run on them; hard-failing would
# block boot until they are provisioned in Secret Manager and existing data is
# re-encrypted). Flip to hard-fail in the follow-up once rotation is done.
_ENCRYPTION_KEY_DEFAULTS = {
    "MFA_ENCRYPTION_KEY": "dev-mfa-encryption-key-32-chars!",
    "OAUTH2_ENCRYPTION_KEY": "W3sKDBj0Wqrq-Fu9cVMd0cKCC0iF9SiNjHVKcIjRjko=",
}


def _secret_is_weak(value: str) -> bool:
    return (value or "") in _DENYLISTED_SECRETS or len(value or "") < _MIN_SECRET_LEN


def validate_secret_for_environment(settings: "Settings") -> None:
    """Fail-safe secret validation at startup for real deployments.

    Call this once from app startup (lifespan), NOT as a Pydantic validator —
    Settings() is constructed all over the test suite (with ENVIRONMENT=test and
    the dev defaults), and a raising validator would break those.

    - SECRET_KEY: HARD FAIL if it is a committed/denylisted literal or shorter
      than 32 chars in a deploy environment (JWT forgery otherwise).
    - Encryption keys (MFA/OAuth2): WARN LOUDLY if still on the committed dev
      default in a deploy environment. (Warn-only until keys are provisioned and
      data re-encrypted; see security remediation plan NEW-S1/§2A.)
    """
    if (settings.ENVIRONMENT or "").strip().lower() not in _DEPLOY_ENVIRONMENTS:
        return

    if _secret_is_weak(settings.SECRET_KEY):
        raise RuntimeError(
            f"SECRET_KEY is a weak/committed default in a "
            f"'{settings.ENVIRONMENT}' deployment. Set SECRET_KEY (or JWT_SECRET / "
            f"JWT_SECRET_KEY) to a random secret of at least {_MIN_SECRET_LEN} "
            f"characters — refusing to start with a predictable JWT signing key."
        )

    for field_name, dev_default in _ENCRYPTION_KEY_DEFAULTS.items():
        value = getattr(settings, field_name, None)
        if value == dev_default:
            _config_logger.critical(
                "SECURITY: %s is the committed development default in a '%s' "
                "deployment. Data encrypted with it is recoverable from the "
                "repository. Provision a real key and re-encrypt existing data. "
                "(This will become a hard startup failure in a follow-up.)",
                field_name, settings.ENVIRONMENT,
            )


# Filesystem root that must never hold PHI. Broadly readable, cleared
# unpredictably, and outside the encrypted volume every BAA Exhibit A commits to.
_PHI_FORBIDDEN_ROOT = "/tmp"


def _resolved(path: str) -> Path:
    """Real path, with symlinks and '..' collapsed.

    Containment is decided on this, never on the configured string: both
    "/mnt/phi/../../tmp/aictrlnet" and a symlink pointing into /tmp read like
    volume paths and land in /tmp, and a relative path resolves against whatever
    working directory the container happened to start in.
    """
    return Path(path or "").expanduser().resolve()


def validate_phi_mode(settings: "Settings") -> None:
    """Refuse to boot a PHI deployment on insecure defaults.

    Call this once from app startup (lifespan), NOT as a Pydantic validator —
    same reason as validate_secret_for_environment above: Settings() is
    constructed all over the test suite with the dev defaults, and a raising
    validator would break those.

    A no-op unless AICTRLNET_PHI_MODE is set, so deployments that handle no PHI
    behave exactly as before. When it is set, the defaults that would otherwise
    apply are refused:

    - DATA_PATH and STAGED_FILES_DIR default under /tmp, so documents land off
      the encrypted volume (risk analysis R-01), and staged files must sit on the
      PHI volume rather than merely somewhere off /tmp.
    - CREDENTIAL_BACKEND defaults to 'environment', i.e. plaintext env vars
      holding the practice's EHR service-account login (R-02). Encrypted 'file',
      'database' and 'vault' backends already exist.
    - ALLOW_DEV_TOKENS is true in the dev compose file, and the static dev bearer
      token would authenticate anyone to a system holding PHI.
    - ENVIRONMENT must name a real deployment, which also arms the SECRET_KEY
      guard above on the same box.

    Every problem is reported at once. Failing on the first would make an
    operator commissioning a practice machine rediscover the next one on the
    following boot.

    Path shape only: this deliberately does not check that the directories exist,
    are writable, or are genuinely encrypted. A volume may be mounted after the
    process starts, and encryption is not observable from inside it. Provisioning
    the volume is the BAA control; this stops the configuration pointing off it.
    """
    if not getattr(settings, "AICTRLNET_PHI_MODE", False):
        return

    forbidden = _resolved(_PHI_FORBIDDEN_ROOT)
    data_path = _resolved(settings.DATA_PATH)
    staged_dir = _resolved(settings.STAGED_FILES_DIR)
    problems = []

    # A relative path is not a mount point: it resolves against whatever working
    # directory the process started in, so the same config lands PHI in different
    # places depending on how the container was launched.
    for name, configured in (
        ("DATA_PATH", settings.DATA_PATH),
        ("STAGED_FILES_DIR", settings.STAGED_FILES_DIR),
    ):
        if not Path(configured or "").is_absolute():
            problems.append(
                f"{name}={configured!r} is not an absolute path, so where it "
                f"lands depends on the process working directory. Required: an "
                f"absolute path on the encrypted PHI volume."
            )

    if data_path.is_relative_to(forbidden):
        problems.append(
            f"DATA_PATH={settings.DATA_PATH!r} resolves to '{data_path}', which is "
            f"under '{forbidden}'. Required: a path on the encrypted PHI volume."
        )

    if staged_dir.is_relative_to(forbidden):
        problems.append(
            f"STAGED_FILES_DIR={settings.STAGED_FILES_DIR!r} resolves to "
            f"'{staged_dir}', which is under '{forbidden}'. Required: a path on "
            f"the encrypted PHI volume."
        )
    elif not staged_dir.is_relative_to(data_path):
        problems.append(
            f"STAGED_FILES_DIR={settings.STAGED_FILES_DIR!r} resolves to "
            f"'{staged_dir}', which is outside DATA_PATH '{data_path}'. Required: "
            f"a directory under DATA_PATH, so staged documents stay on the volume."
        )

    if (settings.CREDENTIAL_BACKEND or "").strip().lower() == "environment":
        problems.append(
            "CREDENTIAL_BACKEND='environment' stores credentials as plaintext "
            "environment variables. Required: 'file', 'database' or 'vault', all "
            "of which encrypt at rest."
        )

    if settings.ALLOW_DEV_TOKENS:
        problems.append(
            "ALLOW_DEV_TOKENS=true accepts the static development bearer token, "
            "which would let any caller authenticate to a system holding PHI. "
            "Required: false."
        )

    environment = (settings.ENVIRONMENT or "").strip().lower()
    if environment not in _DEPLOY_ENVIRONMENTS:
        problems.append(
            f"ENVIRONMENT={settings.ENVIRONMENT!r} is not a deployment "
            f"environment. Required: one of "
            f"{', '.join(sorted(_DEPLOY_ENVIRONMENTS))}."
        )

    if problems:
        raise RuntimeError(
            "AICTRLNET_PHI_MODE is on, but this deployment would place protected "
            "health information at risk. Refusing to start:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nEvery BAA commits that PHI and audit records live on a separate "
            "encrypted volume. Fix the settings above or unset AICTRLNET_PHI_MODE "
            "if this deployment handles no PHI."
        )


def get_settings() -> Settings:
    """Get settings instance."""
    # Force reload of environment variables
    edition = os.getenv("AICTRLNET_EDITION", "community")
    return Settings(EDITION=edition)


# Don't create a global instance - use get_settings() instead
# This ensures environment variables are loaded correctly
_settings = None

def get_cached_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings

# For backward compatibility
settings = get_cached_settings()