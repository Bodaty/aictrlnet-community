"""Configuration management using Pydantic settings."""

from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn, computed_field
from functools import lru_cache
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
    VERSION: str = "2.0.0"
    EDITION: str = Field(default="community", env="AICTRLNET_EDITION")
    
    # Security
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production")
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
    POSTGRES_USER: str = Field(default="aictrlnet")
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
    OLLAMA_MODEL: str = Field(default="llama2")

    # Default LLM Model (can be overridden by environment for cloud deployments)
    # Aligned with UI default (llama3.1-local maps to this)
    DEFAULT_LLM_MODEL: str = Field(default="llama3.1:8b-instruct-q4_K_M", env="DEFAULT_LLM_MODEL")

    # LLM Service URL (if using external LLM service adapter)
    # Set to None to disable the LLM service adapter
    LLM_SERVICE_URL: Optional[str] = Field(default=None, env="LLM_SERVICE_URL")
    
    # Stripe Payment Processing
    STRIPE_SECRET_KEY: str = Field(default="sk_test_dummy")
    STRIPE_WEBHOOK_SECRET: str = Field(default="whsec_dummy")
    FRONTEND_URL: str = Field(default="http://localhost:3000")
    
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
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    USE_CREATE_ALL: bool = Field(default=False, env="USE_CREATE_ALL")  # Default to migrations even in dev
    DATA_PATH: str = Field(default="/tmp/aictrlnet", env="DATA_PATH")
    
    # Performance
    MAX_CONNECTIONS_COUNT: int = Field(default=10)
    MIN_CONNECTIONS_COUNT: int = Field(default=10)


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