"""Basic security service for Community Edition.

Provides essential security features without database persistence.
Advanced features like rate limiting, IP blocking, and security alerts
are available in Business Edition.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from collections import defaultdict

from core.config import settings


logger = logging.getLogger(__name__)


class BasicSecurityService:
    """Basic security service for Community Edition.
    
    Provides:
    - Password validation
    - Basic rate limiting (in-memory)
    - Simple authentication helpers
    
    Advanced features available in Business Edition:
    - Persistent rate limiting
    - IP blocking
    - Security alerts
    - Compliance validation
    - Audit logging
    """
    
    # Password validation rules
    PASSWORD_RULES = {
        "min_length": 8,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_digit": True,
        "require_special": True,
        "special_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?"
    }
    
    def __init__(self):
        """Initialize the basic security service."""
        # In-memory rate limiting
        self._rate_limit_cache = defaultdict(list)
        self._blocked_ips = set()
        self._failed_auth_attempts = defaultdict(int)
        
        # Load configuration
        self.config_file = Path(settings.DATA_PATH) / "security" / "config.json"
        self._load_config()
    
    def _load_config(self):
        """Load security configuration."""
        if self.config_file.exists():
            try:
                config = json.loads(self.config_file.read_text())
                self.PASSWORD_RULES.update(config.get("password_rules", {}))
            except Exception as e:
                logger.warning(f"Failed to load security config: {e}")
    
    async def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password against security rules."""
        errors = []
        score = 100
        
        # Check minimum length
        if len(password) < self.PASSWORD_RULES["min_length"]:
            errors.append(f"Password must be at least {self.PASSWORD_RULES['min_length']} characters long")
            score -= 20
        
        # Check uppercase
        if self.PASSWORD_RULES["require_uppercase"] and not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")
            score -= 15
        
        # Check lowercase
        if self.PASSWORD_RULES["require_lowercase"] and not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")
            score -= 15
        
        # Check digit
        if self.PASSWORD_RULES["require_digit"] and not re.search(r"\d", password):
            errors.append("Password must contain at least one digit")
            score -= 15
        
        # Check special character
        if self.PASSWORD_RULES["require_special"]:
            special_chars = self.PASSWORD_RULES["special_chars"]
            if not any(char in special_chars for char in password):
                errors.append(f"Password must contain at least one special character ({special_chars})")
                score -= 15
        
        # Check for common patterns
        common_patterns = ["password", "123456", "qwerty", "admin"]
        for pattern in common_patterns:
            if pattern in password.lower():
                errors.append(f"Password contains common pattern: {pattern}")
                score -= 20
                break
        
        # Calculate strength
        if score >= 80:
            strength = "strong"
        elif score >= 60:
            strength = "medium"
        else:
            strength = "weak"
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "score": max(0, score),
            "strength": strength
        }
    
    async def check_rate_limit(
        self,
        resource: str,
        identifier: str,
        limit: int = 100,
        window_seconds: int = 60
    ) -> Dict[str, Any]:
        """Check rate limit for a resource (in-memory only)."""
        key = f"{resource}:{identifier}"
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        
        # Clean old entries
        self._rate_limit_cache[key] = [
            timestamp for timestamp in self._rate_limit_cache[key]
            if timestamp > window_start
        ]
        
        # Check limit
        current_count = len(self._rate_limit_cache[key])
        allowed = current_count < limit
        
        if allowed:
            self._rate_limit_cache[key].append(now)
        
        return {
            "allowed": allowed,
            "current": current_count,
            "limit": limit,
            "window_seconds": window_seconds,
            "retry_after": window_seconds if not allowed else None
        }
    
    async def record_failed_auth(self, identifier: str) -> Dict[str, Any]:
        """Record failed authentication attempt."""
        self._failed_auth_attempts[identifier] += 1
        attempts = self._failed_auth_attempts[identifier]
        
        # Simple blocking after 5 failed attempts
        if attempts >= 5:
            self._blocked_ips.add(identifier)
            return {
                "blocked": True,
                "attempts": attempts,
                "reason": "Too many failed authentication attempts"
            }
        
        return {
            "blocked": False,
            "attempts": attempts,
            "remaining": 5 - attempts
        }
    
    async def clear_failed_auth(self, identifier: str):
        """Clear failed authentication attempts."""
        if identifier in self._failed_auth_attempts:
            del self._failed_auth_attempts[identifier]
        if identifier in self._blocked_ips:
            self._blocked_ips.remove(identifier)
    
    async def is_blocked(self, identifier: str) -> bool:
        """Check if an identifier is blocked."""
        return identifier in self._blocked_ips
    
    async def get_security_headers(self) -> Dict[str, str]:
        """Get recommended security headers."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
    
    async def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive data using SHA-256."""
        if salt:
            data = f"{data}{salt}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get current security status (simplified for Community)."""
        return {
            "status": "healthy",
            "blocked_count": len(self._blocked_ips),
            "rate_limit_tracked": len(self._rate_limit_cache),
            "features": {
                "password_validation": True,
                "rate_limiting": "in-memory",
                "ip_blocking": "in-memory",
                "security_alerts": False,
                "audit_logging": False,
                "compliance_validation": False
            },
            "upgrade_available": True,
            "upgrade_benefits": [
                "Persistent rate limiting",
                "Advanced IP blocking",
                "Security alerts and monitoring",
                "Compliance validation",
                "Audit logging"
            ]
        }