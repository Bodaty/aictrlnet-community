"""Basic workflow security validation for Community Edition."""

import re
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels for workflow generation."""
    SAFE = "safe"
    MODERATE = "moderate"
    RISKY = "risky"
    BLOCKED = "blocked"


@dataclass
class SecurityValidation:
    """Result of security validation."""
    allowed: bool
    level: SecurityLevel
    reason: Optional[str] = None


class WorkflowSecurityService:
    """Basic security validation for workflow generation in Community Edition."""
    
    # Basic forbidden patterns
    FORBIDDEN_PATTERNS = [
        # Credential access
        r"\b(password|credential|secret|token|api[_\s]?key)\b",
        # System manipulation
        r"\b(bypass|override|disable)[_\s]?(security|auth)\b",
        # Data exfiltration
        r"\b(export|dump)[_\s]?(all|database|user)\b",
        # Code execution - use word boundaries to avoid false positives with "execution", "evaluate", etc.
        r"\bexec\s*\(|\beval\s*\(|\bsubprocess\.",
    ]
    
    # Rate limits for Community Edition
    COMMUNITY_RATE_LIMITS = {
        "requests_per_minute": 5,
        "requests_per_hour": 30,
        "requests_per_day": 100
    }
    
    def validate_prompt(self, prompt: str, user_id: str) -> SecurityValidation:
        """Validate a workflow generation prompt."""
        # Check length
        if len(prompt) > 5000:
            return SecurityValidation(
                allowed=False,
                level=SecurityLevel.BLOCKED,
                reason="Prompt exceeds maximum length (5000 characters)"
            )
        
        if len(prompt) < 10:
            return SecurityValidation(
                allowed=False,
                level=SecurityLevel.BLOCKED,
                reason="Prompt too short to generate meaningful workflow"
            )
        
        # Check forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.warning(f"Forbidden pattern detected for user {user_id}: {pattern}")
                return SecurityValidation(
                    allowed=False,
                    level=SecurityLevel.BLOCKED,
                    reason="Security risk: Forbidden pattern detected"
                )
        
        # Basic prompt injection detection
        injection_patterns = [
            r"ignore[_\s]?previous[_\s]?instructions",
            r"disregard[_\s]?system",
            r"you[_\s]?are[_\s]?now",
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return SecurityValidation(
                    allowed=False,
                    level=SecurityLevel.BLOCKED,
                    reason="Potential prompt injection detected"
                )
        
        return SecurityValidation(
            allowed=True,
            level=SecurityLevel.SAFE
        )
    
    def validate_generated_workflow(
        self, 
        workflow: Dict[str, Any], 
        user_id: str
    ) -> SecurityValidation:
        """Validate a generated workflow for security risks."""
        workflow_str = str(workflow).lower()
        
        # Check for risky operations
        risky_operations = [
            "database_write",
            "file_write",
            "api_call",
            "external_request"
        ]
        
        risk_count = sum(1 for op in risky_operations if op in workflow_str)
        
        if risk_count > 3:
            return SecurityValidation(
                allowed=False,
                level=SecurityLevel.BLOCKED,
                reason="Workflow contains too many risky operations"
            )
        elif risk_count > 1:
            level = SecurityLevel.RISKY
        elif risk_count > 0:
            level = SecurityLevel.MODERATE
        else:
            level = SecurityLevel.SAFE
        
        return SecurityValidation(
            allowed=True,
            level=level
        )