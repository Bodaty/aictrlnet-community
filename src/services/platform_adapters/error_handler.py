"""Error handling for platform integrations"""
import logging
from typing import Dict, Any, Optional, Type
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PlatformErrorType(str, Enum):
    """Types of platform integration errors"""
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    CONFIGURATION = "configuration"
    PLATFORM_ERROR = "platform_error"
    UNKNOWN = "unknown"


class PlatformError(Exception):
    """Base exception for platform integration errors"""
    
    def __init__(
        self,
        message: str,
        error_type: PlatformErrorType = PlatformErrorType.UNKNOWN,
        platform: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None,
        is_retryable: bool = False
    ):
        super().__init__(message)
        self.error_type = error_type
        self.platform = platform
        self.details = details or {}
        self.retry_after = retry_after
        self.is_retryable = is_retryable
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/storage"""
        return {
            "message": str(self),
            "error_type": self.error_type.value,
            "platform": self.platform,
            "details": self.details,
            "retry_after": self.retry_after,
            "is_retryable": self.is_retryable,
            "timestamp": self.timestamp.isoformat()
        }


class AuthenticationError(PlatformError):
    """Authentication/authorization errors"""
    def __init__(self, message: str, platform: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_type=PlatformErrorType.AUTHENTICATION,
            platform=platform,
            is_retryable=False,
            **kwargs
        )


class RateLimitError(PlatformError):
    """Rate limiting errors"""
    def __init__(
        self,
        message: str,
        platform: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_type=PlatformErrorType.RATE_LIMIT,
            platform=platform,
            retry_after=retry_after,
            is_retryable=True,
            **kwargs
        )


class NetworkError(PlatformError):
    """Network connectivity errors"""
    def __init__(self, message: str, platform: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_type=PlatformErrorType.NETWORK,
            platform=platform,
            is_retryable=True,
            **kwargs
        )


class TimeoutError(PlatformError):
    """Execution timeout errors"""
    def __init__(self, message: str, platform: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_type=PlatformErrorType.TIMEOUT,
            platform=platform,
            is_retryable=True,
            **kwargs
        )


class ValidationError(PlatformError):
    """Input validation errors"""
    def __init__(self, message: str, platform: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_type=PlatformErrorType.VALIDATION,
            platform=platform,
            is_retryable=False,
            **kwargs
        )


class ConfigurationError(PlatformError):
    """Platform configuration errors"""
    def __init__(self, message: str, platform: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_type=PlatformErrorType.CONFIGURATION,
            platform=platform,
            is_retryable=False,
            **kwargs
        )


class PlatformErrorHandler:
    """Handles and classifies platform integration errors"""
    
    # Map of error patterns to error types
    ERROR_PATTERNS = {
        PlatformErrorType.AUTHENTICATION: [
            "unauthorized",
            "authentication failed",
            "invalid credentials",
            "access denied",
            "forbidden",
            "invalid api key",
            "invalid token",
            "expired token"
        ],
        PlatformErrorType.RATE_LIMIT: [
            "rate limit",
            "too many requests",
            "quota exceeded",
            "throttled",
            "429"
        ],
        PlatformErrorType.NETWORK: [
            "connection refused",
            "connection reset",
            "network unreachable",
            "dns resolution failed",
            "connection timeout"
        ],
        PlatformErrorType.TIMEOUT: [
            "timeout",
            "timed out",
            "deadline exceeded",
            "request timeout"
        ],
        PlatformErrorType.VALIDATION: [
            "invalid input",
            "validation failed",
            "invalid format",
            "missing required",
            "invalid parameter"
        ],
        PlatformErrorType.NOT_FOUND: [
            "not found",
            "404",
            "resource not found",
            "workflow not found",
            "does not exist"
        ],
        PlatformErrorType.PERMISSION: [
            "permission denied",
            "insufficient permissions",
            "access restricted",
            "not authorized"
        ]
    }
    
    # Map of HTTP status codes to error types
    STATUS_CODE_MAP = {
        401: PlatformErrorType.AUTHENTICATION,
        403: PlatformErrorType.PERMISSION,
        404: PlatformErrorType.NOT_FOUND,
        429: PlatformErrorType.RATE_LIMIT,
        408: PlatformErrorType.TIMEOUT,
        422: PlatformErrorType.VALIDATION,
        500: PlatformErrorType.PLATFORM_ERROR,
        502: PlatformErrorType.NETWORK,
        503: PlatformErrorType.PLATFORM_ERROR,
        504: PlatformErrorType.TIMEOUT
    }
    
    @classmethod
    def classify_error(
        cls,
        error: Exception,
        platform: Optional[str] = None,
        status_code: Optional[int] = None
    ) -> PlatformError:
        """Classify a generic exception into a specific platform error"""
        
        # If already a PlatformError, return as-is
        if isinstance(error, PlatformError):
            return error
        
        error_message = str(error).lower()
        error_type = PlatformErrorType.UNKNOWN
        is_retryable = False
        retry_after = None
        details = {}
        
        # First check status code if available
        if status_code and status_code in cls.STATUS_CODE_MAP:
            error_type = cls.STATUS_CODE_MAP[status_code]
        else:
            # Check error message patterns
            for err_type, patterns in cls.ERROR_PATTERNS.items():
                for pattern in patterns:
                    if pattern in error_message:
                        error_type = err_type
                        break
                if error_type != PlatformErrorType.UNKNOWN:
                    break
        
        # Determine if retryable based on error type
        retryable_types = {
            PlatformErrorType.RATE_LIMIT,
            PlatformErrorType.NETWORK,
            PlatformErrorType.TIMEOUT,
            PlatformErrorType.PLATFORM_ERROR
        }
        is_retryable = error_type in retryable_types
        
        # Extract retry-after header for rate limits
        if hasattr(error, 'response') and error.response:
            if hasattr(error.response, 'headers'):
                retry_after_header = error.response.headers.get('Retry-After')
                if retry_after_header:
                    try:
                        retry_after = int(retry_after_header)
                    except ValueError:
                        # Might be a date string
                        pass
                
                # Add response details
                details['status_code'] = getattr(error.response, 'status_code', None)
                details['response_text'] = getattr(error.response, 'text', None)
        
        # Create appropriate error class
        error_classes = {
            PlatformErrorType.AUTHENTICATION: AuthenticationError,
            PlatformErrorType.RATE_LIMIT: RateLimitError,
            PlatformErrorType.NETWORK: NetworkError,
            PlatformErrorType.TIMEOUT: TimeoutError,
            PlatformErrorType.VALIDATION: ValidationError,
            PlatformErrorType.CONFIGURATION: ConfigurationError
        }
        
        error_class = error_classes.get(error_type, PlatformError)
        
        return error_class(
            message=str(error),
            platform=platform,
            details=details,
            retry_after=retry_after,
            is_retryable=is_retryable
        )
    
    @classmethod
    def handle_error(
        cls,
        error: Exception,
        platform: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PlatformError:
        """Handle an error and log it appropriately"""
        platform_error = cls.classify_error(error, platform)
        
        # Log the error with appropriate level
        log_context = {
            "platform": platform,
            "error_type": platform_error.error_type.value,
            "is_retryable": platform_error.is_retryable,
            "context": context or {}
        }
        
        if platform_error.error_type in [
            PlatformErrorType.AUTHENTICATION,
            PlatformErrorType.CONFIGURATION,
            PlatformErrorType.PERMISSION
        ]:
            logger.error(f"Platform error: {platform_error}", extra=log_context)
        elif platform_error.is_retryable:
            logger.warning(f"Retryable platform error: {platform_error}", extra=log_context)
        else:
            logger.error(f"Non-retryable platform error: {platform_error}", extra=log_context)
        
        return platform_error
    
    @staticmethod
    def create_error_response(
        error: PlatformError,
        execution_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            "success": False,
            "error": {
                "message": str(error),
                "type": error.error_type.value,
                "platform": error.platform,
                "is_retryable": error.is_retryable,
                "retry_after": error.retry_after,
                "details": error.details,
                "timestamp": error.timestamp.isoformat()
            },
            "execution_id": execution_id
        }


class ErrorRecoveryStrategy:
    """Strategies for recovering from platform errors"""
    
    @staticmethod
    async def handle_authentication_error(
        error: AuthenticationError,
        credentials: Dict[str, Any],
        platform: str
    ) -> Optional[Dict[str, Any]]:
        """Handle authentication errors - might refresh tokens"""
        logger.info(f"Attempting to recover from authentication error for {platform}")
        
        # Check if we have refresh token
        if "refresh_token" in credentials:
            # Platform-specific token refresh logic would go here
            logger.info("Refresh token available, attempting refresh")
            # This would call platform-specific refresh endpoint
            return None
        
        # Can't recover without refresh capability
        logger.warning("No refresh token available, cannot recover")
        return None
    
    @staticmethod
    async def handle_rate_limit_error(
        error: RateLimitError,
        platform: str
    ) -> Dict[str, Any]:
        """Handle rate limit errors - calculate backoff"""
        retry_after = error.retry_after or 60  # Default to 60 seconds
        
        logger.info(
            f"Rate limit hit for {platform}. "
            f"Retry after {retry_after} seconds"
        )
        
        return {
            "action": "retry",
            "delay": retry_after,
            "reason": "rate_limit"
        }
    
    @staticmethod
    async def handle_network_error(
        error: NetworkError,
        attempt: int
    ) -> Dict[str, Any]:
        """Handle network errors - exponential backoff"""
        base_delay = 1
        max_delay = 300
        delay = min(base_delay * (2 ** attempt), max_delay)
        
        logger.info(f"Network error on attempt {attempt}. Retry after {delay} seconds")
        
        return {
            "action": "retry",
            "delay": delay,
            "reason": "network_error"
        }