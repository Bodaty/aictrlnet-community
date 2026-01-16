"""Custom exceptions for the application."""

from typing import Any, Dict, Optional


class BaseAPIException(Exception):
    """Base exception for all API exceptions."""
    
    def __init__(
        self,
        message: str = "An error occurred",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(BaseAPIException):
    """Raised when validation fails."""
    
    def __init__(self, message: str = "Validation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 422, details)


class NotFoundError(BaseAPIException):
    """Raised when a resource is not found."""
    
    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 404, details)


class UnauthorizedError(BaseAPIException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Unauthorized", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 401, details)


class ForbiddenError(BaseAPIException):
    """Raised when user lacks permissions."""
    
    def __init__(self, message: str = "Forbidden", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 403, details)


class ConflictError(BaseAPIException):
    """Raised when there's a conflict with existing data."""
    
    def __init__(self, message: str = "Conflict", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 409, details)


class BadRequestError(BaseAPIException):
    """Raised when the request is malformed."""
    
    def __init__(self, message: str = "Bad request", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 400, details)


class InternalServerError(BaseAPIException):
    """Raised when an internal server error occurs."""
    
    def __init__(self, message: str = "Internal server error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 500, details)


class ServiceUnavailableError(BaseAPIException):
    """Raised when a service is temporarily unavailable."""
    
    def __init__(self, message: str = "Service unavailable", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 503, details)


class FeatureNotAvailableError(BaseAPIException):
    """Raised when a feature is not available in the current edition."""
    
    def __init__(self, message: str = "Feature not available in this edition", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 403, details)


class AdapterError(BaseAPIException):
    """Raised when an adapter operation fails."""

    def __init__(self, message: str = "Adapter operation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 500, details)


class ToolExecutionError(BaseAPIException):
    """Raised when a tool execution fails.

    Used by the AICtrlNet Intelligent Assistant (v4) tool calling system
    when tool invocation encounters an error.
    """

    def __init__(
        self,
        tool_name: str,
        message: str = "Tool execution failed",
        details: Optional[Dict[str, Any]] = None,
        recovery_strategy: Optional[str] = None,
        fallback_tool: Optional[str] = None
    ):
        full_details = details or {}
        full_details["tool_name"] = tool_name
        if recovery_strategy:
            full_details["recovery_strategy"] = recovery_strategy
        if fallback_tool:
            full_details["fallback_tool"] = fallback_tool

        super().__init__(f"Tool '{tool_name}' failed: {message}", 500, full_details)
        self.tool_name = tool_name
        self.recovery_strategy = recovery_strategy
        self.fallback_tool = fallback_tool