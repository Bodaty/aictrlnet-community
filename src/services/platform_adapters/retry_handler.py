"""Retry handler for platform integration executions"""
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Coroutine
from datetime import datetime, timedelta
from functools import wraps
import random

from schemas.platform_integration import ExecutionStatus
from .base import ExecutionResponse

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_errors: Optional[list] = None,
        retryable_status_codes: Optional[list] = None
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_errors = retryable_errors or [
            asyncio.TimeoutError,
            ConnectionError,
            ConnectionResetError,
            ConnectionAbortedError
        ]
        self.retryable_status_codes = retryable_status_codes or [
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504   # Gateway Timeout
        ]


class RetryHandler:
    """Handles retry logic for platform integrations"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt"""
        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_delay
        )
        
        if self.config.jitter:
            # Add random jitter to prevent thundering herd
            jitter = random.uniform(0, delay * 0.1)
            delay += jitter
        
        return delay
    
    def is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable"""
        for error_type in self.config.retryable_errors:
            if isinstance(error, error_type):
                return True
        
        # Check for HTTP status codes in exception
        if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            return error.response.status_code in self.config.retryable_status_codes
        
        return False
    
    def is_retryable_response(self, response: ExecutionResponse) -> bool:
        """Check if execution response indicates retry is needed"""
        # Don't retry successful executions
        if response.status == ExecutionStatus.COMPLETED:
            return False
        
        # Check for rate limiting or temporary failures
        if response.metadata:
            status_code = response.metadata.get('status_code')
            if status_code and status_code in self.config.retryable_status_codes:
                return True
        
        # Check error messages for retryable patterns
        if response.error:
            error_lower = response.error.lower()
            retryable_patterns = [
                'timeout',
                'rate limit',
                'too many requests',
                'service unavailable',
                'temporarily unavailable',
                'connection reset',
                'connection refused'
            ]
            
            for pattern in retryable_patterns:
                if pattern in error_lower:
                    return True
        
        return False
    
    async def execute_with_retry(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs
    ) -> T:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.config.max_attempts} for {func.__name__}")
                
                result = await func(*args, **kwargs)
                
                # If result is ExecutionResponse, check if retry needed
                if isinstance(result, ExecutionResponse) and self.is_retryable_response(result):
                    if attempt < self.config.max_attempts:
                        delay = self.calculate_delay(attempt)
                        logger.warning(
                            f"Execution failed with retryable error: {result.error}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        await asyncio.sleep(delay)
                        continue
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if self.is_retryable_error(e) and attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Retryable error on attempt {attempt}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Non-retryable error or max attempts reached: {str(e)}")
                    raise
        
        # If we get here, we've exhausted all retries
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"Max retry attempts ({self.config.max_attempts}) exceeded")


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """Decorator to add retry logic to async functions"""
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            config = RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter
            )
            handler = RetryHandler(config)
            return await handler.execute_with_retry(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for platform integrations"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs
    ) -> T:
        """Call function through circuit breaker"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception(
                    f"Circuit breaker is open. Service unavailable. "
                    f"Next retry after {self.recovery_timeout} seconds from last failure."
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        if self.last_failure_time is None:
            return False
        
        return datetime.utcnow() >= self.last_failure_time + timedelta(seconds=self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "closed"
        logger.info("Circuit breaker reset to closed state")
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will retry after {self.recovery_timeout} seconds."
            )
        elif self.state == "half-open":
            self.state = "open"
            logger.warning("Circuit breaker reopened after failure in half-open state")


class AdaptiveRetryHandler(RetryHandler):
    """Adaptive retry handler that adjusts based on platform behavior"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        super().__init__(config)
        self.platform_stats: Dict[str, Dict[str, Any]] = {}
    
    def update_platform_stats(
        self,
        platform: str,
        success: bool,
        response_time: Optional[float] = None,
        error_type: Optional[str] = None
    ):
        """Update statistics for a platform"""
        if platform not in self.platform_stats:
            self.platform_stats[platform] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'avg_response_time': 0,
                'error_types': {},
                'last_success': None,
                'last_failure': None
            }
        
        stats = self.platform_stats[platform]
        stats['total_calls'] += 1
        
        if success:
            stats['successful_calls'] += 1
            stats['last_success'] = datetime.utcnow()
            
            if response_time:
                # Update rolling average
                avg = stats['avg_response_time']
                stats['avg_response_time'] = (avg * (stats['successful_calls'] - 1) + response_time) / stats['successful_calls']
        else:
            stats['failed_calls'] += 1
            stats['last_failure'] = datetime.utcnow()
            
            if error_type:
                stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1
    
    def get_adaptive_config(self, platform: str) -> RetryConfig:
        """Get adaptive retry configuration based on platform statistics"""
        if platform not in self.platform_stats:
            return self.config
        
        stats = self.platform_stats[platform]
        success_rate = stats['successful_calls'] / stats['total_calls'] if stats['total_calls'] > 0 else 1.0
        
        # Adjust retry attempts based on success rate
        if success_rate < 0.5:
            # Low success rate - reduce retry attempts
            max_attempts = max(1, self.config.max_attempts - 1)
        elif success_rate > 0.9:
            # High success rate - increase retry attempts
            max_attempts = min(5, self.config.max_attempts + 1)
        else:
            max_attempts = self.config.max_attempts
        
        # Adjust delays based on average response time
        avg_response_time = stats['avg_response_time']
        if avg_response_time > 30:  # Slow platform
            initial_delay = self.config.initial_delay * 2
            max_delay = self.config.max_delay * 2
        else:
            initial_delay = self.config.initial_delay
            max_delay = self.config.max_delay
        
        return RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=self.config.exponential_base,
            jitter=self.config.jitter,
            retryable_errors=self.config.retryable_errors,
            retryable_status_codes=self.config.retryable_status_codes
        )