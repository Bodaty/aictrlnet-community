"""Resilient MCP Executor - Community Edition.

Provides reliable MCP tool execution with retry logic, circuit breakers,
and graceful fallback handling.

This is the base implementation for Community edition. Business and
Enterprise editions extend this with ML-enhanced features and compliance
controls.
"""

from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

from services.mcp_unified import UnifiedMCPService

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Retry strategy options."""
    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class ExecutionResult:
    """Result of an MCP tool execution."""

    def __init__(
        self,
        success: bool,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        attempts: int = 1,
        execution_time_ms: float = 0,
        used_fallback: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize execution result.

        Args:
            success: Whether execution succeeded
            result: The execution result if successful
            error: Error message if failed
            attempts: Number of attempts made
            execution_time_ms: Total execution time in milliseconds
            used_fallback: Whether a fallback was used
            metadata: Additional metadata about the execution
        """
        self.success = success
        self.result = result
        self.error = error
        self.attempts = attempts
        self.execution_time_ms = execution_time_ms
        self.used_fallback = used_fallback
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "attempts": self.attempts,
            "execution_time_ms": self.execution_time_ms,
            "used_fallback": self.used_fallback,
            "metadata": self.metadata
        }


class CircuitBreaker:
    """Circuit breaker for MCP server connections.

    Prevents cascading failures by stopping requests to failing servers.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        half_open_max_calls: int = 3
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout_seconds: Time to wait before half-open
            half_open_max_calls: Calls to allow in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self._last_failure_time:
            return True

        elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout_seconds

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                # Recovered - close the circuit
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info("Circuit breaker closed - service recovered")
        else:
            # Reset failure count on success in closed state
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()
        self._success_count = 0

        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery attempt - reopen
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened - recovery failed")
        elif self._failure_count >= self.failure_threshold:
            # Too many failures - open the circuit
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures"
            )

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        state = self.state  # This updates state if recovery time passed

        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            return self._half_open_calls <= self.half_open_max_calls
        else:  # OPEN
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "recovery_timeout_seconds": self.recovery_timeout_seconds
        }


class ResilientMCPExecutor:
    """Resilient executor for MCP tools.

    Provides retry logic, circuit breakers, and fallback handling
    for reliable MCP tool execution.

    Community Edition Features:
    - Fixed retry with configurable attempts
    - Basic circuit breaker per server
    - Simple fallback support
    """

    def __init__(
        self,
        mcp_service: UnifiedMCPService,
        default_retry_strategy: RetryStrategy = RetryStrategy.FIXED,
        default_max_retries: int = 3,
        default_retry_delay_ms: int = 1000,
        circuit_breaker_enabled: bool = True
    ):
        """Initialize the resilient executor.

        Args:
            mcp_service: The MCP service for tool execution
            default_retry_strategy: Default retry strategy
            default_max_retries: Default maximum retry attempts
            default_retry_delay_ms: Default delay between retries
            circuit_breaker_enabled: Whether to use circuit breakers
        """
        self.mcp_service = mcp_service
        self.default_retry_strategy = default_retry_strategy
        self.default_max_retries = default_max_retries
        self.default_retry_delay_ms = default_retry_delay_ms
        self.circuit_breaker_enabled = circuit_breaker_enabled

        # Circuit breakers per server
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Fallback handlers
        self._fallback_handlers: Dict[str, Callable] = {}

        # Execution statistics
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "retried_executions": 0,
            "fallback_executions": 0,
            "circuit_breaker_rejections": 0
        }

    def _get_circuit_breaker(self, server_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a server."""
        if server_name not in self._circuit_breakers:
            self._circuit_breakers[server_name] = CircuitBreaker()
        return self._circuit_breakers[server_name]

    def register_fallback(
        self,
        tool_name: str,
        fallback_handler: Callable
    ) -> None:
        """Register a fallback handler for a tool.

        Args:
            tool_name: Name of the tool
            fallback_handler: Async callable to handle fallback
        """
        self._fallback_handlers[tool_name] = fallback_handler
        logger.info(f"Registered fallback handler for tool: {tool_name}")

    def _calculate_delay(
        self,
        attempt: int,
        strategy: RetryStrategy,
        base_delay_ms: int
    ) -> float:
        """Calculate delay for retry attempt.

        Args:
            attempt: Current attempt number (1-indexed)
            strategy: Retry strategy to use
            base_delay_ms: Base delay in milliseconds

        Returns:
            Delay in seconds
        """
        if strategy == RetryStrategy.NONE:
            return 0
        elif strategy == RetryStrategy.FIXED:
            return base_delay_ms / 1000
        elif strategy == RetryStrategy.EXPONENTIAL:
            # 2^attempt * base_delay, capped at 30 seconds
            delay = min((2 ** attempt) * base_delay_ms / 1000, 30)
            return delay
        elif strategy == RetryStrategy.LINEAR:
            # attempt * base_delay, capped at 30 seconds
            delay = min(attempt * base_delay_ms / 1000, 30)
            return delay
        else:
            return base_delay_ms / 1000

    async def execute(
        self,
        connection_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        max_retries: Optional[int] = None,
        retry_delay_ms: Optional[int] = None,
        timeout_ms: Optional[int] = None
    ) -> ExecutionResult:
        """Execute an MCP tool with resilience features.

        Args:
            connection_id: MCP server connection ID
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            server_name: Server name for circuit breaker (optional)
            retry_strategy: Override default retry strategy
            max_retries: Override default max retries
            retry_delay_ms: Override default retry delay
            timeout_ms: Execution timeout in milliseconds

        Returns:
            ExecutionResult with outcome details
        """
        start_time = datetime.utcnow()
        self._stats["total_executions"] += 1

        # Use server_name or connection_id for circuit breaker
        circuit_key = server_name or connection_id

        # Check circuit breaker
        if self.circuit_breaker_enabled:
            circuit_breaker = self._get_circuit_breaker(circuit_key)
            if not circuit_breaker.can_execute():
                self._stats["circuit_breaker_rejections"] += 1
                logger.warning(
                    f"Circuit breaker open for {circuit_key}, rejecting request"
                )

                # Try fallback if available
                if tool_name in self._fallback_handlers:
                    return await self._execute_fallback(
                        tool_name, arguments, start_time,
                        "Circuit breaker open"
                    )

                return ExecutionResult(
                    success=False,
                    error=f"Circuit breaker open for server {circuit_key}",
                    attempts=0,
                    execution_time_ms=self._elapsed_ms(start_time),
                    metadata={"circuit_breaker": circuit_breaker.get_status()}
                )

        # Resolve retry parameters
        strategy = retry_strategy or self.default_retry_strategy
        max_attempts = (max_retries or self.default_max_retries) + 1
        delay_ms = retry_delay_ms or self.default_retry_delay_ms

        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            try:
                # Execute the tool
                result = await self._execute_with_timeout(
                    connection_id, tool_name, arguments, timeout_ms
                )

                # Success
                if self.circuit_breaker_enabled:
                    circuit_breaker.record_success()

                self._stats["successful_executions"] += 1
                if attempt > 1:
                    self._stats["retried_executions"] += 1

                return ExecutionResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    execution_time_ms=self._elapsed_ms(start_time),
                    metadata={
                        "retry_strategy": strategy.value,
                        "server": circuit_key
                    }
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"MCP tool execution failed (attempt {attempt}/{max_attempts}): {e}"
                )

                if self.circuit_breaker_enabled:
                    circuit_breaker.record_failure()

                # Check if we should retry
                if attempt < max_attempts and strategy != RetryStrategy.NONE:
                    delay = self._calculate_delay(attempt, strategy, delay_ms)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)

        # All retries exhausted
        self._stats["failed_executions"] += 1

        # Try fallback if available
        if tool_name in self._fallback_handlers:
            return await self._execute_fallback(
                tool_name, arguments, start_time, last_error, max_attempts
            )

        return ExecutionResult(
            success=False,
            error=last_error,
            attempts=max_attempts,
            execution_time_ms=self._elapsed_ms(start_time),
            metadata={
                "retry_strategy": strategy.value,
                "server": circuit_key
            }
        )

    async def _execute_with_timeout(
        self,
        connection_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_ms: Optional[int]
    ) -> Dict[str, Any]:
        """Execute tool with optional timeout.

        Args:
            connection_id: MCP connection ID
            tool_name: Tool name
            arguments: Tool arguments
            timeout_ms: Timeout in milliseconds

        Returns:
            Execution result from MCP service
        """
        if timeout_ms:
            try:
                return await asyncio.wait_for(
                    self.mcp_service.execute_tool(
                        connection_id=connection_id,
                        tool_name=tool_name,
                        arguments=arguments
                    ),
                    timeout=timeout_ms / 1000
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Tool execution timed out after {timeout_ms}ms"
                )
        else:
            return await self.mcp_service.execute_tool(
                connection_id=connection_id,
                tool_name=tool_name,
                arguments=arguments
            )

    async def _execute_fallback(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        start_time: datetime,
        original_error: Optional[str],
        attempts: int = 0
    ) -> ExecutionResult:
        """Execute fallback handler.

        Args:
            tool_name: Tool name
            arguments: Original arguments
            start_time: When execution started
            original_error: The error that triggered fallback
            attempts: Number of attempts before fallback

        Returns:
            ExecutionResult from fallback
        """
        try:
            handler = self._fallback_handlers[tool_name]
            result = await handler(arguments)

            self._stats["fallback_executions"] += 1

            return ExecutionResult(
                success=True,
                result=result,
                attempts=attempts,
                execution_time_ms=self._elapsed_ms(start_time),
                used_fallback=True,
                metadata={
                    "original_error": original_error,
                    "fallback_tool": tool_name
                }
            )
        except Exception as e:
            logger.error(f"Fallback handler failed for {tool_name}: {e}")
            return ExecutionResult(
                success=False,
                error=f"Fallback failed: {str(e)}",
                attempts=attempts,
                execution_time_ms=self._elapsed_ms(start_time),
                used_fallback=True,
                metadata={
                    "original_error": original_error,
                    "fallback_error": str(e)
                }
            )

    def _elapsed_ms(self, start_time: datetime) -> float:
        """Calculate elapsed time in milliseconds."""
        return (datetime.utcnow() - start_time).total_seconds() * 1000

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["successful_executions"] / self._stats["total_executions"]
                if self._stats["total_executions"] > 0 else 0
            ),
            "circuit_breakers": {
                name: cb.get_status()
                for name, cb in self._circuit_breakers.items()
            }
        }

    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "retried_executions": 0,
            "fallback_executions": 0,
            "circuit_breaker_rejections": 0
        }

    def reset_circuit_breaker(self, server_name: str) -> bool:
        """Manually reset a circuit breaker.

        Args:
            server_name: Server to reset

        Returns:
            True if reset, False if not found
        """
        if server_name in self._circuit_breakers:
            self._circuit_breakers[server_name] = CircuitBreaker()
            logger.info(f"Circuit breaker reset for {server_name}")
            return True
        return False

    def get_circuit_breaker_status(
        self,
        server_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get circuit breaker status.

        Args:
            server_name: Specific server, or all if None

        Returns:
            Status dict for server(s)
        """
        if server_name:
            if server_name in self._circuit_breakers:
                return {server_name: self._circuit_breakers[server_name].get_status()}
            return {}

        return {
            name: cb.get_status()
            for name, cb in self._circuit_breakers.items()
        }
