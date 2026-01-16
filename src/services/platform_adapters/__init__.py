"""Platform adapters package"""
from .base import (
    BasePlatformAdapter,
    ExecutionRequest,
    ExecutionResponse,
    ExecutionStatus,
    HealthCheckResult
)
from .registry import (
    PlatformAdapterRegistry,
    PlatformAdapterService,
    register_adapter
)

# Import adapters to trigger registration
from . import n8n_adapter
from . import zapier_adapter
from . import make_adapter
from . import power_automate_adapter
from . import ifttt_adapter

__all__ = [
    "BasePlatformAdapter",
    "ExecutionRequest",
    "ExecutionResponse",
    "ExecutionStatus",
    "HealthCheckResult",
    "PlatformAdapterRegistry",
    "PlatformAdapterService",
    "register_adapter"
]