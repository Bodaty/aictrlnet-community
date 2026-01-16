"""Adapter system for AICtrlNet."""

from .base_adapter import BaseAdapter, AdapterConfig, AdapterCapability
from .registry import AdapterRegistry, adapter_registry
from .factory import AdapterFactory

__all__ = [
    "BaseAdapter",
    "AdapterConfig", 
    "AdapterCapability",
    "AdapterRegistry",
    "adapter_registry",
    "AdapterFactory"
]