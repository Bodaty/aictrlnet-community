"""Service layer for business logic.

This package is extended by the Business and Enterprise editions via
``pkgutil.extend_path`` so that submodules defined in
``editions/business/src/services/`` and ``editions/enterprise/src/services/``
are importable as ``services.X`` alongside the Community submodules, regardless
of PYTHONPATH ordering.
"""

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from .task import TaskService
from .workflow import WorkflowService
from .workflow_template_service import WorkflowTemplateService
from .control_plane import ControlPlaneService
from .memory import MemoryService
from .state import StateService
from .validation import ValidationService
from .rbac_basic import BasicRBACService as RBACService
from .security_basic import BasicSecurityService as SecurityService
from .basic_resource_pool import BasicResourcePoolService
from .api_key_service import APIKeyService
from .webhook_service import WebhookService
# v4: Tool Dispatcher for Intelligent Assistant
from .tool_dispatcher import ToolDispatcher, CORE_TOOLS, Edition
from .tool_aware_conversation import ToolAwareConversationService

__all__ = [
    "TaskService",
    "WorkflowService",
    "WorkflowTemplateService",
    "ControlPlaneService",
    "MemoryService",
    "StateService",
    "ValidationService",
    "RBACService",
    "SecurityService",
    "BasicResourcePoolService",
    "APIKeyService",
    "WebhookService",
    # v4 Tool Dispatcher
    "ToolDispatcher",
    "CORE_TOOLS",
    "Edition",
    "ToolAwareConversationService",
]