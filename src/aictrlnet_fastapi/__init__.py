"""
AICtrlNet FastAPI Community Edition

Open source AI workflow orchestration platform.
"""

__version__ = "2.0.0"
__edition__ = "community"

# Export key components for extension by other editions
from .core.app import AICtrlNetApp, create_app
from .core.config import Settings, get_settings
from .core.database import get_db

__all__ = [
    "AICtrlNetApp",
    "create_app",
    "Settings", 
    "get_settings",
    "get_db",
    "__version__",
    "__edition__",
]