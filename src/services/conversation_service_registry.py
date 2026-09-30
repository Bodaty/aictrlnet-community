"""Which conversation service class this process runs (spec §4a).

Business cannot import Enterprise, so no endpoint can name the right class.
Each edition registers its class at app startup with its rank; the highest
rank wins regardless of registration order. Every call site that builds a
conversation service instantiates `get_conversation_service_class()`.
"""

import logging
from typing import Optional, Tuple, Type

logger = logging.getLogger(__name__)

COMMUNITY_RANK = 0
BUSINESS_RANK = 1
ENTERPRISE_RANK = 2

_registered: Optional[Tuple[int, type]] = None


def register_conversation_service_class(cls: type, rank: int) -> None:
    global _registered
    if _registered is None or rank >= _registered[0]:
        _registered = (rank, cls)
        logger.info(
            "[conversation-registry] using %s.%s (rank %d)", cls.__module__, cls.__name__, rank
        )


def get_conversation_service_class() -> Type:
    if _registered is not None:
        return _registered[1]
    from services.enhanced_conversation_manager import EnhancedConversationService

    return EnhancedConversationService
