"""Spec §4a: the conversation service registry resolves the highest-ranked class."""

from services import conversation_service_registry as registry
from services.enhanced_conversation_manager import EnhancedConversationService


class _Business(EnhancedConversationService):
    pass


class _Enterprise(_Business):
    pass


def test_default_is_the_community_class(monkeypatch):
    monkeypatch.setattr(registry, "_registered", None)
    assert registry.get_conversation_service_class() is EnhancedConversationService


def test_highest_rank_wins_in_any_order(monkeypatch):
    monkeypatch.setattr(registry, "_registered", None)
    registry.register_conversation_service_class(_Enterprise, registry.ENTERPRISE_RANK)
    registry.register_conversation_service_class(_Business, registry.BUSINESS_RANK)
    assert registry.get_conversation_service_class() is _Enterprise

    monkeypatch.setattr(registry, "_registered", None)
    registry.register_conversation_service_class(_Business, registry.BUSINESS_RANK)
    registry.register_conversation_service_class(_Enterprise, registry.ENTERPRISE_RANK)
    assert registry.get_conversation_service_class() is _Enterprise
