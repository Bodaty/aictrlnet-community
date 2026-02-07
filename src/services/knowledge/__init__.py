"""
Knowledge Services Module

Provides system knowledge foundation for the Intelligent Assistant.
"""

from .system_manifest_service import SystemManifestService, get_manifest_service
from .knowledge_indexer import KnowledgeIndexer
from .knowledge_retrieval_service import KnowledgeRetrievalService, KnowledgeItem
from .api_introspection_service import ApiIntrospectionService

__all__ = [
    'SystemManifestService',
    'get_manifest_service',
    'KnowledgeIndexer',
    'KnowledgeRetrievalService',
    'KnowledgeItem',
    'ApiIntrospectionService',
]