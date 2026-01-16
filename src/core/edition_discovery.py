"""
Dynamic Edition Feature Discovery for AICtrlNet.

This module automatically discovers which features belong to which edition
by analyzing the actual API endpoints and module structure.
"""

import os
import importlib
import inspect
from typing import Dict, List, Set, Optional, Any
from enum import Enum
from pathlib import Path
from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)


class Edition(str, Enum):
    """AICtrlNet editions."""
    COMMUNITY = "community"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class EditionFeatureDiscovery:
    """
    Dynamically discovers features based on actual API endpoints and modules.
    This avoids manual maintenance of feature lists.
    """
    
    def __init__(self):
        self._feature_cache = None
        self._endpoint_cache = None
        
    def discover_features(self, force_refresh: bool = False) -> Dict[Edition, Dict[str, List[str]]]:
        """
        Discover all features by analyzing the codebase structure.
        Features are discovered from:
        1. API endpoint paths and routers
        2. Module organization (what's imported where)
        3. Model definitions
        4. Service classes
        """
        if self._feature_cache and not force_refresh:
            return self._feature_cache
            
        features = {
            Edition.COMMUNITY: {},
            Edition.BUSINESS: {},
            Edition.ENTERPRISE: {}
        }
        
        # Discover from API routers
        self._discover_from_routers(features)
        
        # Discover from models
        self._discover_from_models(features)
        
        # Discover from services
        self._discover_from_services(features)
        
        self._feature_cache = features
        return features
    
    def _discover_from_routers(self, features: Dict[Edition, Dict[str, List[str]]]):
        """Discover features from API router definitions."""
        
        # Check which routers are imported in each edition
        edition_routers = {
            Edition.COMMUNITY: self._get_community_routers(),
            Edition.BUSINESS: self._get_business_routers(),
            Edition.ENTERPRISE: self._get_enterprise_routers()
        }
        
        for edition, routers in edition_routers.items():
            features[edition]["endpoints"] = []
            for router_info in routers:
                features[edition]["endpoints"].extend(router_info["endpoints"])
                
    def _get_community_routers(self) -> List[Dict[str, Any]]:
        """Get routers defined in Community edition."""
        routers = []
        
        try:
            # Import the community router module
            from api.v1.community_router import api_router
            
            # Extract endpoint information
            for route in api_router.routes:
                if hasattr(route, 'path'):
                    endpoint_info = {
                        "path": route.path,
                        "methods": list(route.methods) if hasattr(route, 'methods') else [],
                        "name": route.name if hasattr(route, 'name') else None,
                        "tags": route.tags if hasattr(route, 'tags') else []
                    }
                    
                    # Categorize by endpoint prefix
                    if "/tasks" in route.path:
                        routers.append({
                            "category": "task_management",
                            "endpoints": [endpoint_info]
                        })
                    elif "/workflows" in route.path:
                        routers.append({
                            "category": "workflow_management", 
                            "endpoints": [endpoint_info]
                        })
                    elif "/nlp" in route.path:
                        routers.append({
                            "category": "nlp_processing",
                            "endpoints": [endpoint_info]
                        })
                    elif "/conversation" in route.path:
                        routers.append({
                            "category": "multi_turn_conversation",
                            "endpoints": [endpoint_info]
                        })
                    elif "/mcp" in route.path:
                        routers.append({
                            "category": "mcp_integration",
                            "endpoints": [endpoint_info]
                        })
                    elif "/iam" in route.path:
                        routers.append({
                            "category": "internal_agent_messaging",
                            "endpoints": [endpoint_info]
                        })
                    elif "/quality" in route.path or "/data-quality" in route.path:
                        routers.append({
                            "category": "data_quality",
                            "endpoints": [endpoint_info]
                        })
                        
        except ImportError as e:
            logger.warning(f"Could not import community router: {e}")
            
        return routers
    
    def _get_business_routers(self) -> List[Dict[str, Any]]:
        """Get routers defined in Business edition (includes Community)."""
        routers = self._get_community_routers()  # Accretive model
        
        # Business-specific routers would be discovered from business edition repo
        # For now, we know these from the architecture:
        business_specific = [
            {"category": "ai_governance", "endpoints": [
                {"path": "/api/v1/ai-governance", "methods": ["GET", "POST"]},
                {"path": "/api/v1/ml-governance", "methods": ["GET", "POST"]},
                {"path": "/api/v1/agp", "methods": ["GET", "POST", "PUT", "DELETE"]}
            ]},
            {"category": "pod_orchestration", "endpoints": [
                {"path": "/api/v1/pods", "methods": ["GET", "POST", "PUT", "DELETE"]},
                {"path": "/api/v1/swarms", "methods": ["GET", "POST"]}
            ]},
            {"category": "advanced_analytics", "endpoints": [
                {"path": "/api/v1/analytics", "methods": ["GET", "POST"]},
                {"path": "/api/v1/metrics", "methods": ["GET"]}
            ]}
        ]
        
        routers.extend(business_specific)
        return routers
    
    def _get_enterprise_routers(self) -> List[Dict[str, Any]]:
        """Get routers defined in Enterprise edition (includes Business)."""
        routers = self._get_business_routers()  # Accretive model
        
        # Enterprise-specific routers
        enterprise_specific = [
            {"category": "multi_tenancy", "endpoints": [
                {"path": "/api/v1/tenants", "methods": ["GET", "POST", "PUT", "DELETE"]},
                {"path": "/api/v1/tenant-switch", "methods": ["POST"]}
            ]},
            {"category": "federation", "endpoints": [
                {"path": "/api/v1/federation", "methods": ["GET", "POST"]},
                {"path": "/api/v1/federation/partners", "methods": ["GET", "POST", "DELETE"]}
            ]},
            {"category": "geographic_routing", "endpoints": [
                {"path": "/api/v1/regions", "methods": ["GET", "POST"]},
                {"path": "/api/v1/routing", "methods": ["GET", "PUT"]}
            ]}
        ]
        
        routers.extend(enterprise_specific)
        return routers
    
    def _discover_from_models(self, features: Dict[Edition, Dict[str, List[str]]]):
        """Discover features from model definitions."""
        
        # Analyze which models exist in each edition's models directory
        try:
            from models import __all__ as community_models
            features[Edition.COMMUNITY]["models"] = [
                model for model in community_models 
                if not model.startswith("_")
            ]
        except ImportError:
            features[Edition.COMMUNITY]["models"] = []
            
    def _discover_from_services(self, features: Dict[Edition, Dict[str, List[str]]]):
        """Discover features from service classes."""
        
        # Check which services exist
        services_path = Path("services")
        if services_path.exists():
            for service_file in services_path.glob("*.py"):
                if not service_file.name.startswith("_"):
                    service_name = service_file.stem
                    
                    # Categorize services by edition based on naming/imports
                    if any(kw in service_name for kw in ["ai_governance", "ml_governance", "agp"]):
                        features[Edition.BUSINESS].setdefault("services", []).append(service_name)
                    elif any(kw in service_name for kw in ["tenant", "federation", "geographic"]):
                        features[Edition.ENTERPRISE].setdefault("services", []).append(service_name)
                    else:
                        features[Edition.COMMUNITY].setdefault("services", []).append(service_name)
    
    def get_feature_edition(self, feature_name: str) -> Optional[Edition]:
        """
        Determine which edition a feature belongs to by checking where it's defined.
        """
        features = self.discover_features()
        
        # Check each edition (in order) to find minimum required edition
        for edition in [Edition.COMMUNITY, Edition.BUSINESS, Edition.ENTERPRISE]:
            edition_features = features[edition]
            for category, feature_list in edition_features.items():
                if any(feature_name in str(f) for f in feature_list):
                    return edition
        
        return None
    
    def is_feature_available(self, feature_name: str, user_edition: Edition) -> bool:
        """
        Check if a feature is available in the user's edition.
        Uses the accretive model.
        """
        feature_edition = self.get_feature_edition(feature_name)
        if not feature_edition:
            return False
        
        edition_hierarchy = {
            Edition.COMMUNITY: 0,
            Edition.BUSINESS: 1,
            Edition.ENTERPRISE: 2
        }
        
        return edition_hierarchy[user_edition] >= edition_hierarchy[feature_edition]
    
    def get_available_endpoints(self, user_edition: Edition) -> List[str]:
        """
        Get all API endpoints available to a user based on their edition.
        """
        features = self.discover_features()
        endpoints = []
        
        # Community endpoints available to all
        if "endpoints" in features[Edition.COMMUNITY]:
            for router_info in features[Edition.COMMUNITY]["endpoints"]:
                if isinstance(router_info, dict) and "endpoints" in router_info:
                    for ep in router_info["endpoints"]:
                        if isinstance(ep, dict) and "path" in ep:
                            endpoints.append(ep["path"])
        
        # Business endpoints if Business or Enterprise
        if user_edition in [Edition.BUSINESS, Edition.ENTERPRISE]:
            if "endpoints" in features[Edition.BUSINESS]:
                for router_info in features[Edition.BUSINESS]["endpoints"]:
                    if isinstance(router_info, dict) and "endpoints" in router_info:
                        for ep in router_info["endpoints"]:
                            if isinstance(ep, dict) and "path" in ep:
                                endpoints.append(ep["path"])
        
        # Enterprise endpoints if Enterprise
        if user_edition == Edition.ENTERPRISE:
            if "endpoints" in features[Edition.ENTERPRISE]:
                for router_info in features[Edition.ENTERPRISE]["endpoints"]:
                    if isinstance(router_info, dict) and "endpoints" in router_info:
                        for ep in router_info["endpoints"]:
                            if isinstance(ep, dict) and "path" in ep:
                                endpoints.append(ep["path"])
        
        return list(set(endpoints))  # Remove duplicates
    
    def detect_user_intent_edition(self, intent: str, user_edition: Edition) -> Dict[str, Any]:
        """
        Analyze user intent to determine if requested feature is available in their edition.
        Returns availability info and upgrade suggestions if needed.
        """
        intent_lower = intent.lower()
        
        # Keywords that suggest specific edition features
        edition_keywords = {
            Edition.BUSINESS: [
                "ai governance", "ml governance", "risk assessment", 
                "compliance", "pod", "swarm", "advanced analytics",
                "agp", "policy", "evaluation"
            ],
            Edition.ENTERPRISE: [
                "multi-tenant", "tenant", "federation", "cross-region",
                "geographic", "enterprise", "sso", "sox", "hipaa", "gdpr"
            ]
        }
        
        # Check if intent suggests features beyond user's edition
        suggested_edition = Edition.COMMUNITY
        matched_keywords = []
        
        for edition, keywords in edition_keywords.items():
            for keyword in keywords:
                if keyword in intent_lower:
                    matched_keywords.append(keyword)
                    if edition == Edition.ENTERPRISE:
                        suggested_edition = Edition.ENTERPRISE
                    elif edition == Edition.BUSINESS and suggested_edition != Edition.ENTERPRISE:
                        suggested_edition = Edition.BUSINESS
        
        # Determine if upgrade is needed
        edition_hierarchy = {
            Edition.COMMUNITY: 0,
            Edition.BUSINESS: 1,
            Edition.ENTERPRISE: 2
        }
        
        needs_upgrade = edition_hierarchy[suggested_edition] > edition_hierarchy[user_edition]
        
        return {
            "intent": intent,
            "user_edition": user_edition,
            "suggested_edition": suggested_edition,
            "needs_upgrade": needs_upgrade,
            "matched_keywords": matched_keywords,
            "available": not needs_upgrade,
            "upgrade_message": f"This feature requires {suggested_edition} edition" if needs_upgrade else None
        }


# Singleton instance
_discovery = EditionFeatureDiscovery()

def get_edition_discovery() -> EditionFeatureDiscovery:
    """Get the edition feature discovery singleton."""
    return _discovery