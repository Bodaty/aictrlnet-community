"""
Edition Feature Registry for AICtrlNet.

This module maintains a registry of which features belong to which edition,
allowing the conversation system and other components to provide edition-aware responses.
"""

from typing import Dict, List, Set, Optional
from enum import Enum


class Edition(str, Enum):
    """AICtrlNet editions."""
    COMMUNITY = "community"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class FeatureCategory(str, Enum):
    """Feature categories for organization."""
    CORE = "core"
    WORKFLOW = "workflow"
    INTEGRATION = "integration"
    AI_ML = "ai_ml"
    GOVERNANCE = "governance"
    ANALYTICS = "analytics"
    SECURITY = "security"
    PLATFORM = "platform"
    DATA_QUALITY = "data_quality"
    AGENT = "agent"
    COLLABORATION = "collaboration"


class EditionFeatureRegistry:
    """
    Centralized registry of features and their edition availability.
    Uses an accretive model: Business includes all Community features,
    Enterprise includes all Business features.
    """
    
    # Community Edition Features (Open Source)
    COMMUNITY_FEATURES = {
        FeatureCategory.CORE: [
            "task_management",
            "workflow_creation",
            "workflow_execution",
            "basic_scheduling",
            "webhook_integration",
            "api_keys",
            "user_management",
            "basic_auth",
        ],
        FeatureCategory.WORKFLOW: [
            "workflow_templates",
            "workflow_import_export",
            "workflow_versioning",
            "basic_triggers",
            "sequential_execution",
            "parallel_execution",
        ],
        FeatureCategory.INTEGRATION: [
            "mcp_integration",
            "rest_api",
            "webhook_events",
            "basic_adapters",
            "bridge_connections",
        ],
        FeatureCategory.AI_ML: [
            "nlp_processing",
            "multi_turn_conversation",
            "intent_detection",
            "basic_llm_integration",
        ],
        FeatureCategory.DATA_QUALITY: [
            "iso_25012_compliance",
            "data_validation",
            "quality_metrics",
            "basic_profiling",
        ],
        FeatureCategory.AGENT: [
            "iam_messaging",  # Internal Agent Messaging
            "basic_agent_orchestration",
            "agent_discovery",
        ],
    }
    
    # Business Edition Features (Professional)
    BUSINESS_FEATURES = {
        FeatureCategory.AI_ML: [
            "ai_governance",
            "ml_model_tracking",
            "risk_assessment",
            "compliance_monitoring",
            "ml_governance",
            "advanced_llm_features",
            "custom_model_integration",
        ],
        FeatureCategory.GOVERNANCE: [
            "agp_policies",  # AI Governance Policies
            "evaluation_framework",
            "automated_compliance",
            "audit_trails",
            "policy_enforcement",
        ],
        FeatureCategory.ANALYTICS: [
            "advanced_analytics",
            "performance_metrics",
            "resource_optimization",
            "cost_tracking",
            "usage_analytics",
        ],
        FeatureCategory.WORKFLOW: [
            "conditional_branching",
            "loop_constructs",
            "error_handling",
            "retry_policies",
            "advanced_scheduling",
            "workflow_optimization",
        ],
        FeatureCategory.SECURITY: [
            "role_based_access",
            "advanced_auth",
            "mfa_support",
            "api_rate_limiting",
            "security_scanning",
        ],
        FeatureCategory.DATA_QUALITY: [
            "advanced_profiling",
            "anomaly_detection",
            "data_lineage",
            "quality_dashboards",
        ],
        FeatureCategory.AGENT: [
            "pod_orchestration",
            "swarm_intelligence",
            "agent_performance_tracking",
            "memory_management",
            "cache_optimization",
        ],
    }
    
    # Enterprise Edition Features (Enterprise)
    ENTERPRISE_FEATURES = {
        FeatureCategory.PLATFORM: [
            "multi_tenancy",
            "federation",
            "geographic_routing",
            "cross_region_sync",
            "platform_integration",
            "enterprise_sso",
        ],
        FeatureCategory.GOVERNANCE: [
            "enterprise_compliance",
            "sox_compliance",
            "gdpr_compliance",
            "hipaa_compliance",
            "custom_compliance_frameworks",
        ],
        FeatureCategory.COLLABORATION: [
            "workflow_sharing",
            "team_workspaces",
            "approval_workflows",
            "delegation_policies",
        ],
        FeatureCategory.ANALYTICS: [
            "predictive_analytics",
            "ml_insights",
            "executive_dashboards",
            "custom_reporting",
            "data_warehouse_integration",
        ],
        FeatureCategory.SECURITY: [
            "enterprise_sso",
            "advanced_encryption",
            "key_management",
            "security_compliance",
            "threat_detection",
        ],
        FeatureCategory.AGENT: [
            "distributed_pods",
            "cross_tenant_orchestration",
            "enterprise_memory_pools",
            "global_cache_sync",
            "federated_agent_discovery",
        ],
    }
    
    @classmethod
    def get_edition_features(cls, edition: Edition) -> Dict[FeatureCategory, List[str]]:
        """
        Get all features available in a specific edition.
        Implements accretive model.
        """
        features = {}
        
        # Community features are available to all
        for category, feature_list in cls.COMMUNITY_FEATURES.items():
            features.setdefault(category, []).extend(feature_list)
        
        # Business features available to Business and Enterprise
        if edition in [Edition.BUSINESS, Edition.ENTERPRISE]:
            for category, feature_list in cls.BUSINESS_FEATURES.items():
                features.setdefault(category, []).extend(feature_list)
        
        # Enterprise features only for Enterprise
        if edition == Edition.ENTERPRISE:
            for category, feature_list in cls.ENTERPRISE_FEATURES.items():
                features.setdefault(category, []).extend(feature_list)
        
        return features
    
    @classmethod
    def get_feature_edition(cls, feature_name: str) -> Optional[Edition]:
        """
        Determine which edition a feature belongs to.
        Returns the minimum edition required for the feature.
        """
        # Check Community features
        for category, features in cls.COMMUNITY_FEATURES.items():
            if feature_name in features:
                return Edition.COMMUNITY
        
        # Check Business features
        for category, features in cls.BUSINESS_FEATURES.items():
            if feature_name in features:
                return Edition.BUSINESS
        
        # Check Enterprise features
        for category, features in cls.ENTERPRISE_FEATURES.items():
            if feature_name in features:
                return Edition.ENTERPRISE
        
        return None
    
    @classmethod
    def is_feature_available(cls, feature_name: str, user_edition: Edition) -> bool:
        """
        Check if a feature is available in the user's edition.
        """
        feature_edition = cls.get_feature_edition(feature_name)
        if not feature_edition:
            return False
        
        edition_hierarchy = {
            Edition.COMMUNITY: 0,
            Edition.BUSINESS: 1,
            Edition.ENTERPRISE: 2
        }
        
        return edition_hierarchy[user_edition] >= edition_hierarchy[feature_edition]
    
    @classmethod
    def get_upgrade_features(cls, current_edition: Edition) -> Dict[Edition, Dict[FeatureCategory, List[str]]]:
        """
        Get features that would be available by upgrading to higher editions.
        """
        upgrade_features = {}
        
        if current_edition == Edition.COMMUNITY:
            # Show Business features as upgrade option
            upgrade_features[Edition.BUSINESS] = cls.BUSINESS_FEATURES
            # Show Enterprise features as ultimate upgrade
            upgrade_features[Edition.ENTERPRISE] = {
                **cls.BUSINESS_FEATURES,
                **cls.ENTERPRISE_FEATURES
            }
        elif current_edition == Edition.BUSINESS:
            # Show Enterprise features as upgrade option
            upgrade_features[Edition.ENTERPRISE] = cls.ENTERPRISE_FEATURES
        
        return upgrade_features
    
    @classmethod
    def get_feature_description(cls, feature_name: str) -> Dict[str, Any]:
        """
        Get detailed description of a feature for conversation context.
        """
        descriptions = {
            # Core features
            "task_management": {
                "name": "Task Management",
                "description": "Create, manage, and execute tasks with dependencies",
                "category": FeatureCategory.CORE,
                "keywords": ["task", "job", "execution", "scheduling"]
            },
            "workflow_creation": {
                "name": "Workflow Creation",
                "description": "Design and build automated workflows with visual editor",
                "category": FeatureCategory.WORKFLOW,
                "keywords": ["workflow", "automation", "pipeline", "orchestration"]
            },
            
            # AI/ML features
            "ai_governance": {
                "name": "AI Governance",
                "description": "Monitor and govern AI models with compliance tracking",
                "category": FeatureCategory.AI_ML,
                "keywords": ["ai", "governance", "compliance", "model", "risk"]
            },
            "ml_governance": {
                "name": "ML Governance", 
                "description": "Machine learning model lifecycle management and monitoring",
                "category": FeatureCategory.AI_ML,
                "keywords": ["ml", "machine learning", "model", "monitoring"]
            },
            
            # Agent features
            "pod_orchestration": {
                "name": "Pod Orchestration",
                "description": "Coordinate groups of agents working together on complex tasks",
                "category": FeatureCategory.AGENT,
                "keywords": ["pod", "agent", "orchestration", "coordination"]
            },
            "swarm_intelligence": {
                "name": "Swarm Intelligence",
                "description": "Enable collective intelligence across agent groups",
                "category": FeatureCategory.AGENT,
                "keywords": ["swarm", "collective", "intelligence", "distributed"]
            },
            
            # Platform features
            "multi_tenancy": {
                "name": "Multi-Tenancy",
                "description": "Support multiple isolated tenants in a single deployment",
                "category": FeatureCategory.PLATFORM,
                "keywords": ["tenant", "isolation", "multi-tenant", "enterprise"]
            },
            "federation": {
                "name": "Federation",
                "description": "Connect and share workflows across organizations",
                "category": FeatureCategory.PLATFORM,
                "keywords": ["federation", "cross-org", "sharing", "collaboration"]
            },
        }
        
        return descriptions.get(feature_name, {
            "name": feature_name.replace("_", " ").title(),
            "description": f"Feature: {feature_name}",
            "category": cls._get_feature_category(feature_name),
            "keywords": [feature_name]
        })
    
    @classmethod
    def _get_feature_category(cls, feature_name: str) -> Optional[FeatureCategory]:
        """Get the category of a feature."""
        all_features = {
            **cls.COMMUNITY_FEATURES,
            **cls.BUSINESS_FEATURES,
            **cls.ENTERPRISE_FEATURES
        }
        
        for category, features in all_features.items():
            if feature_name in features:
                return category
        
        return None
    
    @classmethod
    def search_features(cls, query: str, user_edition: Edition) -> List[Dict[str, Any]]:
        """
        Search for features based on query and user's edition.
        Returns features available to the user and upgrade options.
        """
        query_lower = query.lower()
        results = {
            "available": [],
            "upgrade_required": []
        }
        
        # Get all features with their editions
        all_features = []
        for feature_list in cls.COMMUNITY_FEATURES.values():
            for feature in feature_list:
                all_features.append((feature, Edition.COMMUNITY))
        for feature_list in cls.BUSINESS_FEATURES.values():
            for feature in feature_list:
                all_features.append((feature, Edition.BUSINESS))
        for feature_list in cls.ENTERPRISE_FEATURES.values():
            for feature in feature_list:
                all_features.append((feature, Edition.ENTERPRISE))
        
        # Search and categorize
        for feature_name, feature_edition in all_features:
            feature_info = cls.get_feature_description(feature_name)
            
            # Check if query matches feature
            if (query_lower in feature_name.lower() or
                query_lower in feature_info.get("description", "").lower() or
                any(query_lower in kw for kw in feature_info.get("keywords", []))):
                
                feature_data = {
                    **feature_info,
                    "feature_id": feature_name,
                    "edition": feature_edition,
                    "available": cls.is_feature_available(feature_name, user_edition)
                }
                
                if feature_data["available"]:
                    results["available"].append(feature_data)
                else:
                    results["upgrade_required"].append(feature_data)
        
        return results


# Singleton instance
_registry = EditionFeatureRegistry()

def get_edition_registry() -> EditionFeatureRegistry:
    """Get the edition feature registry singleton."""
    return _registry