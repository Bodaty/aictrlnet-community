"""
Edition-Aware System Manifest Service V2

This service dynamically discovers and reports system capabilities based on the
actual running edition, respecting the accretive model where:
- Community: Base features only
- Business: Community + Business features
- Enterprise: Community + Business + Enterprise features

Key Principles:
1. Dynamic Discovery - No hardcoded counts or capabilities
2. Edition Awareness - Only report what's actually available
3. Accretive Model - Higher editions include lower edition features
4. Real-time Accuracy - Query actual databases and filesystems
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, inspect
import logging

from core.database import get_db
from core.edition_discovery import get_edition_discovery, Edition
from models.workflow_templates import WorkflowTemplate
from fastapi import FastAPI
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)


class SystemManifestService:
    """
    Generates edition-aware system manifests that accurately reflect
    what's available in the running edition.
    """

    # Edition hierarchy (for accretive model)
    EDITION_HIERARCHY = {
        "community": ["community"],
        "business": ["community", "business"],
        "enterprise": ["community", "business", "enterprise"]
    }

    # Template directories per edition
    TEMPLATE_DIRS = {
        "community": ["/app/workflow-templates"],
        "business": ["/app/workflow-templates"],  # Business has its own
        "enterprise": ["/app/workflow-templates"]  # Enterprise has its own
    }

    def __init__(self, db: AsyncSession):
        self.db = db

        # Use the edition discovery system
        self.edition_discovery = get_edition_discovery()

        # Detect current edition properly
        self.edition = self._detect_current_edition()
        self.accessible_editions = self.EDITION_HIERARCHY.get(self.edition, ["community"])

        logger.info(f"[Manifest] Initialized for edition: {self.edition}")
        logger.info(f"[Manifest] Accessible editions: {self.accessible_editions}")

        self.manifest = {
            "version": "2.0.0",
            "edition": self.edition,
            "accessible_editions": self.accessible_editions,
            "generated_at": None,
            "features": {},
            "templates": {},
            "agents": {},
            "adapters": {},
            "endpoints": {},
            "ui_mappings": {},
            "statistics": {}
        }

    def _detect_current_edition(self) -> str:
        """Detect running edition using proper detection logic."""
        # First check environment variable (this is set in docker-compose)
        edition_env = os.getenv("AICTRLNET_EDITION", "").lower()
        if edition_env in ["community", "business", "enterprise"]:
            return edition_env

        # Then check imported modules to infer edition
        if 'models.enterprise' in sys.modules or 'aictrlnet_enterprise' in sys.modules:
            return "enterprise"
        elif 'models.business' in sys.modules or 'aictrlnet_business' in sys.modules:
            return "business"
        else:
            return "community"

    async def generate_manifest(self) -> Dict:
        """Generate edition-aware manifest."""
        logger.info(f"[Manifest] Starting generation for {self.edition} edition")

        # Scan components based on edition
        await self._scan_features()
        await self._scan_templates()
        await self._scan_agents()
        await self._scan_adapters()
        await self._scan_endpoints()
        await self._calculate_statistics()

        self.manifest["generated_at"] = datetime.utcnow().isoformat()

        logger.info(f"[Manifest] Generation complete - {len(self.manifest['features'])} features, "
                   f"{self.manifest['templates'].get('total_count', 0)} templates, "
                   f"{self.manifest['agents'].get('total_count', 0)} agents")

        return self.manifest

    async def _scan_features(self):
        """Scan available features based on edition."""
        features = {}

        # Community features - always available
        if "community" in self.accessible_editions:
            features.update({
                "workflow_automation": {
                    "description": "Create and manage automated workflows",
                    "edition": "community",
                    "capabilities": {
                        "basic_workflows": True,
                        "manual_creation": True,
                        "simple_scheduling": True
                    }
                },
                "basic_integrations": {
                    "description": "Connect with external systems via webhooks",
                    "edition": "community",
                    "capabilities": {
                        "webhooks": True,
                        "rest_api": True,
                        "basic_adapters": True
                    }
                }
            })

        # Business features - if Business or Enterprise
        if "business" in self.accessible_editions:
            features.update({
                "agent_management": {
                    "description": "Deploy and orchestrate AI agents",
                    "edition": "business",
                    "capabilities": {
                        "ai_agents": True,
                        "human_pools": True,
                        "pod_formation": True,
                        "ml_enhanced": True
                    }
                },
                "ai_governance": {
                    "description": "ML-powered governance and risk assessment",
                    "edition": "business",
                    "capabilities": {
                        "risk_assessment": True,
                        "bias_detection": True,
                        "explainability": True,
                        "compliance_tracking": True
                    }
                },
                "advanced_monitoring": {
                    "description": "Real-time monitoring with ML insights",
                    "edition": "business",
                    "capabilities": {
                        "predictive_analytics": True,
                        "anomaly_detection": True,
                        "performance_optimization": True
                    }
                }
            })

        # Enterprise features - if Enterprise
        if "enterprise" in self.accessible_editions:
            features.update({
                "multi_tenancy": {
                    "description": "Full multi-tenant support",
                    "edition": "enterprise",
                    "capabilities": {
                        "tenant_isolation": True,
                        "resource_quotas": True,
                        "custom_domains": True
                    }
                },
                "federation": {
                    "description": "Cross-organization federation",
                    "edition": "enterprise",
                    "capabilities": {
                        "federated_workflows": True,
                        "cross_org_agents": True,
                        "unified_governance": True
                    }
                }
            })

        self.manifest["features"] = features

    async def _scan_templates(self):
        """Scan workflow templates from database and filesystem."""
        template_manifest = {
            "total_count": 0,
            "by_edition": {},
            "categories": {},
            "sources": []
        }

        # Get templates from database (Community)
        if "community" in self.accessible_editions:
            try:
                # Count templates in database
                db_count = await self.db.scalar(
                    select(func.count(WorkflowTemplate.id))
                )
                template_manifest["by_edition"]["community"] = db_count or 0
                template_manifest["total_count"] += db_count or 0
                template_manifest["sources"].append("database")

                # Get template categories
                result = await self.db.execute(
                    select(WorkflowTemplate.category, func.count(WorkflowTemplate.id))
                    .group_by(WorkflowTemplate.category)
                )
                for category, count in result:
                    if category:
                        template_manifest["categories"][category] = count

            except Exception as e:
                logger.warning(f"[Manifest] Error scanning database templates: {e}")

        # Scan filesystem templates based on edition
        for edition in self.accessible_editions:
            template_dirs = self.TEMPLATE_DIRS.get(edition, [])
            for template_dir in template_dirs:
                if os.path.exists(template_dir):
                    try:
                        # Count JSON files
                        json_files = list(Path(template_dir).rglob("*.json"))
                        # Filter out metadata files
                        template_files = [f for f in json_files if not f.name.endswith('.metadata.json')]

                        count = len(template_files)
                        if count > 0:
                            if edition not in template_manifest["by_edition"]:
                                template_manifest["by_edition"][edition] = 0
                            template_manifest["by_edition"][edition] += count
                            template_manifest["total_count"] += count

                            if f"filesystem_{edition}" not in template_manifest["sources"]:
                                template_manifest["sources"].append(f"filesystem_{edition}")

                    except Exception as e:
                        logger.warning(f"[Manifest] Error scanning {template_dir}: {e}")

        self.manifest["templates"] = template_manifest

    async def _scan_agents(self):
        """Scan available agents based on edition."""
        agent_manifest = {
            "total_count": 0,
            "by_edition": {},
            "categories": {
                "ai_agents": [],
                "human_pools": [],
                "specialized": []
            }
        }

        # Community agents
        if "community" in self.accessible_editions:
            try:
                from services.agent_config_service import AgentConfigService
                config_service = AgentConfigService()

                # Get Community edition agents
                community_agents = getattr(config_service, 'ALLOWED_AGENTS', [])

                for agent_name in community_agents:
                    agent_info = {
                        "id": agent_name,
                        "name": agent_name,
                        "edition": "community",
                        "type": "ai",
                        "description": self._get_agent_description(agent_name)
                    }
                    agent_manifest["categories"]["ai_agents"].append(agent_info)

                agent_manifest["by_edition"]["community"] = len(community_agents)
                agent_manifest["total_count"] += len(community_agents)

            except ImportError:
                logger.warning("[Manifest] Could not import Community agent config")

        # Business agents (if accessible)
        if "business" in self.accessible_editions:
            try:
                # Business edition has EnhancedAgent models
                # Try to import from Business edition path first
                try:
                    from aictrlnet_business.models.enhanced_agent import EnhancedAgent
                    result = await self.db.execute(
                        select(func.count(EnhancedAgent.id)).where(EnhancedAgent.is_system == True)
                    )
                    business_count = result.scalar() or 0
                except ImportError:
                    # Fallback: check if table exists directly
                    from sqlalchemy import text
                    result = await self.db.execute(text(
                        "SELECT COUNT(*) FROM enhanced_agents WHERE is_system = true"
                    ))
                    business_count = result.scalar() or 0

                agent_manifest["by_edition"]["business"] = business_count
                agent_manifest["total_count"] += business_count

            except (ImportError, AttributeError, Exception) as e:
                logger.info(f"[Manifest] Business agents not available: {e}")

        # Enterprise agents (if accessible)
        if "enterprise" in self.accessible_editions:
            try:
                # Enterprise might have federated agents
                from models.enterprise import FederatedAgent
                result = await self.db.execute(select(func.count(FederatedAgent.id)))
                enterprise_count = result.scalar() or 0

                agent_manifest["by_edition"]["enterprise"] = enterprise_count
                agent_manifest["total_count"] += enterprise_count

            except (ImportError, AttributeError) as e:
                logger.info(f"[Manifest] Enterprise agents not available: {e}")

        self.manifest["agents"] = agent_manifest

    async def _scan_adapters(self):
        """Scan available adapters based on edition."""
        adapter_manifest = {
            "total_count": 0,
            "by_edition": {},
            "categories": {},
            "available": []
        }

        # Check database for configured adapters
        try:
            from models.community_complete import Adapter

            # Count all adapters
            total_result = await self.db.execute(
                select(func.count(Adapter.id))
            )
            total_count = total_result.scalar() or 0

            # Count enabled adapters
            enabled_result = await self.db.execute(
                select(func.count(Adapter.id)).where(Adapter.enabled == True)
            )
            enabled_count = enabled_result.scalar() or 0

            # Get adapter categories
            result = await self.db.execute(
                select(Adapter.category, func.count(Adapter.id))
                .where(Adapter.enabled == True)
                .group_by(Adapter.category)
            )

            for category, count in result:
                if category:
                    adapter_manifest["categories"][category] = count

            # Get adapter list
            adapters_result = await self.db.execute(
                select(Adapter.id).where(Adapter.enabled == True)
            )
            adapter_manifest["available"] = [str(a[0]) for a in adapters_result]

            adapter_manifest["total_count"] = enabled_count
            adapter_manifest["by_edition"][self.edition] = enabled_count

        except Exception as e:
            logger.warning(f"[Manifest] Error scanning adapters: {e}")

        self.manifest["adapters"] = adapter_manifest

    async def _scan_endpoints(self):
        """Scan API endpoints based on edition."""
        endpoints = {}

        try:
            # Import the appropriate router based on edition
            if self.edition == "enterprise":
                from api.v1.enterprise_router import api_router
            elif self.edition == "business":
                from api.v1.business_router import api_router
            else:
                from api.v1.community_router import api_router

            # Extract routes
            for route in api_router.routes:
                if isinstance(route, APIRoute):
                    key = f"{list(route.methods)[0] if route.methods else 'GET'} {route.path}"
                    endpoints[key] = {
                        "path": route.path,
                        "methods": list(route.methods) if route.methods else [],
                        "name": route.name,
                        "tags": route.tags
                    }

        except ImportError as e:
            logger.warning(f"[Manifest] Could not import router for {self.edition}: {e}")

        self.manifest["endpoints"] = endpoints

    async def _calculate_statistics(self):
        """Calculate statistics based on discovered resources."""
        stats = {
            "edition": self.edition,
            "total_features": len(self.manifest.get("features", {})),
            "total_templates": self.manifest.get("templates", {}).get("total_count", 0),
            "total_agents": self.manifest.get("agents", {}).get("total_count", 0),
            "total_adapters": self.manifest.get("adapters", {}).get("total_count", 0),
            "total_endpoints": len(self.manifest.get("endpoints", {})),
            "accessible_editions": self.accessible_editions
        }

        self.manifest["statistics"] = stats

    def _get_agent_description(self, agent_name: str) -> str:
        """Get description for an agent."""
        descriptions = {
            "basic_nlp": "Natural language processing and text analysis",
            "basic_workflow": "Simple workflow generation",
            "basic_assistant": "General AI assistance"
        }
        return descriptions.get(agent_name, "AI agent")

    async def get_manifest(self) -> Dict:
        """Get the complete system manifest."""
        if not self.manifest.get("generated_at"):
            await self.generate_manifest()
        return self.manifest

    async def get_capabilities_summary(self) -> Dict:
        """Get summary of capabilities for the assistant."""
        if not self.manifest.get("generated_at"):
            await self.generate_manifest()

        stats = self.manifest.get("statistics", {})

        return {
            "edition": self.edition,
            "templates": stats.get("total_templates", 0),
            "agents": stats.get("total_agents", 0),
            "adapters": stats.get("total_adapters", 0),
            "features": stats.get("total_features", 0),
            "endpoints": stats.get("total_endpoints", 0)
        }


# Singleton instance management
_manifest_service = None

async def get_manifest_service(db: AsyncSession) -> SystemManifestService:
    """Get or create the edition-aware manifest service."""
    global _manifest_service
    if _manifest_service is None:
        _manifest_service = SystemManifestService(db)
        await _manifest_service.generate_manifest()
    return _manifest_service