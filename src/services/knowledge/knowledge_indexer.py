"""
Knowledge Indexer Service

Indexes all system components (templates, agents, adapters) for intelligent retrieval.
Creates searchable, categorized knowledge base for the assistant.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import hashlib

from models.workflow_templates import WorkflowTemplate
# AIAgent is in Business edition, not Community
# from models.ai_agent import AIAgent
# Adapter registry not available in simplified imports
# from adapters.registry import adapter_registry


class KnowledgeIndexer:
    """
    Indexes and organizes system knowledge for fast retrieval.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.index = {
            "templates": {},
            "agents": {},
            "adapters": {},
            "patterns": {},
            "metadata": {
                "indexed_at": None,
                "total_items": 0,
                "version": "1.0.0"
            }
        }

    async def build_index(self) -> Dict:
        """
        Build complete knowledge index.
        """
        print("[KnowledgeIndexer] Building knowledge index...")

        # Index all components
        await self._index_templates()
        await self._index_agents()
        await self._index_adapters()
        await self._extract_patterns()

        self.index["metadata"]["indexed_at"] = datetime.utcnow().isoformat()
        self.index["metadata"]["total_items"] = (
            len(self.index["templates"]) +
            len(self.index["agents"]) +
            len(self.index["adapters"])
        )

        print(f"[KnowledgeIndexer] Indexed {self.index['metadata']['total_items']} items")
        return self.index

    async def _index_templates(self):
        """
        Index all workflow templates with rich metadata.
        """
        # Query templates from database
        query = select(WorkflowTemplate)
        result = await self.db.execute(query)
        templates = result.scalars().all()

        for template in templates:
            template_id = str(template.id)

            # Create rich index entry
            self.index["templates"][template_id] = {
                "id": template_id,
                "name": template.name,
                "description": template.description,
                "category": template.category or "general",
                "tags": self._extract_tags(template),
                "search_terms": self._generate_search_terms(template),
                "use_cases": self._extract_use_cases(template),
                "complexity": self._calculate_complexity(template),
                "popularity": template.usage_count if hasattr(template, 'usage_count') else 0,
                "requirements": self._extract_requirements(template),
                "related_features": self._identify_related_features(template),
                "industry_fit": self._determine_industry_fit(template)
            }

    async def _index_agents(self):
        """
        Index all AI agents with capabilities.
        For Community edition, use predefined agent data.
        """
        # AIAgent model is in Business edition
        # For Community, use hardcoded agent catalog data
        agents_data = [
            {
                "id": "agent-1",
                "name": "Data Processing Agent",
                "type": "ai",
                "description": "Processes and transforms data",
                "capabilities": ["data_processing", "etl", "validation"],
                "framework": "langchain",
                "model": "gpt-4"
            },
            # Add more predefined agents as needed
        ]

        for agent in agents_data:
            agent_id = agent["id"]
            self.index["agents"][agent_id] = agent

    async def _index_adapters(self):
        """
        Index all adapters with integration capabilities.
        For Community edition, use predefined adapter data.
        """
        # Adapter registry not available in simplified structure
        # Use hardcoded adapter catalog data
        adapters = {
            "http": {"category": "network", "description": "HTTP/REST API adapter"},
            "database": {"category": "data", "description": "Database connection adapter"},
            "file": {"category": "storage", "description": "File system adapter"},
            # Add more predefined adapters
        }

        for adapter_type, adapter_info in adapters.items():
            # Create rich index entry
            self.index["adapters"][adapter_type] = {
                "type": adapter_type,
                "category": adapter_info.get("category", "general"),
                "description": adapter_info.get("description", ""),
                "search_terms": self._generate_adapter_search_terms(adapter_type, adapter_info),
                "capabilities": adapter_info.get("capabilities", []),
                "requirements": adapter_info.get("requirements", {}),
                "integration_complexity": self._assess_integration_complexity(adapter_info),
                "popular_use_cases": self._identify_adapter_use_cases(adapter_type),
                "compatible_with": self._identify_compatibility(adapter_type),
                "setup_time": self._estimate_setup_time(adapter_info)
            }

    async def _extract_patterns(self):
        """
        Extract common patterns from indexed items.
        """
        # Common workflow patterns
        self.index["patterns"]["workflow_patterns"] = [
            {
                "pattern": "approval_chain",
                "description": "Multi-step approval workflows",
                "templates": self._find_templates_by_pattern("approval"),
                "use_case": "Document approval, expense approval, leave requests"
            },
            {
                "pattern": "data_pipeline",
                "description": "Data processing and transformation",
                "templates": self._find_templates_by_pattern("data|etl|transform"),
                "use_case": "ETL processes, report generation, data sync"
            },
            {
                "pattern": "notification_automation",
                "description": "Automated alerts and notifications",
                "templates": self._find_templates_by_pattern("notify|alert|remind"),
                "use_case": "Status updates, deadline reminders, escalations"
            },
            {
                "pattern": "integration_sync",
                "description": "System-to-system synchronization",
                "templates": self._find_templates_by_pattern("sync|integrate|connect"),
                "use_case": "CRM sync, inventory updates, cross-platform data"
            }
        ]

        # Common agent patterns
        self.index["patterns"]["agent_patterns"] = [
            {
                "pattern": "customer_facing",
                "description": "Agents that interact with customers",
                "agents": self._find_agents_by_capability("customer|support|service"),
                "use_case": "Customer support, sales assistance, onboarding"
            },
            {
                "pattern": "analytical",
                "description": "Data analysis and insights",
                "agents": self._find_agents_by_capability("analyze|report|insight"),
                "use_case": "Business intelligence, forecasting, optimization"
            },
            {
                "pattern": "creative",
                "description": "Content creation and generation",
                "agents": self._find_agents_by_capability("write|create|generate"),
                "use_case": "Content marketing, documentation, copywriting"
            }
        ]

        # Common integration patterns
        self.index["patterns"]["integration_patterns"] = [
            {
                "pattern": "payment_processing",
                "description": "Payment and billing integrations",
                "adapters": ["stripe", "paypal", "square"],
                "templates": self._find_templates_by_pattern("payment|billing|invoice")
            },
            {
                "pattern": "communication_hub",
                "description": "Multi-channel communication",
                "adapters": ["slack", "email", "sms", "teams"],
                "templates": self._find_templates_by_pattern("notify|message|communicate")
            },
            {
                "pattern": "data_warehouse",
                "description": "Data storage and analytics",
                "adapters": ["bigquery", "snowflake", "databricks"],
                "templates": self._find_templates_by_pattern("analytics|report|data")
            }
        ]

    def _extract_tags(self, template: WorkflowTemplate) -> List[str]:
        """
        Extract tags from template.
        """
        tags = []

        # From explicit tags field
        if hasattr(template, 'tags') and template.tags:
            if isinstance(template.tags, list):
                tags.extend(template.tags)
            elif isinstance(template.tags, str):
                tags.extend(template.tags.split(','))

        # Extract from name and description
        if template.name:
            tags.extend(template.name.lower().split('_'))
        if template.description:
            # Extract key words from description
            words = template.description.lower().split()
            key_words = [w for w in words if len(w) > 4 and w.isalpha()]
            tags.extend(key_words[:5])

        return list(set(tags))

    def _generate_search_terms(self, template: WorkflowTemplate) -> List[str]:
        """
        Generate search terms for template.
        """
        terms = []

        # From template fields
        if template.name:
            terms.append(template.name.lower())
            terms.extend(template.name.lower().replace('_', ' ').split())

        if template.description:
            terms.append(template.description.lower())

        if hasattr(template, 'category') and template.category:
            terms.append(template.category.lower())

        # Add synonyms
        synonyms = {
            "invoice": ["billing", "payment", "receipt"],
            "customer": ["client", "user", "consumer"],
            "order": ["purchase", "transaction", "sale"],
            "email": ["message", "notification", "alert"],
            "report": ["analytics", "dashboard", "metrics"]
        }

        for term in terms[:]:
            for key, values in synonyms.items():
                if key in term:
                    terms.extend(values)

        return list(set(terms))

    def _extract_use_cases(self, template: WorkflowTemplate) -> List[str]:
        """
        Extract use cases from template.
        """
        use_cases = []

        # Based on category and name
        if hasattr(template, 'category'):
            if template.category == "sales":
                use_cases.extend(["lead generation", "opportunity tracking", "quote generation"])
            elif template.category == "hr":
                use_cases.extend(["onboarding", "leave management", "performance review"])
            elif template.category == "finance":
                use_cases.extend(["expense approval", "invoice processing", "budget tracking"])

        return use_cases

    def _calculate_complexity(self, template: WorkflowTemplate) -> str:
        """
        Calculate template complexity.
        """
        # Simple heuristic based on template configuration
        if hasattr(template, 'configuration'):
            config_size = len(str(template.configuration))
            if config_size < 500:
                return "simple"
            elif config_size < 2000:
                return "moderate"
            else:
                return "complex"
        return "simple"

    def _extract_requirements(self, template: WorkflowTemplate) -> Dict:
        """
        Extract requirements for template.
        """
        requirements = {
            "edition": "community",
            "adapters": [],
            "agents": [],
            "permissions": []
        }

        # Parse from configuration if available
        if hasattr(template, 'configuration') and template.configuration:
            try:
                config = json.loads(template.configuration) if isinstance(template.configuration, str) else template.configuration
                if "requirements" in config:
                    requirements.update(config["requirements"])
            except:
                pass

        return requirements

    def _identify_related_features(self, template: WorkflowTemplate) -> List[str]:
        """
        Identify related features for template.
        """
        features = []

        name_lower = template.name.lower() if template.name else ""

        if "approval" in name_lower:
            features.append("approval_chains")
        if "data" in name_lower or "etl" in name_lower:
            features.append("data_processing")
        if "email" in name_lower or "notify" in name_lower:
            features.append("notifications")
        if "api" in name_lower or "integrate" in name_lower:
            features.append("integrations")

        return features

    def _determine_industry_fit(self, template: WorkflowTemplate) -> List[str]:
        """
        Determine which industries this template fits.
        """
        industries = []

        name_lower = template.name.lower() if template.name else ""
        desc_lower = template.description.lower() if template.description else ""

        # Healthcare
        if any(term in name_lower + desc_lower for term in ["patient", "medical", "health", "clinic"]):
            industries.append("healthcare")

        # E-commerce
        if any(term in name_lower + desc_lower for term in ["order", "cart", "product", "inventory"]):
            industries.append("ecommerce")

        # Finance
        if any(term in name_lower + desc_lower for term in ["invoice", "payment", "transaction", "accounting"]):
            industries.append("finance")

        # Generic (fits all)
        if not industries or "general" in name_lower:
            industries.append("all")

        return industries

    def _parse_capabilities(self, agent: Dict) -> List[str]:
        """
        Parse agent capabilities.
        """
        capabilities = []

        if hasattr(agent, 'capabilities'):
            if isinstance(agent.capabilities, list):
                capabilities = agent.capabilities
            elif isinstance(agent.capabilities, str):
                try:
                    capabilities = json.loads(agent.capabilities)
                except:
                    capabilities = [agent.capabilities]

        # Add default capabilities based on type
        if hasattr(agent, 'agent_type'):
            if agent.agent_type == "customer_service":
                capabilities.extend(["respond_to_queries", "escalate_issues", "track_tickets"])
            elif agent.agent_type == "data_analyst":
                capabilities.extend(["analyze_data", "generate_reports", "identify_trends"])

        return list(set(capabilities))

    def _generate_agent_search_terms(self, agent: Dict) -> List[str]:
        """
        Generate search terms for agent.
        """
        terms = []

        if agent.name:
            terms.append(agent.name.lower())
            terms.extend(agent.name.lower().replace('_', ' ').split())

        if hasattr(agent, 'description') and agent.description:
            terms.append(agent.description.lower())

        if hasattr(agent, 'agent_type') and agent.agent_type:
            terms.append(agent.agent_type.lower())

        return list(set(terms))

    def _identify_specializations(self, agent: Dict) -> List[str]:
        """
        Identify agent specializations.
        """
        specs = []

        if hasattr(agent, 'agent_type'):
            type_lower = agent.agent_type.lower()
            if "customer" in type_lower or "support" in type_lower:
                specs.append("customer_service")
            if "data" in type_lower or "analyst" in type_lower:
                specs.append("data_analysis")
            if "write" in type_lower or "content" in type_lower:
                specs.append("content_creation")

        return specs

    def _identify_integration_points(self, agent: Dict) -> List[str]:
        """
        Identify where agent can integrate.
        """
        return ["workflows", "pods", "api", "webhooks"]

    def _extract_performance_metrics(self, agent: Dict) -> Dict:
        """
        Extract agent performance metrics.
        """
        return {
            "avg_response_time": "2s",
            "success_rate": "95%",
            "capacity": "100 tasks/hour"
        }

    def _calculate_cost_profile(self, agent: Dict) -> Dict:
        """
        Calculate agent cost profile.
        """
        # Simple cost estimation
        if hasattr(agent, 'model'):
            if "gpt-4" in agent.model.lower():
                return {"tier": "premium", "est_cost": "$0.03/request"}
            elif "gpt-3.5" in agent.model.lower():
                return {"tier": "standard", "est_cost": "$0.002/request"}

        return {"tier": "basic", "est_cost": "$0.001/request"}

    def _generate_adapter_search_terms(self, adapter_type: str, adapter_info: Dict) -> List[str]:
        """
        Generate search terms for adapter.
        """
        terms = [adapter_type.lower()]

        if "description" in adapter_info:
            terms.append(adapter_info["description"].lower())

        # Add category
        if "category" in adapter_info:
            terms.append(adapter_info["category"].lower())

        return terms

    def _assess_integration_complexity(self, adapter_info: Dict) -> str:
        """
        Assess integration complexity.
        """
        if "requirements" in adapter_info:
            req_count = len(adapter_info["requirements"])
            if req_count <= 1:
                return "simple"
            elif req_count <= 3:
                return "moderate"
            else:
                return "complex"
        return "simple"

    def _identify_adapter_use_cases(self, adapter_type: str) -> List[str]:
        """
        Identify adapter use cases.
        """
        use_cases_map = {
            "openai": ["content generation", "code assistance", "analysis"],
            "slack": ["team notifications", "alerts", "collaboration"],
            "stripe": ["payment processing", "subscriptions", "invoicing"],
            "email": ["notifications", "marketing", "transactional"],
            "shopify": ["ecommerce", "order management", "inventory"],
            "salesforce": ["crm", "sales tracking", "customer management"]
        }

        return use_cases_map.get(adapter_type, ["general integration"])

    def _identify_compatibility(self, adapter_type: str) -> List[str]:
        """
        Identify what this adapter is compatible with.
        """
        return ["workflows", "agents", "webhooks", "api"]

    def _estimate_setup_time(self, adapter_info: Dict) -> str:
        """
        Estimate setup time for adapter.
        """
        complexity = self._assess_integration_complexity(adapter_info)
        if complexity == "simple":
            return "5 minutes"
        elif complexity == "moderate":
            return "15 minutes"
        else:
            return "30+ minutes"

    def _find_templates_by_pattern(self, pattern: str) -> List[str]:
        """
        Find templates matching pattern.
        """
        matches = []
        pattern_lower = pattern.lower()

        for template_id, template_data in self.index["templates"].items():
            if pattern_lower in template_data["name"].lower() or \
               pattern_lower in template_data["description"].lower():
                matches.append(template_id)

        return matches[:5]  # Return top 5

    def _find_agents_by_capability(self, capability: str) -> List[str]:
        """
        Find agents with capability.
        """
        matches = []
        capability_lower = capability.lower()

        for agent_id, agent_data in self.index["agents"].items():
            if any(capability_lower in cap.lower() for cap in agent_data["capabilities"]):
                matches.append(agent_id)

        return matches[:5]  # Return top 5

    async def search(self, query: str, limit: int = 10) -> Dict:
        """
        Search the index for relevant items.
        """
        query_lower = query.lower()
        results = {
            "templates": [],
            "agents": [],
            "adapters": []
        }

        # Search templates
        for template_id, template_data in self.index["templates"].items():
            score = self._calculate_relevance_score(query_lower, template_data["search_terms"])
            if score > 0:
                results["templates"].append({
                    "id": template_id,
                    "name": template_data["name"],
                    "score": score,
                    "data": template_data
                })

        # Search agents
        for agent_id, agent_data in self.index["agents"].items():
            # Generate search terms if not present
            if "search_terms" not in agent_data:
                search_terms = []
                if "name" in agent_data:
                    search_terms.extend(agent_data["name"].lower().split())
                if "description" in agent_data:
                    search_terms.extend(agent_data["description"].lower().split())
                if "capabilities" in agent_data:
                    search_terms.extend(agent_data["capabilities"])
                agent_data["search_terms"] = search_terms

            score = self._calculate_relevance_score(query_lower, agent_data["search_terms"])
            if score > 0:
                results["agents"].append({
                    "id": agent_id,
                    "name": agent_data["name"],
                    "score": score,
                    "data": agent_data
                })

        # Search adapters
        for adapter_type, adapter_data in self.index["adapters"].items():
            # Generate search terms if not present
            if "search_terms" not in adapter_data:
                search_terms = []
                if "name" in adapter_data:
                    search_terms.extend(adapter_data["name"].lower().split())
                if "description" in adapter_data:
                    search_terms.extend(adapter_data["description"].lower().split())
                if "category" in adapter_data:
                    search_terms.append(adapter_data["category"])
                adapter_data["search_terms"] = search_terms

            score = self._calculate_relevance_score(query_lower, adapter_data["search_terms"])
            if score > 0:
                results["adapters"].append({
                    "type": adapter_type,
                    "score": score,
                    "data": adapter_data
                })

        # Sort by score and limit
        for category in results:
            results[category] = sorted(results[category], key=lambda x: x["score"], reverse=True)[:limit]

        return results

    def _calculate_relevance_score(self, query: str, search_terms: List[str]) -> float:
        """
        Calculate relevance score for search.
        """
        score = 0.0

        # Exact match
        if query in search_terms:
            score += 10.0

        # Partial matches
        for term in search_terms:
            if query in term:
                score += 5.0
            elif term in query:
                score += 3.0

            # Word overlap
            query_words = set(query.split())
            term_words = set(term.split())
            overlap = query_words.intersection(term_words)
            score += len(overlap) * 2.0

        return score