"""
Knowledge Retrieval Service

RAG-based retrieval for system knowledge with semantic search and context-aware ranking.
This service enables the intelligent assistant to find relevant information about AICtrlNet.
"""

import json
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter
from sqlalchemy.ext.asyncio import AsyncSession


class KnowledgeItem:
    """
    Represents a piece of knowledge retrieved from the system.
    """
    def __init__(self, type: str, id: str, name: str, data: Dict, relevance: float = 1.0):
        self.type = type  # template, agent, adapter, feature
        self.id = id
        self.name = name
        self.data = data
        self.relevance = relevance

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "data": self.data,
            "relevance": self.relevance
        }


class KnowledgeRetrievalService:
    """
    RAG-based retrieval service for system knowledge.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.manifest_service = None
        self.indexer = None
        # Community edition uses keyword-based search with TF-IDF
        self.keyword_index = {}
        self.document_frequency = {}  # For TF-IDF calculation
        self.total_documents = 0  # Total number of indexed documents
        self._initialized = False

    async def initialize(self):
        """
        Initialize the retrieval service.
        """
        if self._initialized:
            return

        # Import here to avoid circular dependencies
        from services.knowledge.system_manifest_service import get_manifest_service
        from services.knowledge.knowledge_indexer import KnowledgeIndexer

        # Initialize manifest and indexer
        self.manifest_service = await get_manifest_service(self.db)
        self.indexer = KnowledgeIndexer(self.db)
        await self.indexer.build_index()

        # Build keyword index for Community edition
        await self._build_keyword_index()

        self._initialized = True
        print("[KnowledgeRetrieval] Initialized with manifest and index")

    async def find_relevant_knowledge(
        self,
        query: str,
        context: Dict,
        limit: int = 10
    ) -> List[KnowledgeItem]:
        """
        Find most relevant system knowledge for user query.
        Community Edition: Uses keyword-based search.
        """
        if not self._initialized:
            await self.initialize()

        # Community edition: Use keyword search
        search_results = await self._keyword_search(query, limit=limit*2)

        # Step 2: Feature matching from manifest
        feature_matches = await self._match_features(query)

        # Step 3: Pattern matching
        pattern_matches = await self._match_patterns(query, context)

        # Step 4: Combine and rank
        all_items = []

        # Add search results with boosted scores for templates/agents/adapters
        # These are concrete, actionable results that users want to see first
        for template in search_results.get("templates", []):
            all_items.append(KnowledgeItem(
                type="template",
                id=template["id"],
                name=template["name"],
                data=template["data"],
                relevance=template["score"] * 5.0  # Boost template scores significantly
            ))

        for agent in search_results.get("agents", []):
            all_items.append(KnowledgeItem(
                type="agent",
                id=agent["id"],
                name=agent["name"],
                data=agent["data"],
                relevance=agent["score"] * 5.0  # Boost agent scores
            ))

        for adapter in search_results.get("adapters", []):
            all_items.append(KnowledgeItem(
                type="adapter",
                id=adapter["type"],
                name=adapter["type"],
                data=adapter["data"],
                relevance=adapter["score"] * 5.0  # Boost adapter scores
            ))

        # Add feature matches with moderate relevance
        # Features are informational, not actionable, so lower priority
        for feature_name, feature_data in feature_matches.items():
            all_items.append(KnowledgeItem(
                type="feature",
                id=feature_name,
                name=feature_name,
                data=feature_data,
                relevance=3.0  # Moderate relevance - helpful but not primary
            ))

        # Add pattern matches with high relevance
        for pattern in pattern_matches:
            all_items.append(KnowledgeItem(
                type="pattern",
                id=pattern["pattern"],
                name=pattern["description"],
                data=pattern,
                relevance=4.0  # Patterns are useful context
            ))

        # Step 5: Re-rank based on context
        ranked_items = await self._rerank_by_context(all_items, context)

        # Step 6: Augment with examples
        augmented_items = await self._add_examples(ranked_items)

        return augmented_items[:limit]

    async def _match_features(self, query: str) -> Dict:
        """
        Match query against system features.
        """
        if not self.manifest_service:
            return {}

        query_lower = query.lower()
        matches = {}

        for feature_name, feature_data in self.manifest_service.manifest["features"].items():
            # Check if feature name or description matches
            if feature_name.lower() in query_lower or \
               query_lower in feature_data.get("description", "").lower():
                matches[feature_name] = feature_data

            # Check capabilities
            for cap_name, cap_data in feature_data.get("capabilities", {}).items():
                # Handle both dict and boolean capability values
                if isinstance(cap_data, dict):
                    if cap_name.lower() in query_lower or \
                       query_lower in cap_data.get("description", "").lower():
                        matches[feature_name] = feature_data
                        break
                else:
                    # For boolean values, just check the capability name
                    if cap_name.lower() in query_lower:
                        matches[feature_name] = feature_data
                        break

        return matches

    async def _match_patterns(self, query: str, context: Dict) -> List[Dict]:
        """
        Match query against known patterns.
        """
        if not self.indexer:
            return []

        query_lower = query.lower()
        matches = []

        # Check workflow patterns
        for pattern in self.indexer.index.get("patterns", {}).get("workflow_patterns", []):
            if pattern["pattern"] in query_lower or \
               any(term in query_lower for term in pattern.get("use_case", "").lower().split()):
                matches.append(pattern)

        # Check agent patterns
        for pattern in self.indexer.index.get("patterns", {}).get("agent_patterns", []):
            if pattern["pattern"] in query_lower or \
               any(term in query_lower for term in pattern.get("use_case", "").lower().split()):
                matches.append(pattern)

        # Check integration patterns
        for pattern in self.indexer.index.get("patterns", {}).get("integration_patterns", []):
            if pattern["pattern"] in query_lower:
                matches.append(pattern)

        return matches

    async def _rerank_by_context(
        self,
        items: List[KnowledgeItem],
        context: Dict
    ) -> List[KnowledgeItem]:
        """
        Re-rank items based on conversation context.
        """
        # Extract context factors
        user_industry = context.get("user_industry")
        user_size = context.get("company_size")
        recent_topics = context.get("recent_topics", [])
        user_edition = context.get("edition", "community")

        # Re-rank based on context
        for item in items:
            # Boost if matches user industry
            if user_industry and "industry_fit" in item.data:
                if user_industry in item.data["industry_fit"]:
                    item.relevance *= 1.5

            # Boost if matches recent topics
            for topic in recent_topics:
                if topic.lower() in item.name.lower():
                    item.relevance *= 1.3

            # Adjust for edition
            if "required_edition" in item.data:
                req_edition = item.data["required_edition"]
                if req_edition == user_edition:
                    item.relevance *= 1.2
                elif self._edition_includes(user_edition, req_edition):
                    item.relevance *= 1.1
                else:
                    item.relevance *= 0.5  # Reduce relevance if not available

            # Boost popular items
            if "popularity" in item.data and item.data["popularity"] > 100:
                item.relevance *= 1.2

        # Sort by relevance
        return sorted(items, key=lambda x: x.relevance, reverse=True)

    async def _add_examples(self, items: List[KnowledgeItem]) -> List[KnowledgeItem]:
        """
        Augment items with relevant examples.
        """
        for item in items:
            if item.type == "template":
                # Add example usage
                item.data["example_usage"] = self._generate_template_example(item.data)

            elif item.type == "agent":
                # Add example configuration
                item.data["example_config"] = self._generate_agent_example(item.data)

            elif item.type == "feature":
                # Add example commands
                item.data["example_commands"] = self._generate_feature_examples(item.data)

        return items

    def _generate_template_example(self, template_data: Dict) -> Dict:
        """
        Generate example usage for template.
        """
        return {
            "description": f"Use the {template_data['name']} template",
            "command": f"Create a workflow from {template_data['name']} template",
            "parameters": template_data.get("requirements", {}),
            "expected_result": f"A new workflow based on {template_data['name']} will be created"
        }

    def _generate_agent_example(self, agent_data: Dict) -> Dict:
        """
        Generate example configuration for agent.
        """
        agent_type = agent_data.get('type', agent_data.get('agent_type', 'assistant'))
        return {
            "description": f"Configure {agent_data.get('name', 'AI')} agent",
            "command": f"Create a {agent_type} agent named {agent_data.get('name', 'AI Agent')}",
            "configuration": {
                "framework": agent_data.get("framework", "langchain"),
                "model": agent_data.get("model", "gpt-4"),
                "capabilities": agent_data.get("capabilities", [])[:3]
            }
        }

    def _generate_feature_examples(self, feature_data: Dict) -> List[str]:
        """
        Generate example commands for feature.
        """
        examples = []

        if "capabilities" in feature_data:
            for cap_name, cap_data in feature_data["capabilities"].items():
                if "examples" in cap_data:
                    examples.extend(cap_data["examples"][:2])

        if not examples:
            examples = [f"Use {feature_data.get('description', 'this feature')}"]

        return examples

    def _edition_includes(self, user_edition: str, required_edition: str) -> bool:
        """
        Check if user edition includes required edition.
        """
        edition_hierarchy = {
            "community": 0,
            "business": 1,
            "enterprise": 2
        }

        user_level = edition_hierarchy.get(user_edition, 0)
        required_level = edition_hierarchy.get(required_edition, 0)

        return user_level >= required_level

    async def get_feature_details(self, feature: str) -> Dict:
        """
        Get comprehensive details about a specific feature.
        """
        if not self._initialized:
            await self.initialize()

        details = await self.manifest_service.get_feature_details(feature)

        # Augment with examples and patterns
        if details:
            details["examples"] = self._generate_feature_examples(details.get("feature", {}))
            details["related_patterns"] = await self._find_related_patterns(feature)

        return details

    async def _find_related_patterns(self, feature: str) -> List[Dict]:
        """
        Find patterns related to a feature.
        """
        patterns = []

        if self.indexer and "patterns" in self.indexer.index:
            for pattern_type in ["workflow_patterns", "agent_patterns", "integration_patterns"]:
                for pattern in self.indexer.index["patterns"].get(pattern_type, []):
                    if feature.lower() in pattern.get("description", "").lower():
                        patterns.append(pattern)

        return patterns

    async def get_capabilities_summary(self) -> Dict:
        """
        Get a summary of system capabilities for the assistant.
        """
        if not self._initialized:
            await self.initialize()

        # Use the manifest service's get_capabilities_summary method
        if hasattr(self.manifest_service, 'get_capabilities_summary'):
            capabilities = await self.manifest_service.get_capabilities_summary()
            # Add extra fields the enhanced conversation manager expects
            capabilities["automation_coverage"] = 0.0  # TODO: Calculate properly
            capabilities["learning_status"] = "active" if self._initialized else "initializing"
            capabilities["active_automations"] = 0
            capabilities["ready"] = True
            return capabilities

        # Fallback to direct access if method doesn't exist
        stats = self.manifest_service.manifest.get("statistics", {})

        return {
            "templates": stats.get("total_templates", 0),
            "agents": stats.get("total_agents", 0),
            "adapters": stats.get("total_adapters", 0),
            "features": stats.get("total_features", 0),
            "endpoints": stats.get("total_endpoints", 0),
            "automation_coverage": stats.get("automation_coverage", 0.0),
            "learning_status": "active" if self._initialized else "initializing",
            "active_automations": 0,  # Will be populated when workflow tracking is implemented
            "ready": True
        }

    async def suggest_next_actions(
        self,
        current_action: str,
        context: Dict
    ) -> List[Dict]:
        """
        Suggest next actions based on current action and context.
        """
        suggestions = []

        # Common next actions based on current action
        next_actions_map = {
            "create_workflow": [
                {"action": "add_agent", "description": "Add an agent to the workflow"},
                {"action": "configure_trigger", "description": "Set up workflow trigger"},
                {"action": "test_workflow", "description": "Test the workflow"}
            ],
            "create_agent": [
                {"action": "configure_capabilities", "description": "Configure agent capabilities"},
                {"action": "add_to_workflow", "description": "Add agent to a workflow"},
                {"action": "create_pod", "description": "Create a pod with this agent"}
            ],
            "configure_integration": [
                {"action": "test_connection", "description": "Test the integration"},
                {"action": "map_fields", "description": "Map data fields"},
                {"action": "create_sync_workflow", "description": "Create sync workflow"}
            ]
        }

        # Get suggestions based on current action
        if current_action in next_actions_map:
            suggestions.extend(next_actions_map[current_action])

        # Add context-based suggestions
        if context.get("has_workflows") and not context.get("has_agents"):
            suggestions.append({
                "action": "create_agent",
                "description": "Create an agent to enhance your workflows"
            })

        if context.get("has_agents") and not context.get("has_pods"):
            suggestions.append({
                "action": "create_pod",
                "description": "Form a pod to coordinate multiple agents"
            })

        return suggestions[:5]  # Limit to 5 suggestions

    async def _build_keyword_index(self):
        """
        Build keyword index with TF-IDF preparation for Community edition search.
        """
        if not self.indexer or not self.indexer.index:
            return

        self.keyword_index = {
            "templates": {},
            "agents": {},
            "adapters": {},
            "features": {}
        }

        # Reset document frequency counters
        self.document_frequency = {}
        self.total_documents = 0

        # Index templates and build document frequencies
        for template_id, template in self.indexer.index.get("templates", {}).items():
            keywords = self._extract_keywords(template)
            self.keyword_index["templates"][template_id] = {
                "data": template,
                "keywords": keywords,
                "term_frequency": self._calculate_term_frequency(keywords)
            }
            self._update_document_frequency(keywords)
            self.total_documents += 1

        # Index agents
        for agent_id, agent in self.indexer.index.get("agents", {}).items():
            keywords = self._extract_keywords(agent)
            self.keyword_index["agents"][agent_id] = {
                "data": agent,
                "keywords": keywords,
                "term_frequency": self._calculate_term_frequency(keywords)
            }
            self._update_document_frequency(keywords)
            self.total_documents += 1

        # Index adapters
        for adapter_id, adapter in self.indexer.index.get("adapters", {}).items():
            keywords = self._extract_keywords(adapter)
            self.keyword_index["adapters"][adapter_id] = {
                "data": adapter,
                "keywords": keywords,
                "term_frequency": self._calculate_term_frequency(keywords)
            }
            self._update_document_frequency(keywords)
            self.total_documents += 1

    def _extract_keywords(self, item: Dict) -> List[str]:
        """
        Extract keywords from an item for indexing.
        """
        keywords = []

        # Extract from name
        if "name" in item:
            keywords.extend(self._tokenize(item["name"]))

        # Extract from description
        if "description" in item:
            keywords.extend(self._tokenize(item["description"]))

        # Extract from tags
        if "tags" in item:
            keywords.extend(item["tags"])

        # Extract from capabilities
        if "capabilities" in item:
            for cap in item["capabilities"]:
                if isinstance(cap, str):
                    keywords.extend(self._tokenize(cap))

        return list(set(keywords))  # Remove duplicates

    def _calculate_term_frequency(self, keywords: List[str]) -> Dict[str, float]:
        """
        Calculate term frequency for keywords.
        """
        if not keywords:
            return {}

        tf = {}
        total = len(keywords)

        for keyword in keywords:
            tf[keyword] = keywords.count(keyword) / total

        return tf

    def _update_document_frequency(self, keywords: List[str]):
        """
        Update document frequency for IDF calculation.
        """
        unique_keywords = set(keywords)
        for keyword in unique_keywords:
            self.document_frequency[keyword] = self.document_frequency.get(keyword, 0) + 1

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into keywords.
        """
        if not text:
            return []

        # Convert to lowercase and split by word boundaries
        words = re.findall(r'\w+', text.lower())

        # Filter out common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}

        return [w for w in words if w not in stop_words and len(w) > 2]

    async def _keyword_search(self, query: str, limit: int = 20) -> Dict:
        """
        Community Edition: Keyword-based search using the indexer.
        The indexer has better keyword matching than TF-IDF for this use case.
        """
        if not self.indexer:
            return {"templates": [], "agents": [], "adapters": []}

        # Use the indexer's search method which works well
        indexer_results = await self.indexer.search(query, limit=limit)

        # Convert indexer results to our expected format
        results = {
            "templates": [],
            "agents": [],
            "adapters": []
        }

        # Templates from indexer
        for template in indexer_results.get("templates", []):
            results["templates"].append({
                "id": template.get("id", ""),
                "name": template.get("name", ""),
                "data": template,
                "score": template.get("score", 1.0)
            })

        # Agents from indexer
        for agent in indexer_results.get("agents", []):
            results["agents"].append({
                "id": agent.get("id", ""),
                "name": agent.get("name", ""),
                "data": agent,
                "score": agent.get("score", 1.0)
            })

        # Adapters from indexer
        for adapter in indexer_results.get("adapters", []):
            results["adapters"].append({
                "type": adapter.get("type", ""),
                "data": adapter,
                "score": adapter.get("score", 1.0)
            })

        return results

    def _calculate_keyword_score(self, query_keywords: List[str], item_data: Dict) -> float:
        """
        Calculate TF-IDF relevance score for Community edition.
        """
        if not query_keywords or not item_data.get("keywords"):
            return 0.0

        item_keywords = item_data.get("keywords", [])
        term_frequency = item_data.get("term_frequency", {})

        # Calculate TF-IDF score
        score = 0.0
        matched_terms = 0

        for query_term in query_keywords:
            if query_term in term_frequency:
                # TF (Term Frequency) - already calculated
                tf = term_frequency[query_term]

                # IDF (Inverse Document Frequency)
                if query_term in self.document_frequency and self.total_documents > 0:
                    import math
                    idf = math.log(self.total_documents / (1 + self.document_frequency[query_term]))
                else:
                    idf = 1.0  # Default IDF if not found

                # TF-IDF score
                score += tf * idf
                matched_terms += 1

        # Normalize by number of query terms to favor documents matching more query terms
        if len(query_keywords) > 0:
            coverage_bonus = (matched_terms / len(query_keywords)) * 5.0
            score = score * (1 + coverage_bonus)

        # Boost if all query terms are found
        if matched_terms == len(query_keywords):
            score *= 2.0

        return score