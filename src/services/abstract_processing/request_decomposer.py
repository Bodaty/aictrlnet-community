"""Request Decomposer Service for AICtrlNet Intelligent Assistant.

This service breaks down high-level, abstract requests into concrete,
actionable components that can be executed by the system.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from core.config import get_settings
from services.knowledge.system_manifest_service import SystemManifestService
from services.llm_helpers import get_system_llm_settings
from llm import llm_service
from llm import UserLLMSettings

settings = get_settings()
logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of components that can be decomposed from a request."""
    WORKFLOW = "workflow"
    AGENT = "agent"
    POD = "pod"
    INTEGRATION = "integration"
    DATA_PROCESSING = "data_processing"
    MONITORING = "monitoring"
    NOTIFICATION = "notification"
    REPORT = "report"


@dataclass
class DecomposedComponent:
    """A single component extracted from an abstract request."""
    type: ComponentType
    name: str
    description: str
    requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    estimated_complexity: str = "medium"  # simple, medium, complex
    capabilities_needed: List[str] = field(default_factory=list)


@dataclass
class DecomposedRequest:
    """Complete decomposition of an abstract request."""
    original_request: str
    business_goal: str
    components: List[DecomposedComponent]
    phases: List[Dict[str, Any]]
    total_complexity: str
    estimated_effort: str
    success_metrics: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)


class RequestDecomposer:
    """Service for decomposing abstract requests into actionable components."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.manifest_service = SystemManifestService(db)
        self.llm_service = llm_service

        # Common business domains and their typical components
        self.domain_patterns = {
            "sales": {
                "keywords": ["sales", "crm", "leads", "deals", "pipeline", "quotes"],
                "typical_components": [
                    ("workflow", "lead_qualification", "Qualify and score incoming leads"),
                    ("workflow", "quote_generation", "Generate quotes based on requirements"),
                    ("agent", "sales_assistant", "AI assistant for sales team"),
                    ("monitoring", "sales_dashboard", "Track pipeline and conversion metrics"),
                    ("notification", "deal_alerts", "Alert on deal stage changes"),
                ]
            },
            "customer_service": {
                "keywords": ["support", "customer", "ticket", "helpdesk", "service"],
                "typical_components": [
                    ("workflow", "ticket_routing", "Route tickets to appropriate agents"),
                    ("workflow", "auto_response", "Auto-respond to common queries"),
                    ("agent", "support_bot", "First-level support chatbot"),
                    ("monitoring", "sla_tracking", "Track SLA compliance"),
                    ("report", "satisfaction_report", "Customer satisfaction metrics"),
                ]
            },
            "marketing": {
                "keywords": ["marketing", "campaign", "email", "social", "content"],
                "typical_components": [
                    ("workflow", "campaign_automation", "Automate campaign execution"),
                    ("workflow", "content_publishing", "Schedule and publish content"),
                    ("agent", "content_generator", "AI content creation assistant"),
                    ("monitoring", "engagement_tracking", "Track engagement metrics"),
                    ("report", "roi_analysis", "Campaign ROI analysis"),
                ]
            },
            "hr": {
                "keywords": ["hr", "human resources", "employee", "onboarding", "recruitment"],
                "typical_components": [
                    ("workflow", "onboarding_process", "Employee onboarding automation"),
                    ("workflow", "leave_management", "Leave request processing"),
                    ("agent", "hr_assistant", "HR query assistant"),
                    ("monitoring", "attendance_tracking", "Track attendance patterns"),
                    ("notification", "policy_updates", "Policy change notifications"),
                ]
            },
            "finance": {
                "keywords": ["finance", "accounting", "invoice", "expense", "budget"],
                "typical_components": [
                    ("workflow", "invoice_processing", "Automate invoice processing"),
                    ("workflow", "expense_approval", "Expense approval workflow"),
                    ("agent", "finance_assistant", "Financial query assistant"),
                    ("monitoring", "budget_tracking", "Track budget vs actual"),
                    ("report", "financial_summary", "Monthly financial reports"),
                ]
            }
        }

    async def decompose_request(self,
                               abstract_request: str,
                               context: Optional[Dict] = None) -> DecomposedRequest:
        """Decompose an abstract request into actionable components.

        Args:
            abstract_request: High-level request from user
            context: Optional context about user/business

        Returns:
            DecomposedRequest with all components identified
        """
        # Identify business domain using LLM-based semantic understanding
        domain = await self._identify_domain(abstract_request)

        # Extract business goal
        business_goal = await self._extract_business_goal(abstract_request, domain)

        # Identify required components
        components = await self._identify_components(abstract_request, domain, context)

        # Organize into phases
        phases = self._organize_phases(components)

        # Calculate complexity
        total_complexity = self._calculate_complexity(components)

        # Estimate effort
        estimated_effort = self._estimate_effort(components, total_complexity)

        # Define success metrics
        success_metrics = self._define_success_metrics(domain, business_goal)

        # Extract constraints and assumptions
        constraints = self._extract_constraints(abstract_request, context)
        assumptions = self._identify_assumptions(abstract_request, domain)

        return DecomposedRequest(
            original_request=abstract_request,
            business_goal=business_goal,
            components=components,
            phases=phases,
            total_complexity=total_complexity,
            estimated_effort=estimated_effort,
            success_metrics=success_metrics,
            constraints=constraints,
            assumptions=assumptions
        )

    async def _identify_domain(self, request: str) -> Optional[str]:
        """Identify the business domain from the request using LLM semantic understanding."""
        # Try LLM-based semantic domain identification
        if self.llm_service:
            try:
                # Build domain descriptions for better matching
                domain_descriptions = []
                for domain, pattern in self.domain_patterns.items():
                    keywords = ", ".join(pattern["keywords"])
                    domain_descriptions.append(f"- {domain}: {keywords}")

                prompt = f"""Identify the primary business domain for this request.

Request: "{request}"

Available domains:
{chr(10).join(domain_descriptions)}

Consider the full context and business process described, not just individual keywords.
For example, "customer onboarding" relates to HR (employee/customer onboarding process), not customer service.
"automate sales" relates to sales (CRM, pipeline), not customer service.

Return ONLY the domain name from the list above that best matches this request.
If none match well, return "None"."""

                # Use system settings for domain identification
                user_settings = get_system_llm_settings()

                response = await self.llm_service.generate(
                    prompt=prompt,
                    user_settings=user_settings,
                    task_type="classification",
                    temperature=0.1
                )

                if response and response.text:
                    # Extract just the domain name from potentially verbose response
                    response_text = response.text.strip().lower()

                    # First, check if any known domain appears in the response
                    for domain in self.domain_patterns.keys():
                        if domain in response_text:
                            logger.info(f"LLM identified domain: {domain} for request: {request[:50]}...")
                            return domain

                    # If no known domain found and response is short, treat it as the domain
                    if len(response_text) < 30:
                        if response_text in self.domain_patterns:
                            logger.info(f"LLM identified domain: {response_text} for request: {request[:50]}...")
                            return response_text

                    # Otherwise, log the issue
                    if response_text != "none":
                        logger.warning(f"LLM returned unknown domain: {response_text}")
            except Exception as e:
                logger.warning(f"LLM domain identification failed: {e}, falling back to keyword matching")

        # Fallback to keyword matching if LLM fails
        request_lower = request.lower()
        domain_scores = {}
        for domain, pattern in self.domain_patterns.items():
            score = sum(1 for keyword in pattern["keywords"]
                       if keyword in request_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            fallback_domain = max(domain_scores, key=domain_scores.get)
            logger.info(f"Fallback keyword matching identified domain: {fallback_domain}")
            return fallback_domain

        return None

    async def _extract_business_goal(self, request: str, domain: Optional[str]) -> str:
        """Extract the core business goal from the request."""
        # Try to use LLM for better extraction
        if self.llm_service:
            try:
                prompt = f"""Extract the main business goal from this request:
                Request: {request}
                Domain: {domain or 'general'}

                Provide a clear, concise business goal in one sentence."""

                # Use system settings for business goal extraction
                user_settings = get_system_llm_settings()

                response = await self.llm_service.generate(
                    prompt=prompt,
                    user_settings=user_settings,
                    task_type="analysis"
                )

                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")

        # Fallback to pattern-based extraction
        if "automate" in request.lower():
            # Extract what comes after "automate"
            match = re.search(r"automate\s+(?:my\s+)?(.+?)(?:\.|$)", request, re.IGNORECASE)
            if match:
                return f"Automate {match.group(1)}"

        # Default based on domain
        if domain:
            return f"Improve {domain} operations through automation"

        return "Streamline business processes through automation"

    async def _identify_components(self,
                                  request: str,
                                  domain: Optional[str],
                                  context: Optional[Dict]) -> List[DecomposedComponent]:
        """Identify required components for the request."""
        components = []

        # Get available capabilities from manifest
        manifest = await self.manifest_service.generate_manifest()
        available_capabilities = set()

        if manifest:
            if "workflow_templates" in manifest:
                available_capabilities.update(manifest["workflow_templates"].keys())
            if "agents" in manifest:
                available_capabilities.update(manifest["agents"].keys())

        # Start with domain-specific components if domain identified
        if domain and domain in self.domain_patterns:
            for comp_type, name, description in self.domain_patterns[domain]["typical_components"]:
                # Check if request mentions or implies this component
                if self._is_component_relevant(request, name, description):
                    component = DecomposedComponent(
                        type=ComponentType(comp_type),
                        name=name,
                        description=description,
                        estimated_complexity=self._estimate_component_complexity(comp_type, description),
                        capabilities_needed=self._identify_capabilities_for_component(
                            comp_type, name, available_capabilities
                        )
                    )
                    components.append(component)

        # Add general components based on request analysis
        request_lower = request.lower()

        # Check for workflow needs
        if any(word in request_lower for word in ["process", "workflow", "automate", "streamline"]):
            if not any(c.type == ComponentType.WORKFLOW for c in components):
                components.append(DecomposedComponent(
                    type=ComponentType.WORKFLOW,
                    name="main_automation_workflow",
                    description="Primary automation workflow for the requested process",
                    estimated_complexity="complex"
                ))

        # Check for monitoring needs
        if any(word in request_lower for word in ["track", "monitor", "measure", "metrics"]):
            if not any(c.type == ComponentType.MONITORING for c in components):
                components.append(DecomposedComponent(
                    type=ComponentType.MONITORING,
                    name="performance_monitoring",
                    description="Monitor and track key performance metrics",
                    estimated_complexity="medium"
                ))

        # Check for reporting needs
        if any(word in request_lower for word in ["report", "analyze", "insights", "analytics"]):
            if not any(c.type == ComponentType.REPORT for c in components):
                components.append(DecomposedComponent(
                    type=ComponentType.REPORT,
                    name="analytics_reporting",
                    description="Generate reports and analytics",
                    estimated_complexity="medium"
                ))

        # Add dependencies between components
        self._identify_dependencies(components)

        return components

    def _is_component_relevant(self, request: str, component_name: str, description: str) -> bool:
        """Check if a component is relevant to the request."""
        request_lower = request.lower()

        # Check if component name keywords appear in request
        name_words = component_name.replace("_", " ").split()
        if any(word in request_lower for word in name_words):
            return True

        # Check if description keywords appear in request
        desc_words = description.lower().split()
        important_words = [w for w in desc_words if len(w) > 4]  # Skip small words
        if sum(1 for word in important_words if word in request_lower) >= 2:
            return True

        # Check for "entire", "complete", "full" - implies all components needed
        if any(word in request_lower for word in ["entire", "complete", "full", "comprehensive"]):
            return True

        return False

    def _estimate_component_complexity(self, comp_type: str, description: str) -> str:
        """Estimate the complexity of a component."""
        # Complex indicators
        complex_indicators = ["integrate", "ai", "machine learning", "multi", "complex", "advanced"]
        if any(indicator in description.lower() for indicator in complex_indicators):
            return "complex"

        # Simple indicators
        simple_indicators = ["basic", "simple", "alert", "notify", "track"]
        if any(indicator in description.lower() for indicator in simple_indicators):
            return "simple"

        # Default by type
        if comp_type in ["agent", "pod"]:
            return "complex"
        elif comp_type in ["notification", "report"]:
            return "simple"

        return "medium"

    def _identify_capabilities_for_component(self,
                                            comp_type: str,
                                            name: str,
                                            available: set) -> List[str]:
        """Identify which available capabilities are needed for a component."""
        needed = []

        # Match based on name similarity
        for capability in available:
            if name in capability.lower() or capability.lower() in name:
                needed.append(capability)

        # Add type-specific capabilities
        if comp_type == "workflow":
            workflow_caps = [cap for cap in available if "workflow" in cap.lower()]
            needed.extend(workflow_caps[:2])  # Add up to 2 workflow capabilities

        return list(set(needed))  # Remove duplicates

    def _identify_dependencies(self, components: List[DecomposedComponent]):
        """Identify dependencies between components."""
        for i, component in enumerate(components):
            # Monitoring depends on workflows
            if component.type == ComponentType.MONITORING:
                workflow_names = [c.name for c in components
                                if c.type == ComponentType.WORKFLOW]
                component.dependencies.extend(workflow_names)

            # Reports depend on monitoring
            elif component.type == ComponentType.REPORT:
                monitoring_names = [c.name for c in components
                                  if c.type == ComponentType.MONITORING]
                component.dependencies.extend(monitoring_names)

            # Notifications depend on workflows
            elif component.type == ComponentType.NOTIFICATION:
                workflow_names = [c.name for c in components
                                if c.type == ComponentType.WORKFLOW]
                component.dependencies.extend(workflow_names[:1])  # Just primary workflow

    def _organize_phases(self, components: List[DecomposedComponent]) -> List[Dict[str, Any]]:
        """Organize components into implementation phases."""
        phases = []

        # Phase 1: Foundation (workflows and integrations)
        foundation = [c for c in components
                     if c.type in [ComponentType.WORKFLOW, ComponentType.INTEGRATION]]
        if foundation:
            phases.append({
                "name": "Foundation",
                "description": "Set up core workflows and integrations",
                "components": [c.name for c in foundation],
                "duration": "1-2 weeks",
                "priority": "high"
            })

        # Phase 2: Intelligence (agents and data processing)
        intelligence = [c for c in components
                       if c.type in [ComponentType.AGENT, ComponentType.POD, ComponentType.DATA_PROCESSING]]
        if intelligence:
            phases.append({
                "name": "Intelligence",
                "description": "Add AI agents and intelligent processing",
                "components": [c.name for c in intelligence],
                "duration": "2-3 weeks",
                "priority": "medium"
            })

        # Phase 3: Insights (monitoring and reporting)
        insights = [c for c in components
                   if c.type in [ComponentType.MONITORING, ComponentType.REPORT, ComponentType.NOTIFICATION]]
        if insights:
            phases.append({
                "name": "Insights",
                "description": "Enable monitoring, reporting, and notifications",
                "components": [c.name for c in insights],
                "duration": "1 week",
                "priority": "medium"
            })

        return phases

    def _calculate_complexity(self, components: List[DecomposedComponent]) -> str:
        """Calculate overall complexity of the solution."""
        if not components:
            return "simple"

        complexity_scores = {"simple": 1, "medium": 2, "complex": 3}
        total_score = sum(complexity_scores[c.estimated_complexity] for c in components)
        avg_score = total_score / len(components)

        if avg_score >= 2.5:
            return "complex"
        elif avg_score >= 1.5:
            return "medium"
        else:
            return "simple"

    def _estimate_effort(self, components: List[DecomposedComponent], complexity: str) -> str:
        """Estimate the total effort required."""
        base_effort = {
            "simple": 1,
            "medium": 2,
            "complex": 4
        }

        # Calculate total effort units
        total_units = sum(base_effort[c.estimated_complexity] for c in components)

        # Convert to time estimate
        if total_units <= 3:
            return "1-2 days"
        elif total_units <= 8:
            return "3-5 days"
        elif total_units <= 15:
            return "1-2 weeks"
        elif total_units <= 30:
            return "2-4 weeks"
        else:
            return "1-2 months"

    def _define_success_metrics(self, domain: Optional[str], business_goal: str) -> List[str]:
        """Define success metrics for the solution."""
        metrics = []

        # Domain-specific metrics
        domain_metrics = {
            "sales": [
                "Lead conversion rate improvement",
                "Sales cycle time reduction",
                "Pipeline visibility increase"
            ],
            "customer_service": [
                "Average response time reduction",
                "Customer satisfaction score improvement",
                "Ticket resolution rate increase"
            ],
            "marketing": [
                "Campaign execution time reduction",
                "Engagement rate improvement",
                "Content production efficiency increase"
            ],
            "hr": [
                "Onboarding time reduction",
                "Employee satisfaction improvement",
                "HR query response time reduction"
            ],
            "finance": [
                "Invoice processing time reduction",
                "Error rate reduction",
                "Financial reporting speed improvement"
            ]
        }

        if domain and domain in domain_metrics:
            metrics.extend(domain_metrics[domain])

        # General metrics based on goal
        if "automate" in business_goal.lower():
            metrics.append("Manual task reduction by 70%+")
        if "improve" in business_goal.lower():
            metrics.append("Process efficiency improvement by 40%+")
        if "reduce" in business_goal.lower():
            metrics.append("Cost reduction by 25%+")

        # Always include these
        metrics.extend([
            "System adoption rate above 80%",
            "ROI within 6 months"
        ])

        return list(set(metrics))[:5]  # Return up to 5 unique metrics

    def _extract_constraints(self, request: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Extract constraints from the request and context."""
        constraints = {}

        request_lower = request.lower()

        # Timeline constraints
        if "urgent" in request_lower or "asap" in request_lower:
            constraints["timeline"] = "urgent"
        elif "this month" in request_lower:
            constraints["timeline"] = "1 month"
        elif "this quarter" in request_lower:
            constraints["timeline"] = "3 months"

        # Budget constraints
        if "budget" in request_lower or "cost" in request_lower:
            constraints["budget_sensitive"] = True

        # Compliance constraints
        if any(word in request_lower for word in ["compliant", "compliance", "regulatory"]):
            constraints["compliance_required"] = True

        # Scale constraints
        if "enterprise" in request_lower or "large scale" in request_lower:
            constraints["scale"] = "enterprise"
        elif "small" in request_lower or "startup" in request_lower:
            constraints["scale"] = "small"

        # Add context constraints
        if context:
            if "company_size" in context:
                constraints["company_size"] = context["company_size"]
            if "industry" in context:
                constraints["industry"] = context["industry"]

        return constraints

    def _identify_assumptions(self, request: str, domain: Optional[str]) -> List[str]:
        """Identify assumptions being made about the request."""
        assumptions = []

        # Domain assumptions
        if domain:
            assumptions.append(f"Primary focus is on {domain} processes")

        # Integration assumptions
        if "integrate" not in request.lower():
            assumptions.append("Standard integrations with common tools will suffice")

        # Scale assumptions
        if "scale" not in request.lower():
            assumptions.append("Solution will handle moderate transaction volumes")

        # User assumptions
        assumptions.append("Users have basic technical proficiency")

        # Data assumptions
        assumptions.append("Historical data is available for analysis")

        return assumptions