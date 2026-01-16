"""Solution Synthesizer Service for AICtrlNet Intelligent Assistant.

This service synthesizes complete, actionable solutions from decomposed requests
and clarification answers, creating implementation-ready plans.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from services.abstract_processing.request_decomposer import (
    DecomposedRequest, DecomposedComponent, ComponentType
)
from services.abstract_processing.clarification_engine import ClarificationContext
from services.action_planner import ActionPlanner, ActionPlan, ActionStep
from services.knowledge.system_manifest_service import SystemManifestService

logger = logging.getLogger(__name__)


class ImplementationPhase(Enum):
    """Phases of solution implementation."""
    FOUNDATION = "foundation"  # Core infrastructure and integrations
    AUTOMATION = "automation"  # Workflows and process automation
    INTELLIGENCE = "intelligence"  # AI agents and smart features
    OPTIMIZATION = "optimization"  # Monitoring and optimization
    SCALING = "scaling"  # Scale and performance tuning


@dataclass
class ResourceRequirement:
    """Resource needed for solution implementation."""
    type: str  # compute, storage, integration, human
    name: str
    specification: Dict[str, Any]
    estimated_cost: Optional[float] = None
    availability: str = "immediate"  # immediate, setup_required, external


@dataclass
class RiskMitigation:
    """Risk and its mitigation strategy."""
    risk: str
    probability: str  # low, medium, high
    impact: str  # low, medium, high
    mitigation: str
    contingency: Optional[str] = None


@dataclass
class ImplementationPlan:
    """Detailed implementation plan for a solution."""
    phases: List[Dict[str, Any]]
    total_duration: str
    resource_requirements: List[ResourceRequirement]
    dependencies: Dict[str, List[str]]  # component -> dependencies
    risk_mitigations: List[RiskMitigation]
    success_metrics: List[str]
    rollback_strategy: Optional[str] = None
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CompleteSolution:
    """Complete synthesized solution ready for implementation."""
    id: str
    original_request: str
    business_goal: str
    solution_name: str
    description: str

    # Solution components
    workflows: List[Dict[str, Any]]
    agents: List[Dict[str, Any]]
    integrations: List[Dict[str, Any]]
    monitoring: List[Dict[str, Any]]

    # Implementation details
    implementation_plan: ImplementationPlan

    # Estimates
    expected_automation_rate: float
    expected_time_savings: float  # hours per week
    expected_error_reduction: float  # percentage
    expected_roi_months: int

    # Explanation
    explanation: str
    assumptions: List[str]
    constraints: Dict[str, Any]

    # Optional fields
    action_plan: Optional[ActionPlan] = None  # Executable action plan

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    confidence_score: float = 0.8


class SolutionSynthesizer:
    """Service for synthesizing complete solutions from decomposed requests."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.manifest_service = SystemManifestService(db)
        self.action_planner = ActionPlanner(db)

        # Component templates for quick synthesis
        self.component_templates = {
            "sales_automation": {
                "workflows": [
                    {
                        "name": "lead_qualification",
                        "description": "Qualify and score incoming leads",
                        "steps": ["capture", "enrich", "score", "route"],
                        "automation_level": 0.85
                    },
                    {
                        "name": "opportunity_management",
                        "description": "Manage sales opportunities through pipeline",
                        "steps": ["create", "track", "update", "forecast"],
                        "automation_level": 0.70
                    }
                ],
                "agents": [
                    {
                        "name": "sales_assistant",
                        "type": "ai",
                        "capabilities": ["answer_queries", "provide_insights", "generate_quotes"]
                    }
                ],
                "integrations": ["crm", "email", "calendar"],
                "monitoring": ["pipeline_dashboard", "conversion_metrics"]
            },
            "customer_service": {
                "workflows": [
                    {
                        "name": "ticket_routing",
                        "description": "Route support tickets to right agents",
                        "steps": ["categorize", "prioritize", "assign", "track"],
                        "automation_level": 0.90
                    }
                ],
                "agents": [
                    {
                        "name": "support_bot",
                        "type": "ai",
                        "capabilities": ["answer_faqs", "collect_info", "escalate"]
                    }
                ],
                "integrations": ["helpdesk", "knowledge_base", "chat"],
                "monitoring": ["sla_dashboard", "satisfaction_metrics"]
            }
        }

    async def synthesize_solution(self,
                                 decomposed: DecomposedRequest,
                                 context: ClarificationContext) -> CompleteSolution:
        """Synthesize a complete solution from decomposed request and context.

        Args:
            decomposed: The decomposed request with components
            context: Clarification context with answers

        Returns:
            Complete solution ready for implementation
        """
        # Generate solution ID
        solution_id = str(uuid.uuid4())

        # Create solution name
        solution_name = self._generate_solution_name(decomposed, context)

        # Build solution components
        workflows = await self._synthesize_workflows(decomposed, context)
        agents = await self._synthesize_agents(decomposed, context)
        integrations = await self._synthesize_integrations(decomposed, context)
        monitoring = await self._synthesize_monitoring(decomposed, context)

        # Create implementation plan
        implementation_plan = await self._create_implementation_plan(
            decomposed, workflows, agents, integrations, monitoring, context
        )

        # Generate executable action plan
        action_plan = await self._generate_action_plan(
            workflows, agents, integrations, monitoring, implementation_plan
        )

        # Calculate estimates
        estimates = self._calculate_estimates(
            workflows, agents, integrations, decomposed.total_complexity
        )

        # Generate explanation
        explanation = self._generate_explanation(
            decomposed, workflows, agents, integrations, monitoring, estimates
        )

        return CompleteSolution(
            id=solution_id,
            original_request=decomposed.original_request,
            business_goal=decomposed.business_goal,
            solution_name=solution_name,
            description=f"Complete solution for: {decomposed.business_goal}",
            workflows=workflows,
            agents=agents,
            integrations=integrations,
            monitoring=monitoring,
            implementation_plan=implementation_plan,
            action_plan=action_plan,
            expected_automation_rate=estimates["automation_rate"],
            expected_time_savings=estimates["time_savings"],
            expected_error_reduction=estimates["error_reduction"],
            expected_roi_months=estimates["roi_months"],
            explanation=explanation,
            assumptions=decomposed.assumptions,
            constraints=decomposed.constraints,
            confidence_score=self._calculate_confidence(decomposed, context)
        )

    def _generate_solution_name(self,
                               decomposed: DecomposedRequest,
                               context: ClarificationContext) -> str:
        """Generate a descriptive name for the solution."""
        # Extract key terms from business goal
        goal_terms = decomposed.business_goal.lower().split()
        important_terms = [t for t in goal_terms
                          if t not in ["the", "a", "an", "to", "for", "and", "or"]]

        # Get scale if available
        scale = context.previous_answers.get("user_count", "")

        # Build name
        if "sales" in decomposed.business_goal.lower():
            base_name = "Sales Automation Solution"
        elif "customer" in decomposed.business_goal.lower() or "support" in decomposed.business_goal.lower():
            base_name = "Customer Service Solution"
        elif "marketing" in decomposed.business_goal.lower():
            base_name = "Marketing Automation Solution"
        elif "hr" in decomposed.business_goal.lower():
            base_name = "HR Process Solution"
        else:
            base_name = "Business Automation Solution"

        # Add scale qualifier if available
        if scale == "200+":
            return f"Enterprise {base_name}"
        elif scale == "1-10":
            return f"Team {base_name}"

        return base_name

    async def _synthesize_workflows(self,
                                   decomposed: DecomposedRequest,
                                   context: ClarificationContext) -> List[Dict[str, Any]]:
        """Synthesize workflow components."""
        workflows = []

        # Get workflow components from decomposition
        workflow_components = [c for c in decomposed.components
                              if c.type == ComponentType.WORKFLOW]

        # Get available workflow templates from manifest
        manifest = await self.manifest_service.generate_manifest()
        available_templates = {}
        if manifest and "workflow_templates" in manifest:
            available_templates = manifest["workflow_templates"]

        for component in workflow_components:
            # Try to match with available template using basic matching
            template_id = None
            for tid, template in available_templates.items():
                if component.name in template.get("name", "").lower():
                    template_id = tid
                    break

            workflow = {
                "id": str(uuid.uuid4()),
                "name": component.name,
                "description": component.description,
                "template_id": template_id,
                "complexity": component.estimated_complexity,
                "steps": self._generate_workflow_steps(component, context),
                "triggers": self._generate_workflow_triggers(component, context),
                "automation_level": 0.7 if component.estimated_complexity == "simple" else 0.5
            }

            workflows.append(workflow)

        return workflows

    async def _synthesize_agents(self,
                                decomposed: DecomposedRequest,
                                context: ClarificationContext) -> List[Dict[str, Any]]:
        """Synthesize agent components."""
        agents = []

        agent_components = [c for c in decomposed.components
                           if c.type == ComponentType.AGENT]

        for component in agent_components:
            agent = {
                "id": str(uuid.uuid4()),
                "name": component.name,
                "description": component.description,
                "type": "ai" if "ai" in component.name.lower() else "human",
                "capabilities": self._determine_agent_capabilities(component),
                "integration_points": self._determine_agent_integrations(component, context)
            }
            agents.append(agent)

        return agents

    async def _synthesize_integrations(self,
                                      decomposed: DecomposedRequest,
                                      context: ClarificationContext) -> List[Dict[str, Any]]:
        """Synthesize integration components."""
        integrations = []

        # Get explicitly mentioned integrations
        integration_components = [c for c in decomposed.components
                                 if c.type == ComponentType.INTEGRATION]

        # Add integrations from clarification answers
        if "existing_systems" in context.previous_answers:
            systems = context.previous_answers["existing_systems"]
            if isinstance(systems, str):
                systems = [systems]

            for system in systems:
                if system != "none":
                    integration = {
                        "id": str(uuid.uuid4()),
                        "type": system,
                        "name": f"{system}_integration",
                        "description": f"Integration with {system} system",
                        "configuration": self._get_integration_config(system),
                        "data_flow": "bidirectional"
                    }
                    integrations.append(integration)

        # Add integrations from components
        for component in integration_components:
            integration = {
                "id": str(uuid.uuid4()),
                "type": "custom",
                "name": component.name,
                "description": component.description,
                "configuration": {},
                "data_flow": "bidirectional"
            }
            integrations.append(integration)

        return integrations

    async def _synthesize_monitoring(self,
                                    decomposed: DecomposedRequest,
                                    context: ClarificationContext) -> List[Dict[str, Any]]:
        """Synthesize monitoring components."""
        monitoring = []

        monitoring_components = [c for c in decomposed.components
                                if c.type == ComponentType.MONITORING]

        for component in monitoring_components:
            monitor = {
                "id": str(uuid.uuid4()),
                "name": component.name,
                "description": component.description,
                "type": "dashboard" if "dashboard" in component.name else "metrics",
                "metrics": self._determine_metrics(component, decomposed.business_goal),
                "refresh_interval": "real-time" if component.estimated_complexity == "complex" else "5min",
                "alerts": self._determine_alerts(component)
            }
            monitoring.append(monitor)

        # Always add basic monitoring if none specified
        if not monitoring:
            monitoring.append({
                "id": str(uuid.uuid4()),
                "name": "system_health",
                "description": "Basic system health monitoring",
                "type": "dashboard",
                "metrics": ["uptime", "performance", "errors"],
                "refresh_interval": "5min",
                "alerts": ["system_down", "high_error_rate"]
            })

        return monitoring

    def _generate_workflow_steps(self,
                                component: DecomposedComponent,
                                context: ClarificationContext) -> List[Dict[str, Any]]:
        """Generate workflow steps based on component."""
        steps = []

        # Use templates if available
        if "lead" in component.name:
            steps = [
                {"name": "capture", "type": "input", "description": "Capture lead information"},
                {"name": "validate", "type": "validation", "description": "Validate lead data"},
                {"name": "score", "type": "processing", "description": "Score lead quality"},
                {"name": "route", "type": "action", "description": "Route to sales team"}
            ]
        elif "ticket" in component.name:
            steps = [
                {"name": "receive", "type": "input", "description": "Receive support ticket"},
                {"name": "categorize", "type": "processing", "description": "Categorize issue"},
                {"name": "prioritize", "type": "processing", "description": "Set priority"},
                {"name": "assign", "type": "action", "description": "Assign to agent"}
            ]
        else:
            # Generic workflow steps
            steps = [
                {"name": "trigger", "type": "input", "description": "Workflow trigger"},
                {"name": "process", "type": "processing", "description": "Process data"},
                {"name": "decide", "type": "decision", "description": "Make decision"},
                {"name": "execute", "type": "action", "description": "Execute action"}
            ]

        # Add approval step if needed
        if context.previous_answers.get("approval_needed") not in [None, "none"]:
            steps.insert(-1, {"name": "approve", "type": "approval", "description": "Get approval"})

        return steps

    def _generate_workflow_triggers(self,
                                   component: DecomposedComponent,
                                   context: ClarificationContext) -> List[Dict[str, Any]]:
        """Generate workflow triggers."""
        triggers = []

        # Time-based triggers
        if "daily" in component.description.lower() or "recurring" in component.description.lower():
            triggers.append({
                "type": "schedule",
                "schedule": "daily",
                "time": "09:00"
            })

        # Event-based triggers
        triggers.append({
            "type": "event",
            "event": "data_received",
            "source": "api"
        })

        # Manual trigger always available
        triggers.append({
            "type": "manual",
            "description": "Manual trigger via UI"
        })

        return triggers

    def _determine_agent_capabilities(self, component: DecomposedComponent) -> List[str]:
        """Determine agent capabilities based on component."""
        capabilities = []

        name_lower = component.name.lower()
        desc_lower = component.description.lower()

        if "sales" in name_lower or "sales" in desc_lower:
            capabilities = ["answer_queries", "provide_recommendations", "generate_quotes", "track_deals"]
        elif "support" in name_lower or "support" in desc_lower:
            capabilities = ["answer_faqs", "troubleshoot", "escalate", "collect_feedback"]
        elif "assistant" in name_lower:
            capabilities = ["answer_questions", "provide_guidance", "automate_tasks"]
        else:
            capabilities = ["process_requests", "provide_information"]

        return capabilities

    def _determine_agent_integrations(self,
                                     component: DecomposedComponent,
                                     context: ClarificationContext) -> List[str]:
        """Determine what systems the agent should integrate with."""
        integrations = []

        # Based on agent type
        if "sales" in component.name.lower():
            integrations = ["crm", "email", "calendar"]
        elif "support" in component.name.lower():
            integrations = ["ticketing", "knowledge_base"]

        # Add from context
        if "existing_systems" in context.previous_answers:
            systems = context.previous_answers["existing_systems"]
            if isinstance(systems, list):
                integrations.extend(systems)

        return list(set(integrations))

    def _get_integration_config(self, system_type: str) -> Dict[str, Any]:
        """Get integration configuration template."""
        configs = {
            "crm": {
                "auth_type": "oauth2",
                "sync_frequency": "real-time",
                "entities": ["contacts", "leads", "opportunities", "accounts"]
            },
            "email": {
                "auth_type": "oauth2",
                "protocol": "imap/smtp",
                "sync_frequency": "real-time"
            },
            "chat": {
                "auth_type": "webhook",
                "events": ["message", "mention", "command"]
            },
            "database": {
                "auth_type": "connection_string",
                "sync_frequency": "batch",
                "batch_size": 1000
            }
        }

        return configs.get(system_type, {"auth_type": "api_key"})

    def _determine_metrics(self,
                         component: DecomposedComponent,
                         business_goal: str) -> List[str]:
        """Determine metrics to track."""
        metrics = []

        # Based on component name
        if "sales" in component.name.lower() or "sales" in business_goal.lower():
            metrics = ["conversion_rate", "pipeline_value", "deal_velocity", "win_rate"]
        elif "support" in component.name.lower() or "service" in business_goal.lower():
            metrics = ["response_time", "resolution_time", "satisfaction_score", "ticket_volume"]
        elif "marketing" in component.name.lower():
            metrics = ["engagement_rate", "lead_generation", "campaign_roi", "content_performance"]
        else:
            metrics = ["process_time", "error_rate", "completion_rate", "efficiency"]

        return metrics

    def _determine_alerts(self, component: DecomposedComponent) -> List[str]:
        """Determine what alerts to set up."""
        alerts = []

        if component.estimated_complexity == "complex":
            alerts = ["threshold_breach", "anomaly_detected", "sla_violation", "system_error"]
        else:
            alerts = ["threshold_breach", "system_error"]

        return alerts

    async def _create_implementation_plan(self,
                                         decomposed: DecomposedRequest,
                                         workflows: List[Dict],
                                         agents: List[Dict],
                                         integrations: List[Dict],
                                         monitoring: List[Dict],
                                         context: ClarificationContext) -> ImplementationPlan:
        """Create detailed implementation plan."""
        # Organize into phases
        phases = []

        # Phase 1: Foundation
        if integrations:
            phases.append({
                "name": "Foundation",
                "description": "Set up integrations and infrastructure",
                "components": [i["name"] for i in integrations],
                "duration": "3-5 days",
                "milestone": "All systems connected"
            })

        # Phase 2: Automation
        if workflows:
            phases.append({
                "name": "Automation",
                "description": "Implement automated workflows",
                "components": [w["name"] for w in workflows],
                "duration": "1-2 weeks",
                "milestone": "Core processes automated"
            })

        # Phase 3: Intelligence
        if agents:
            phases.append({
                "name": "Intelligence",
                "description": "Deploy AI agents",
                "components": [a["name"] for a in agents],
                "duration": "1 week",
                "milestone": "Agents operational"
            })

        # Phase 4: Optimization
        if monitoring:
            phases.append({
                "name": "Optimization",
                "description": "Set up monitoring and optimization",
                "components": [m["name"] for m in monitoring],
                "duration": "2-3 days",
                "milestone": "Full visibility achieved"
            })

        # Calculate total duration
        total_duration = self._calculate_total_duration(phases)

        # Identify resource requirements
        resources = self._identify_resources(workflows, agents, integrations, context)

        # Map dependencies
        dependencies = self._map_dependencies(workflows, agents, integrations, monitoring)

        # Identify risks
        risks = self._identify_risks(decomposed, context)

        # Define success metrics
        success_metrics = decomposed.success_metrics

        return ImplementationPlan(
            phases=phases,
            total_duration=total_duration,
            resource_requirements=resources,
            dependencies=dependencies,
            risk_mitigations=risks,
            success_metrics=success_metrics,
            rollback_strategy="Phased rollback with checkpoint restoration"
        )

    async def _generate_action_plan(self,
                                   workflows: List[Dict],
                                   agents: List[Dict],
                                   integrations: List[Dict],
                                   monitoring: List[Dict],
                                   implementation_plan: ImplementationPlan) -> ActionPlan:
        """Generate executable action plan."""
        steps = []

        # Generate steps for each phase
        for phase in implementation_plan.phases:
            if phase["name"] == "Foundation":
                for integration in integrations:
                    steps.append(ActionStep(
                        id=str(uuid.uuid4()),
                        name=f"Set up {integration['name']}",
                        type="setup_integration",
                        description=integration["description"],
                        params={"integration": integration},
                        duration_seconds=3600,  # 1 hour estimate
                        requires_confirmation=True
                    ))

            elif phase["name"] == "Automation":
                for workflow in workflows:
                    steps.append(ActionStep(
                        id=str(uuid.uuid4()),
                        name=f"Create {workflow['name']} workflow",
                        type="create_workflow",
                        description=workflow["description"],
                        params={"workflow": workflow},
                        duration_seconds=1800,  # 30 min estimate
                        requires_confirmation=False
                    ))

            elif phase["name"] == "Intelligence":
                for agent in agents:
                    steps.append(ActionStep(
                        id=str(uuid.uuid4()),
                        name=f"Deploy {agent['name']}",
                        type="deploy_agent",
                        description=agent["description"],
                        params={"agent": agent},
                        duration_seconds=2400,  # 40 min estimate
                        requires_confirmation=True
                    ))

            elif phase["name"] == "Optimization":
                for monitor in monitoring:
                    steps.append(ActionStep(
                        id=str(uuid.uuid4()),
                        name=f"Configure {monitor['name']}",
                        type="setup_monitoring",
                        description=monitor["description"],
                        params={"monitor": monitor},
                        duration_seconds=900,  # 15 min estimate
                        requires_confirmation=False
                    ))

        # Calculate total time
        total_seconds = sum(s.duration_seconds for s in steps)

        return ActionPlan(
            id=str(uuid.uuid4()),
            name="Complete Solution Implementation",
            description=f"Implementation plan for: {implementation_plan.phases[0]['description']}",
            steps=steps,
            estimated_time_seconds=total_seconds,
            confidence_score=0.8
        )

    def _calculate_total_duration(self, phases: List[Dict]) -> str:
        """Calculate total implementation duration."""
        # Simple addition of phase durations
        total_days = 0
        for phase in phases:
            duration = phase["duration"]
            if "day" in duration:
                # Extract number of days
                if "-" in duration:
                    # Take average of range
                    parts = duration.split("-")
                    min_days = int(''.join(c for c in parts[0] if c.isdigit()))
                    max_days = int(''.join(c for c in parts[1].split()[0] if c.isdigit()))
                    total_days += (min_days + max_days) / 2
                else:
                    days = int(''.join(c for c in duration if c.isdigit()))
                    total_days += days
            elif "week" in duration:
                # Convert weeks to days
                if "-" in duration:
                    parts = duration.split("-")
                    min_weeks = int(''.join(c for c in parts[0] if c.isdigit()))
                    max_weeks = int(''.join(c for c in parts[1].split()[0] if c.isdigit()))
                    total_days += ((min_weeks + max_weeks) / 2) * 5  # 5 working days per week
                else:
                    weeks = int(''.join(c for c in duration if c.isdigit()))
                    total_days += weeks * 5

        # Convert back to readable format
        if total_days <= 5:
            return f"{int(total_days)} days"
        elif total_days <= 20:
            return f"{int(total_days/5)} weeks"
        else:
            return f"{int(total_days/20)} months"

    def _identify_resources(self,
                           workflows: List[Dict],
                           agents: List[Dict],
                           integrations: List[Dict],
                           context: ClarificationContext) -> List[ResourceRequirement]:
        """Identify required resources."""
        resources = []

        # Compute resources based on scale
        scale = context.previous_answers.get("user_count", "11-50")
        if scale == "200+":
            compute_spec = {"cpu": "8 cores", "ram": "32GB", "storage": "500GB"}
        elif scale == "51-200":
            compute_spec = {"cpu": "4 cores", "ram": "16GB", "storage": "200GB"}
        else:
            compute_spec = {"cpu": "2 cores", "ram": "8GB", "storage": "100GB"}

        resources.append(ResourceRequirement(
            type="compute",
            name="server_infrastructure",
            specification=compute_spec,
            availability="setup_required"
        ))

        # Integration resources
        for integration in integrations:
            resources.append(ResourceRequirement(
                type="integration",
                name=f"{integration['type']}_connector",
                specification={"type": integration["type"], "config": integration.get("configuration", {})},
                availability="immediate"
            ))

        # Human resources for setup
        resources.append(ResourceRequirement(
            type="human",
            name="implementation_specialist",
            specification={"hours": len(workflows) * 2 + len(agents) * 3},
            availability="immediate"
        ))

        return resources

    def _map_dependencies(self,
                         workflows: List[Dict],
                         agents: List[Dict],
                         integrations: List[Dict],
                         monitoring: List[Dict]) -> Dict[str, List[str]]:
        """Map dependencies between components."""
        dependencies = {}

        # Workflows depend on integrations
        for workflow in workflows:
            dependencies[workflow["name"]] = [i["name"] for i in integrations]

        # Agents depend on workflows and integrations
        for agent in agents:
            dependencies[agent["name"]] = (
                [w["name"] for w in workflows[:1]] +  # Depend on first workflow
                [i["name"] for i in integrations[:2]]  # And first 2 integrations
            )

        # Monitoring depends on everything
        for monitor in monitoring:
            dependencies[monitor["name"]] = (
                [w["name"] for w in workflows] +
                [a["name"] for a in agents]
            )

        return dependencies

    def _identify_risks(self,
                       decomposed: DecomposedRequest,
                       context: ClarificationContext) -> List[RiskMitigation]:
        """Identify implementation risks and mitigations."""
        risks = []

        # Complexity risk
        if decomposed.total_complexity == "complex":
            risks.append(RiskMitigation(
                risk="High complexity may lead to delays",
                probability="medium",
                impact="high",
                mitigation="Phased approach with regular checkpoints",
                contingency="Simplify scope if needed"
            ))

        # Integration risk
        if context.previous_answers.get("existing_systems") not in [None, "none"]:
            risks.append(RiskMitigation(
                risk="Integration challenges with existing systems",
                probability="medium",
                impact="medium",
                mitigation="Thorough API testing and fallback options",
                contingency="Manual data transfer if needed"
            ))

        # Scale risk
        if context.previous_answers.get("user_count") == "200+":
            risks.append(RiskMitigation(
                risk="Performance issues at scale",
                probability="low",
                impact="high",
                mitigation="Load testing and performance optimization",
                contingency="Scale infrastructure as needed"
            ))

        # Timeline risk
        if decomposed.constraints.get("timeline") == "urgent":
            risks.append(RiskMitigation(
                risk="Tight timeline may compromise quality",
                probability="high",
                impact="medium",
                mitigation="Focus on MVP features first",
                contingency="Extend timeline if critical issues arise"
            ))

        return risks

    def _calculate_estimates(self,
                            workflows: List[Dict],
                            agents: List[Dict],
                            integrations: List[Dict],
                            complexity: str) -> Dict[str, float]:
        """Calculate solution estimates."""
        # Automation rate based on components
        base_automation = 0.3  # 30% base
        automation_rate = base_automation
        automation_rate += len(workflows) * 0.15  # Each workflow adds 15%
        automation_rate += len(agents) * 0.10  # Each agent adds 10%
        automation_rate = min(0.95, automation_rate)  # Cap at 95%

        # Time savings (hours per week)
        time_per_workflow = 5  # Each workflow saves 5 hours/week
        time_per_agent = 3  # Each agent saves 3 hours/week
        time_savings = len(workflows) * time_per_workflow + len(agents) * time_per_agent

        # Error reduction
        base_error_reduction = 0.2  # 20% base
        error_reduction = base_error_reduction
        error_reduction += len(workflows) * 0.1  # Each workflow reduces errors by 10%
        error_reduction = min(0.8, error_reduction)  # Cap at 80%

        # ROI calculation (months)
        if complexity == "simple":
            roi_months = 3
        elif complexity == "medium":
            roi_months = 6
        else:
            roi_months = 9

        return {
            "automation_rate": automation_rate,
            "time_savings": time_savings,
            "error_reduction": error_reduction,
            "roi_months": roi_months
        }

    def _generate_explanation(self,
                            decomposed: DecomposedRequest,
                            workflows: List[Dict],
                            agents: List[Dict],
                            integrations: List[Dict],
                            monitoring: List[Dict],
                            estimates: Dict[str, float]) -> str:
        """Generate human-readable explanation of the solution."""
        explanation = f"""Based on your request to {decomposed.original_request}, I've created a comprehensive solution:

**What I'll Build:**
- {len(workflows)} automated workflows to streamline your processes
- {len(agents)} AI/human agents to provide intelligent assistance
- {len(integrations)} system integrations to connect your tools
- {len(monitoring)} monitoring dashboards for full visibility

**Implementation Approach:**
The solution will be implemented in phases to ensure smooth deployment:
1. First, I'll set up the necessary integrations and infrastructure
2. Then, I'll implement the automated workflows
3. Next, I'll deploy the intelligent agents
4. Finally, I'll configure monitoring and optimization

**Expected Outcomes:**
- {estimates['automation_rate']*100:.0f}% task automation, reducing manual work significantly
- {estimates['time_savings']:.0f} hours saved per week through automation
- {estimates['error_reduction']*100:.0f}% reduction in errors and inconsistencies
- Return on investment within {estimates['roi_months']} months

**Key Benefits:**
- Streamlined operations with minimal manual intervention
- Real-time visibility into all processes
- Scalable solution that grows with your needs
- Improved accuracy and consistency

This solution is tailored to your specific needs and constraints, ensuring maximum value delivery."""

        return explanation

    def _calculate_confidence(self,
                            decomposed: DecomposedRequest,
                            context: ClarificationContext) -> float:
        """Calculate confidence score for the solution."""
        confidence = 0.5  # Base confidence

        # Increase confidence based on information completeness
        if context.previous_answers:
            confidence += len(context.previous_answers) * 0.05  # Each answer adds 5%

        # Adjust based on complexity
        if decomposed.total_complexity == "simple":
            confidence += 0.2
        elif decomposed.total_complexity == "medium":
            confidence += 0.1
        else:
            confidence -= 0.1

        # Adjust based on domain familiarity
        if any(domain in decomposed.business_goal.lower()
               for domain in ["sales", "support", "marketing"]):
            confidence += 0.1  # Familiar domain

        return min(0.95, max(0.3, confidence))  # Clamp between 30% and 95%