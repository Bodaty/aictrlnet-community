"""
Action Planner Service for Intelligent Assistant.

Plans and previews actions before execution, breaking complex actions into steps,
estimating time and resources, and identifying potential issues.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from services.edition_boundary_detector import EditionBoundaryDetector, EditionTier, EditionBoundary
from services.conversation_escalation_handler import ConversationEscalationHandler
from services.llm_helpers import get_user_llm_settings
from llm import LLMService, UserLLMSettings
from services.unified_analysis_parser import UnifiedAnalysisParser, UnifiedAnalysisResult

logger = logging.getLogger(__name__)

# Unified initial analysis prompt template (LLM Call Optimization)
UNIFIED_INITIAL_ANALYSIS_PROMPT = """
Analyze the following user request and provide a structured response with three components:

USER REQUEST: {user_query}

Provide your analysis in the following JSON format:

{{
  "intent": {{
    "primary_intent": "<workflow_generation|question|clarification>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
  }},
  "domain": {{
    "industry": "<industry_name>",
    "domain_type": "<specific_domain>",
    "confidence": <0.0-1.0>,
    "keywords": ["<keyword1>", "<keyword2>", ...]
  }},
  "company_context": {{
    "company_size": "<startup|small|medium|large|enterprise|unknown>",
    "business_type": "<b2b|b2c|b2b2c|unknown>",
    "automation_needs": ["<need1>", "<need2>", ...],
    "specific_processes": ["<process1>", "<process2>", ...]
  }}
}}

IMPORTANT:
- Return ONLY the JSON object with no additional text before or after it
- Your response must start with { and end with }
- Do not include markdown code fences (```json or ```)
- Do not add explanatory text, comments, or notes after the JSON
- Be concise and accurate
- Use standard industry terminology
- If information is not explicit in the request, use "unknown" or leave empty arrays
- Confidence should reflect certainty based on available information
"""


class StepType(str, Enum):
    """Types of action steps."""
    CREATE_RESOURCE = "create_resource"
    CONFIGURE = "configure"
    CONNECT = "connect"
    ENABLE_FEATURE = "enable_feature"
    VALIDATE = "validate"
    DEPLOY = "deploy"


@dataclass
class ActionStep:
    """Represents a single step in an action plan."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    type: StepType = StepType.CREATE_RESOURCE
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: int = 10
    requires_confirmation: bool = False
    dependencies: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)
    rollback_action: Optional[str] = None


@dataclass
class ResourceRequirement:
    """Resource required for action execution."""
    name: str
    type: str  # api_key, license, permission, etc.
    status: str = "required"  # required, optional, available
    description: str = ""


@dataclass
class ActionPlan:
    """Complete action plan with all details."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    steps: List[ActionStep] = field(default_factory=list)
    estimated_time_seconds: int = 0
    required_resources: List[ResourceRequirement] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)
    rollback_strategy: Optional[str] = None
    confidence_score: float = 0.95
    # Removed created_at to avoid JSON serialization issues


@dataclass
class ActionPreview:
    """Preview of what will be created/modified."""
    summary: str = ""
    details: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    resources_to_create: List[Dict[str, str]] = field(default_factory=list)
    estimated_time: str = ""
    requires_approval: bool = False


class ActionPlanner:
    """
    Plans and previews actions before execution.
    Breaks down complex requests into manageable steps.
    """

    def __init__(self, db: AsyncSession, user_edition: str = "community"):
        self.db = db
        self.manifest_service = None
        self.knowledge_retrieval = None
        self.llm_service = None
        self._initialized = False
        # Keep templates for fallback compatibility
        self._plan_templates = self._initialize_plan_templates()
        # Edition escalation components
        self.user_edition = EditionTier(user_edition)
        self.boundary_detector = EditionBoundaryDetector(self.user_edition)
        self.escalation_handler = ConversationEscalationHandler()

    def _initialize_plan_templates(self) -> Dict[str, Any]:
        """Initialize common action plan templates."""
        return {
            "customer_service_setup": {
                "name": "Customer Service Automation",
                "steps": [
                    {
                        "name": "Create Customer Service Agent",
                        "type": StepType.CREATE_RESOURCE,
                        "duration": 5,
                        "params": {"agent_type": "customer_service", "capabilities": ["respond", "escalate", "log"]}
                    },
                    {
                        "name": "Create Ticket Routing Workflow",
                        "type": StepType.CREATE_RESOURCE,
                        "duration": 10,
                        "params": {"template": "ticket_routing", "auto_assign": True}
                    },
                    {
                        "name": "Configure Response Templates",
                        "type": StepType.CONFIGURE,
                        "duration": 15,
                        "params": {"templates": ["greeting", "acknowledgment", "resolution"]}
                    },
                    {
                        "name": "Connect Communication Channels",
                        "type": StepType.CONNECT,
                        "duration": 20,
                        "params": {"channels": ["email", "slack", "web_chat"]},
                        "requires_confirmation": True
                    },
                    {
                        "name": "Enable Sentiment Analysis",
                        "type": StepType.ENABLE_FEATURE,
                        "duration": 5,
                        "params": {"feature": "sentiment_analysis", "threshold": 0.3}
                    }
                ],
                "resources": [
                    {"name": "slack_api_key", "type": "api_key", "status": "optional"},
                    {"name": "email_smtp_credentials", "type": "credentials", "status": "required"}
                ]
            },
            "data_processing_pipeline": {
                "name": "Data Processing Pipeline",
                "steps": [
                    {
                        "name": "Create Data Ingestion Agent",
                        "type": StepType.CREATE_RESOURCE,
                        "duration": 5,
                        "params": {"agent_type": "data_processor", "batch_size": 1000}
                    },
                    {
                        "name": "Configure Data Sources",
                        "type": StepType.CONFIGURE,
                        "duration": 10,
                        "params": {"sources": ["database", "api", "file_upload"]}
                    },
                    {
                        "name": "Create Transformation Workflow",
                        "type": StepType.CREATE_RESOURCE,
                        "duration": 15,
                        "params": {"transformations": ["clean", "normalize", "enrich"]}
                    },
                    {
                        "name": "Setup Data Validation",
                        "type": StepType.CONFIGURE,
                        "duration": 10,
                        "params": {"validation_rules": ["completeness", "accuracy", "consistency"]}
                    },
                    {
                        "name": "Configure Output Destinations",
                        "type": StepType.CONFIGURE,
                        "duration": 10,
                        "params": {"destinations": ["warehouse", "dashboard", "api"]},
                        "requires_confirmation": True
                    }
                ],
                "resources": [
                    {"name": "database_connection", "type": "connection_string", "status": "required"},
                    {"name": "storage_bucket", "type": "storage", "status": "required"}
                ]
            },
            "monitoring_system": {
                "name": "Monitoring & Alerting System",
                "steps": [
                    {
                        "name": "Create Monitoring Agent",
                        "type": StepType.CREATE_RESOURCE,
                        "duration": 5,
                        "params": {"agent_type": "monitor", "check_interval": 60}
                    },
                    {
                        "name": "Configure Metrics",
                        "type": StepType.CONFIGURE,
                        "duration": 10,
                        "params": {"metrics": ["uptime", "response_time", "error_rate"]}
                    },
                    {
                        "name": "Setup Alert Rules",
                        "type": StepType.CONFIGURE,
                        "duration": 10,
                        "params": {"alerts": ["threshold", "anomaly", "trend"]}
                    },
                    {
                        "name": "Configure Notifications",
                        "type": StepType.CONFIGURE,
                        "duration": 5,
                        "params": {"channels": ["email", "slack", "sms"]},
                        "requires_confirmation": True
                    }
                ],
                "resources": [
                    {"name": "notification_endpoints", "type": "configuration", "status": "required"}
                ]
            }
        }

    async def _initialize_services(self):
        """Initialize Phase 1 services for dynamic discovery."""
        if self._initialized:
            return

        from services.knowledge.system_manifest_service import SystemManifestService
        from services.knowledge.knowledge_retrieval_service import KnowledgeRetrievalService
        from services.knowledge.knowledge_cache import knowledge_cache

        # Initialize manifest service
        self.manifest_service = SystemManifestService(self.db)

        # Initialize knowledge retrieval using cache to prevent repeated initialization
        # This addresses the 74x initialization issue causing timeout
        self.knowledge_retrieval = await knowledge_cache.get_or_create(
            KnowledgeRetrievalService,
            self.db
        )

        # Try to get LLM service (Business edition)
        try:
            from llm.service import llm_service
            self.llm_service = llm_service
            logger.info("LLM service available for dynamic plan generation")
        except ImportError:
            logger.info("LLM service not available in Community edition - using keyword matching")

        self._initialized = True
        logger.info("Action Planner initialized with Phase 1 services")

    async def _generate_dynamic_plan(
        self,
        intent: str,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Optional[ActionPlan]:
        """
        Generate a dynamic action plan using discovered capabilities.
        For abstract requests, use Phase 4 processing.
        """
        # Get user query from context
        user_query = context.get('user_query', intent)

        # Check if this is an abstract/high-level request
        if await self._is_abstract_request(user_query):
            logger.info(f"[ActionPlanner] Detected abstract request: {user_query}")
            return await self._handle_abstract_request(user_query, context, parameters)

        # Find relevant knowledge using Phase 1 retrieval
        relevant_items = await self.knowledge_retrieval.find_relevant_knowledge(
            query=user_query,
            context=context,
            limit=5
        )

        if not relevant_items:
            logger.info("No relevant knowledge found for dynamic planning")
            return None

        # Generate manifest to understand available capabilities
        manifest = await self.manifest_service.generate_manifest()

        # If we have LLM service, use it for intelligent plan generation
        if self.llm_service:
            return await self._generate_llm_plan(
                user_query, relevant_items, manifest, parameters, context
            )
        else:
            # Community edition: Use keyword matching and templates
            return await self._generate_template_based_plan(
                user_query, relevant_items, parameters
            )

    async def _generate_llm_plan(
        self,
        user_query: str,
        relevant_items: List,
        manifest: Dict,
        parameters: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> ActionPlan:
        """
        Use LLM to generate an intelligent plan based on discovered capabilities.
        """
        # Build context for LLM
        context_str = f"""
User Request: {user_query}

Available Resources:
"""
        for item in relevant_items:
            context_str += f"- {item.name}: {item.data.get('description', 'Available')}\n"

        context_str += f"\nSystem Capabilities: {len(manifest.get('agents', {}))} agents, {len(manifest.get('templates', {}))} templates"

        # Call LLM to generate plan
        prompt = f"""Generate an action plan for: {user_query}

{context_str}

CRITICAL INSTRUCTIONS:
You are creating HIGH-LEVEL plans that delegate to INTELLIGENT services.
DO NOT create detailed implementation steps - our services handle that intelligently!

For these request types, create ONLY ONE delegating step:

1. COMPANY/BUSINESS AUTOMATION (automate my company/business/startup/agency/firm/store):
   Create ONE step that delegates to CompanyAutomationOrchestrator
   The orchestrator will intelligently detect the industry and create appropriate resources

2. WORKFLOW CREATION (create workflow/build workflow/setup process):
   Create ONE step that delegates to workflow generation service
   The service will semantically match templates

3. AGENT CREATION (create agent/build bot/setup assistant):
   Create ONE step that delegates to agent creation service
   The service will match capabilities

EXAMPLES:

For "automate my marketing company" or "automate my startup":
{{
  "steps": [{{
    "name": "Analyzing and setting up company automation",
    "type": "create_resource",
    "duration_seconds": 60,
    "params": {{
      "type": "company_automation",
      "request": "automate my marketing company"
    }}
  }}]
}}

For "create a customer support agent":
{{
  "steps": [{{
    "name": "Creating intelligent agent",
    "type": "create_resource",
    "duration_seconds": 30,
    "params": {{
      "type": "agent",
      "request": "create a customer support agent"
    }}
  }}]
}}

For "build a workflow to process invoices":
{{
  "steps": [{{
    "name": "Creating workflow",
    "type": "create_resource",
    "duration_seconds": 30,
    "params": {{
      "type": "workflow",
      "request": "build a workflow to process invoices"
    }}
  }}]
}}

IMPORTANT RULES:
- Create HIGH-LEVEL delegating steps, not detailed implementation
- The step name should be generic like "Analyzing and setting up company automation"
- Include the FULL original request in params.request for intelligent matching
- DO NOT specify industry, agent_type, or template - let services decide
- DO NOT create multiple steps for company automation - just one delegating step

Return ONLY the JSON object, no explanations."""

        try:
            # Get user settings if we have user context
            user_settings = None
            if context:
                # Try to get user_id from context (could be in session or directly)
                user_id = context.get('user_id')
                if not user_id and context.get('session'):
                    user_id = context['session'].get('user_id') if isinstance(context['session'], dict) else getattr(context.get('session'), 'user_id', None)

                if user_id:
                    user_settings = await get_user_llm_settings(
                        db=self.db,
                        user_id=user_id,
                        task_type="planning"
                    )
                    logger.info(f"Using LLM settings for user {user_id}: {user_settings.selected_model}")

            response = await self.llm_service.generate(
                prompt=prompt,
                user_settings=user_settings,  # Pass user settings to respect preferences
                task_type="planning",
                response_format="json"
            )
            # Log the raw LLM response for debugging
            logger.info(f"LLM full response for '{user_query}': {response.text}")

            # Parse LLM response into ActionPlan
            return self._parse_llm_plan(response.text, parameters)
        except Exception as e:
            logger.error(f"LLM plan generation failed: {e}")
            return None

    async def _generate_template_based_plan(
        self,
        user_query: str,
        relevant_items: List,
        parameters: Dict[str, Any]
    ) -> ActionPlan:
        """
        Community edition: Generate plan using templates and keyword matching.
        """
        plan = ActionPlan(
            name=f"Dynamic Plan - {user_query[:30]}",
            description=f"Automated plan for: {user_query}"
        )

        # Create steps based on relevant items found
        for item in relevant_items:
            if item.type == 'template':
                step = ActionStep(
                    name=f"Apply {item.name} Template",
                    type=StepType.CREATE_RESOURCE,
                    duration_seconds=30,
                    params={
                        'template_id': item.id,
                        'template_name': item.name,
                        **parameters
                    }
                )
                plan.steps.append(step)
            elif item.type == 'agent':
                step = ActionStep(
                    name=f"Configure {item.name} Agent",
                    type=StepType.CONFIGURE,
                    duration_seconds=20,
                    params={
                        'agent_id': item.id,
                        'agent_type': item.data.get('type', 'generic'),
                        **parameters
                    }
                )
                plan.steps.append(step)

        # If no specific items found, create generic workflow step
        if not plan.steps:
            step = ActionStep(
                name="Create Custom Workflow",
                type=StepType.CREATE_RESOURCE,
                duration_seconds=30,
                params={
                    'type': 'workflow',
                    'description': user_query,
                    **parameters
                }
            )
            plan.steps.append(step)

        plan.estimated_time_seconds = sum(step.duration_seconds for step in plan.steps)
        return plan

    def _parse_llm_plan(self, llm_response: str, parameters: Dict[str, Any]) -> ActionPlan:
        """
        Parse LLM response into ActionPlan structure.
        """
        import json

        plan = ActionPlan(
            name=parameters.get("name", "AI-Generated Plan"),
            description=parameters.get("description", "")
        )

        try:
            # Try to extract JSON from response - handle both object and array formats
            json_str = None
            if '{' in llm_response:
                json_start = llm_response.index('{')
                json_end = llm_response.rfind('}') + 1
                json_str = llm_response[json_start:json_end]
            elif '[' in llm_response:
                # LLM returned just an array, wrap it
                json_start = llm_response.index('[')
                json_end = llm_response.rfind(']') + 1
                array_str = llm_response[json_start:json_end]
                json_str = f'{{"steps": {array_str}}}'

            if json_str:
                plan_data = json.loads(json_str)
                logger.info(f"Parsed LLM plan data: {plan_data}")

                for step_data in plan_data.get('steps', []):
                    # Try to parse step type, skip invalid ones
                    step_type_str = step_data.get('type', 'create_resource')
                    try:
                        step_type = StepType(step_type_str)
                    except ValueError:
                        logger.warning(f"Skipping step with invalid type '{step_type_str}': {step_data.get('name', 'Unknown')}")
                        continue  # Skip this step but continue with others

                    step = ActionStep(
                        name=step_data.get('name', 'Action Step'),
                        type=step_type,
                        duration_seconds=step_data.get('duration_seconds', 30),
                        params=step_data.get('params', {})
                    )
                    plan.steps.append(step)

                # If all steps were invalid, treat as parse failure
                if not plan.steps and plan_data.get('steps'):
                    logger.warning("All steps had invalid types, using fallback")
                    raise ValueError("No valid steps found after filtering")
        except Exception as e:
            logger.warning(f"Failed to parse LLM response, using fallback: {e}")
            # If parsing fails, create a simple step based on user's request
            # Infer resource type from parameters or user query
            resource_type = parameters.get('type', 'workflow')  # Default to workflow
            resource_name = parameters.get('name', parameters.get('user_query', 'Resource'))

            # Try to infer type from intent or query
            if 'agent' in parameters.get('user_query', '').lower():
                resource_type = 'agent'
                step_name = f"Create {resource_name}"
            else:
                step_name = "Execute AI-Planned Workflow"

            step = ActionStep(
                name=step_name,
                type=StepType.CREATE_RESOURCE,
                duration_seconds=60,
                params={
                    'type': resource_type,
                    'name': resource_name,
                    'ai_generated': True,
                    **parameters
                }
            )
            plan.steps.append(step)

        plan.estimated_time_seconds = sum(step.duration_seconds for step in plan.steps)
        return plan

    async def plan_action(
        self,
        intent: str,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> ActionPlan:
        """
        Create a detailed action plan based on intent and context.
        Now uses dynamic discovery instead of hardcoded templates.

        Args:
            intent: The detected intent (e.g., "create_customer_service")
            context: Current conversation context
            parameters: Extracted parameters from conversation

        Returns:
            Detailed action plan with steps, time estimates, and resources
        """
        # Initialize services if not already done
        await self._initialize_services()

        # First, try dynamic discovery approach
        dynamic_plan = await self._generate_dynamic_plan(intent, context, parameters)
        if dynamic_plan:
            logger.info(f"Generated dynamic plan with {len(dynamic_plan.steps)} steps")
            return dynamic_plan

        # Fallback to template matching for backward compatibility
        logger.info("Falling back to template-based planning")
        plan = ActionPlan(
            name=parameters.get("name", f"Action Plan - {intent}"),
            description=parameters.get("description", "")
        )

        # Map intent to plan template
        template_key = self._map_intent_to_template(intent)

        if template_key and template_key in self._plan_templates:
            template = self._plan_templates[template_key]

            # Build steps from template
            for step_data in template["steps"]:
                step = ActionStep(
                    name=step_data["name"],
                    type=step_data.get("type", StepType.CREATE_RESOURCE),
                    duration_seconds=step_data.get("duration", 10),
                    params=self._customize_params(step_data.get("params", {}), parameters),
                    requires_confirmation=step_data.get("requires_confirmation", False)
                )

                # Add potential issues based on context
                step.potential_issues = self._identify_step_issues(step, context)

                # Define rollback action
                step.rollback_action = self._define_rollback(step)

                plan.steps.append(step)

            # Add resource requirements
            for resource_data in template.get("resources", []):
                resource = ResourceRequirement(
                    name=resource_data["name"],
                    type=resource_data["type"],
                    status=resource_data.get("status", "required"),
                    description=resource_data.get("description", "")
                )
                plan.required_resources.append(resource)
        else:
            # Create a simple single-step plan for unknown intents
            # Special handling for workflow creation intents
            if 'workflow' in intent.lower() or intent.lower() in ['create', 'setup', 'build']:
                step = ActionStep(
                    name=f"Create Workflow",
                    type=StepType.CREATE_RESOURCE,
                    duration_seconds=30,
                    params={
                        'type': 'workflow',
                        'description': context.get('user_query', f"Execute {intent}"),
                        **parameters
                    },
                    requires_confirmation=False  # Already confirmed in Phase 2
                )
            else:
                step = ActionStep(
                    name=f"Execute {intent}",
                    type=StepType.CREATE_RESOURCE,
                    duration_seconds=30,
                    params=parameters,
                    requires_confirmation=True
                )
            plan.steps.append(step)

        # Calculate total estimated time
        plan.estimated_time_seconds = sum(step.duration_seconds for step in plan.steps)

        # Identify overall potential issues
        plan.potential_issues = self._identify_plan_issues(plan, context)

        # Define rollback strategy
        plan.rollback_strategy = self._define_rollback_strategy(plan)

        return plan

    async def preview_action(self, plan: ActionPlan) -> ActionPreview:
        """
        Generate a user-friendly preview of the action plan.

        Args:
            plan: The action plan to preview

        Returns:
            ActionPreview with summary and details
        """
        preview = ActionPreview()

        # Generate summary
        preview.summary = f"Will execute {len(plan.steps)} steps to {plan.name}"

        # Add step details
        for i, step in enumerate(plan.steps, 1):
            icon = "✅" if not step.requires_confirmation else "⚠️"
            time_str = self._format_duration(step.duration_seconds)
            detail = f"{icon} Step {i}: {step.name} ({time_str})"
            preview.details.append(detail)

        # Add warnings for potential issues
        for issue in plan.potential_issues:
            preview.warnings.append(f"⚠️ {issue}")

        # List resources to be created
        for step in plan.steps:
            if step.type == StepType.CREATE_RESOURCE:
                resource_info = {
                    "type": step.params.get("type", "resource"),
                    "name": step.params.get("name", step.name),
                    "description": step.description or f"Created by {step.name}"
                }
                preview.resources_to_create.append(resource_info)

        # Format estimated time
        preview.estimated_time = self._format_duration(plan.estimated_time_seconds)

        # Check if approval needed
        preview.requires_approval = any(
            step.requires_confirmation for step in plan.steps
        )

        return preview

    def _map_intent_to_template(self, intent: str) -> Optional[str]:
        """Map intent to plan template key."""
        intent_mapping = {
            "create_customer_service": "customer_service_setup",
            "setup_customer_support": "customer_service_setup",
            "create_data_pipeline": "data_processing_pipeline",
            "setup_data_processing": "data_processing_pipeline",
            "create_monitoring": "monitoring_system",
            "setup_alerts": "monitoring_system"
        }
        return intent_mapping.get(intent.lower())

    def _customize_params(
        self,
        template_params: Dict[str, Any],
        user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge template params with user-provided params."""
        customized = template_params.copy()

        # Override with user params where provided
        for key, value in user_params.items():
            if key in customized and isinstance(customized[key], dict) and isinstance(value, dict):
                customized[key].update(value)
            else:
                customized[key] = value

        return customized

    def _identify_step_issues(
        self,
        step: ActionStep,
        context: Dict[str, Any]
    ) -> List[str]:
        """Identify potential issues for a step."""
        issues = []

        # Check for missing resources
        if step.type == StepType.CONNECT:
            if "slack" in str(step.params) and not context.get("has_slack_integration"):
                issues.append("Slack integration not configured")

        # Check for permission issues
        if step.requires_confirmation:
            issues.append("Requires user confirmation to proceed")

        return issues

    def _identify_plan_issues(
        self,
        plan: ActionPlan,
        context: Dict[str, Any]
    ) -> List[str]:
        """Identify potential issues for the entire plan."""
        issues = []

        # Check resource availability
        for resource in plan.required_resources:
            if resource.status == "required" and not context.get(f"has_{resource.name}"):
                issues.append(f"Missing required resource: {resource.name}")

        # Check time constraints
        if plan.estimated_time_seconds > 300:  # More than 5 minutes
            issues.append("This operation may take several minutes to complete")

        return issues

    def _define_rollback(self, step: ActionStep) -> Optional[str]:
        """Define rollback action for a step."""
        rollback_actions = {
            StepType.CREATE_RESOURCE: f"delete_{step.params.get('type', 'resource')}",
            StepType.CONFIGURE: f"reset_configuration",
            StepType.CONNECT: f"disconnect",
            StepType.ENABLE_FEATURE: f"disable_feature"
        }
        return rollback_actions.get(step.type)

    def _define_rollback_strategy(self, plan: ActionPlan) -> str:
        """Define overall rollback strategy for the plan."""
        if len(plan.steps) <= 3:
            return "Simple rollback: Undo each step in reverse order"
        else:
            return "Progressive rollback: Undo steps from failure point, preserving stable components"

    def _format_duration(self, seconds: int) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            if remaining_seconds:
                return f"{minutes}m {remaining_seconds}s"
            return f"{minutes}m"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            if remaining_minutes:
                return f"{hours}h {remaining_minutes}m"
            return f"{hours}h"

    async def _is_abstract_request(self, query: str) -> bool:
        """
        Detect if a request is abstract/high-level and needs Phase 4 processing.
        """
        # Keywords that indicate abstract/high-level requests
        abstract_indicators = [
            "entire", "complete", "all", "whole", "full",
            "system", "process", "business", "operations",
            "improve", "optimize", "enhance", "transform",
            "setup", "establish", "implement", "deploy",
            "everything", "comprehensive", "end-to-end"
        ]

        # Company automation patterns - these ARE abstract and need intelligent processing
        company_automation_patterns = [
            "automate my company",
            "automate my business",
            "automate my startup",
            "automate my agency",
            "automate my firm",
            "automate my store",
            "automate my restaurant",
            "automate my clinic",
            "automate my practice"
        ]

        # Vague requests that need clarification
        vague_patterns = [
            "i want to", "i need to", "help me", "can you",
            "set up", "create a", "build a", "make a"
        ]

        query_lower = query.lower()

        # Check for company automation specifically - these ARE abstract
        is_company_automation = any(pattern in query_lower for pattern in company_automation_patterns)
        if is_company_automation:
            logger.info(f"Detected company automation request as abstract: {query}")
            return True

        # Check for abstract indicators
        has_abstract = any(indicator in query_lower for indicator in abstract_indicators)

        # Check for vague patterns with broad scope
        is_vague = any(pattern in query_lower for pattern in vague_patterns)

        # Check if request is too short and broad
        is_broad = len(query.split()) < 10 and ("automate" in query_lower or "improve" in query_lower)

        return has_abstract or (is_vague and is_broad)

    async def _handle_abstract_request(
        self,
        user_query: str,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> ActionPlan:
        """
        Handle abstract requests using Phase 4 processing:
        1. Decompose the request
        2. Generate clarifying questions if needed
        3. Synthesize a complete solution
        """
        # Import Phase 4 services
        from services.abstract_processing import (
            RequestDecomposer,
            ClarificationEngine,
            SolutionSynthesizer,
            ClarificationContext
        )

        # Initialize Phase 4 services
        decomposer = RequestDecomposer(self.db)
        clarification_engine = ClarificationEngine(self.db)
        synthesizer = SolutionSynthesizer(self.db)

        # Step 1: Decompose the abstract request
        logger.info(f"[ActionPlanner] Decomposing abstract request: {user_query}")
        decomposed = await decomposer.decompose_request(user_query)

        # Step 2: Check if we need clarification
        clarification_context = ClarificationContext(
            decomposed_request=decomposed,
            user_history=context.get('user_history', {}),
            industry=context.get('industry'),
            company_size=context.get('company_size')
        )

        # Generate clarifying questions (for now, we'll use defaults)
        questions = await clarification_engine.generate_clarifying_questions(
            clarification_context,
            max_questions=3
        )

        # If we have critical questions, we should ask them
        # For now, we'll proceed with defaults
        if questions:
            # In a full implementation, we'd return these questions to the user
            # For now, use default answers
            answers = {}
            for q in questions:
                if q.options and len(q.options) > 0:
                    answers[q.id] = q.options[0].value

            clarification_context = await clarification_engine.process_answers(
                questions, answers, clarification_context
            )

        # Step 3: Synthesize complete solution
        logger.info(f"[ActionPlanner] Synthesizing solution for: {decomposed.business_goal}")
        solution = await synthesizer.synthesize_solution(decomposed, clarification_context)

        # Convert solution to ActionPlan
        return self._solution_to_action_plan(solution, decomposed, parameters)

    def _solution_to_action_plan(
        self,
        solution,
        decomposed,
        parameters: Dict[str, Any]
    ) -> ActionPlan:
        """
        Convert a Phase 4 solution into an ActionPlan.
        """
        plan = ActionPlan(
            name=solution.solution_name,
            description=f"{decomposed.business_goal}\n\n{solution.description}"
        )

        # Add workflow creation steps
        for workflow in solution.workflows:
            step = ActionStep(
                name=f"Create {workflow['name']}",
                description=workflow.get('description', ''),
                type=StepType.CREATE_RESOURCE,
                duration_seconds=workflow.get('estimated_time', 60),
                params={
                    'type': 'workflow',
                    'name': workflow['name'],
                    'triggers': workflow.get('triggers', []),
                    'actions': workflow.get('actions', []),
                    **parameters
                },
                requires_confirmation=workflow.get('requires_approval', False)
            )
            plan.steps.append(step)

        # Add agent creation steps
        for agent in solution.agents:
            step = ActionStep(
                name=f"Create {agent['name']}",
                description=f"Create {agent.get('type', 'ai')} agent",
                type=StepType.CREATE_RESOURCE,
                duration_seconds=30,
                params={
                    'type': 'agent',
                    'agent_type': agent.get('type', 'ai'),
                    'name': agent['name'],
                    'capabilities': agent.get('capabilities', []),
                    'configuration': agent.get('configuration', {})
                }
            )
            plan.steps.append(step)

        # Add integration steps
        for integration in solution.integrations:
            step = ActionStep(
                name=f"Setup {integration['name']} Integration",
                description=f"Connect to {integration.get('platform', 'external system')}",
                type=StepType.CONNECT,
                duration_seconds=integration.get('setup_time', 45),
                params={
                    'platform': integration.get('platform'),
                    'type': integration.get('type'),
                    'configuration': integration.get('configuration', {})
                },
                requires_confirmation=True
            )
            plan.steps.append(step)

        # Add monitoring configuration
        if solution.monitoring:
            step = ActionStep(
                name="Configure Monitoring & Analytics",
                description="Setup performance tracking and analytics",
                type=StepType.CONFIGURE,
                duration_seconds=30,
                params={
                    'metrics': [m['name'] for m in solution.monitoring],
                    'alerts': [m.get('alert_threshold') for m in solution.monitoring if 'alert_threshold' in m]
                }
            )
            plan.steps.append(step)

        # Calculate total time
        plan.estimated_time_seconds = sum(step.duration_seconds for step in plan.steps)

        # Add metadata about expected outcomes
        plan.metadata = {
            'automation_rate': f"{solution.expected_automation_rate * 100:.0f}%",
            'time_savings': f"{solution.expected_time_savings:.0f} hours/week",
            'roi_months': solution.expected_roi_months,
            'confidence': f"{solution.confidence_score * 100:.0f}%",
            'implementation_phases': decomposed.phases
        }

        # Add resource requirements
        for resource in solution.implementation_plan.resource_requirements:
            plan.required_resources.append({
                'name': resource.name,
                'type': resource.type,
                'status': 'required' if resource.availability == 'immediate' else 'optional',
                'estimated_cost': resource.estimated_cost or 0
            })

        # Add risks as potential issues
        for risk in solution.implementation_plan.risk_mitigations:
            plan.potential_issues.append(
                f"{risk.risk}: {risk.mitigation}"
            )

        return plan

    async def perform_unified_initial_analysis(
        self,
        user_query: str,
        user_llm_settings: 'UserLLMSettings'
    ) -> UnifiedAnalysisResult:
        """
        Perform unified initial analysis combining intent, domain, and company context.

        This replaces the sequential calls to:
        - classify_intent() (Phase 1)
        - identify_domain() (Phase 2)
        - analyze_company_automation() (Phase 3)

        By combining these three FAST-tier LLM calls into one, we reduce latency by ~4-6 seconds
        while maintaining accuracy. This is a key optimization for workflow generation speed.

        Args:
            user_query: User's input text
            user_llm_settings: User's LLM preferences (should use FAST tier for this operation)

        Returns:
            UnifiedAnalysisResult containing all three analysis components:
            - Intent classification (primary_intent, confidence, reasoning)
            - Domain identification (industry, domain_type, keywords)
            - Company context (company_size, business_type, automation_needs)

        Raises:
            ValueError: If LLM response cannot be parsed

        Example:
            >>> result = await action_planner.perform_unified_initial_analysis(
            ...     user_query="generate a workflow for my restaurant",
            ...     user_llm_settings=user_settings  # with FAST tier
            ... )
            >>> print(result.primary_intent)  # "workflow_generation"
            >>> print(result.industry)  # "hospitality"
            >>> print(result.company_size)  # "small"
        """
        # Start performance monitoring
        from datetime import datetime
        start_time = datetime.now()

        try:
            # Format the unified prompt with user query
            formatted_prompt = UNIFIED_INITIAL_ANALYSIS_PROMPT.format(
                user_query=user_query
            )

            logger.info(
                f"[Unified Analysis] Starting unified initial analysis for query: '{user_query[:100]}'",
                extra={
                    "optimization": "unified_initial_analysis",
                    "start_time": start_time.isoformat(),
                    "query_length": len(user_query)
                }
            )

            # Get LLM service instance
            from llm.service import LLMService
            llm_service = LLMService()

            # Measure LLM call time
            llm_start = datetime.now()

            # Call LLM service with FAST tier for quick structured extraction
            llm_response = await llm_service.generate(
                prompt=formatted_prompt,
                user_settings=user_llm_settings,
                task_type="intent_classification",  # Maps to FAST tier
                temperature=0.2,  # Low temperature for structured output
                response_format="json",
                max_tokens=1024,  # Sufficient for full JSON response with all fields (~450-500 tokens needed)
                system_prompt="You are a precise analysis system. Always respond with valid JSON matching the schema provided."
            )

            llm_elapsed = (datetime.now() - llm_start).total_seconds()

            logger.info(
                f"[Unified Analysis] LLM response received in {llm_elapsed:.2f}s "
                f"(model={llm_response.model_used}, tokens={llm_response.tokens_used})",
                extra={
                    "optimization": "unified_initial_analysis",
                    "llm_latency_seconds": llm_elapsed,
                    "model_used": llm_response.model_used,
                    "tokens_used": llm_response.tokens_used
                }
            )

            # Parse the LLM response using UnifiedAnalysisParser
            parse_start = datetime.now()
            parser = UnifiedAnalysisParser()
            result = await parser.parse_response(llm_response.text)
            parse_elapsed = (datetime.now() - parse_start).total_seconds()

            # Calculate total elapsed time
            total_elapsed = (datetime.now() - start_time).total_seconds()

            # Log success with comprehensive metrics
            logger.info(
                f"[LLM Optimization] ✅ Unified analysis completed in {total_elapsed:.2f}s "
                f"(LLM: {llm_elapsed:.2f}s, Parse: {parse_elapsed:.2f}s) | "
                f"Intent: {result.primary_intent} (conf={result.intent_confidence:.2f}), "
                f"Industry: {result.industry} (conf={result.domain_confidence:.2f}), "
                f"Company: {result.company_size}",
                extra={
                    "optimization": "unified_initial_analysis",
                    "success": True,
                    "total_latency_seconds": total_elapsed,
                    "llm_latency_seconds": llm_elapsed,
                    "parse_latency_seconds": parse_elapsed,
                    "intent_confidence": result.intent_confidence,
                    "domain_confidence": result.domain_confidence,
                    "primary_intent": result.primary_intent,
                    "industry": result.industry,
                    "domain_type": result.domain_type,
                    "company_size": result.company_size,
                    "business_type": result.business_type,
                    "automation_needs_count": len(result.automation_needs),
                    "domain_keywords_count": len(result.domain_keywords)
                }
            )

            return result

        except ValueError as e:
            # Calculate elapsed time for failure case
            total_elapsed = (datetime.now() - start_time).total_seconds()

            logger.error(
                f"[LLM Optimization] ❌ Unified analysis FAILED after {total_elapsed:.2f}s: {e}",
                extra={
                    "optimization": "unified_initial_analysis",
                    "success": False,
                    "total_latency_seconds": total_elapsed,
                    "error_type": "ValueError",
                    "error_message": str(e)
                }
            )
            logger.error(f"[Unified Analysis] Raw LLM response: {llm_response.text[:500]}")
            raise

        except Exception as e:
            # Calculate elapsed time for failure case
            total_elapsed = (datetime.now() - start_time).total_seconds()

            logger.error(
                f"[LLM Optimization] ❌ Unified analysis FAILED after {total_elapsed:.2f}s: {e}",
                extra={
                    "optimization": "unified_initial_analysis",
                    "success": False,
                    "total_latency_seconds": total_elapsed,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            logger.exception("Full traceback:")
            raise ValueError(f"Unified analysis failed: {e}") from e

    async def check_edition_boundary(
        self,
        intent: str,
        context: Dict[str, Any],
        requested_adapters: List[str] = None,
        current_usage: Dict[str, int] = None
    ) -> Optional[Dict[str, Any]]:
        """Check if the user's request crosses an edition boundary.

        Args:
            intent: User's primary intent
            context: Conversation context
            requested_adapters: List of adapters needed
            current_usage: Current usage metrics

        Returns:
            Dict with escalation response if boundary detected, None otherwise
        """
        # Analyze intent to determine required features
        requested_features = self.boundary_detector.analyze_intent_for_features(
            intent,
            context
        )

        # Detect boundary
        boundary = self.boundary_detector.detect_boundary(
            intent=intent,
            requested_features=requested_features,
            requested_adapters=requested_adapters or [],
            current_usage=current_usage,
            context=context
        )

        if boundary:
            # Generate conversational escalation message
            escalation_response = self.escalation_handler.generate_escalation_message(
                boundary=boundary,
                include_workaround=True,
                include_upgrade_link=True
            )

            # Add follow-up suggestions
            escalation_response['follow_up_suggestions'] = (
                self.escalation_handler.generate_follow_up_suggestions(
                    boundary, context
                )
            )

            # Format for chat UI
            escalation_response['ui_format'] = (
                self.escalation_handler.format_for_chat_ui(escalation_response)
            )

            logger.info(
                f"Edition boundary detected: {boundary.boundary_type.value} "
                f"(current: {boundary.current_edition.value}, "
                f"required: {boundary.required_edition.value})"
            )

            return escalation_response

        return None