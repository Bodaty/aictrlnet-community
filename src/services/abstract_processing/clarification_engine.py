"""Clarification Engine Service for AICtrlNet Intelligent Assistant.

This service generates smart clarifying questions to refine abstract requests
and gather missing information needed for complete solution synthesis.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from services.abstract_processing.request_decomposer import DecomposedRequest, ComponentType
from services.pattern_learning_service import PatternLearningService

logger = logging.getLogger(__name__)


class QuestionPriority(Enum):
    """Priority levels for clarifying questions."""
    CRITICAL = "critical"  # Must be answered for any solution
    HIGH = "high"  # Important for optimal solution
    MEDIUM = "medium"  # Helpful for customization
    LOW = "low"  # Nice to have details


class QuestionType(Enum):
    """Types of clarifying questions."""
    SCALE = "scale"  # About size/volume
    TIMELINE = "timeline"  # About deadlines
    INTEGRATION = "integration"  # About systems to connect
    WORKFLOW = "workflow"  # About process details
    PREFERENCE = "preference"  # About user preferences
    CONSTRAINT = "constraint"  # About limitations
    GOAL = "goal"  # About business objectives


@dataclass
class QuestionOption:
    """An option for a multiple-choice question."""
    value: str
    label: str
    description: Optional[str] = None
    implications: Optional[Dict[str, Any]] = None


@dataclass
class ClarifyingQuestion:
    """A clarifying question to ask the user."""
    id: str
    text: str
    type: QuestionType
    priority: QuestionPriority
    options: Optional[List[QuestionOption]] = None
    default_value: Optional[str] = None
    why_asking: str = ""
    skip_option: Optional[str] = None
    depends_on: Optional[str] = None  # ID of another question
    validation: Optional[Dict[str, Any]] = None


@dataclass
class ClarificationContext:
    """Context for generating clarifying questions."""
    decomposed_request: DecomposedRequest
    user_history: Optional[Dict] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None
    previous_answers: Dict[str, Any] = field(default_factory=dict)


class ClarificationEngine:
    """Service for generating smart clarifying questions."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.pattern_service = PatternLearningService(db)

        # Question templates by type
        self.question_templates = {
            QuestionType.SCALE: {
                "user_count": {
                    "text": "How many users will be using this system?",
                    "options": [
                        QuestionOption("1-10", "Small team (1-10 users)"),
                        QuestionOption("11-50", "Department (11-50 users)"),
                        QuestionOption("51-200", "Division (51-200 users)"),
                        QuestionOption("200+", "Enterprise (200+ users)")
                    ],
                    "why": "This helps me size the infrastructure appropriately"
                },
                "transaction_volume": {
                    "text": "What's your expected daily transaction volume?",
                    "options": [
                        QuestionOption("low", "Low (< 100/day)", "Simple infrastructure"),
                        QuestionOption("medium", "Medium (100-1000/day)", "Standard setup"),
                        QuestionOption("high", "High (1000-10000/day)", "Scaled infrastructure"),
                        QuestionOption("very_high", "Very High (10000+/day)", "Enterprise scale")
                    ],
                    "why": "This determines the performance requirements"
                }
            },
            QuestionType.TIMELINE: {
                "deadline": {
                    "text": "When do you need this implemented?",
                    "options": [
                        QuestionOption("asap", "As soon as possible"),
                        QuestionOption("1week", "Within a week"),
                        QuestionOption("2weeks", "Within 2 weeks"),
                        QuestionOption("1month", "Within a month"),
                        QuestionOption("flexible", "Flexible timeline")
                    ],
                    "why": "This helps me prioritize the implementation phases"
                },
                "rollout": {
                    "text": "How would you like to roll this out?",
                    "options": [
                        QuestionOption("all", "All at once"),
                        QuestionOption("phased", "Phased approach"),
                        QuestionOption("pilot", "Pilot group first")
                    ],
                    "why": "This affects the deployment strategy"
                }
            },
            QuestionType.INTEGRATION: {
                "existing_systems": {
                    "text": "Which existing systems need to be integrated?",
                    "options": [
                        QuestionOption("crm", "CRM System (Salesforce, HubSpot, etc.)"),
                        QuestionOption("erp", "ERP System (SAP, Oracle, etc.)"),
                        QuestionOption("email", "Email (Gmail, Outlook, etc.)"),
                        QuestionOption("chat", "Chat/Messaging (Slack, Teams, etc.)"),
                        QuestionOption("database", "Database (SQL, MongoDB, etc.)"),
                        QuestionOption("api", "Custom APIs"),
                        QuestionOption("none", "No integrations needed")
                    ],
                    "why": "This determines which connectors and adapters to set up"
                },
                "data_sources": {
                    "text": "Where is your data currently stored?",
                    "options": [
                        QuestionOption("cloud", "Cloud storage (AWS, Google, Azure)"),
                        QuestionOption("onprem", "On-premises servers"),
                        QuestionOption("saas", "SaaS applications"),
                        QuestionOption("mixed", "Mixed environments")
                    ],
                    "why": "This helps configure data access and security"
                }
            },
            QuestionType.WORKFLOW: {
                "current_process": {
                    "text": "How is this process currently handled?",
                    "options": [
                        QuestionOption("manual", "Completely manual"),
                        QuestionOption("partial", "Partially automated"),
                        QuestionOption("spreadsheets", "Spreadsheets and documents"),
                        QuestionOption("existing_tool", "Existing tool/software"),
                        QuestionOption("not_handled", "Not currently handled")
                    ],
                    "why": "This helps me understand the starting point"
                },
                "approval_needed": {
                    "text": "Does this process require approvals?",
                    "options": [
                        QuestionOption("none", "No approvals needed"),
                        QuestionOption("single", "Single approval"),
                        QuestionOption("multi", "Multiple approvals"),
                        QuestionOption("conditional", "Conditional approvals")
                    ],
                    "why": "This determines the workflow complexity"
                }
            },
            QuestionType.PREFERENCE: {
                "ui_preference": {
                    "text": "How would you prefer to interact with the system?",
                    "options": [
                        QuestionOption("chat", "Chat/conversational interface"),
                        QuestionOption("dashboard", "Visual dashboard"),
                        QuestionOption("api", "API/programmatic access"),
                        QuestionOption("mixed", "All of the above")
                    ],
                    "why": "This shapes the user interface design"
                },
                "notification_preference": {
                    "text": "How should the system notify you of important events?",
                    "options": [
                        QuestionOption("email", "Email notifications"),
                        QuestionOption("sms", "SMS/text messages"),
                        QuestionOption("slack", "Slack messages"),
                        QuestionOption("in_app", "In-app notifications"),
                        QuestionOption("webhook", "Webhook to custom system")
                    ],
                    "why": "This configures the notification system"
                }
            },
            QuestionType.CONSTRAINT: {
                "budget": {
                    "text": "Do you have a budget range in mind?",
                    "options": [
                        QuestionOption("minimal", "Minimize costs", "Use free/open source where possible"),
                        QuestionOption("standard", "Standard budget", "Balance cost and features"),
                        QuestionOption("premium", "Premium features ok", "Best solution regardless of cost"),
                        QuestionOption("not_sure", "Not sure yet")
                    ],
                    "why": "This helps select appropriate service tiers"
                },
                "compliance": {
                    "text": "Are there any compliance requirements?",
                    "options": [
                        QuestionOption("none", "No specific requirements"),
                        QuestionOption("gdpr", "GDPR (European privacy)"),
                        QuestionOption("hipaa", "HIPAA (Healthcare)"),
                        QuestionOption("sox", "SOX (Financial)"),
                        QuestionOption("other", "Other industry-specific")
                    ],
                    "why": "This ensures regulatory compliance"
                }
            },
            QuestionType.GOAL: {
                "primary_goal": {
                    "text": "What's your primary goal with this automation?",
                    "options": [
                        QuestionOption("efficiency", "Increase efficiency", "Reduce time spent on tasks"),
                        QuestionOption("accuracy", "Improve accuracy", "Reduce errors and mistakes"),
                        QuestionOption("scale", "Enable scaling", "Handle more volume"),
                        QuestionOption("cost", "Reduce costs", "Lower operational expenses"),
                        QuestionOption("experience", "Improve experience", "Better user/customer experience")
                    ],
                    "why": "This helps optimize the solution for your goals"
                },
                "success_metric": {
                    "text": "How will you measure success?",
                    "options": [
                        QuestionOption("time", "Time saved"),
                        QuestionOption("volume", "Transactions processed"),
                        QuestionOption("satisfaction", "User satisfaction"),
                        QuestionOption("revenue", "Revenue impact"),
                        QuestionOption("cost", "Cost reduction")
                    ],
                    "why": "This helps define success metrics"
                }
            }
        }

    async def generate_clarifying_questions(self,
                                          context: ClarificationContext,
                                          max_questions: int = 3) -> List[ClarifyingQuestion]:
        """Generate clarifying questions based on the decomposed request.

        Args:
            context: Context including decomposed request and user info
            max_questions: Maximum number of questions to ask

        Returns:
            List of clarifying questions, prioritized
        """
        questions = []

        # Analyze what information is missing
        missing_info = self._identify_missing_information(context)

        # Generate questions for missing information
        for info_type, info_details in missing_info.items():
            if info_details["priority"] == QuestionPriority.CRITICAL:
                question = self._create_question(info_type, info_details, context)
                if question:
                    questions.append(question)

        # Add high priority questions if room
        if len(questions) < max_questions:
            for info_type, info_details in missing_info.items():
                if info_details["priority"] == QuestionPriority.HIGH:
                    question = self._create_question(info_type, info_details, context)
                    if question and len(questions) < max_questions:
                        questions.append(question)

        # Learn from patterns - what questions were helpful in similar requests
        pattern_questions = await self._get_pattern_based_questions(context)
        for pq in pattern_questions:
            if len(questions) < max_questions and pq not in questions:
                questions.append(pq)

        # Sort by priority and limit
        questions.sort(key=lambda q: self._question_priority_value(q.priority), reverse=True)
        return questions[:max_questions]

    def _identify_missing_information(self,
                                     context: ClarificationContext) -> Dict[str, Dict]:
        """Identify what information is missing for complete solution."""
        missing = {}
        decomposed = context.decomposed_request

        # Check scale information
        if not context.company_size and not decomposed.constraints.get("scale"):
            missing["user_count"] = {
                "type": QuestionType.SCALE,
                "priority": QuestionPriority.HIGH,
                "template": "user_count"
            }

        # Check timeline if not specified
        if not decomposed.constraints.get("timeline"):
            missing["deadline"] = {
                "type": QuestionType.TIMELINE,
                "priority": QuestionPriority.HIGH,
                "template": "deadline"
            }

        # Check integration needs if workflows exist
        has_workflows = any(c.type == ComponentType.WORKFLOW for c in decomposed.components)
        if has_workflows and "integration" not in context.previous_answers:
            missing["existing_systems"] = {
                "type": QuestionType.INTEGRATION,
                "priority": QuestionPriority.CRITICAL,
                "template": "existing_systems"
            }

        # Check workflow details if complex
        if decomposed.total_complexity == "complex":
            if "approval" not in context.previous_answers:
                missing["approval_needed"] = {
                    "type": QuestionType.WORKFLOW,
                    "priority": QuestionPriority.MEDIUM,
                    "template": "approval_needed"
                }

        # Check compliance for certain industries
        if context.industry in ["healthcare", "finance", "legal"]:
            if "compliance" not in context.previous_answers:
                missing["compliance"] = {
                    "type": QuestionType.CONSTRAINT,
                    "priority": QuestionPriority.CRITICAL,
                    "template": "compliance"
                }

        # Check primary goal if not clear
        if not decomposed.business_goal or "improve" in decomposed.business_goal.lower():
            missing["primary_goal"] = {
                "type": QuestionType.GOAL,
                "priority": QuestionPriority.HIGH,
                "template": "primary_goal"
            }

        return missing

    def _create_question(self,
                        info_id: str,
                        info_details: Dict,
                        context: ClarificationContext) -> Optional[ClarifyingQuestion]:
        """Create a clarifying question from template."""
        question_type = info_details["type"]
        template_id = info_details["template"]

        if question_type not in self.question_templates:
            return None

        if template_id not in self.question_templates[question_type]:
            return None

        template = self.question_templates[question_type][template_id]

        # Check if already answered
        if template_id in context.previous_answers:
            return None

        question = ClarifyingQuestion(
            id=info_id,
            text=template["text"],
            type=question_type,
            priority=info_details["priority"],
            options=template.get("options"),
            why_asking=template.get("why", "This helps customize the solution"),
            skip_option="Use smart defaults"
        )

        # Add default based on patterns or context
        if context.user_history:
            default = self._get_default_from_history(template_id, context.user_history)
            if default:
                question.default_value = default

        return question

    async def _get_pattern_based_questions(self,
                                          context: ClarificationContext) -> List[ClarifyingQuestion]:
        """Get questions based on learned patterns."""
        questions = []

        # Get relevant patterns
        pattern_context = {
            "primary_intent": context.decomposed_request.business_goal,
            "domain": context.industry
        }
        patterns = await self.pattern_service.get_relevant_patterns(pattern_context)

        # Extract question patterns
        for pattern in patterns:
            if pattern.pattern_type == "parameter":
                # These patterns show which parameters were needed
                pattern_data = pattern.pattern_data
                if "extracted_params" in pattern_data:
                    for param in pattern_data["extracted_params"]:
                        # Map parameter to question type
                        question = self._param_to_question(param)
                        if question:
                            questions.append(question)

        return questions

    def _param_to_question(self, param_name: str) -> Optional[ClarifyingQuestion]:
        """Convert a parameter name to a clarifying question."""
        param_questions = {
            "user_count": ("user_count", QuestionType.SCALE),
            "timeline": ("deadline", QuestionType.TIMELINE),
            "systems": ("existing_systems", QuestionType.INTEGRATION),
            "budget": ("budget", QuestionType.CONSTRAINT),
            "compliance": ("compliance", QuestionType.CONSTRAINT)
        }

        if param_name in param_questions:
            template_id, q_type = param_questions[param_name]
            if q_type in self.question_templates and template_id in self.question_templates[q_type]:
                template = self.question_templates[q_type][template_id]
                return ClarifyingQuestion(
                    id=f"pattern_{param_name}",
                    text=template["text"],
                    type=q_type,
                    priority=QuestionPriority.MEDIUM,
                    options=template.get("options"),
                    why_asking="Previous similar requests needed this information"
                )
        return None

    def _get_default_from_history(self, template_id: str, history: Dict) -> Optional[str]:
        """Get default answer based on user history."""
        # Check if user has answered this before
        if "previous_answers" in history:
            if template_id in history["previous_answers"]:
                return history["previous_answers"][template_id]

        # Check common defaults
        if "company_size" in history:
            size = history["company_size"]
            if template_id == "user_count":
                size_mapping = {
                    "small": "1-10",
                    "medium": "11-50",
                    "large": "51-200",
                    "enterprise": "200+"
                }
                return size_mapping.get(size)

        return None

    def _question_priority_value(self, priority: QuestionPriority) -> int:
        """Convert priority to numeric value for sorting."""
        values = {
            QuestionPriority.CRITICAL: 4,
            QuestionPriority.HIGH: 3,
            QuestionPriority.MEDIUM: 2,
            QuestionPriority.LOW: 1
        }
        return values.get(priority, 0)

    async def process_answers(self,
                            questions: List[ClarifyingQuestion],
                            answers: Dict[str, str],
                            context: ClarificationContext) -> ClarificationContext:
        """Process user answers and update context.

        Args:
            questions: Questions that were asked
            answers: User's answers (question_id -> answer)
            context: Current context

        Returns:
            Updated context with answers incorporated
        """
        # Update previous answers
        context.previous_answers.update(answers)

        # Process each answer
        for question in questions:
            if question.id in answers:
                answer = answers[question.id]

                # Find the selected option
                if question.options:
                    for option in question.options:
                        if option.value == answer:
                            # Apply implications if any
                            if option.implications:
                                self._apply_implications(option.implications, context)
                            break

                # Update constraints based on answer type
                if question.type == QuestionType.SCALE:
                    context.decomposed_request.constraints["scale"] = answer
                elif question.type == QuestionType.TIMELINE:
                    context.decomposed_request.constraints["timeline"] = answer
                elif question.type == QuestionType.CONSTRAINT:
                    if "compliance" in question.id:
                        context.decomposed_request.constraints["compliance"] = answer
                    elif "budget" in question.id:
                        context.decomposed_request.constraints["budget"] = answer
                elif question.type == QuestionType.GOAL:
                    if "primary_goal" in question.id:
                        context.decomposed_request.business_goal = f"Primary goal: {answer}"

        return context

    def _apply_implications(self, implications: Dict[str, Any], context: ClarificationContext):
        """Apply implications from an answer to the context."""
        for key, value in implications.items():
            if key == "complexity":
                context.decomposed_request.total_complexity = value
            elif key == "components":
                # Add or modify components based on implication
                pass  # Implementation depends on specific needs
            else:
                # Add to constraints
                context.decomposed_request.constraints[key] = value

    def generate_followup_questions(self,
                                   initial_answer: str,
                                   question_type: QuestionType) -> List[ClarifyingQuestion]:
        """Generate follow-up questions based on an answer.

        Args:
            initial_answer: The answer that triggers follow-up
            question_type: Type of the original question

        Returns:
            List of follow-up questions
        """
        followups = []

        # Integration follow-ups
        if question_type == QuestionType.INTEGRATION and initial_answer != "none":
            if initial_answer == "crm":
                followups.append(ClarifyingQuestion(
                    id="crm_system",
                    text="Which CRM system are you using?",
                    type=QuestionType.INTEGRATION,
                    priority=QuestionPriority.HIGH,
                    options=[
                        QuestionOption("salesforce", "Salesforce"),
                        QuestionOption("hubspot", "HubSpot"),
                        QuestionOption("dynamics", "Microsoft Dynamics"),
                        QuestionOption("other", "Other CRM")
                    ],
                    why_asking="This helps configure the specific integration"
                ))

        # Scale follow-ups
        elif question_type == QuestionType.SCALE and initial_answer == "200+":
            followups.append(ClarifyingQuestion(
                id="enterprise_features",
                text="Which enterprise features are important?",
                type=QuestionType.PREFERENCE,
                priority=QuestionPriority.MEDIUM,
                options=[
                    QuestionOption("sso", "Single Sign-On"),
                    QuestionOption("multi_tenant", "Multi-tenancy"),
                    QuestionOption("audit", "Audit logging"),
                    QuestionOption("ha", "High availability")
                ],
                why_asking="This helps configure enterprise-specific features"
            ))

        return followups