"""
Enhanced Conversation Manager Service with Knowledge Grounding

Extends the existing conversation manager with system knowledge capabilities.
This enables the intelligent assistant to provide informed, context-aware responses.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from services.conversation_manager import ConversationManagerService
from services.knowledge.knowledge_retrieval_service import KnowledgeRetrievalService, KnowledgeItem
from services.knowledge.system_manifest_service import get_manifest_service
from services.action_planner import ActionPlanner
from services.progressive_executor import ProgressiveExecutor
from services.pattern_learning_service import PatternLearningService
from services.proactive_assistant import ProactiveAssistant
from services.system_prompt_assembler import SystemPromptAssembler
from schemas.conversation import ConversationResponse, IntentDetectionResponse

logger = logging.getLogger(__name__)


class EnhancedConversationService(ConversationManagerService):
    """
    Extends existing conversation service with system knowledge grounding.
    Maintains backward compatibility while adding intelligent assistance.
    """

    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self.knowledge_service = KnowledgeRetrievalService(db)
        self.manifest_service = None
        self.action_planner = ActionPlanner(db)
        self.progressive_executor = ProgressiveExecutor(db)
        self.pattern_service = PatternLearningService(db)
        self.proactive_assistant = ProactiveAssistant(db)
        self.prompt_assembler = SystemPromptAssembler(db)
        self._knowledge_initialized = False

        # Initialize LLM service for intelligent responses
        # NOTE: Use _enhanced_llm_service to avoid conflict with ToolAwareConversationService's llm_service property
        self._enhanced_llm_service = None
        try:
            from llm.service import LLMService
            self._enhanced_llm_service = LLMService()
            logger.info("[EnhancedConversation] LLM service initialized for intelligent responses")
        except ImportError:
            logger.info("[EnhancedConversation] LLM service not available - using fallback")

    async def initialize_knowledge(self):
        """Initialize knowledge services if not already done."""
        if self._knowledge_initialized:
            return

        await self.knowledge_service.initialize()
        self.manifest_service = await get_manifest_service(self.db)
        self._knowledge_initialized = True
        logger.info("[EnhancedConversation] Knowledge services initialized")

    async def process_message(
        self,
        session_id: UUID,
        content: str,
        user_id: str
    ) -> ConversationResponse:
        """
        Process message with knowledge-grounded responses.
        Overrides parent method to add intelligence.
        """
        # Ensure knowledge is initialized
        await self.initialize_knowledge()

        # Get session with fresh data
        session = await self.get_session(session_id)
        if not session:
            session = await self.create_session(user_id)

        # CRITICAL: Ensure context is loaded properly
        if session.context is None:
            session.context = {}
            logger.warning(f"[Phase 2] Session context was None, initialized to empty dict")

        logger.info(f"[Phase 2] Processing message - State: {session.state}, Context keys: {list(session.context.keys())}")

        # Store user message
        user_message = await self._store_message(
            session_id=session.id,
            role="user",
            content=content
        )

        # Check if user is confirming an action
        if session.state == "confirming_action":
            # Check for confirmation phrases
            content_lower = content.lower()
            if any(phrase in content_lower for phrase in ['yes', 'proceed', 'confirm', 'go ahead', 'do it', 'ok', 'sure']):
                # Check if action plan exists
                if 'pending_action_plan' in session.context:
                    session.context['action_confirmed'] = True
                    logger.info(f"[Phase 2] User confirmed action with plan: {len(session.context['pending_action_plan'].get('steps', []))} steps")
                else:
                    logger.warning(f"[Phase 2] No action plan in context during confirmation!")
                    # Try to recreate the action plan if we have intent
                    if session.primary_intent:
                        logger.info(f"[Phase 2] Recreating action plan for: {session.primary_intent}")
                        from schemas.conversation import IntentDetectionResponse
                        intent_result = IntentDetectionResponse(
                            intent=session.primary_intent,
                            confidence=session.intent_confidence or 0.7,
                            entities=session.extracted_params or {}
                        )
                        # Get the original query from the session messages history
                        # The previous message (before "yes") contains the actual request
                        messages = session.messages[-2:] if len(session.messages) > 1 else []
                        original_query = messages[0].content if messages and len(messages) > 1 else content

                        action_plan = await self._create_action_plan(session, intent_result, original_query)
                        if action_plan:
                            session.context['action_confirmed'] = True
                            logger.info(f"[Phase 2] Recreated action plan successfully")
                    else:
                        logger.error(f"[Phase 2] Cannot confirm - no action plan and no intent!")

                # Mark context as modified for SQLAlchemy
                from sqlalchemy.orm import attributes
                attributes.flag_modified(session, 'context')
                await self.db.commit()
                await self.db.refresh(session)
                logger.info(f"[Phase 2] Confirmation processed")
            elif any(phrase in content_lower for phrase in ['no', 'cancel', 'stop', 'abort', 'wait']):
                session.context['action_confirmed'] = False
                session.context['pending_action_plan'] = None  # Clear the plan
                await self.db.commit()  # Persist the cancellation
                logger.info(f"[Phase 2] User cancelled action")

        # FIX: If action was just confirmed, execute it and return appropriate response
        # This must happen BEFORE generating response from "yes" query
        if session.state == "confirming_action" and session.context.get('action_confirmed'):
            logger.info(f"[Phase 2] Action confirmed - executing immediately before response generation")

            # Execute the confirmed action
            await self._trigger_action(session)

            # Refresh session to get automation_result
            await self.db.refresh(session)

            # Get automation_result from context
            automation_result = session.context.get('automation_result')

            # Generate response from automation_result instead of template search
            if automation_result:
                response_content = await self._format_automation_result_response(automation_result)
                logger.info(f"[Phase 2] Generated response from automation_result")
            else:
                # Fallback: action executed but no automation_result
                response_content = "Action completed successfully."
                logger.warning(f"[Phase 2] No automation_result found after action execution")

            # Store the message with proper response
            assistant_message = await self._store_message(
                session_id=session.id,
                role="assistant",
                content=response_content,
                detected_intent=session.primary_intent or "company_automation",
                intent_confidence=0.95,
                entities={},
                suggested_actions=automation_result.get('actions', []) if automation_result else []
            )

            # Update session state to executing_action
            await self.update_session_state(session.id, "executing_action")

            # Analyze session for patterns after successful action
            try:
                await self.pattern_service.analyze_session(str(session.id))
                logger.info("[Phase 3] Pattern analysis completed")
            except Exception as e:
                logger.warning(f"Pattern analysis failed: {e}")

            return ConversationResponse(
                session_id=session.id,
                message=assistant_message,
                state="executing_action",
                context=session.context,
                quick_actions=automation_result.get('actions', []) if automation_result else [],
                requires_clarification=False,
                clarification_options=[],
                automation_result=automation_result
            )

        # Detect intent with context (uses parent method)
        intent_result = await self._detect_intent_with_context(session, content)

        # NEW: Augment with system knowledge
        knowledge_items = await self.knowledge_service.find_relevant_knowledge(
            query=content,
            context=session.context,
            limit=5
        )

        # Update session with detected intent (use >= instead of >)
        if intent_result.confidence >= 0.6:  # Lower threshold to match state logic
            session.primary_intent = intent_result.intent
            session.intent_confidence = intent_result.confidence
            session.extracted_params = {
                **session.extracted_params,
                **intent_result.entities
            }
            # Persist the session changes
            await self.db.commit()
            logger.info(f"[Phase 2] Intent set: {session.primary_intent} (confidence: {intent_result.confidence})")

        # Generate knowledge-informed response
        response_content = await self._generate_informed_response(
            session=session,
            intent_result=intent_result,
            knowledge_items=knowledge_items,
            user_query=content
        )

        # Get quick actions based on knowledge
        quick_actions = await self._generate_smart_actions(
            session=session,
            intent_result=intent_result,
            knowledge_items=knowledge_items
        )

        # Determine next state
        next_state = await self._determine_smart_state(
            session=session,
            intent_result=intent_result,
            knowledge_items=knowledge_items
        )

        # Handle state-specific actions using Phase 2 services BEFORE storing message
        logger.info(f"[Phase 2] State transition: {session.state} → {next_state}, primary_intent: {session.primary_intent}")

        if next_state == "confirming_action" and session.primary_intent:
            logger.info(f"[Phase 2] Creating action plan for: {session.primary_intent}")
            # When ready to confirm, create the action plan (pass user's query)
            action_plan = await self._create_action_plan(session, intent_result, content)
            if action_plan:
                # Replace knowledge search with action plan when we have one
                response_content = await self._format_action_plan_response(action_plan, content)
                logger.info(f"[Phase 2] Action plan created with {len(action_plan.steps)} steps")
        elif next_state == "confirming_action":
            logger.warning(f"[Phase 2] Confirming state but no primary_intent set!")

        # Store assistant response with potentially updated content
        assistant_message = await self._store_message(
            session_id=session.id,
            role="assistant",
            content=response_content,
            detected_intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            entities=intent_result.entities,
            suggested_actions=quick_actions
        )

        # Update session state
        await self.update_session_state(session.id, next_state)

        # Handle execution after state update
        if next_state == "executing_action" and session.primary_intent:
            # Refresh session to get latest context with action plan
            session = await self.get_session(session.id)
            logger.info(f"[Phase 2] Executing action: {session.primary_intent}")
            # Execute the confirmed action
            await self._trigger_action(session)

            # Analyze session for patterns after successful action
            try:
                await self.pattern_service.analyze_session(str(session.id))
                logger.info("[Phase 3] Pattern analysis completed")
            except Exception as e:
                logger.warning(f"Pattern analysis failed: {e}")
        elif next_state == "executing_action":
            logger.warning(f"[Phase 2] Execution state but no primary_intent!")

        # Check if we should show proactive suggestions (with timeout to prevent blocking)
        suggestions = None
        try:
            async with asyncio.timeout(5.0):  # 5 second timeout to prevent blocking workflow confirmation
                if await self.proactive_assistant.should_show_suggestions(user_id, str(session.id)):
                    raw_suggestions = await self.proactive_assistant.generate_suggestions(
                        user_id, session.context
                    )
                    if raw_suggestions:
                        suggestions = await self.proactive_assistant.format_suggestions_for_ui(raw_suggestions)
                        logger.info(f"[Phase 3] Generated {len(raw_suggestions)} proactive suggestions")
        except asyncio.TimeoutError:
            logger.warning("[Phase 3] Proactive suggestions generation timed out (>5s), continuing without suggestions")
        except Exception as e:
            logger.warning(f"[Phase 3] Failed to generate suggestions: {e}")

        # Extract automation_result from context for rich UX display
        automation_result = session.context.get('automation_result')
        if automation_result:
            logger.info(f"[EnhancedConversationManager] Extracting automation_result for response: {list(automation_result.keys())}")

        response = ConversationResponse(
            session_id=session.id,
            message=assistant_message,
            state=next_state,
            context=session.context,
            quick_actions=quick_actions,
            requires_clarification=next_state == "clarifying_details",
            clarification_options=await self._get_clarification_options(session, intent_result),
            automation_result=automation_result
        )

        # Add suggestions if available
        if suggestions:
            response.proactive_suggestions = suggestions

        return response

    async def _generate_informed_response(
        self,
        session,
        intent_result: IntentDetectionResponse,
        knowledge_items: List[KnowledgeItem],
        user_query: str
    ) -> str:
        """
        Generate response grounded in system knowledge.
        """
        # Handle different intent types with knowledge
        # Map various creation intents to the create action
        creation_intents = ["create", "create_workflow", "create_agent", "form_pod", "automate"]

        if intent_result.intent == "how_to":
            return await self._generate_how_to_response(knowledge_items, user_query)
        elif intent_result.intent == "where_is":
            return await self._generate_navigation_response(knowledge_items, user_query)
        elif intent_result.intent in creation_intents:
            return await self._generate_creation_response(knowledge_items, intent_result)
        elif intent_result.intent == "status":
            return await self._generate_status_response(session)
        elif intent_result.intent in ["capabilities", "get_help"]:
            # Handle both "capabilities" (legacy) and "get_help" (current) intents
            return await self._generate_capabilities_response()
        else:
            # General knowledge-informed response
            return await self._generate_general_response(knowledge_items, user_query, intent_result)

    async def _generate_how_to_response(
        self,
        knowledge_items: List[KnowledgeItem],
        user_query: str
    ) -> str:
        """Generate 'how to' response with step-by-step guidance."""
        if not knowledge_items:
            return "I can help you with that. Could you provide more details about what you'd like to do?"

        # Get the most relevant item
        top_item = knowledge_items[0]

        if top_item.type == "feature":
            feature_data = top_item.data
            capabilities = feature_data.get("capabilities", {})

            response = f"To {user_query.lower().replace('how to', '').strip()}, you can:\n\n"

            # List available methods
            for i, (cap_name, cap_data) in enumerate(capabilities.items(), 1):
                response += f"{i}. **{cap_data.get('description', cap_name)}**\n"
                if cap_data.get('ui_path'):
                    response += f"   - Navigate to: `{cap_data['ui_path']}`\n"
                if cap_data.get('endpoint'):
                    response += f"   - Or use API: `{cap_data['endpoint']}`\n"

            # Add examples if available
            if feature_data.get('example_commands'):
                response += "\nExamples:\n"
                for example in feature_data['example_commands'][:2]:
                    response += f"- {example}\n"

            response += "\nWould you like me to help you with any of these options?"

        elif top_item.type == "template":
            template_data = top_item.data
            response = f"You can use the **{template_data['name']}** template:\n\n"
            response += f"**Description**: {template_data.get('description', 'Workflow template')}\n"
            response += f"**Category**: {template_data.get('category', 'general')}\n"
            response += f"**Complexity**: {template_data.get('complexity', 'simple')}\n\n"
            response += "Would you like me to create a workflow from this template?"

        else:
            # Generic response
            response = f"Based on your query, I found {len(knowledge_items)} relevant options. "
            response += "Here are the top suggestions:\n\n"
            for item in knowledge_items[:3]:
                response += f"- **{item.name}** ({item.type}): {item.data.get('description', '')}\n"

        return response

    async def _generate_navigation_response(
        self,
        knowledge_items: List[KnowledgeItem],
        user_query: str
    ) -> str:
        """Generate navigation response with UI locations."""
        if not knowledge_items:
            return "I'm not sure where that feature is located. Could you provide more details?"

        top_item = knowledge_items[0]

        if top_item.data.get('ui_path'):
            location = top_item.data['ui_path']
            response = f"You can find {top_item.name} at:\n\n"
            response += f"📍 **Location**: `{location}`\n\n"
            response += f"Or I can take you there directly: [Go to {top_item.name}]({location})"
        else:
            response = f"The {top_item.name} feature is accessible through:\n"
            if top_item.data.get('endpoint'):
                response += f"- API: `{top_item.data['endpoint']}`\n"
            response += "\nWould you like more information about this feature?"

        return response

    async def _generate_creation_response(
        self,
        knowledge_items: List[KnowledgeItem],
        intent_result: IntentDetectionResponse
    ) -> str:
        """Generate response for creation intents."""
        target = intent_result.entities.get('target', 'resource')

        # Find relevant templates
        templates = [item for item in knowledge_items if item.type == "template"]
        agents = [item for item in knowledge_items if item.type == "agent"]

        response = f"I'll help you create {target}. "

        if templates:
            response += f"I found {len(templates)} relevant templates:\n\n"
            for template in templates[:3]:
                response += f"- **{template.name}**: {template.data.get('description', '')}\n"
            response += "\nWould you like to use one of these templates, or create from scratch?"
        elif agents:
            response += f"I can help you create an agent. Here are some options:\n\n"
            for agent in agents[:3]:
                response += f"- **{agent.name}** ({agent.data.get('type', 'ai')}): "
                response += f"{', '.join(agent.data.get('capabilities', [])[:2])}\n"
            response += "\nWhat type of agent would you like to create?"
        else:
            response += "What specific features would you like to include?"

        return response

    async def _generate_status_response(self, session) -> str:
        """Generate status response about current session/system."""
        capabilities = await self.knowledge_service.get_capabilities_summary()

        response = "**System Status**:\n\n"
        response += f"✅ System ready with:\n"
        response += f"- {capabilities['templates']} workflow templates\n"
        response += f"- {capabilities['agents']} pre-configured agents\n"
        response += f"- {capabilities['adapters']} integrations\n"
        response += f"- {capabilities['automation_coverage']} automation coverage\n\n"

        if session.primary_intent:
            response += f"**Current Context**: Working on '{session.primary_intent}'\n"

        response += "\nWhat would you like to work on?"
        return response

    async def _generate_capabilities_response(self) -> str:
        """Generate response about system capabilities."""
        capabilities = await self.knowledge_service.get_capabilities_summary()

        response = "I'm AICtrlNet's intelligent assistant. I can help you with:\n\n"
        response += "**Core Capabilities**:\n"
        response += "🔄 **Workflow Automation** - Create and manage automated workflows\n"
        response += "🤖 **AI Agent Management** - Deploy and orchestrate AI agents\n"
        response += "🔌 **Integrations** - Connect with external systems\n"
        response += "📊 **Monitoring** - Track performance and analytics\n"
        response += "🛡️ **Governance** - Ensure compliance and quality\n\n"

        response += "**Quick Stats**:\n"
        response += f"- {capabilities['templates']} ready-to-use templates\n"
        response += f"- {capabilities['agents']} pre-configured AI agents\n"
        response += f"- {capabilities['adapters']} integrations available\n\n"

        response += "**Example Scenario**:\n"
        response += "💡 *\"I want to automate customer onboarding\"*\n\n"
        response += "I'll help you create a workflow that:\n"
        response += "• Sends personalized welcome emails via SendGrid\n"
        response += "• Creates onboarding tasks in your project management tool\n"
        response += "• Assigns an AI agent to answer initial customer questions\n"
        response += "• Tracks progress and sends automated follow-ups\n"
        response += "• Notifies your team when human intervention is needed\n\n"

        response += "**Try asking me to**:\n"
        response += "- 'Create a workflow for invoice processing'\n"
        response += "- 'Set up customer service automation'\n"
        response += "- 'Show me what needs attention'\n"
        response += "- 'Automate my sales pipeline'"

        return response

    async def _generate_general_response(
        self,
        knowledge_items: List[KnowledgeItem],
        user_query: str,
        intent_result: IntentDetectionResponse
    ) -> str:
        """Generate general knowledge-informed response."""
        if not knowledge_items:
            # Fallback to capabilities overview
            return await self._generate_capabilities_response()

        response = f"Based on your query about '{user_query}', here's what I found:\n\n"

        # Group items by type
        by_type = {}
        for item in knowledge_items:
            if item.type not in by_type:
                by_type[item.type] = []
            by_type[item.type].append(item)

        # Display grouped results - show ALL items (UI can scroll)
        for item_type, items in by_type.items():
            if item_type == "template":
                response += f"**📄 Templates** ({len(items)}):\n"
            elif item_type == "agent":
                response += f"**🤖 Agents** ({len(items)}):\n"
            elif item_type == "feature":
                response += f"**⚡ Features** ({len(items)}):\n"
            else:
                response += f"**🔧 {item_type.title()}s** ({len(items)}):\n"

            # Show ALL items with full descriptions - UI can scroll
            for item in items:
                description = item.data.get('description', 'No description available')
                response += f"- **{item.name}**: {description}\n"

        response += "\nWould you like to explore any of these options?"
        return response

    async def _generate_smart_actions(
        self,
        session,
        intent_result: IntentDetectionResponse,
        knowledge_items: List[KnowledgeItem]
    ) -> List[Dict[str, Any]]:
        """Generate smart quick actions based on context and knowledge."""
        actions = []

        # Get suggestions from knowledge service
        if session.primary_intent:
            suggestions = await self.knowledge_service.suggest_next_actions(
                current_action=session.primary_intent,
                context=session.context
            )
            # Convert to dict format expected by schema
            for s in suggestions[:3]:
                actions.append({
                    "action": s.get('action', s.get('description', '')),
                    "description": s.get('description', ''),
                    "type": s.get('type', 'suggestion')
                })

        # Add knowledge-based actions
        for item in knowledge_items[:2]:
            if item.type == "template":
                actions.append({
                    "action": "use_template",
                    "description": f"Use {item.name} template",
                    "type": "template",
                    "data": {"template_id": item.id, "template_name": item.name}
                })
            elif item.type == "agent":
                actions.append({
                    "action": "deploy_agent",
                    "description": f"Deploy {item.name} agent",
                    "type": "agent",
                    "data": {"agent_id": item.id, "agent_name": item.name}
                })
            elif item.type == "feature":
                actions.append({
                    "action": "explore_feature",
                    "description": f"Explore {item.name}",
                    "type": "feature",
                    "data": {"feature_id": item.id, "feature_name": item.name}
                })

        # Add default smart actions if needed
        if len(actions) < 3:
            defaults = [
                {"action": "show_examples", "description": "Show me examples", "type": "help"},
                {"action": "show_capabilities", "description": "What can you do?", "type": "help"},
                {"action": "guide_me", "description": "Guide me step-by-step", "type": "help"}
            ]
            actions.extend(defaults[:3-len(actions)])

        return actions[:4]  # Limit to 4 actions

    async def _determine_smart_state(
        self,
        session,
        intent_result: IntentDetectionResponse,
        knowledge_items: List[KnowledgeItem]
    ) -> str:
        """Determine next conversation state intelligently based on conversation flow."""
        current_state = session.state

        # Clear action intents that should trigger planning
        action_intents = [
            'create_workflow',
            'create_agent',
            'create_pod',  # Fixed: was 'form_pod'
            'company_automation',  # Company-level automation
            'deploy',
            'execute'
        ]
        is_action_intent = intent_result.intent in action_intents

        # Reasonable confidence threshold (0.6 not 0.8)
        has_sufficient_confidence = intent_result.confidence >= 0.6

        # State transitions based on conversation flow
        if current_state == "greeting":
            if is_action_intent and has_sufficient_confidence:
                # User wants to do something specific
                if intent_result.missing_params:
                    return "clarifying_details"  # Need more info
                else:
                    return "confirming_action"  # Ready to confirm
            elif intent_result.intent:
                return "gathering_intent"  # Have some intent but not actionable
            else:
                return "gathering_intent"  # Still exploring

        elif current_state == "gathering_intent":
            if is_action_intent and has_sufficient_confidence:
                if intent_result.missing_params:
                    return "clarifying_details"
                else:
                    return "confirming_action"  # Ready for action
            else:
                # Don't get stuck forever - move forward after a few tries
                return "gathering_intent"  # But could add counter logic

        elif current_state == "clarifying_details":
            # Check if we now have all required params
            if not intent_result.missing_params:
                return "confirming_action"
            else:
                return "clarifying_details"  # Still gathering params

        elif current_state == "confirming_action":
            # Check if user confirmed
            if session.context.get('action_confirmed', False):
                return "executing_action"
            else:
                return "confirming_action"  # Still waiting for confirmation

        elif current_state == "executing_action":
            return "completed"

        elif current_state == "completed":
            # Reset for new interaction
            return "greeting"

        # Fallback to current state
        return current_state

    async def _trigger_action(self, session):
        """
        Execute the confirmed action plan using Phase 2 ProgressiveExecutor.
        This implements the guided action execution from the spec.
        """
        try:
            # Ensure we have fresh session data
            await self.db.refresh(session)

            # Debug logging
            logger.info(f"[Phase 2] Executing action - State: {session.state}, Intent: {session.primary_intent}")
            logger.info(f"[Phase 2] Context keys: {list(session.context.keys()) if session.context else 'None'}")

            # Handle empty context
            if not session.context:
                logger.error("[Phase 2] Session context is None or empty")
                session.context = {}

            # Get the action plan from session context
            plan_data = session.context.get('pending_action_plan')
            if not plan_data:
                logger.warning("[Phase 2] No action plan in context, attempting fallback")

                # Fallback: Create simple plan from intent
                if session.primary_intent:
                    import uuid
                    logger.info(f"[Phase 2] Creating fallback plan for: {session.primary_intent}")
                    plan_data = {
                        'id': str(uuid.uuid4()),
                        'name': f"Execute {session.primary_intent}",
                        'description': f"Auto-generated plan for {session.primary_intent}",
                        'steps': [{
                            'id': str(uuid.uuid4()),
                            'action': session.primary_intent,
                            'description': f"Execute {session.primary_intent}",
                            'parameters': session.extracted_params or {},
                            'order': 1
                        }]
                    }
                else:
                    logger.error("[Phase 2] No action plan and no intent to fall back on")
                    return

            # Reconstruct the ActionPlan from the stored data
            from services.action_planner import ActionPlan, ActionStep

            action_plan = ActionPlan(
                id=plan_data.get('id', ''),
                name=plan_data.get('name', ''),
                description=plan_data.get('description', ''),
                steps=[ActionStep(**step) for step in plan_data.get('steps', [])]
            )

            logger.info(f"[Phase 2] Executing plan with {len(action_plan.steps)} steps")

            # Execute using ProgressiveExecutor
            # Note: execute_plan expects (plan, context, confirm_callback)
            execution_result = await self.progressive_executor.execute_plan(
                plan=action_plan,
                context=session.context,
                confirm_callback=None  # Already confirmed
            )

            # Store execution result in session
            # ExecutionProgress has 'status' not 'success'
            from services.progressive_executor import ExecutionStatus

            # Extract workflow ID and automation result if created
            workflow_id = None
            automation_result = None
            if execution_result.results:
                for result in execution_result.results:
                    if result.output:
                        # Extract workflow ID
                        if 'workflow_id' in result.output:
                            workflow_id = result.output['workflow_id']
                            logger.info(f"[Phase 2] Workflow created with ID: {workflow_id}")
                        # Extract company automation result for AutomationSummaryCard
                        # Check for new type='automation_complete' FIRST (has rich UI data)
                        if result.output.get('type') == 'automation_complete':
                            automation_result = result.output  # Full result with summary, workflows, agents, etc.
                            logger.info(f"[Phase 2] Automation complete result extracted with rich UI data: {list(automation_result.keys())}")
                        # Also check for legacy type='company_automation'
                        elif result.output.get('type') == 'company_automation':
                            automation_result = result.output
                            logger.info(f"[Phase 2] Company automation result extracted (legacy)")
                        # Fallback to company_automation sub-dict for backward compatibility
                        elif 'company_automation' in result.output:
                            automation_result = result.output['company_automation']
                            logger.info(f"[Phase 2] Company automation completed (sub-dict): {automation_result.get('organization_id')}")

            session.context['execution_result'] = {
                'success': execution_result.status == ExecutionStatus.COMPLETED,
                'status': execution_result.status,
                'completed_steps': execution_result.completed_steps,
                'failed_steps': execution_result.failed_steps,
                'current_step': execution_result.current_step_index,
                'total_steps': execution_result.total_steps,
                'workflow_id': workflow_id  # Add workflow ID for frontend navigation
            }

            # Add automation_result to context for frontend AutomationSummaryCard
            if automation_result:
                session.context['automation_result'] = automation_result
                logger.info(f"[Phase 2] Added automation_result to context: {list(automation_result.keys())}")

            # CRITICAL: Mark context as modified so SQLAlchemy persists JSON changes
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(session, 'context')
            await self.db.commit()

            logger.info(f"[Phase 2] Execution status: {execution_result.status}, workflow_id: {workflow_id}")

        except Exception as e:
            logger.error(f"[Phase 2] Error during action execution: {e}")
            import traceback
            logger.error(f"[Phase 2] Traceback: {traceback.format_exc()}")

    def _generate_action_preview(self, action_plan) -> str:
        """Generate a user-friendly preview of the action plan."""
        response = "I'll help you with that. Here's what I'll do:\n\n"

        for i, step in enumerate(action_plan.steps, 1):
            response += f"**Step {i}**: {step.description}\n"
            if step.estimated_duration:
                response += f"   ⏱️ Estimated time: {step.estimated_duration} seconds\n"

        response += f"\n**Total estimated time**: {action_plan.estimated_time_seconds} seconds\n"
        response += "\nWould you like me to proceed with this plan?"

        return response

    async def handle_confirmation(self, session, confirmed: bool):
        """
        Handle user confirmation of action plan and execute progressively if confirmed.
        """
        if not confirmed:
            await self.add_assistant_message(
                session.id,
                "No problem! Feel free to ask me to adjust the plan or try something else."
            )
            await self.update_session_state(session.id, "greeting")
            return

        # Get the action plan from context
        context = session.context or {}
        action_plan_data = context.get("action_plan")

        if not action_plan_data:
            await self.add_assistant_message(
                session.id,
                "I couldn't find the action plan. Let me start over."
            )
            return

        # Execute progressively with the ProgressiveExecutor
        try:
            # Convert plan data back to ActionPlan object
            from services.action_planner import ActionPlan, ActionStep

            action_plan = ActionPlan(
                steps=[ActionStep(**step) for step in action_plan_data["steps"]],
                estimated_time_seconds=action_plan_data["estimated_time"],
                required_resources=action_plan_data.get("required_resources", []),
                potential_issues=action_plan_data.get("potential_issues", [])
            )

            # Execute with progress updates
            progress = await self.progressive_executor.execute_plan(
                plan=action_plan,
                context=context,
                confirm_callback=None  # We already confirmed
            )

            # Report completion
            from services.progressive_executor import ExecutionStatus
            if progress.status == ExecutionStatus.COMPLETED:
                await self.add_assistant_message(
                    session.id,
                    f"✅ Successfully completed all {len(progress.completed_steps)} steps!\n\n" +
                    "Your automation is now active and running."
                )
            elif progress.status == ExecutionStatus.FAILED:
                await self.add_assistant_message(
                    session.id,
                    f"⚠️ Completed {len(progress.completed_steps)} of {len(action_plan.steps)} steps.\n\n" +
                    f"Issue: {progress.error_message}\n\n" +
                    "Would you like me to retry or adjust the remaining steps?"
                )
            else:
                await self.add_assistant_message(
                    session.id,
                    f"❌ Unable to complete the action: {progress.error_message}\n\n" +
                    "Let me try a different approach."
                )

            # Update session state
            new_state = "completed" if progress.status == "completed" else "needs_attention"
            await self.update_session_state(session.id, new_state)

        except Exception as e:
            logger.error(f"Error during progressive execution: {e}")
            await self.add_assistant_message(
                session.id,
                "I encountered an error during execution. Let me try again."
            )

    async def _create_action_plan(self, session, intent_result, user_query=None):
        """
        Create action plan using Phase 2 ActionPlanner.
        Returns the plan and stores it in session context.
        """
        try:
            intent = session.primary_intent or intent_result.intent
            params = session.extracted_params or intent_result.entities
            context = session.context or {}

            # Add user's original query to context for NLP service
            if user_query:
                context['user_query'] = user_query

            # Add user_id to context for LLM model selection
            if session.user_id:
                context['user_id'] = session.user_id

            # LLM Call Optimization: Perform unified initial analysis ONCE at the START
            # This replaces 3 sequential FAST-tier calls with 1 unified call (4-6s savings)
            unified_result = None
            if user_query and intent in ['create_workflow', 'workflow_generation', 'automate', 'create']:
                try:
                    from core.database import get_db
                    from models.user import User

                    # Get user LLM settings for tier-based model selection
                    user_llm_settings = None
                    if session.user_id:
                        async for db in get_db():
                            user = await db.get(User, session.user_id)
                            if user and hasattr(user, 'llm_settings'):
                                user_llm_settings = user.llm_settings
                            break

                    # Perform unified analysis (intent + domain + company context in ONE call)
                    unified_result = await self.action_planner.perform_unified_initial_analysis(
                        user_query=user_query,
                        user_llm_settings=user_llm_settings
                    )

                    # Store unified results in session context for downstream use
                    if unified_result:
                        context['unified_analysis'] = {
                            'intent': unified_result.primary_intent,
                            'intent_confidence': unified_result.intent_confidence,
                            'industry': unified_result.industry,
                            'domain_type': unified_result.domain_type,
                            'domain_confidence': unified_result.domain_confidence,
                            'company_size': unified_result.company_size,
                            'business_type': unified_result.business_type,
                            'automation_needs': unified_result.automation_needs,
                            'specific_processes': unified_result.specific_processes
                        }
                        logger.info(
                            f"[LLM Optimization] Unified analysis results stored in session context - "
                            f"Intent: {unified_result.primary_intent}, Industry: {unified_result.industry}"
                        )
                except Exception as e:
                    logger.warning(f"[LLM Optimization] Unified analysis failed, falling back to sequential: {e}")

            # Create action plan
            action_plan = await self.action_planner.plan_action(
                intent=intent,
                context=context,
                parameters=params
            )

            # Store in session context for confirmation/execution
            # ActionPlan is a dataclass, so we need to convert it manually
            import dataclasses

            plan_dict = dataclasses.asdict(action_plan)

            # Ensure context exists
            if session.context is None:
                session.context = {}

            session.context['pending_action_plan'] = plan_dict

            # CRITICAL: Mark context as modified so SQLAlchemy knows to update it
            from sqlalchemy.orm import attributes
            attributes.flag_modified(session, 'context')

            # Persist the action plan to database
            await self.db.commit()
            await self.db.refresh(session)

            # Verify it was saved
            if 'pending_action_plan' in session.context:
                logger.info(f"[Phase 2] Successfully persisted action plan for {intent} with {len(action_plan.steps)} steps")
            else:
                logger.error(f"[Phase 2] Failed to persist action plan!")
            return action_plan

        except Exception as e:
            logger.error(f"[Phase 2] Failed to create action plan: {e}")
            import traceback
            logger.error(f"[Phase 2] Traceback: {traceback.format_exc()}")
            return None

    async def _format_action_confirmation(self, action_plan, current_response):
        """
        Format the action plan for user confirmation.
        Adds preview of what will be done.
        """
        confirmation = f"\n\n**📋 Action Plan Ready**\n"
        confirmation += f"Name: {action_plan.name}\n"
        if action_plan.description:
            confirmation += f"Description: {action_plan.description}\n"
        confirmation += f"Steps to execute ({len(action_plan.steps)}):\n"

        for i, step in enumerate(action_plan.steps, 1):
            confirmation += f"  {i}. {step.name}: {step.description}\n"

        if hasattr(action_plan, 'estimated_time_seconds') and action_plan.estimated_time_seconds:
            confirmation += f"\nEstimated time: {action_plan.estimated_time_seconds} seconds\n"

        confirmation += f"\n**Confirm to proceed?** (yes/no)"

        return current_response + confirmation

    async def _format_action_plan_response(self, action_plan, user_query):
        """
        Format a clean, user-friendly action plan response.
        """
        # Use LLM to intelligently rephrase the user's request
        response_intro = await self._generate_intelligent_response_intro(user_query)
        response = f"{response_intro}\n\n"

        # Present the action plan clearly
        response += "**Here's what I'll do:**\n\n"

        # Format steps with proper descriptions
        for i, step in enumerate(action_plan.steps, 1):
            step_name = step.name.replace('_', ' ').title()

            # Provide meaningful step descriptions based on step type
            if hasattr(step, 'type'):
                if step.type == 'CREATE_RESOURCE':
                    if 'customer' in user_query.lower() or 'service' in user_query.lower():
                        response += f"**Step {i}:** Set up Customer Service System\n"
                        response += f"   • Create support ticket routing workflow\n"
                        response += f"   • Configure auto-response templates\n"
                        response += f"   • Set up escalation rules\n\n"
                    else:
                        response += f"**Step {i}:** {step_name}\n"
                        response += f"   • Initialize workflow components\n\n"
                elif step.type == 'CONFIGURE':
                    response += f"**Step {i}:** Configure Integrations\n"
                    if 'email' in user_query.lower():
                        response += f"   • Connect email service provider\n"
                    if 'slack' in user_query.lower():
                        response += f"   • Set up Slack workspace connection\n"
                        response += f"   • Configure notification channels\n"
                    response += "\n"
                elif step.type == 'CONNECT':
                    response += f"**Step {i}:** Connect Services\n"
                    response += f"   • Establish API connections\n"
                    response += f"   • Verify authentication\n\n"
                elif step.type == 'ENABLE_FEATURE':
                    response += f"**Step {i}:** Enable Features\n"
                    response += f"   • Activate automation rules\n"
                    response += f"   • Start monitoring\n\n"
                elif step.type == 'VALIDATE':
                    response += f"**Step {i}:** Validate Setup\n"
                    response += f"   • Test connections\n"
                    response += f"   • Verify workflow triggers\n\n"
                else:
                    response += f"**Step {i}:** {step_name}\n"
                    if step.description:
                        response += f"   • {step.description}\n\n"
                    else:
                        response += f"   • Configure and initialize\n\n"
            else:
                # Fallback for simple steps
                if 'workflow' in step_name.lower():
                    response += f"**Step {i}:** Create Customer Service Workflow\n"
                    response += f"   • Design workflow structure\n"
                    response += f"   • Set up triggers and actions\n"
                    response += f"   • Configure routing rules\n\n"
                else:
                    response += f"**Step {i}:** {step_name}\n\n"

        # Add any warnings or requirements
        if hasattr(action_plan, 'required_resources') and action_plan.required_resources:
            response += "**Requirements:**\n"
            for resource in action_plan.required_resources:
                # Handle both dict and object formats
                status = resource.get('status') if isinstance(resource, dict) else getattr(resource, 'status', None)
                name = resource.get('name') if isinstance(resource, dict) else getattr(resource, 'name', 'Unknown')
                if status == 'required':
                    response += f"   ⚠️ {name.replace('_', ' ').title()}\n"

        # Time estimate in human-friendly format
        if hasattr(action_plan, 'estimated_time_seconds') and action_plan.estimated_time_seconds:
            time_est = action_plan.estimated_time_seconds
            if time_est < 60:
                response += f"\n⏱️ **Estimated time:** {time_est} seconds\n"
            else:
                minutes = time_est // 60
                response += f"\n⏱️ **Estimated time:** About {minutes} minute{'s' if minutes > 1 else ''}\n"

        response += "\n**Ready to proceed?** Just say 'yes' to start, or let me know if you'd like to modify anything."

        return response

    async def _format_automation_result_response(self, automation_result: dict) -> str:
        """
        Format a user-friendly response from automation_result after action execution.
        This is called when user confirms with 'yes' and action completes.
        """
        company_name = automation_result.get('company_name', 'Your company')
        industry = automation_result.get('industry', 'your industry')
        summary = automation_result.get('summary', {})

        workflows_created = summary.get('workflows_created', 0)
        agents_assigned = summary.get('agents_assigned', 0)
        pods_created = summary.get('pods_created', 0)
        automation_level = summary.get('automation_level', 0)

        response = f"**Automation setup complete for {company_name}!**\n\n"
        response += f"**Industry:** {industry.title()}\n\n"

        response += "**Summary:**\n"
        response += f"• {workflows_created} workflows created\n"
        response += f"• {agents_assigned} AI agents assigned\n"
        response += f"• {pods_created} pods created\n"
        response += f"• {automation_level}% automation level\n\n"

        # List workflows
        workflows = automation_result.get('workflows', [])
        if workflows:
            response += "**Workflows Created:**\n"
            for wf in workflows[:5]:  # Show first 5
                wf_name = wf.get('name', 'Unknown').replace('_', ' ').title()
                response += f"• {wf_name}\n"
            if len(workflows) > 5:
                response += f"• ... and {len(workflows) - 5} more\n"
            response += "\n"

        # List key agents
        agents = automation_result.get('agents', [])
        if agents:
            response += "**Key Agents:**\n"
            for agent in agents[:5]:  # Show first 5
                agent_name = agent.get('name', 'Unknown')
                response += f"• {agent_name}\n"
            if len(agents) > 5:
                response += f"• ... and {len(agents) - 5} more\n"
            response += "\n"

        # Add next steps from company_automation if available
        company_automation = automation_result.get('company_automation', {})
        next_steps = company_automation.get('next_steps', [])
        if next_steps:
            response += "**Next Steps:**\n"
            for step in next_steps[:3]:
                response += f"• {step}\n"
            response += "\n"

        response += "Use the navigation buttons below to explore your new automation setup."

        return response

    async def _generate_intelligent_response_intro(self, user_query: str) -> str:
        """
        Generate an intelligent, grammatically correct response introduction
        using LLM to understand and rephrase the user's request.
        """
        if self._enhanced_llm_service:
            try:
                # Use LLM to generate a natural response intro
                prompt = f"""Given this user request: "{user_query}"

Generate a brief, professional response introduction that:
1. Acknowledges what the user wants to do
2. Uses proper grammar
3. Is conversational but professional
4. Ends with a period

Examples:
- User: "I want to automate customer onboarding"
  Response: "I'll help you automate customer onboarding."

- User: "generate a email marketing workflow"
  Response: "I'll create an email marketing workflow for you."

- User: "I need to set up invoice processing"
  Response: "I'll help you set up invoice processing."

Now generate a response for: "{user_query}"
Response (just the sentence, no quotes):"""

                llm_response = await self._enhanced_llm_service.generate(
                    prompt=prompt,
                    task_type="response_formatting",
                    temperature=0.3,
                    max_tokens=50
                )

                if llm_response and llm_response.text:
                    response_intro = llm_response.text.strip()
                    # Ensure it ends with period
                    if response_intro and not response_intro.endswith('.'):
                        response_intro += '.'
                    logger.debug(f"[LLM Response] Generated intro: {response_intro}")
                    return response_intro
            except Exception as e:
                logger.warning(f"[LLM Response] Failed to generate intro: {e}")

        # Fallback to simple but grammatically correct string manipulation
        action_phrase = user_query.lower().strip()

        # Remove common prefixes
        prefixes = ["i want to ", "i need to ", "i'd like to ", "help me ", "please "]
        for prefix in prefixes:
            if action_phrase.startswith(prefix):
                action_phrase = action_phrase[len(prefix):]
                break

        # Handle "generate" requests specially
        if action_phrase.startswith("generate "):
            action_phrase = action_phrase.replace("generate ", "create ")

        return f"I'll help you {action_phrase}."

    # =========================================================================
    # v4 Tool-Aware Streaming with Knowledge Integration
    # =========================================================================

    async def stream_tool_execution(
        self,
        content: str,
        user_id: str,
        session_context: Optional[Dict[str, Any]] = None
    ) -> "AsyncGenerator[Dict[str, Any], None]":
        """
        v4 Tool-aware streaming with FULL knowledge integration.

        Unlike the standalone ToolAwareConversationService, this method:
        1. Queries knowledge base for relevant templates, agents, adapters
        2. Builds an industry-aware system prompt with discovered capabilities
        3. Uses the state machine for multi-turn clarification when needed
        4. Leverages ML-enhanced semantic search (Business edition)

        This ensures that "Set up a new legal matter for e-discovery" produces
        intelligent, context-aware responses rather than generic tool calls.
        """
        from typing import AsyncGenerator
        from services.tool_dispatcher import ToolDispatcher, Edition
        from llm.models import ToolDefinition

        await self.initialize_knowledge()

        yield {"event": "thinking", "data": {"message": "Analyzing your request..."}}

        # Step 1: Query knowledge base for relevant resources
        knowledge_items = await self.knowledge_service.find_relevant_knowledge(
            query=content,
            context=session_context or {},
            limit=10
        )

        # Categorize knowledge items
        templates = [k for k in knowledge_items if k.type == "template"]
        agents = [k for k in knowledge_items if k.type == "agent"]
        adapters = [k for k in knowledge_items if k.type == "adapter"]
        features = [k for k in knowledge_items if k.type == "feature"]

        yield {
            "event": "knowledge_retrieved",
            "data": {
                "templates": len(templates),
                "agents": len(agents),
                "adapters": len(adapters),
                "features": len(features),
                "message": f"Found {len(knowledge_items)} relevant resources"
            }
        }

        # Step 2: Build system prompt via assembler
        system_prompt = await self.prompt_assembler.assemble(
            edition="community",
            knowledge_items=knowledge_items,
            session_context=session_context,
        )

        # Step 3: Get tool definitions
        tool_dispatcher = ToolDispatcher(self.db, Edition.COMMUNITY)
        tools = tool_dispatcher.get_available_tools()

        yield {
            "event": "tools_ready",
            "data": {
                "tools_available": len(tools),
                "message": "Tools initialized"
            }
        }

        # Step 4: Generate response with tools using knowledge-aware prompt
        if not self._enhanced_llm_service:
            yield {"event": "error", "data": {"message": "LLM service unavailable"}}
            return

        try:
            llm_response = await self._enhanced_llm_service.generate_with_tools(
                prompt=content,
                tools=tools,
                system_prompt=system_prompt,
                task_type="tool_use"
            )

            if llm_response.has_tool_calls():
                yield {
                    "event": "tools_identified",
                    "data": {
                        "tools": [tc.name for tc in llm_response.tool_calls],
                        "count": len(llm_response.tool_calls)
                    }
                }

                # Execute each tool with progress updates
                tool_results = []
                for i, tool_call in enumerate(llm_response.tool_calls):
                    yield {
                        "event": "tool_start",
                        "data": {
                            "tool": tool_call.name,
                            "index": i + 1,
                            "total": len(llm_response.tool_calls),
                            "arguments": tool_call.arguments
                        }
                    }

                    result = await tool_dispatcher.invoke(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        user_id=user_id,
                        context=session_context or {}
                    )

                    if result.data is None:
                        result.data = {}
                    result.data['tool_name'] = tool_call.name
                    tool_results.append(result)

                    if result.success:
                        yield {
                            "event": "tool_complete",
                            "data": {
                                "tool": tool_call.name,
                                "success": True,
                                "result": result.data,
                                "execution_time_ms": result.execution_time_ms
                            }
                        }
                    else:
                        yield {
                            "event": "tool_error",
                            "data": {
                                "tool": tool_call.name,
                                "success": False,
                                "error": result.error,
                                "recovery_strategy": result.recovery_strategy.value if result.recovery_strategy else None
                            }
                        }

                # Generate final response with knowledge context
                response_content = await self._generate_tool_response_with_knowledge(
                    original_query=content,
                    tool_calls=llm_response.tool_calls,
                    tool_results=tool_results,
                    knowledge_items=knowledge_items,
                    llm_text=llm_response.text
                )

                # Apply JSON stripping to the final response
                response_content = self._strip_json_from_response(response_content)

                # Extract workflow_id if a workflow was created (for frontend navigation)
                created_workflow_id = None
                created_workflow_name = None
                for result in tool_results:
                    if result.success and result.data:
                        if result.data.get('workflow_id'):
                            created_workflow_id = result.data.get('workflow_id')
                            created_workflow_name = result.data.get('workflow_name')
                            break

                response_data = {
                    "content": response_content,
                    "tools_used": [tc.name for tc in llm_response.tool_calls],
                    "all_successful": all(r.success for r in tool_results),
                    "knowledge_context": {
                        "templates_found": len(templates),
                        "agents_found": len(agents)
                    }
                }

                # Include workflow navigation info if a workflow was created
                if created_workflow_id:
                    response_data["created_workflow"] = {
                        "id": created_workflow_id,
                        "name": created_workflow_name,
                        "edit_url": f"/workflows/{created_workflow_id}/edit"
                    }

                yield {
                    "event": "response",
                    "data": response_data
                }
            else:
                # No tools needed - generate knowledge-aware response
                response_content = await self._generate_knowledge_aware_response(
                    content, knowledge_items, llm_response.text
                )

                # Apply JSON stripping to the final response
                response_content = self._strip_json_from_response(response_content)

                yield {
                    "event": "response",
                    "data": {
                        "content": response_content,
                        "tools_used": [],
                        "knowledge_context": {
                            "templates_found": len(templates),
                            "agents_found": len(agents)
                        }
                    }
                }

            yield {"event": "complete", "data": {"status": "done"}}

        except Exception as e:
            logger.error(f"[v4 Knowledge] Streaming error: {e}")
            import traceback
            logger.error(f"[v4 Knowledge] Traceback: {traceback.format_exc()}")
            yield {
                "event": "error",
                "data": {"message": str(e)}
            }

    async def _generate_tool_response_with_knowledge(
        self,
        original_query: str,
        tool_calls: List[Any],
        tool_results: List[Any],
        knowledge_items: List["KnowledgeItem"],
        llm_text: Optional[str] = None
    ) -> str:
        """Generate a response that incorporates both tool results and knowledge context."""
        # Start with intro based on what was done
        successful = [r for r in tool_results if r.success]
        failed = [r for r in tool_results if not r.success]

        response = ""

        if successful:
            if len(successful) == 1:
                tool_name = tool_calls[0].name.replace('_', ' ')
                response += f"I've completed the {tool_name} operation.\n\n"
            else:
                response += f"I've completed {len(successful)} actions:\n\n"
                for tc, result in zip(tool_calls, tool_results):
                    if result.success:
                        response += f"- **{tc.name.replace('_', ' ')}**: Done\n"
                response += "\n"

        # Add relevant knowledge suggestions
        templates = [k for k in knowledge_items if k.type == "template"]
        if templates and 'search' in original_query.lower():
            response += "**Relevant Templates Found:**\n"
            for t in templates[:3]:
                desc = t.data.get('description', '')
                response += f"- **{t.name}**: {desc}\n"
            response += "\n"

        if failed:
            response += "**Some operations had issues:**\n"
            for tc, result in zip(tool_calls, tool_results):
                if not result.success:
                    response += f"- {tc.name}: {result.error}\n"
            response += "\nWould you like me to try a different approach?\n"
        else:
            response += "Is there anything else you'd like me to help with?"

        return response

    def _strip_json_from_response(self, text: str) -> str:
        """Remove any raw JSON blocks and common LLM artifacts from the response.

        Cleans up:
        - JSON code blocks
        - Raw JSON objects with "action"/"message" keys (extract message)
        - Tool documentation/schemas
        - Role labels like "Assistant:"
        - Excessive formatting
        """
        import re
        import json

        if not text:
            return text

        cleaned = text

        # Pattern 0: Extract message from JSON wrapper
        # Handle common LLM patterns: {"action": "...", "message": "..."} or {"action": "...", "execution_plan": "..."}
        try:
            if cleaned.strip().startswith('{') and cleaned.strip().endswith('}'):
                parsed = json.loads(cleaned.strip())
                if isinstance(parsed, dict):
                    # Priority order: message > execution_plan > query
                    if 'message' in parsed:
                        cleaned = parsed['message']
                    elif 'execution_plan' in parsed:
                        # LLM described what it would do but didn't execute
                        # Return a natural-sounding response asking for details
                        cleaned = "Happy to help with that! I need a few quick details:\n1. What's the matter/workflow name?\n2. How many documents approximately?\n3. What type of processing?"
                    elif 'query' in parsed:
                        # LLM tried to search but didn't execute
                        cleaned = f"I can help you with {parsed.get('action', 'that')}. What specific details would you like?"
        except (json.JSONDecodeError, TypeError):
            pass  # Not valid JSON, continue with other cleanup

        # Pattern 1: Remove JSON code blocks like ```json {...} ``` or ``` {...} ```
        code_block_pattern = r'```(?:json)?\s*\{[^`]*\}\s*```'
        cleaned = re.sub(code_block_pattern, '', cleaned, flags=re.DOTALL)

        # Pattern 2: Remove raw JSON at the start of the response
        start_json_pattern = r'^\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*'
        match = re.match(start_json_pattern, cleaned, re.DOTALL)
        if match:
            cleaned = cleaned[match.end():]

        # Pattern 3: Remove "Available Tools:" sections and everything after
        available_tools_pattern = r'\*\*Available Tools:\*\*.*$'
        cleaned = re.sub(available_tools_pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Pattern 4: Remove tool documentation blocks (Tool: xxx, Description:, Parameters:)
        tool_doc_pattern = r'Tool:\s*\w+.*?(?=\n\n|\Z)'
        cleaned = re.sub(tool_doc_pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Pattern 5: Remove role labels at the start
        role_label_pattern = r'^\*?\*?(?:Assistant|User|System):\*?\*?\s*'
        cleaned = re.sub(role_label_pattern, '', cleaned, flags=re.IGNORECASE)

        # Pattern 6: Remove sentences about recommending/using tools
        tool_recommend_pattern = r'(?:To|I recommend|I\'ll use|Using)\s+(?:set up|create|use)?\s*.*?(?:tool|the `\w+`)\.\s*'
        cleaned = re.sub(tool_recommend_pattern, '', cleaned, flags=re.IGNORECASE)

        # Pattern 7: Remove "Please let me know if you'd like to use any other tool" type phrases
        tool_offer_pattern = r'Please let me know if you\'?d? like to use any (?:other )?tool.*?\.?\s*'
        cleaned = re.sub(tool_offer_pattern, '', cleaned, flags=re.IGNORECASE)

        # Pattern 8: Remove parameter schema blocks (- name: string (required) etc.)
        param_schema_pattern = r'Parameters:\s*\n(?:\s*-\s*\w+:.*\n?)+'
        cleaned = re.sub(param_schema_pattern, '', cleaned, flags=re.IGNORECASE)

        # Clean up any excessive newlines and whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)  # Remove trailing spaces on lines
        cleaned = cleaned.strip()

        return cleaned if cleaned else text

    async def _generate_knowledge_aware_response(
        self,
        user_query: str,
        knowledge_items: List["KnowledgeItem"],
        llm_text: Optional[str] = None
    ) -> str:
        """Generate a response based on knowledge when no tools are called."""
        # Strip any accidental JSON from LLM response
        if llm_text:
            llm_text = self._strip_json_from_response(llm_text)

        if not knowledge_items:
            return llm_text or "I'd be happy to help. Could you provide more details about what you'd like to accomplish?"

        # Categorize knowledge
        templates = [k for k in knowledge_items if k.type == "template"]
        agents = [k for k in knowledge_items if k.type == "agent"]

        response = ""
        if llm_text:
            response = llm_text + "\n\n"

        if templates:
            response += f"**Found {len(templates)} Relevant Templates:**\n\n"
            for t in templates[:5]:
                desc = t.data.get('description', 'Workflow template')
                category = t.data.get('category', 'general')
                response += f"- **{t.name}** ({category})\n  {desc}\n\n"
            response += "Would you like me to create a workflow using one of these templates?\n"

        elif agents:
            response += f"**Available AI Agents:**\n\n"
            for a in agents[:3]:
                response += f"- **{a.name}**: "
                caps = a.data.get('capabilities', [])
                response += ', '.join(caps[:3]) if caps else 'General assistance'
                response += "\n"
            response += "\nWould you like to deploy one of these agents?\n"

        else:
            response += "I found some relevant resources. "
            response += "What specific action would you like to take?"

        return response

    # =========================================================================
    # v5 UNIFIED CONVERSATION FLOW - LLM as Brain
    # =========================================================================

    async def process_message_v2(
        self,
        session_id: UUID,
        content: str,
        user_id: str,
        stream: bool = True
    ) -> "AsyncGenerator[Dict[str, Any], None]":
        """
        v5 Unified conversation with LLM-as-brain architecture.

        Key differences from v1-v4:
        1. Full conversation history passed to LLM
        2. LLM decides whether to ask questions or take action
        3. No rigid state machine - LLM IS the state machine
        4. Parameters accumulate naturally via conversation memory
        5. Single unified flow for all conversation types

        This is a GENERAL-PURPOSE implementation that works for any domain:
        - Legal e-discovery workflows
        - Customer onboarding automation
        - Invoice processing
        - Any workflow/agent/integration task

        Args:
            session_id: Required session ID for persistence and history
            content: User's message
            user_id: User identifier
            stream: Whether to yield SSE events (default True)

        Yields:
            Dict events: thinking, knowledge_retrieved, tool_start, tool_complete, response, complete
        """
        from typing import AsyncGenerator
        from services.tool_dispatcher import ToolDispatcher, Edition

        await self.initialize_knowledge()

        # =====================================================================
        # Step 1: Load session and full conversation history
        # =====================================================================
        session = await self.get_session(session_id)
        if not session:
            yield {"event": "error", "data": {"message": f"Session {session_id} not found"}}
            return

        if stream:
            yield {"event": "thinking", "data": {"message": "Understanding your request..."}}

        # Build conversation history from session messages (async to avoid lazy load issues)
        conversation_history = await self._build_conversation_history(session_id)
        logger.info(f"[v5] Loaded {len(conversation_history)} messages from history")

        # =====================================================================
        # Step 2: Store user message immediately (before processing)
        # =====================================================================
        user_message = await self._store_message(
            session_id=session_id,
            role="user",
            content=content
        )

        # Add current message to history for this call
        conversation_history.append({"role": "user", "content": content})

        # =====================================================================
        # Step 3: Retrieve knowledge context (templates, agents, adapters)
        # =====================================================================
        knowledge_items = await self.knowledge_service.find_relevant_knowledge(
            query=content,
            context=session.context or {},
            limit=10
        )

        if stream:
            templates = [k for k in knowledge_items if k.type == "template"]
            agents = [k for k in knowledge_items if k.type == "agent"]
            yield {
                "event": "knowledge_retrieved",
                "data": {
                    "templates": len(templates),
                    "agents": len(agents),
                    "total": len(knowledge_items),
                    "message": f"Found {len(knowledge_items)} relevant resources"
                }
            }

        # =====================================================================
        # Step 4: Build comprehensive LLM context (the key to v5)
        # =====================================================================
        system_prompt = await self.prompt_assembler.assemble(
            edition="community",
            session=session,
            knowledge_items=knowledge_items,
            conversation_history=conversation_history,
            user_id=user_id,
        )

        # Get available tools
        tool_dispatcher = ToolDispatcher(self.db, Edition.COMMUNITY)
        tools = tool_dispatcher.get_available_tools()

        if stream:
            yield {
                "event": "context_ready",
                "data": {
                    "tools_available": len(tools),
                    "history_messages": len(conversation_history),
                    "message": "Context assembled"
                }
            }

        # =====================================================================
        # Step 5: LLM generation with full context - LLM DECIDES what to do
        # =====================================================================
        if not self._enhanced_llm_service:
            yield {"event": "error", "data": {"message": "LLM service unavailable"}}
            return

        # Format history into prompt (since our LLM service takes prompt, not messages)
        full_prompt = self._format_history_as_prompt(conversation_history)

        try:
            llm_response = await self._enhanced_llm_service.generate_with_tools(
                prompt=full_prompt,
                tools=tools,
                system_prompt=system_prompt,
                task_type="tool_use",
                temperature=0.4  # Slightly higher for more natural conversation
            )

            # =====================================================================
            # Step 6: Execute tools if LLM decided to call any
            # =====================================================================
            tool_results = []

            # First check for proper tool calls
            has_proper_tool_calls = llm_response.has_tool_calls()

            # Also check if LLM output JSON that looks like a tool call
            # (some models output JSON text instead of proper function calls)
            inferred_tool_call = None
            if not has_proper_tool_calls and llm_response.text:
                # Pass user's message to validate that extracted data matches user input
                inferred_tool_call = self._parse_json_tool_call(llm_response.text, content)

            if has_proper_tool_calls or inferred_tool_call:
                # Build list of tool calls to execute (proper or inferred)
                tool_calls_to_execute = []
                if has_proper_tool_calls:
                    tool_calls_to_execute = llm_response.tool_calls
                elif inferred_tool_call:
                    # Create a simple object with name and arguments
                    class InferredToolCall:
                        def __init__(self, name, arguments):
                            self.name = name
                            self.arguments = arguments
                    tool_calls_to_execute = [InferredToolCall(
                        inferred_tool_call['tool'],
                        inferred_tool_call.get('arguments', {})
                    )]

                if stream:
                    yield {
                        "event": "tools_identified",
                        "data": {
                            "tools": [tc.name for tc in tool_calls_to_execute],
                            "count": len(tool_calls_to_execute)
                        }
                    }

                for i, tool_call in enumerate(tool_calls_to_execute):
                    if stream:
                        yield {
                            "event": "tool_start",
                            "data": {
                                "tool": tool_call.name,
                                "index": i + 1,
                                "total": len(tool_calls_to_execute)
                            }
                        }

                    # Build context with knowledge items for domain-aware generation
                    # Per WORKFLOW_GENERATION_FIX_IMPLEMENTATION.md Change 1.1
                    tool_context = dict(session.context) if session.context else {}
                    if knowledge_items:
                        # Convert knowledge items to serializable format for tool handlers
                        tool_context['knowledge_items'] = [
                            {'name': k.name, 'type': k.type, 'description': getattr(k, 'description', ''),
                             'category': getattr(k, 'category', ''), 'id': str(getattr(k, 'id', ''))}
                            for k in knowledge_items
                        ]
                        tool_context['templates'] = [
                            {'name': k.name, 'description': getattr(k, 'description', ''),
                             'category': getattr(k, 'category', ''), 'id': str(getattr(k, 'id', ''))}
                            for k in knowledge_items if k.type == 'template'
                        ]

                    result = await tool_dispatcher.invoke(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        user_id=user_id,
                        context=tool_context
                    )

                    if result.data is None:
                        result.data = {}
                    result.data['tool_name'] = tool_call.name
                    tool_results.append(result)

                    if stream:
                        yield {
                            "event": "tool_complete",
                            "data": {
                                "tool": tool_call.name,
                                "success": result.success,
                                "result": result.data if result.success else None,
                                "error": result.error if not result.success else None
                            }
                        }

            # =====================================================================
            # Step 7: Generate final response
            # =====================================================================
            response_content = self._build_v5_response(
                llm_response=llm_response,
                tool_results=tool_results,
                knowledge_items=knowledge_items
            )

            # Clean any JSON artifacts from response
            response_content = self._strip_json_from_response(response_content)

            # =====================================================================
            # Step 8: Persist assistant response and update session context
            # =====================================================================
            assistant_message = await self._store_message(
                session_id=session_id,
                role="assistant",
                content=response_content,
                llm_model_used=getattr(llm_response, 'model_used', None)
            )

            # Update session context with any new information
            await self._update_session_context_v5(
                session=session,
                user_content=content,
                assistant_content=response_content,
                tool_results=tool_results
            )

            # =====================================================================
            # Step 9: Return final response
            # =====================================================================
            # Extract created_workflow info for frontend navigation
            # First try from tool_results, then fall back to session context
            created_workflow_info = None

            # Method 1: Check tool_results directly
            logger.info(f"[v5 created_workflow] tool_results count: {len(tool_results) if tool_results else 0}")
            if tool_results:
                for result in tool_results:
                    logger.info(f"[v5 created_workflow] Checking result: success={result.success}, data={result.data}")
                    if result.success and result.data and result.data.get('workflow_id'):
                        created_workflow_info = {
                            "id": result.data.get('workflow_id'),
                            "name": result.data.get('workflow_name', 'Workflow'),
                            "edit_url": f"/workflows/{result.data.get('workflow_id')}/edit"
                        }
                        logger.info(f"[v5 created_workflow] Found workflow from tool_results: {created_workflow_info}")
                        break

            # Method 2: Fall back to session context if tool_results didn't have it
            if not created_workflow_info:
                v5_params = session.context.get('v5_parameters', {})
                logger.info(f"[v5 created_workflow] Checking session context: {v5_params}")
                if v5_params.get('workflow_id'):
                    created_workflow_info = {
                        "id": v5_params.get('workflow_id'),
                        "name": v5_params.get('workflow_name') or v5_params.get('last_created_workflow', 'Workflow'),
                        "edit_url": f"/workflows/{v5_params.get('workflow_id')}/edit"
                    }
                    logger.info(f"[v5 created_workflow] Found workflow from session context: {created_workflow_info}")

            response_data = {
                "content": response_content,
                "message_id": str(assistant_message.id),
                "tools_executed": [tc.name for tc in llm_response.tool_calls] if llm_response.has_tool_calls() else [],
                "all_successful": all(r.success for r in tool_results) if tool_results else True,
                "session_context": {
                    "parameters": session.context.get('v5_parameters', {}),
                    "turn_count": len(conversation_history)
                }
            }

            # Include workflow navigation info if a workflow was created
            if created_workflow_info:
                response_data["created_workflow"] = created_workflow_info

            yield {
                "event": "response",
                "data": response_data
            }

            yield {"event": "complete", "data": {"status": "done"}}

        except Exception as e:
            logger.error(f"[v5] Error in process_message_v2: {e}")
            import traceback
            logger.error(f"[v5] Traceback: {traceback.format_exc()}")
            yield {"event": "error", "data": {"message": str(e)}}

    async def _build_conversation_history(self, session_id: UUID) -> List[Dict[str, str]]:
        """
        Build conversation history from session messages.

        Returns list of {"role": "user"|"assistant", "content": "..."} dicts.
        Limited to last N messages to stay within context limits.

        Uses async query to avoid SQLAlchemy lazy loading issues.
        """
        from sqlalchemy import select, desc
        from models.conversation import ConversationMessage

        MAX_HISTORY_MESSAGES = 20  # Configurable

        # Query messages directly with async
        result = await self.db.execute(
            select(ConversationMessage)
            .filter(ConversationMessage.session_id == session_id)
            .order_by(desc(ConversationMessage.timestamp))
            .limit(MAX_HISTORY_MESSAGES)
        )
        messages = result.scalars().all()

        # Reverse to get chronological order (oldest first)
        messages = list(reversed(messages))

        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content
            })

        return history

    def _format_history_as_prompt(self, history: List[Dict[str, str]]) -> str:
        """
        Format conversation history as a prompt string.

        Since our LLMService takes a single prompt (not messages array),
        we format the history into a readable conversation format.
        """
        if not history:
            return ""

        formatted = "CONVERSATION HISTORY:\n"
        formatted += "=" * 40 + "\n\n"

        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"**{role_label}:** {msg['content']}\n\n"

        formatted += "=" * 40 + "\n"
        formatted += "Please respond to the user's latest message above.\n"

        return formatted

    def _parse_json_tool_call(self, text: str, user_message: str = "") -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM text that looks like a tool call.

        Only triggers if the extracted arguments contain information that
        appears to come from the user's message (not made up).

        Args:
            text: LLM response text
            user_message: The user's original message (to validate data)

        Returns: {"tool": "tool_name", "arguments": {...}} or None
        """
        import json
        import re

        if not text:
            return None

        try:
            # Try to find JSON in the text
            text_stripped = text.strip()

            # Direct JSON object
            if text_stripped.startswith('{') and text_stripped.endswith('}'):
                parsed = json.loads(text_stripped)
            else:
                # Try to extract JSON from text
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text_stripped, re.DOTALL)
                if not json_match:
                    return None
                parsed = json.loads(json_match.group())

            # Check for known tool call patterns
            if isinstance(parsed, dict):
                tool_name = None
                arguments = {}

                # Pattern 1: {"action": "create_workflow", ...}
                if 'action' in parsed:
                    tool_name = parsed['action']
                    # Everything except 'action' and meta fields becomes arguments
                    arguments = {k: v for k, v in parsed.items()
                                if k not in ['action', 'execution_plan', 'query', 'message']}

                    # Map common argument names
                    if 'workflow_name' in arguments and 'name' not in arguments:
                        arguments['name'] = arguments.pop('workflow_name')

                # Pattern 2: {"name": "...", "description": "..."} (implicit create_workflow)
                elif 'name' in parsed and 'description' in parsed:
                    tool_name = 'create_workflow'
                    arguments = parsed

                if tool_name and tool_name == 'create_workflow':
                    # Validate: only execute if the name contains SPECIFIC information
                    # that the user provided (not generic words like "legal", "workflow", etc.)
                    name = arguments.get('name', '')
                    user_lower = user_message.lower()

                    # Skip generic words that appear in typical requests
                    generic_words = {'legal', 'matter', 'workflow', 'create', 'make', 'set', 'new',
                                    'discovery', 'document', 'processing', 'e-discovery', 'ediscovery'}

                    # Get significant words from the proposed name (length > 4, not generic)
                    name_words = [w for w in name.lower().split()
                                 if len(w) > 4 and w not in generic_words]

                    # Check if any non-generic significant words match user input
                    matching_specific = sum(1 for w in name_words if w in user_lower)

                    # Also check for proper nouns (capitalized words that aren't at sentence start)
                    # These are likely user-provided names
                    words_in_name = name.split()
                    proper_nouns = [w for w in words_in_name[1:] if w[0].isupper()] if len(words_in_name) > 1 else []
                    has_user_proper_noun = any(pn.lower() in user_lower for pn in proper_nouns)

                    if matching_specific >= 1 or has_user_proper_noun:
                        # Found specific user-provided content
                        return {
                            'tool': tool_name,
                            'arguments': arguments
                        }
                    else:
                        # Name seems generic or made up
                        logger.debug(f"[v5] Skipping inferred tool call - name '{name}' doesn't contain user-specific info")
                        return None

                elif tool_name:
                    # Other tools - execute them
                    return {
                        'tool': tool_name,
                        'arguments': arguments
                    }

        except (json.JSONDecodeError, TypeError):
            pass

        return None

    def _build_v5_response(
        self,
        llm_response,
        tool_results: List,
        knowledge_items: List["KnowledgeItem"]
    ) -> str:
        """
        Build the final response, incorporating tool results if any.

        Priority:
        1. If tools executed successfully - generate a natural confirmation
        2. If no tools executed - use LLM's text (cleaned of JSON)
        3. Fallback - ask clarifying question
        """
        # If tools were executed, prioritize tool result response
        if tool_results:
            successful = [r for r in tool_results if r.success]
            failed = [r for r in tool_results if not r.success]

            # Check if all tools failed with "not found" - likely hallucination
            all_not_found = all(
                r.error and "not found" in str(r.error).lower()
                for r in failed
            ) if failed and not successful else False

            if all_not_found:
                # LLM hallucinated tool names, provide helpful response
                return "I apologize, but I encountered an issue processing your request. Let me help you directly instead.\n\nCould you tell me more specifically what you'd like to set up? For example:\n- What would you like to name this workflow/matter?\n- What should it do?"

            if successful:
                # Build natural response from tool results
                response_parts = []
                created_workflow = None  # Track for next step suggestions

                for result in successful:
                    data = result.data or {}
                    tool_name = data.get('tool_name', '')

                    # Handle workflow creation - RICH FORMAT (v5 spec requirement)
                    if data.get('workflow_name'):
                        wf_name = data.get('workflow_name')
                        wf_id = data.get('workflow_id', '')
                        wf_id_short = wf_id[:8] if wf_id else ''
                        created_workflow = {'name': wf_name, 'id': wf_id}

                        # Rich response format with configuration details
                        response_parts.append(f"✅ **Created workflow: {wf_name}**")
                        if wf_id_short:
                            response_parts.append(f"   ID: `{wf_id_short}...`")

                        # Add configuration summary if available
                        if data.get('description'):
                            response_parts.append(f"   Description: {data.get('description')}")

                        # Show features enabled (from spec lines 826-841)
                        response_parts.append("\n**Configuration:**")
                        response_parts.append("- Status: Ready to configure")
                        response_parts.append("- Automation: Enabled")
                        if data.get('industry'):
                            response_parts.append(f"- Industry: {data.get('industry').title()}")

                    # Handle list_workflows or discover_workflows - summarize the workflows
                    elif data.get('workflows') and tool_name in ('list_workflows', 'discover_workflows'):
                        workflows = data.get('workflows', [])
                        count = data.get('count', len(workflows))
                        if count > 0:
                            response_parts.append(f"Found **{count} workflows**:\n")
                            for wf in workflows[:10]:  # Show first 10
                                name = wf.get('name', 'Unnamed')
                                status = wf.get('status', 'unknown')
                                response_parts.append(f"- **{name}** ({status})")
                        else:
                            response_parts.append("No workflows found.")

                    # Handle list_templates
                    elif tool_name == 'list_templates':
                        templates = data.get('templates', [])
                        count = data.get('count', len(templates))
                        if count > 0:
                            response_parts.append(f"Found **{count} templates**:\n")
                            for t in templates[:10]:
                                name = t.get('name', 'Unnamed')
                                response_parts.append(f"- **{name}**")
                        else:
                            response_parts.append("No templates found in the database. Templates may need to be loaded.")

                    # Handle list_agents
                    elif tool_name == 'list_agents':
                        agents = data.get('agents', [])
                        count = data.get('count', len(agents))
                        if count > 0:
                            response_parts.append(f"Found **{count} agents**:\n")
                            for a in agents[:10]:
                                name = a.get('name', 'Unnamed')
                                response_parts.append(f"- **{name}**")
                        else:
                            response_parts.append("No agents currently configured.")

                    # Handle list_integrations
                    elif tool_name == 'list_integrations':
                        integrations = data.get('integrations', [])
                        count = len(integrations)
                        if count > 0:
                            response_parts.append(f"Found **{count} integrations**:\n")
                            for i in integrations[:10]:
                                name = i.get('name', 'Unnamed')
                                response_parts.append(f"- **{name}**")
                        else:
                            response_parts.append("No integrations configured yet.")

                    # Handle help
                    elif tool_name == 'get_help' and data.get('help'):
                        response_parts.append(data.get('help'))

                    # Handle system status
                    elif tool_name == 'get_system_status':
                        status = data.get('status', 'unknown')
                        response_parts.append(f"System status: **{status}**")
                        if data.get('services'):
                            for svc, svc_status in data.get('services', {}).items():
                                response_parts.append(f"- {svc}: {svc_status}")

                    # Handle generic message
                    elif data.get('message'):
                        response_parts.append(data.get('message'))

                    # Fallback
                    else:
                        response_parts.append(f"{tool_name.replace('_', ' ').title()} completed.")

                response = "\n".join(response_parts)

                # =================================================================
                # NEXT STEP SUGGESTIONS (v5 spec requirement - lines 836-841)
                # After a successful action, offer relevant follow-up options
                # =================================================================
                if created_workflow:
                    response += "\n\n**Would you like me to:**"
                    response += "\n1. View/Edit this workflow in the editor"
                    response += "\n2. Add a description or configure this workflow"
                    response += "\n3. Set up triggers or schedule"
                    response += "\n4. Connect to an integration (Slack, Email, etc.)"
                    response += "\n5. Create another workflow"
                    response += "\n\nJust tell me what you'd like to do next!"

                    # Store workflow_id in session for frontend navigation
                    # This will be included in the response for the frontend to use
                    if hasattr(self, '_last_created_workflow_id'):
                        self._last_created_workflow_id = created_workflow.get('id')
                    else:
                        self._last_created_workflow_id = created_workflow.get('id')

                if failed:
                    response += "\n\n⚠️ Some actions encountered issues:\n"
                    for result in failed:
                        tool_name = result.data.get('tool_name', 'action') if result.data else 'action'
                        response += f"- {tool_name}: {result.error}\n"

                return response.strip()

            # All failed but not "not found"
            if failed:
                # Check if any failure is a "needs_clarification" - use that message instead of error
                for result in failed:
                    if result.data and result.data.get('needs_clarification'):
                        return result.data.get('message', "Could you provide more details?")

                # Otherwise show errors
                response = "I encountered some issues:\n"
                for result in failed:
                    tool_name = result.data.get('tool_name', 'action') if result.data else 'action'
                    response += f"- {tool_name}: {result.error}\n"
                return response

        # No tools executed - use LLM's response (will be cleaned by _strip_json_from_response)
        response = llm_response.text or ""

        # If LLM response is just JSON, don't use it
        if response.strip().startswith('{') and response.strip().endswith('}'):
            # LLM output JSON instead of natural language
            response = ""

        # If no response at all, generate one based on knowledge
        if not response and knowledge_items:
            templates = [k for k in knowledge_items if k.type == "template"]
            if templates:
                response = "I found some relevant resources that might help. "
                response += "Could you tell me more about what you'd like to accomplish?"

        # Fallback
        if not response:
            response = "I'm here to help. Could you tell me more about what you'd like to do?"

        return response

    async def _update_session_context_v5(
        self,
        session,
        user_content: str,
        assistant_content: str,
        tool_results: List
    ):
        """
        Update session context after a conversation turn.

        v5 ENHANCEMENT: Now includes EXPLICIT parameter extraction from user messages,
        not just relying on LLM memory. This makes parameter tracking more reliable.
        """
        from datetime import datetime
        from sqlalchemy.orm import attributes

        if session.context is None:
            session.context = {}

        # Initialize v5 parameters dict if needed
        if 'v5_parameters' not in session.context:
            session.context['v5_parameters'] = {}

        # =================================================================
        # EXPLICIT PARAMETER EXTRACTION (spec requirement)
        # Extract parameters from user's message each turn
        # =================================================================
        extracted_params = self._extract_parameters_from_turn(user_content)
        if extracted_params:
            # Merge with existing parameters (new values override old)
            session.context['v5_parameters'].update(extracted_params)
            logger.info(f"[v5 Params] Extracted from turn: {extracted_params}")
            logger.info(f"[v5 Params] Accumulated: {session.context['v5_parameters']}")

        # Track tool execution results
        if tool_results:
            if 'v5_tool_history' not in session.context:
                session.context['v5_tool_history'] = []

            for result in tool_results:
                session.context['v5_tool_history'].append({
                    'tool': result.data.get('tool_name') if result.data else None,
                    'success': result.success,
                    'timestamp': str(datetime.utcnow())
                })

            # Store workflow/agent details if created
            for result in tool_results:
                if result.success and result.data:
                    if 'workflow_id' in result.data:
                        session.context['v5_parameters']['workflow_id'] = result.data['workflow_id']
                        session.context['v5_parameters']['last_created_workflow'] = result.data.get('workflow_name', '')
                    if 'workflow_name' in result.data:
                        session.context['v5_parameters']['workflow_name'] = result.data['workflow_name']
                    if 'agent_id' in result.data:
                        session.context['v5_parameters']['agent_id'] = result.data['agent_id']

        # Track turn count
        session.context['v5_parameters']['turn_count'] = session.context['v5_parameters'].get('turn_count', 0) + 1

        # Mark context as modified
        attributes.flag_modified(session, 'context')
        await self.db.commit()

    def _extract_parameters_from_turn(self, user_content: str) -> Dict[str, Any]:
        """
        Extract explicit parameters from user's message.

        This is the v5 spec requirement for explicit parameter accumulation
        instead of relying solely on LLM memory.

        Extracts:
        - Names: "Call it X", "Name it X", direct names after being asked
        - Descriptions: "It should do X", "for X purposes"
        - Industry: Legal, healthcare, finance keywords
        - Numeric values: document counts, amounts
        """
        import re
        extracted = {}
        content_lower = user_content.lower()

        # =================================================================
        # Extract NAME patterns
        # =================================================================
        # Pattern: "Call it X" or "Name it X" or "Let's call it X"
        name_patterns = [
            r"call it\s+['\"]?([^'\".,!?]+)['\"]?",
            r"name it\s+['\"]?([^'\".,!?]+)['\"]?",
            r"let'?s? call it\s+['\"]?([^'\".,!?]+)['\"]?",
            r"named?\s+['\"]([^'\"]+)['\"]",
            r'^["\']([^"\']+)["\']$',  # Just a quoted name
        ]
        for pattern in name_patterns:
            match = re.search(pattern, user_content, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if name and len(name) > 2:
                    extracted['workflow_name'] = name
                    break

        # If no pattern matched but this looks like a direct name response
        # (short message after being asked for a name)
        if 'workflow_name' not in extracted:
            # Single line, relatively short, no question marks
            if len(user_content) < 100 and '\n' not in user_content and '?' not in user_content:
                # Check if it's not a command/action word
                action_words = ['create', 'show', 'list', 'help', 'what', 'how', 'why', 'yes', 'no']
                first_word = user_content.split()[0].lower() if user_content.split() else ''
                if first_word not in action_words:
                    # Could be a name - let LLM validate, but store as potential
                    extracted['potential_name'] = user_content.strip()

        # =================================================================
        # Extract DESCRIPTION patterns
        # =================================================================
        desc_patterns = [
            r"it should\s+(.+?)(?:\.|$)",
            r"that\s+(?:will|should|can)\s+(.+?)(?:\.|$)",
            r"for\s+(.+?)\s+purposes?",
            r"to\s+handle\s+(.+?)(?:\.|$)",
        ]
        for pattern in desc_patterns:
            match = re.search(pattern, user_content, re.IGNORECASE)
            if match:
                desc = match.group(1).strip()
                if desc and len(desc) > 5:
                    extracted['description'] = desc
                    break

        # =================================================================
        # Extract INDUSTRY context
        # =================================================================
        industry_keywords = {
            'legal': ['legal', 'e-discovery', 'ediscovery', 'litigation', 'attorney', 'law', 'court', 'matter', 'deposition'],
            'healthcare': ['healthcare', 'medical', 'patient', 'hipaa', 'clinical', 'hospital'],
            'finance': ['financial', 'banking', 'trading', 'investment', 'accounting', 'invoice'],
            'hr': ['employee', 'onboarding', 'hiring', 'recruitment', 'hr', 'human resources'],
        }
        for industry, keywords in industry_keywords.items():
            if any(kw in content_lower for kw in keywords):
                extracted['industry'] = industry
                break

        # =================================================================
        # Extract NUMERIC values
        # =================================================================
        # Document count: "4000 documents", "thousands of files"
        doc_count_patterns = [
            r'(\d+(?:,\d{3})*)\s*(?:documents?|files?|records?|emails?)',
            r'(\d+)k\s*(?:documents?|files?)',
        ]
        for pattern in doc_count_patterns:
            match = re.search(pattern, user_content, re.IGNORECASE)
            if match:
                count_str = match.group(1).replace(',', '')
                if 'k' in user_content[match.start():match.end()].lower():
                    count_str = str(int(count_str) * 1000)
                extracted['document_count'] = int(count_str)
                break

        return extracted