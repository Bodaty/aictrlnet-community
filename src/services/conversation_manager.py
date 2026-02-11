"""
Conversation Manager Service for Multi-Turn NLP System.

This service manages conversation sessions, maintains context, and coordinates
with existing NLP service for backward compatibility.
"""

import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, desc, select

from models.conversation import (
    ConversationSession,
    ConversationMessage,
    ConversationAction,
    ConversationIntent,
    ConversationPattern
)
from schemas.conversation import (
    ConversationSessionCreate,
    ConversationMessageCreate,
    ConversationActionExecute,
    ConversationResponse,
    IntentDetectionResponse
)
from core.config import get_settings
from core.edition_discovery import get_edition_discovery, Edition
from services.action_orchestrator import ActionOrchestrator, ActionType, ParameterRequirements, SmartDefaults
from services.llm_helpers import get_user_llm_settings
from llm import llm_service, UserLLMSettings

logger = logging.getLogger(__name__)
settings = get_settings()
edition_discovery = get_edition_discovery()


class ConversationManagerService:
    """
    Manages multi-turn conversations with context persistence and intent refinement.
    Designed to work alongside existing single-shot NLP service for backward compatibility.
    Uses AsyncSession to work with the FastAPI async architecture.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        # Use LLM service directly for conversation intelligence
        self.llm = llm_service
        self.action_orchestrator = None  # Initialized per session with user edition
        self.smart_defaults = SmartDefaults()
        
    # === Session Management ===
    
    async def create_session(
        self, 
        user_id: str,
        initial_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ConversationSession:
        """Create a new conversation session."""
        session = ConversationSession(
            id=uuid4(),
            user_id=user_id,
            state="greeting",
            started_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            context=context or {},
            extracted_params={},
            session_config={
                "multi_turn_enabled": True,
                "edition": "community",
                "max_turns": 20,
                "timeout_minutes": 30
            },
            is_active=True,
            requires_human=False
        )
        
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        
        # Process initial message if provided
        if initial_message:
            await self.process_message(session.id, initial_message, user_id)
            
        return session
    
    async def find_or_create_channel_session(
        self,
        channel_type: str,
        sender_id: str,
        platform_metadata: Dict[str, Any],
    ) -> ConversationSession:
        """Find an existing active session for this channel sender, or create one.

        Channel-agnostic session lookup: if the same sender has an active session
        with a matching channel binding, reuse it. If they arrive from a different
        channel but we can identify the same user, add a new binding to the
        existing session.
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=30)

        # Look for active sessions that have a binding for this channel + sender
        result = await self.db.execute(
            select(ConversationSession).filter(
                and_(
                    ConversationSession.is_active == True,
                    ConversationSession.last_activity > cutoff_time,
                )
            ).order_by(desc(ConversationSession.last_activity))
        )
        sessions = result.scalars().all()

        for session in sessions:
            bindings = session.channel_bindings or {}
            binding = bindings.get(channel_type, {})
            if binding.get("sender_id") == sender_id:
                return session

        # No matching session found — create a new one
        # Use sender_id as user_id placeholder for channel users
        # (real user mapping would come from a user-linking service)
        user_id = f"{channel_type}:{sender_id}"

        new_binding = {
            channel_type: {
                "sender_id": sender_id,
                **platform_metadata,
            }
        }

        session = ConversationSession(
            id=uuid4(),
            user_id=user_id,
            state="greeting",
            started_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            context={},
            extracted_params={},
            channel_bindings=new_binding,
            primary_channel=channel_type,
            session_config={
                "multi_turn_enabled": True,
                "edition": "community",
                "max_turns": 20,
                "timeout_minutes": 30,
            },
            is_active=True,
            requires_human=False,
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_session(self, session_id: UUID) -> Optional[ConversationSession]:
        """Get a conversation session by ID."""
        result = await self.db.execute(
            select(ConversationSession).filter(ConversationSession.id == session_id)
        )
        return result.scalar_one_or_none()
    
    async def get_active_sessions(self, user_id: str) -> List[ConversationSession]:
        """Get all active sessions for a user."""
        from sqlalchemy.orm import selectinload
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=30)
        result = await self.db.execute(
            select(ConversationSession)
            .filter(
                and_(
                    ConversationSession.user_id == user_id,
                    ConversationSession.is_active == True,
                    ConversationSession.last_activity > cutoff_time
                )
            )
            .options(
                selectinload(ConversationSession.messages),
                selectinload(ConversationSession.actions)
            )
            .order_by(desc(ConversationSession.last_activity))
        )
        return result.scalars().all()
    
    async def update_session_state(
        self, 
        session_id: UUID, 
        new_state: str,
        context_update: Optional[Dict[str, Any]] = None
    ) -> ConversationSession:
        """Update session state and optionally merge context."""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        session.state = new_state
        session.last_activity = datetime.utcnow()
        
        if context_update:
            session.context = {**session.context, **context_update}
            
        await self.db.commit()
        await self.db.refresh(session)
        return session
    
    # === Message Processing ===
    
    async def process_message(
        self,
        session_id: UUID,
        content: str,
        user_id: str,
        channel_type: str = "web",
        external_message_id: Optional[str] = None,
    ) -> ConversationResponse:
        """
        Process a message in the conversation context.
        This is the main entry point for multi-turn processing.
        Channel info is stored on the message but does NOT affect processing logic.
        """
        session = await self.get_session(session_id)
        if not session:
            # Create new session if not found
            session = await self.create_session(user_id)

        # Store user message
        user_message = await self._store_message(
            session_id=session.id,
            role="user",
            content=content,
            channel_type=channel_type,
            external_message_id=external_message_id,
        )

        # IMPORTANT: Check for confirmations/cancellations BEFORE running unified analysis
        # when in "confirming_action" state, otherwise confirmation synonyms like "continue"
        # will be treated as new intents instead of confirmations
        if session.state == "confirming_action":
            if self._is_confirmation(content):
                # User confirmed - skip intent detection and go straight to execution
                intent_result = IntentResult(
                    intent=session.primary_intent or "unknown",
                    confidence=1.0,
                    entities={},
                    missing_params=[]
                )
                session.context["confirmed"] = True
            elif self._is_cancellation(content):
                # User cancelled - create cancellation intent
                intent_result = IntentResult(
                    intent="cancel",
                    confidence=1.0,
                    entities={},
                    missing_params=[]
                )
                session.context["cancelled"] = True
            else:
                # Not a confirmation or cancellation - detect intent normally
                intent_result = await self._detect_intent_with_context(session, content)
        else:
            # Not in confirming_action state - detect intent normally
            intent_result = await self._detect_intent_with_context(session, content)
        
        # Update session with detected intent
        if intent_result.confidence > 0.7:
            session.primary_intent = intent_result.intent
            session.intent_confidence = intent_result.confidence
            session.extracted_params = {
                **session.extracted_params,
                **intent_result.entities
            }
        
        # Determine next state and response
        base_response, next_state, quick_actions = await self._determine_response(
            session, intent_result, content
        )
        
        # Make response edition-aware
        response_content = self._generate_edition_aware_response(
            base_response,
            session,
            intent_result
        )
        
        # Store assistant response
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
        
        # Check if we should execute an action
        if next_state == "executing_action" and session.primary_intent:
            await self._trigger_action(session)

        # Extract automation_result from context for rich UX display
        automation_result = session.context.get('automation_result')

        return ConversationResponse(
            session_id=session.id,
            message=assistant_message,
            state=next_state,
            context=session.context,
            quick_actions=quick_actions,
            requires_clarification=next_state == "clarifying_details",
            clarification_options=await self._get_clarification_options(session, intent_result),
            automation_result=automation_result
        )
    
    async def _store_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        **kwargs
    ) -> ConversationMessage:
        """Store a message in the conversation history."""
        message = ConversationMessage(
            id=uuid4(),
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            message_config=kwargs.get('message_config', {}),
            suggested_actions=kwargs.get('suggested_actions', []),
            detected_intent=kwargs.get('detected_intent'),
            intent_confidence=kwargs.get('intent_confidence'),
            entities=kwargs.get('entities', {}),
            llm_model_used=kwargs.get('llm_model_used'),
            token_count=kwargs.get('token_count'),
            processing_time_ms=kwargs.get('processing_time_ms'),
            channel_type=kwargs.get('channel_type', 'web'),
            external_message_id=kwargs.get('external_message_id'),
        )

        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)

        # Auto-name conversation from first user message if not already named
        if role == "user":
            session = await self.get_session(session_id)
            if session and not session.name:
                # Count user messages to check if this is the first one
                result = await self.db.execute(
                    select(ConversationMessage).filter(
                        and_(
                            ConversationMessage.session_id == session_id,
                            ConversationMessage.role == "user"
                        )
                    )
                )
                user_messages = result.scalars().all()

                # If this is the first user message, generate name from content
                if len(user_messages) == 1:
                    # Take first 50 characters, remove newlines, and clean up
                    name_content = content.replace('\n', ' ').replace('\r', ' ').strip()
                    if len(name_content) > 50:
                        # Truncate at word boundary if possible
                        truncated = name_content[:50]
                        last_space = truncated.rfind(' ')
                        if last_space > 30:  # Only use word boundary if it's not too short
                            name_content = truncated[:last_space] + "..."
                        else:
                            name_content = truncated + "..."

                    session.name = name_content
                    await self.db.commit()
                    logger.info(f"[Conversation] Auto-named session {session_id}: '{name_content}'")

        return message
    
    # === Intent Detection ===
    
    async def _detect_intent_with_context(
        self,
        session: ConversationSession,
        content: str
    ) -> IntentDetectionResponse:
        """
        Detect intent considering conversation history and context.
        Integrates with existing NLP service for backward compatibility.
        """
        # Get recent messages for context
        result = await self.db.execute(
            select(ConversationMessage)
            .filter(ConversationMessage.session_id == session.id)
            .order_by(desc(ConversationMessage.timestamp))
            .limit(5)
        )
        recent_messages = result.scalars().all()
        
        # Prepare context for NLP service
        context_str = self._build_context_string(session, recent_messages)

        logger.info("[DEBUG] About to call LLM for intent detection")

        # FIRST: Check for high-confidence pattern matches that should bypass LLM
        # This ensures consistent behavior regardless of LLM quality
        import re
        content_lower = content.lower()

        # Pattern 1: Company automation - "automate my [company/business/firm/agency/...]"
        company_automation_pattern = r'automate\s+(my|our|the)\s+(\w+\s+)*(company|business|firm|agency|startup|organization|enterprise|shop|store|practice|clinic|studio|restaurant|hotel|salon)'
        if re.search(company_automation_pattern, content_lower):
            logger.info(f"[ConversationManager] Pattern match: company_automation (bypassing LLM)")
            nlp_result = {'action': 'company_automation', 'confidence': 0.95, 'entities': {}}
            session.context['_ml_enhanced'] = False  # Pattern match, not ML
            return IntentDetectionResponse(
                intent='company_automation',
                confidence=0.95,
                entities={},
                requires_params=[],
                has_params=[],
                missing_params=[]
            )

        # Pattern 2: Company automation with industry mention
        if 'automate' in content_lower and 'industry' in content_lower:
            logger.info(f"[ConversationManager] Pattern match: company_automation (industry mention, bypassing LLM)")
            return IntentDetectionResponse(
                intent='company_automation',
                confidence=0.9,
                entities={},
                requires_params=[],
                has_params=[],
                missing_params=[]
            )

        # Use LLM service directly for intelligent intent detection
        try:
            # Build conversation context for the LLM
            conversation_history = []
            for msg in recent_messages[-3:]:  # Last 3 messages for context
                conversation_history.append(f"{msg.role}: {msg.content}")

            context_str = "\n".join(conversation_history) if conversation_history else "No previous context"

            # Create a comprehensive prompt for intent detection
            intent_prompt = f"""You are analyzing a user's request in a conversation system.

Conversation History:
{context_str}

Current User Request: {content}

Analyze the user's intent and extract key information.

IMPORTANT - PRIORITIZE THESE INFORMATIONAL INTENTS FIRST:
1. get_help: User is asking about capabilities, help, or "what can you do"
   Examples: "what can you do", "help", "capabilities", "how do I", "can you"

2. search_resources: User is searching for existing resources or asking to "show me" something
   Examples: "show me", "find", "search for", "list my", "what workflows do I have"

THEN CHECK THESE ACTION INTENTS:
3. company_automation: User wants to automate their ENTIRE business/company/organization/startup (ABSTRACT, COMPANY-WIDE)
   Examples: "automate my company", "automate my business", "automate my startup", "automate my entire organization"
   KEY: Must be about automating the WHOLE company, not a specific process

4. create_workflow: User wants to create a SPECIFIC workflow/process for a particular task or function (CONCRETE, TASK-SPECIFIC)
   Examples:
   - "create workflow for X"
   - "automate [SPECIFIC PROCESS/TASK]" (e.g., "automate customer onboarding", "automate my sales pipeline")
   - "set up [SPECIFIC FUNCTION] automation" (e.g., "set up customer service automation")
   - "build process for Y"
   - "generate workflow for Z"
   KEY: Must be about a SPECIFIC task/process/function, even if it says "automate"

5. create_agent: User wants to create an AI or human agent
   Examples: "create an agent", "make an agent", "new agent"

6. create_pod: User wants to form a pod/team of agents
   Examples: "form a pod", "create pod", "make a team"

7. ai_governance: User wants AI/ML governance features (Business edition)
8. multi_tenancy: User wants multi-tenant features (Enterprise edition)

KEY DISTINCTION RULES:
- If user asks "what can you do" → get_help (NOT create_workflow)
- If user asks "help" or "how do I" → get_help
- If user asks "automate my company/business/startup" (WHOLE COMPANY) → company_automation
- If user asks "automate [specific task/process/function]" (SPECIFIC) → create_workflow
- If user asks "create workflow for [specific thing]" → create_workflow
- If user asks "set up [specific] automation" → create_workflow

CRITICAL DISTINCTION - "automate" keyword:
- "automate my company" = company_automation (entire business)
- "automate my startup" = company_automation (entire business)
- "automate customer onboarding" = create_workflow (specific process)
- "automate my sales pipeline" = create_workflow (specific function)
- "automate invoice processing" = create_workflow (specific task)
- "set up customer service automation" = create_workflow (specific department function)

Extract any entities mentioned such as workflow_name, agent_name, pod_name, search_query.

CRITICAL: Respond with ONLY valid JSON, no explanation text before or after.

{{
  "action": "<detected_intent>",
  "confidence": <0.0 to 1.0>,
  "entities": {{
    "entity_name": "entity_value"
  }},
  "reasoning": "Brief explanation of why you chose this intent"
}}"""

            # Get user's LLM settings with proper preference resolution
            model_override = session.context.get('preferred_model')  # MCP or context preference
            user_settings = await get_user_llm_settings(
                db=self.db,
                user_id=session.user_id,
                model_override=model_override,
                max_tokens=300
            )

            # Call LLM service for intent detection
            llm_response = await self.llm.generate(
                prompt=intent_prompt,
                user_settings=user_settings,
                task_type="intent_classification",
                temperature=0.2,  # Low temperature for consistent classification
                max_tokens=300,
                cache_key=f"intent_{session.id}_{hash(content)}"
            )

            # Parse the LLM response - try to extract JSON from text
            import json
            response_text = llm_response.text.strip()

            # DEBUG: Log the actual LLM response
            logger.info(f"[DEBUG] LLM raw response: {response_text[:200]}")

            # Try to find JSON in the response
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                logger.info(f"[DEBUG] Extracted JSON string (full): {json_str}")
                nlp_result = json.loads(json_str)
            else:
                # Fallback if no JSON found
                raise ValueError("No JSON found in LLM response")

            # Log successful LLM usage
            logger.info(
                f"[ConversationManager] LLM intent detection successful - "
                f"Model: {llm_response.model_used}, "
                f"Intent: {nlp_result.get('action')}, "
                f"Confidence: {nlp_result.get('confidence')}"
            )

            # Store LLM metadata in session context for transparency
            session.context['last_llm_model'] = llm_response.model_used
            session.context['last_llm_tokens'] = llm_response.tokens_used

        except Exception as e:
            logger.warning(f"LLM intent detection failed: {e}, falling back to pattern matching")
            # Fall back to basic pattern matching only if LLM fails
            nlp_result = self._basic_intent_detection(content)
        
        # Extract intent from NLP result
        intent = self._extract_intent_from_nlp(nlp_result)
        entities = self._extract_entities_from_nlp(nlp_result)
        
        # Check predefined intents
        predefined_intent = await self._match_predefined_intent(content, entities)
        if predefined_intent:
            intent = predefined_intent.name
            
        # Determine required and missing parameters
        required_params = []
        if predefined_intent:
            required_params = predefined_intent.required_params or []
            
        has_params = list(entities.keys())
        missing_params = [p for p in required_params if p not in has_params]
        
        return IntentDetectionResponse(
            intent=intent,
            confidence=nlp_result.get('confidence', 0.5),
            entities=entities,
            requires_params=required_params,
            has_params=has_params,
            missing_params=missing_params
        )
    
    async def _match_predefined_intent(
        self,
        content: str,
        entities: Dict[str, Any]
    ) -> Optional[ConversationIntent]:
        """Match user input against predefined intents."""
        # Query active intents
        result = await self.db.execute(
            select(ConversationIntent).filter(ConversationIntent.is_active == True)
        )
        intents = result.scalars().all()
        
        content_lower = content.lower()
        best_match = None
        best_score = 0.0
        
        for intent in intents:
            score = 0.0
            
            # Check example phrases
            for phrase in (intent.example_phrases or []):
                if phrase.lower() in content_lower:
                    score += 0.5
                    
            # Check for entity matches
            if intent.required_params:
                matching_params = sum(
                    1 for p in intent.required_params if p in entities
                )
                score += matching_params / len(intent.required_params)
                
            if score > best_score:
                best_score = score
                best_match = intent
                
        return best_match if best_score > 0.3 else None
    
    # === Response Generation ===
    
    async def _determine_response(
        self,
        session: ConversationSession,
        intent_result: IntentDetectionResponse,
        user_input: str
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Determine the appropriate response and next state.
        Returns (response_text, next_state, quick_actions)
        """
        current_state = session.state
        quick_actions = []
        
        # State machine logic
        if current_state == "greeting":
            if intent_result.intent:
                if intent_result.missing_params:
                    response = f"I understand you want to {intent_result.intent}. "
                    response += f"Let me gather some information first. "
                    response += self._get_clarification_question(
                        intent_result.missing_params[0]
                    )
                    next_state = "clarifying_details"
                else:
                    response = f"I'll help you with {intent_result.intent}. Let me process that for you."
                    next_state = "confirming_action"
            else:
                response = "Hello! I'm here to help you with AICtrlNet. What would you like to do today?"
                next_state = "gathering_intent"
                quick_actions = self._get_common_actions()
                
        elif current_state == "gathering_intent":
            if intent_result.intent:
                response = f"Got it! You want to {intent_result.intent}. "
                if intent_result.missing_params:
                    response += self._get_clarification_question(
                        intent_result.missing_params[0]
                    )
                    next_state = "clarifying_details"
                else:
                    response += "Shall I proceed?"
                    next_state = "confirming_action"
            else:
                response = "I'm not sure I understand. Could you tell me more about what you're trying to do?"
                next_state = "gathering_intent"
                quick_actions = self._get_common_actions()
                
        elif current_state == "clarifying_details":
            # Store the clarification in extracted params
            param_name = intent_result.missing_params[0] if intent_result.missing_params else None
            
            # Check if user is uncertain
            uncertainty_signals = ["don't know", "not sure", "unsure", "whatever",
                                  "default", "you choose", "you decide", "anything"]
            is_uncertain = any(signal in user_input.lower() for signal in uncertainty_signals)
            
            if is_uncertain and param_name:
                # Use smart defaults
                action_type = self._map_intent_to_action_type(session.primary_intent)
                defaults = self.smart_defaults.generate_defaults(action_type, session.extracted_params)
                
                if param_name in defaults:
                    default_info = defaults[param_name]
                    session.extracted_params[param_name] = default_info["value"]
                    session.extracted_params[f"{param_name}_was_defaulted"] = True
                    
                    # Create action orchestrator with user edition
                    if not self.action_orchestrator:
                        user_edition = self._get_user_edition(session)
                        self.action_orchestrator = ActionOrchestrator(self.db, user_edition)
                    
                    response = self.action_orchestrator.generate_default_explanation_message(
                        param_name,
                        default_info["value"],
                        default_info["explanation"]
                    )
                else:
                    # No default available, store user input
                    if param_name:
                        session.extracted_params[param_name] = user_input
                    response = "Got it!"
            elif param_name:
                # User provided specific value
                session.extracted_params[param_name] = user_input
                session.extracted_params[f"{param_name}_was_defaulted"] = False
                response = "Got it!"
            else:
                response = "I understand."
            
            # Check if we need more parameters using ActionOrchestrator logic
            action_type = self._map_intent_to_action_type(session.primary_intent)
            if not self.action_orchestrator:
                user_edition = self._get_user_edition(session)
                self.action_orchestrator = ActionOrchestrator(self.db, user_edition)
            
            # Update conversation context
            session.context["questions_asked"] = session.context.get("questions_asked", 0) + 1
            session.context["last_user_message"] = user_input
            
            should_ask, next_param, question = await self.action_orchestrator.should_ask_for_more(
                action_type,
                session.extracted_params,
                session.context
            )
            
            if should_ask:
                # Ask for next parameter
                response = f"{response} {question}"
                next_state = "clarifying_details"
            else:
                # Have enough params, prepare with defaults and confirm
                enriched = await self.action_orchestrator.prepare_action_with_defaults(
                    action_type,
                    session.extracted_params
                )
                session.extracted_params = enriched["params"]
                
                # Build confirmation message
                if enriched["defaults_used"]:
                    defaults_msg = "\n".join([
                        f"• {param}: {enriched['explanations'].get(param, 'default value')}"
                        for param in enriched["defaults_used"]
                    ])
                    response = f"{response}\n\nI'll proceed with these settings:\n{defaults_msg}\n\nReady to create?"
                else:
                    response = f"Perfect! I have all the information I need. Ready to {session.primary_intent}?"
                
                next_state = "confirming_action"
                quick_actions = [
                    {"label": "Yes, proceed", "action": "confirm"},
                    {"label": "Cancel", "action": "cancel"}
                ]
                
        elif current_state == "confirming_action":
            if self._is_confirmation(user_input):
                response = f"Executing {session.primary_intent}..."
                next_state = "executing_action"
            elif self._is_cancellation(user_input):
                response = "Cancelled. Is there anything else I can help you with?"
                next_state = "gathering_intent"
                quick_actions = self._get_common_actions()
            else:
                response = "Would you like me to proceed? Please say 'yes' to continue or 'no' to cancel."
                next_state = "confirming_action"
                
        elif current_state == "executing_action":
            response = "Task completed successfully! Is there anything else you'd like to do?"
            next_state = "completed"
            quick_actions = self._get_common_actions()
            
        else:  # completed or abandoned
            if intent_result.intent:
                response = f"Let me help you with {intent_result.intent}."
                next_state = "gathering_intent"
            else:
                response = "What else can I help you with?"
                next_state = "gathering_intent"
                quick_actions = self._get_common_actions()
                
        return response, next_state, quick_actions
    
    # === Action Execution ===
    
    async def _trigger_action(self, session: ConversationSession):
        """Trigger the action for the confirmed intent."""
        action = ConversationAction(
            id=uuid4(),
            session_id=session.id,
            action_type=session.primary_intent or "unknown",
            action_params=session.extracted_params,
            status="pending",
            created_at=datetime.utcnow()
        )
        
        self.db.add(action)
        await self.db.commit()
        
        # Execute through ActionOrchestrator
        try:
            if not self.action_orchestrator:
                user_edition = self._get_user_edition(session)
                self.action_orchestrator = ActionOrchestrator(self.db, user_edition)
            
            action_type = self._map_intent_to_action_type(session.primary_intent)
            # Enrich context with conversation provenance so downstream
            # workflow executions carry triggered_by="conversation"
            action_context = {
                **session.context,
                "triggered_by": "conversation",
                "session_id": str(session.id),
                "primary_channel": session.primary_channel or "web",
            }
            result = await self.action_orchestrator.execute_action(
                action_type=action_type,
                params=session.extracted_params,
                user_id=session.user_id,
                context=action_context
            )
            
            # Update action with result
            action.status = "completed" if result.get("success") else "failed"
            action.completed_at = datetime.utcnow()
            action.result = result
            await self.db.commit()
            
            return result
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            action.status = "failed"
            action.completed_at = datetime.utcnow()
            action.result = {"error": str(e)}
            await self.db.commit()
            
            return {"success": False, "error": str(e)}
    
    # === Helper Methods (sync methods that don't need async) ===
    
    def _map_intent_to_action_type(self, intent: str) -> ActionType:
        """Map detected intent to ActionType enum."""
        intent_lower = (intent or "").lower()

        # Company automation intent - highest priority for abstract requests
        if "automation" in intent_lower or "automate" in intent_lower or "company" in intent_lower:
            if any(word in intent_lower for word in ["company", "business", "organization", "enterprise"]):
                return ActionType.COMPANY_AUTOMATION

        if "workflow" in intent_lower:
            return ActionType.CREATE_WORKFLOW
        elif "agent" in intent_lower:
            return ActionType.CREATE_AGENT
        elif "pod" in intent_lower:
            return ActionType.FORM_POD
        elif "task" in intent_lower:
            return ActionType.EXECUTE_TASK
        elif "search" in intent_lower or "find" in intent_lower:
            return ActionType.SEARCH
        else:
            return ActionType.UNKNOWN
    
    def _build_context_string(
        self,
        session: ConversationSession,
        recent_messages: List[ConversationMessage]
    ) -> str:
        """Build context string from session and message history."""
        context_parts = []
        
        # Add session context
        if session.primary_intent:
            context_parts.append(f"Intent: {session.primary_intent}")
        if session.extracted_params:
            context_parts.append(f"Parameters: {json.dumps(session.extracted_params)}")
            
        # Add message history
        for msg in reversed(recent_messages[-3:]):
            context_parts.append(f"{msg.role}: {msg.content[:100]}")
            
        return " | ".join(context_parts)
    
    def _get_user_edition(self, session: ConversationSession) -> Edition:
        """Get user's edition from session context."""
        # Check session context for edition
        edition_str = session.context.get("edition", "community")
        
        # Map string to Edition enum
        edition_map = {
            "community": Edition.COMMUNITY,
            "business": Edition.BUSINESS,
            "enterprise": Edition.ENTERPRISE
        }
        
        return edition_map.get(edition_str.lower(), Edition.COMMUNITY)
    
    def _basic_intent_detection(self, content: str) -> Dict[str, Any]:
        """Basic intent detection without NLP service."""
        content_lower = content.lower()

        # Simple keyword-based intent detection
        # FIRST: Check for company automation (higher priority than generic workflow)
        # Pattern: "automate my [company/business/firm/agency/startup/organization]"
        import re
        company_automation_pattern = r'automate\s+(my|our|the)\s+(\w+\s+)*(company|business|firm|agency|startup|organization|enterprise|shop|store|practice|clinic|studio|restaurant|hotel|salon)'
        if re.search(company_automation_pattern, content_lower):
            return {'action': 'company_automation', 'confidence': 0.95}

        # Also check for "[industry] industry" pattern with automate
        if 'automate' in content_lower and 'industry' in content_lower:
            return {'action': 'company_automation', 'confidence': 0.9}

        # Check for automation and creation keywords (generic workflow)
        if any(word in content_lower for word in ['automate', 'automation', 'automated']):
            # Automation is essentially creating workflows
            return {'action': 'create_workflow', 'confidence': 0.8}
        elif any(word in content_lower for word in ['workflow', 'create workflow', 'make workflow', 'build workflow']):
            return {'action': 'create_workflow', 'confidence': 0.7}
        elif any(word in content_lower for word in ['setup', 'set up', 'implement', 'build']) and \
             any(word in content_lower for word in ['process', 'system', 'solution']):
            # Setup/implement process/system indicates workflow creation
            return {'action': 'create_workflow', 'confidence': 0.7}
        elif any(word in content_lower for word in ['agent', 'create agent', 'make agent']):
            return {'action': 'create_agent', 'confidence': 0.7}
        elif any(word in content_lower for word in ['pod', 'form pod', 'create pod']):
            return {'action': 'form_pod', 'confidence': 0.7, 'edition_required': 'business'}
        elif any(word in content_lower for word in ['ai governance', 'ml governance', 'risk assessment']):
            return {'action': 'ai_governance', 'confidence': 0.8, 'edition_required': 'business'}
        elif any(word in content_lower for word in ['tenant', 'multi-tenant', 'federation']):
            return {'action': 'multi_tenancy', 'confidence': 0.8, 'edition_required': 'enterprise'}
        elif any(word in content_lower for word in ['search', 'find', 'look for']):
            return {'action': 'search', 'confidence': 0.6}
        elif any(word in content_lower for word in ['help', 'what can', 'how do']):
            return {'action': 'help', 'confidence': 0.8}
        else:
            return {'action': '', 'confidence': 0.3}
    
    def _extract_intent_from_nlp(self, nlp_result: Dict[str, Any]) -> str:
        """Extract intent from NLP service result."""
        # Return the action directly - LLM now uses canonical intent names
        return nlp_result.get('action', 'unknown')
    
    def _extract_entities_from_nlp(self, nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from NLP service result."""
        entities = {}
        
        # Extract from parameters
        if 'parameters' in nlp_result:
            entities.update(nlp_result['parameters'])
            
        # Extract from parsed fields
        for field in ['workflow_name', 'agent_type', 'pod_name', 'search_query']:
            if field in nlp_result and nlp_result[field]:
                entities[field] = nlp_result[field]
                
        return entities
    
    def _get_clarification_question(self, param_name: str) -> str:
        """Get appropriate clarification question for a parameter."""
        questions = {
            'workflow_name': "What would you like to name this workflow?",
            'workflow_type': "What type of workflow do you want to create? (e.g., approval, data processing, monitoring)",
            'agent_name': "What should I call this agent?",
            'agent_type': "What type of agent do you need? (e.g., monitoring, execution, analysis)",
            'pod_name': "What name should I give to this pod?",
            'pod_members': "Which agents should be part of this pod?",
            'search_query': "What would you like to search for?",
            'resource_type': "What type of resource are you looking for?",
        }
        return questions.get(param_name, f"Could you provide the {param_name}?")
    
    async def _get_clarification_options(
        self,
        session: ConversationSession,
        intent_result: IntentDetectionResponse
    ) -> List[Dict[str, Any]]:
        """Get clarification options for the current context."""
        if not intent_result.missing_params:
            return []
            
        param = intent_result.missing_params[0]
        
        # Provide common options based on parameter type
        options_map = {
            'workflow_type': [
                {'value': 'approval', 'label': 'Approval Workflow'},
                {'value': 'data_processing', 'label': 'Data Processing'},
                {'value': 'monitoring', 'label': 'Monitoring Workflow'}
            ],
            'agent_type': [
                {'value': 'monitor', 'label': 'Monitoring Agent'},
                {'value': 'executor', 'label': 'Execution Agent'},
                {'value': 'analyzer', 'label': 'Analysis Agent'}
            ]
        }
        
        return options_map.get(param, [])
    
    def _get_common_actions(self) -> List[Dict[str, Any]]:
        """Get common quick actions."""
        return [
            {"label": "Create Workflow", "action": "create_workflow"},
            {"label": "Create Agent", "action": "create_agent"},
            {"label": "Form Pod", "action": "form_pod"},
            {"label": "Search Resources", "action": "search"},
            {"label": "Get Help", "action": "help"}
        ]
    
    def _is_confirmation(self, text: str) -> bool:
        """Check if text is a confirmation."""
        confirmations = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'proceed', 'confirm', 'go ahead', 'continue']
        return any(word in text.lower() for word in confirmations)
    
    def _is_cancellation(self, text: str) -> bool:
        """Check if text is a cancellation."""
        cancellations = ['no', 'nope', 'cancel', 'stop', 'abort', 'nevermind', 'forget it']
        return any(word in text.lower() for word in cancellations)
    
    # === Pattern Learning ===
    
    async def record_pattern(self, session_id: UUID):
        """Record successful conversation pattern for learning."""
        session = await self.get_session(session_id)
        if not session or session.state != "completed":
            return
            
        result = await self.db.execute(
            select(ConversationMessage)
            .filter(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.timestamp)
        )
        messages = result.scalars().all()
        
        if len(messages) < 2:
            return
            
        # Create pattern hash from intent and parameter keys
        pattern_key = f"{session.primary_intent}:{sorted(session.extracted_params.keys())}"
        pattern_hash = str(hash(pattern_key))[:64]
        
        # Check if pattern exists
        result = await self.db.execute(
            select(ConversationPattern)
            .filter(ConversationPattern.pattern_hash == pattern_hash)
        )
        pattern = result.scalar_one_or_none()
        
        if pattern:
            # Update existing pattern
            pattern.occurrence_count += 1
            pattern.success_count += 1
            pattern.last_seen = datetime.utcnow()
            pattern.average_turns = (
                (pattern.average_turns * (pattern.occurrence_count - 1) + len(messages)) 
                / pattern.occurrence_count
            )
        else:
            # Create new pattern
            pattern = ConversationPattern(
                id=uuid4(),
                pattern_hash=pattern_hash,
                pattern_type="intent_flow",
                conversation_flow={
                    "intent": session.primary_intent,
                    "params": list(session.extracted_params.keys()),
                    "message_count": len(messages)
                },
                success_criteria={"state": "completed"},
                occurrence_count=1,
                success_count=1,
                average_turns=len(messages),
                average_duration_seconds=(
                    (session.ended_at - session.started_at).total_seconds()
                    if session.ended_at else 0
                ),
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                confidence_score=0.5,
                is_promoted=False
            )
            self.db.add(pattern)
            
        await self.db.commit()
    
    # === Edition-Aware Methods ===
    
    def _check_edition_availability(self, intent: str, user_edition: Edition) -> Tuple[bool, Optional[str]]:
        """
        Check if an intent/feature is available in the user's edition.
        Returns (is_available, upgrade_message).
        """
        # Check with edition discovery
        edition_check = edition_discovery.detect_user_intent_edition(intent, user_edition)
        
        if edition_check["needs_upgrade"]:
            upgrade_msg = (
                f"This feature requires {edition_check['suggested_edition']} edition. "
                f"Would you like to learn more about upgrading?"
            )
            return False, upgrade_msg
        
        return True, None
    
    def _generate_edition_aware_response(
        self, 
        base_response: str, 
        session: ConversationSession,
        intent_result: IntentDetectionResponse
    ) -> str:
        """
        Modify response based on edition availability.
        """
        user_edition = self._get_user_edition(session)
        
        # Check if the detected intent is available
        if intent_result.intent:
            is_available, upgrade_msg = self._check_edition_availability(
                intent_result.intent, 
                user_edition
            )
            
            if not is_available:
                # Modify response to mention upgrade
                return (
                    f"I understand you'd like to {intent_result.intent.replace('_', ' ')}. "
                    f"{upgrade_msg} "
                    f"In the meantime, here are some features available in your {user_edition} edition: "
                    f"{self._get_available_features_summary(user_edition)}"
                )
        
        return base_response
    
    def _get_available_features_summary(self, edition: Edition) -> str:
        """Get a summary of available features for the edition."""
        # Get available endpoints for this edition
        available_endpoints = edition_discovery.get_available_endpoints(edition)
        
        # Categorize main features
        features = []
        if any('/workflow' in ep for ep in available_endpoints):
            features.append("workflow automation")
        if any('/task' in ep for ep in available_endpoints):
            features.append("task management")
        if any('/nlp' in ep for ep in available_endpoints):
            features.append("natural language processing")
        if any('/conversation' in ep for ep in available_endpoints):
            features.append("multi-turn conversations")
        if edition in [Edition.BUSINESS, Edition.ENTERPRISE]:
            if any('/ai-governance' in ep for ep in available_endpoints):
                features.append("AI governance")
            if any('/pod' in ep for ep in available_endpoints):
                features.append("pod orchestration")
        if edition == Edition.ENTERPRISE:
            if any('/tenant' in ep for ep in available_endpoints):
                features.append("multi-tenancy")
            if any('/federation' in ep for ep in available_endpoints):
                features.append("federation")
        
        return ", ".join(features) if features else "core features"

    async def _generate_llm_response(
        self,
        context: str,
        intent: str,
        entities: Dict[str, Any],
        state: str,
        user_input: str
    ) -> Optional[str]:
        """
        Generate a natural response using LLM service.
        Returns None if LLM fails, allowing fallback to template responses.
        """
        try:
            response_prompt = f"""You are AICtrlNet's intelligent assistant in a conversation.

Current state: {state}
User said: {user_input}
Detected intent: {intent}
Entities: {entities}

Previous context:
{context}

Generate a helpful, conversational response that:
1. Acknowledges what the user wants
2. Guides them naturally toward their goal
3. Asks for any missing information conversationally
4. Maintains context from the conversation

Keep the response concise and friendly."""

            # Use LLM service for response generation
            # Note: Using system user here since this is generating system responses
            from services.llm_helpers import get_system_llm_settings
            user_settings = get_system_llm_settings(
                max_tokens=300,
                temperature=0.7
            )

            llm_response = await self.llm.generate(
                prompt=response_prompt,
                user_settings=user_settings,
                task_type="conversation",
                temperature=0.7,
                max_tokens=300
            )

            return llm_response.text

        except Exception as e:
            logger.warning(f"LLM response generation failed: {e}")
            return None