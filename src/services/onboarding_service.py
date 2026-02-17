"""Onboarding Interview Service for Personal Agent setup.

Manages the 5-chapter, 8-question interview that configures the user's
personal agent personality, captures user context, and routes to the
right onboarding track.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from models.personal_agent import PersonalAgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Interview question definitions
# =============================================================================

INTERVIEW_QUESTIONS: Dict[Tuple[int, int], Dict[str, Any]] = {
    # Chapter 1: "Who You Are"
    (1, 1): {
        "text": "What's your role? Knowing this helps me tailor my responses to your needs.",
        "type": "free_text_with_suggestions",
        "options": [
            {"value": "software_engineer", "label": "Software Engineer"},
            {"value": "business_owner", "label": "Business Owner"},
            {"value": "marketing_manager", "label": "Marketing Manager"},
            {"value": "cto", "label": "CTO / Technical Leader"},
            {"value": "student", "label": "Student / Learner"},
            {"value": "researcher", "label": "Researcher"},
        ],
        "field": "user_context.role",
    },
    (1, 2): {
        "text": "What brings you here today?",
        "type": "choice",
        "options": [
            {"value": "automate_business", "label": "Automate my business",
             "description": "Set up automation workflows for your business processes"},
            {"value": "connect_agent", "label": "Connect my AI agent",
             "description": "Govern and connect an external AI agent runtime"},
            {"value": "explore", "label": "Explore the platform",
             "description": "Browse features and see what's possible"},
            {"value": "build_extend", "label": "Build / extend workflows",
             "description": "Create custom workflows and integrations"},
        ],
        "field": "user_context.primary_intent",
    },

    # Chapter 2: "How to Talk to You"
    (2, 1): {
        "text": "Pick a vibe — here's how I'd answer \"What can this platform do?\" in different tones:",
        "type": "tone_preview",
        "options": [
            {"value": "friendly",
             "label": "Friendly",
             "preview": "Hey! This platform helps you automate your work with AI-powered workflows. Think of it as your business autopilot — you set the rules, and I handle the rest."},
            {"value": "professional",
             "label": "Professional",
             "preview": "The platform provides enterprise-grade workflow automation with AI governance capabilities. It enables you to design, deploy, and monitor automated business processes with full compliance controls."},
            {"value": "casual",
             "label": "Casual",
             "preview": "So basically, it's a tool that lets you set up automations for pretty much anything. You tell it what to do, it does it. Pretty handy."},
        ],
        "field": "personality.tone",
    },
    (2, 2): {
        "text": "How much detail do you prefer?",
        "type": "style_preview",
        "options": [
            {"value": "concise",
             "label": "Keep it short",
             "preview": "Created workflow 'Daily Report'. It runs at 9 AM and emails your team a summary. Want to customize it?"},
            {"value": "detailed",
             "label": "Give me the details",
             "preview": "I've created a new workflow called 'Daily Report' with the following configuration:\n- **Trigger**: Daily at 9:00 AM UTC\n- **Action**: Aggregate metrics from connected data sources\n- **Output**: Formatted email report sent to your team distribution list\n- **Status**: Active and ready to run\n\nWould you like to customize the trigger time, add filters, or connect additional data sources?"},
            {"value": "step-by-step",
             "label": "Walk me through it",
             "preview": "Let me walk you through what I just set up:\n\n**Step 1**: I created a workflow called 'Daily Report'\n**Step 2**: I set it to trigger every day at 9 AM\n**Step 3**: It will pull metrics from your connected sources\n**Step 4**: Then format and email a summary to your team\n\nHere's what to do next..."},
        ],
        "field": "personality.style",
    },

    # Chapter 3: "What to Help With"
    (3, 1): {
        "text": "What are you working on? Tell me about your projects or areas of focus, and I'll tailor my suggestions.",
        "type": "free_text",
        "options": [],
        "field": "personality.expertise_areas",
    },
    (3, 2): {
        "text": "When I can take action, what's your preference?",
        "type": "choice",
        "options": [
            {"value": "observe",
             "label": "Just show me insights",
             "description": "I'll monitor and report, but never take action without you asking"},
            {"value": "suggest",
             "label": "Suggest things, I'll decide",
             "description": "I'll recommend actions and wait for your approval"},
            {"value": "supervised",
             "label": "Go ahead with simple stuff, ask for big things",
             "description": "I'll handle routine tasks automatically and confirm before bigger changes"},
            {"value": "autonomous",
             "label": "Full autopilot — I trust you",
             "description": "I'll take action proactively and notify you of what I did"},
        ],
        "field": "user_context.comfort_level",
    },

    # Chapter 4: "The Details"
    (4, 1): {
        "text": "Give your assistant a name — or keep the suggestion below.",
        "type": "name_input",
        "options": [],
        "field": "agent_name",
    },

    # Chapter 5: "The Reveal"
    (5, 1): {
        "text": "Here's your personalized assistant — ready to go!",
        "type": "reveal",
        "options": [],
        "field": None,
    },
}

# Questions per chapter (for navigation)
QUESTIONS_PER_CHAPTER = {1: 2, 2: 2, 3: 2, 4: 1, 5: 1}
CHAPTER_TITLES = {
    1: "Who You Are",
    2: "How to Talk to You",
    3: "What to Help With",
    4: "The Details",
    5: "The Reveal + Your Path",
}

# =============================================================================
# Personality type mapping (30 combos)
# =============================================================================

PERSONALITY_TYPES = {
    ("friendly", "concise"): "The Quick Helper",
    ("friendly", "detailed"): "The Detailed Helper",
    ("friendly", "balanced"): "The Helper",
    ("friendly", "step-by-step"): "The Guide",
    ("friendly", "conversational"): "The Companion",
    ("professional", "concise"): "The Executive",
    ("professional", "detailed"): "The Strategist",
    ("professional", "balanced"): "The Consultant",
    ("professional", "step-by-step"): "The Analyst",
    ("professional", "conversational"): "The Advisor",
    ("casual", "concise"): "The Quick Buddy",
    ("casual", "detailed"): "The Explainer",
    ("casual", "balanced"): "The Buddy",
    ("casual", "step-by-step"): "The Tutor",
    ("casual", "conversational"): "The Chat Buddy",
    ("technical", "concise"): "The Engineer",
    ("technical", "detailed"): "The Architect",
    ("technical", "balanced"): "The Tech Lead",
    ("technical", "step-by-step"): "The Debugger",
    ("technical", "conversational"): "The Hacker",
    ("supportive", "concise"): "The Quick Coach",
    ("supportive", "detailed"): "The Mentor",
    ("supportive", "balanced"): "The Coach",
    ("supportive", "step-by-step"): "The Teacher",
    ("supportive", "conversational"): "The Encourager",
    ("formal", "concise"): "The Brief",
    ("formal", "detailed"): "The Scholar",
    ("formal", "balanced"): "The Counselor",
    ("formal", "step-by-step"): "The Professor",
    ("formal", "conversational"): "The Diplomat",
}

# Name suggestions by personality type prefix
NAME_SUGGESTIONS = {
    "friendly": ["Atlas", "Nova", "Scout", "Sunny"],
    "professional": ["Sage", "Sterling", "Morgan", "Maxwell"],
    "casual": ["Chip", "Dash", "Ziggy", "Pixel"],
    "technical": ["Logic", "Vector", "Nexus", "Cipher"],
    "supportive": ["Haven", "Echo", "Ember", "Bloom"],
    "formal": ["Ashford", "Sinclair", "Reeves", "Mercer"],
}


class OnboardingService:
    """Manages the personal agent onboarding interview."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    async def _get_config(self, user_id: str) -> PersonalAgentConfig:
        """Get or create the PersonalAgentConfig for a user."""
        result = await self.db.execute(
            select(PersonalAgentConfig).where(
                PersonalAgentConfig.user_id == user_id
            )
        )
        config = result.scalar_one_or_none()

        if config is None:
            config = PersonalAgentConfig(
                id=str(uuid.uuid4()),
                user_id=user_id,
                agent_name="My Assistant",
                personality={"tone": "friendly", "style": "concise", "expertise_areas": []},
                preferences={
                    "notifications": {"enabled": True, "frequency": "daily"},
                    "auto_actions": {"enabled": False, "require_confirmation": True},
                },
                active_workflows=[],
                max_workflows=5,
                status="active",
                onboarding_state={
                    "status": "not_started",
                    "current_chapter": 1,
                    "completed_chapters": [],
                },
                user_context={},
            )
            self.db.add(config)
            try:
                await self.db.commit()
                await self.db.refresh(config)
            except IntegrityError:
                await self.db.rollback()
                result = await self.db.execute(
                    select(PersonalAgentConfig).where(
                        PersonalAgentConfig.user_id == user_id
                    )
                )
                config = result.scalar_one()

        return config

    async def get_state(self, user_id: str) -> Dict[str, Any]:
        """Get current onboarding state."""
        config = await self._get_config(user_id)
        state = config.onboarding_state or {
            "status": "not_started",
            "current_chapter": 1,
            "completed_chapters": [],
        }
        # Add computed personality type if completed
        if state.get("status") == "completed":
            personality = config.personality or {}
            state["personality_type"] = self._compute_personality_type(
                personality.get("tone", "friendly"),
                personality.get("style", "concise"),
            )
        return state

    async def start_or_resume(self, user_id: str) -> Dict[str, Any]:
        """Start or resume the onboarding interview."""
        config = await self._get_config(user_id)
        state = dict(config.onboarding_state or {})

        if state.get("status") == "completed":
            # Already completed — return summary
            return self._build_summary_response(config)

        if state.get("status") != "in_progress":
            state["status"] = "in_progress"
            state["current_chapter"] = 1
            state["completed_chapters"] = state.get("completed_chapters", [])
            state["started_at"] = datetime.now(timezone.utc).isoformat()
            config.onboarding_state = state
            await self.db.commit()

        chapter = state.get("current_chapter", 1)
        question = self._get_current_question_in_chapter(state, chapter)

        q_def = INTERVIEW_QUESTIONS.get((chapter, question))
        if not q_def:
            # Past last question — complete
            return await self._complete_interview(config)

        return {
            "status": state["status"],
            "current_chapter": chapter,
            "current_question": question,
            "question_text": q_def["text"],
            "question_type": q_def["type"],
            "options": self._prepare_options(q_def, config),
            "total_chapters": 5,
            "completed_chapters": state.get("completed_chapters", []),
            "message": f"Chapter {chapter} of 5 — {CHAPTER_TITLES.get(chapter, '')}",
        }

    async def process_answer(
        self, user_id: str, chapter: int, question: int, answer: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process an answer and apply it to the config."""
        config = await self._get_config(user_id)
        state = dict(config.onboarding_state or {})

        if state.get("status") not in ("in_progress", "not_started"):
            if state.get("status") == "completed":
                # Allow re-answering even when completed (redo flow)
                state["status"] = "in_progress"
            else:
                return {"applied": False, "message": "Interview not active."}

        q_def = INTERVIEW_QUESTIONS.get((chapter, question))
        if not q_def:
            return {"applied": False, "message": f"Invalid question ({chapter}, {question})."}

        # Apply answer to the appropriate field
        self._apply_answer(config, q_def, answer)

        # Mark chapter complete if this was the last question in it
        completed = state.get("completed_chapters", [])
        max_q = QUESTIONS_PER_CHAPTER.get(chapter, 1)
        if question >= max_q and chapter not in completed:
            completed.append(chapter)
            state["completed_chapters"] = sorted(completed)

        # Advance to next question/chapter
        next_chapter, next_question = self._advance(chapter, question)

        if next_chapter is None:
            # Interview complete
            state["status"] = "completed"
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            personality = config.personality or {}
            state["personality_type"] = self._compute_personality_type(
                personality.get("tone", "friendly"),
                personality.get("style", "concise"),
            )
            config.onboarding_state = state
            config.updated_at = datetime.utcnow()
            await self.db.commit()

            return {
                "chapter": chapter,
                "question": question,
                "applied": True,
                "next_chapter": None,
                "next_question": None,
                "personality_preview": self._build_personality_preview(config),
                "message": "Interview complete! Your assistant is ready.",
                "suggested_action": self._compute_suggested_action(config),
            }

        state["current_chapter"] = next_chapter
        state["current_question"] = next_question
        config.onboarding_state = state
        config.updated_at = datetime.utcnow()
        await self.db.commit()

        # Build next question info
        next_q_def = INTERVIEW_QUESTIONS.get((next_chapter, next_question))
        return {
            "chapter": chapter,
            "question": question,
            "applied": True,
            "next_chapter": next_chapter,
            "next_question": next_question,
            "next_question_text": next_q_def["text"] if next_q_def else None,
            "next_question_type": next_q_def["type"] if next_q_def else None,
            "next_options": self._prepare_options(next_q_def, config) if next_q_def else [],
            "personality_preview": self._build_personality_preview(config),
            "message": self._build_transition_message(chapter, next_chapter, config),
        }

    async def skip(self, user_id: str, skip_all: bool = False,
                   chapter: Optional[int] = None, question: Optional[int] = None) -> Dict[str, Any]:
        """Skip a question, chapter, or the entire interview."""
        config = await self._get_config(user_id)
        state = dict(config.onboarding_state or {})

        if skip_all:
            state["status"] = "skipped"
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            config.onboarding_state = state
            config.updated_at = datetime.utcnow()
            await self.db.commit()
            return {
                "status": "skipped",
                "message": "Interview skipped. You can always set up your assistant later from Settings.",
            }

        # Skip current question and advance
        current_ch = chapter or state.get("current_chapter", 1)
        current_q = question or self._get_current_question_in_chapter(state, current_ch)
        next_chapter, next_question = self._advance(current_ch, current_q)

        if next_chapter is None:
            return await self._complete_interview(config)

        completed = state.get("completed_chapters", [])
        max_q = QUESTIONS_PER_CHAPTER.get(current_ch, 1)
        if current_q >= max_q and current_ch not in completed:
            completed.append(current_ch)
            state["completed_chapters"] = sorted(completed)

        state["current_chapter"] = next_chapter
        state["status"] = "in_progress"
        config.onboarding_state = state
        config.updated_at = datetime.utcnow()
        await self.db.commit()

        next_q_def = INTERVIEW_QUESTIONS.get((next_chapter, next_question))
        return {
            "status": "in_progress",
            "current_chapter": next_chapter,
            "current_question": next_question,
            "question_text": next_q_def["text"] if next_q_def else "",
            "question_type": next_q_def["type"] if next_q_def else "choice",
            "options": self._prepare_options(next_q_def, config) if next_q_def else [],
            "message": "Skipped. Here's the next question.",
        }

    async def reset(self, user_id: str) -> Dict[str, Any]:
        """Reset the interview to start over."""
        config = await self._get_config(user_id)
        config.onboarding_state = {
            "status": "not_started",
            "current_chapter": 1,
            "completed_chapters": [],
        }
        config.user_context = {}
        config.updated_at = datetime.utcnow()
        await self.db.commit()
        return {"status": "not_started", "message": "Interview reset. Start fresh anytime."}

    # ------------------------------------------------------------------
    # Tool-facing method (called by update_onboarding tool during chat)
    # ------------------------------------------------------------------

    async def update_from_conversation(
        self, user_id: str, chapter: int, question: int, value: str = ""
    ) -> Dict[str, Any]:
        """Process an onboarding answer submitted via the conversation tool.

        Called by the ``update_onboarding`` tool during chat.  Accepts the
        same parameter names as the tool definition (chapter, question, value).
        Auto-starts the interview if not already in progress.
        """
        # LLMs may send chapter/question as strings — coerce to int
        try:
            chapter = int(chapter)
            question = int(question)
        except (TypeError, ValueError):
            return {"applied": False, "message": f"Invalid chapter/question: {chapter}, {question}"}

        config = await self._get_config(user_id)
        state = dict(config.onboarding_state or {})

        # Auto-start if needed
        if state.get("status") in ("not_started", None):
            state["status"] = "in_progress"
            state["current_chapter"] = 1
            state["completed_chapters"] = []
            state["started_at"] = datetime.now(timezone.utc).isoformat()
            config.onboarding_state = state
            await self.db.commit()

        answer = {"value": value}
        return await self.process_answer(user_id, chapter, question, answer)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_answer(self, config: PersonalAgentConfig, q_def: Dict[str, Any], answer: Dict[str, Any]) -> None:
        """Apply an answer to the appropriate config field."""
        field = q_def.get("field")
        if not field:
            return

        # Try standard keys first, then fall back to field-specific key
        # e.g. for field="user_context.role", also check answer["role"]
        field_key = field.rsplit(".", 1)[-1] if "." in field else field
        value = answer.get("value", answer.get("text", answer.get(field_key, "")))

        if field == "personality.tone":
            personality = dict(config.personality or {})
            personality["tone"] = value
            config.personality = personality

        elif field == "personality.style":
            personality = dict(config.personality or {})
            personality["style"] = value
            config.personality = personality

        elif field == "personality.expertise_areas":
            personality = dict(config.personality or {})
            # Parse free text into list
            if isinstance(value, list):
                areas = value
            elif isinstance(value, str):
                areas = [a.strip() for a in value.replace(";", ",").split(",") if a.strip()]
            else:
                areas = []
            personality["expertise_areas"] = areas[:10]
            config.personality = personality

        elif field == "agent_name":
            if value and isinstance(value, str) and value.strip():
                config.agent_name = value.strip()

        elif field.startswith("user_context."):
            ctx_key = field.split(".", 1)[1]
            user_context = dict(config.user_context or {})
            user_context[ctx_key] = value
            config.user_context = user_context

    def _advance(self, chapter: int, question: int) -> Tuple[Optional[int], Optional[int]]:
        """Advance to next question. Returns (None, None) when interview is done."""
        max_q = QUESTIONS_PER_CHAPTER.get(chapter, 1)
        if question < max_q:
            return chapter, question + 1

        # Move to next chapter
        next_ch = chapter + 1
        if next_ch > 5:
            return None, None
        return next_ch, 1

    def _get_current_question_in_chapter(self, state: Dict, chapter: int) -> int:
        """Determine which question we're on in a chapter."""
        # Use tracked current_question if on the current chapter
        if state.get("current_chapter") == chapter and "current_question" in state:
            return state["current_question"]
        completed = state.get("completed_chapters", [])
        if chapter in completed:
            return QUESTIONS_PER_CHAPTER.get(chapter, 1)
        return 1

    def _prepare_options(self, q_def: Dict[str, Any], config: PersonalAgentConfig) -> List[Dict[str, Any]]:
        """Prepare question options, including dynamic suggestions."""
        if q_def["type"] == "name_input":
            tone = (config.personality or {}).get("tone", "friendly")
            suggestions = NAME_SUGGESTIONS.get(tone, ["Atlas", "Nova"])
            return [{"value": s, "label": s} for s in suggestions]

        if q_def["type"] == "reveal":
            return []

        return q_def.get("options", [])

    def _compute_personality_type(self, tone: str, style: str) -> str:
        """Derive the personality type label from tone + style."""
        return PERSONALITY_TYPES.get((tone, style), "The Assistant")

    def _generate_name_suggestion(self, config: PersonalAgentConfig) -> str:
        """Suggest an agent name based on personality."""
        tone = (config.personality or {}).get("tone", "friendly")
        suggestions = NAME_SUGGESTIONS.get(tone, ["Atlas"])
        return suggestions[0]

    def _build_personality_preview(self, config: PersonalAgentConfig) -> Dict[str, Any]:
        """Build a preview of the current personality state."""
        personality = config.personality or {}
        tone = personality.get("tone", "friendly")
        style = personality.get("style", "concise")
        return {
            "agent_name": config.agent_name,
            "tone": tone,
            "style": style,
            "expertise_areas": personality.get("expertise_areas", []),
            "personality_type": self._compute_personality_type(tone, style),
        }

    def _build_transition_message(self, from_chapter: int, to_chapter: int,
                                   config: PersonalAgentConfig) -> str:
        """Build a transition message between chapters."""
        if from_chapter == to_chapter:
            return "Got it! Next question."

        personality = config.personality or {}
        tone = personality.get("tone", "friendly")

        # Use the tone the user just picked for immediate feedback
        transitions = {
            "friendly": {
                2: "Great, now let's figure out how you like to communicate!",
                3: "Awesome! Now let's talk about what you're working on.",
                4: "Almost done! One more thing...",
                5: "And here's the big reveal!",
            },
            "professional": {
                2: "Noted. Let's configure your communication preferences.",
                3: "Moving to expertise configuration.",
                4: "Final configuration step.",
                5: "Here is your configuration summary.",
            },
            "casual": {
                2: "Cool, let's figure out our vibe!",
                3: "Nice! What are you into?",
                4: "Quick one more...",
                5: "Drumroll please...",
            },
        }

        tone_transitions = transitions.get(tone, transitions["friendly"])
        return tone_transitions.get(to_chapter, f"Moving to Chapter {to_chapter}.")

    def _compute_suggested_action(self, config: PersonalAgentConfig) -> Dict[str, Any]:
        """Compute the suggested next action based on role + intent."""
        ctx = config.user_context or {}
        intent = ctx.get("primary_intent", "explore")
        role = ctx.get("role", "").lower()

        is_technical = role in ("software_engineer", "cto", "researcher") or "engineer" in role or "developer" in role

        if intent == "automate_business":
            return {
                "type": "tool",
                "tool_name": "get_help",
                "label": "Tell me about your company and I'll create an automation plan",
                "description": "Start by describing your business processes, and I'll suggest workflows to automate them.",
                "args": {"topic": "automation"},
            }
        elif intent == "connect_agent":
            return {
                "type": "guide",
                "label": "Connect your agent runtime",
                "description": "Learn how to connect and govern external AI agents through the platform.",
                "guide_ref": "connecting-openclaw-to-aictrlnet",
            }
        elif intent == "build_extend" or is_technical:
            return {
                "type": "tool",
                "tool_name": "list_workflow_templates",
                "label": "Browse workflow templates",
                "description": "Explore 183+ workflow templates to find one that fits your use case.",
                "args": {"limit": 10},
            }
        else:
            # Generic explore
            return {
                "type": "suggestions",
                "label": "Here are 3 things to try",
                "items": [
                    {"label": "Ask me anything", "description": "Try asking 'What can this platform do?'"},
                    {"label": "Create a workflow", "description": "Say 'Create a daily report workflow'"},
                    {"label": "Browse templates", "description": "Say 'Show me workflow templates'"},
                ],
            }

    def _build_summary_response(self, config: PersonalAgentConfig) -> Dict[str, Any]:
        """Build the completed interview summary."""
        preview = self._build_personality_preview(config)
        state = config.onboarding_state or {}
        return {
            "status": "completed",
            "current_chapter": 5,
            "current_question": 1,
            "question_text": "Here's your personalized assistant — ready to go!",
            "question_type": "reveal",
            "options": [],
            "total_chapters": 5,
            "completed_chapters": [1, 2, 3, 4, 5],
            "message": f"Meet {preview['agent_name']} — Your {preview['personality_type']}",
            "personality_preview": preview,
            "suggested_action": self._compute_suggested_action(config),
        }

    async def _complete_interview(self, config: PersonalAgentConfig) -> Dict[str, Any]:
        """Mark interview as completed and return summary."""
        state = dict(config.onboarding_state or {})
        personality = config.personality or {}
        state["status"] = "completed"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        state["completed_chapters"] = [1, 2, 3, 4, 5]
        state["personality_type"] = self._compute_personality_type(
            personality.get("tone", "friendly"),
            personality.get("style", "concise"),
        )
        config.onboarding_state = state
        config.updated_at = datetime.utcnow()
        await self.db.commit()
        return self._build_summary_response(config)
