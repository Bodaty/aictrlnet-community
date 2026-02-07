"""Tool-Aware Conversation Service for AICtrlNet Intelligent Assistant v4.

Community Edition - Base tool-calling conversation capabilities.

This service extends the ConversationService with tool-calling capabilities,
enabling the LLM to invoke tools during conversation to perform actions, gather
information, and orchestrate workflows.

The tool-aware conversation flow:
1. User message → LLM with tool definitions
2. LLM decides: direct response OR tool call(s)
3. If tool call(s): execute via ToolDispatcher, feed results back to LLM
4. LLM generates final response incorporating tool results
5. Update conversation state appropriately

Business Edition extends this with ML-enhanced tool selection and analytics.
"""

import logging
import time
from typing import Optional, List, Dict, Any, AsyncGenerator
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from services.tool_dispatcher import ToolDispatcher, CORE_TOOLS, Edition
from services.system_prompt_assembler import SystemPromptAssembler
from schemas.conversation import ConversationResponse
from llm.service import LLMService
from llm.models import ToolDefinition, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolAwareConversationService:
    """Base conversation service with tool-calling capabilities.

    Community Edition provides:
    - Tool-aware LLM responses via generate_with_tools()
    - Tool execution via ToolDispatcher
    - Tool result feedback loop
    - executing_tools state management
    - SSE event streaming for tool execution progress

    Business Edition extends with:
    - ML-enhanced tool selection
    - Analytics and tracking
    - AI Governance integration
    """

    def __init__(self, db: AsyncSession, edition: Edition = Edition.COMMUNITY):
        """Initialize the tool-aware conversation service.

        Args:
            db: Database session
            edition: Current edition (affects available tools)
        """
        self.db = db
        self.edition = edition
        self.tool_dispatcher = ToolDispatcher(db, edition)
        self.prompt_assembler = SystemPromptAssembler(db)
        self._tools_initialized = False

        # Tool execution tracking
        self._tool_execution_history: Dict[str, List[Dict[str, Any]]] = {}

        # LLM service (lazy loaded)
        self._llm_service: Optional[LLMService] = None

        logger.info(f"[v4] ToolAwareConversationService initialized for {edition.value} edition")

    @property
    def llm_service(self) -> Optional[LLMService]:
        """Lazy-load LLM service."""
        if self._llm_service is None:
            try:
                # LLMService is a singleton - no args needed
                self._llm_service = LLMService()
            except Exception as e:
                logger.warning(f"[v4] Could not initialize LLM service: {e}")
        return self._llm_service

    async def _ensure_tools_initialized(self) -> None:
        """Ensure tool dispatcher services are loaded."""
        if self._tools_initialized:
            return

        # The dispatcher lazy-loads services on first invoke
        self._tools_initialized = True
        logger.info(f"[v4] Tool dispatcher ready with {len(self.tool_dispatcher.get_available_tools())} tools")

    def _get_tool_definitions_for_llm(self) -> List[ToolDefinition]:
        """Get tool definitions available for the current edition."""
        return self.tool_dispatcher.get_available_tools()

    async def execute_tool_calls(
        self,
        tool_calls: List[ToolCall],
        user_id: str,
        session_context: Optional[Dict[str, Any]] = None
    ) -> List[ToolResult]:
        """Execute tool calls via the dispatcher.

        Args:
            tool_calls: List of tool calls from LLM
            user_id: User ID for authorization
            session_context: Current session context

        Returns:
            List of ToolResult objects
        """
        results = []

        for tool_call in tool_calls:
            logger.info(f"[v4] Executing tool: {tool_call.name}")

            result = await self.tool_dispatcher.invoke(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                user_id=user_id,
                context=session_context or {}
            )

            # Add tool name to result data for tracking
            if result.data is None:
                result.data = {}
            result.data['tool_name'] = tool_call.name

            results.append(result)

            logger.info(f"[v4] Tool {tool_call.name} completed: success={result.success}")

        return results

    async def determine_post_tool_state(
        self,
        tool_results: List[ToolResult]
    ) -> str:
        """Determine the conversation state after tool execution.

        State transitions based on tool results:
        - All successful → completed or gathering_intent
        - Some failed → clarifying_details
        - Needs confirmation → confirming_action
        """
        all_successful = all(r.success for r in tool_results)
        any_needs_clarification = any(
            r.recovery_strategy and r.recovery_strategy.value == "clarify"
            for r in tool_results
        )

        if any_needs_clarification:
            return "clarifying_details"
        elif all_successful:
            # Check if this was a terminal action
            terminal_tools = ['execute_workflow', 'execute_agent', 'delete_workflow']
            executed_tools = [
                r.data.get('tool_name') for r in tool_results if r.data
            ]
            if any(t in terminal_tools for t in executed_tools):
                return "completed"
            else:
                return "gathering_intent"  # Ready for more
        else:
            return "gathering_intent"

    async def stream_tool_execution(
        self,
        content: str,
        user_id: str,
        session_context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream tool execution progress as SSE events.

        Yields events for:
        - tool_start: When a tool begins execution
        - tool_progress: Progress updates during execution
        - tool_complete: When a tool finishes
        - tool_error: If a tool fails
        - response: Final response text

        This is used by the SSE endpoint for real-time UI updates.
        """
        await self._ensure_tools_initialized()

        # Get tools
        tools = self._get_tool_definitions_for_llm()
        system_prompt = await self.prompt_assembler.assemble(
            edition=self.edition.value,
            session_context=session_context,
        )

        yield {"event": "thinking", "data": {"message": "Analyzing your request..."}}

        if not self.llm_service:
            yield {"event": "error", "data": {"message": "LLM service unavailable"}}
            return

        try:
            llm_response = await self.llm_service.generate_with_tools(
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

                    result = await self.tool_dispatcher.invoke(
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

                # Generate final response — use LLM text if available, else simple summary
                if llm_response.text:
                    response_content = llm_response.text
                else:
                    successful = [r for r in tool_results if r.success]
                    response_content = f"Executed {len(successful)} tool(s) successfully."

                yield {
                    "event": "response",
                    "data": {
                        "content": response_content,
                        "tools_used": [tc.name for tc in llm_response.tool_calls],
                        "all_successful": all(r.success for r in tool_results)
                    }
                }
            else:
                # No tools - just response
                yield {
                    "event": "response",
                    "data": {
                        "content": llm_response.text or "I'm not sure how to help with that. Could you be more specific?",
                        "tools_used": []
                    }
                }

            yield {"event": "complete", "data": {"status": "done"}}

        except Exception as e:
            logger.error(f"[v4] Streaming error: {e}")
            yield {
                "event": "error",
                "data": {"message": str(e)}
            }
