"""AI Process node implementation for AI/ML processing tasks."""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..base_node import BaseNode
from ..models import NodeConfig
from ..template_utils import (
    resolve_templates,
    get_adapter_credentials_for_tenant,
)
from events.event_bus import event_bus
from adapters.registry import adapter_registry
from adapters.models import AdapterConfig, AdapterCategory, AdapterRequest, AdapterStatus


logger = logging.getLogger(__name__)


_JSON_LIKE_PREFIX = re.compile(r"^\s*[\{\[]")
_CODE_FENCE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    m = _CODE_FENCE.match(text.strip())
    return m.group(1) if m else text


def _escape_control_chars_in_strings(text: str) -> str:
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            code = ord(ch)
            if code < 0x20:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                elif ch == "\b":
                    out.append("\\b")
                elif ch == "\f":
                    out.append("\\f")
                else:
                    out.append("\\u%04x" % code)
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def _try_parse_llm_json(text: Any) -> Optional[Any]:
    """Best-effort parse of LLM output into a Python object.

    LLMs frequently emit "almost-JSON" — newlines and tabs inside string
    literals instead of escape sequences, sometimes wrapped in a markdown
    code fence. We strip the fence, try strict json.loads, and on failure
    do a single repair pass that escapes raw control chars inside string
    literals. Returns None if both attempts fail or the input doesn't
    look JSON-shaped.
    """
    if not isinstance(text, str) or not text:
        return None
    candidate = _strip_code_fence(text)
    if not _JSON_LIKE_PREFIX.match(candidate):
        return None
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        try:
            return json.loads(_escape_control_chars_in_strings(candidate))
        except (ValueError, TypeError):
            return None


# --- Narrative-style routing -------------------------------------------------
# Small Ollama models (1-3B) drift on dollar amounts inside prose even when the
# numbers are provided verbatim. For nodes whose prompt or name signals
# narrative work, we (a) prepend a verbatim-copy guardrail rendered from a
# shared template and (b) for Ollama-family adapters, upgrade the model to
# the QUALITY tier default (currently llama3.1:8b-instruct-q4_K_M).

# Narrative-task signals: prose-writing intent, not output-shape mentions.
# We deliberately exclude bare "summary"/"report" because templates often
# reference them as JSON output keys (e.g. {"summary": {...}}) without
# intending narrative work. "executive_summary" is specific to the CFO
# pattern; bare "summary" is not. We do NOT negate on "JSON only" because
# narrative-in-JSON is the normal case (executive_summary is a JSON string
# field that happens to contain prose) — drift hits the prose content, not
# the JSON structure.
_NARRATIVE_KEYWORDS = (
    "narrative",
    "commentary",
    "executive_summary",
    "executive summary",
)
_SMALL_OLLAMA_MODELS = frozenset({"llama3.2:1b", "llama3.2:3b"})
_OLLAMA_ADAPTER_IDS = frozenset({"ollama"})

# Module-level singleton — PromptTemplateLoader walks the prompts dir on init,
# so we instantiate once instead of per LLM call.
_prompt_loader = None


def _get_prompt_loader():
    global _prompt_loader
    if _prompt_loader is None:
        from services.prompt_template_loader import PromptTemplateLoader
        _prompt_loader = PromptTemplateLoader()
    return _prompt_loader


def _normalize_response_text(response) -> None:
    """In-place: ensure response.data['text'] holds the LLM content.

    Different providers return content under different keys:
      - Ollama /api/generate → data['response']
      - Ollama /api/chat     → data['message']['content']
      - OpenAI / Claude chat → data['choices'][0]['message']['content']
      - Anthropic blocks     → data['content'] = [{'type': 'text', 'text': ...}]
    Callers read response.data['text'] regardless.
    """
    if not getattr(response, "data", None):
        return
    if response.data.get("text"):
        return
    text_value: Any = ""
    # Ollama generate
    if not text_value:
        text_value = response.data.get("response", "")
    # Ollama chat: nested message.content
    if not text_value:
        msg = response.data.get("message")
        if isinstance(msg, dict):
            text_value = msg.get("content", "")
    # OpenAI / Claude chat: choices[0].message.content
    if not text_value and isinstance(response.data.get("choices"), list):
        choices = response.data["choices"]
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                text_value = (first.get("message") or {}).get("content", "")
    # Anthropic content blocks (list form)
    if not text_value:
        content_field = response.data.get("content")
        if isinstance(content_field, list):
            text_value = "".join(
                b.get("text", "") for b in content_field if isinstance(b, dict)
            )
        elif isinstance(content_field, str):
            text_value = content_field
    response.data["text"] = text_value or ""


class AIProcessNode(BaseNode):
    """Node for AI/ML processing tasks.
    
    Supports various AI operations:
    - Text generation
    - Sentiment analysis
    - Classification
    - Summarization
    - Entity extraction
    - Embeddings generation
    - Translation
    - Q&A
    """
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the AI process node. Returns output dict for BaseNode.run() to wrap."""
        # Get AI processing parameters
        ai_task = self.config.parameters.get("ai_task", "generate")
        # UI saves as "adapter", setup scripts use "adapter_id", industry
        # templates use "adapter_type" — accept all three
        adapter_id = (
            self.config.parameters.get("adapter_id")
            or self.config.parameters.get("adapter")
            or self.config.parameters.get("adapter_type")
        )

        if not adapter_id:
            # Try org-level LLM settings before falling back to system default
            adapter_id = await self._resolve_org_adapter()

        if not adapter_id:
            # No org setting — use system default LLM
            adapter_id = await self._auto_select_adapter(ai_task)

        # Trial restriction: override node-specific adapter to system default
        adapter_id = await self._enforce_trial_restriction(adapter_id)

        # Stash so _process_* methods can decide whether narrative routing
        # (model upgrade) applies. Only Ollama-family adapters get upgraded —
        # see _maybe_upgrade_model_for_narrative.
        self._current_adapter_id = adapter_id

        # Check LLM usage limits before calling (Business+ only)
        await self._check_llm_limits()

        # Get AI adapter class from registry
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"AI adapter {adapter_id} not found")

        # Look up UI-configured credentials for this adapter, scoped to the
        # executing org (B2 tiered model): tenant key -> shared/free-tier key
        # -> adapter env fallback. The tenant-aware getter matches a NULL
        # (shared) row explicitly and never returns another tenant's row, so
        # one org's LLM key can't leak to another. context["tenant_id"] is
        # threaded into every node by the workflow runtime.
        credentials = await get_adapter_credentials_for_tenant(
            adapter_id, context.get("tenant_id")
        ) or {}

        # Create adapter instance with proper config + credentials.
        # Propagate the node's "timeout" so long generations (e.g. large
        # parse/synthesis prompts) aren't killed by the adapter's short
        # default. Defaults to 300s — the lenient value adapters already
        # intend for local models — instead of the 60s AdapterConfig default.
        # Clamp to a 1800s ceiling: many auto-generated system templates carry
        # garbage timeout values (e.g. 19800s ≈ 5.5h); without a cap a stalled
        # LLM call would hang the workflow for hours. 1800s is generous for the
        # heaviest legitimate parse on a slow local model while bounding the tail.
        try:
            node_timeout = int(self.config.parameters.get("timeout", 300))
        except (TypeError, ValueError):
            node_timeout = 300
        node_timeout = max(1, min(node_timeout, 1800))
        adapter_config = AdapterConfig(
            name=adapter_id,
            category=AdapterCategory.AI,
            version="1.0.0",
            description=f"AI adapter for {ai_task}",
            api_key=credentials.get("api_key"),
            credentials=credentials,
            timeout_seconds=node_timeout,
        )
        adapter = adapter_class(adapter_config)
        try:
            await adapter.start()
        except Exception as start_err:
            # Adapter may already be registered from a prior execution — that's OK
            if "already registered" in str(start_err):
                await adapter.initialize()
                adapter.status = AdapterStatus.READY
                adapter._initialized = True
            else:
                raise

        # Build template context: input_data keys available both at top level
        # and under "input_data" prefix so {{key}} and {{input_data.key}} work.
        tmpl_ctx = {"input_data": input_data, **input_data}

        # Resolve templates in parameters that may contain {{...}} placeholders
        resolved_params = resolve_templates(dict(self.config.parameters), tmpl_ctx)
        # Patch self.config.parameters so downstream helpers see resolved values
        self.config.parameters.update(resolved_params)

        # Track wall time across the LLM call so the audit row + metering
        # can record response_time. Started just before the adapter dispatch
        # so we measure the model's latency, not setup overhead.
        _llm_started_at = time.monotonic()
        _llm_status = "success"
        _llm_error: Optional[str] = None

        # Process based on task type
        if ai_task == "generate":
            output_data = await self._process_generation(adapter, input_data)
        elif ai_task == "sentiment":
            output_data = await self._process_sentiment(adapter, input_data)
        elif ai_task == "classify":
            output_data = await self._process_classification(adapter, input_data)
        elif ai_task == "summarize":
            output_data = await self._process_summarization(adapter, input_data)
        elif ai_task == "extract_entities":
            output_data = await self._process_entity_extraction(adapter, input_data)
        elif ai_task == "embeddings":
            output_data = await self._process_embeddings(adapter, input_data)
        elif ai_task == "translate":
            output_data = await self._process_translation(adapter, input_data)
        elif ai_task == "qa":
            output_data = await self._process_qa(adapter, input_data)
        elif ai_task == "custom":
            output_data = await self._process_custom(adapter, input_data)
        else:
            raise ValueError(f"Unsupported AI task: {ai_task}")

        logger.debug(f"AI node {self.config.id} output keys: {list(output_data.keys())}, task={ai_task}")

        # Track LLM usage for metering (Business+ only)
        await self._track_llm_call(adapter_id, ai_task, output_data)

        # AI Governance audit log entry — every LLM invocation gets a row in
        # ai_audit_logs so /ai-governance?tab=audit-logs reflects what
        # actually ran. Business-only; no-ops when Business model isn't
        # importable (Community-standalone edition).
        await self._audit_llm_call(
            adapter_id=adapter_id,
            ai_task=ai_task,
            output_data=output_data,
            context=context,
            response_time_s=time.monotonic() - _llm_started_at,
            status=_llm_status,
            error_message=_llm_error,
        )

        # Publish completion event
        await event_bus.publish(
            "node.executed",
            {
                "node_id": self.config.id,
                "node_type": "aiProcess",
                "ai_task": ai_task,
                "adapter_id": adapter_id
            }
        )

        return output_data
    
    async def _resolve_org_adapter(self) -> Optional[str]:
        """Check org-level LLM settings for a preferred adapter.

        Returns adapter registry key if the org has configured a preferred
        provider and it's available, otherwise None.
        """
        try:
            from core.tenant_context import get_current_tenant_id
            from core.database import get_session_maker
            from llm.org_llm_settings import get_org_llm_settings

            tenant_id = get_current_tenant_id()
            async with get_session_maker()() as db:
                settings = await get_org_llm_settings(tenant_id, db)

            if not settings or not settings.preferred_provider:
                return None

            # In trial mode, skip org settings — use system default
            if settings.trial_mode and not settings.has_own_key():
                return None

            # Map provider to adapter key
            provider_to_adapter = {
                "ollama": ["ollama"],
                "vertex_ai": ["llm-service", "vertex_ai", "vertex-ai", "gemini"],
                "gemini": ["llm-service", "gemini", "vertex_ai"],
                "openai": ["openai"],
                "anthropic": ["claude", "anthropic"],
                "deepseek": ["deepseek"],
            }
            available = list(adapter_registry._adapter_classes.keys())
            candidates = provider_to_adapter.get(
                settings.preferred_provider,
                [settings.preferred_provider, "llm-service"]
            )
            for c in candidates:
                if c in available:
                    logger.info(f"Using org-level adapter '{c}' (provider: {settings.preferred_provider})")
                    return c

            return None
        except Exception as e:
            logger.debug(f"Org adapter resolution failed (using system default): {e}")
            return None

    async def _enforce_trial_restriction(self, adapter_id: str) -> str:
        """In trial mode, override node-specific adapters to system default.

        Trial tenants use the system default LLM (Bodaty pays). Node-level
        adapter overrides (e.g. adapter_type: "openai") are ignored to prevent
        cost leakage. After BYOK, node-level selection is honored.
        """
        try:
            from core.tenant_context import get_current_tenant_id
            from core.database import get_session_maker
            from llm.org_llm_settings import get_org_llm_settings

            tenant_id = get_current_tenant_id()
            async with get_session_maker()() as db:
                settings = await get_org_llm_settings(tenant_id, db)

            if not settings or not settings.trial_mode:
                return adapter_id  # Not in trial — honor node selection

            if settings.has_own_key():
                return adapter_id  # Has own key — honor node selection

            # In trial without own key — force system default
            system_adapter = await self._auto_select_adapter("generate")
            if adapter_id != system_adapter:
                logger.info(
                    f"Trial restriction: overriding node adapter '{adapter_id}' "
                    f"with system default '{system_adapter}'"
                )
                return system_adapter
            return adapter_id
        except Exception as e:
            logger.debug(f"Trial restriction check skipped: {e}")
            return adapter_id

    async def _check_llm_limits(self) -> None:
        """Check LLM usage limits before making a call.

        Only enforced when Business edition metering is available.
        Community edition skips this check silently.
        """
        try:
            from core.tenant_context import get_current_tenant_id
            from core.database import get_session_maker
            from aictrlnet_business.services.usage_metering import (
                UsageMeteringService, UsageLimitEnforcer
            )

            tenant_id = get_current_tenant_id()
            async with get_session_maker()() as db:
                metering = UsageMeteringService(db)
                enforcer = UsageLimitEnforcer(metering)

                allowed = await enforcer.check_and_enforce(tenant_id, "llm_calls")
                if not allowed:
                    raise ValueError(
                        "LLM call limit exceeded for this billing period. "
                        "Configure your own API key in Settings > AI Model Configuration "
                        "to continue using AI features."
                    )

                # Emit 80% warning
                limit_check = await metering.check_usage_limits(tenant_id, "llm_calls")
                pct = limit_check.get("percentage_used", 0)
                if pct >= 80:
                    await event_bus.publish(
                        "llm.usage_warning",
                        {
                            "tenant_id": tenant_id,
                            "percentage_used": pct,
                            "current_usage": limit_check.get("current_usage", 0),
                            "limit": limit_check.get("limit", 0),
                            "message": (
                                f"You've used {pct:.0f}% of your trial AI calls. "
                                "Configure your own API key in settings to keep "
                                "workflows running."
                            ),
                        }
                    )
        except ImportError:
            pass  # Community edition — no metering
        except ValueError:
            raise  # Re-raise limit exceeded
        except Exception as e:
            logger.debug(f"LLM limit check skipped: {e}")

    async def _track_llm_call(
        self, adapter_id: str, ai_task: str, output_data: Dict[str, Any]
    ) -> None:
        """Track an LLM call for usage metering.

        Tracks both aggregate llm_calls and per-provider usage
        (e.g. llm_ollama, llm_openai). Only when Business edition is available.
        """
        try:
            from core.tenant_context import get_current_tenant_id
            from core.database import get_session_maker
            from aictrlnet_business.services.usage_metering import UsageMeteringService

            tenant_id = get_current_tenant_id()
            usage = output_data.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            model_name = output_data.get("model", "")

            async with get_session_maker()() as db:
                metering = UsageMeteringService(db)

                # 1. Aggregate LLM call count
                await metering.track_usage(tenant_id, "llm_calls", 1)

                # 2. Per-provider tracking (e.g. llm_ollama, llm_openai)
                provider_key = f"llm_{adapter_id}"
                await metering.track_usage(tenant_id, provider_key, 1)

                # 3. Token tracking per provider
                if total_tokens:
                    token_key = f"llm_tokens_{adapter_id}"
                    await metering.track_usage(tenant_id, token_key, total_tokens)

                # 4. Billable event with full metadata
                await metering.record_billable_event(
                    tenant_id,
                    "llm_call",
                    {
                        "adapter_id": adapter_id,
                        "ai_task": ai_task,
                        "node_id": self.config.id,
                        "model": model_name,
                        "tokens": total_tokens,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                    }
                )
        except ImportError:
            pass  # Community edition — no metering
        except Exception as e:
            logger.debug(f"LLM usage tracking skipped: {e}")

    async def _audit_llm_call(
        self,
        *,
        adapter_id: str,
        ai_task: str,
        output_data: Dict[str, Any],
        context: Dict[str, Any],
        response_time_s: float,
        status: str,
        error_message: Optional[str],
    ) -> None:
        """Write one ai_audit_logs row per LLM invocation.

        Powers the Audit Logs tab on /ai-governance. Business-only —
        AIAuditLog lives in aictrlnet_business; this no-ops when the
        model isn't importable (Community-standalone edition) or when
        the write fails. Never raises into the workflow.
        """
        try:
            from core.database import get_session_maker
            from aictrlnet_business.models.ai_governance_complete import AIAuditLog

            usage = output_data.get("usage") or {}
            tokens_used = usage.get("total_tokens") or 0
            model_name = output_data.get("model") or adapter_id

            row = AIAuditLog(
                action="model_invoked",
                model_id=str(model_name) if model_name else None,
                user_id=context.get("user_id"),
                workflow_id=context.get("workflow_id"),
                task_id=self.config.id,
                tokens_used=tokens_used or None,
                response_time=response_time_s,
                status=status,
                error_message=error_message,
                details={
                    "ai_task": ai_task,
                    "adapter_id": adapter_id,
                    "node_name": self.config.name,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
            )

            async with get_session_maker()() as db:
                db.add(row)
                await db.commit()
        except ImportError:
            pass  # Community edition — no AI audit logging
        except Exception as e:
            logger.debug(f"AI audit log write skipped: {e}")

    async def _auto_select_adapter(self, ai_task: str) -> str:
        """Auto-select adapter using the system default LLM provider.

        Resolution order:
        1. Map DEFAULT_LLM_MODEL to its provider (ollama, vertex_ai, etc.)
        2. Find a matching adapter in the registry
        3. Fall back to llm-service (bridges to the LLM service which has all providers)
        4. Fall back to any available AI adapter
        """
        from llm.tier_resolver import get_environment_default_provider

        available = list(adapter_registry._adapter_classes.keys())
        if not available:
            raise ValueError("No AI adapters available")

        # Map provider string to adapter registry keys (handles naming variants)
        provider_to_adapter = {
            "ollama": ["ollama"],
            "vertex_ai": ["llm-service", "vertex_ai", "vertex-ai", "gemini"],
            "gemini": ["llm-service", "gemini", "google-gemini", "vertex_ai"],
            "openai": ["openai"],
            "anthropic": ["claude", "anthropic"],
            "deepseek": ["deepseek"],
            "dashscope": ["dashscope"],
        }

        provider = get_environment_default_provider()
        candidates = provider_to_adapter.get(provider, [provider, "llm-service"])

        for candidate in candidates:
            if candidate in available:
                logger.info(f"Auto-selected adapter '{candidate}' (system default provider: {provider})")
                return candidate

        # Fallback: cost-preference order
        for fb in ["ollama", "llm-service", "openai", "claude", "deepseek", "huggingface"]:
            if fb in available:
                logger.info(f"Auto-selected fallback adapter '{fb}'")
                return fb

        return available[0]
    
    async def _call_adapter(self, adapter: Any, capability: str, parameters: dict) -> "AdapterResponse":
        """Call adapter with fallback from generic capabilities to chat."""
        from adapters.models import AdapterRequest
        from llm.tier_resolver import get_environment_default_model

        # Ensure a model is always set — adapters like Ollama require it
        if not parameters.get("model"):
            parameters["model"] = get_environment_default_model()

        request = AdapterRequest(capability=capability, parameters=parameters)
        try:
            response = await adapter.execute(request)
            if response.status != "error":
                _normalize_response_text(response)
                return response
        except Exception:
            pass

        # Fallback: convert to chat (works with Ollama, Claude, OpenAI, etc.)
        prompt = parameters.get("prompt") or parameters.get("text", "")
        messages = [{"role": "user", "content": prompt}]
        chat_request = AdapterRequest(
            capability="chat",
            parameters={
                "messages": messages,
                "model": parameters.get("model"),
                "max_tokens": parameters.get("max_tokens", 1000),
                "temperature": parameters.get("temperature", 0.7),
            }
        )
        response = await adapter.execute(chat_request)
        if response.status != "error":
            _normalize_response_text(response)
        return response

    async def _build_agent_persona_block(self) -> Optional[str]:
        """Return a persona-context block when this node references an agent.

        Looks at NodeConfig.parameters for agent_id (UUID) or agent (name string).
        Returns a short structured block built from EnhancedAgent.identity +
        behavioral_rules, or None if no agent is referenced or Business models
        aren't available.
        """
        params = self.config.parameters or {}
        agent_id = params.get("agent_id")
        agent_name = params.get("agent")
        if not agent_id and not agent_name:
            return None

        # Cross-edition import — Business model. Silently skip if unavailable.
        try:
            import sys
            if '/workspace/editions/business/src' not in sys.path:
                sys.path.insert(0, '/workspace/editions/business/src')
            from aictrlnet_business.models.enhanced_agent import EnhancedAgent
            from core.database import get_db
            from sqlalchemy import select as _select
        except Exception:
            return None

        try:
            async for db in get_db():
                stmt = _select(EnhancedAgent)
                if agent_id:
                    stmt = stmt.where(EnhancedAgent.id == agent_id)
                else:
                    stmt = stmt.where(EnhancedAgent.name == agent_name)
                result = await db.execute(stmt)
                agent = result.unique().scalar_one_or_none()
                break
        except Exception as e:
            logger.debug(f"Persona lookup failed for node {self.config.id}: {e}")
            return None

        if not agent:
            return None

        identity = agent.identity if isinstance(agent.identity, dict) else {}
        rules = agent.behavioral_rules if isinstance(agent.behavioral_rules, dict) else {}
        if not identity and not rules:
            return None

        lines = [f"## Active Agent: {agent.name}"]
        if identity.get("personality"):
            lines.append(f"- Personality: {identity['personality']}")
        if identity.get("voice"):
            lines.append(f"- Voice: {identity['voice']}")
        if identity.get("expertise_domain"):
            lines.append(f"- Expertise: {identity['expertise_domain']}")
        if identity.get("communication_style"):
            lines.append(f"- Communication style: {identity['communication_style']}")
        if rules.get("confirmation_required"):
            actions = ", ".join(str(a) for a in rules["confirmation_required"][:5])
            lines.append(f"- Requires confirmation for: {actions}")
        if rules.get("autonomous_actions"):
            actions = ", ".join(str(a) for a in rules["autonomous_actions"][:5])
            lines.append(f"- Can act autonomously on: {actions}")
        return "\n".join(lines) if len(lines) > 1 else None

    def _is_narrative_node(self, prompt: str) -> bool:
        """Detect narrative-style aiProcess nodes by keyword in prompt or name.

        Narrative nodes are where small-model arithmetic drift hits hardest —
        they receive precomputed numbers and just have to re-state them in
        prose. Triggering on these keywords applies the verbatim-copy
        guardrail and (for Ollama) upgrades the model to QUALITY tier.

        Keywords are specific to prose-writing intent; bare "summary"/
        "report" are excluded because they appear as JSON output keys in
        non-narrative templates.
        """
        haystack = (str(prompt or "") + " " + str(self.config.name or "")).lower()
        return any(k in haystack for k in _NARRATIVE_KEYWORDS)

    def _apply_narrative_guardrail(self, prompt: str, input_data: Dict[str, Any]) -> str:
        """Prepend a verbatim-copy guardrail block to a narrative prompt.

        Reads narrative_with_numbers.md via the shared PromptTemplateLoader
        and injects the upstream input dict as the precomputed_numbers block.
        Caps the JSON blob at 4000 chars to avoid blowing the context window
        on unexpectedly large upstream outputs.
        """
        try:
            import json as _json
            numbers_blob = _json.dumps(input_data, default=str, indent=2)[:4000]
            guardrail = _get_prompt_loader().get_section(
                "narrative_with_numbers",
                {"precomputed_numbers": numbers_blob},
            )
        except Exception as exc:
            logger.warning("Narrative guardrail render failed: %s", exc)
            return prompt
        if not guardrail:
            return prompt
        return f"{guardrail}\n\n{prompt}"

    def _maybe_upgrade_model_for_narrative(
        self, prompt: str, parameters: Dict[str, Any]
    ) -> None:
        """In-place: route narrative nodes to QUALITY tier when no model is set.

        Triggers ONLY when:
          - The resolved adapter is Ollama-family (avoids injecting an Ollama
            model tag into an OpenAI/Claude request, which would 400-error).
          - The node prompt looks narrative.
          - The user has NOT explicitly chosen a model. Per the "by default"
            spec, an explicit `model` in node config wins; we don't second-
            guess production deployments that pin a small model for cost or
            speed reasons.
        """
        adapter_id = getattr(self, "_current_adapter_id", None)
        if adapter_id not in _OLLAMA_ADAPTER_IDS:
            return
        if not self._is_narrative_node(prompt):
            return
        if parameters.get("model"):
            # Explicit pin — leave as-is.
            return
        try:
            from llm.tier_resolver import ModelTier, get_system_default_for_tier
            upgraded = get_system_default_for_tier(ModelTier.QUALITY)
        except Exception as exc:
            logger.warning("Tier-resolver lookup failed; using default: %s", exc)
            return
        if not upgraded:
            return
        logger.info(
            "Narrative node %s — defaulting Ollama model to QUALITY tier (%s)",
            self.config.id,
            upgraded,
        )
        parameters["model"] = upgraded

    async def _process_generation(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text generation task."""
        prompt = (
            input_data.get("prompt")
            or self.config.parameters.get("prompt")
            or self.config.parameters.get("description")
        )
        if not prompt:
            # Synthesize a prompt from the node name and any analysis/agent fields
            # in the config. Workflow templates store rich descriptions in these.
            parts = []
            if self.config.name and self.config.name != self.config.id:
                parts.append(f"Perform the following task: {self.config.name}.")
            agent = self.config.parameters.get("agent")
            if agent:
                parts.append(f"You are the {agent}.")
            analysis = self.config.parameters.get("analysis")
            if analysis and isinstance(analysis, list):
                parts.append(f"Analyze: {', '.join(analysis)}.")
            fields = self.config.parameters.get("required_fields")
            if fields and isinstance(fields, list):
                parts.append(f"Return results for: {', '.join(fields)}.")
            if input_data:
                parts.append(f"Input data: {str(input_data)[:500]}")
            prompt = " ".join(parts) if parts else f"Process node: {self.config.name or self.config.id}"

        # If this node references a named agent, prepend its persona block so the
        # LLM call carries the agent's identity. Best-effort: silently skips when
        # Business edition is not present or no matching agent is registered.
        persona_block = await self._build_agent_persona_block()
        if persona_block:
            prompt = f"{persona_block}\n\n{prompt}"

        # Narrative-style nodes (executive_summary, by-fund commentary, board
        # narrative, etc.) get a verbatim-copy guardrail block prepended so
        # the LLM is told NOT to recompute or invent numbers. The numbers
        # come from upstream deterministic compute.
        if self._is_narrative_node(prompt):
            prompt = self._apply_narrative_guardrail(prompt, input_data)

        call_params = {
            "prompt": prompt,
            "max_tokens": self.config.parameters.get("max_tokens", 1000),
            "temperature": self.config.parameters.get("temperature", 0.7),
            "top_p": self.config.parameters.get("top_p", 1.0),
            "model": self.config.parameters.get("model"),
        }
        # For Ollama-family adapters, narrative prompts run on QUALITY tier
        # (llama3.1:8b-instruct-q4_K_M) instead of the small balanced
        # defaults that drift on dollar amounts.
        self._maybe_upgrade_model_for_narrative(prompt, call_params)

        response = await self._call_adapter(adapter, "generate", call_params)

        if response.status == "error":
            raise Exception(f"Generation failed: {response.error}")

        gen_text = response.data.get("text", "")
        result = {
            "generated_text": gen_text,
            "usage": response.data.get("usage", {}),
            "model": response.data.get("model"),
        }
        # When the LLM emits JSON (most agent prompts do), also expose a
        # parsed structured form so downstream consumers (approvals UI,
        # workflow nodes) don't have to repair LLM-quirky JSON each time.
        # parsed_output is None when the text isn't JSON-shaped.
        parsed = _try_parse_llm_json(gen_text)
        if parsed is not None:
            result["parsed_output"] = parsed
            # Surface the parsed top-level keys directly on the result so that, with the
            # engine's {**input_data, **output_data} accumulation, downstream templates can
            # address them as {{proposal.subject}} / {{ap_entry.memo}} / {{classification.intent}}
            # rather than {{parsed_output.proposal.subject}}. This is what the starter
            # adapter `params`, `parameter_mapping`, and approval `description_template`
            # all assume. Never clobber the reserved result keys.
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    if key not in result:
                        result[key] = value
        return result
    
    async def _process_sentiment(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sentiment analysis task."""
        text = input_data.get("text") or self.config.parameters.get("text")
        if not text:
            raise ValueError("Text is required for sentiment analysis")

        response = await self._call_adapter(adapter, "generate", {
            "prompt": f"Analyze the sentiment of the following text and respond with only 'positive', 'negative', or 'neutral':\n\n{text}",
            "max_tokens": 10,
            "temperature": 0,
            "model": self.config.parameters.get("model"),
        })

        if response.status == "error":
            raise Exception(f"Sentiment analysis failed: {response.error}")

        sentiment = response.data.get("text", "").strip().lower()
        return {
            "sentiment": sentiment,
            "confidence": 0.8
        }

        # Extract classification result (unreachable but kept for native support)
        classifications = response.data.get("classifications", [])
        if classifications:
            top_class = max(classifications, key=lambda x: x.get("confidence", 0))
            return {
                "sentiment": top_class.get("label"),
                "confidence": top_class.get("confidence"),
                "scores": {c["label"]: c["confidence"] for c in classifications}
            }
        
        return {"sentiment": "neutral", "confidence": 0.5}
    
    async def _process_classification(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text classification task."""
        # Get text and categories
        text = input_data.get("text") or self.config.parameters.get("text")
        categories = input_data.get("categories") or self.config.parameters.get("categories")

        if not text:
            # Fall back to the resolved prompt as the text to classify
            prompt = self.config.parameters.get("prompt")
            if prompt:
                text = prompt
                logger.info("Classification: using resolved prompt as text input")
            else:
                raise ValueError("Text is required for classification")
        if not categories:
            raise ValueError("Categories are required for classification")
        
        # Build request
        request = AdapterRequest(
            capability="classify",
            parameters={
                "text": text,
                "categories": categories,
                "multi_label": self.config.parameters.get("multi_label", False),
                "model": self.config.parameters.get("model")
            }
        )
        
        # Most adapters don't support native "classify" — go straight to
        # generation-based classification via _call_adapter (which falls back
        # to chat_completion automatically).
        categories_str = ", ".join(categories)
        response = await self._call_adapter(adapter, "generate", {
            "prompt": f"Classify the following text into one of these categories [{categories_str}]. Respond with only the category name:\n\n{text}",
            "max_tokens": 50,
            "temperature": 0,
            "model": self.config.parameters.get("model"),
        })

        if response.status == "error":
            raise Exception(f"Classification failed: {response.error}")

        category = response.data.get("text", "").strip()
        if category:
            return {
                "category": category,
                "confidence": 0.8,
                "multi_label": False
            }
        
        # Extract classification results
        classifications = response.data.get("classifications", [])
        if self.config.parameters.get("multi_label", False):
            # Return all classifications above threshold
            threshold = self.config.parameters.get("threshold", 0.5)
            selected = [c for c in classifications if c.get("confidence", 0) >= threshold]
            return {
                "categories": [c["label"] for c in selected],
                "scores": {c["label"]: c["confidence"] for c in classifications},
                "multi_label": True
            }
        else:
            # Return top classification
            if classifications:
                top_class = max(classifications, key=lambda x: x.get("confidence", 0))
                return {
                    "category": top_class.get("label"),
                    "confidence": top_class.get("confidence"),
                    "scores": {c["label"]: c["confidence"] for c in classifications},
                    "multi_label": False
                }
        
        return {"category": None, "confidence": 0, "multi_label": False}
    
    async def _process_summarization(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text summarization task."""
        # Get text to summarize
        text = input_data.get("text") or self.config.parameters.get("text")
        if not text:
            raise ValueError("Text is required for summarization")
        
        # Get summarization parameters
        max_length = self.config.parameters.get("max_length", 200)
        style = self.config.parameters.get("style", "concise")  # concise, detailed, bullet_points
        
        # Build prompt based on style
        if style == "bullet_points":
            prompt = f"Summarize the following text in bullet points:\n\n{text}"
        elif style == "detailed":
            prompt = f"Provide a detailed summary of the following text:\n\n{text}"
        else:
            prompt = f"Provide a concise summary of the following text in no more than {max_length} words:\n\n{text}"
        
        # Build request
        request = AdapterRequest(
            capability="summarize",
            parameters={
                "text": text,
                "max_length": max_length,
                "model": self.config.parameters.get("model")
            }
        )
        
        response = await self._call_adapter(adapter, "generate", {
            "prompt": prompt,
            "max_tokens": max_length * 2,
            "temperature": 0.3,
            "model": self.config.parameters.get("model"),
        })

        if response.status == "error":
            raise Exception(f"Summarization failed: {response.error}")

        return {
            "summary": response.data.get("text", ""),
            "method": "generation"
        }
    
    async def _process_entity_extraction(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process entity extraction task."""
        # Get text to analyze
        text = input_data.get("text") or self.config.parameters.get("text")
        if not text:
            raise ValueError("Text is required for entity extraction")
        
        # Get entity types to extract
        entity_types = self.config.parameters.get("entity_types", 
            ["person", "organization", "location", "date", "email", "phone", "url"])
        
        # Build prompt for extraction
        prompt = f"""Extract the following types of entities from the text: {', '.join(entity_types)}.
        
Format the response as JSON with entity type as keys and lists of found entities as values.
        
Text: {text}"""
        
        # Build request
        request = AdapterRequest(
            capability="generate",
            parameters={
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0,
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Entity extraction failed: {response.error}")
        
        # Try to parse JSON response
        import json
        result_text = response.data.get("text", "")
        try:
            # Extract JSON from response
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                entities = json.loads(result_text[start:end])
            else:
                entities = {}
        except:
            # Fallback to empty entities
            entities = {entity_type: [] for entity_type in entity_types}
        
        return {
            "entities": entities,
            "entity_count": sum(len(v) for v in entities.values())
        }
    
    async def _process_embeddings(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process embeddings generation task."""
        # Get text(s) to embed
        text = input_data.get("text") or self.config.parameters.get("text")
        texts = input_data.get("texts") or self.config.parameters.get("texts")
        
        if not text and not texts:
            raise ValueError("Text or texts are required for embeddings generation")
        
        # Prepare input list
        if texts:
            input_texts = texts if isinstance(texts, list) else [texts]
        else:
            input_texts = [text]
        
        # Build request
        request = AdapterRequest(
            capability="embeddings",
            parameters={
                "texts": input_texts,
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Embeddings generation failed: {response.error}")
        
        embeddings = response.data.get("embeddings", [])
        
        return {
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings)
        }
    
    async def _process_translation(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text translation task."""
        # Get text to translate
        text = input_data.get("text") or self.config.parameters.get("text")
        if not text:
            raise ValueError("Text is required for translation")
        
        # Get languages
        source_lang = self.config.parameters.get("source_language", "auto")
        target_lang = self.config.parameters.get("target_language", "en")
        
        # Build prompt for translation
        if source_lang == "auto":
            prompt = f"Translate the following text to {target_lang}:\n\n{text}"
        else:
            prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}"
        
        # Build request
        request = AdapterRequest(
            capability="generate",
            parameters={
                "prompt": prompt,
                "max_tokens": len(text) * 2,  # Rough estimate
                "temperature": 0.3,
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Translation failed: {response.error}")
        
        return {
            "translated_text": response.data.get("text", ""),
            "source_language": source_lang,
            "target_language": target_lang
        }
    
    async def _process_qa(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process question answering task."""
        # Get question and context
        question = input_data.get("question") or self.config.parameters.get("question")
        context = input_data.get("context") or self.config.parameters.get("context")
        
        if not question:
            raise ValueError("Question is required for Q&A task")
        
        # Build prompt
        if context:
            prompt = f"""Based on the following context, answer the question.
            
Context: {context}
            
Question: {question}
            
Answer:"""
        else:
            prompt = question
        
        # Build request
        request = AdapterRequest(
            capability="generate",
            parameters={
                "prompt": prompt,
                "max_tokens": self.config.parameters.get("max_answer_length", 500),
                "temperature": 0.3,
                "model": self.config.parameters.get("model")
            }
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Q&A failed: {response.error}")
        
        return {
            "answer": response.data.get("text", ""),
            "question": question,
            "has_context": context is not None
        }
    
    async def _process_custom(self, adapter: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process custom AI task."""
        # Get custom capability and parameters
        capability = self.config.parameters.get("capability", "generate")
        request_params = self.config.parameters.get("request_parameters", {})
        
        # Merge with input data
        for key, value in input_data.items():
            if key not in request_params:
                request_params[key] = value
        
        # Build request
        request = AdapterRequest(
            capability=capability,
            parameters=request_params
        )
        
        # Execute request
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Custom AI task failed: {response.error}")
        
        return response.data
    
    def validate_config(self) -> bool:
        """Validate node configuration."""
        ai_task = self.config.parameters.get("ai_task", "generate")
        
        # Validate task-specific requirements
        if ai_task in ["generate", "sentiment", "classify", "summarize", "extract_entities", "translate", "qa"]:
            # These tasks need either input text or configured text
            pass  # Will be validated at runtime
        elif ai_task == "embeddings":
            # Needs text or texts
            pass  # Will be validated at runtime
        elif ai_task == "custom":
            if not self.config.parameters.get("capability"):
                raise ValueError("capability parameter is required for custom AI task")
        
        return True