"""MCP tool executor — delegates tool calls to existing AICtrlNet services.

Execution pipeline (see docs/architecture/MCP_SERVER_ARCHITECTURE_SPEC.md):

1. Plan gate       — ``plan_gate.enforce_plan`` — raises ``PlanError``
2. Scope check     — new-taxonomy + legacy compatibility expansion
3. Compliance gate — Enterprise-only ``MCPComplianceService.enforce_compliance``
4. Rate bucket     — Redis per-(principal, tool, window); raises ``RateError``
5. With-metering   — idempotency lookup, atomic quota charge, timeout,
                     handler execution, refund on ``RefundableError``,
                     idempotency store
6. Audit log       — Enterprise-only ``MCPComplianceService.audit_mcp_operation``
7. Metrics emit    — ``observability.record_invocation``

Each handler receives ``(arguments, db, user_id)`` and returns a plain
dict. The protocol layer wraps the result in MCP content format.
"""

import logging
import time
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from . import observability
from .metering import (
    QuotaError,
    RefundableError,
    ToolTimeoutError,
    with_metering,
)
from .plan_gate import (
    PLAN_MUTATING_TOOLS,
    PlanError,
    PlanService,
    enforce_plan,
)
from .rate_bucket import RateError, check_rate
from .scopes import scopes_satisfy
from .tools import TOOL_SCOPES

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    pass


class ScopeError(Exception):
    pass


class ComplianceError(Exception):
    pass


# Re-export pipeline errors for the protocol layer
__all__ = [
    "ComplianceError",
    "PlanError",
    "QuotaError",
    "RateError",
    "ScopeError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "execute_tool",
]


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    db: AsyncSession,
    user_id: str,
    api_key: Optional[Any] = None,
    tenant_id: Optional[str] = None,
    plan_service: Optional[PlanService] = None,
) -> Dict[str, Any]:
    """Execute an MCP tool through the full pipeline."""

    # One PlanService per HTTP request (passed in from protocol.py).
    # If not supplied, create a per-call one (safe but loses batch caching).
    plan_service = plan_service or PlanService(db)
    start = time.monotonic()
    status = "unknown"
    plan_tier = "unknown"

    try:
        # 1. Plan gate
        try:
            plan_tier = await enforce_plan(tool_name, tenant_id, plan_service)
        except PlanError as e:
            observability.record_plan_denied(
                tool=tool_name,
                current_plan=e.current,
                required_plan=e.required,
                tenant_id=tenant_id,
            )
            status = "plan_denied"
            raise

        # 2. Scope check (API-key auth only; JWT users bypass — full access)
        if api_key is not None:
            required = TOOL_SCOPES.get(tool_name, [])
            granted = getattr(api_key, "scopes", []) or []
            if required and not scopes_satisfy(granted, required):
                missing = [s for s in required if not scopes_satisfy(granted, [s])]
                observability.record_scope_denied(
                    tool=tool_name,
                    auth_type="api_key",
                    missing=missing,
                    tenant_id=tenant_id,
                )
                status = "scope_denied"
                raise ScopeError(
                    f"API key missing required scope(s) for {tool_name}: {missing}"
                )

        # 3. Enterprise compliance check
        await _enforce_compliance_if_enterprise(tool_name, tenant_id, db, plan_tier=plan_tier)

        # 4. Per-tool rate bucket (Redis / in-memory fallback)
        try:
            await check_rate(
                tool_name=tool_name,
                api_key=api_key,
                user_id=user_id,
                tenant_id=tenant_id,  # A10: tenant-scoped principal
            )
        except RateError as e:
            observability.record_rate_denied(
                tool=tool_name,
                window=e.window,
                limit=e.limit,
                tenant_id=tenant_id,
            )
            status = "rate_limited"
            raise

        # 5. With-metering (idempotency + atomic quota + timeout + refund)
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            raise ToolExecutionError(f"Unknown tool: {tool_name}")

        async def run_handler():
            return await handler(arguments, db, user_id)

        try:
            result = await with_metering(
                tool_name=tool_name,
                args=arguments,
                tenant_id=tenant_id or "default",
                edition=plan_tier,
                db=db,
                handler=run_handler,
            )
        except QuotaError as e:
            observability.record_quota_denied(
                tool=tool_name,
                meter=e.meter,
                limit=e.limit,
                used=e.used,
                tenant_id=tenant_id,
            )
            status = "quota_exceeded"
            raise
        except ToolTimeoutError:
            status = "timeout"
            raise
        except (RefundableError, ToolExecutionError, ScopeError, ComplianceError):
            status = "handler_error"
            raise
        except Exception as e:
            status = "handler_error"
            raise ToolExecutionError(str(e)) from e

        # Bust plan cache if this tool may have changed the plan/subscription
        if tool_name in PLAN_MUTATING_TOOLS:
            plan_service.bust_cache(tenant_id)

        status = "success"
        return result

    finally:
        duration = time.monotonic() - start
        # 7. Metrics — always emit, regardless of outcome
        observability.record_invocation(
            tool=tool_name,
            status=status,
            plan=plan_tier,
            duration_seconds=duration,
            tenant_id=tenant_id,
        )
        # 6. Audit — always log for Enterprise tenants, even on failure
        await _audit_if_enterprise(
            tool_name,
            arguments,
            None if status != "success" else "ok",
            user_id,
            tenant_id,
            duration * 1000,
            status,
            db,
            plan_tier=plan_tier,
        )


async def _enforce_compliance_if_enterprise(
    tool_name: str,
    tenant_id: Optional[str],
    db: AsyncSession,
    plan_tier: str = "community",
) -> None:
    """Enforce Enterprise compliance gate before tool execution.

    Wave 7 A2: **fail-secure for Enterprise plan tier**. If the
    compliance service is reachable but errors mid-check, previously
    we logged a warning and let the tool through — which is unsafe for
    SOC2/HIPAA-scoped deploys. Now: Enterprise-plan callers fail with
    ComplianceError on unexpected errors; Community/Business keep the
    warning-and-continue behavior (they don't expect compliance
    enforcement).

    Controlled by ``MCP_COMPLIANCE_REQUIRED_FOR_ENTERPRISE`` (default
    ``true``). Operators can flip to ``false`` for an intentional
    graceful-degradation window.
    """
    try:
        from aictrlnet_enterprise.services.mcp_compliance import MCPComplianceService  # type: ignore
    except ImportError:
        return
    try:
        svc = MCPComplianceService()
        compliant, reason = await svc.enforce_compliance(
            server_id="aictrlnet-mcp-transport",
            tenant_id=tenant_id or "default",
            capability=tool_name,
            db=db,
        )
        if not compliant:
            observability.record_compliance_denied(
                tool=tool_name,
                reason=reason or "compliance denied",
                tenant_id=tenant_id,
            )
            raise ComplianceError(reason or "Compliance check failed")
    except ComplianceError:
        raise
    except Exception as e:
        import os

        fail_secure = (
            plan_tier == "enterprise"
            and os.environ.get("MCP_COMPLIANCE_REQUIRED_FOR_ENTERPRISE", "true").lower() == "true"
        )
        if fail_secure:
            observability.record_compliance_denied(
                tool=tool_name,
                reason=f"compliance service error: {e}",
                tenant_id=tenant_id,
            )
            raise ComplianceError(
                f"Enterprise compliance service unavailable: {e}"
            ) from e
        logger.warning("Compliance check errored for %s: %s", tool_name, e)


async def _audit_if_enterprise(
    tool_name: str,
    request_data: dict,
    response_data: Any,
    user_id: str,
    tenant_id: Optional[str],
    duration_ms: float,
    status: str,
    db: AsyncSession,
    plan_tier: str = "community",
) -> None:
    """Persist the MCP tool call to the Enterprise audit log.

    Wave 7 A12: **fail-closed for Enterprise plan tier**. Previously
    audit failures were silently logged as warnings — unacceptable for
    SOC2/HIPAA-scoped deploys where the audit trail is a compliance
    requirement. Now: if the audit service errors for an Enterprise
    caller, the tool call surfaces as a ComplianceError so the caller
    knows the trail is incomplete and can decide what to do.

    Controlled by the same ``MCP_COMPLIANCE_REQUIRED_FOR_ENTERPRISE``
    flag as A2.
    """
    try:
        from aictrlnet_enterprise.services.mcp_compliance import MCPComplianceService  # type: ignore
    except ImportError:
        return
    try:
        svc = MCPComplianceService()
        await svc.audit_mcp_operation(
            tenant_id=tenant_id or "default",
            user_id=user_id,
            operation_type=f"mcp_tool:{tool_name}",
            operation_status=status,
            server_id="aictrlnet-mcp-transport",
            request_data=request_data,
            response_data=response_data,
            duration_ms=duration_ms,
            db=db,
        )
    except Exception as e:
        import os

        fail_secure = (
            plan_tier == "enterprise"
            and os.environ.get("MCP_COMPLIANCE_REQUIRED_FOR_ENTERPRISE", "true").lower() == "true"
        )
        if fail_secure:
            logger.error("Enterprise audit log failed — raising: %s", e)
            raise ComplianceError(
                f"Enterprise audit trail incomplete: {e}"
            ) from e
        logger.warning("Enterprise audit logging failed: %s", e)


# ---------------------------------------------------------------------------
# Tool handlers (unchanged from v1 — new-wave handlers will be appended
# as those waves land)
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402


async def _handle_create_workflow(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.nlp import NLPService

    svc = NLPService(db)
    result = await svc.process_natural_language(
        prompt=arguments["description"],
        context={"mode": "create", "user_id": user_id},
        user_id=user_id,
    )
    if "error" in result and result["error"]:
        raise ToolExecutionError(result["error"])

    workflow = result.get("workflow") or {}
    definition = workflow.get("definition") or {}
    nodes = definition.get("nodes") or []
    return {
        "workflow_id": workflow.get("id"),
        "name": workflow.get("name"),
        "node_count": len(nodes),
        "status": "created",
        "message": f"Created workflow '{workflow.get('name', 'Untitled')}' with {len(nodes)} nodes",
    }


async def _handle_list_workflows(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.workflow_service import WorkflowService

    svc = WorkflowService(db)
    workflows = await svc.list_workflows(
        limit=arguments.get("limit", 20),
        offset=arguments.get("offset", 0),
    )
    return {
        "workflows": [
            {
                "id": str(w.id),
                "name": w.name,
                "status": getattr(w, "status", "unknown"),
                "created_at": str(getattr(w, "created_at", "")),
            }
            for w in workflows
        ],
        "count": len(workflows),
    }


async def _handle_get_workflow(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.workflow_service import WorkflowService

    svc = WorkflowService(db)
    workflow = await svc.get_workflow(arguments["workflow_id"])
    if not workflow:
        raise ToolExecutionError(f"Workflow {arguments['workflow_id']} not found")

    definition = getattr(workflow, "definition", {}) or {}
    nodes = definition.get("nodes", [])
    return {
        "id": str(workflow.id),
        "name": workflow.name,
        "description": getattr(workflow, "description", ""),
        "status": getattr(workflow, "status", "unknown"),
        "node_count": len(nodes),
        "definition": definition,
        "created_at": str(getattr(workflow, "created_at", "")),
    }


async def _handle_execute_workflow(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.workflow_service import WorkflowService

    svc = WorkflowService(db)
    # Pass dry_run through. WorkflowService composes with
    # dry_run_interceptor.resolve_dry_run, which OR-merges request /
    # session / agent / pod scopes — a true value at any layer wins.
    dry_run = bool(arguments.get("dry_run", False))
    execute_kwargs = {
        "workflow_id": arguments["workflow_id"],
        "input_data": arguments.get("input_data"),
        "user_id": user_id,
    }
    try:
        import inspect
        if "dry_run" in inspect.signature(svc.execute_workflow).parameters:
            execute_kwargs["dry_run"] = dry_run
    except Exception:
        pass
    execution = await svc.execute_workflow(**execute_kwargs)
    return {
        "execution_id": str(execution.id),
        "status": getattr(execution, "status", "started"),
        "dry_run": dry_run,
        "dry_run_source": getattr(execution, "dry_run_source", "request" if dry_run else None),
        "message": f"Workflow execution started: {execution.id}",
    }


async def _handle_get_execution_status(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.workflow_execution import WorkflowExecutionService

    svc = WorkflowExecutionService(db)
    details = await svc.get_execution_details(arguments["execution_id"])
    if not details:
        raise ToolExecutionError(f"Execution {arguments['execution_id']} not found")
    return details


async def _handle_list_templates(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.workflow_template_service import create_workflow_template_service
    from schemas.workflow_templates import TemplateListRequest

    svc = create_workflow_template_service()
    request = TemplateListRequest(
        category=arguments.get("category"),
        limit=arguments.get("limit", 20),
        skip=0,
    )
    templates, total = await svc.list_templates(db=db, user_id=user_id, request=request)
    return {
        "templates": [
            {
                "id": str(t.id),
                "name": t.name,
                "description": getattr(t, "description", ""),
                "category": getattr(t, "category", ""),
            }
            for t in templates
        ],
        "total": total,
    }


async def _handle_assess_quality(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from mcp_server.services.quality import MCPQualityService

    svc = MCPQualityService(db)
    return await svc.assess_quality(
        content=arguments["content"],
        content_type=arguments.get("content_type", "text"),
        criteria=arguments.get("criteria"),
    )


async def _handle_send_message(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.enhanced_conversation_manager import EnhancedConversationService

    svc = EnhancedConversationService(db)
    active_sessions = await svc.get_active_sessions(user_id)
    if active_sessions:
        session = active_sessions[0]
    else:
        session = await svc.create_session(user_id)

    response = await svc.process_message(
        session_id=session.id,
        content=arguments["message"],
        user_id=user_id,
    )
    if hasattr(response, "dict"):
        return response.dict()
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return {"session_id": str(session.id), "response": str(response)}


async def _handle_evaluate_policy(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    import re
    import sys
    import time as _time

    if "/workspace/editions/business/src" not in sys.path:
        sys.path.insert(0, "/workspace/editions/business/src")
    if "/workspace/editions/community/src" not in sys.path:
        sys.path.insert(0, "/workspace/editions/community/src")

    from aictrlnet_business.services.agp_evaluation import AGPEvaluationService
    from core.tenant_context import get_current_tenant_id

    svc = AGPEvaluationService(db)
    tenant_id = get_current_tenant_id()
    policy = await svc.get_policy(arguments["policy_id"], tenant_id=tenant_id)
    if not policy:
        raise ToolExecutionError(f"Policy {arguments['policy_id']} not found")

    content = arguments["content"]
    rules_raw = _json.loads(policy.rules) if isinstance(policy.rules, str) else (policy.rules or [])
    rules = rules_raw if isinstance(rules_raw, list) else [rules_raw] if rules_raw else []

    start = _time.time()
    violations = []
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        conditions = rule.get("conditions", {})
        cond_type = conditions.get("type", "")
        cond_value = conditions.get("value", "")
        pattern = conditions.get("criteria", {}).get("pattern", "")
        matched = False

        if cond_type == "contains":
            targets = cond_value if isinstance(cond_value, list) else [cond_value]
            matched = any(t.lower() in content.lower() for t in targets if t)
        elif cond_type == "regex":
            try:
                matched = bool(re.search(cond_value, content))
            except re.error:
                pass
        elif cond_type == "content" and pattern:
            try:
                matched = bool(re.search(pattern, content))
            except re.error:
                pass

        if matched:
            action_info = rule.get("actions", {})
            violations.append({
                "rule_id": rule.get("id", "unknown"),
                "rule_name": action_info.get("params", {}).get("reason", "Unnamed rule"),
                "severity": rule.get("metadata", {}).get("severity", "medium"),
                "action": action_info.get("type", "block"),
            })

    elapsed = (_time.time() - start) * 1000
    return {
        "passed": len(violations) == 0,
        "policy_id": arguments["policy_id"],
        "policy_name": policy.name,
        "violations": violations,
        "matched_rules": len(violations),
        "total_rules": len(rules),
        "execution_time_ms": round(elapsed, 2),
    }


async def _handle_list_policies(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    import sys

    if "/workspace/editions/business/src" not in sys.path:
        sys.path.insert(0, "/workspace/editions/business/src")
    if "/workspace/editions/community/src" not in sys.path:
        sys.path.insert(0, "/workspace/editions/community/src")

    from aictrlnet_business.services.agp_evaluation import AGPEvaluationService
    from core.tenant_context import get_current_tenant_id

    svc = AGPEvaluationService(db)
    tenant_id = get_current_tenant_id()
    policies = await svc.get_policies(tenant_id=tenant_id)
    return {
        "policies": [
            {
                "id": str(p.id),
                "name": p.name,
                "description": getattr(p, "description", ""),
                "policy_type": getattr(p, "policy_type", ""),
                "enabled": getattr(p, "enabled", True),
            }
            for p in (policies[: arguments.get("limit", 20)])
        ],
        "count": len(policies),
    }


async def _handle_check_compliance(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    import sys

    if "/workspace/editions/enterprise/src" not in sys.path:
        sys.path.insert(0, "/workspace/editions/enterprise/src")
    if "/workspace/editions/business/src" not in sys.path:
        sys.path.insert(0, "/workspace/editions/business/src")
    if "/workspace/editions/community/src" not in sys.path:
        sys.path.insert(0, "/workspace/editions/community/src")

    from aictrlnet_enterprise.services.mcp_compliance import MCPComplianceService
    from core.tenant_context import get_current_tenant_id

    svc = MCPComplianceService()
    tenant_id = get_current_tenant_id()
    return await svc.check_server_compliance(
        server_id=arguments["server_id"],
        tenant_id=tenant_id,
        db=db,
    )


# ---------------------------------------------------------------------------
# Wave 1 handlers — Three-Layer Reach + Control Spectrum + Approvals
# ---------------------------------------------------------------------------


async def _handle_list_adapters(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.adapter import AdapterService

    svc = AdapterService(db)
    adapters, total = await svc.discover_adapters(
        edition="community",
        category=arguments.get("category"),
        search=arguments.get("search"),
        skip=arguments.get("offset", 0),
        limit=arguments.get("limit", 50),
        include_unavailable=bool(arguments.get("include_unavailable", False)),
    )
    return {
        "adapters": [
            {
                "id": str(getattr(a, "id", "")),
                "name": getattr(a, "name", None),
                "category": getattr(a, "category", None),
                "edition": getattr(a, "edition", None),
                "description": getattr(a, "description", None),
                "available": getattr(a, "available", True),
            }
            for a in adapters
        ],
        "total": total,
    }


async def _handle_get_adapter(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.adapter import AdapterService

    svc = AdapterService(db)
    adapter = await svc.get_adapter(arguments["adapter_id"], edition="community")
    if not adapter:
        raise ToolExecutionError(f"Adapter {arguments['adapter_id']} not found")
    return {
        "id": str(getattr(adapter, "id", "")),
        "name": getattr(adapter, "name", None),
        "category": getattr(adapter, "category", None),
        "edition": getattr(adapter, "edition", None),
        "description": getattr(adapter, "description", None),
        "capabilities": getattr(adapter, "capabilities", None),
        "auth_type": getattr(adapter, "auth_type", None),
        "available": getattr(adapter, "available", True),
    }


async def _handle_list_my_adapter_configs(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.adapter_config import AdapterConfigService

    svc = AdapterConfigService(db)
    configs, total = await svc.list_user_configs(
        user_id=user_id,
        adapter_type=arguments.get("adapter_type"),
        enabled_only=bool(arguments.get("enabled_only", False)),
        skip=arguments.get("offset", 0),
        limit=arguments.get("limit", 50),
    )
    return {
        "configs": [
            {
                "id": str(c.id),
                "adapter_type": getattr(c, "adapter_type", None),
                "name": getattr(c, "name", None),
                "enabled": getattr(c, "enabled", True),
                "created_at": str(getattr(c, "created_at", "")),
            }
            for c in configs
        ],
        "total": total,
    }


async def _handle_test_adapter_config(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.adapter_config import AdapterConfigService

    svc = AdapterConfigService(db)
    config = await svc.get_user_config(user_id, arguments["config_id"])
    if not config:
        raise ToolExecutionError(f"Config {arguments['config_id']} not found")
    result = await svc.test_config(config, timeout=arguments.get("timeout_seconds", 10))
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"result": str(result)}


async def _handle_nl_to_workflow(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.nlp import NLPService

    svc = NLPService(db)
    result = await svc.process_natural_language(
        prompt=arguments["text"],
        context={"mode": "create", "user_id": user_id, **(arguments.get("context") or {})},
        user_id=user_id,
    )
    if "error" in result and result["error"]:
        raise ToolExecutionError(result["error"])

    workflow = result.get("workflow") or {}
    definition = workflow.get("definition") or {}
    nodes = definition.get("nodes") or []
    return {
        "workflow_id": workflow.get("id"),
        "name": workflow.get("name"),
        "node_count": len(nodes),
        "nodes": [
            {"id": n.get("id"), "type": n.get("type"), "label": n.get("label")}
            for n in nodes
        ],
        "status": "created",
    }


async def _handle_analyze_intent(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.nlp import NLPService

    svc = NLPService(db)
    text = arguments["text"]
    intent = await svc._analyze_intent(text)  # noqa: SLF001
    confidence = 0.0
    if isinstance(intent, dict):
        confidence = float(intent.get("confidence", 0.0))
    elif hasattr(intent, "confidence"):
        confidence = float(intent.confidence)
    return {"text": text, "intent": intent, "confidence": confidence}


async def _handle_get_workflow_autonomy(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import select

    from core.tenant_context import get_current_tenant_id
    from models.community_complete import WorkflowDefinition
    from services.autonomy_resolver import CommunityAutonomyResolver
    from services.autonomy_taxonomy import (
        level_to_auto_approve_threshold,
        level_to_phase,
        phase_descriptor,
    )

    workflow_id = arguments["workflow_id"]
    tenant_id = get_current_tenant_id()

    row = (
        await db.execute(
            select(
                WorkflowDefinition.autonomy_level,
                WorkflowDefinition.autonomy_locked,
                WorkflowDefinition.tenant_id,
            ).where(WorkflowDefinition.id == workflow_id)
        )
    ).one_or_none()
    if row is None:
        raise ToolExecutionError(f"Workflow {workflow_id} not found")
    level, locked, wf_tenant = row

    resolver = CommunityAutonomyResolver(db)
    resolved = await resolver.resolve(
        tenant_id=tenant_id or str(wf_tenant) or "default",
        user_id=user_id,
        workflow_id=workflow_id,
    )
    descriptor = phase_descriptor(resolved.phase)
    eff_level = resolved.level if resolved.level is not None else level
    return {
        "workflow_id": workflow_id,
        "autonomy_level": level,
        "autonomy_locked": bool(locked or False),
        "effective_level": eff_level,
        "phase": resolved.phase or (level_to_phase(level) if level is not None else None),
        "phase_label": descriptor.label,
        "phase_description": descriptor.tagline,
        "auto_approve_threshold": resolved.auto_approve_threshold
        or (level_to_auto_approve_threshold(level) if level is not None else None),
        "source": resolved.source,
    }


async def _handle_preview_autonomy(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    import json as _json
    from sqlalchemy import select

    from models.community_complete import WorkflowDefinition
    from services.autonomy_taxonomy import (
        level_to_auto_approve_threshold,
        level_to_phase,
        node_risk as static_node_risk,
    )

    workflow_id = arguments["workflow_id"]
    level = int(arguments["level"])
    if not 0 <= level <= 100:
        raise ToolExecutionError("level must be between 0 and 100")

    definition = (
        await db.execute(
            select(WorkflowDefinition.definition).where(
                WorkflowDefinition.id == workflow_id
            )
        )
    ).scalar_one_or_none()
    if definition is None:
        raise ToolExecutionError(f"Workflow {workflow_id} not found")
    if isinstance(definition, str):
        try:
            definition = _json.loads(definition)
        except _json.JSONDecodeError:
            definition = {}

    threshold = level_to_auto_approve_threshold(level)
    nodes = []
    would_gate = 0
    auto_approve = 0
    for node in definition.get("nodes", []) if isinstance(definition, dict) else []:
        node_type = str(node.get("type", ""))
        params = node.get("parameters") or {}
        override = params.get("risk_override") if isinstance(params, dict) else None
        risk = static_node_risk(node_type, risk_override=override)
        gated = risk > threshold
        if gated:
            would_gate += 1
        else:
            auto_approve += 1
        nodes.append(
            {
                "node_id": str(node.get("id", "")),
                "node_type": node_type,
                "risk": risk,
                "would_gate": gated,
            }
        )
    return {
        "workflow_id": workflow_id,
        "level": level,
        "phase": level_to_phase(level),
        "auto_approve_threshold": threshold,
        "total_nodes": len(nodes),
        "would_gate_count": would_gate,
        "auto_approve_count": auto_approve,
        "nodes": nodes,
    }


async def _handle_set_workflow_autonomy(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import select, update

    from core.tenant_context import get_current_tenant_id
    from models.community_complete import WorkflowDefinition

    workflow_id = arguments["workflow_id"]
    caller_tenant = get_current_tenant_id()

    wf_tenant = (
        await db.execute(
            select(WorkflowDefinition.tenant_id).where(
                WorkflowDefinition.id == workflow_id
            )
        )
    ).scalar_one_or_none()
    if wf_tenant is None:
        raise ToolExecutionError(f"Workflow {workflow_id} not found")
    if str(wf_tenant) != str(caller_tenant):
        raise ToolExecutionError("Cross-tenant workflow update denied")

    values: Dict[str, Any] = {}
    if "autonomy_level" in arguments:
        lvl = int(arguments["autonomy_level"])
        if not 0 <= lvl <= 100:
            raise ToolExecutionError("autonomy_level must be 0-100")
        values["autonomy_level"] = lvl
    if "autonomy_locked" in arguments:
        values["autonomy_locked"] = bool(arguments["autonomy_locked"])
    if not values:
        raise ToolExecutionError("No autonomy fields to update")

    await db.execute(
        update(WorkflowDefinition)
        .where(WorkflowDefinition.id == workflow_id)
        .values(**values)
    )
    await db.commit()
    return await _handle_get_workflow_autonomy(arguments, db, user_id)


def _ensure_business_sys_path() -> None:
    import sys

    if "/workspace/editions/business/src" not in sys.path:
        sys.path.insert(0, "/workspace/editions/business/src")
    if "/workspace/editions/community/src" not in sys.path:
        sys.path.insert(0, "/workspace/editions/community/src")


async def _handle_research_api(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.api_research_service import APIResearchService

    svc = APIResearchService()
    spec = await svc.research_api(
        api_name=arguments["api_name"],
        documentation_url=arguments.get("documentation_url"),
        user_context=arguments.get("user_context"),
    )
    return spec


async def _handle_generate_adapter(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    import asyncio as _asyncio

    from aictrlnet_business.services.generated_adapter_service import (
        GeneratedAdapterService,
    )

    svc = GeneratedAdapterService(db)
    record = await svc.create_generated_adapter(
        name=arguments["name"],
        user_id=user_id,
        description=arguments.get("description"),
        generation_mode=arguments.get("generation_mode", "python_code"),
        api_name=arguments["api_name"],
    )
    adapter_id = record["id"]

    async def _run_generation():
        try:
            from core.database import AsyncSessionLocal

            async with AsyncSessionLocal() as bg_db:
                bg_svc = GeneratedAdapterService(bg_db)
                await bg_svc.generate_adapter(
                    adapter_id=adapter_id,
                    api_name=arguments["api_name"],
                    base_url=arguments["base_url"],
                    auth_type=arguments["auth_type"],
                    capabilities=arguments["capabilities"],
                    auth_config=arguments.get("auth_config"),
                )
        except Exception as e:
            logger.exception("Background generate_adapter failed: %s", e)

    _asyncio.create_task(_run_generation())

    return {
        "adapter_id": adapter_id,
        "name": record.get("name"),
        "status": record.get("status"),
        "poll_tool": "get_generated_adapter_status",
        "message": (
            "Generation kicked off in the background. Poll "
            "get_generated_adapter_status until status is approved | "
            "awaiting_approval | failed."
        ),
    }


async def _handle_self_extend(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    import asyncio as _asyncio

    from aictrlnet_business.services.api_research_service import APIResearchService
    from aictrlnet_business.services.generated_adapter_service import (
        GeneratedAdapterService,
    )

    research_svc = APIResearchService()
    spec = await research_svc.research_api(
        api_name=arguments["api_name"],
        documentation_url=arguments.get("documentation_url"),
        user_context=arguments.get("user_context"),
    )

    adapter_svc = GeneratedAdapterService(db)
    record = await adapter_svc.create_generated_adapter(
        name=arguments["api_name"],
        user_id=user_id,
        description=f"Self-extended from research on '{arguments['api_name']}'",
        generation_mode=arguments.get("generation_mode", "python_code"),
        api_name=arguments["api_name"],
        api_documentation_url=arguments.get("documentation_url"),
    )
    adapter_id = record["id"]

    async def _run_generation():
        try:
            from core.database import AsyncSessionLocal

            async with AsyncSessionLocal() as bg_db:
                bg_svc = GeneratedAdapterService(bg_db)
                await bg_svc.generate_adapter(
                    adapter_id=adapter_id,
                    api_name=spec.get("api_name") or arguments["api_name"],
                    base_url=spec.get("base_url") or "",
                    auth_type=spec.get("auth_type") or "none",
                    capabilities=spec.get("capabilities") or [],
                    auth_config=spec.get("auth_config"),
                    research_data=spec,
                )
        except Exception as e:
            logger.exception("Background self_extend generation failed: %s", e)

    _asyncio.create_task(_run_generation())

    return {
        "adapter_id": adapter_id,
        "api_name": spec.get("api_name") or arguments["api_name"],
        "research": {
            "base_url": spec.get("base_url"),
            "auth_type": spec.get("auth_type"),
            "capabilities_count": len(spec.get("capabilities") or []),
            "confidence": spec.get("confidence"),
            "source": spec.get("source"),
        },
        "status": record.get("status"),
        "poll_tool": "get_generated_adapter_status",
    }


async def _handle_list_generated_adapters(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.generated_adapter_service import (
        GeneratedAdapterService,
    )

    svc = GeneratedAdapterService(db)
    records = await svc.list_generated_adapters(
        user_id=user_id if arguments.get("mine_only") else None,
        status=arguments.get("status"),
        limit=arguments.get("limit", 50),
        offset=arguments.get("offset", 0),
    )
    return {"adapters": records, "count": len(records)}


async def _handle_get_generated_adapter_status(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.generated_adapter_service import (
        GeneratedAdapterService,
    )

    svc = GeneratedAdapterService(db)
    record = await svc.get_by_id(arguments["adapter_id"], include_code=False)
    if not record:
        raise ToolExecutionError(f"Adapter {arguments['adapter_id']} not found")
    return {
        "id": record.get("id"),
        "name": record.get("name"),
        "status": record.get("status"),
        "status_message": record.get("status_message"),
        "risk_score": record.get("risk_score"),
        "risk_level": record.get("risk_level"),
        "risk_details": record.get("risk_details"),
        "validation_passed": record.get("validation_passed"),
        "validation_errors": record.get("validation_errors"),
        "generation_mode": record.get("generation_mode"),
        "capabilities": record.get("capabilities"),
        "created_at": record.get("created_at"),
    }


async def _handle_get_generated_adapter_source(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.generated_adapter_service import (
        GeneratedAdapterService,
    )

    svc = GeneratedAdapterService(db)
    record = await svc.get_by_id(arguments["adapter_id"], include_code=True)
    if not record:
        raise ToolExecutionError(f"Adapter {arguments['adapter_id']} not found")
    return {
        "id": record.get("id"),
        "status": record.get("status"),
        "generation_mode": record.get("generation_mode"),
        "generated_code": record.get("generated_code"),
        "adapter_class_name": record.get("adapter_class_name"),
        "declarative_spec": record.get("declarative_spec"),
        "ast_analysis": record.get("ast_analysis"),
    }


async def _handle_approve_adapter(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.generated_adapter_service import (
        GeneratedAdapterService,
    )

    svc = GeneratedAdapterService(db)
    result = await svc.approve_adapter(arguments["adapter_id"], approved_by=user_id)
    if not result:
        raise ToolExecutionError(f"Adapter {arguments['adapter_id']} not found")
    return {
        "id": result.get("id"),
        "status": result.get("status"),
        "message": result.get("status_message"),
    }


async def _handle_reject_adapter(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.generated_adapter_service import (
        GeneratedAdapterService,
    )

    svc = GeneratedAdapterService(db)
    result = await svc.reject_adapter(
        arguments["adapter_id"],
        rejected_by=user_id,
        reason=arguments.get("reason"),
    )
    if not result:
        raise ToolExecutionError(f"Adapter {arguments['adapter_id']} not found")
    return {
        "id": result.get("id"),
        "status": result.get("status"),
        "message": result.get("status_message"),
    }


async def _handle_activate_adapter(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.generated_adapter_service import (
        GeneratedAdapterService,
    )

    svc = GeneratedAdapterService(db)
    try:
        result = await svc.activate_adapter(arguments["adapter_id"])
    except ValueError as e:
        raise ToolExecutionError(str(e)) from e
    return {
        "id": result.get("id"),
        "status": result.get("status"),
        "registered_adapter_type": result.get("registered_adapter_type"),
        "message": result.get("status_message"),
    }


BROWSER_SERVICE_URL = "http://browser-service:8005/browser/execute"


async def _handle_browser_execute(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    import os

    import httpx

    from .browser_safety import BrowserSafetyError, validate_actions

    try:
        actions = validate_actions(
            arguments.get("actions") or [],
            tenant_id=None,
            feature_flags=None,
        )
    except BrowserSafetyError as e:
        raise ToolExecutionError(f"browser_execute denied: {e}") from e

    payload = {
        "actions": actions,
        "timeout_ms": int(arguments.get("timeout_ms", 30_000)),
    }
    if "viewport" in arguments and isinstance(arguments["viewport"], dict):
        payload["viewport"] = arguments["viewport"]

    url = os.environ.get("BROWSER_SERVICE_URL", BROWSER_SERVICE_URL)
    try:
        async with httpx.AsyncClient(timeout=payload["timeout_ms"] / 1000 + 5) as client:
            resp = await client.post(url, json=payload)
    except httpx.HTTPError as e:
        raise ToolExecutionError(f"browser-service unreachable: {e}") from e

    if resp.status_code != 200:
        raise ToolExecutionError(
            f"browser-service error: {resp.status_code} — {resp.text[:200]}"
        )
    data = resp.json()
    return {
        "success": data.get("success"),
        "total_duration_ms": data.get("total_duration_ms"),
        "page_url": data.get("page_url"),
        "page_title": data.get("page_title"),
        "results": data.get("results", []),
    }


async def _handle_list_pending_approvals(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.approval import ApprovalService

    svc = ApprovalService(db)
    requests = await svc.list_requests(
        skip=arguments.get("offset", 0),
        limit=arguments.get("limit", 50),
        status="pending",
        workflow_id=arguments.get("workflow_id"),
        user_id=user_id if arguments.get("mine_only") else None,
        resource_type=arguments.get("resource_type"),
    )
    return {
        "requests": [
            {
                "id": str(r.id),
                "workflow_id": str(r.workflow_id),
                "requester_id": r.requester_id,
                "status": str(r.status),
                "resource_type": getattr(r, "resource_type", None),
                "resource_id": getattr(r, "resource_id", None),
                "context": getattr(r, "context", None),
                "reason": getattr(r, "reason", None),
                "created_at": str(getattr(r, "created_at", "")),
            }
            for r in requests
        ],
        "count": len(requests),
    }


async def _handle_get_approval(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.approval import ApprovalService

    svc = ApprovalService(db)
    req = await svc.get_request(arguments["request_id"])
    if not req:
        raise ToolExecutionError(
            f"Approval request {arguments['request_id']} not found"
        )
    return {
        "id": str(req.id),
        "workflow_id": str(req.workflow_id),
        "requester_id": req.requester_id,
        "status": str(req.status),
        "resource_type": getattr(req, "resource_type", None),
        "resource_id": getattr(req, "resource_id", None),
        "context": getattr(req, "context", None),
        "reason": getattr(req, "reason", None),
        "meta_data": getattr(req, "meta_data", None),
        "created_at": str(getattr(req, "created_at", "")),
    }


async def _handle_approve_request(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.approval import ApprovalService

    svc = ApprovalService(db)
    try:
        result = await svc.approve_request(
            request_id=arguments["request_id"],
            approver_id=user_id,
            comments=arguments.get("comments"),
        )
    except ValueError as e:
        raise ToolExecutionError(str(e)) from e
    return result


async def _handle_reject_request(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.approval import ApprovalService

    svc = ApprovalService(db)
    try:
        result = await svc.reject_request(
            request_id=arguments["request_id"],
            approver_id=user_id,
            reason=arguments["reason"],
        )
    except ValueError as e:
        raise ToolExecutionError(str(e)) from e
    return result


# ---------------------------------------------------------------------------
# Wave 2 handlers — API-key economy + governance visibility
# ---------------------------------------------------------------------------


async def _handle_list_api_keys(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.api_key_service import APIKeyService

    svc = APIKeyService(db)
    keys = await svc.list_user_api_keys(user_id=user_id)
    return {
        "api_keys": [
            k.to_dict() if hasattr(k, "to_dict") else {
                "id": getattr(k, "id", None),
                "name": getattr(k, "name", None),
                "key_identifier": f"{getattr(k, 'key_prefix', '')}...{getattr(k, 'key_suffix', '')}",
                "scopes": getattr(k, "scopes", []) or [],
                "is_active": getattr(k, "is_active", True),
                "expires_at": str(getattr(k, "expires_at", "") or ""),
                "last_used_at": str(getattr(k, "last_used_at", "") or ""),
                "usage_count": getattr(k, "usage_count", 0),
            }
            for k in keys
        ],
        "count": len(keys) if hasattr(keys, "__len__") else 0,
    }


async def _handle_get_api_key_usage(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Aggregate APIKeyLog rows for per-tool usage over a window.

    The MCP audit log entry we emit on every tool call (via
    MCPComplianceService.audit_mcp_operation) keys by
    operation_type=f"mcp_tool:{tool_name}", which is where the per-tool
    grain lives for Enterprise. Community falls back to the api_key_logs
    table's endpoint column as a proxy for tool name.
    """
    from datetime import datetime, timedelta

    from sqlalchemy import func, select

    from models.api_key import APIKey, APIKeyLog

    days = int(arguments.get("days", 30))
    days = max(1, min(days, 90))
    since = datetime.utcnow() - timedelta(days=days)

    # Scope to the caller's keys only (or a specific one they own)
    keys_q = select(APIKey).where(APIKey.user_id == user_id)
    if arguments.get("api_key_id"):
        keys_q = keys_q.where(APIKey.id == arguments["api_key_id"])
    key_rows = (await db.execute(keys_q)).scalars().all()
    key_ids = [str(k.id) for k in key_rows]

    if not key_ids:
        return {"usage": [], "days": days, "total_calls": 0}

    stmt = (
        select(APIKeyLog.endpoint, func.count().label("n"))
        .where(APIKeyLog.api_key_id.in_(key_ids))
        .where(APIKeyLog.created_at >= since)
        .group_by(APIKeyLog.endpoint)
        .order_by(func.count().desc())
    )
    rows = (await db.execute(stmt)).all()
    total = sum(r.n for r in rows)
    return {
        "usage": [{"endpoint": r.endpoint, "calls": int(r.n)} for r in rows],
        "days": days,
        "total_calls": int(total),
        "api_key_ids_included": key_ids,
    }


async def _handle_get_subscription(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.subscription import SubscriptionService
    except ImportError:
        # Community deploys get a minimal synthetic response.
        return {
            "plan": "community",
            "status": "active",
            "edition": "community",
            "features": {},
            "limits": {},
            "available": False,
            "message": "Subscription service not available in this edition",
        }

    svc = SubscriptionService(db)
    sub = await svc.get_current_subscription(user_id=user_id)
    if not sub:
        return {
            "plan": None,
            "status": "none",
            "edition": "community",
            "message": "No active subscription; defaulting to community tier",
        }
    if hasattr(sub, "model_dump"):
        return sub.model_dump()
    if hasattr(sub, "dict"):
        return sub.dict()
    return {"subscription": str(sub)}


async def _handle_get_upgrade_options(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.subscription import SubscriptionService
    except ImportError:
        return {"plans": [], "available": False}

    svc = SubscriptionService(db)
    plans = await svc.get_subscription_plans(
        edition=arguments.get("target_edition")
    )
    out = []
    for p in plans:
        if hasattr(p, "model_dump"):
            out.append(p.model_dump())
        elif hasattr(p, "dict"):
            out.append(p.dict())
        else:
            out.append({"plan": str(p)})
    return {"plans": out, "count": len(out)}


# -- Governance visibility (Business) --


async def _handle_list_ai_policies(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from sqlalchemy import select

    from aictrlnet_business.models.ai_governance import AIPolicy
    from core.tenant_context import get_current_tenant_id

    tenant_id = get_current_tenant_id() or "default-tenant"
    query = select(AIPolicy).where(AIPolicy.tenant_id == tenant_id)
    if arguments.get("policy_type"):
        query = query.where(AIPolicy.policy_type == arguments["policy_type"])
    if "enabled" in arguments:
        query = query.where(AIPolicy.enabled == bool(arguments["enabled"]))
    query = query.order_by(AIPolicy.created_at.desc())
    policies = (await db.execute(query)).scalars().all()
    return {
        "policies": [
            {
                "id": str(p.id),
                "name": p.name,
                "description": getattr(p, "description", None),
                "policy_type": getattr(p, "policy_type", None),
                "severity": getattr(p, "severity", None),
                "enabled": getattr(p, "enabled", True),
                "applies_to": getattr(p, "applies_to", None),
                "created_at": str(getattr(p, "created_at", "")),
            }
            for p in policies
        ],
        "count": len(policies),
    }


async def _handle_create_policy(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.models.ai_governance import AIAuditLog, AIPolicy
    from core.tenant_context import get_current_tenant_id

    tenant_id = get_current_tenant_id() or "default-tenant"
    policy = AIPolicy(
        name=arguments["name"],
        description=arguments.get("description"),
        policy_type=arguments["policy_type"],
        rules=arguments["rules"],
        applies_to=arguments.get("applies_to") or [],
        severity=arguments.get("severity", "medium"),
        enabled=bool(arguments.get("enabled", True)),
        resource_metadata=arguments.get("resource_metadata") or {},
        tenant_id=tenant_id,
    )
    db.add(policy)
    await db.commit()
    await db.refresh(policy)

    # Mirror the endpoint behavior: create an audit log for the creation
    try:
        audit = AIAuditLog(
            action="policy_created",
            user_id=user_id,
            details={
                "policy_id": str(policy.id),
                "policy_name": policy.name,
                "policy_type": policy.policy_type,
            },
            status="success",
        )
        db.add(audit)
        await db.commit()
    except Exception as e:
        logger.warning("Failed to log policy creation: %s", e)

    return {
        "id": str(policy.id),
        "name": policy.name,
        "policy_type": policy.policy_type,
        "severity": policy.severity,
        "enabled": policy.enabled,
        "tenant_id": policy.tenant_id,
        "created_at": str(policy.created_at),
    }


async def _handle_get_ai_audit_logs(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from datetime import datetime

    from sqlalchemy import select

    from aictrlnet_business.models.ai_governance import AIAuditLog

    query = select(AIAuditLog)
    if arguments.get("action"):
        query = query.where(AIAuditLog.action == arguments["action"])
    if arguments.get("model_id"):
        query = query.where(AIAuditLog.model_id == arguments["model_id"])
    if arguments.get("user_id"):
        query = query.where(AIAuditLog.user_id == arguments["user_id"])
    if arguments.get("status"):
        query = query.where(AIAuditLog.status == arguments["status"])
    if arguments.get("start_date"):
        try:
            start = datetime.fromisoformat(arguments["start_date"].replace("Z", "+00:00"))
            query = query.where(AIAuditLog.timestamp >= start)
        except ValueError:
            raise ToolExecutionError("Invalid start_date; use ISO-8601")
    if arguments.get("end_date"):
        try:
            end = datetime.fromisoformat(arguments["end_date"].replace("Z", "+00:00"))
            query = query.where(AIAuditLog.timestamp <= end)
        except ValueError:
            raise ToolExecutionError("Invalid end_date; use ISO-8601")

    limit = min(int(arguments.get("limit", 100)), 1000)
    offset = int(arguments.get("offset", 0))
    query = query.order_by(AIAuditLog.timestamp.desc()).limit(limit).offset(offset)

    logs = (await db.execute(query)).scalars().all()
    return {
        "logs": [
            {
                "id": str(log.id),
                "action": log.action,
                "user_id": log.user_id,
                "model_id": getattr(log, "model_id", None),
                "status": log.status,
                "details": getattr(log, "details", None),
                "timestamp": str(getattr(log, "timestamp", "")),
            }
            for log in logs
        ],
        "count": len(logs),
        "limit": limit,
        "offset": offset,
    }


async def _handle_list_violations(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    from aictrlnet_business.services.governance_service import GovernanceService
    from core.tenant_context import get_current_tenant_id

    svc = GovernanceService(db)
    tenant_id = get_current_tenant_id() or "default"
    violations = await svc.list_violations(
        skip=int(arguments.get("offset", 0)),
        limit=min(int(arguments.get("limit", 100)), 1000),
        policy_id=arguments.get("policy_id"),
        severity=arguments.get("severity"),
        resolved=arguments.get("resolved"),
        tenant_id=tenant_id,
    )
    return {
        "violations": [
            {
                "id": str(v.id),
                "policy_id": str(v.policy_id) if v.policy_id else None,
                "resource_type": getattr(v, "resource_type", None),
                "resource_id": getattr(v, "resource_id", None),
                "violation_type": getattr(v, "violation_type", None),
                "details": getattr(v, "details", None),
                "severity": getattr(v, "severity", None),
                "resolved": getattr(v, "resolved", False),
                "resolved_at": str(getattr(v, "resolved_at", "")) if getattr(v, "resolved_at", None) else None,
                "resolved_by": getattr(v, "resolved_by", None),
                "created_at": str(getattr(v, "created_at", "")),
            }
            for v in violations
        ],
        "count": len(violations),
    }


# ---------------------------------------------------------------------------
# Wave 3 handlers — trial metering surface
# ---------------------------------------------------------------------------


async def _handle_get_trial_status(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Combined usage + limits + upgrade prompts for the caller's tenant.

    Also pulls the MCP-layer meter counters (mcp_meters table) so
    llm_calls and browser_actions totals match what the metering
    decorator is actually charging.
    """
    from datetime import datetime, timezone

    from sqlalchemy import text

    from core.tenant_context import get_current_tenant_id
    from services.usage_service import UsageService

    tenant_id = get_current_tenant_id() or "community"
    svc = UsageService(db)

    # 1. Community edition usage + limits + upgrade prompts
    try:
        status = await svc.get_usage_status(tenant_id)
        status_payload = (
            status.model_dump() if hasattr(status, "model_dump")
            else status.dict() if hasattr(status, "dict")
            else {"status": str(status)}
        )
    except Exception as e:
        logger.warning("get_usage_status failed: %s", e)
        status_payload = {"error": "usage_status_unavailable", "message": str(e)}

    # 2. MCP-layer meter counters (llm_calls, browser_actions) — the
    # canonical answer to v11.2 "trial metering works through MCP".
    meters: Dict[str, Dict[str, Any]] = {}
    try:
        rows = (
            await db.execute(
                text(
                    """
                    SELECT meter, counter, limit_override, period_start, period_end
                    FROM mcp_meters
                    WHERE tenant_id = :t
                      AND period_start = date_trunc('month', now())
                    """
                ),
                {"t": tenant_id},
            )
        ).all()
        for meter, counter, limit_override, period_start, period_end in rows:
            meters[meter] = {
                "used": int(counter),
                "limit_override": int(limit_override) if limit_override is not None else None,
                "period_start": str(period_start),
                "period_end": str(period_end),
            }
    except Exception as e:
        logger.warning("mcp_meters lookup failed: %s", e)

    # 3. Default limits from the metering config so the response is
    # self-contained (no need for the caller to look up DEFAULT_LIMITS).
    from .metering import DEFAULT_LIMITS
    # Edition is carried on the user's plan; re-resolve via plan_gate
    # to keep the service consistent with the gate's decisions.
    from .plan_gate import PlanService

    plan_svc = PlanService(db)
    edition = await plan_svc.get_effective_edition(tenant_id)
    default_limits = DEFAULT_LIMITS.get(edition, DEFAULT_LIMITS["community"])

    # Merge: ensure every configured meter appears even if no usage yet
    for meter, limit in default_limits.items():
        entry = meters.get(meter, {"used": 0, "limit_override": None})
        effective_limit = entry.get("limit_override") or limit
        entry["limit"] = int(effective_limit)
        entry["remaining"] = max(0, int(effective_limit) - entry["used"])
        entry["percent_used"] = (
            round(entry["used"] * 100.0 / effective_limit, 2)
            if effective_limit
            else 0.0
        )
        meters[meter] = entry

    over_80 = [m for m, e in meters.items() if e.get("percent_used", 0) >= 80]
    upgrade_url = None
    if over_80 or status_payload.get("needs_upgrade"):
        upgrade_url = f"/pricing?from={edition}&reason=trial_threshold"

    return {
        "edition": edition,
        "tenant_id": tenant_id,
        "mcp_meters": meters,
        "status": status_payload,
        "upgrade_url": upgrade_url,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


async def _handle_get_usage_report(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Per-resource usage aggregation over a rolling window.

    Pulls from usage_tracking (if present — Business+ detail) and falls
    back to the basic_usage_metrics snapshot if not. Scoped to the
    caller's tenant only.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import text

    from core.tenant_context import get_current_tenant_id

    tenant_id = get_current_tenant_id() or "community"
    days = max(1, min(int(arguments.get("days", 30)), 90))
    since = datetime.now(timezone.utc) - timedelta(days=days)
    resource_filter = arguments.get("resource_type")

    # Per-day per-resource breakdown. usage_tracking is Business-edition
    # (detailed); skip gracefully if the table isn't there.
    breakdown: list = []
    by_resource: Dict[str, float] = {}
    total = 0.0
    source = "usage_tracking"
    try:
        sql = """
            SELECT date_trunc('day', timestamp) AS day,
                   resource_type,
                   SUM(quantity) AS qty
            FROM usage_tracking
            WHERE tenant_id = :t
              AND timestamp >= :since
        """
        params = {"t": tenant_id, "since": since}
        if resource_filter:
            sql += " AND resource_type = :rt"
            params["rt"] = resource_filter
        sql += " GROUP BY day, resource_type ORDER BY day DESC, resource_type"

        rows = (await db.execute(text(sql), params)).all()
        for day, resource_type, qty in rows:
            breakdown.append(
                {
                    "day": str(day),
                    "resource_type": resource_type,
                    "quantity": float(qty),
                }
            )
            by_resource[resource_type] = by_resource.get(resource_type, 0.0) + float(qty)
            total += float(qty)
    except Exception as e:
        logger.info("usage_tracking unavailable (%s); falling back to basic metrics", e)
        source = "basic_usage_metrics"
        # Fall back to current snapshot
        try:
            from services.usage_service import UsageService

            usage = await UsageService(db).get_current_usage(tenant_id)
            snapshot = (
                usage.model_dump() if hasattr(usage, "model_dump")
                else usage.dict() if hasattr(usage, "dict")
                else {}
            )
            for k, v in snapshot.items():
                if isinstance(v, (int, float)) and k not in ("id",):
                    by_resource[k] = float(v)
                    total += float(v)
        except Exception as e2:
            logger.warning("basic_usage_metrics fallback failed: %s", e2)

    return {
        "tenant_id": tenant_id,
        "days": days,
        "source": source,
        "total": total,
        "by_resource": by_resource,
        "breakdown": breakdown,
        "resource_filter": resource_filter,
    }


# ---------------------------------------------------------------------------
# Wave 4 handlers — horizontal business + living platform
# ---------------------------------------------------------------------------


# ---- Tasks ----

def _task_to_dict(t) -> Dict[str, Any]:
    return {
        "id": str(getattr(t, "id", "")),
        "name": getattr(t, "name", None),
        "description": getattr(t, "description", None),
        "status": getattr(t, "status").value if hasattr(getattr(t, "status", None), "value") else str(getattr(t, "status", "")),
        "metadata": getattr(t, "task_metadata", None),
        "created_at": str(getattr(t, "created_at", "")),
    }


async def _handle_create_task(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.task_service import TaskService

    svc = TaskService(db)
    task = await svc.create_task(
        name=arguments["name"],
        description=arguments.get("description", ""),
        metadata=arguments.get("metadata"),
    )
    return _task_to_dict(task)


async def _handle_list_tasks(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.task_service import TaskService

    svc = TaskService(db)
    filters = {}
    if arguments.get("status"):
        filters["status"] = arguments["status"]
    tasks = await svc.list_tasks(
        filters=filters or None,
        limit=min(int(arguments.get("limit", 100)), 500),
        offset=int(arguments.get("offset", 0)),
    )
    return {"tasks": [_task_to_dict(t) for t in tasks], "count": len(tasks)}


async def _handle_get_task(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.task_service import TaskService

    svc = TaskService(db)
    task = await svc.get_task(arguments["task_id"])
    if not task:
        raise ToolExecutionError(f"Task {arguments['task_id']} not found")
    return _task_to_dict(task)


async def _handle_update_task(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from core.exceptions import NotFoundError
    from services.task_service import TaskService

    svc = TaskService(db)
    try:
        task = await svc.update_task(
            task_id=arguments["task_id"],
            name=arguments.get("name"),
            description=arguments.get("description"),
            status=arguments.get("status"),
            metadata=arguments.get("metadata"),
        )
    except NotFoundError as e:
        raise ToolExecutionError(str(e)) from e
    return _task_to_dict(task)


async def _handle_complete_task(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from datetime import datetime

    from core.exceptions import NotFoundError
    from services.task_service import TaskService

    svc = TaskService(db)
    meta = {"completed_at": datetime.utcnow().isoformat()}
    if arguments.get("result") is not None:
        meta["result"] = arguments["result"]
    try:
        task = await svc.update_task(
            task_id=arguments["task_id"], status="completed", metadata=meta
        )
    except NotFoundError as e:
        raise ToolExecutionError(str(e)) from e
    return _task_to_dict(task)


# ---- Memory (in-memory per-user dict, Community) ----
# Replicates the semantics of editions/community/src/api/v1/endpoints/memory_basic.py.
# Uses a module-local store — same process-scoped lifetime as the HTTP endpoint.

_MCP_MEMORY_STORE: Dict[str, Dict[str, Any]] = {}
_MCP_MEMORY_MAX_KEYS = 1000
_MCP_MEMORY_MAX_SIZE = 10 * 1024 * 1024  # 10 MB


def _mcp_user_memory(user_id: str) -> Dict[str, Any]:
    return _MCP_MEMORY_STORE.setdefault(user_id, {})


def _mcp_memory_size(values: Dict[str, Any]) -> int:
    return sum(len(str(v)) for v in values.values())


async def _handle_get_memory(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    mem = _mcp_user_memory(user_id)
    key = arguments["key"]
    return {"key": key, "value": mem.get(key), "exists": key in mem}


async def _handle_set_memory(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    mem = _mcp_user_memory(user_id)
    key = arguments["key"]
    value = arguments["value"]
    projected = dict(mem)
    projected[key] = value
    if len(projected) > _MCP_MEMORY_MAX_KEYS:
        raise ToolExecutionError(f"Memory key limit exceeded ({_MCP_MEMORY_MAX_KEYS})")
    if _mcp_memory_size(projected) > _MCP_MEMORY_MAX_SIZE:
        raise ToolExecutionError("Memory size limit exceeded (10 MB)")
    mem[key] = value
    return {"key": key, "stored": True, "total_keys": len(mem)}


async def _handle_delete_memory(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    mem = _mcp_user_memory(user_id)
    key = arguments["key"]
    existed = key in mem
    mem.pop(key, None)
    return {"key": key, "deleted": existed}


# ---- Conversations + Channels ----


async def _handle_list_conversations(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.enhanced_conversation_manager import EnhancedConversationService

    svc = EnhancedConversationService(db)
    sessions = await svc.get_active_sessions(user_id)
    # get_active_sessions returns either list or dict{"sessions": [...]}
    if isinstance(sessions, dict):
        sessions_list = sessions.get("sessions") or []
    else:
        sessions_list = list(sessions or [])
    return {
        "sessions": [
            {
                "id": str(getattr(s, "id", "") or (s.get("id") if isinstance(s, dict) else "")),
                "created_at": str(
                    getattr(s, "created_at", "")
                    or (s.get("created_at") if isinstance(s, dict) else "")
                ),
                "channel": getattr(s, "channel", None)
                or (s.get("channel") if isinstance(s, dict) else None),
            }
            for s in sessions_list
        ],
        "count": len(sessions_list),
    }


async def _handle_get_conversation(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import desc, select

    try:
        from models.community_complete import ConversationMessage, ConversationSession
    except ImportError:
        raise ToolExecutionError("Conversation models unavailable")

    session_id = arguments["session_id"]
    limit = min(int(arguments.get("message_limit", 50)), 500)

    session = (
        await db.execute(
            select(ConversationSession).where(ConversationSession.id == session_id)
        )
    ).scalar_one_or_none()
    if not session:
        raise ToolExecutionError(f"Conversation {session_id} not found")

    msgs = (
        await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(desc(ConversationMessage.created_at))
            .limit(limit)
        )
    ).scalars().all()

    return {
        "id": str(session.id),
        "user_id": getattr(session, "user_id", None),
        "created_at": str(getattr(session, "created_at", "")),
        "messages": [
            {
                "id": str(getattr(m, "id", "")),
                "role": getattr(m, "role", None),
                "content": getattr(m, "content", None),
                "created_at": str(getattr(m, "created_at", "")),
            }
            for m in reversed(msgs)  # oldest first
        ],
    }


async def _handle_list_linked_channels(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    try:
        from api.v1.endpoints.channel_link import list_linked_channels

        # The endpoint takes user_id + db; duck-type current_user arg
        class _U:
            def __init__(self, uid):
                self.id = uid

        result = await list_linked_channels(current_user=_U(user_id), db=db)
    except Exception as e:
        logger.info("channel_link service unavailable: %s", e)
        return {"channels": [], "available": False}
    if isinstance(result, dict):
        return result
    return {"channels": result, "count": len(result) if hasattr(result, "__len__") else 0}


async def _handle_request_channel_link_code(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    try:
        from api.v1.endpoints.channel_link import request_link_code

        class _U:
            def __init__(self, uid):
                self.id = uid

        # The endpoint is a FastAPI handler — call its inner logic via request
        result = await request_link_code(
            channel_type=arguments["channel_type"],
            db=db,
            current_user=_U(user_id),
        )
        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return {"code": str(result)}
    except Exception as e:
        raise ToolExecutionError(f"request_channel_link_code failed: {e}") from e


async def _handle_unlink_channel(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    try:
        from api.v1.endpoints.channel_link import unlink_channel

        class _U:
            def __init__(self, uid):
                self.id = uid

        result = await unlink_channel(
            channel_type=arguments["channel_type"],
            channel_user_id=arguments["channel_user_id"],
            db=db,
            current_user=_U(user_id),
        )
        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return {"unlinked": True}
    except Exception as e:
        raise ToolExecutionError(f"unlink_channel failed: {e}") from e


async def _handle_send_channel_message(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Send a message via a linked external channel. Enforces channel
    ownership by calling list_linked_channels first and checking that
    the requested channel belongs to the caller.
    """
    # 1. Verify caller owns a linked channel of this type
    linked = await _handle_list_linked_channels({}, db, user_id)
    channels = linked.get("channels") or []
    wanted = arguments["channel_type"].lower()
    matches = [
        c for c in channels
        if (c.get("channel_type") if isinstance(c, dict) else getattr(c, "channel_type", None)) == wanted
    ]
    if not matches:
        raise ToolExecutionError(
            f"No linked channel of type '{wanted}' — link it first via "
            f"request_channel_link_code"
        )

    # 2. Dispatch through conversation service (which wraps the channel
    # gateway). EnhancedConversationService.process_message already
    # supports multi-channel delivery.
    from services.enhanced_conversation_manager import EnhancedConversationService

    svc = EnhancedConversationService(db)
    active_sessions = await svc.get_active_sessions(user_id)
    if isinstance(active_sessions, dict):
        active_sessions = active_sessions.get("sessions") or []
    if active_sessions:
        session = active_sessions[0]
    else:
        session = await svc.create_session(user_id)

    session_id = getattr(session, "id", None) or (
        session.get("id") if isinstance(session, dict) else None
    )
    response = await svc.process_message(
        session_id=session_id,
        content=arguments["message"],
        user_id=user_id,
    )
    payload: Dict[str, Any]
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif hasattr(response, "dict"):
        payload = response.dict()
    elif isinstance(response, dict):
        payload = response
    else:
        payload = {"response": str(response)}
    payload["channel_type"] = wanted
    return payload


async def _handle_list_notifications(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from sqlalchemy import desc, select

        from aictrlnet_business.models.notification import Notification  # type: ignore
    except Exception:
        return {"notifications": [], "available": False}

    query = select(Notification).where(Notification.user_id == user_id)
    if arguments.get("unread_only"):
        query = query.where(Notification.read == False)  # noqa: E712
    query = (
        query.order_by(desc(Notification.created_at))
        .limit(min(int(arguments.get("limit", 50)), 200))
        .offset(int(arguments.get("offset", 0)))
    )
    rows = (await db.execute(query)).scalars().all()
    return {
        "notifications": [
            {
                "id": str(getattr(n, "id", "")),
                "type": getattr(n, "type", None),
                "title": getattr(n, "title", None),
                "body": getattr(n, "body", None),
                "read": getattr(n, "read", False),
                "created_at": str(getattr(n, "created_at", "")),
            }
            for n in rows
        ],
        "count": len(rows),
    }


async def _handle_mark_notification_read(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from datetime import datetime

        from sqlalchemy import select

        from aictrlnet_business.models.notification import Notification  # type: ignore
    except Exception:
        return {"marked": False, "available": False}

    n = (
        await db.execute(
            select(Notification).where(
                Notification.id == arguments["notification_id"],
                Notification.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if not n:
        raise ToolExecutionError(
            f"Notification {arguments['notification_id']} not found"
        )
    n.read = True
    n.read_at = datetime.utcnow()
    await db.commit()
    return {"id": str(n.id), "marked": True}


# ---- Knowledge ----


async def _handle_query_knowledge(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.knowledge.knowledge_retrieval_service import KnowledgeRetrievalService

    svc = KnowledgeRetrievalService(db)
    results = await svc.find_relevant_knowledge(
        query=arguments["query"],
        context=arguments.get("context"),
        limit=min(int(arguments.get("limit", 5)), 20),
    )
    return {
        "results": [
            (r.model_dump() if hasattr(r, "model_dump")
             else r.dict() if hasattr(r, "dict")
             else {"content": str(r)})
            for r in results
        ],
        "count": len(results) if hasattr(results, "__len__") else 0,
    }


async def _handle_suggest_next_actions(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.knowledge.knowledge_retrieval_service import KnowledgeRetrievalService

    svc = KnowledgeRetrievalService(db)
    actions = await svc.suggest_next_actions(
        current_action=arguments["current_action"],
        context=arguments.get("context"),
    )
    return {"suggestions": actions}


async def _handle_get_capabilities_summary(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.knowledge.knowledge_retrieval_service import KnowledgeRetrievalService

    svc = KnowledgeRetrievalService(db)
    summary = await svc.get_capabilities_summary()
    return {"summary": summary}


# ---- Templates ----


async def _handle_search_templates(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from schemas.workflow_templates import TemplateListRequest
    from services.workflow_template_service import create_workflow_template_service

    svc = create_workflow_template_service()
    request = TemplateListRequest(
        category=arguments.get("category"),
        search=arguments.get("query"),
        limit=min(int(arguments.get("limit", 20)), 100),
        skip=0,
    )
    templates, total = await svc.list_templates(
        db=db, user_id=user_id, request=request
    )
    return {
        "templates": [
            {
                "id": str(t.id),
                "name": t.name,
                "description": getattr(t, "description", ""),
                "category": getattr(t, "category", ""),
                "complexity": getattr(t, "complexity", None),
            }
            for t in templates
        ],
        "total": total,
    }


async def _handle_instantiate_template(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from services.workflow_template_service import create_workflow_template_service

    svc = create_workflow_template_service()
    # instantiate_template on the service creates a workflow from the template;
    # method name may vary — try the canonical one first.
    method = (
        getattr(svc, "instantiate_template", None)
        or getattr(svc, "create_from_template", None)
    )
    if not method:
        raise ToolExecutionError("Template instantiation not available in this edition")
    workflow = await method(
        db=db,
        user_id=user_id,
        template_id=arguments["template_id"],
        name=arguments.get("name"),
        parameters=arguments.get("parameters") or {},
    )
    return {
        "workflow_id": str(getattr(workflow, "id", "") or
                           (workflow.get("id") if isinstance(workflow, dict) else "")),
        "name": getattr(workflow, "name", None) or
                (workflow.get("name") if isinstance(workflow, dict) else None),
        "template_id": arguments["template_id"],
    }


# ---- Files ----


async def _handle_upload_file(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Accept a base64-encoded file, stage it, and return a file_id.

    Mirrors the behavior of POST /file-upload/upload but takes the
    content inline via args instead of multipart. Size + MIME rules
    are identical: 10 MB cap, allow-listed MIME, magic-byte check.
    """
    import base64
    import os
    import uuid
    from datetime import datetime

    try:
        from models.community_complete import StagedFile
    except ImportError:
        raise ToolExecutionError("StagedFile model unavailable")

    max_size = 50 * 1024 * 1024  # aligned with file_upload endpoint
    try:
        raw = base64.b64decode(arguments["content_base64"], validate=True)
    except Exception as e:
        raise ToolExecutionError(f"Invalid base64: {e}") from e
    if len(raw) > max_size:
        raise ToolExecutionError(
            f"File exceeds {max_size // (1024 * 1024)} MB limit"
        )

    # Basic filename sanitization — reject traversal
    filename = os.path.basename(arguments["filename"] or "file.bin").strip()
    if not filename or ".." in filename or "/" in filename:
        raise ToolExecutionError("Invalid filename")

    storage_dir = os.environ.get("STAGED_FILES_DIR", "/tmp/aictrlnet/staged_files")
    os.makedirs(storage_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    storage_path = os.path.join(storage_dir, file_id)
    with open(storage_path, "wb") as f:
        f.write(raw)

    staged = StagedFile(
        id=file_id,
        user_id=user_id,
        filename=filename,
        content_type=arguments.get("content_type", "application/octet-stream"),
        file_size=len(raw),
        storage_path=storage_path,
        workflow_id=arguments.get("workflow_id"),
        created_at=datetime.utcnow(),
    )
    db.add(staged)
    await db.commit()
    await db.refresh(staged)
    return {
        "file_id": file_id,
        "filename": filename,
        "content_type": staged.content_type,
        "file_size": len(raw),
        "workflow_id": staged.workflow_id,
    }


async def _handle_list_staged_files(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import desc, select

    try:
        from models.community_complete import StagedFile
    except ImportError:
        return {"files": [], "available": False}

    rows = (
        await db.execute(
            select(StagedFile)
            .where(StagedFile.user_id == user_id)
            .order_by(desc(StagedFile.created_at))
            .limit(min(int(arguments.get("limit", 50)), 500))
            .offset(int(arguments.get("offset", 0)))
        )
    ).scalars().all()
    return {
        "files": [
            {
                "file_id": str(f.id),
                "filename": f.filename,
                "content_type": f.content_type,
                "file_size": f.file_size,
                "created_at": str(getattr(f, "created_at", "")),
            }
            for f in rows
        ],
        "count": len(rows),
    }


async def _handle_get_staged_file(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    import base64
    from sqlalchemy import select

    try:
        from models.community_complete import StagedFile
    except ImportError:
        raise ToolExecutionError("StagedFile model unavailable")

    f = (
        await db.execute(
            select(StagedFile).where(
                StagedFile.id == arguments["file_id"],
                StagedFile.user_id == user_id,  # IDOR defense
            )
        )
    ).scalar_one_or_none()
    if not f:
        raise ToolExecutionError(f"File {arguments['file_id']} not found")

    out: Dict[str, Any] = {
        "file_id": str(f.id),
        "filename": f.filename,
        "content_type": f.content_type,
        "file_size": f.file_size,
        "created_at": str(getattr(f, "created_at", "")),
    }
    if arguments.get("include_content"):
        try:
            with open(f.storage_path, "rb") as fh:
                out["content_base64"] = base64.b64encode(fh.read()).decode("ascii")
        except Exception as e:
            raise ToolExecutionError(f"Could not read file content: {e}") from e
    return out


# ---- Data Quality ----


async def _handle_assess_data_quality(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    try:
        from services.data_quality_service import DataQualityService
    except ImportError:
        return {"score": None, "available": False, "message": "Data quality service not installed"}

    svc = DataQualityService(db)
    assess = getattr(svc, "assess", None) or getattr(svc, "assess_quality", None)
    if not assess:
        return {"score": None, "available": False}
    result = await assess(
        data=arguments["data"],
        dimensions=arguments.get("dimensions"),
        rules=arguments.get("rules"),
    )
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"result": result}


async def _handle_list_quality_dimensions(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    # Static dimension catalog — matches the DIMENSIONS referenced by
    # community data-quality endpoints.
    return {
        "dimensions": [
            {"name": "accuracy", "description": "Values match real-world truth"},
            {"name": "completeness", "description": "No missing values where expected"},
            {"name": "consistency", "description": "Values agree across records"},
            {"name": "timeliness", "description": "Values are current + up-to-date"},
            {"name": "uniqueness", "description": "No unintended duplicates"},
            {"name": "validity", "description": "Values conform to schema/format"},
        ]
    }


# ---- Agents ----


async def _handle_list_agents(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from sqlalchemy import select

        from aictrlnet_business.models.agent_registry import AgentRegistry  # type: ignore
        from core.tenant_context import get_current_tenant_id

        tenant_id = get_current_tenant_id() or "default"
        query = select(AgentRegistry).where(AgentRegistry.tenant_id == tenant_id)
        if arguments.get("agent_type"):
            query = query.where(AgentRegistry.agent_type == arguments["agent_type"])
        if arguments.get("enabled_only"):
            query = query.where(AgentRegistry.enabled == True)  # noqa: E712
        query = query.limit(int(arguments.get("limit", 50))).offset(
            int(arguments.get("offset", 0))
        )
        rows = (await db.execute(query)).scalars().all()
        return {
            "agents": [
                {
                    "id": str(getattr(a, "id", "")),
                    "name": getattr(a, "name", None),
                    "agent_type": getattr(a, "agent_type", None),
                    "enabled": getattr(a, "enabled", True),
                    "description": getattr(a, "description", None),
                }
                for a in rows
            ],
            "count": len(rows),
        }
    except Exception as e:
        logger.info("agent_registry unavailable: %s", e)
        return {"agents": [], "available": False}


async def _handle_get_agent_capabilities(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from sqlalchemy import select

        from aictrlnet_business.models.agent_registry import AgentRegistry  # type: ignore
    except Exception:
        raise ToolExecutionError("Agent registry unavailable in this edition")

    agent = (
        await db.execute(
            select(AgentRegistry).where(AgentRegistry.id == arguments["agent_id"])
        )
    ).scalar_one_or_none()
    if not agent:
        raise ToolExecutionError(f"Agent {arguments['agent_id']} not found")
    return {
        "id": str(agent.id),
        "name": agent.name,
        "agent_type": getattr(agent, "agent_type", None),
        "capabilities": getattr(agent, "capabilities", None),
        "tools": getattr(agent, "tools", None),
        "enabled": getattr(agent, "enabled", True),
        "autonomy_level": getattr(agent, "autonomy_level", None),
    }


async def _handle_set_agent_autonomy(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from sqlalchemy import select, update

        from aictrlnet_business.models.agent_registry import AgentRegistry  # type: ignore
        from core.tenant_context import get_current_tenant_id
    except Exception:
        raise ToolExecutionError("Agent registry unavailable")

    tenant_id = get_current_tenant_id() or "default"
    agent = (
        await db.execute(
            select(AgentRegistry).where(
                AgentRegistry.id == arguments["agent_id"],
                AgentRegistry.tenant_id == tenant_id,
            )
        )
    ).scalar_one_or_none()
    if not agent:
        raise ToolExecutionError(
            f"Agent {arguments['agent_id']} not found in tenant"
        )

    level = int(arguments["autonomy_level"])
    if not 0 <= level <= 100:
        raise ToolExecutionError("autonomy_level must be 0-100")
    await db.execute(
        update(AgentRegistry)
        .where(AgentRegistry.id == agent.id)
        .values(autonomy_level=level)
    )
    await db.commit()
    return {"id": str(agent.id), "autonomy_level": level}


async def _handle_execute_agent(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from services.agent_execution_service import AgentExecutionService
    except ImportError:
        try:
            from aictrlnet_business.services.agent_execution import AgentExecutionService  # type: ignore
        except Exception:
            raise ToolExecutionError("Agent execution service unavailable")

    svc = AgentExecutionService(db)
    method = (
        getattr(svc, "execute", None)
        or getattr(svc, "execute_agent", None)
        or getattr(svc, "run", None)
    )
    if not method:
        raise ToolExecutionError("AgentExecutionService has no execute method")
    result = await method(
        agent_id=arguments["agent_id"],
        prompt=arguments["prompt"],
        input_data=arguments.get("input_data"),
        user_id=user_id,
    )
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    if isinstance(result, dict):
        return result
    return {"result": str(result)}


# ---- LLM Registry ----


async def _handle_list_llm_models(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from sqlalchemy import select

        from aictrlnet_business.models.llm_registry import LLMModel  # type: ignore
    except Exception:
        return {"models": [], "available": False}

    query = select(LLMModel)
    if arguments.get("provider"):
        query = query.where(LLMModel.provider == arguments["provider"])
    if arguments.get("enabled_only", True):
        query = query.where(LLMModel.enabled == True)  # noqa: E712
    rows = (await db.execute(query)).scalars().all()
    return {
        "models": [
            {
                "id": str(getattr(m, "id", "")),
                "provider": getattr(m, "provider", None),
                "model_name": getattr(m, "model_name", None),
                "enabled": getattr(m, "enabled", True),
                "capabilities": getattr(m, "capabilities", None),
            }
            for m in rows
        ],
        "count": len(rows),
    }


async def _handle_get_llm_recommendation(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.llm_registry_service import LLMRegistryService  # type: ignore
    except Exception:
        return {"recommendation": None, "available": False}

    svc = LLMRegistryService(db)
    method = (
        getattr(svc, "recommend", None)
        or getattr(svc, "get_recommendation", None)
    )
    if not method:
        return {"recommendation": None, "available": False}
    rec = await method(
        task_type=arguments["task_type"],
        constraints=arguments.get("constraints") or {},
    )
    if hasattr(rec, "model_dump"):
        return rec.model_dump()
    if hasattr(rec, "dict"):
        return rec.dict()
    return {"recommendation": rec}


# ---- Living Platform — Patterns ----


async def _handle_list_pattern_candidates(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.learning_loop_service import (  # type: ignore
            LearningLoopService,
        )
    except Exception:
        return {"candidates": [], "available": False}

    svc = LearningLoopService(db)
    list_method = (
        getattr(svc, "list_pattern_candidates", None)
        or getattr(svc, "list_candidates", None)
    )
    if not list_method:
        return {"candidates": [], "available": False}
    candidates = await list_method(
        status=arguments.get("status"),
        min_confidence=arguments.get("min_confidence"),
        limit=min(int(arguments.get("limit", 50)), 200),
        offset=int(arguments.get("offset", 0)),
    )
    return {
        "candidates": [
            (c.model_dump() if hasattr(c, "model_dump")
             else c.dict() if hasattr(c, "dict")
             else {"id": str(getattr(c, "id", c))})
            for c in candidates
        ],
        "count": len(candidates) if hasattr(candidates, "__len__") else 0,
    }


async def _handle_promote_pattern_to_template(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.learning_loop_service import (  # type: ignore
            LearningLoopService,
        )
    except Exception:
        raise ToolExecutionError("Learning loop service unavailable in this edition")

    svc = LearningLoopService(db)
    method = (
        getattr(svc, "promote_to_template", None)
        or getattr(svc, "convert_to_template", None)
    )
    if not method:
        raise ToolExecutionError("Promotion method unavailable")
    result = await method(
        pattern_id=arguments["pattern_id"],
        template_name=arguments.get("template_name"),
        category=arguments.get("category"),
        user_id=user_id,
    )
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"template_id": str(result) if result else None}


# ---- Living Platform — Org Discovery ----


async def _handle_org_discovery_scan(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.org_discovery_service import (  # type: ignore
            OrgDiscoveryService,
        )
    except Exception:
        raise ToolExecutionError("Org discovery service unavailable")

    svc = OrgDiscoveryService(db)
    scan = (
        getattr(svc, "scan", None)
        or getattr(svc, "trigger_scan", None)
    )
    if not scan:
        raise ToolExecutionError("scan method unavailable")
    result = await scan(
        user_id=user_id,
        sources=arguments.get("sources") or [],
    )
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"scan": result}


async def _handle_get_org_landscape(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.org_discovery_service import (  # type: ignore
            OrgDiscoveryService,
        )
    except Exception:
        return {"landscape": None, "available": False}

    svc = OrgDiscoveryService(db)
    method = (
        getattr(svc, "get_landscape", None)
        or getattr(svc, "get_profile", None)
    )
    if not method:
        return {"landscape": None, "available": False}
    landscape = await method(user_id=user_id)
    if hasattr(landscape, "model_dump"):
        return landscape.model_dump()
    if hasattr(landscape, "dict"):
        return landscape.dict()
    return {"landscape": landscape}


async def _handle_get_org_recommendations(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.org_discovery_service import (  # type: ignore
            OrgDiscoveryService,
        )
    except Exception:
        return {"recommendations": [], "available": False}

    svc = OrgDiscoveryService(db)
    method = (
        getattr(svc, "get_recommendations", None)
        or getattr(svc, "recommend", None)
    )
    if not method:
        return {"recommendations": [], "available": False}
    recs = await method(user_id=user_id, focus=arguments.get("focus"))
    return {
        "recommendations": [
            (r.model_dump() if hasattr(r, "model_dump")
             else r.dict() if hasattr(r, "dict")
             else r)
            for r in (recs if isinstance(recs, list) else [recs])
        ]
    }


# ---- Living Platform — Company Automation ----


async def _handle_automate_company(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Run Bobby's full company-automation orchestrator. Loads the
    matching industry pack automatically (or uses ``industry`` override)
    and returns the activation plan id + workflow/agent summary.
    """
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.schemas.company_automation import (  # type: ignore
            AutomateCompanyRequest,
        )
        from aictrlnet_business.services.company_automation_orchestrator import (  # type: ignore
            CompanyAutomationOrchestrator,
        )
    except Exception as e:
        raise ToolExecutionError(
            f"Company automation orchestrator unavailable: {e}"
        ) from e

    from uuid import UUID

    org_id = arguments.get("organization_id")
    try:
        org_uuid = UUID(org_id) if org_id else None
    except ValueError as e:
        raise ToolExecutionError(f"Invalid organization_id: {e}") from e

    request = AutomateCompanyRequest(
        request=arguments["request"],
        organization_id=org_uuid,
        use_bodaty_template=bool(arguments.get("use_bodaty_template", True)),
        industry_override=arguments.get("industry"),
    )

    orchestrator = CompanyAutomationOrchestrator()
    try:
        result = await orchestrator.automate_company(
            db=db, request=request, user_id=user_id
        )
    except ValueError as e:
        raise ToolExecutionError(str(e)) from e

    payload = (
        result.model_dump() if hasattr(result, "model_dump")
        else result.dict() if hasattr(result, "dict")
        else (result if isinstance(result, dict) else {"result": str(result)})
    )
    # Normalize key names — the orchestrator returns activation_plan_id
    # but we advertise plan_id for the polling contract.
    if "activation_plan_id" in payload and "plan_id" not in payload:
        payload["plan_id"] = str(payload["activation_plan_id"])
    payload.setdefault("poll_tool", "get_company_automation_status")
    # Cast UUIDs so json.dumps in the protocol layer doesn't choke
    if "organization_id" in payload and payload["organization_id"] is not None:
        payload["organization_id"] = str(payload["organization_id"])
    if "activation_plan_id" in payload and payload["activation_plan_id"] is not None:
        payload["activation_plan_id"] = str(payload["activation_plan_id"])
    return payload


async def _handle_get_company_automation_status(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Query CompanyActivationPlan by id and return the status + phases.

    Mirrors the GET /plans/{plan_id} endpoint so Claude polls the same
    source of truth the HitLai UI does.
    """
    _ensure_business_sys_path()
    try:
        from sqlalchemy import select

        from aictrlnet_business.models.company_automation import (  # type: ignore
            CompanyActivationPlan,
        )
    except Exception:
        return {"status": "unavailable", "available": False}

    plan = (
        await db.execute(
            select(CompanyActivationPlan).where(
                CompanyActivationPlan.id == arguments["plan_id"]
            )
        )
    ).scalar_one_or_none()
    if not plan:
        raise ToolExecutionError(f"Plan {arguments['plan_id']} not found")

    return {
        "plan_id": str(plan.id),
        "organization_id": str(getattr(plan, "organization_id", "") or ""),
        "name": getattr(plan, "name", None),
        "status": getattr(plan, "status", None),
        "industry": getattr(plan, "industry", None),
        "phases": getattr(plan, "phases", None),
        "current_phase": getattr(plan, "current_phase", None),
        "progress_percent": getattr(plan, "progress_percent", None),
        "created_at": str(getattr(plan, "created_at", "")),
        "updated_at": str(getattr(plan, "updated_at", "")),
    }


async def _handle_list_industry_packs(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """List the 41 available industry packs. Each pack bundles
    compliance, workflows, agents, integrations, KPIs, ROI calc for
    a vertical. Used by automate_company to customize the activation
    plan."""
    _ensure_business_sys_path()
    try:
        from services.industry_pack_loader import get_industry_pack_loader  # type: ignore
    except Exception:
        try:
            from services.industry_pack_loader import IndustryPackLoader  # type: ignore
            loader = IndustryPackLoader()
        except Exception as e:
            return {
                "packs": [],
                "available": False,
                "message": f"Industry pack loader unavailable: {e}",
            }
    else:
        loader = get_industry_pack_loader()

    method = getattr(loader, "list_industries", None)
    if method is None:
        return {"packs": [], "available": False}
    industries = method()
    return {
        "packs": industries,
        "count": len(industries) if hasattr(industries, "__len__") else 0,
    }


async def _handle_detect_industry(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Detect the best-fit industry pack for arbitrary text.

    Uses the same loader the orchestrator uses — keyword + alias match.
    Returns None if no pack matches (caller can fall through to a
    generic template)."""
    _ensure_business_sys_path()
    try:
        from services.industry_pack_loader import get_industry_pack_loader  # type: ignore
    except Exception:
        try:
            from services.industry_pack_loader import IndustryPackLoader  # type: ignore
            loader = IndustryPackLoader()
        except Exception as e:
            return {
                "industry": None,
                "available": False,
                "message": f"Industry pack loader unavailable: {e}",
            }
    else:
        loader = get_industry_pack_loader()

    industry_id = loader.detect_industry(arguments["text"]) if hasattr(
        loader, "detect_industry"
    ) else None
    return {
        "industry": industry_id,
        "text_length": len(arguments["text"]),
        "matched": industry_id is not None,
    }


# ---- Quality Verification ----


async def _handle_verify_quality(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.quality_verification_service import (  # type: ignore
            QualityVerificationService,
        )
    except Exception:
        # Fall back to the community assess_quality handler as a degraded path
        from mcp_server.services.quality import MCPQualityService

        svc = MCPQualityService(db)
        return await svc.assess_quality(
            content=arguments["content"],
            content_type=arguments.get("content_type", "text"),
            criteria={"standards": arguments.get("standards")},
        )

    svc = QualityVerificationService(db)
    method = (
        getattr(svc, "verify", None)
        or getattr(svc, "verify_quality", None)
    )
    if not method:
        return {"passed": None, "available": False}
    result = await method(
        content=arguments["content"],
        content_type=arguments.get("content_type", "text"),
        standards=arguments.get("standards") or [],
    )
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"result": result}


# ---------------------------------------------------------------------------
# Wave 5 handlers — Institute (education-led GTM, v11.1)
# ---------------------------------------------------------------------------
#
# The Institute platform service is under development. Tools return
# feature_pending in a structured shape so Claude can surface the status
# accurately — v11.3's claim-validation rule: never claim a feature is
# live if the backing code isn't shipped. Once InstituteService lands,
# these handlers can delegate to it with minimal change.


_INSTITUTE_PENDING_MESSAGE = (
    "AICtrlNet Institute platform is under development. This MCP tool "
    "is live but the backing service has not shipped yet. Track v11.1 "
    "education-led GTM milestone for availability."
)


def _institute_service(db):
    """Lazy-import helper. Returns None if the service module does not
    exist yet (pre-launch)."""
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.institute_service import (  # type: ignore
            InstituteService,
        )
    except Exception:
        return None
    return InstituteService(db)


async def _handle_list_institute_modules(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _institute_service(db)
    if svc is None:
        return {
            "status": "feature_pending",
            "message": _INSTITUTE_PENDING_MESSAGE,
            "available": False,
            "tier": arguments.get("tier"),
            "audience": arguments.get("audience"),
            "modules": [],
        }
    method = (
        getattr(svc, "list_modules", None)
        or getattr(svc, "get_modules", None)
    )
    if not method:
        return {"status": "feature_pending", "available": False, "modules": []}
    modules = await method(
        tier=arguments.get("tier"),
        audience=arguments.get("audience"),
    )
    return {
        "status": "available",
        "available": True,
        "modules": [
            (m.model_dump() if hasattr(m, "model_dump")
             else m.dict() if hasattr(m, "dict")
             else m)
            for m in modules
        ],
    }


async def _handle_enroll_in_module(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _institute_service(db)
    if svc is None:
        return {
            "status": "feature_pending",
            "message": _INSTITUTE_PENDING_MESSAGE,
            "available": False,
            "module_id": arguments["module_id"],
        }
    enroll = getattr(svc, "enroll", None) or getattr(svc, "enroll_user", None)
    if not enroll:
        return {"status": "feature_pending", "available": False}
    result = await enroll(user_id=user_id, module_id=arguments["module_id"])
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"enrollment": result}


async def _handle_get_certification_status(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _institute_service(db)
    if svc is None:
        return {
            "status": "feature_pending",
            "message": _INSTITUTE_PENDING_MESSAGE,
            "available": False,
            "completed_modules": [],
            "in_progress_modules": [],
            "certifications": [],
        }
    method = (
        getattr(svc, "get_certifications", None)
        or getattr(svc, "get_certification_status", None)
    )
    if not method:
        return {"status": "feature_pending", "available": False}
    result = await method(user_id=user_id)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"certifications": result}


# ---------------------------------------------------------------------------
# Wave 6 handlers — Enterprise admin (analytics, audit, compliance, fleet,
# license, federated knowledge, cross-tenant). All handlers:
#
# 1. Add the enterprise src path to sys.path so their service imports work.
# 2. try/except ImportError → graceful degradation for non-Enterprise
#    deploys. (Enterprise-tier plan gate already prevents the handler
#    from being called on Community/Business deployments, but if a
#    Community deploy somehow routes here, we degrade rather than crash.)
# 3. Use get_current_tenant_id() to scope reads — no cross-tenant leak.
# ---------------------------------------------------------------------------


def _ensure_enterprise_sys_path() -> None:
    import sys

    for p in (
        "/workspace/editions/enterprise/src",
        "/workspace/editions/business/src",
        "/workspace/editions/community/src",
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def _enterprise_pending(feature: str) -> Dict[str, Any]:
    return {
        "status": "feature_pending",
        "available": False,
        "feature": feature,
        "message": (
            "Enterprise service unavailable on this deploy. This MCP tool "
            "is plan-gated to Enterprise tier; the backing service ships "
            "with the Enterprise edition package."
        ),
    }


# ---- Analytics ----


async def _handle_query_analytics(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.analytics import AnalyticsService
    except Exception:
        return _enterprise_pending("analytics")

    svc = AnalyticsService(db)
    try:
        result = await svc.query_analytics(
            metric_type=arguments["metric_type"],
            filters=arguments.get("filters") or {},
            group_by=arguments.get("group_by") or [],
            time_range=arguments.get("time_range", "7d"),
        )
    except TypeError:
        # Service signature may use a different arg name
        result = await svc.query_analytics(arguments)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"result": result}


async def _handle_get_dashboard_metrics(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.analytics import AnalyticsService
        from core.tenant_context import get_current_tenant_id
    except Exception:
        return _enterprise_pending("analytics")

    svc = AnalyticsService(db)
    tenant_id = get_current_tenant_id() or "default"
    result = await svc.get_dashboard_metrics(tenant_id=tenant_id)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"metrics": result}


async def _handle_get_metric_trends(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.analytics import AnalyticsService
        from core.tenant_context import get_current_tenant_id
    except Exception:
        return _enterprise_pending("analytics")

    svc = AnalyticsService(db)
    tenant_id = get_current_tenant_id() or "default"
    method = (
        getattr(svc, "get_trend_data", None) or getattr(svc, "get_trends", None)
    )
    if not method:
        return _enterprise_pending("analytics.trends")
    result = await method(
        metric_type=arguments["metric_type"],
        time_range=arguments.get("time_range", "30d"),
        tenant_id=tenant_id,
    )
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"trends": result}


# ---- Audit ----


async def _handle_get_audit_logs(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from datetime import datetime

        from aictrlnet_enterprise.services.audit_service import AuditService
        from core.tenant_context import get_current_tenant_id
    except Exception:
        return _enterprise_pending("audit")

    svc = AuditService(db)
    tenant_id = get_current_tenant_id() or "default"

    def _parse(d):
        if not d:
            return None
        try:
            return datetime.fromisoformat(d.replace("Z", "+00:00"))
        except ValueError:
            return None

    result = await svc.get_audit_logs(
        resource_type=arguments.get("resource_type"),
        resource_id=arguments.get("resource_id"),
        action=arguments.get("action"),
        severity=arguments.get("severity"),
        success=arguments.get("success"),
        user_id=arguments.get("user_id"),
        start_date=_parse(arguments.get("start_date")),
        end_date=_parse(arguments.get("end_date")),
        limit=min(int(arguments.get("limit", 100)), 1000),
        offset=int(arguments.get("offset", 0)),
        order_by=arguments.get("order_by", "timestamp"),
        order_direction=arguments.get("order_direction", "desc"),
        tenant_id=tenant_id,
    )
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {"logs": result if isinstance(result, list) else [result]}


async def _handle_get_audit_summary(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.audit_service import AuditService
        from core.tenant_context import get_current_tenant_id
    except Exception:
        return _enterprise_pending("audit")

    svc = AuditService(db)
    tenant_id = get_current_tenant_id() or "default"
    days = max(1, min(int(arguments.get("days", 7)), 90))
    try:
        result = await svc.get_audit_summary(days=days, tenant_id=tenant_id)
    except TypeError:
        result = await svc.get_audit_summary(days)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"summary": result}


# ---- Compliance ----


async def _handle_run_compliance_check(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.compliance import ComplianceService
        from core.tenant_context import get_current_tenant_id
    except Exception:
        return _enterprise_pending("compliance")

    svc = ComplianceService(db)
    tenant_id = get_current_tenant_id() or "default"
    try:
        result = await svc.run_compliance_check(
            standards=arguments.get("standards") or [],
            scope=arguments.get("scope") or {},
            tenant_id=tenant_id,
        )
    except TypeError:
        result = await svc.run_compliance_check(
            arguments.get("standards") or [], arguments.get("scope") or {}
        )
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"result": result}


async def _handle_list_compliance_standards(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.compliance import ComplianceService
    except Exception:
        return _enterprise_pending("compliance")

    svc = ComplianceService(db)
    standards = await svc.get_compliance_standards()
    return {
        "standards": [
            (s.model_dump() if hasattr(s, "model_dump")
             else s.dict() if hasattr(s, "dict")
             else s)
            for s in (standards if isinstance(standards, list) else [standards])
        ]
    }


async def _handle_get_enterprise_risk_assessment(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from core.tenant_context import get_current_tenant_id

        # The enterprise endpoint lives in ai_governance_enterprise — reach
        # it through the service underneath.
        from aictrlnet_enterprise.services.ai_governance_enterprise import (  # type: ignore
            EnterpriseAIGovernanceService,
        )
    except Exception:
        # Fall back to the business-tier risk aggregation so Enterprise
        # callers still get a value even if the enterprise service isn't
        # yet factored out.
        try:
            _ensure_business_sys_path()
            from aictrlnet_business.services.risk_assessment_service import (  # type: ignore
                RiskAssessmentService,
            )

            svc = RiskAssessmentService(db)
            result = await svc.get_summary(user_id=user_id)
            if hasattr(result, "model_dump"):
                return result.model_dump()
            if hasattr(result, "dict"):
                return result.dict()
            return {"summary": result}
        except Exception:
            return _enterprise_pending("risk_assessment")

    svc = EnterpriseAIGovernanceService(db)
    tenant_id = get_current_tenant_id() or "default"
    method = (
        getattr(svc, "get_risk_summary", None)
        or getattr(svc, "get_enterprise_risk_summary", None)
    )
    if not method:
        return _enterprise_pending("risk_assessment")
    result = await method(tenant_id=tenant_id)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"summary": result}


# ---- Organizations + Tenants ----


async def _handle_list_organizations(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from sqlalchemy import select

        from aictrlnet_enterprise.models.organization import Organization  # type: ignore
    except Exception:
        return _enterprise_pending("organizations")

    rows = (await db.execute(select(Organization))).scalars().all()
    return {
        "organizations": [
            {
                "id": str(getattr(o, "id", "")),
                "name": getattr(o, "name", None),
                "slug": getattr(o, "slug", None),
                "created_at": str(getattr(o, "created_at", "")),
            }
            for o in rows
        ],
        "count": len(rows),
    }


async def _handle_list_tenants(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from sqlalchemy import select

        from aictrlnet_enterprise.models.tenant import Tenant  # type: ignore
    except Exception:
        try:
            from sqlalchemy import select

            # Community tenants table (present in all editions)
            from models.community_complete import Tenant  # type: ignore
        except Exception:
            return _enterprise_pending("tenants")

    query = (
        select(Tenant)
        .limit(min(int(arguments.get("limit", 100)), 500))
        .offset(int(arguments.get("offset", 0)))
    )
    rows = (await db.execute(query)).scalars().all()
    return {
        "tenants": [
            {
                "id": str(getattr(t, "id", "")),
                "name": getattr(t, "name", None),
                "edition": getattr(t, "edition", None),
                "created_at": str(getattr(t, "created_at", "")),
            }
            for t in rows
        ],
        "count": len(rows),
    }


# ---- Federated knowledge + Cross-tenant ----


async def _handle_federated_knowledge_query(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        # Knowledge federation lives in knowledge_enterprise endpoint; try
        # to find its service backing.
        from aictrlnet_enterprise.services.federated_knowledge_service import (  # type: ignore
            FederatedKnowledgeService,
        )
    except Exception:
        # Fallback: call the community knowledge retrieval service
        # scoped to the caller's tenant only.
        try:
            from services.knowledge.knowledge_retrieval_service import (
                KnowledgeRetrievalService,
            )

            svc = KnowledgeRetrievalService(db)
            results = await svc.find_relevant_knowledge(
                query=arguments["query"],
                context=None,
                limit=min(int(arguments.get("limit", 10)), 100),
            )
            return {
                "results": [
                    (r.model_dump() if hasattr(r, "model_dump")
                     else r.dict() if hasattr(r, "dict")
                     else {"content": str(r)})
                    for r in results
                ],
                "federated": False,
                "note": "Federated knowledge service not available — scoped to caller's tenant only",
            }
        except Exception:
            return _enterprise_pending("federated_knowledge")

    svc = FederatedKnowledgeService(db)
    method = (
        getattr(svc, "search", None)
        or getattr(svc, "query", None)
        or getattr(svc, "find", None)
    )
    if not method:
        return _enterprise_pending("federated_knowledge")
    results = await method(
        query=arguments["query"],
        tenant_ids=arguments.get("tenant_ids"),
        limit=min(int(arguments.get("limit", 10)), 100),
        user_id=user_id,
    )
    return {
        "results": results if isinstance(results, list) else [results],
        "federated": True,
    }


async def _handle_get_cross_tenant_insights(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.cross_tenant_service import (  # type: ignore
            CrossTenantService,
        )
    except Exception:
        return _enterprise_pending("cross_tenant")

    svc = CrossTenantService(db)
    metric_type = arguments.get("metric_type", "summary")
    method_name = f"get_cross_tenant_{metric_type}"
    method = (
        getattr(svc, method_name, None)
        or getattr(svc, "get_summary", None)
        or getattr(svc, "get_insights", None)
    )
    if not method:
        return _enterprise_pending("cross_tenant")
    result = await method(
        time_range=arguments.get("time_range", "30d"),
        user_id=user_id,
    )
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"insights": result}


# ---- Fleet Management ----


async def _handle_list_fleet_agents(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.fleet_management import FleetManagementService  # type: ignore
    except Exception:
        return _enterprise_pending("fleet_management")

    svc = FleetManagementService(db)
    method = (
        getattr(svc, "list_runtimes", None)
        or getattr(svc, "list_fleet_agents", None)
        or getattr(svc, "list_agents", None)
    )
    if not method:
        return _enterprise_pending("fleet_management")
    try:
        result = await method(
            status=arguments.get("status"),
            limit=min(int(arguments.get("limit", 100)), 500),
            offset=int(arguments.get("offset", 0)),
        )
    except TypeError:
        result = await method()
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    if isinstance(result, list):
        return {"fleet_agents": result, "count": len(result)}
    return {"fleet_agents": result}


async def _handle_get_fleet_autonomy_summary(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.fleet_management import FleetManagementService  # type: ignore
    except Exception:
        return _enterprise_pending("fleet_management")

    svc = FleetManagementService(db)
    method = (
        getattr(svc, "get_fleet_summary", None)
        or getattr(svc, "get_autonomy_summary", None)
        or getattr(svc, "get_fleet_risk_overview", None)
    )
    if not method:
        return _enterprise_pending("fleet_management")
    try:
        result = await method()
    except TypeError:
        result = await method(user_id=user_id)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"summary": result}


# ---- License Management ----


async def _handle_get_license_status(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.license_management import (  # type: ignore
            LicenseManagementService,
        )
    except Exception:
        return _enterprise_pending("license_management")

    svc = LicenseManagementService(db)
    method = (
        getattr(svc, "get_subscription_status", None)
        or getattr(svc, "get_status", None)
        or getattr(svc, "get_license", None)
    )
    if not method:
        return _enterprise_pending("license_management")
    try:
        result = await method(user_id=user_id)
    except TypeError:
        result = await method()
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"license": result}


async def _handle_list_license_entitlements(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.license_management import (  # type: ignore
            LicenseManagementService,
        )
    except Exception:
        return _enterprise_pending("license_management")

    svc = LicenseManagementService(db)
    method = (
        getattr(svc, "list_entitlements", None)
        or getattr(svc, "get_entitlements", None)
        or getattr(svc, "get_features", None)
    )
    if not method:
        return _enterprise_pending("license_management")
    try:
        result = await method(user_id=user_id)
    except TypeError:
        result = await method()
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    if isinstance(result, list):
        return {"entitlements": result, "count": len(result)}
    return {"entitlements": result}


# ---------------------------------------------------------------------------
# Wave 7 B1.1 — MCP Client (federate external MCP servers)
# ---------------------------------------------------------------------------


async def _handle_register_mcp_server(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Persist an external MCP server config to ``mcp_servers`` table."""
    import time
    import uuid

    from sqlalchemy import text

    from core.tenant_context import get_current_tenant_id

    server_id = str(uuid.uuid4())
    tenant = get_current_tenant_id() or "default"
    transport = arguments.get("transport", "http")
    now = time.time()

    try:
        await db.execute(
            text(
                """
                INSERT INTO mcp_servers
                  (id, name, url, api_key, service_type, status,
                   last_checked, created_at, updated_at,
                   transport_type, command, args)
                VALUES (:id, :name, :url, :api_key, 'external', 'pending',
                        :now, :now, :now, :transport, :command, :args)
                """
            ),
            {
                "id": server_id,
                "name": arguments["name"],
                "url": arguments.get("url"),
                "api_key": arguments.get("api_key"),
                "now": now,
                "transport": transport,
                "command": arguments.get("command"),
                "args": ",".join(arguments.get("args") or []) or None,
            },
        )
        await db.commit()
    except Exception as e:
        raise ToolExecutionError(f"register_mcp_server failed: {e}") from e

    return {
        "server_id": server_id,
        "name": arguments["name"],
        "transport": transport,
        "status": "pending",
        "message": "Server registered. Call discover_mcp_server_tools to verify connectivity.",
    }


async def _handle_discover_mcp_server_tools(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Connect to a registered external MCP server and enumerate its tools."""
    from sqlalchemy import text

    row = (
        await db.execute(
            text("SELECT id, name, url, transport_type, command, args, api_key FROM mcp_servers WHERE id = :id"),
            {"id": arguments["server_id"]},
        )
    ).first()
    if row is None:
        raise ToolExecutionError(f"MCP server {arguments['server_id']} not registered")

    try:
        from adapters.implementations.ai.mcp_client_adapter import (
            MCPConnection,
            MCPServerConfig,
            MCPTransportType,
        )
    except Exception as e:
        return {"status": "feature_pending", "available": False, "message": str(e)}

    config = MCPServerConfig(
        name=row[1],
        transport=MCPTransportType(row[3]),
        url=row[2],
        command=row[4],
        args=(row[5] or "").split(",") if row[5] else None,
        api_key=row[6],
    )
    conn = MCPConnection(config)
    try:
        ok = await conn.connect()
        if not ok:
            return {"status": "connect_failed", "server_id": arguments["server_id"]}
        tools_list = getattr(conn, "tools", {}) or {}
        return {
            "server_id": arguments["server_id"],
            "tools": [
                {
                    "name": getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None),
                    "description": getattr(t, "description", None) or (t.get("description") if isinstance(t, dict) else None),
                }
                for t in (tools_list.values() if isinstance(tools_list, dict) else tools_list)
            ],
            "count": len(tools_list),
        }
    finally:
        disconnect = getattr(conn, "disconnect", None)
        if callable(disconnect):
            try:
                await disconnect()
            except Exception:
                pass


async def _handle_invoke_external_mcp_tool(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Call a tool on a registered external MCP server."""
    from sqlalchemy import text

    row = (
        await db.execute(
            text("SELECT id, name, url, transport_type, command, args, api_key FROM mcp_servers WHERE id = :id"),
            {"id": arguments["server_id"]},
        )
    ).first()
    if row is None:
        raise ToolExecutionError(f"MCP server {arguments['server_id']} not registered")

    try:
        from adapters.implementations.ai.mcp_client_adapter import (
            MCPConnection,
            MCPServerConfig,
            MCPTransportType,
        )
    except Exception as e:
        return {"status": "feature_pending", "available": False, "message": str(e)}

    config = MCPServerConfig(
        name=row[1],
        transport=MCPTransportType(row[3]),
        url=row[2],
        command=row[4],
        args=(row[5] or "").split(",") if row[5] else None,
        api_key=row[6],
    )
    conn = MCPConnection(config)
    try:
        if not await conn.connect():
            raise ToolExecutionError("Connection to external MCP server failed")
        try:
            result = await conn.call_tool(
                arguments["tool_name"], arguments.get("arguments") or {}
            )
        except Exception as e:
            raise ToolExecutionError(f"External tool error: {e}") from e
        return {
            "server_id": arguments["server_id"],
            "tool_name": arguments["tool_name"],
            "result": result,
        }
    finally:
        disconnect = getattr(conn, "disconnect", None)
        if callable(disconnect):
            try:
                await disconnect()
            except Exception:
                pass


async def _handle_list_registered_mcp_servers(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text

    rows = (
        await db.execute(
            text(
                "SELECT id, name, transport_type, url, status, last_checked "
                "FROM mcp_servers ORDER BY name"
            )
        )
    ).all()
    return {
        "servers": [
            {
                "id": r[0],
                "name": r[1],
                "transport": r[2],
                "url": r[3],
                "status": r[4],
                "last_checked": r[5],
            }
            for r in rows
        ],
        "count": len(rows),
    }


async def _handle_unregister_mcp_server(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text

    try:
        await db.execute(
            text("DELETE FROM mcp_servers WHERE id = :id"),
            {"id": arguments["server_id"]},
        )
        await db.commit()
    except Exception as e:
        raise ToolExecutionError(f"unregister_mcp_server failed: {e}") from e
    return {"server_id": arguments["server_id"], "unregistered": True}


# ---------------------------------------------------------------------------
# Wave 7 B1.2 — Credential Management
# ---------------------------------------------------------------------------
#
# Uses CredentialService with whatever backend is configured. Backends
# that don't support list/rotate/validate degrade to feature_pending
# rather than crashing. Secrets are never returned from list_ and
# get_ defaults to reveal=false.


def _credential_service(db):
    try:
        from core.services.credential_service import CredentialService  # type: ignore
    except Exception:
        return None
    try:
        return CredentialService(db)
    except Exception:
        try:
            # Older constructors
            return CredentialService()
        except Exception:
            return None


async def _handle_create_credential(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _credential_service(db)
    if svc is None:
        return {"status": "feature_pending", "available": False}
    store = getattr(svc, "store_credentials", None) or getattr(svc, "create_credentials", None)
    if store is None:
        return {"status": "feature_pending", "available": False}
    ok = await store(
        arguments["platform"], arguments["credential_id"], arguments["credentials"]
    )
    return {
        "platform": arguments["platform"],
        "credential_id": arguments["credential_id"],
        "stored": bool(ok),
    }


async def _handle_list_credentials(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _credential_service(db)
    if svc is None:
        return {"credentials": [], "available": False, "status": "feature_pending"}
    list_method = getattr(svc, "list_credentials", None)
    if list_method is None:
        # Graceful fallback — some backends don't implement list
        return {
            "credentials": [],
            "available": False,
            "status": "feature_pending",
            "message": "Credential backend does not support listing; use get_credential per id.",
        }
    items = await list_method(platform=arguments.get("platform"))
    return {
        "credentials": [
            {k: v for k, v in (item.items() if isinstance(item, dict) else {}) if k not in ("value", "secret", "credentials")}
            for item in (items or [])
        ],
        "count": len(items) if items else 0,
    }


async def _handle_get_credential(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _credential_service(db)
    if svc is None:
        return {"status": "feature_pending", "available": False}
    fetch = getattr(svc, "get_credentials", None)
    if fetch is None:
        return {"status": "feature_pending", "available": False}
    try:
        payload = await fetch(arguments["platform"], arguments["credential_id"])
    except Exception as e:
        raise ToolExecutionError(f"get_credential failed: {e}") from e
    if payload is None:
        raise ToolExecutionError(
            f"Credential {arguments['platform']}/{arguments['credential_id']} not found"
        )

    # Audit-friendly default: return metadata only unless explicitly asked.
    if not arguments.get("reveal"):
        if isinstance(payload, dict):
            return {
                "platform": arguments["platform"],
                "credential_id": arguments["credential_id"],
                "fields_available": sorted(list(payload.keys())),
                "revealed": False,
            }
    return {
        "platform": arguments["platform"],
        "credential_id": arguments["credential_id"],
        "credentials": payload,
        "revealed": True,
    }


async def _handle_delete_credential(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _credential_service(db)
    if svc is None:
        return {"status": "feature_pending", "available": False}
    delete = getattr(svc, "delete_credentials", None)
    if delete is None:
        return {"status": "feature_pending", "available": False}
    ok = await delete(arguments["platform"], arguments["credential_id"])
    return {
        "platform": arguments["platform"],
        "credential_id": arguments["credential_id"],
        "deleted": bool(ok),
    }


async def _handle_rotate_credential(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Rotate: delete + store. Increments rotation_count if the
    backend exposes a ``rotate_credentials`` method; otherwise falls
    back to delete+store."""
    svc = _credential_service(db)
    if svc is None:
        return {"status": "feature_pending", "available": False}

    rotate = getattr(svc, "rotate_credentials", None)
    if rotate:
        result = await rotate(
            arguments["platform"],
            arguments["credential_id"],
            arguments["new_credentials"],
        )
        return {
            "platform": arguments["platform"],
            "credential_id": arguments["credential_id"],
            "rotated": bool(result),
        }

    # Fallback: delete + store
    delete = getattr(svc, "delete_credentials", None)
    store = getattr(svc, "store_credentials", None) or getattr(svc, "create_credentials", None)
    if not delete or not store:
        return {"status": "feature_pending", "available": False}
    await delete(arguments["platform"], arguments["credential_id"])
    ok = await store(
        arguments["platform"],
        arguments["credential_id"],
        arguments["new_credentials"],
    )
    return {
        "platform": arguments["platform"],
        "credential_id": arguments["credential_id"],
        "rotated": bool(ok),
        "note": "Rotated via delete+store (backend lacks native rotate_credentials).",
    }


async def _handle_validate_credential(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _credential_service(db)
    if svc is None:
        return {"status": "feature_pending", "available": False}
    validate = getattr(svc, "validate_credentials", None)
    if validate is None:
        return {
            "valid": None,
            "available": False,
            "status": "feature_pending",
            "message": "Credential backend does not support validation yet.",
        }
    try:
        result = await validate(arguments["platform"], arguments["credential_id"])
    except Exception as e:
        return {"valid": False, "error": str(e)}
    return {
        "platform": arguments["platform"],
        "credential_id": arguments["credential_id"],
        "valid": bool(result),
    }


# ---------------------------------------------------------------------------
# Wave 7 B1.3 — Personal Agent Hub
# ---------------------------------------------------------------------------


def _personal_agent_service(db):
    try:
        from services.personal_agent_service import PersonalAgentService  # type: ignore
    except Exception:
        return None
    try:
        return PersonalAgentService(db)
    except Exception:
        return None


def _pa_dump(obj) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


async def _handle_get_personal_agent_config(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _personal_agent_service(db)
    if svc is None:
        return {"status": "feature_pending", "available": False}
    config = await svc.get_or_create_config(user_id)
    return {"config": _pa_dump(config)}


async def _handle_update_personal_agent_config(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _personal_agent_service(db)
    if svc is None:
        return {"status": "feature_pending", "available": False}
    method = getattr(svc, "update_config", None)
    if not method:
        return {"status": "feature_pending", "available": False}
    config = await method(user_id, arguments)
    return {"config": _pa_dump(config), "updated": True}


async def _handle_get_personal_agent_activity(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _personal_agent_service(db)
    if svc is None:
        return {"activity": [], "available": False, "status": "feature_pending"}
    feed = getattr(svc, "get_activity_feed", None)
    if not feed:
        return {"activity": [], "available": False}
    result = await feed(user_id, limit=int(arguments.get("limit", 50)))
    return {"activity": _pa_dump(result)}


async def _handle_connect_external_agent(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Attach an external agent (BYOA) to the personal-agent config.

    Uses update_config under the hood — external_agents is a list on
    the personal agent config.
    """
    svc = _personal_agent_service(db)
    if svc is None:
        return {"status": "feature_pending", "available": False}
    # Read current config, append external agent, write back.
    config = await svc.get_or_create_config(user_id)
    current = _pa_dump(config) or {}
    externals = current.get("external_agents") or []
    externals.append({
        "agent_id": arguments["agent_id"],
        "agent_type": arguments["agent_type"],
        "endpoint": arguments.get("endpoint"),
        "credentials_ref": arguments.get("credentials_ref"),
    })
    update = getattr(svc, "update_config", None)
    if update:
        result = await update(user_id, {"external_agents": externals})
        return {"external_agents": externals, "config": _pa_dump(result)}
    return {
        "status": "feature_pending",
        "available": False,
        "message": "Personal agent service lacks update_config.",
    }


async def _handle_create_personal_workflow(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _personal_agent_service(db)
    if svc is None:
        return {"status": "feature_pending", "available": False}
    add = getattr(svc, "add_workflow", None)
    if not add:
        return {"status": "feature_pending", "available": False}
    result = await add(
        user_id=user_id,
        workflow_id=arguments["workflow_id"],
        role=arguments.get("role", "primary"),
    )
    return {
        "workflow_id": arguments["workflow_id"],
        "role": arguments.get("role", "primary"),
        "result": _pa_dump(result),
    }


async def _handle_promote_personal_workflow(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Feature_pending — org-promotion service not yet shipped."""
    return {
        "status": "feature_pending",
        "available": False,
        "message": (
            "Personal-to-org workflow promotion service not yet factored "
            "out. Tracked under Wave 7 B1.3 follow-up."
        ),
        "workflow_id": arguments["workflow_id"],
    }


# ---------------------------------------------------------------------------
# Wave 7 B1.4 — Marketplace (publish / compose / sync — feature_pending)
# ---------------------------------------------------------------------------


def _marketplace_service(db):
    try:
        from services.marketplace_service import MarketplaceService  # type: ignore
    except Exception:
        return None
    try:
        return MarketplaceService(db)
    except Exception:
        return None


async def _handle_list_org_marketplace_items(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _marketplace_service(db)
    if svc is None:
        return {"items": [], "available": False, "status": "feature_pending"}
    browse = getattr(svc, "browse", None)
    if not browse:
        return {"items": [], "available": False}
    result = await browse(
        category=arguments.get("category"),
        limit=int(arguments.get("limit", 50)),
        offset=int(arguments.get("offset", 0)),
    )
    return {"items": _pa_dump(result)}


async def _handle_publish_to_org_marketplace(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _marketplace_service(db)
    publish = getattr(svc, "publish", None) if svc else None
    if publish is None:
        return {
            "status": "feature_pending",
            "available": False,
            "message": (
                "MarketplaceService.publish not yet implemented "
                "(Feature #19 P4 Planned)."
            ),
            "item_id": arguments.get("item_id"),
        }
    result = await publish(
        user_id=user_id,
        item_type=arguments["item_type"],
        item_id=arguments["item_id"],
        title=arguments.get("title"),
        description=arguments.get("description"),
        visibility=arguments.get("visibility", "org"),
    )
    return {"published": True, "result": _pa_dump(result)}


async def _handle_compose_marketplace_items(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _marketplace_service(db)
    compose = getattr(svc, "compose", None) if svc else None
    if compose is None:
        return {
            "status": "feature_pending",
            "available": False,
            "message": "Composition engine not yet shipped.",
            "item_ids": arguments["item_ids"],
        }
    result = await compose(
        user_id=user_id,
        item_ids=arguments["item_ids"],
        name=arguments["name"],
        description=arguments.get("description"),
    )
    return {"composed": True, "result": _pa_dump(result)}


async def _handle_sync_public_marketplace_updates(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _marketplace_service(db)
    sync = getattr(svc, "sync_public_updates", None) if svc else None
    if sync is None:
        return {
            "status": "feature_pending",
            "available": False,
            "message": "Public registry sync not yet shipped.",
        }
    result = await sync(user_id=user_id)
    return {"synced": True, "result": _pa_dump(result)}


# ---------------------------------------------------------------------------
# Wave 7 B1.5 — External platform native execution
# ---------------------------------------------------------------------------


async def _execute_platform(
    adapter_module: str,
    adapter_class: str,
    db: AsyncSession,
    user_id: str,
    credential_id: str,
    target_id: str,
    input_data: Optional[Dict[str, Any]] = None,
    id_field: str = "workflow_id",
) -> Dict[str, Any]:
    """Shared platform-execution path — loads credential, instantiates
    adapter, calls execute."""
    # Load credential via CredentialService
    try:
        from core.services.credential_service import CredentialService
        cred_svc = CredentialService(db) if hasattr(CredentialService, "__init__") else CredentialService()
        # Platform name is the module slug before _adapter
        platform = adapter_module.split(".")[-1].replace("_adapter", "")
        creds = await cred_svc.get_credentials(platform, credential_id)
    except Exception as e:
        raise ToolExecutionError(f"Could not load credential {credential_id}: {e}") from e

    # Instantiate adapter
    try:
        import importlib
        mod = importlib.import_module(adapter_module)
        AdapterCls = getattr(mod, adapter_class)
        adapter = AdapterCls(credentials=creds)
    except Exception as e:
        return {"status": "feature_pending", "available": False, "message": str(e)}

    execute = getattr(adapter, "execute_workflow", None) or getattr(adapter, "execute", None)
    if not execute:
        return {"status": "feature_pending", "available": False}

    try:
        result = await execute(target_id, input_data or {})
    except Exception as e:
        raise ToolExecutionError(f"Platform execution failed: {e}") from e

    return {
        id_field: target_id,
        "result": _pa_dump(result) if hasattr(result, "model_dump") or isinstance(result, dict) else str(result),
    }


async def _handle_execute_n8n_workflow(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    return await _execute_platform(
        "services.platform_adapters.n8n_adapter", "N8nAdapter",
        db, user_id,
        arguments["credential_id"], arguments["workflow_id"],
        arguments.get("input_data"), id_field="workflow_id",
    )


async def _handle_execute_zapier_zap(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    return await _execute_platform(
        "services.platform_adapters.zapier_adapter", "ZapierAdapter",
        db, user_id,
        arguments["credential_id"], arguments["zap_id"],
        arguments.get("input_data"), id_field="zap_id",
    )


async def _handle_execute_make_scenario(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    return await _execute_platform(
        "services.platform_adapters.make_adapter", "MakeAdapter",
        db, user_id,
        arguments["credential_id"], arguments["scenario_id"],
        arguments.get("input_data"), id_field="scenario_id",
    )


async def _handle_execute_power_automate_flow(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    return await _execute_platform(
        "services.platform_adapters.power_automate_adapter", "PowerAutomateAdapter",
        db, user_id,
        arguments["credential_id"], arguments["flow_id"],
        arguments.get("input_data"), id_field="flow_id",
    )


# ---------------------------------------------------------------------------
# Wave 7 B1.6 — OpenClaw + A2A delegation (mostly feature_pending)
# ---------------------------------------------------------------------------


async def _handle_evaluate_runtime_action(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    # Try dedicated runtime_gateway_service first, then fall back to
    # agent_runtime_bridge, then feature_pending.
    for module_path, class_name in (
        ("aictrlnet_enterprise.services.runtime_gateway_service", "RuntimeGatewayService"),
        ("aictrlnet_business.services.agent_runtime_bridge", "AgentRuntimeBridge"),
    ):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            Cls = getattr(mod, class_name, None)
            if Cls is None:
                continue
            svc = Cls(db) if "db" in Cls.__init__.__code__.co_varnames else Cls()
            method = (
                getattr(svc, "evaluate_action", None)
                or getattr(svc, "pre_action_evaluate", None)
                or getattr(svc, "evaluate_runtime_action", None)
            )
            if method:
                result = await method(
                    agent_id=arguments["agent_id"],
                    action=arguments["action"],
                    context=arguments.get("context") or {},
                )
                return {"result": _pa_dump(result)}
        except Exception:
            continue
    return {
        "status": "feature_pending",
        "available": False,
        "message": (
            "OpenClaw RuntimeGatewayService not yet factored out. "
            "Tools for runtime-action evaluation will light up when the "
            "service lands."
        ),
    }


async def _handle_get_delegation_chain(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    for module_path, class_name in (
        ("aictrlnet_enterprise.services.runtime_gateway_service", "RuntimeGatewayService"),
        ("aictrlnet_business.services.agent_runtime_bridge", "AgentRuntimeBridge"),
    ):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            Cls = getattr(mod, class_name, None)
            if Cls is None:
                continue
            svc = Cls(db) if "db" in Cls.__init__.__code__.co_varnames else Cls()
            method = (
                getattr(svc, "get_delegation_chain", None)
                or getattr(svc, "get_chain", None)
            )
            if method:
                result = await method(invocation_id=arguments["invocation_id"])
                return {"chain": _pa_dump(result)}
        except Exception:
            continue
    return {
        "status": "feature_pending",
        "available": False,
        "chain": None,
    }


async def _handle_list_a2a_agents(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        # A2A agent registry — try standard path
        from aictrlnet_business.services.a2a_service import A2AService  # type: ignore
        svc = A2AService(db)
        method = getattr(svc, "list_agents", None) or getattr(svc, "discover_agents", None)
        if method:
            result = await method(capabilities=arguments.get("capabilities"))
            return {"agents": _pa_dump(result)}
    except Exception:
        pass
    return {"agents": [], "available": False, "status": "feature_pending"}


async def _handle_register_runtime_webhook(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    return {
        "status": "feature_pending",
        "available": False,
        "message": "Runtime webhook registration pending RuntimeGatewayService.",
        "name": arguments.get("name"),
    }


async def _handle_list_runtime_webhooks(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    return {"webhooks": [], "available": False, "status": "feature_pending"}


# ---------------------------------------------------------------------------
# Wave 7 B1.7 — Pods + Swarm Intelligence
# ---------------------------------------------------------------------------


def _swarm_service(db):
    _ensure_business_sys_path()
    for path in (
        "aictrlnet_business.services.swarm_intelligence_service",
        "aictrlnet_business.services.pod_service",
    ):
        try:
            import importlib
            mod = importlib.import_module(path)
            for cls_name in ("SwarmIntelligenceService", "PodService"):
                Cls = getattr(mod, cls_name, None)
                if Cls:
                    try:
                        return Cls(db)
                    except TypeError:
                        return Cls()
        except Exception:
            continue
    return None


async def _handle_form_pod(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _swarm_service(db)
    if not svc:
        return {"status": "feature_pending", "available": False}
    method = getattr(svc, "form_pod", None) or getattr(svc, "create_pod", None)
    if not method:
        return {"status": "feature_pending", "available": False}
    result = await method(
        name=arguments.get("name"),
        agent_ids=arguments["agent_ids"],
        objective=arguments["objective"],
        collaboration_contract=arguments.get("collaboration_contract"),
        user_id=user_id,
    )
    return {"pod": _pa_dump(result)}


async def _handle_list_pods(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _swarm_service(db)
    if not svc:
        return {"pods": [], "available": False, "status": "feature_pending"}
    method = getattr(svc, "list_pods", None)
    if not method:
        return {"pods": [], "available": False}
    result = await method(status=arguments.get("status"), limit=arguments.get("limit", 50))
    return {"pods": _pa_dump(result)}


async def _handle_get_pod_status(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _swarm_service(db)
    if not svc:
        return {"status": "feature_pending", "available": False}
    method = getattr(svc, "get_pod_status", None) or getattr(svc, "get_pod", None)
    if not method:
        return {"status": "feature_pending", "available": False}
    result = await method(pod_id=arguments["pod_id"])
    return {"pod": _pa_dump(result)}


async def _handle_dispatch_swarm(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    svc = _swarm_service(db)
    if not svc:
        return {"status": "feature_pending", "available": False}
    method = getattr(svc, "dispatch_swarm", None) or getattr(svc, "dispatch", None)
    if not method:
        return {"status": "feature_pending", "available": False}
    result = await method(
        agent_ids=arguments["agent_ids"],
        task=arguments["task"],
        aggregation_strategy=arguments.get("aggregation_strategy", "best_of_n"),
        user_id=user_id,
    )
    return {"swarm_result": _pa_dump(result)}


# ---------------------------------------------------------------------------
# Wave 7 B1.8 — Framework cascading + ML matching
# ---------------------------------------------------------------------------


async def _handle_get_framework_cascade(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    # Static default cascade per features doc; tenant override would
    # be read from an LLMRegistry config row if that's shipped.
    return {
        "priority": [
            "langchain",
            "autogpt",
            "crewai",
            "autogen",
            "semantic_kernel",
        ],
        "source": "default",
        "note": (
            "Default cascade per features doc §AI Framework Cascading. "
            "Tenant-level override via set_framework_priority is "
            "feature_pending until LLMRegistry stores per-tenant priority."
        ),
    }


async def _handle_set_framework_priority(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    return {
        "status": "feature_pending",
        "available": False,
        "message": (
            "Tenant-level framework priority override not yet shipped. "
            "Edit framework order via server env vars for now."
        ),
        "priority": arguments["priority"],
    }


async def _handle_get_execution_framework_trace(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    # Look up execution record; frameworks-used would be in metadata
    try:
        from sqlalchemy import text
        row = (
            await db.execute(
                text(
                    "SELECT id, metadata FROM workflow_executions "
                    "WHERE id = :id LIMIT 1"
                ),
                {"id": arguments["execution_id"]},
            )
        ).first()
        if row:
            meta = row[1] if isinstance(row[1], dict) else (
                __import__("json").loads(row[1]) if isinstance(row[1], str) else {}
            )
            return {
                "execution_id": arguments["execution_id"],
                "framework_used": meta.get("framework_used") if isinstance(meta, dict) else None,
                "cascade_attempts": meta.get("cascade_attempts", []) if isinstance(meta, dict) else [],
            }
    except Exception:
        pass
    return {
        "execution_id": arguments["execution_id"],
        "framework_used": None,
        "cascade_attempts": [],
        "status": "feature_pending",
    }


async def _handle_match_agents_to_task(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.intelligent_matching_utils import match_agents  # type: ignore
    except Exception:
        try:
            from aictrlnet_business.services import intelligent_matching_utils as imu  # type: ignore
            match_agents = getattr(imu, "match_agents", None) or getattr(imu, "match_agents_to_task", None)
        except Exception:
            match_agents = None
    if match_agents is None:
        return {
            "matches": [],
            "available": False,
            "status": "feature_pending",
            "message": "intelligent_matching_utils lacks an exported match_agents function.",
        }
    try:
        results = await match_agents(
            task=arguments["task"], top_k=int(arguments.get("top_k", 3)), db=db
        ) if _is_coroutine_fn(match_agents) else match_agents(
            task=arguments["task"], top_k=int(arguments.get("top_k", 3))
        )
    except Exception as e:
        return {"matches": [], "error": str(e)}
    return {"matches": _pa_dump(results), "task": arguments["task"]}


def _is_coroutine_fn(fn) -> bool:
    import asyncio
    return asyncio.iscoroutinefunction(fn)


# ---------------------------------------------------------------------------
# Wave 7 B1.9 — Activity Timeline + Operations Status
# ---------------------------------------------------------------------------


async def _handle_get_activity_timeline(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    """Aggregate recent events across workflows / executions / approvals
    / audit. Simple union of the last N entries across those tables."""
    from datetime import datetime

    from sqlalchemy import text

    from core.tenant_context import get_current_tenant_id

    tenant_id = get_current_tenant_id() or "default"
    limit = min(int(arguments.get("limit", 100)), 500)
    event_types = arguments.get("event_types") or [
        "workflow_execution",
        "approval_decision",
        "audit_log",
    ]
    since_s = arguments.get("since")
    events: list = []

    if "workflow_execution" in event_types:
        try:
            rows = (
                await db.execute(
                    text(
                        "SELECT id, workflow_id, status, started_at "
                        "FROM workflow_executions "
                        "WHERE tenant_id = :t "
                        "ORDER BY started_at DESC LIMIT :l"
                    ),
                    {"t": tenant_id, "l": limit},
                )
            ).all()
            for r in rows:
                events.append({
                    "type": "workflow_execution",
                    "id": str(r[0]),
                    "workflow_id": str(r[1]) if r[1] else None,
                    "status": r[2],
                    "at": str(r[3]) if r[3] else None,
                })
        except Exception:
            pass

    # Sort combined events by timestamp (best-effort; strings OK since
    # ISO-8601 sorts correctly)
    events.sort(key=lambda e: e.get("at") or "", reverse=True)
    return {
        "events": events[:limit],
        "count": len(events[:limit]),
        "tenant_id": tenant_id,
    }


async def _handle_get_operations_status(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text

    from core.tenant_context import get_current_tenant_id

    tenant_id = get_current_tenant_id() or "default"
    counts = {}
    for label, sql in (
        ("workflows_active", "SELECT COUNT(*) FROM workflow_executions WHERE tenant_id = :t AND status IN ('running','pending')"),
        ("pods_active", "SELECT COUNT(*) FROM mcp_pods WHERE tenant_id = :t AND status = 'active'"),
        ("approvals_pending", "SELECT COUNT(*) FROM approval_requests WHERE tenant_id = :t AND status = 'pending'"),
    ):
        try:
            row = (await db.execute(text(sql), {"t": tenant_id})).first()
            counts[label] = int(row[0]) if row else 0
        except Exception:
            counts[label] = None
    return {"tenant_id": tenant_id, "counts": counts}


# ---------------------------------------------------------------------------
# Wave 7 Track B2 handlers (compact — direct SQL against Track C tables)
# ---------------------------------------------------------------------------


# ---- B2.1 Cost ----

async def _handle_get_cost_analytics(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.mcp_cost_service import MCPCostService  # type: ignore
        svc = MCPCostService(db)
        method = getattr(svc, "get_cost_analytics", None) or getattr(svc, "get_analytics", None)
        if method:
            return {"analytics": _pa_dump(await method(
                period=arguments.get("period", "month"),
                breakdown_by=arguments.get("breakdown_by"),
            ))}
    except Exception:
        pass
    return {"status": "feature_pending", "available": False}


async def _handle_get_platform_cost_estimate(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    try:
        from services.platform_cost_optimizer import PlatformCostOptimizer  # type: ignore
        opt = PlatformCostOptimizer(db) if "db" in PlatformCostOptimizer.__init__.__code__.co_varnames else PlatformCostOptimizer()
        method = getattr(opt, "estimate_cost", None) or getattr(opt, "get_estimate", None)
        if method:
            return {"estimates": _pa_dump(await method(
                workflow_id=arguments["workflow_id"],
                input_size=arguments.get("input_size"),
            ))}
    except Exception:
        pass
    return {"status": "feature_pending", "available": False, "workflow_id": arguments["workflow_id"]}


async def _handle_optimize_workflow_cost(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    try:
        from services.platform_cost_optimizer import PlatformCostOptimizer  # type: ignore
        opt = PlatformCostOptimizer(db) if "db" in PlatformCostOptimizer.__init__.__code__.co_varnames else PlatformCostOptimizer()
        method = getattr(opt, "suggest_optimizations", None) or getattr(opt, "optimize", None)
        if method:
            return {"suggestions": _pa_dump(await method(workflow_id=arguments["workflow_id"]))}
    except Exception:
        pass
    return {"status": "feature_pending", "available": False}


async def _handle_analyze_cost_trends(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_enterprise_sys_path()
    try:
        from aictrlnet_enterprise.services.analytics import AnalyticsService  # type: ignore
        svc = AnalyticsService(db)
        method = getattr(svc, "analyze_cost_trends", None) or getattr(svc, "get_cost_trends", None)
        if method:
            return {"trends": _pa_dump(await method(days=arguments.get("days", 90)))}
    except Exception:
        pass
    return {"status": "feature_pending", "available": False}


# ---- B2.2 SLA (direct SQL against mcp_slas / mcp_sla_violations) ----

async def _handle_create_sla(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    import uuid
    from sqlalchemy import text
    from core.tenant_context import get_current_tenant_id

    sla_id = str(uuid.uuid4())
    tenant_id = get_current_tenant_id() or "default"
    try:
        await db.execute(
            text(
                """
                INSERT INTO mcp_slas (id, tenant_id, name, description, resource_type,
                                      resource_id, metric, target_value, window_seconds,
                                      severity, enabled, created_by)
                VALUES (:id, :t, :name, :desc, :rt, :rid, :metric, :target, :window, :sev, true, :u)
                """
            ),
            {
                "id": sla_id, "t": tenant_id, "name": arguments["name"],
                "desc": arguments.get("description"),
                "rt": arguments["resource_type"], "rid": arguments.get("resource_id"),
                "metric": arguments["metric"], "target": arguments["target_value"],
                "window": arguments.get("window_seconds", 3600),
                "sev": arguments.get("severity", "medium"), "u": user_id,
            },
        )
        await db.commit()
    except Exception as e:
        raise ToolExecutionError(f"create_sla failed: {e}") from e
    return {"sla_id": sla_id, "name": arguments["name"]}


async def _handle_list_slas(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text
    from core.tenant_context import get_current_tenant_id

    tenant_id = get_current_tenant_id() or "default"
    sql = "SELECT id, name, resource_type, metric, target_value, enabled FROM mcp_slas WHERE tenant_id = :t"
    if arguments.get("enabled_only", True):
        sql += " AND enabled = true"
    rows = (await db.execute(text(sql), {"t": tenant_id})).all()
    return {
        "slas": [
            {"id": r[0], "name": r[1], "resource_type": r[2], "metric": r[3],
             "target_value": r[4], "enabled": r[5]}
            for r in rows
        ],
        "count": len(rows),
    }


async def _handle_get_sla_status(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text
    row = (await db.execute(
        text("SELECT id, name, metric, target_value, window_seconds FROM mcp_slas WHERE id = :id"),
        {"id": arguments["sla_id"]},
    )).first()
    if not row:
        raise ToolExecutionError(f"SLA {arguments['sla_id']} not found")
    # Violations count
    viols = (await db.execute(
        text("SELECT COUNT(*) FROM mcp_sla_violations WHERE sla_id = :id AND resolved = false"),
        {"id": arguments["sla_id"]},
    )).first()
    return {
        "sla_id": row[0], "name": row[1], "metric": row[2],
        "target_value": row[3], "window_seconds": row[4],
        "unresolved_violations": int(viols[0]) if viols else 0,
    }


async def _handle_get_sla_violations(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text
    from core.tenant_context import get_current_tenant_id

    tenant_id = get_current_tenant_id() or "default"
    conds = ["tenant_id = :t"]
    params: Dict[str, Any] = {"t": tenant_id}
    if "resolved" in arguments:
        conds.append("resolved = :r")
        params["r"] = bool(arguments["resolved"])
    if arguments.get("severity"):
        conds.append("severity = :sev")
        params["sev"] = arguments["severity"]
    sql = f"SELECT id, sla_id, observed_value, target_value, severity, resolved, detected_at FROM mcp_sla_violations WHERE {' AND '.join(conds)} ORDER BY detected_at DESC LIMIT :l"
    params["l"] = min(int(arguments.get("limit", 100)), 1000)
    rows = (await db.execute(text(sql), params)).all()
    return {
        "violations": [
            {"id": r[0], "sla_id": r[1], "observed": r[2], "target": r[3],
             "severity": r[4], "resolved": r[5], "detected_at": str(r[6])}
            for r in rows
        ],
        "count": len(rows),
    }


async def _handle_get_sla_metrics(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text
    from core.tenant_context import get_current_tenant_id

    tenant_id = get_current_tenant_id() or "default"
    period = arguments.get("period", "month")
    interval_map = {"day": "1 day", "week": "7 days", "month": "30 days"}
    interval = interval_map.get(period, "30 days")
    try:
        total = (await db.execute(
            text(f"SELECT COUNT(*) FROM mcp_slas WHERE tenant_id = :t"),
            {"t": tenant_id},
        )).first()
        breaches = (await db.execute(
            text(f"SELECT COUNT(*) FROM mcp_sla_violations WHERE tenant_id = :t AND detected_at >= now() - interval '{interval}'"),
            {"t": tenant_id},
        )).first()
        return {
            "period": period,
            "total_slas": int(total[0]) if total else 0,
            "breaches_in_period": int(breaches[0]) if breaches else 0,
        }
    except Exception as e:
        return {"status": "feature_pending", "error": str(e)}


# ---- B2.3 Workflow versioning (direct SQL) ----

async def _handle_list_workflow_versions(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text
    rows = (await db.execute(
        text("SELECT version, created_by, created_at, change_summary FROM mcp_workflow_versions WHERE workflow_id = :w ORDER BY version DESC"),
        {"w": arguments["workflow_id"]},
    )).all()
    return {
        "workflow_id": arguments["workflow_id"],
        "versions": [
            {"version": r[0], "created_by": r[1], "created_at": str(r[2]), "change_summary": r[3]}
            for r in rows
        ],
    }


async def _handle_get_workflow_version(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text
    row = (await db.execute(
        text("SELECT version, definition_json, created_at, change_summary FROM mcp_workflow_versions WHERE workflow_id = :w AND version = :v"),
        {"w": arguments["workflow_id"], "v": arguments["version"]},
    )).first()
    if not row:
        raise ToolExecutionError(f"Version {arguments['version']} of {arguments['workflow_id']} not found")
    import json as _json
    try:
        definition = _json.loads(row[1]) if isinstance(row[1], str) else row[1]
    except Exception:
        definition = row[1]
    return {
        "workflow_id": arguments["workflow_id"],
        "version": row[0],
        "definition": definition,
        "created_at": str(row[2]),
        "change_summary": row[3],
    }


async def _handle_rollback_workflow(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    # Feature_pending — requires writing a new version from target_version
    # back to the workflow, which needs WorkflowService integration.
    return {
        "status": "feature_pending",
        "available": False,
        "message": "Rollback requires WorkflowService integration; list/get/compare are available now.",
        "workflow_id": arguments["workflow_id"],
        "target_version": arguments["target_version"],
    }


async def _handle_compare_workflow_versions(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    import json as _json

    from sqlalchemy import text

    rows = (await db.execute(
        text("SELECT version, definition_json FROM mcp_workflow_versions WHERE workflow_id = :w AND version IN (:a, :b)"),
        {"w": arguments["workflow_id"], "a": arguments["version_a"], "b": arguments["version_b"]},
    )).all()
    defs = {}
    for r in rows:
        try:
            defs[r[0]] = _json.loads(r[1]) if isinstance(r[1], str) else r[1]
        except Exception:
            defs[r[0]] = {}
    a = defs.get(arguments["version_a"]) or {}
    b = defs.get(arguments["version_b"]) or {}
    # Simple diff: node-count, edges-count
    a_nodes = len(a.get("nodes", []) if isinstance(a, dict) else [])
    b_nodes = len(b.get("nodes", []) if isinstance(b, dict) else [])
    return {
        "workflow_id": arguments["workflow_id"],
        "version_a": arguments["version_a"],
        "version_b": arguments["version_b"],
        "summary": {
            "nodes_a": a_nodes,
            "nodes_b": b_nodes,
            "nodes_delta": b_nodes - a_nodes,
        },
        "definition_a": a,
        "definition_b": b,
    }


# ---- B2.4 RBAC (direct SQL against mcp_roles / mcp_role_permissions / mcp_user_roles) ----

async def _handle_list_roles(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text
    from core.tenant_context import get_current_tenant_id
    tenant_id = get_current_tenant_id() or "default"
    rows = (await db.execute(
        text("SELECT id, name, description, is_system FROM mcp_roles WHERE tenant_id = :t"),
        {"t": tenant_id},
    )).all()
    return {
        "roles": [
            {"id": r[0], "name": r[1], "description": r[2], "is_system": r[3]}
            for r in rows
        ],
        "count": len(rows),
    }


async def _handle_get_role(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text
    role = (await db.execute(
        text("SELECT id, name, description, is_system FROM mcp_roles WHERE id = :id"),
        {"id": arguments["role_id"]},
    )).first()
    if not role:
        raise ToolExecutionError(f"Role {arguments['role_id']} not found")
    perms = (await db.execute(
        text("SELECT resource, action, scope FROM mcp_role_permissions WHERE role_id = :id"),
        {"id": arguments["role_id"]},
    )).all()
    return {
        "id": role[0], "name": role[1], "description": role[2], "is_system": role[3],
        "permissions": [
            {"resource": p[0], "action": p[1], "scope": p[2]} for p in perms
        ],
    }


async def _handle_create_role(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    import uuid
    from sqlalchemy import text
    from core.tenant_context import get_current_tenant_id

    role_id = str(uuid.uuid4())
    tenant_id = get_current_tenant_id() or "default"
    try:
        await db.execute(
            text(
                "INSERT INTO mcp_roles (id, tenant_id, name, description, is_system) "
                "VALUES (:id, :t, :name, :desc, false)"
            ),
            {"id": role_id, "t": tenant_id, "name": arguments["name"],
             "desc": arguments.get("description")},
        )
        for perm in (arguments.get("permissions") or []):
            await db.execute(
                text(
                    "INSERT INTO mcp_role_permissions (id, role_id, resource, action) "
                    "VALUES (:id, :r, :res, :act)"
                ),
                {"id": str(uuid.uuid4()), "r": role_id,
                 "res": perm["resource"], "act": perm["action"]},
            )
        await db.commit()
    except Exception as e:
        raise ToolExecutionError(f"create_role failed: {e}") from e
    return {"role_id": role_id, "name": arguments["name"]}


async def _handle_grant_role(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    import uuid
    from sqlalchemy import text
    from core.tenant_context import get_current_tenant_id

    tenant_id = get_current_tenant_id() or "default"
    try:
        await db.execute(
            text(
                "INSERT INTO mcp_user_roles (id, user_id, role_id, tenant_id, granted_by) "
                "VALUES (:id, :u, :r, :t, :by) ON CONFLICT DO NOTHING"
            ),
            {"id": str(uuid.uuid4()), "u": arguments["user_id"],
             "r": arguments["role_id"], "t": tenant_id, "by": user_id},
        )
        await db.commit()
    except Exception as e:
        raise ToolExecutionError(f"grant_role failed: {e}") from e
    return {"user_id": arguments["user_id"], "role_id": arguments["role_id"], "granted": True}


async def _handle_revoke_role(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    from sqlalchemy import text
    try:
        await db.execute(
            text("DELETE FROM mcp_user_roles WHERE user_id = :u AND role_id = :r"),
            {"u": arguments["user_id"], "r": arguments["role_id"]},
        )
        await db.commit()
    except Exception as e:
        raise ToolExecutionError(f"revoke_role failed: {e}") from e
    return {"user_id": arguments["user_id"], "role_id": arguments["role_id"], "revoked": True}


async def _handle_list_permissions(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    # Static catalog — one entry per scope in our taxonomy
    from .scopes import READ, WRITE
    perms = []
    for s in sorted(READ):
        _, resource = s.split(":", 1)
        perms.append({"resource": resource, "action": "read", "scope": s})
    for s in sorted(WRITE):
        _, resource = s.split(":", 1)
        perms.append({"resource": resource, "action": "write", "scope": s})
    return {"permissions": perms, "count": len(perms)}


# ---- B2.5 Template CRUD ----

async def _handle_create_template(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    try:
        from services.workflow_template_service import create_workflow_template_service
        svc = create_workflow_template_service()
        method = getattr(svc, "create_template", None)
        if not method:
            return {"status": "feature_pending", "available": False}
        result = await method(
            db=db, user_id=user_id,
            name=arguments["name"],
            description=arguments.get("description"),
            category=arguments["category"],
            definition=arguments.get("definition"),
            parameters_schema=arguments.get("parameters_schema"),
            source_workflow_id=arguments.get("source_workflow_id"),
        )
        return {"template_id": str(getattr(result, "id", result))}
    except Exception as e:
        return {"status": "feature_pending", "available": False, "error": str(e)}


async def _handle_update_template(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    try:
        from services.workflow_template_service import create_workflow_template_service
        svc = create_workflow_template_service()
        method = getattr(svc, "update_template", None)
        if not method:
            return {"status": "feature_pending", "available": False}
        result = await method(
            db=db, user_id=user_id,
            template_id=arguments["template_id"],
            definition=arguments.get("definition"),
            parameters_schema=arguments.get("parameters_schema"),
            change_summary=arguments.get("change_summary"),
        )
        return {"template_id": arguments["template_id"], "updated": bool(result)}
    except Exception as e:
        return {"status": "feature_pending", "available": False, "error": str(e)}


async def _handle_delete_template(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    try:
        from services.workflow_template_service import create_workflow_template_service
        svc = create_workflow_template_service()
        method = getattr(svc, "delete_template", None)
        if not method:
            return {"status": "feature_pending", "available": False}
        result = await method(db=db, user_id=user_id, template_id=arguments["template_id"])
        return {"template_id": arguments["template_id"], "deleted": bool(result)}
    except Exception as e:
        return {"status": "feature_pending", "available": False, "error": str(e)}


# ---- B2.6 MFA + OAuth2 admin ----

async def _handle_get_mfa_status(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    try:
        from services.mfa_service import MFAService
        svc = MFAService(db)
        method = getattr(svc, "get_mfa_status", None) or getattr(svc, "get_status", None)
        if method:
            return {"mfa_status": _pa_dump(await method(user_id))}
    except Exception:
        pass
    return {"status": "feature_pending", "available": False}


async def _handle_list_oauth2_clients(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from sqlalchemy import select
        from aictrlnet_business.models.oauth2 import OAuth2Client  # type: ignore
        from core.tenant_context import get_current_tenant_id
        tenant_id = get_current_tenant_id() or "default"
        rows = (await db.execute(
            select(OAuth2Client).where(OAuth2Client.tenant_id == tenant_id)
        )).scalars().all()
        return {
            "clients": [
                {"id": c.id, "client_name": c.client_name, "client_type": c.client_type,
                 "is_active": c.is_active, "allowed_scopes": c.allowed_scopes,
                 "created_at": str(getattr(c, "created_at", ""))}
                for c in rows
            ],
            "count": len(rows),
        }
    except Exception:
        return {"status": "feature_pending", "available": False, "clients": []}


async def _handle_revoke_oauth2_token(
    arguments: Dict[str, Any], db: AsyncSession, user_id: str
) -> Dict[str, Any]:
    _ensure_business_sys_path()
    try:
        from aictrlnet_business.services.oauth2_service_async import OAuth2ServiceAsync  # type: ignore
        svc = OAuth2ServiceAsync(db)
        method = getattr(svc, "revoke_token", None) or getattr(svc, "revoke_access_token", None)
        if method:
            await method(arguments["token_id"])
            return {"token_id": arguments["token_id"], "revoked": True}
    except Exception:
        pass
    return {"status": "feature_pending", "available": False}


TOOL_HANDLERS = {
    # Original 11
    "create_workflow": _handle_create_workflow,
    "list_workflows": _handle_list_workflows,
    "get_workflow": _handle_get_workflow,
    "execute_workflow": _handle_execute_workflow,
    "get_execution_status": _handle_get_execution_status,
    "list_templates": _handle_list_templates,
    "assess_quality": _handle_assess_quality,
    "send_message": _handle_send_message,
    "evaluate_policy": _handle_evaluate_policy,
    "list_policies": _handle_list_policies,
    "check_compliance": _handle_check_compliance,
    # Wave 1: Adapters (Layer 1)
    "list_adapters": _handle_list_adapters,
    "get_adapter": _handle_get_adapter,
    "list_my_adapter_configs": _handle_list_my_adapter_configs,
    "test_adapter_config": _handle_test_adapter_config,
    # Wave 1: NL entry
    "nl_to_workflow": _handle_nl_to_workflow,
    "analyze_intent": _handle_analyze_intent,
    # Wave 1: Autonomy (Control Spectrum)
    "get_workflow_autonomy": _handle_get_workflow_autonomy,
    "preview_autonomy": _handle_preview_autonomy,
    "set_workflow_autonomy": _handle_set_workflow_autonomy,
    # Wave 1: Self-extending (Layer 2)
    "research_api": _handle_research_api,
    "generate_adapter": _handle_generate_adapter,
    "self_extend": _handle_self_extend,
    "list_generated_adapters": _handle_list_generated_adapters,
    "get_generated_adapter_status": _handle_get_generated_adapter_status,
    "get_generated_adapter_source": _handle_get_generated_adapter_source,
    "approve_adapter": _handle_approve_adapter,
    "reject_adapter": _handle_reject_adapter,
    "activate_adapter": _handle_activate_adapter,
    # Wave 1: Browser (Layer 3)
    "browser_execute": _handle_browser_execute,
    # Wave 1: Approvals (Layer 4)
    "list_pending_approvals": _handle_list_pending_approvals,
    "get_approval": _handle_get_approval,
    "approve_request": _handle_approve_request,
    "reject_request": _handle_reject_request,
    # Wave 2: API-key + subscription introspection
    "list_api_keys": _handle_list_api_keys,
    "get_api_key_usage": _handle_get_api_key_usage,
    "get_subscription": _handle_get_subscription,
    "get_upgrade_options": _handle_get_upgrade_options,
    # Wave 2: AI governance visibility
    "list_ai_policies": _handle_list_ai_policies,
    "create_policy": _handle_create_policy,
    "get_ai_audit_logs": _handle_get_ai_audit_logs,
    "list_violations": _handle_list_violations,
    # Wave 3: Trial metering surface
    "get_trial_status": _handle_get_trial_status,
    "get_usage_report": _handle_get_usage_report,
    # Wave 4: Tasks
    "create_task": _handle_create_task,
    "list_tasks": _handle_list_tasks,
    "get_task": _handle_get_task,
    "update_task": _handle_update_task,
    "complete_task": _handle_complete_task,
    # Wave 4: Memory
    "get_memory": _handle_get_memory,
    "set_memory": _handle_set_memory,
    "delete_memory": _handle_delete_memory,
    # Wave 4: Conversations + Channels
    "list_conversations": _handle_list_conversations,
    "get_conversation": _handle_get_conversation,
    "list_linked_channels": _handle_list_linked_channels,
    "request_channel_link_code": _handle_request_channel_link_code,
    "unlink_channel": _handle_unlink_channel,
    "send_channel_message": _handle_send_channel_message,
    "list_notifications": _handle_list_notifications,
    "mark_notification_read": _handle_mark_notification_read,
    # Wave 4: Knowledge
    "query_knowledge": _handle_query_knowledge,
    "suggest_next_actions": _handle_suggest_next_actions,
    "get_capabilities_summary": _handle_get_capabilities_summary,
    # Wave 4: Templates
    "search_templates": _handle_search_templates,
    "instantiate_template": _handle_instantiate_template,
    # Wave 4: Files
    "upload_file": _handle_upload_file,
    "list_staged_files": _handle_list_staged_files,
    "get_staged_file": _handle_get_staged_file,
    # Wave 4: Data Quality
    "assess_data_quality": _handle_assess_data_quality,
    "list_quality_dimensions": _handle_list_quality_dimensions,
    # Wave 4: Agents
    "list_agents": _handle_list_agents,
    "get_agent_capabilities": _handle_get_agent_capabilities,
    "set_agent_autonomy": _handle_set_agent_autonomy,
    "execute_agent": _handle_execute_agent,
    # Wave 4: LLM Registry
    "list_llm_models": _handle_list_llm_models,
    "get_llm_recommendation": _handle_get_llm_recommendation,
    # Wave 4: Living Platform
    "list_pattern_candidates": _handle_list_pattern_candidates,
    "promote_pattern_to_template": _handle_promote_pattern_to_template,
    "org_discovery_scan": _handle_org_discovery_scan,
    "get_org_landscape": _handle_get_org_landscape,
    "get_org_recommendations": _handle_get_org_recommendations,
    "automate_company": _handle_automate_company,
    "get_company_automation_status": _handle_get_company_automation_status,
    "list_industry_packs": _handle_list_industry_packs,
    "detect_industry": _handle_detect_industry,
    "verify_quality": _handle_verify_quality,
    # Wave 5: Institute
    "list_institute_modules": _handle_list_institute_modules,
    "enroll_in_module": _handle_enroll_in_module,
    "get_certification_status": _handle_get_certification_status,
    # Wave 6: Analytics
    "query_analytics": _handle_query_analytics,
    "get_dashboard_metrics": _handle_get_dashboard_metrics,
    "get_metric_trends": _handle_get_metric_trends,
    # Wave 6: Audit
    "get_audit_logs": _handle_get_audit_logs,
    "get_audit_summary": _handle_get_audit_summary,
    # Wave 6: Compliance
    "run_compliance_check": _handle_run_compliance_check,
    "list_compliance_standards": _handle_list_compliance_standards,
    "get_enterprise_risk_assessment": _handle_get_enterprise_risk_assessment,
    # Wave 6: Organizations + Tenants
    "list_organizations": _handle_list_organizations,
    "list_tenants": _handle_list_tenants,
    # Wave 6: Federated knowledge + cross-tenant
    "federated_knowledge_query": _handle_federated_knowledge_query,
    "get_cross_tenant_insights": _handle_get_cross_tenant_insights,
    # Wave 6: Fleet management
    "list_fleet_agents": _handle_list_fleet_agents,
    "get_fleet_autonomy_summary": _handle_get_fleet_autonomy_summary,
    # Wave 6: License management
    "get_license_status": _handle_get_license_status,
    "list_license_entitlements": _handle_list_license_entitlements,
    # Wave 7 B1.1: MCP Client
    "register_mcp_server": _handle_register_mcp_server,
    "discover_mcp_server_tools": _handle_discover_mcp_server_tools,
    "invoke_external_mcp_tool": _handle_invoke_external_mcp_tool,
    "list_registered_mcp_servers": _handle_list_registered_mcp_servers,
    "unregister_mcp_server": _handle_unregister_mcp_server,
    # Wave 7 B1.2: Credentials
    "create_credential": _handle_create_credential,
    "list_credentials": _handle_list_credentials,
    "get_credential": _handle_get_credential,
    "delete_credential": _handle_delete_credential,
    "rotate_credential": _handle_rotate_credential,
    "validate_credential": _handle_validate_credential,
    # Wave 7 B1.3: Personal Agent Hub
    "get_personal_agent_config": _handle_get_personal_agent_config,
    "update_personal_agent_config": _handle_update_personal_agent_config,
    "get_personal_agent_activity": _handle_get_personal_agent_activity,
    "connect_external_agent": _handle_connect_external_agent,
    "create_personal_workflow": _handle_create_personal_workflow,
    "promote_personal_workflow": _handle_promote_personal_workflow,
    # Wave 7 B1.4: Marketplace
    "list_org_marketplace_items": _handle_list_org_marketplace_items,
    "publish_to_org_marketplace": _handle_publish_to_org_marketplace,
    "compose_marketplace_items": _handle_compose_marketplace_items,
    "sync_public_marketplace_updates": _handle_sync_public_marketplace_updates,
    # Wave 7 B1.5: External platform native execution
    "execute_n8n_workflow": _handle_execute_n8n_workflow,
    "execute_zapier_zap": _handle_execute_zapier_zap,
    "execute_make_scenario": _handle_execute_make_scenario,
    "execute_power_automate_flow": _handle_execute_power_automate_flow,
    # Wave 7 B1.6: OpenClaw + A2A
    "evaluate_runtime_action": _handle_evaluate_runtime_action,
    "get_delegation_chain": _handle_get_delegation_chain,
    "list_a2a_agents": _handle_list_a2a_agents,
    "register_runtime_webhook": _handle_register_runtime_webhook,
    "list_runtime_webhooks": _handle_list_runtime_webhooks,
    # Wave 7 B1.7: Pods + Swarm
    "form_pod": _handle_form_pod,
    "list_pods": _handle_list_pods,
    "get_pod_status": _handle_get_pod_status,
    "dispatch_swarm": _handle_dispatch_swarm,
    # Wave 7 B1.8: Framework cascading + ML matching
    "get_framework_cascade": _handle_get_framework_cascade,
    "set_framework_priority": _handle_set_framework_priority,
    "get_execution_framework_trace": _handle_get_execution_framework_trace,
    "match_agents_to_task": _handle_match_agents_to_task,
    # Wave 7 B1.9: Activity Timeline + Operations
    "get_activity_timeline": _handle_get_activity_timeline,
    "get_operations_status": _handle_get_operations_status,
    # Wave 7 B2.1: Cost
    "get_cost_analytics": _handle_get_cost_analytics,
    "get_platform_cost_estimate": _handle_get_platform_cost_estimate,
    "optimize_workflow_cost": _handle_optimize_workflow_cost,
    "analyze_cost_trends": _handle_analyze_cost_trends,
    # Wave 7 B2.2: SLA
    "create_sla": _handle_create_sla,
    "list_slas": _handle_list_slas,
    "get_sla_status": _handle_get_sla_status,
    "get_sla_violations": _handle_get_sla_violations,
    "get_sla_metrics": _handle_get_sla_metrics,
    # Wave 7 B2.3: Workflow versioning
    "list_workflow_versions": _handle_list_workflow_versions,
    "get_workflow_version": _handle_get_workflow_version,
    "rollback_workflow": _handle_rollback_workflow,
    "compare_workflow_versions": _handle_compare_workflow_versions,
    # Wave 7 B2.4: RBAC
    "list_roles": _handle_list_roles,
    "get_role": _handle_get_role,
    "create_role": _handle_create_role,
    "grant_role": _handle_grant_role,
    "revoke_role": _handle_revoke_role,
    "list_permissions": _handle_list_permissions,
    # Wave 7 B2.5: Template CRUD
    "create_template": _handle_create_template,
    "update_template": _handle_update_template,
    "delete_template": _handle_delete_template,
    # Wave 7 B2.6: MFA + OAuth2 admin
    "get_mfa_status": _handle_get_mfa_status,
    "list_oauth2_clients": _handle_list_oauth2_clients,
    "revoke_oauth2_token": _handle_revoke_oauth2_token,
}
