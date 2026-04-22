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
        await _enforce_compliance_if_enterprise(tool_name, tenant_id, db)

        # 4. Per-tool rate bucket (Redis / in-memory fallback)
        try:
            await check_rate(
                tool_name=tool_name,
                api_key=api_key,
                user_id=user_id,
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
        )


async def _enforce_compliance_if_enterprise(
    tool_name: str, tenant_id: Optional[str], db: AsyncSession
) -> None:
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
) -> None:
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


TOOL_HANDLERS = {
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
}
