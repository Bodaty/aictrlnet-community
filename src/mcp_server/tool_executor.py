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
}
