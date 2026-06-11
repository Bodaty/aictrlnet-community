"""Platform Integration Node for external automation platforms.

Executes a workflow on an external automation platform (n8n, Zapier, Make,
Power Automate, IFTTT) when the node is fully configured with a real
``PlatformNodeConfig`` (platform + workflow_id + credential_id).

Auto-generated system templates carry *descriptive* platformIntegration nodes
(free-text ``platform`` like "Multi-Channel Communications System", no
workflow_id / credential_id). Those can't address a live platform, so the node
degrades gracefully: it completes with an audit-visible ``skipped`` output
instead of crashing the workflow. No integration result is fabricated.
"""
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from core.database import get_db  # noqa: F401  (kept for compatibility)
from models.platform_integration import PlatformExecution, PlatformCredential, PlatformType
from schemas.platform_integration import PlatformNodeConfig
from services.platform_credential_service import PlatformCredentialService
from services.platform_adapters import (
    PlatformAdapterService,
    ExecutionRequest,
    ExecutionStatus,
)
from sqlalchemy import update as sa_update
from ..base_node import BaseNode
from ..models import NodeConfig

logger = logging.getLogger(__name__)


class PlatformIntegrationNode(BaseNode):
    """Node for integrating with external automation platforms."""

    node_type = "platform"

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Parsed lazily in execute(); descriptive nodes never form a valid one.
        self.platform_config: Optional[PlatformNodeConfig] = None

    def _build_platform_config(self) -> Optional[PlatformNodeConfig]:
        """Build a real PlatformNodeConfig from node parameters, or None.

        Returns None for descriptive placeholder nodes that lack the operational
        fields (valid platform enum + workflow_id + credential_id). A node that
        HAS operational fields but fails validation is a misconfigured live
        integration — raise so the author sees the error instead of a silent
        'skipped' no-op.
        """
        try:
            return PlatformNodeConfig(**self.config.parameters)
        except Exception as exc:
            p = self.config.parameters or {}
            if p.get("platform") and (p.get("workflow_id") or p.get("credential_id")):
                raise ValueError(
                    f"platformIntegration node {self.config.id}: "
                    f"invalid platform configuration: {exc}"
                )
            return None

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the platform integration node. Returns an output dict."""
        self.platform_config = self._build_platform_config()

        if self.platform_config is None:
            # Descriptive / unconfigured node — skip honestly, don't crash.
            platform_label = str(self.config.parameters.get("platform", "")) or "unspecified"
            logger.info(
                "platformIntegration node %s skipped — no live platform integration "
                "configured (descriptive node, platform=%r)",
                self.config.id, platform_label,
            )
            return {
                "status": "skipped",
                "executed": False,
                "reason": "no platform integration configured",
                "platform": platform_label,
                "node_id": self.config.id,
            }

        return await self._execute_platform(input_data, context)

    async def _execute_platform(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the configured workflow on the external platform."""
        start_time = datetime.utcnow()
        pc = self.platform_config

        db = context.get("db")
        if db is None:
            raise ValueError("Database session not provided in context")
        user_id = context.get("user_id")
        if not user_id:
            raise ValueError("User context not provided for platform integration")

        credential_service = PlatformCredentialService(db)
        adapter_service = PlatformAdapterService(db)

        credential_data = await credential_service.get_credential_data(
            pc.credential_id, user_id
        )
        if not credential_data:
            raise ValueError("Platform credentials not found or unauthorized")

        adapter = adapter_service.get_adapter_instance(pc.platform)
        if not adapter:
            raise ValueError(f"Platform adapter not found for {pc.platform.value}")

        platform_input = self._apply_mapping(input_data, pc.input_mapping)

        workflow_instance_id = context.get("workflow_instance_id") or context.get("workflow_id")
        execution_id = context.get("execution_id")

        # Best-effort audit record of the platform call.
        db_execution = PlatformExecution(
            workflow_id=workflow_instance_id,
            node_id=self.config.id,
            execution_id=execution_id,
            platform=pc.platform.value,
            external_workflow_id=pc.workflow_id,
            credential_id=pc.credential_id,
            input_data=platform_input,
            status="running",
            started_at=start_time,
        )
        try:
            db.add(db_execution)
            await db.commit()
        except Exception as rec_err:
            logger.warning("Could not persist platform execution record: %s", rec_err)

        execution_request = ExecutionRequest(
            workflow_id=pc.workflow_id,
            input_data=platform_input,
            timeout=pc.timeout,
            retry_count=0,
            metadata={"node_id": self.config.id, "workflow_instance_id": workflow_instance_id},
        )

        try:
            response = await adapter.execute_workflow(credential_data, execution_request)
        except Exception as exec_err:
            logger.error("Platform execution failed: %s", exec_err, exc_info=True)
            try:
                db_execution.status = "failed"
                db_execution.error_data = {"error": str(exec_err)}
                db_execution.completed_at = datetime.utcnow()
                await db.commit()
            except Exception:
                pass
            raise

        # Persist result + bump credential usage (both best-effort).
        try:
            db_execution.external_execution_id = response.execution_id
            db_execution.status = response.status.value
            db_execution.output_data = response.output_data
            db_execution.error_data = {"error": response.error} if response.error else None
            db_execution.completed_at = response.completed_at or datetime.utcnow()
            db_execution.duration_ms = response.duration_ms
            db_execution.estimated_cost = response.cost_estimate or 0
            await db.execute(
                sa_update(PlatformCredential)
                .where(PlatformCredential.id == pc.credential_id)
                .values(execution_count=PlatformCredential.execution_count + 1)
            )
            await db.commit()
        except Exception as save_err:
            logger.warning("Could not persist platform execution result: %s", save_err)

        if response.status == ExecutionStatus.COMPLETED:
            output_data = self._apply_mapping(response.output_data or {}, pc.output_mapping)
            output_data["_platform_metadata"] = {
                "platform": pc.platform.value,
                "execution_id": response.execution_id,
                "duration_ms": response.duration_ms,
                "cost_estimate": response.cost_estimate,
            }
            return output_data

        if response.status == ExecutionStatus.TIMEOUT:
            raise TimeoutError(f"Platform execution timed out after {pc.timeout}s")

        raise RuntimeError(
            response.error or f"Platform execution failed with status: {response.status}"
        )

    def _apply_mapping(
        self,
        data: Dict[str, Any],
        mapping: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply field mapping transformation."""
        if not mapping:
            return data

        result = {}
        for target_field, source_spec in mapping.items():
            if isinstance(source_spec, str):
                if source_spec.startswith("$."):
                    try:
                        import jsonpath_ng
                        expr = jsonpath_ng.parse(source_spec)
                        matches = expr.find(data)
                        if matches:
                            result[target_field] = matches[0].value
                    except Exception:
                        field = source_spec[2:]
                        if field in data:
                            result[target_field] = data[field]
                else:
                    if source_spec in data:
                        result[target_field] = data[source_spec]
            elif isinstance(source_spec, dict):
                source_field = source_spec.get("field")
                transform = source_spec.get("transform")
                default = source_spec.get("default")
                value = data.get(source_field, default)
                if value is not None and transform:
                    if transform == "string":
                        value = str(value)
                    elif transform == "number":
                        try:
                            value = float(value)
                        except Exception:
                            value = default
                    elif transform == "boolean":
                        value = bool(value)
                    elif transform == "json":
                        if isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except Exception:
                                pass
                if value is not None:
                    result[target_field] = value
            else:
                result[target_field] = source_spec

        return result
