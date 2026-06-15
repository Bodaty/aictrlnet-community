"""Trello adapter — governed Trello board/card integration.

A first-class adapter (modeled on the Slack adapter) so every Trello call a
workflow makes inherits the BaseAdapter cross-cutting stack: audit logging,
credential management, rate-limiting, retries, metrics, and control-plane
governance. Replaces the hand-rolled `webhook`-node + `urllib` Trello wiring.

Auth: Trello uses query-param auth (`?key=...&token=...`) on every request, not
a bearer header. Credentials resolve from (in order) the AdapterConfig, then the
env fallback `TRELLO_API_KEY` / `TRELLO_API_TOKEN` so self-hosted deployments
work out of the box.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability,
    AdapterRequest,
    AdapterResponse,
    AdapterConfig,
    AdapterCategory,
)
from events.event_bus import event_bus

logger = logging.getLogger(__name__)


class TrelloAdapter(BaseAdapter):
    """Adapter for the Trello REST API."""

    # Declarative capability table — single source for both get_capabilities()
    # and execute(). Each entry: HTTP method, path template (filled from the
    # request's friendly params + {token}), `params` mapping {trello_param ->
    # friendly_param}, any constant query params, required friendly params, a
    # UI category, and parameter docs for the capability schema.
    _CAPS: Dict[str, Dict[str, Any]] = {
        "create_card": {
            "method": "POST", "path": "/cards", "cat": "cards",
            "params": {"idList": "list_id", "name": "name", "desc": "desc",
                       "due": "due", "idLabels": "label_ids", "idMembers": "member_ids"},
            "required": ["list_id", "name"],
            "docs": {"list_id": "List ID to create the card in", "name": "Card title",
                     "desc": "Card description", "due": "Due date (ISO 8601)",
                     "label_ids": "Label IDs (list)", "member_ids": "Member IDs to assign (list)"},
        },
        "get_card": {
            "method": "GET", "path": "/cards/{card_id}", "cat": "cards",
            "params": {"fields": "fields"}, "required": ["card_id"],
            "docs": {"card_id": "Card ID", "fields": "Comma-separated fields to return"},
        },
        "move_card": {
            "method": "PUT", "path": "/cards/{card_id}", "cat": "cards",
            "params": {"idList": "target_list_id"}, "required": ["card_id", "target_list_id"],
            "docs": {"card_id": "Card ID", "target_list_id": "Destination list ID"},
        },
        "update_card": {
            "method": "PUT", "path": "/cards/{card_id}", "cat": "cards",
            "params": {"name": "name", "desc": "desc", "due": "due",
                       "closed": "closed", "idList": "target_list_id"},
            "required": ["card_id"],
            "docs": {"card_id": "Card ID", "name": "New title", "desc": "New description",
                     "due": "Due date", "closed": "Archive (true/false)", "target_list_id": "Move to list ID"},
        },
        "add_comment": {
            "method": "POST", "path": "/cards/{card_id}/actions/comments", "cat": "cards",
            "params": {"text": "text"}, "required": ["card_id", "text"],
            "docs": {"card_id": "Card ID", "text": "Comment text"},
        },
        "list_cards": {
            "method": "GET", "path": "/lists/{list_id}/cards", "cat": "lists",
            "params": {"fields": "fields"}, "required": ["list_id"],
            "docs": {"list_id": "List ID", "fields": "Comma-separated fields to return"},
        },
        "archive_card": {
            "method": "PUT", "path": "/cards/{card_id}", "cat": "cards",
            "params": {}, "const": {"closed": "true"}, "required": ["card_id"],
            "docs": {"card_id": "Card ID"},
        },
        "list_lists": {
            "method": "GET", "path": "/boards/{board_id}/lists", "cat": "lists",
            "params": {}, "required": ["board_id"], "docs": {"board_id": "Board ID"},
        },
        "create_list": {
            "method": "POST", "path": "/lists", "cat": "lists",
            "params": {"name": "name", "idBoard": "board_id"}, "required": ["board_id", "name"],
            "docs": {"board_id": "Board ID", "name": "List name"},
        },
        "add_label": {
            "method": "POST", "path": "/cards/{card_id}/idLabels", "cat": "labels",
            "params": {"value": "label_id"}, "required": ["card_id", "label_id"],
            "docs": {"card_id": "Card ID", "label_id": "Label ID to attach"},
        },
        "list_labels": {
            "method": "GET", "path": "/boards/{board_id}/labels", "cat": "labels",
            "params": {}, "required": ["board_id"], "docs": {"board_id": "Board ID"},
        },
        "add_checklist": {
            "method": "POST", "path": "/cards/{card_id}/checklists", "cat": "checklists",
            "params": {"name": "name"}, "required": ["card_id", "name"],
            "docs": {"card_id": "Card ID", "name": "Checklist name"},
        },
        "get_board": {
            "method": "GET", "path": "/boards/{board_id}", "cat": "boards",
            "params": {"fields": "fields"}, "required": ["board_id"],
            "docs": {"board_id": "Board ID", "fields": "Comma-separated fields"},
        },
        "get_members": {
            "method": "GET", "path": "/boards/{board_id}/members", "cat": "members",
            "params": {}, "required": ["board_id"], "docs": {"board_id": "Board ID"},
        },
        "assign_member": {
            "method": "POST", "path": "/cards/{card_id}/idMembers", "cat": "members",
            "params": {"value": "member_id"}, "required": ["card_id", "member_id"],
            "docs": {"card_id": "Card ID", "member_id": "Member ID to assign"},
        },
        "register_webhook": {
            "method": "POST", "path": "/webhooks", "cat": "webhooks",
            "params": {"callbackURL": "callback_url", "idModel": "id_model",
                       "description": "description"},
            "required": ["callback_url", "id_model"],
            "docs": {"callback_url": "Public HTTPS callback URL (must 200 a HEAD at creation)",
                     "id_model": "Trello model ID to watch (board/list/card)",
                     "description": "Webhook description"},
        },
        "delete_webhook": {
            "method": "DELETE", "path": "/webhooks/{webhook_id}", "cat": "webhooks",
            "params": {}, "required": ["webhook_id"], "docs": {"webhook_id": "Webhook ID"},
        },
        "list_webhooks": {
            "method": "GET", "path": "/tokens/{token}/webhooks", "cat": "webhooks",
            "params": {}, "required": [], "docs": {},
        },
    }

    def __init__(self, config: AdapterConfig):
        config.category = AdapterCategory.INTEGRATION
        super().__init__(config)

        self.client: Optional[httpx.AsyncClient] = None
        self.base_url = config.base_url or "https://api.trello.com/1"
        self.discovery_only = (
            config.custom_config.get("discovery_only", False) if config.custom_config else False
        )

        creds = config.credentials or {}
        self.api_key = config.api_key or creds.get("api_key") or os.getenv("TRELLO_API_KEY")
        self.token = (
            creds.get("api_token") or creds.get("token") or config.api_secret
            or os.getenv("TRELLO_API_TOKEN")
        )

        if not self.discovery_only and not (self.api_key and self.token):
            raise ValueError("Trello adapter requires an api_key and api_token")

    @property
    def _auth(self) -> Dict[str, str]:
        return {"key": self.api_key, "token": self.token}

    async def initialize(self) -> None:
        if self.discovery_only:
            logger.info("Trello adapter initialized in discovery mode")
            return
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.config.timeout_seconds)
        # Validate credentials against the current member.
        resp = await self.client.get("/members/me", params=self._auth)
        resp.raise_for_status()
        me = resp.json()
        self.member = {"id": me.get("id"), "username": me.get("username")}
        logger.info(f"Trello adapter initialized for member: {self.member.get('username')}")

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Trello adapter shutdown")

    def get_capabilities(self) -> List[AdapterCapability]:
        caps: List[AdapterCapability] = []
        for name, spec in self._CAPS.items():
            caps.append(
                AdapterCapability(
                    name=name,
                    description=f"Trello: {name.replace('_', ' ')}",
                    category=spec["cat"],
                    parameters={
                        friendly: {"type": "string", "description": doc}
                        for friendly, doc in spec.get("docs", {}).items()
                    },
                    required_parameters=spec["required"],
                    async_supported=True,
                    estimated_duration_seconds=0.5,
                )
            )
        return caps

    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        self.validate_request(request)
        spec = self._CAPS.get(request.capability)
        if not spec:
            raise ValueError(f"Unknown capability: {request.capability}")

        start = datetime.utcnow()
        try:
            p = request.parameters
            path = spec["path"].format(**{**p, "token": self.token})
            params: Dict[str, Any] = dict(spec.get("const", {}))
            for trello_key, friendly in spec["params"].items():
                if friendly in p and p[friendly] is not None:
                    value = p[friendly]
                    params[trello_key] = ",".join(map(str, value)) if isinstance(value, list) else value

            data = await self._api(spec["method"], path, params)
            # Trello list endpoints return a bare JSON array; wrap it so downstream
            # consumers (esp. the workflow `adapter` node, which sets keys on the
            # result dict) always receive a dict.
            if isinstance(data, list):
                data = {"items": data, "count": len(data)}

            await event_bus.publish(
                f"adapter.trello.{request.capability}",
                {"capability": request.capability, "path": path},
                source_id=self.id,
                source_type="adapter",
            )
            return self._success(request, data, start)
        except Exception as e:  # noqa: BLE001 — surface as an error response, not a crash
            return self._error(request, e, start)

    async def _api(self, method: str, path: str, params: Dict[str, Any]) -> Any:
        resp = await self.client.request(method, path, params={**self._auth, **params})
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}

    @staticmethod
    def _success(request: AdapterRequest, data: Any, start: datetime) -> AdapterResponse:
        return AdapterResponse(
            request_id=request.id,
            capability=request.capability,
            status="success",
            data=data,
            duration_ms=(datetime.utcnow() - start).total_seconds() * 1000,
        )

    @staticmethod
    def _error(request: AdapterRequest, err: Exception, start: datetime) -> AdapterResponse:
        return AdapterResponse(
            request_id=request.id,
            capability=request.capability,
            status="error",
            error=str(err),
            duration_ms=(datetime.utcnow() - start).total_seconds() * 1000,
        )

    async def _perform_health_check(self) -> Dict[str, Any]:
        try:
            resp = await self.client.get("/members/me", params=self._auth)
            if resp.status_code == 200:
                return {"status": "healthy", "member": resp.json().get("username")}
            return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
