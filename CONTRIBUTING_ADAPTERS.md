# Contributing Adapters to AICtrlNet

Add a new adapter in under 30 minutes by following this guide.

## Architecture Overview

All adapters extend `BaseAdapter` from `adapters/base_adapter.py` and implement four methods:

| Method | Purpose |
|--------|---------|
| `initialize()` | Connect to external service, validate credentials |
| `shutdown()` | Clean up connections |
| `get_capabilities()` | Declare what the adapter can do |
| `execute(request)` | Handle an incoming `AdapterRequest` and return `AdapterResponse` |

Adapters live under `editions/community/src/adapters/implementations/` in a category folder:
- `ai/` — LLM and ML model providers
- `communication/` — Messaging platforms (Slack, Telegram, WhatsApp, etc.)
- `human/` — Human-in-the-loop services
- `payment/` — Payment processors
- `ai_agents/` — External AI agent frameworks

## Step-by-Step: Add a Communication Adapter

### 1. Create the adapter file

```
editions/community/src/adapters/implementations/communication/myplatform_adapter.py
```

Use the template in `adapter-template/` as a starting point, or copy an existing adapter like `telegram_adapter.py`.

### 2. Implement the four required methods

```python
"""MyPlatform adapter for AICtrlNet."""

import logging
from typing import Any, Dict, List

import httpx

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterConfig, AdapterRequest,
    AdapterResponse, AdapterStatus,
)

logger = logging.getLogger(__name__)

MYPLATFORM_API_BASE = "https://api.myplatform.com/v1"


class MyPlatformAdapter(BaseAdapter):
    """Adapter for MyPlatform messaging API."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.api_key = config.credentials.get("api_key", "") if config.credentials else ""
        self._client = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=MYPLATFORM_API_BASE,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )
        self.status = AdapterStatus.READY
        self._initialized = True

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        self.status = AdapterStatus.STOPPED

    def get_capabilities(self) -> List[AdapterCapability]:
        return [
            AdapterCapability(
                name="send_message",
                description="Send a text message",
                parameters={"to": "str", "text": "str"},
            ),
        ]

    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        handler = {
            "send_message": self._send_message,
        }.get(request.capability)

        if not handler:
            return AdapterResponse(
                status="error",
                error=f"Unknown capability: {request.capability}",
            )

        return await handler(request.parameters)

    async def _send_message(self, params: Dict[str, Any]) -> AdapterResponse:
        resp = await self._client.post("/messages", json={
            "to": params["to"],
            "text": params["text"],
        })
        resp.raise_for_status()
        data = resp.json()
        return AdapterResponse(status="success", data=data)
```

### 3. Register in the factory

Edit `editions/community/src/adapters/factory.py` and add your adapter to `ADAPTER_MAPPINGS`:

```python
ADAPTER_MAPPINGS = {
    # ...existing adapters...
    "myplatform": "adapters.implementations.communication.myplatform_adapter.MyPlatformAdapter",
}
```

Also add the platform to `_determine_category`:

```python
communication_adapters = ["slack", "email", "webhook", "discord", "telegram", "whatsapp", "myplatform"]
```

### 4. Export from `__init__.py`

Add to `editions/community/src/adapters/implementations/communication/__init__.py`:

```python
from .myplatform_adapter import MyPlatformAdapter
```

### 5. (Optional) Add webhook normalization

If your platform sends inbound messages via webhooks, add a normalizer method to `editions/community/src/services/channel_normalizer.py`:

```python
def normalize_myplatform(self, payload: Dict[str, Any]) -> Optional[InboundMessage]:
    return InboundMessage(
        channel_type="myplatform",
        sender_id=payload["from"]["id"],
        message_text=payload["message"]["text"],
        external_message_id=payload["message"]["id"],
        platform_metadata={"chat_id": payload["chat"]["id"]},
    )
```

Then register it in `editions/community/src/api/v1/endpoints/channel_webhook.py`:

```python
_NORMALIZERS = {
    # ...existing...
    "myplatform": normalizer.normalize_myplatform,
}
```

### 6. (Optional) Add notification node support

If you want workflows to send notifications through your platform, edit `editions/community/src/nodes/implementations/notification_node.py`:

1. Add `"myplatform"` to `valid_channels` in `validate_config()`
2. Add a handler branch in `execute()`
3. Add a `_send_myplatform()` method following the existing patterns

## Testing

```bash
# Validate imports
make validate

# Run community tests
make test

# Check health
make health
```

## Checklist

- [ ] Adapter file created with all 4 methods
- [ ] Registered in `factory.py` ADAPTER_MAPPINGS
- [ ] Exported from `__init__.py`
- [ ] `make validate` passes (no import errors)
- [ ] (Optional) Webhook normalizer added
- [ ] (Optional) Notification node support added
