"""Twilio SMS/MMS adapter implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import httpx
from datetime import datetime
import json
import hashlib
import hmac
from urllib.parse import urlencode

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory
)
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class TwilioAdapter(BaseAdapter):
    """Adapter for Twilio SMS/MMS API integration."""

    TWILIO_API_BASE = "https://api.twilio.com/2010-04-01/Accounts"

    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.COMMUNICATION
        super().__init__(config)

        self.client: Optional[httpx.AsyncClient] = None

        # Check for discovery mode
        self.discovery_only = config.custom_config.get("discovery_only", False) if config.custom_config else False

        # Twilio credentials from config
        self.account_sid = config.credentials.get("account_sid") if config.credentials else None
        self.auth_token = config.credentials.get("auth_token") if config.credentials else None
        self.from_number = config.credentials.get("from_number") if config.credentials else None

        # Optional: auth token used for webhook signature validation (can differ from API auth token)
        self.webhook_auth_token = (
            config.credentials.get("webhook_auth_token", self.auth_token)
            if config.credentials else None
        )

        # Status callback URL for delivery receipts (optional)
        self.status_callback_url = config.credentials.get("status_callback_url") if config.credentials else None

        # Build the base URL for this account
        if self.account_sid:
            self.base_url = f"{self.TWILIO_API_BASE}/{self.account_sid}"
        else:
            self.base_url = None

        # Account info populated during initialize()
        self.account_info: Optional[Dict[str, Any]] = None

        # Skip validation in discovery mode
        if not self.discovery_only:
            if not self.account_sid or not self.auth_token:
                raise ValueError("Twilio account_sid and auth_token are required")
            if not self.from_number:
                raise ValueError("Twilio from_number is required")

    async def initialize(self) -> None:
        """Initialize the Twilio adapter and verify credentials."""
        # Skip initialization in discovery mode
        if self.discovery_only:
            logger.info("Twilio adapter initialized in discovery mode")
            return

        # Create HTTP client with Basic Auth (account_sid:auth_token)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=(self.account_sid, self.auth_token),
            timeout=self.config.timeout_seconds,
            headers={
                "Accept": "application/json",
            }
        )

        # Verify credentials by fetching account info
        try:
            response = await self.client.get(".json")
            response.raise_for_status()
            result = response.json()

            account_status = result.get("status")
            if account_status != "active":
                raise ValueError(
                    f"Twilio account is not active (status: {account_status})"
                )

            self.account_info = {
                "sid": result.get("sid"),
                "friendly_name": result.get("friendly_name"),
                "status": result.get("status"),
                "type": result.get("type"),
                "date_created": result.get("date_created"),
            }

            logger.info(
                f"Twilio adapter initialized for account: "
                f"{self.account_info['friendly_name']} ({self.account_info['sid']})"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to verify Twilio credentials: HTTP {e.response.status_code}")
            raise ValueError(f"Twilio authentication failed: {e.response.status_code}") from e
        except Exception as e:
            logger.error(f"Failed to initialize Twilio adapter: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the adapter and close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Twilio adapter shutdown")

    def get_capabilities(self) -> List[AdapterCapability]:
        """Return Twilio adapter capabilities."""
        return [
            AdapterCapability(
                name="send_sms",
                description="Send an SMS message via Twilio",
                category="messaging",
                parameters={
                    "to": {
                        "type": "string",
                        "description": "Recipient phone number in E.164 format (e.g., +15551234567)"
                    },
                    "body": {
                        "type": "string",
                        "description": "SMS message body (up to 1600 characters)"
                    },
                    "from_number": {
                        "type": "string",
                        "description": "Override sender phone number (E.164 format)"
                    },
                    "status_callback": {
                        "type": "string",
                        "description": "URL for delivery status webhooks"
                    },
                    "messaging_service_sid": {
                        "type": "string",
                        "description": "Twilio Messaging Service SID (alternative to from_number)"
                    },
                },
                required_parameters=["to", "body"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0079,
                cost_currency="USD",
            ),
            AdapterCapability(
                name="send_mms",
                description="Send an MMS message with media via Twilio",
                category="messaging",
                parameters={
                    "to": {
                        "type": "string",
                        "description": "Recipient phone number in E.164 format"
                    },
                    "body": {
                        "type": "string",
                        "description": "MMS message body"
                    },
                    "media_url": {
                        "type": "string|array",
                        "description": "URL(s) of media to attach (up to 10, publicly accessible)"
                    },
                    "from_number": {
                        "type": "string",
                        "description": "Override sender phone number (E.164 format)"
                    },
                    "status_callback": {
                        "type": "string",
                        "description": "URL for delivery status webhooks"
                    },
                    "messaging_service_sid": {
                        "type": "string",
                        "description": "Twilio Messaging Service SID (alternative to from_number)"
                    },
                },
                required_parameters=["to", "media_url"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.02,
                cost_currency="USD",
            ),
            AdapterCapability(
                name="get_message",
                description="Retrieve details of a sent or received Twilio message",
                category="messaging",
                parameters={
                    "message_sid": {
                        "type": "string",
                        "description": "Twilio Message SID (e.g., SMxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)"
                    },
                },
                required_parameters=["message_sid"],
                async_supported=True,
                estimated_duration_seconds=0.5,
            ),
        ]

    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request to Twilio."""
        # Validate request
        self.validate_request(request)

        # Route to appropriate handler
        handlers = {
            "send_sms": self._handle_send_sms,
            "send_mms": self._handle_send_mms,
            "get_message": self._handle_get_message,
        }

        handler = handlers.get(request.capability)
        if not handler:
            raise ValueError(f"Unknown capability: {request.capability}")

        return await handler(request)

    async def _handle_send_sms(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending an SMS message."""
        start_time = datetime.utcnow()

        try:
            # Build form data for Twilio Messages API
            form_data = {
                "To": request.parameters["to"],
                "Body": request.parameters["body"],
            }

            # Determine the sender: explicit from_number, messaging_service_sid, or default
            messaging_service_sid = request.parameters.get("messaging_service_sid")
            sender = request.parameters.get("from_number", self.from_number)

            if messaging_service_sid:
                form_data["MessagingServiceSid"] = messaging_service_sid
            else:
                form_data["From"] = sender

            # Optional status callback
            status_callback = request.parameters.get(
                "status_callback", self.status_callback_url
            )
            if status_callback:
                form_data["StatusCallback"] = status_callback

            # POST to /Messages.json (form-encoded, as Twilio requires)
            response = await self.client.post(
                "/Messages.json",
                data=form_data,
            )
            response.raise_for_status()
            result = response.json()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Publish SMS sent event
            await event_bus.publish(
                "adapter.twilio.sms_sent",
                {
                    "to": form_data["To"],
                    "from": result.get("from"),
                    "sid": result.get("sid"),
                    "status": result.get("status"),
                    "num_segments": result.get("num_segments"),
                },
                source_id=self.id,
                source_type="adapter",
            )

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "sid": result.get("sid"),
                    "to": result.get("to"),
                    "from": result.get("from"),
                    "body": result.get("body"),
                    "status": result.get("status"),
                    "direction": result.get("direction"),
                    "num_segments": result.get("num_segments"),
                    "price": result.get("price"),
                    "price_unit": result.get("price_unit"),
                    "date_created": result.get("date_created"),
                },
                duration_ms=duration_ms,
                cost=float(result["price"]) if result.get("price") else None,
                metadata={"account_sid": result.get("account_sid")},
            )

        except httpx.HTTPStatusError as e:
            error_body = self._parse_twilio_error(e.response)
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=error_body,
                error_code=str(e.response.status_code),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def _handle_send_mms(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending an MMS message with media."""
        start_time = datetime.utcnow()

        try:
            # Build form data
            form_data = {
                "To": request.parameters["to"],
            }

            # Body is optional for MMS (media can stand alone)
            if body := request.parameters.get("body"):
                form_data["Body"] = body

            # Determine the sender
            messaging_service_sid = request.parameters.get("messaging_service_sid")
            sender = request.parameters.get("from_number", self.from_number)

            if messaging_service_sid:
                form_data["MessagingServiceSid"] = messaging_service_sid
            else:
                form_data["From"] = sender

            # Handle media URLs -- Twilio accepts multiple MediaUrl params
            media_url = request.parameters["media_url"]
            media_urls = [media_url] if isinstance(media_url, str) else list(media_url)

            # Optional status callback
            status_callback = request.parameters.get(
                "status_callback", self.status_callback_url
            )
            if status_callback:
                form_data["StatusCallback"] = status_callback

            # Twilio accepts multiple MediaUrl fields in form-encoded data.
            # httpx handles repeated keys when we pass a list of tuples.
            form_items = list(form_data.items())
            for url in media_urls:
                form_items.append(("MediaUrl", url))

            # POST to /Messages.json
            response = await self.client.post(
                "/Messages.json",
                data=form_items,
            )
            response.raise_for_status()
            result = response.json()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Publish MMS sent event
            await event_bus.publish(
                "adapter.twilio.mms_sent",
                {
                    "to": result.get("to"),
                    "from": result.get("from"),
                    "sid": result.get("sid"),
                    "status": result.get("status"),
                    "num_media": result.get("num_media"),
                    "media_urls": media_urls,
                },
                source_id=self.id,
                source_type="adapter",
            )

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "sid": result.get("sid"),
                    "to": result.get("to"),
                    "from": result.get("from"),
                    "body": result.get("body"),
                    "status": result.get("status"),
                    "direction": result.get("direction"),
                    "num_media": result.get("num_media"),
                    "price": result.get("price"),
                    "price_unit": result.get("price_unit"),
                    "date_created": result.get("date_created"),
                    "media_urls": media_urls,
                },
                duration_ms=duration_ms,
                cost=float(result["price"]) if result.get("price") else None,
                metadata={"account_sid": result.get("account_sid")},
            )

        except httpx.HTTPStatusError as e:
            error_body = self._parse_twilio_error(e.response)
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=error_body,
                error_code=str(e.response.status_code),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def _handle_get_message(self, request: AdapterRequest) -> AdapterResponse:
        """Handle retrieving a message by SID."""
        start_time = datetime.utcnow()

        try:
            message_sid = request.parameters["message_sid"]

            response = await self.client.get(f"/Messages/{message_sid}.json")
            response.raise_for_status()
            result = response.json()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "sid": result.get("sid"),
                    "to": result.get("to"),
                    "from": result.get("from"),
                    "body": result.get("body"),
                    "status": result.get("status"),
                    "direction": result.get("direction"),
                    "num_segments": result.get("num_segments"),
                    "num_media": result.get("num_media"),
                    "price": result.get("price"),
                    "price_unit": result.get("price_unit"),
                    "error_code": result.get("error_code"),
                    "error_message": result.get("error_message"),
                    "date_created": result.get("date_created"),
                    "date_updated": result.get("date_updated"),
                    "date_sent": result.get("date_sent"),
                    "uri": result.get("uri"),
                },
                duration_ms=duration_ms,
                metadata={"account_sid": result.get("account_sid")},
            )

        except httpx.HTTPStatusError as e:
            error_body = self._parse_twilio_error(e.response)
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=error_body,
                error_code=str(e.response.status_code),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def process_webhook(self, payload: Dict[str, str]) -> Dict[str, Any]:
        """Process an inbound Twilio webhook payload.

        Twilio sends webhook data as form-encoded key-value pairs. The caller
        is responsible for parsing the form data into a dict before passing it
        here. Typical fields include:

            MessageSid, AccountSid, From, To, Body, NumMedia,
            MediaUrl0, MediaContentType0, ...

        Args:
            payload: Parsed form-encoded fields from the Twilio webhook POST.

        Returns:
            A normalized dict with the extracted message data, suitable for
            publishing on the event bus.
        """
        message_sid = payload.get("MessageSid", "")
        account_sid = payload.get("AccountSid", "")
        from_number = payload.get("From", "")
        to_number = payload.get("To", "")
        body = payload.get("Body", "")
        num_media = int(payload.get("NumMedia", "0"))
        sms_status = payload.get("SmsStatus", "")
        message_status = payload.get("MessageStatus", sms_status)

        # Collect media attachments (Twilio indexes them as MediaUrl0, MediaUrl1, ...)
        media = []
        for i in range(num_media):
            media_url = payload.get(f"MediaUrl{i}")
            media_content_type = payload.get(f"MediaContentType{i}")
            if media_url:
                media.append({
                    "url": media_url,
                    "content_type": media_content_type,
                })

        # Determine if this is a status callback or an inbound message
        is_status_callback = bool(message_status) and not body and not from_number
        is_inbound = bool(body) or num_media > 0

        normalized = {
            "message_sid": message_sid,
            "account_sid": account_sid,
            "from": from_number,
            "to": to_number,
            "body": body,
            "status": message_status,
            "num_media": num_media,
            "media": media,
            "is_inbound": is_inbound,
            "is_status_callback": is_status_callback,
            "raw_payload": payload,
        }

        # Publish the appropriate event
        if is_status_callback:
            await event_bus.publish(
                "adapter.twilio.status_callback",
                {
                    "message_sid": message_sid,
                    "status": message_status,
                    "to": to_number,
                },
                source_id=self.id,
                source_type="adapter",
            )
        else:
            await event_bus.publish(
                "adapter.twilio.message_received",
                {
                    "message_sid": message_sid,
                    "from": from_number,
                    "to": to_number,
                    "body": body,
                    "num_media": num_media,
                },
                source_id=self.id,
                source_type="adapter",
            )

        return normalized

    def validate_twilio_signature(
        self,
        url: str,
        params: Dict[str, str],
        signature: str,
        auth_token: Optional[str] = None,
    ) -> bool:
        """Validate an X-Twilio-Signature header to verify webhook authenticity.

        Twilio signs every webhook request. The signature is an HMAC-SHA1 of
        the full request URL with POST parameters appended (sorted by key),
        keyed with the account's auth token.

        See: https://www.twilio.com/docs/usage/security#validating-requests

        Args:
            url: The full URL that Twilio posted to (including scheme and query string).
            params: The POST body parameters as a dict.
            signature: The value of the X-Twilio-Signature header.
            auth_token: Auth token to use for validation. Falls back to
                        self.webhook_auth_token or self.auth_token.

        Returns:
            True if the signature is valid, False otherwise.
        """
        token = auth_token or self.webhook_auth_token or self.auth_token
        if not token:
            logger.warning("Cannot validate Twilio signature: no auth token available")
            return False

        # Build the data string: URL + sorted POST params concatenated
        data_string = url
        for key in sorted(params.keys()):
            data_string += key + params[key]

        # Compute HMAC-SHA1 and base64-encode
        import base64

        computed_sig = base64.b64encode(
            hmac.new(
                token.encode("utf-8"),
                data_string.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("utf-8")

        return hmac.compare_digest(computed_sig, signature)

    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Twilio-specific health check by fetching account info."""
        if self.discovery_only:
            return {"status": "healthy", "mode": "discovery"}

        try:
            response = await self.client.get(".json")
            response.raise_for_status()
            result = response.json()

            account_status = result.get("status", "unknown")

            if account_status == "active":
                return {
                    "status": "healthy",
                    "account_sid": result.get("sid"),
                    "account_name": result.get("friendly_name"),
                    "account_status": account_status,
                }
            else:
                return {
                    "status": "unhealthy",
                    "account_status": account_status,
                    "error": f"Twilio account status is '{account_status}', expected 'active'",
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def _parse_twilio_error(self, response: httpx.Response) -> str:
        """Extract a human-readable error message from a Twilio error response.

        Twilio error responses are JSON with 'code', 'message', and 'more_info'
        fields. This method tries to parse that structure and falls back to the
        raw response text.
        """
        try:
            error_data = response.json()
            code = error_data.get("code", "")
            message = error_data.get("message", "")
            more_info = error_data.get("more_info", "")
            return f"Twilio error {code}: {message} (see {more_info})"
        except Exception:
            return f"Twilio HTTP {response.status_code}: {response.text}"
