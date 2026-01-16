"""Stripe payment adapter implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import json
from datetime import datetime
from decimal import Decimal

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory
)
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class StripeAdapter(BaseAdapter):
    """Adapter for Stripe payment processing."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.PAYMENT
        super().__init__(config)
        
        # Discovery mode support
        self.discovery_only = config.custom_config.get("discovery_only", False) if config.custom_config else False
        
        self.client: Optional[httpx.AsyncClient] = None
        self.base_url = config.base_url or "https://api.stripe.com/v1"
        self.api_key = config.api_key or (config.credentials.get("secret_key") if config.credentials else None)
        
        # Skip validation in discovery mode
        if not self.discovery_only and not self.api_key:
            raise ValueError("Stripe API key is required")
    
    async def initialize(self) -> None:
        """Initialize the Stripe adapter."""
        # Skip initialization in discovery mode
        if self.discovery_only:
            logger.info("Stripe adapter initialized in discovery mode")
            return
        
        # Create HTTP client with Stripe-specific auth
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=(self.api_key, ""),  # Stripe uses basic auth with API key as username
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Stripe-Version": "2023-10-16"  # API version
            },
            timeout=self.config.timeout_seconds
        )
        
        # Test connection by fetching account info
        try:
            response = await self.client.get("/account")
            response.raise_for_status()
            logger.info("Stripe adapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Stripe adapter: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Stripe adapter shutdown")
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return Stripe adapter capabilities."""
        return [
            AdapterCapability(
                name="create_payment_intent",
                description="Create a payment intent for collecting payment",
                category="payment_processing",
                parameters={
                    "amount": {"type": "integer", "description": "Amount in cents (e.g., 1999 for $19.99)"},
                    "currency": {"type": "string", "description": "Three-letter currency code", "default": "usd"},
                    "description": {"type": "string", "description": "Payment description"},
                    "metadata": {"type": "object", "description": "Additional metadata"},
                    "customer": {"type": "string", "description": "Stripe customer ID"},
                    "payment_method_types": {"type": "array", "description": "Payment method types", "default": ["card"]},
                    "setup_future_usage": {"type": "string", "description": "Save payment method for future use"}
                },
                required_parameters=["amount", "currency"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0  # Stripe charges per transaction, not API call
            ),
            AdapterCapability(
                name="capture_payment",
                description="Capture a previously authorized payment",
                category="payment_processing",
                parameters={
                    "payment_intent_id": {"type": "string", "description": "Payment intent ID to capture"},
                    "amount_to_capture": {"type": "integer", "description": "Amount to capture (optional, defaults to full amount)"}
                },
                required_parameters=["payment_intent_id"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="create_customer",
                description="Create a new customer",
                category="customer_management",
                parameters={
                    "email": {"type": "string", "description": "Customer email"},
                    "name": {"type": "string", "description": "Customer name"},
                    "description": {"type": "string", "description": "Customer description"},
                    "metadata": {"type": "object", "description": "Additional metadata"},
                    "phone": {"type": "string", "description": "Customer phone"},
                    "address": {"type": "object", "description": "Customer address"}
                },
                required_parameters=["email"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="update_customer",
                description="Update an existing customer",
                category="customer_management",
                parameters={
                    "customer_id": {"type": "string", "description": "Customer ID to update"},
                    "email": {"type": "string", "description": "New email"},
                    "name": {"type": "string", "description": "New name"},
                    "description": {"type": "string", "description": "New description"},
                    "metadata": {"type": "object", "description": "New metadata"}
                },
                required_parameters=["customer_id"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="create_subscription",
                description="Create a subscription for a customer",
                category="subscription_management",
                parameters={
                    "customer": {"type": "string", "description": "Customer ID"},
                    "items": {"type": "array", "description": "Subscription items with price IDs"},
                    "trial_period_days": {"type": "integer", "description": "Trial period in days"},
                    "metadata": {"type": "object", "description": "Additional metadata"},
                    "payment_behavior": {"type": "string", "description": "Payment behavior", "default": "default_incomplete"}
                },
                required_parameters=["customer", "items"],
                async_supported=True,
                estimated_duration_seconds=1.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="cancel_subscription",
                description="Cancel a subscription",
                category="subscription_management",
                parameters={
                    "subscription_id": {"type": "string", "description": "Subscription ID to cancel"},
                    "cancel_at_period_end": {"type": "boolean", "description": "Cancel at end of period", "default": False},
                    "cancellation_details": {"type": "object", "description": "Cancellation details"}
                },
                required_parameters=["subscription_id"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="create_refund",
                description="Create a refund for a payment",
                category="payment_processing",
                parameters={
                    "payment_intent": {"type": "string", "description": "Payment intent ID to refund"},
                    "amount": {"type": "integer", "description": "Amount to refund in cents (optional, defaults to full)"},
                    "reason": {"type": "string", "description": "Refund reason"},
                    "metadata": {"type": "object", "description": "Additional metadata"}
                },
                required_parameters=["payment_intent"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="list_transactions",
                description="List transactions with filtering",
                category="reporting",
                parameters={
                    "limit": {"type": "integer", "description": "Number of records to return", "default": 10},
                    "starting_after": {"type": "string", "description": "Cursor for pagination"},
                    "created": {"type": "object", "description": "Filter by creation date"},
                    "customer": {"type": "string", "description": "Filter by customer ID"},
                    "status": {"type": "string", "description": "Filter by status"}
                },
                required_parameters=[],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="create_payment_method",
                description="Create a payment method",
                category="payment_processing",
                parameters={
                    "type": {"type": "string", "description": "Payment method type (card, bank_account, etc.)"},
                    "card": {"type": "object", "description": "Card details (for PCI compliance, use tokens)"},
                    "billing_details": {"type": "object", "description": "Billing details"},
                    "metadata": {"type": "object", "description": "Additional metadata"}
                },
                required_parameters=["type"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="attach_payment_method",
                description="Attach a payment method to a customer",
                category="payment_processing",
                parameters={
                    "payment_method_id": {"type": "string", "description": "Payment method ID"},
                    "customer": {"type": "string", "description": "Customer ID"}
                },
                required_parameters=["payment_method_id", "customer"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            )
        ]
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request to Stripe."""
        # Validate request
        self.validate_request(request)
        
        # Route to appropriate handler
        capability_handlers = {
            "create_payment_intent": self._handle_create_payment_intent,
            "capture_payment": self._handle_capture_payment,
            "create_customer": self._handle_create_customer,
            "update_customer": self._handle_update_customer,
            "create_subscription": self._handle_create_subscription,
            "cancel_subscription": self._handle_cancel_subscription,
            "create_refund": self._handle_create_refund,
            "list_transactions": self._handle_list_transactions,
            "create_payment_method": self._handle_create_payment_method,
            "attach_payment_method": self._handle_attach_payment_method
        }
        
        handler = capability_handlers.get(request.capability)
        if not handler:
            raise ValueError(f"Unknown capability: {request.capability}")
        
        return await handler(request)
    
    async def _handle_create_payment_intent(self, request: AdapterRequest) -> AdapterResponse:
        """Handle payment intent creation."""
        start_time = datetime.utcnow()
        
        try:
            # Prepare form data (Stripe uses form encoding)
            data = {
                "amount": request.parameters["amount"],
                "currency": request.parameters.get("currency", "usd")
            }
            
            # Add optional parameters
            if "description" in request.parameters:
                data["description"] = request.parameters["description"]
            if "customer" in request.parameters:
                data["customer"] = request.parameters["customer"]
            if "metadata" in request.parameters:
                # Convert metadata dict to form fields
                for key, value in request.parameters["metadata"].items():
                    data[f"metadata[{key}]"] = str(value)
            if "payment_method_types" in request.parameters:
                # Convert array to form fields
                for i, pm_type in enumerate(request.parameters["payment_method_types"]):
                    data[f"payment_method_types[{i}]"] = pm_type
            if "setup_future_usage" in request.parameters:
                data["setup_future_usage"] = request.parameters["setup_future_usage"]
            
            # Make API request
            response = await self.client.post("/payment_intents", data=data)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Publish event
            await event_bus.publish(
                "adapter.stripe.payment_intent.created",
                {
                    "payment_intent_id": result["id"],
                    "amount": result["amount"],
                    "currency": result["currency"],
                    "status": result["status"]
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "payment_intent": result
                },
                duration_ms=duration_ms,
                cost=0.0,  # Stripe charges per transaction, not API call
                metadata={
                    "payment_intent_id": result["id"],
                    "client_secret": result.get("client_secret")
                }
            )
            
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_message = error_data.get("error", {}).get("message", str(e))
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=error_message,
                error_code=f"HTTP_{e.response.status_code}",
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_capture_payment(self, request: AdapterRequest) -> AdapterResponse:
        """Handle payment capture."""
        start_time = datetime.utcnow()
        
        try:
            payment_intent_id = request.parameters["payment_intent_id"]
            
            data = {}
            if "amount_to_capture" in request.parameters:
                data["amount_to_capture"] = request.parameters["amount_to_capture"]
            
            response = await self.client.post(
                f"/payment_intents/{payment_intent_id}/capture",
                data=data if data else None
            )
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Publish event
            await event_bus.publish(
                "adapter.stripe.payment_intent.captured",
                {
                    "payment_intent_id": result["id"],
                    "amount_captured": result.get("amount_captured", result["amount"]),
                    "status": result["status"]
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "payment_intent": result
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_create_customer(self, request: AdapterRequest) -> AdapterResponse:
        """Handle customer creation."""
        start_time = datetime.utcnow()
        
        try:
            data = {
                "email": request.parameters["email"]
            }
            
            # Add optional parameters
            if "name" in request.parameters:
                data["name"] = request.parameters["name"]
            if "description" in request.parameters:
                data["description"] = request.parameters["description"]
            if "phone" in request.parameters:
                data["phone"] = request.parameters["phone"]
            if "metadata" in request.parameters:
                for key, value in request.parameters["metadata"].items():
                    data[f"metadata[{key}]"] = str(value)
            if "address" in request.parameters:
                addr = request.parameters["address"]
                for field in ["line1", "line2", "city", "state", "postal_code", "country"]:
                    if field in addr:
                        data[f"address[{field}]"] = addr[field]
            
            response = await self.client.post("/customers", data=data)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Publish event
            await event_bus.publish(
                "adapter.stripe.customer.created",
                {
                    "customer_id": result["id"],
                    "email": result["email"]
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "customer": result
                },
                duration_ms=duration_ms,
                cost=0.0,
                metadata={"customer_id": result["id"]}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_update_customer(self, request: AdapterRequest) -> AdapterResponse:
        """Handle customer update."""
        start_time = datetime.utcnow()
        
        try:
            customer_id = request.parameters["customer_id"]
            
            data = {}
            for field in ["email", "name", "description", "phone"]:
                if field in request.parameters:
                    data[field] = request.parameters[field]
            
            if "metadata" in request.parameters:
                for key, value in request.parameters["metadata"].items():
                    data[f"metadata[{key}]"] = str(value)
            
            response = await self.client.post(f"/customers/{customer_id}", data=data)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "customer": result
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_create_subscription(self, request: AdapterRequest) -> AdapterResponse:
        """Handle subscription creation."""
        start_time = datetime.utcnow()
        
        try:
            data = {
                "customer": request.parameters["customer"]
            }
            
            # Add subscription items
            items = request.parameters["items"]
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    data[f"items[{i}][price]"] = item.get("price")
                    if "quantity" in item:
                        data[f"items[{i}][quantity]"] = item["quantity"]
                else:
                    # Simple price ID
                    data[f"items[{i}][price]"] = item
            
            # Add optional parameters
            if "trial_period_days" in request.parameters:
                data["trial_period_days"] = request.parameters["trial_period_days"]
            if "payment_behavior" in request.parameters:
                data["payment_behavior"] = request.parameters["payment_behavior"]
            if "metadata" in request.parameters:
                for key, value in request.parameters["metadata"].items():
                    data[f"metadata[{key}]"] = str(value)
            
            response = await self.client.post("/subscriptions", data=data)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Publish event
            await event_bus.publish(
                "adapter.stripe.subscription.created",
                {
                    "subscription_id": result["id"],
                    "customer": result["customer"],
                    "status": result["status"]
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "subscription": result
                },
                duration_ms=duration_ms,
                cost=0.0,
                metadata={"subscription_id": result["id"]}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_cancel_subscription(self, request: AdapterRequest) -> AdapterResponse:
        """Handle subscription cancellation."""
        start_time = datetime.utcnow()
        
        try:
            subscription_id = request.parameters["subscription_id"]
            
            data = {}
            if request.parameters.get("cancel_at_period_end", False):
                # Soft cancel - subscription remains active until period end
                data["cancel_at_period_end"] = "true"
            else:
                # Immediate cancel
                response = await self.client.delete(f"/subscriptions/{subscription_id}")
                response.raise_for_status()
                result = response.json()
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                await event_bus.publish(
                    "adapter.stripe.subscription.cancelled",
                    {
                        "subscription_id": result["id"],
                        "cancelled_at": result.get("canceled_at")
                    },
                    source_id=self.id,
                    source_type="adapter"
                )
                
                return AdapterResponse(
                    request_id=request.id,
                    capability=request.capability,
                    status="success",
                    data={"subscription": result},
                    duration_ms=duration_ms,
                    cost=0.0
                )
            
            # For soft cancel
            if "cancellation_details" in request.parameters:
                details = request.parameters["cancellation_details"]
                if "comment" in details:
                    data["cancellation_details[comment]"] = details["comment"]
                if "feedback" in details:
                    data["cancellation_details[feedback]"] = details["feedback"]
            
            response = await self.client.post(f"/subscriptions/{subscription_id}", data=data)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "subscription": result
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_create_refund(self, request: AdapterRequest) -> AdapterResponse:
        """Handle refund creation."""
        start_time = datetime.utcnow()
        
        try:
            data = {
                "payment_intent": request.parameters["payment_intent"]
            }
            
            # Add optional parameters
            if "amount" in request.parameters:
                data["amount"] = request.parameters["amount"]
            if "reason" in request.parameters:
                data["reason"] = request.parameters["reason"]
            if "metadata" in request.parameters:
                for key, value in request.parameters["metadata"].items():
                    data[f"metadata[{key}]"] = str(value)
            
            response = await self.client.post("/refunds", data=data)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Publish event
            await event_bus.publish(
                "adapter.stripe.refund.created",
                {
                    "refund_id": result["id"],
                    "amount": result["amount"],
                    "status": result["status"]
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "refund": result
                },
                duration_ms=duration_ms,
                cost=0.0,
                metadata={"refund_id": result["id"]}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_list_transactions(self, request: AdapterRequest) -> AdapterResponse:
        """Handle transaction listing."""
        start_time = datetime.utcnow()
        
        try:
            params = {
                "limit": request.parameters.get("limit", 10)
            }
            
            # Add optional filters
            if "starting_after" in request.parameters:
                params["starting_after"] = request.parameters["starting_after"]
            if "customer" in request.parameters:
                params["customer"] = request.parameters["customer"]
            if "status" in request.parameters:
                params["status"] = request.parameters["status"]
            if "created" in request.parameters:
                created = request.parameters["created"]
                if isinstance(created, dict):
                    for key, value in created.items():
                        params[f"created[{key}]"] = value
            
            # Use charges endpoint for transactions
            response = await self.client.get("/charges", params=params)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "transactions": result.get("data", []),
                    "has_more": result.get("has_more", False),
                    "total_count": len(result.get("data", []))
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_create_payment_method(self, request: AdapterRequest) -> AdapterResponse:
        """Handle payment method creation."""
        start_time = datetime.utcnow()
        
        try:
            data = {
                "type": request.parameters["type"]
            }
            
            # Add type-specific data
            if request.parameters["type"] == "card" and "card" in request.parameters:
                # Note: In production, use Stripe.js or Elements for PCI compliance
                card = request.parameters["card"]
                for field in ["number", "exp_month", "exp_year", "cvc"]:
                    if field in card:
                        data[f"card[{field}]"] = card[field]
            
            # Add billing details
            if "billing_details" in request.parameters:
                billing = request.parameters["billing_details"]
                for field in ["email", "name", "phone"]:
                    if field in billing:
                        data[f"billing_details[{field}]"] = billing[field]
                
                if "address" in billing:
                    addr = billing["address"]
                    for field in ["line1", "line2", "city", "state", "postal_code", "country"]:
                        if field in addr:
                            data[f"billing_details[address][{field}]"] = addr[field]
            
            if "metadata" in request.parameters:
                for key, value in request.parameters["metadata"].items():
                    data[f"metadata[{key}]"] = str(value)
            
            response = await self.client.post("/payment_methods", data=data)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "payment_method": result
                },
                duration_ms=duration_ms,
                cost=0.0,
                metadata={"payment_method_id": result["id"]}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_attach_payment_method(self, request: AdapterRequest) -> AdapterResponse:
        """Handle attaching payment method to customer."""
        start_time = datetime.utcnow()
        
        try:
            payment_method_id = request.parameters["payment_method_id"]
            
            data = {
                "customer": request.parameters["customer"]
            }
            
            response = await self.client.post(
                f"/payment_methods/{payment_method_id}/attach",
                data=data
            )
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "payment_method": result
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Stripe-specific health check."""
        try:
            # Check account endpoint
            response = await self.client.get("/account")
            response.raise_for_status()
            
            account = response.json()
            
            return {
                "status": "healthy",
                "account_id": account.get("id"),
                "account_type": account.get("type"),
                "charges_enabled": account.get("charges_enabled", False),
                "payouts_enabled": account.get("payouts_enabled", False)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }