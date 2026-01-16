"""Fiverr adapter implementation for hiring freelancers."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
import httpx
import json
from datetime import datetime
import time

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory, AdapterStatus
)
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class FiverrAdapter(BaseAdapter):
    """Adapter for Fiverr API integration."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.HUMAN
        super().__init__(config)
        
        self.client: Optional[httpx.AsyncClient] = None
        
        # Extract configuration
        self.api_key = config.api_key or config.credentials.get("api_key")
        self.api_secret = config.credentials.get("api_secret")
        self.base_url = config.base_url or "https://api.fiverr.com/v1"
        
        if not self.api_key:
            raise ValueError("Fiverr API key is required")
    
    async def initialize(self) -> None:
        """Initialize the Fiverr adapter."""
        # Create HTTP client
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if self.api_secret:
            headers["X-API-Secret"] = self.api_secret
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.config.timeout_seconds
        )
        
        # Test connection
        try:
            # Note: Fiverr's actual API endpoints may differ
            response = await self.client.get("/categories")
            response.raise_for_status()
            logger.info("Fiverr adapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Fiverr adapter: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Fiverr adapter shutdown")
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return Fiverr adapter capabilities."""
        return [
            AdapterCapability(
                name="search_gigs",
                description="Search for gigs (services) on Fiverr",
                category="gig_search",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "description": "Category slug"},
                    "subcategory": {"type": "string", "description": "Subcategory slug"},
                    "min_price": {"type": "number", "description": "Minimum price"},
                    "max_price": {"type": "number", "description": "Maximum price"},
                    "delivery_time": {"type": "integer", "description": "Max delivery days"},
                    "seller_level": {"type": "array", "description": "Seller levels (new, level_one, level_two, top_rated)"},
                    "language": {"type": "string", "description": "Seller language"},
                    "sort_by": {"type": "string", "description": "Sort option", "default": "relevance"},
                    "limit": {"type": "integer", "description": "Number of results", "default": 20}
                },
                required_parameters=["query"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.01
            ),
            AdapterCapability(
                name="get_gig_details",
                description="Get detailed information about a gig",
                category="gig_management",
                parameters={
                    "gig_id": {"type": "string", "description": "Gig ID"}
                },
                required_parameters=["gig_id"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.005
            ),
            AdapterCapability(
                name="get_seller_profile",
                description="Get seller profile information",
                category="seller_management",
                parameters={
                    "seller_id": {"type": "string", "description": "Seller username or ID"}
                },
                required_parameters=["seller_id"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.005
            ),
            AdapterCapability(
                name="create_order",
                description="Create an order for a gig",
                category="order_management",
                parameters={
                    "gig_id": {"type": "string", "description": "Gig ID to order"},
                    "package_id": {"type": "string", "description": "Package ID (basic, standard, premium)"},
                    "quantity": {"type": "integer", "description": "Order quantity", "default": 1},
                    "requirements": {"type": "object", "description": "Order requirements"},
                    "extras": {"type": "array", "description": "Gig extras to include"}
                },
                required_parameters=["gig_id", "package_id"],
                async_supported=True,
                estimated_duration_seconds=3.0,
                cost_per_request=0.02
            ),
            AdapterCapability(
                name="send_message",
                description="Send a message to a seller",
                category="communication",
                parameters={
                    "seller_id": {"type": "string", "description": "Seller ID"},
                    "subject": {"type": "string", "description": "Message subject"},
                    "message": {"type": "string", "description": "Message content"},
                    "gig_id": {"type": "string", "description": "Related gig ID (optional)"}
                },
                required_parameters=["seller_id", "message"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.005
            ),
            AdapterCapability(
                name="get_order_status",
                description="Get the status of an order",
                category="order_tracking",
                parameters={
                    "order_id": {"type": "string", "description": "Order ID"}
                },
                required_parameters=["order_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.002
            ),
            AdapterCapability(
                name="submit_requirements",
                description="Submit requirements for an order",
                category="order_management",
                parameters={
                    "order_id": {"type": "string", "description": "Order ID"},
                    "requirements": {"type": "object", "description": "Requirements data"},
                    "attachments": {"type": "array", "description": "File attachments"}
                },
                required_parameters=["order_id", "requirements"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.01
            ),
            AdapterCapability(
                name="request_revision",
                description="Request a revision for delivered work",
                category="order_management",
                parameters={
                    "order_id": {"type": "string", "description": "Order ID"},
                    "delivery_id": {"type": "string", "description": "Delivery ID"},
                    "message": {"type": "string", "description": "Revision request details"}
                },
                required_parameters=["order_id", "delivery_id", "message"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.005
            ),
            AdapterCapability(
                name="accept_delivery",
                description="Accept a delivery and release payment",
                category="order_completion",
                parameters={
                    "order_id": {"type": "string", "description": "Order ID"},
                    "delivery_id": {"type": "string", "description": "Delivery ID"},
                    "rating": {"type": "integer", "description": "Rating (1-5)"},
                    "review": {"type": "string", "description": "Review text"}
                },
                required_parameters=["order_id", "delivery_id"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.01
            ),
            AdapterCapability(
                name="get_categories",
                description="Get Fiverr service categories",
                category="metadata",
                parameters={
                    "parent_id": {"type": "string", "description": "Parent category ID (optional)"}
                },
                required_parameters=[],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.001
            ),
            AdapterCapability(
                name="create_custom_offer",
                description="Create a custom offer for a buyer",
                category="custom_orders",
                parameters={
                    "buyer_id": {"type": "string", "description": "Buyer username"},
                    "description": {"type": "string", "description": "Offer description"},
                    "price": {"type": "number", "description": "Offer price"},
                    "delivery_time": {"type": "integer", "description": "Delivery time in days"},
                    "revisions": {"type": "integer", "description": "Number of revisions"},
                    "extras": {"type": "array", "description": "Additional services"}
                },
                required_parameters=["buyer_id", "description", "price", "delivery_time"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.01
            ),
            AdapterCapability(
                name="get_seller_analytics",
                description="Get seller performance analytics",
                category="analytics",
                parameters={
                    "seller_id": {"type": "string", "description": "Seller ID"},
                    "date_from": {"type": "string", "description": "Start date (ISO format)"},
                    "date_to": {"type": "string", "description": "End date (ISO format)"},
                    "metrics": {"type": "array", "description": "Metrics to retrieve"}
                },
                required_parameters=["seller_id"],
                async_supported=True,
                estimated_duration_seconds=1.5,
                cost_per_request=0.01
            )
        ]
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request using Fiverr."""
        if not self._initialized or self.status != AdapterStatus.RUNNING:
            raise RuntimeError("Adapter not initialized or not running")
        
        if not self.client:
            raise RuntimeError("HTTP client not initialized")
        
        start_time = time.time()
        
        try:
            # Apply rate limiting if configured
            if self._rate_limiter:
                async with self._rate_limiter:
                    response = await self._execute_request(request)
            else:
                response = await self._execute_request(request)
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.total_latency += (time.time() - start_time)
            
            # Publish event
            await event_bus.publish({
                "type": "adapter.request.completed",
                "adapter_id": self.id,
                "capability": request.capability,
                "duration": time.time() - start_time,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return response
            
        except Exception as e:
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.total_latency += (time.time() - start_time)
            
            # Publish error event
            await event_bus.publish({
                "type": "adapter.request.failed",
                "adapter_id": self.id,
                "capability": request.capability,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.error(f"Error executing Fiverr request: {str(e)}")
            raise
    
    async def _execute_request(self, request: AdapterRequest) -> AdapterResponse:
        """Execute the actual Fiverr request."""
        capability = request.capability
        params = request.parameters
        
        capability_map = {
            "search_gigs": self._search_gigs,
            "get_gig_details": self._get_gig_details,
            "get_seller_profile": self._get_seller_profile,
            "create_order": self._create_order,
            "send_message": self._send_message,
            "get_order_status": self._get_order_status,
            "submit_requirements": self._submit_requirements,
            "request_revision": self._request_revision,
            "accept_delivery": self._accept_delivery,
            "get_categories": self._get_categories,
            "create_custom_offer": self._create_custom_offer,
            "get_seller_analytics": self._get_seller_analytics
        }
        
        handler = capability_map.get(capability)
        if not handler:
            raise ValueError(f"Unknown capability: {capability}")
        
        return await handler(params)
    
    async def _search_gigs(self, params: Dict[str, Any]) -> AdapterResponse:
        """Search for gigs."""
        search_params = {
            "query": params["query"],
            "limit": params.get("limit", 20),
            "sort_by": params.get("sort_by", "relevance")
        }
        
        # Add optional filters
        if "category" in params:
            search_params["category"] = params["category"]
        if "subcategory" in params:
            search_params["subcategory"] = params["subcategory"]
        if "min_price" in params:
            search_params["min_price"] = params["min_price"]
        if "max_price" in params:
            search_params["max_price"] = params["max_price"]
        if "delivery_time" in params:
            search_params["delivery_time"] = params["delivery_time"]
        if "seller_level" in params:
            search_params["seller_level"] = ",".join(params["seller_level"])
        if "language" in params:
            search_params["language"] = params["language"]
        
        response = await self.client.get("/gigs/search", params=search_params)
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "gigs": result.get("gigs", []),
                "total": result.get("total", 0),
                "page": result.get("page", 1),
                "per_page": result.get("per_page", 20)
            },
            metadata={
                "query": params["query"],
                "filters_applied": len(search_params) - 3  # Minus query, limit, sort_by
            }
        )
    
    async def _get_gig_details(self, params: Dict[str, Any]) -> AdapterResponse:
        """Get gig details."""
        gig_id = params["gig_id"]
        
        response = await self.client.get(f"/gigs/{gig_id}")
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "gig": result.get("gig", {}),
                "packages": result.get("gig", {}).get("packages", []),
                "extras": result.get("gig", {}).get("extras", []),
                "faqs": result.get("gig", {}).get("faqs", [])
            },
            metadata={
                "gig_id": gig_id
            }
        )
    
    async def _get_seller_profile(self, params: Dict[str, Any]) -> AdapterResponse:
        """Get seller profile."""
        seller_id = params["seller_id"]
        
        response = await self.client.get(f"/sellers/{seller_id}")
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "seller": result.get("seller", {}),
                "stats": result.get("seller", {}).get("stats", {}),
                "skills": result.get("seller", {}).get("skills", []),
                "languages": result.get("seller", {}).get("languages", [])
            },
            metadata={
                "seller_id": seller_id
            }
        )
    
    async def _create_order(self, params: Dict[str, Any]) -> AdapterResponse:
        """Create an order."""
        order_data = {
            "gig_id": params["gig_id"],
            "package_id": params["package_id"],
            "quantity": params.get("quantity", 1),
            "extras": params.get("extras", [])
        }
        
        if "requirements" in params:
            order_data["requirements"] = params["requirements"]
        
        response = await self.client.post("/orders", json=order_data)
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "order_id": result.get("order", {}).get("id"),
                "status": result.get("order", {}).get("status", "pending"),
                "total_price": result.get("order", {}).get("total_price"),
                "delivery_date": result.get("order", {}).get("delivery_date")
            },
            metadata={
                "gig_id": params["gig_id"],
                "package_id": params["package_id"]
            }
        )
    
    async def _send_message(self, params: Dict[str, Any]) -> AdapterResponse:
        """Send a message."""
        message_data = {
            "recipient_id": params["seller_id"],
            "message": params["message"],
            "subject": params.get("subject", "New Message")
        }
        
        if "gig_id" in params:
            message_data["gig_id"] = params["gig_id"]
        
        response = await self.client.post("/messages", json=message_data)
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "message_id": result.get("message", {}).get("id"),
                "thread_id": result.get("message", {}).get("thread_id"),
                "sent_at": result.get("message", {}).get("sent_at")
            },
            metadata={
                "recipient_id": params["seller_id"]
            }
        )
    
    async def _get_order_status(self, params: Dict[str, Any]) -> AdapterResponse:
        """Get order status."""
        order_id = params["order_id"]
        
        response = await self.client.get(f"/orders/{order_id}")
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "order": result.get("order", {}),
                "status": result.get("order", {}).get("status"),
                "deliveries": result.get("order", {}).get("deliveries", []),
                "time_left": result.get("order", {}).get("time_left")
            },
            metadata={
                "order_id": order_id
            }
        )
    
    async def _submit_requirements(self, params: Dict[str, Any]) -> AdapterResponse:
        """Submit order requirements."""
        order_id = params["order_id"]
        requirements_data = {
            "requirements": params["requirements"],
            "attachments": params.get("attachments", [])
        }
        
        response = await self.client.post(
            f"/orders/{order_id}/requirements",
            json=requirements_data
        )
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "submitted": True,
                "order_status": result.get("order", {}).get("status", "in_progress")
            },
            metadata={
                "order_id": order_id
            }
        )
    
    async def _request_revision(self, params: Dict[str, Any]) -> AdapterResponse:
        """Request a revision."""
        order_id = params["order_id"]
        delivery_id = params["delivery_id"]
        
        revision_data = {
            "message": params["message"]
        }
        
        response = await self.client.post(
            f"/orders/{order_id}/deliveries/{delivery_id}/revisions",
            json=revision_data
        )
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "revision_id": result.get("revision", {}).get("id"),
                "status": "revision_requested"
            },
            metadata={
                "order_id": order_id,
                "delivery_id": delivery_id
            }
        )
    
    async def _accept_delivery(self, params: Dict[str, Any]) -> AdapterResponse:
        """Accept a delivery."""
        order_id = params["order_id"]
        delivery_id = params["delivery_id"]
        
        acceptance_data = {}
        if "rating" in params:
            acceptance_data["rating"] = params["rating"]
        if "review" in params:
            acceptance_data["review"] = params["review"]
        
        response = await self.client.post(
            f"/orders/{order_id}/deliveries/{delivery_id}/accept",
            json=acceptance_data
        )
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "accepted": True,
                "order_status": "completed",
                "review_id": result.get("review", {}).get("id") if "review" in params else None
            },
            metadata={
                "order_id": order_id,
                "delivery_id": delivery_id
            }
        )
    
    async def _get_categories(self, params: Dict[str, Any]) -> AdapterResponse:
        """Get categories."""
        query_params = {}
        if "parent_id" in params:
            query_params["parent_id"] = params["parent_id"]
        
        response = await self.client.get("/categories", params=query_params)
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "categories": result.get("categories", []),
                "total": len(result.get("categories", []))
            },
            metadata={
                "parent_id": params.get("parent_id", "root")
            }
        )
    
    async def _create_custom_offer(self, params: Dict[str, Any]) -> AdapterResponse:
        """Create a custom offer."""
        offer_data = {
            "buyer_username": params["buyer_id"],
            "description": params["description"],
            "price": params["price"],
            "delivery_time": params["delivery_time"],
            "revisions": params.get("revisions", 1),
            "extras": params.get("extras", [])
        }
        
        response = await self.client.post("/custom-offers", json=offer_data)
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "offer_id": result.get("offer", {}).get("id"),
                "status": result.get("offer", {}).get("status", "pending"),
                "expires_at": result.get("offer", {}).get("expires_at")
            },
            metadata={
                "buyer_id": params["buyer_id"],
                "price": params["price"]
            }
        )
    
    async def _get_seller_analytics(self, params: Dict[str, Any]) -> AdapterResponse:
        """Get seller analytics."""
        seller_id = params["seller_id"]
        
        query_params = {}
        if "date_from" in params:
            query_params["from"] = params["date_from"]
        if "date_to" in params:
            query_params["to"] = params["date_to"]
        if "metrics" in params:
            query_params["metrics"] = ",".join(params["metrics"])
        
        response = await self.client.get(
            f"/sellers/{seller_id}/analytics",
            params=query_params
        )
        response.raise_for_status()
        result = response.json()
        
        return AdapterResponse(
            success=True,
            data={
                "analytics": result.get("analytics", {}),
                "period": {
                    "from": query_params.get("from", "all_time"),
                    "to": query_params.get("to", "now")
                }
            },
            metadata={
                "seller_id": seller_id,
                "metrics_count": len(params.get("metrics", []))
            }
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Fiverr connection."""
        if not self.client:
            return {
                "status": "unhealthy",
                "error": "Client not initialized"
            }
        
        try:
            # Test by getting categories
            response = await self.client.get("/categories", params={"limit": 1})
            response.raise_for_status()
            
            return {
                "status": "healthy",
                "api_version": "v1",
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }