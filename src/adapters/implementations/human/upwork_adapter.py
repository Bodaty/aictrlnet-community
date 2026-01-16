"""Upwork adapter implementation for hiring freelancers."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
import httpx
import json
from datetime import datetime
import time
from urllib.parse import urlencode

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory, AdapterStatus
)
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class UpworkAdapter(BaseAdapter):
    """Adapter for Upwork API integration."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.HUMAN
        super().__init__(config)
        
        self.client: Optional[httpx.AsyncClient] = None
        
        # Extract configuration
        self.client_id = config.credentials.get("client_id")
        self.client_secret = config.credentials.get("client_secret")
        self.access_token = config.credentials.get("access_token")
        self.access_secret = config.credentials.get("access_secret")
        self.base_url = config.base_url or "https://www.upwork.com/api"
        
        if not all([self.client_id, self.client_secret, self.access_token, self.access_secret]):
            raise ValueError("Upwork OAuth credentials required (client_id, client_secret, access_token, access_secret)")
    
    async def initialize(self) -> None:
        """Initialize the Upwork adapter."""
        # Upwork uses OAuth 1.0a, which requires signing requests
        # For simplicity, we'll assume tokens are already obtained
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=self.config.timeout_seconds
        )
        
        # Test connection by getting user info
        try:
            # Note: Real implementation would need OAuth 1.0a signing
            response = await self._make_request("GET", "/auth/v1/info.json")
            logger.info("Upwork adapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Upwork adapter: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Upwork adapter shutdown")
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return Upwork adapter capabilities."""
        return [
            AdapterCapability(
                name="search_freelancers",
                description="Search for freelancers on Upwork",
                category="freelancer_search",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "skills": {"type": "array", "description": "Required skills"},
                    "category": {"type": "string", "description": "Job category"},
                    "subcategory": {"type": "string", "description": "Job subcategory"},
                    "hourly_rate_min": {"type": "number", "description": "Minimum hourly rate"},
                    "hourly_rate_max": {"type": "number", "description": "Maximum hourly rate"},
                    "location": {"type": "string", "description": "Freelancer location"},
                    "english_level": {"type": "string", "description": "English proficiency level"},
                    "limit": {"type": "integer", "description": "Number of results", "default": 20}
                },
                required_parameters=["query"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.01
            ),
            AdapterCapability(
                name="get_freelancer_profile",
                description="Get detailed freelancer profile",
                category="freelancer_management",
                parameters={
                    "freelancer_id": {"type": "string", "description": "Freelancer ID or username"}
                },
                required_parameters=["freelancer_id"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.005
            ),
            AdapterCapability(
                name="post_job",
                description="Post a job on Upwork",
                category="job_posting",
                parameters={
                    "title": {"type": "string", "description": "Job title"},
                    "description": {"type": "string", "description": "Job description"},
                    "category": {"type": "string", "description": "Job category"},
                    "subcategory": {"type": "string", "description": "Job subcategory"},
                    "skills": {"type": "array", "description": "Required skills"},
                    "scope": {"type": "string", "description": "Project scope (small, medium, large)"},
                    "duration": {"type": "string", "description": "Project duration"},
                    "budget": {"type": "number", "description": "Project budget"},
                    "hourly": {"type": "boolean", "description": "Hourly or fixed price", "default": False},
                    "visibility": {"type": "string", "description": "Job visibility", "default": "public"}
                },
                required_parameters=["title", "description", "category"],
                async_supported=True,
                estimated_duration_seconds=3.0,
                cost_per_request=0.02
            ),
            AdapterCapability(
                name="invite_to_job",
                description="Invite a freelancer to apply for a job",
                category="job_management",
                parameters={
                    "job_id": {"type": "string", "description": "Job ID"},
                    "freelancer_id": {"type": "string", "description": "Freelancer ID"},
                    "message": {"type": "string", "description": "Invitation message"}
                },
                required_parameters=["job_id", "freelancer_id"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.01
            ),
            AdapterCapability(
                name="send_offer",
                description="Send a job offer to a freelancer",
                category="hiring",
                parameters={
                    "freelancer_id": {"type": "string", "description": "Freelancer ID"},
                    "job_id": {"type": "string", "description": "Job ID"},
                    "title": {"type": "string", "description": "Offer title"},
                    "message": {"type": "string", "description": "Offer message"},
                    "hourly_rate": {"type": "number", "description": "Hourly rate (for hourly jobs)"},
                    "fixed_price": {"type": "number", "description": "Fixed price (for fixed jobs)"},
                    "weekly_limit": {"type": "integer", "description": "Weekly hour limit (hourly only)"},
                    "start_date": {"type": "string", "description": "Start date (ISO format)"}
                },
                required_parameters=["freelancer_id", "job_id", "title"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.02
            ),
            AdapterCapability(
                name="create_milestone",
                description="Create a milestone for a contract",
                category="contract_management",
                parameters={
                    "contract_id": {"type": "string", "description": "Contract ID"},
                    "description": {"type": "string", "description": "Milestone description"},
                    "amount": {"type": "number", "description": "Milestone amount"},
                    "due_date": {"type": "string", "description": "Due date (ISO format)"}
                },
                required_parameters=["contract_id", "description", "amount"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.01
            ),
            AdapterCapability(
                name="submit_work_diary",
                description="Submit work diary entry (for tracking)",
                category="time_tracking",
                parameters={
                    "contract_id": {"type": "string", "description": "Contract ID"},
                    "memo": {"type": "string", "description": "Work description"},
                    "hours": {"type": "number", "description": "Hours worked"},
                    "date": {"type": "string", "description": "Work date (ISO format)"}
                },
                required_parameters=["contract_id", "memo", "hours"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.005
            ),
            AdapterCapability(
                name="get_job_applications",
                description="Get applications for a job",
                category="job_management",
                parameters={
                    "job_id": {"type": "string", "description": "Job ID"},
                    "status": {"type": "string", "description": "Application status filter"},
                    "limit": {"type": "integer", "description": "Number of results", "default": 50}
                },
                required_parameters=["job_id"],
                async_supported=True,
                estimated_duration_seconds=1.5,
                cost_per_request=0.01
            ),
            AdapterCapability(
                name="release_payment",
                description="Release payment for completed work",
                category="payment",
                parameters={
                    "contract_id": {"type": "string", "description": "Contract ID"},
                    "milestone_id": {"type": "string", "description": "Milestone ID (for fixed price)"},
                    "amount": {"type": "number", "description": "Amount to release"},
                    "memo": {"type": "string", "description": "Payment memo"}
                },
                required_parameters=["contract_id"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.02
            ),
            AdapterCapability(
                name="leave_feedback",
                description="Leave feedback for a freelancer",
                category="feedback",
                parameters={
                    "contract_id": {"type": "string", "description": "Contract ID"},
                    "score": {"type": "number", "description": "Overall score (1-5)"},
                    "skills": {"type": "number", "description": "Skills score (1-5)"},
                    "quality": {"type": "number", "description": "Quality score (1-5)"},
                    "availability": {"type": "number", "description": "Availability score (1-5)"},
                    "communication": {"type": "number", "description": "Communication score (1-5)"},
                    "deadlines": {"type": "number", "description": "Deadlines score (1-5)"},
                    "comment": {"type": "string", "description": "Feedback comment"}
                },
                required_parameters=["contract_id", "score", "comment"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.01
            )
        ]
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a request to Upwork API with OAuth signing."""
        # In a real implementation, this would include OAuth 1.0a signing
        # For now, we'll simulate the request
        
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        # Add OAuth parameters (simplified)
        if "params" not in kwargs:
            kwargs["params"] = {}
        
        kwargs["params"].update({
            "oauth_consumer_key": self.client_id,
            "oauth_token": self.access_token,
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_nonce": str(int(time.time() * 1000))
        })
        
        response = await self.client.request(method, endpoint, **kwargs)
        response.raise_for_status()
        return response.json()
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request using Upwork."""
        if not self._initialized or self.status != AdapterStatus.RUNNING:
            raise RuntimeError("Adapter not initialized or not running")
        
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
            
            logger.error(f"Error executing Upwork request: {str(e)}")
            raise
    
    async def _execute_request(self, request: AdapterRequest) -> AdapterResponse:
        """Execute the actual Upwork request."""
        capability = request.capability
        params = request.parameters
        
        capability_map = {
            "search_freelancers": self._search_freelancers,
            "get_freelancer_profile": self._get_freelancer_profile,
            "post_job": self._post_job,
            "invite_to_job": self._invite_to_job,
            "send_offer": self._send_offer,
            "create_milestone": self._create_milestone,
            "submit_work_diary": self._submit_work_diary,
            "get_job_applications": self._get_job_applications,
            "release_payment": self._release_payment,
            "leave_feedback": self._leave_feedback
        }
        
        handler = capability_map.get(capability)
        if not handler:
            raise ValueError(f"Unknown capability: {capability}")
        
        return await handler(params)
    
    async def _search_freelancers(self, params: Dict[str, Any]) -> AdapterResponse:
        """Search for freelancers."""
        search_params = {
            "q": params["query"],
            "limit": params.get("limit", 20)
        }
        
        # Add optional filters
        if "skills" in params:
            search_params["skills"] = ";".join(params["skills"])
        if "category" in params:
            search_params["category2"] = params["category"]
        if "subcategory" in params:
            search_params["subcategory2"] = params["subcategory"]
        if "hourly_rate_min" in params:
            search_params["rate_min"] = params["hourly_rate_min"]
        if "hourly_rate_max" in params:
            search_params["rate_max"] = params["hourly_rate_max"]
        if "location" in params:
            search_params["loc"] = params["location"]
        if "english_level" in params:
            search_params["english"] = params["english_level"]
        
        result = await self._make_request(
            "GET",
            "/profiles/v2/search/providers.json",
            params=search_params
        )
        
        return AdapterResponse(
            success=True,
            data={
                "freelancers": result.get("providers", []),
                "total": result.get("paging", {}).get("total", 0),
                "count": result.get("paging", {}).get("count", 0)
            },
            metadata={
                "query": params["query"],
                "filters_applied": len(search_params) - 2  # Minus q and limit
            }
        )
    
    async def _get_freelancer_profile(self, params: Dict[str, Any]) -> AdapterResponse:
        """Get freelancer profile details."""
        freelancer_id = params["freelancer_id"]
        
        result = await self._make_request(
            "GET",
            f"/profiles/v1/providers/{freelancer_id}.json"
        )
        
        return AdapterResponse(
            success=True,
            data={
                "profile": result.get("profile", {}),
                "skills": result.get("profile", {}).get("skills", []),
                "portfolio": result.get("profile", {}).get("portfolio", []),
                "certifications": result.get("profile", {}).get("certifications", [])
            },
            metadata={
                "freelancer_id": freelancer_id
            }
        )
    
    async def _post_job(self, params: Dict[str, Any]) -> AdapterResponse:
        """Post a new job."""
        job_data = {
            "title": params["title"],
            "description": params["description"],
            "category": params["category"],
            "subcategory": params.get("subcategory", ""),
            "skills": params.get("skills", []),
            "scope": params.get("scope", "medium"),
            "duration": params.get("duration", "1-3 months"),
            "visibility": params.get("visibility", "public")
        }
        
        # Add budget info
        if params.get("hourly", False):
            job_data["budget_type"] = "hourly"
            job_data["hourly_budget_min"] = params.get("budget", 0)
            job_data["hourly_budget_max"] = params.get("budget", 0) * 1.5
        else:
            job_data["budget_type"] = "fixed"
            job_data["budget"] = params.get("budget", 0)
        
        result = await self._make_request(
            "POST",
            "/hr/v2/jobs.json",
            json=job_data
        )
        
        return AdapterResponse(
            success=True,
            data={
                "job_id": result.get("job", {}).get("id"),
                "job_url": result.get("job", {}).get("url"),
                "status": result.get("job", {}).get("status", "draft")
            },
            metadata={
                "title": params["title"],
                "budget_type": job_data["budget_type"]
            }
        )
    
    async def _invite_to_job(self, params: Dict[str, Any]) -> AdapterResponse:
        """Invite a freelancer to a job."""
        invitation_data = {
            "job_id": params["job_id"],
            "provider_id": params["freelancer_id"],
            "message": params.get("message", "You're invited to apply for this job.")
        }
        
        result = await self._make_request(
            "POST",
            f"/hr/v1/jobs/{params['job_id']}/invites.json",
            json=invitation_data
        )
        
        return AdapterResponse(
            success=True,
            data={
                "invitation_id": result.get("invitation", {}).get("id"),
                "status": "sent"
            },
            metadata={
                "job_id": params["job_id"],
                "freelancer_id": params["freelancer_id"]
            }
        )
    
    async def _send_offer(self, params: Dict[str, Any]) -> AdapterResponse:
        """Send a job offer."""
        offer_data = {
            "provider_id": params["freelancer_id"],
            "job_id": params["job_id"],
            "title": params["title"],
            "message": params.get("message", ""),
            "start_date": params.get("start_date", datetime.utcnow().isoformat())
        }
        
        # Add payment terms
        if "hourly_rate" in params:
            offer_data["charge_type"] = "hourly"
            offer_data["charge_rate"] = params["hourly_rate"]
            offer_data["weekly_hours_limit"] = params.get("weekly_limit", 40)
        else:
            offer_data["charge_type"] = "fixed"
            offer_data["amount"] = params.get("fixed_price", 0)
        
        result = await self._make_request(
            "POST",
            "/offers/v1/clients/offers.json",
            json=offer_data
        )
        
        return AdapterResponse(
            success=True,
            data={
                "offer_id": result.get("offer", {}).get("id"),
                "status": result.get("offer", {}).get("status", "pending")
            },
            metadata={
                "freelancer_id": params["freelancer_id"],
                "job_id": params["job_id"]
            }
        )
    
    async def _create_milestone(self, params: Dict[str, Any]) -> AdapterResponse:
        """Create a milestone."""
        milestone_data = {
            "description": params["description"],
            "amount": params["amount"],
            "due_date": params.get("due_date", "")
        }
        
        result = await self._make_request(
            "POST",
            f"/hr/v2/contracts/{params['contract_id']}/milestones.json",
            json=milestone_data
        )
        
        return AdapterResponse(
            success=True,
            data={
                "milestone_id": result.get("milestone", {}).get("id"),
                "status": result.get("milestone", {}).get("status", "active")
            },
            metadata={
                "contract_id": params["contract_id"],
                "amount": params["amount"]
            }
        )
    
    async def _submit_work_diary(self, params: Dict[str, Any]) -> AdapterResponse:
        """Submit work diary entry."""
        diary_data = {
            "memo": params["memo"],
            "hours": params["hours"],
            "date": params.get("date", datetime.utcnow().date().isoformat())
        }
        
        result = await self._make_request(
            "POST",
            f"/team/v2/workdiaries/contracts/{params['contract_id']}.json",
            json=diary_data
        )
        
        return AdapterResponse(
            success=True,
            data={
                "diary_id": result.get("diary", {}).get("id"),
                "status": "submitted"
            },
            metadata={
                "contract_id": params["contract_id"],
                "hours": params["hours"]
            }
        )
    
    async def _get_job_applications(self, params: Dict[str, Any]) -> AdapterResponse:
        """Get job applications."""
        query_params = {
            "limit": params.get("limit", 50)
        }
        
        if "status" in params:
            query_params["status"] = params["status"]
        
        result = await self._make_request(
            "GET",
            f"/hr/v2/jobs/{params['job_id']}/applications.json",
            params=query_params
        )
        
        return AdapterResponse(
            success=True,
            data={
                "applications": result.get("applications", []),
                "total": result.get("paging", {}).get("total", 0)
            },
            metadata={
                "job_id": params["job_id"],
                "returned_count": len(result.get("applications", []))
            }
        )
    
    async def _release_payment(self, params: Dict[str, Any]) -> AdapterResponse:
        """Release payment."""
        payment_data = {
            "amount": params.get("amount", 0),
            "comments": params.get("memo", "Payment released")
        }
        
        if "milestone_id" in params:
            endpoint = f"/hr/v2/milestones/{params['milestone_id']}/approve.json"
        else:
            endpoint = f"/hr/v2/contracts/{params['contract_id']}/payments.json"
        
        result = await self._make_request(
            "POST",
            endpoint,
            json=payment_data
        )
        
        return AdapterResponse(
            success=True,
            data={
                "payment_id": result.get("payment", {}).get("id"),
                "status": "released",
                "amount": params.get("amount", 0)
            },
            metadata={
                "contract_id": params["contract_id"]
            }
        )
    
    async def _leave_feedback(self, params: Dict[str, Any]) -> AdapterResponse:
        """Leave feedback."""
        feedback_data = {
            "score": params["score"],
            "comment": params["comment"],
            "scores": {
                "skills": params.get("skills", params["score"]),
                "quality": params.get("quality", params["score"]),
                "availability": params.get("availability", params["score"]),
                "communication": params.get("communication", params["score"]),
                "deadlines": params.get("deadlines", params["score"])
            }
        }
        
        result = await self._make_request(
            "POST",
            f"/hr/v2/contracts/{params['contract_id']}/feedback.json",
            json=feedback_data
        )
        
        return AdapterResponse(
            success=True,
            data={
                "feedback_id": result.get("feedback", {}).get("id"),
                "status": "submitted"
            },
            metadata={
                "contract_id": params["contract_id"],
                "score": params["score"]
            }
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Upwork connection."""
        if not self.client:
            return {
                "status": "unhealthy",
                "error": "Client not initialized"
            }
        
        try:
            # Test by getting user info
            result = await self._make_request("GET", "/auth/v1/info.json")
            
            return {
                "status": "healthy",
                "authenticated_user": result.get("auth_user", {}).get("name", "Unknown"),
                "response_time": 0.5  # Simulated
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }