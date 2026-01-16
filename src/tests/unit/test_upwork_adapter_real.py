"""Unit tests for the actual Upwork adapter implementation."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import time
import httpx

from adapters.implementations.human.upwork_adapter import UpworkAdapter
from adapters.models import (
    AdapterConfig, AdapterRequest, AdapterResponse,
    AdapterStatus, AdapterCategory
)


@pytest.fixture
def upwork_config():
    """Create test Upwork adapter configuration."""
    return AdapterConfig(
        adapter_id="test-upwork",
        adapter_type="upwork",
        name="Test Upwork Adapter",
        category=AdapterCategory.HUMAN,
        credentials={
            "client_id": "test-client",
            "client_secret": "test-secret",
            "access_token": "test-token",
            "access_secret": "test-access-secret"
        },
        settings={},
        timeout_seconds=30
    )


class TestUpworkAdapter:
    """Test the actual Upwork adapter implementation."""
    
    def test_adapter_creation(self, upwork_config):
        """Test creating Upwork adapter instance."""
        adapter = UpworkAdapter(upwork_config)
        
        assert adapter.config.category == AdapterCategory.HUMAN
        assert adapter.client_id == "test-client"
        assert adapter.client_secret == "test-secret"
        assert adapter.access_token == "test-token"
        assert adapter.access_secret == "test-access-secret"
        assert adapter.base_url == "https://www.upwork.com/api"
    
    def test_adapter_requires_all_oauth_credentials(self):
        """Test that adapter raises error without complete OAuth credentials."""
        config = AdapterConfig(
            adapter_id="test-no-creds",
            adapter_type="upwork",
            name="No Creds Adapter",
            category=AdapterCategory.HUMAN,
            credentials={
                "client_id": "test-client",
                # Missing client_secret, access_token, access_secret
            },
            settings={}
        )
        
        with pytest.raises(ValueError, match="Upwork OAuth credentials required"):
            UpworkAdapter(config)
    
    @pytest.mark.asyncio
    async def test_adapter_initialize(self, upwork_config):
        """Test adapter initialization."""
        adapter = UpworkAdapter(upwork_config)
        
        await adapter.initialize()
        
        assert adapter.client is not None
        assert adapter._initialized is True
        assert adapter.status == AdapterStatus.RUNNING
        
        # Cleanup
        await adapter.shutdown()
    
    @pytest.mark.asyncio
    async def test_adapter_shutdown(self, upwork_config):
        """Test adapter shutdown."""
        adapter = UpworkAdapter(upwork_config)
        
        await adapter.initialize()
        await adapter.shutdown()
        
        assert adapter.client is None
        assert adapter.status == AdapterStatus.STOPPED
    
    def test_get_capabilities(self, upwork_config):
        """Test getting adapter capabilities."""
        adapter = UpworkAdapter(upwork_config)
        capabilities = adapter.get_capabilities()
        
        # Check that all expected capabilities are present
        capability_names = [cap.name for cap in capabilities]
        expected_capabilities = [
            "search_freelancers",
            "get_freelancer_profile",
            "post_job",
            "invite_to_job",
            "send_offer",
            "create_milestone",
            "submit_work_diary",
            "get_job_applications",
            "release_payment",
            "leave_feedback"
        ]
        
        for expected in expected_capabilities:
            assert expected in capability_names
        
        # Check specific capability details
        search_cap = next(cap for cap in capabilities if cap.name == "search_freelancers")
        assert "query" in search_cap.parameters
        assert "query" in search_cap.required_parameters
        assert search_cap.category == "search"
    
    @pytest.mark.asyncio
    async def test_execute_not_initialized(self, upwork_config):
        """Test that execute raises error when not initialized."""
        adapter = UpworkAdapter(upwork_config)
        
        request = AdapterRequest(
            capability="search_freelancers",
            parameters={"query": "python developer"}
        )
        
        with pytest.raises(RuntimeError, match="Adapter not initialized or not running"):
            await adapter.execute(request)
    
    @pytest.mark.asyncio
    async def test_search_freelancers(self, upwork_config):
        """Test searching for freelancers."""
        adapter = UpworkAdapter(upwork_config)
        
        # Mock the HTTP client
        with patch.object(adapter, '_make_request') as mock_request:
            mock_request.return_value = {
                "freelancers": [
                    {
                        "id": "123456",
                        "name": "John Doe",
                        "title": "Python Developer",
                        "skills": ["Python", "Django", "FastAPI"],
                        "rate": 50.0,
                        "hours": 1000,
                        "jobs_success": 95.5,
                        "location": {
                            "country": "United States",
                            "city": "New York"
                        }
                    }
                ],
                "total": 1,
                "offset": 0
            }
            
            await adapter.initialize()
            
            request = AdapterRequest(
                capability="search_freelancers",
                parameters={
                    "query": "python developer",
                    "skills": ["Python", "FastAPI"],
                    "min_rate": 30,
                    "max_rate": 100,
                    "limit": 20
                }
            )
            
            response = await adapter.execute(request)
            
            # Verify the request was made correctly
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"  # method
            assert "/profiles/v2/search/freelancers" in call_args[0][1]  # endpoint
            
            # Check params
            params = call_args[1].get("params", {})
            assert params["q"] == "python developer"
            assert params["skills"] == "Python;FastAPI"
            assert params["hourlyRate"] == "[30 TO 100]"
            
            # Verify response
            assert response.success is True
            assert len(response.data["freelancers"]) == 1
            assert response.data["freelancers"][0]["name"] == "John Doe"
            assert response.capability == "search_freelancers"
    
    @pytest.mark.asyncio
    async def test_post_job(self, upwork_config):
        """Test posting a job."""
        adapter = UpworkAdapter(upwork_config)
        
        with patch.object(adapter, '_make_request') as mock_request:
            mock_request.return_value = {
                "job": {
                    "id": "job-789",
                    "ciphertext": "~01234567890abcdef",
                    "title": "Python Developer Needed",
                    "url": "https://www.upwork.com/jobs/~01234567890abcdef",
                    "created_time": "2025-01-26T12:00:00Z",
                    "status": "open"
                }
            }
            
            await adapter.initialize()
            
            request = AdapterRequest(
                capability="post_job",
                parameters={
                    "title": "Python Developer Needed",
                    "description": "Looking for experienced Python developer",
                    "category": "web-development",
                    "subcategory": "web-programming",
                    "skills": ["Python", "FastAPI", "PostgreSQL"],
                    "scope": "medium",
                    "duration": "1-3 months",
                    "budget": 5000,
                    "hourly": False,
                    "visibility": "public"
                }
            )
            
            response = await adapter.execute(request)
            
            # Verify the request
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"  # method
            assert "/hr/v2/jobs" in call_args[0][1]  # endpoint
            
            # Check job data
            job_data = call_args[1].get("json", {})
            assert job_data["title"] == "Python Developer Needed"
            assert job_data["description"] == "Looking for experienced Python developer"
            assert job_data["category"] == "web-development"
            
            # Verify response
            assert response.success is True
            assert response.data["job"]["id"] == "job-789"
            assert response.data["job"]["status"] == "open"
    
    @pytest.mark.asyncio
    async def test_create_milestone(self, upwork_config):
        """Test creating a milestone."""
        adapter = UpworkAdapter(upwork_config)
        
        with patch.object(adapter, '_make_request') as mock_request:
            mock_request.return_value = {
                "milestone": {
                    "id": "milestone-123",
                    "description": "Complete Phase 1",
                    "amount": 1000.0,
                    "status": "active",
                    "created_time": "2025-01-26T12:00:00Z"
                }
            }
            
            await adapter.initialize()
            
            request = AdapterRequest(
                capability="create_milestone",
                parameters={
                    "contract_id": "contract-456",
                    "description": "Complete Phase 1",
                    "amount": 1000.0,
                    "due_date": "2025-02-15"
                }
            )
            
            response = await adapter.execute(request)
            
            # Verify the request
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"  # method
            assert "/finance/v2/milestones" in call_args[0][1]  # endpoint
            
            # Verify response
            assert response.success is True
            assert response.data["milestone"]["amount"] == 1000.0
            assert response.data["milestone"]["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self, upwork_config):
        """Test handling of API errors."""
        adapter = UpworkAdapter(upwork_config)
        
        with patch.object(adapter, '_make_request') as mock_request:
            mock_request.side_effect = httpx.HTTPError("API Error: Unauthorized")
            
            await adapter.initialize()
            
            request = AdapterRequest(
                capability="search_freelancers",
                parameters={"query": "test"}
            )
            
            # Should raise the error
            with pytest.raises(httpx.HTTPError):
                await adapter.execute(request)
            
            # Check metrics were updated
            assert adapter.metrics.total_requests == 1
            assert adapter.metrics.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, upwork_config):
        """Test rate limiting functionality."""
        # Configure with rate limit
        upwork_config.rate_limit = {"requests_per_minute": 60}
        adapter = UpworkAdapter(upwork_config)
        
        with patch.object(adapter, '_make_request') as mock_request:
            mock_request.return_value = {"freelancers": [], "total": 0}
            
            await adapter.initialize()
            
            # Should apply rate limiting
            assert adapter._rate_limiter is not None
            
            request = AdapterRequest(
                capability="search_freelancers",
                parameters={"query": "test"}
            )
            
            # Execute should respect rate limit
            response = await adapter.execute(request)
            assert response.success is True