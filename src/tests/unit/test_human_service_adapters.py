"""Unit tests for Human Service Adapters (Upwork, Fiverr)."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx

from adapters.implementations.human.upwork_adapter import UpworkAdapter
from adapters.implementations.human.fiverr_adapter import FiverrAdapter
from adapters.models import AdapterConfig, AdapterRequest, AdapterCategory


class TestUpworkAdapter:
    """Test Upwork adapter implementation."""
    
    @pytest.fixture
    def upwork_config(self):
        return AdapterConfig(
            adapter_id="test-upwork",
            adapter_type="upwork",
            name="Test Upwork",
            category=AdapterCategory.HUMAN,
            credentials={
                "client_id": "test-client",
                "client_secret": "test-secret",
                "access_token": "test-token"
            }
        )
    
    def test_adapter_creation(self, upwork_config):
        """Test creating Upwork adapter."""
        adapter = UpworkAdapter(upwork_config)
        assert adapter.config.category == AdapterCategory.HUMAN
        assert adapter.base_url == "https://www.upwork.com/api"
    
    def test_capabilities(self, upwork_config):
        """Test Upwork capabilities."""
        adapter = UpworkAdapter(upwork_config)
        capabilities = adapter.get_capabilities()
        
        cap_names = [cap.name for cap in capabilities]
        assert "search_freelancers" in cap_names
        assert "post_job" in cap_names
        assert "create_contract" in cap_names
        assert "send_message" in cap_names
        assert "create_milestone" in cap_names
    
    @pytest.mark.asyncio
    async def test_search_freelancers(self, upwork_config):
        """Test searching for freelancers."""
        adapter = UpworkAdapter(upwork_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock search response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "freelancers": [
                    {
                        "id": "123",
                        "name": "John Doe",
                        "title": "Python Developer",
                        "rate": 50,
                        "skills": ["Python", "FastAPI"],
                        "rating": 4.8
                    }
                ],
                "total": 1
            }
            mock_client.get.return_value = mock_response
            
            await adapter.initialize()
            
            request = AdapterRequest(
                capability="search_freelancers",
                parameters={
                    "skills": ["Python", "FastAPI"],
                    "min_rate": 30,
                    "max_rate": 100
                }
            )
            
            response = await adapter.execute(request)
            
            assert response.success is True
            assert len(response.data["freelancers"]) == 1
            assert response.data["freelancers"][0]["name"] == "John Doe"
    
    @pytest.mark.asyncio
    async def test_post_job(self, upwork_config):
        """Test posting a job."""
        adapter = UpworkAdapter(upwork_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {
                "job_id": "job-456",
                "status": "posted",
                "url": "https://upwork.com/jobs/job-456"
            }
            mock_client.post.return_value = mock_response
            
            await adapter.initialize()
            
            request = AdapterRequest(
                capability="post_job",
                parameters={
                    "title": "Python Developer Needed",
                    "description": "Looking for FastAPI expert",
                    "category": "web-development",
                    "budget": 5000,
                    "duration": "1-3 months"
                }
            )
            
            response = await adapter.execute(request)
            
            assert response.success is True
            assert response.data["job_id"] == "job-456"
            assert response.data["status"] == "posted"


class TestFiverrAdapter:
    """Test Fiverr adapter implementation."""
    
    @pytest.fixture
    def fiverr_config(self):
        return AdapterConfig(
            adapter_id="test-fiverr",
            adapter_type="fiverr",
            name="Test Fiverr",
            category=AdapterCategory.HUMAN,
            credentials={
                "api_key": "test-api-key"
            }
        )
    
    def test_adapter_creation(self, fiverr_config):
        """Test creating Fiverr adapter."""
        adapter = FiverrAdapter(fiverr_config)
        assert adapter.config.category == AdapterCategory.HUMAN
        assert adapter.base_url == "https://api.fiverr.com/v1"
    
    def test_capabilities(self, fiverr_config):
        """Test Fiverr capabilities."""
        adapter = FiverrAdapter(fiverr_config)
        capabilities = adapter.get_capabilities()
        
        cap_names = [cap.name for cap in capabilities]
        assert "search_gigs" in cap_names
        assert "create_order" in cap_names
        assert "send_message" in cap_names
        assert "get_order_status" in cap_names
    
    @pytest.mark.asyncio
    async def test_search_gigs(self, fiverr_config):
        """Test searching for gigs."""
        adapter = FiverrAdapter(fiverr_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "gigs": [
                    {
                        "id": "gig-789",
                        "title": "I will build FastAPI applications",
                        "seller": {
                            "username": "fastapi_expert",
                            "rating": 4.9
                        },
                        "price": 150,
                        "delivery_time": 3
                    }
                ],
                "total": 1
            }
            mock_client.get.return_value = mock_response
            
            await adapter.initialize()
            
            request = AdapterRequest(
                capability="search_gigs",
                parameters={
                    "query": "FastAPI development",
                    "min_price": 50,
                    "max_price": 500
                }
            )
            
            response = await adapter.execute(request)
            
            assert response.success is True
            assert len(response.data["gigs"]) == 1
            assert "FastAPI" in response.data["gigs"][0]["title"]