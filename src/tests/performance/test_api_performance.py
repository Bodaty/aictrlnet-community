"""Performance tests for API endpoints."""

import pytest
import asyncio
import time
from httpx import AsyncClient
from unittest.mock import patch, Mock
import statistics

from main import app


@pytest.fixture
async def fast_client():
    """Create client with shorter timeout for performance tests."""
    async with AsyncClient(app=app, base_url="http://test", timeout=5.0) as client:
        # Mock authentication
        with patch("core.security.get_current_active_user") as mock_auth:
            mock_auth.return_value = {
                "id": "perf-test-user",
                "username": "perfuser",
                "email": "perf@example.com",
                "is_active": True
            }
            client.headers["Authorization"] = "Bearer perf-token"
            yield client


class TestAPIPerformance:
    """Test API endpoint performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_adapter_discovery_performance(self, fast_client):
        """Test performance of adapter discovery endpoint."""
        response_times = []
        
        # Mock the adapter service
        with patch("services.adapter.AdapterService.discover_adapters") as mock_discover:
            mock_discover.return_value = (
                [{"type": f"adapter-{i}", "name": f"Adapter {i}"} for i in range(50)],
                50
            )
            
            # Run multiple requests
            for _ in range(100):
                start_time = time.time()
                response = await fast_client.get("/api/v1/adapters/discover")
                end_time = time.time()
                
                assert response.status_code == 200
                response_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        max_time = max(response_times)
        
        print(f"\nAdapter Discovery Performance:")
        print(f"  Average response time: {avg_time:.2f}ms")
        print(f"  95th percentile: {p95_time:.2f}ms")
        print(f"  Max response time: {max_time:.2f}ms")
        
        # Performance assertions
        assert avg_time < 50  # Average should be under 50ms
        assert p95_time < 100  # 95th percentile should be under 100ms
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_requests(self, fast_client):
        """Test API performance under concurrent load."""
        
        async def make_request(client, endpoint):
            """Make a single request and return response time."""
            start_time = time.time()
            response = await client.get(endpoint)
            end_time = time.time()
            return {
                "duration": (end_time - start_time) * 1000,
                "status": response.status_code
            }
        
        # Mock services
        with patch("services.workflow.WorkflowService.list_workflows") as mock_list:
            mock_list.return_value = ([], 0)  # Empty list, 0 total
            
            # Test different concurrency levels
            concurrency_levels = [10, 50, 100]
            
            for concurrency in concurrency_levels:
                tasks = [
                    make_request(fast_client, "/api/v1/workflows")
                    for _ in range(concurrency)
                ]
                
                start_time = time.time()
                results = await asyncio.gather(*tasks)
                total_time = (time.time() - start_time) * 1000
                
                # Calculate statistics
                response_times = [r["duration"] for r in results]
                success_count = sum(1 for r in results if r["status"] == 200)
                
                avg_time = statistics.mean(response_times)
                throughput = (concurrency / total_time) * 1000  # Requests per second
                
                print(f"\nConcurrency Level: {concurrency}")
                print(f"  Total time: {total_time:.2f}ms")
                print(f"  Average response time: {avg_time:.2f}ms")
                print(f"  Throughput: {throughput:.2f} req/s")
                print(f"  Success rate: {(success_count/concurrency)*100:.1f}%")
                
                # All requests should succeed
                assert success_count == concurrency
                # Average response time should stay reasonable
                assert avg_time < 200
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_large_payload_performance(self, fast_client):
        """Test performance with large request/response payloads."""
        
        # Create a large workflow definition
        large_workflow = {
            "name": "Large Workflow",
            "description": "Performance test workflow",
            "definition": {
                "nodes": [
                    {
                        "id": f"node-{i}",
                        "type": "process",
                        "name": f"Process Node {i}",
                        "data": {"index": i, "config": {"param": "x" * 100}}
                    }
                    for i in range(100)
                ],
                "edges": [
                    {"from": f"node-{i}", "to": f"node-{i+1}"}
                    for i in range(99)
                ]
            }
        }
        
        with patch("services.workflow.WorkflowService.create_workflow") as mock_create:
            mock_workflow = Mock()
            mock_workflow.id = "large-workflow-001"
            mock_workflow.to_dict.return_value = large_workflow
            mock_create.return_value = mock_workflow
            
            # Measure creation time
            start_time = time.time()
            response = await fast_client.post(
                "/api/v1/workflows",
                json=large_workflow
            )
            creation_time = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            
            print(f"\nLarge Payload Performance:")
            print(f"  Workflow nodes: 100")
            print(f"  Creation time: {creation_time:.2f}ms")
            print(f"  Payload size: ~{len(str(large_workflow))/1024:.1f}KB")
            
            # Should handle large payloads efficiently
            assert creation_time < 500  # Under 500ms for large workflow
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_database_query_performance(self, fast_client):
        """Test performance of database-heavy endpoints."""
        
        # Test pagination with different page sizes
        page_sizes = [10, 50, 100]
        
        for page_size in page_sizes:
            with patch("services.task.TaskService.list_tasks") as mock_list:
                # Mock returning the requested number of tasks
                mock_tasks = [
                    Mock(id=f"task-{i}", name=f"Task {i}")
                    for i in range(page_size)
                ]
                mock_list.return_value = (mock_tasks, 1000)  # 1000 total tasks
                
                start_time = time.time()
                response = await fast_client.get(
                    f"/api/v1/tasks?limit={page_size}"
                )
                query_time = (time.time() - start_time) * 1000
                
                assert response.status_code == 200
                
                print(f"\nDatabase Query Performance (limit={page_size}):")
                print(f"  Query time: {query_time:.2f}ms")
                print(f"  Time per item: {query_time/page_size:.2f}ms")
                
                # Query time should scale reasonably
                assert query_time < 100 + (page_size * 2)  # Base + per-item overhead