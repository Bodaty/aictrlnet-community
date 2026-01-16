#!/usr/bin/env python3
"""Test Redis caching functionality."""

import asyncio
import time
import httpx
from rich.console import Console
from rich.table import Table

console = Console()

# Test configuration
BASE_URL = "http://localhost:8001"  # Business edition
AUTH_HEADER = {"Authorization": "Bearer dev-token-for-testing"}


async def test_cache_functionality():
    """Test Redis cache functionality."""
    async with httpx.AsyncClient() as client:
        # 1. Check cache status
        console.print("\n[bold cyan]1. Checking cache status...[/bold cyan]")
        response = await client.get(
            f"{BASE_URL}/api/v1/cache/stats",
            headers=AUTH_HEADER
        )
        if response.status_code == 200:
            stats = response.json()
            console.print(f"Cache Status: {stats['status']}")
            if stats['status'] == 'enabled':
                console.print(f"Total Keys: {stats['stats']['total_keys']}")
                console.print(f"Hit Rate: {stats['stats']['hit_rate']}%")
        else:
            console.print(f"[red]Failed to get cache stats: {response.status_code}[/red]")
            return

        # 2. Create a task to cache
        console.print("\n[bold cyan]2. Creating a test task...[/bold cyan]")
        task_data = {
            "name": "Cached Test Task",
            "description": "This task will be cached",
            "status": "pending"
        }
        response = await client.post(
            f"{BASE_URL}/api/v1/tasks",
            json=task_data,
            headers=AUTH_HEADER
        )
        if response.status_code == 201:
            task = response.json()
            task_id = task['id']
            console.print(f"Created task: {task_id}")
        else:
            console.print(f"[red]Failed to create task: {response.status_code}[/red]")
            return

        # 3. Fetch task multiple times to test caching
        console.print("\n[bold cyan]3. Testing cache performance...[/bold cyan]")
        
        # First fetch (cache miss)
        start_time = time.time()
        response = await client.get(
            f"{BASE_URL}/api/v1/tasks/{task_id}",
            headers=AUTH_HEADER
        )
        first_fetch_time = time.time() - start_time
        console.print(f"First fetch (cache miss): {first_fetch_time*1000:.2f}ms")

        # Second fetch (cache hit)
        start_time = time.time()
        response = await client.get(
            f"{BASE_URL}/api/v1/tasks/{task_id}",
            headers=AUTH_HEADER
        )
        second_fetch_time = time.time() - start_time
        console.print(f"Second fetch (cache hit): {second_fetch_time*1000:.2f}ms")

        # Performance improvement
        improvement = ((first_fetch_time - second_fetch_time) / first_fetch_time) * 100
        console.print(f"[green]Performance improvement: {improvement:.1f}%[/green]")

        # 4. Test cache invalidation
        console.print("\n[bold cyan]4. Testing cache invalidation...[/bold cyan]")
        
        # Update the task
        update_data = {"description": "Updated description"}
        response = await client.put(
            f"{BASE_URL}/api/v1/tasks/{task_id}",
            json=update_data,
            headers=AUTH_HEADER
        )
        console.print("Updated task (cache should be invalidated)")

        # Fetch again (should be cache miss)
        start_time = time.time()
        response = await client.get(
            f"{BASE_URL}/api/v1/tasks/{task_id}",
            headers=AUTH_HEADER
        )
        third_fetch_time = time.time() - start_time
        console.print(f"Third fetch (after update): {third_fetch_time*1000:.2f}ms")

        # 5. List cache keys (Enterprise only)
        console.print("\n[bold cyan]5. Testing cache key listing (Enterprise only)...[/bold cyan]")
        response = await client.get(
            f"{BASE_URL}/api/v1/cache/keys?pattern=task:*",
            headers=AUTH_HEADER
        )
        if response.status_code == 200:
            keys_data = response.json()
            console.print(f"Found {keys_data['total_found']} task cache keys")
            for key in keys_data['keys'][:5]:  # Show first 5
                console.print(f"  - {key}")
        else:
            console.print("[yellow]Cache key listing requires Enterprise edition[/yellow]")

        # 6. Clear cache
        console.print("\n[bold cyan]6. Clearing cache...[/bold cyan]")
        response = await client.delete(
            f"{BASE_URL}/api/v1/cache/clear?pattern=task:*",
            headers=AUTH_HEADER
        )
        if response.status_code == 204:
            console.print("[green]Cache cleared successfully[/green]")
        else:
            console.print(f"[red]Failed to clear cache: {response.status_code}[/red]")

        # Clean up - delete the test task
        await client.delete(
            f"{BASE_URL}/api/v1/tasks/{task_id}",
            headers=AUTH_HEADER
        )
        console.print("\n[green]Test completed successfully![/green]")


async def test_cache_with_multiple_requests():
    """Test cache with concurrent requests."""
    console.print("\n[bold cyan]Testing cache with concurrent requests...[/bold cyan]")
    
    async with httpx.AsyncClient() as client:
        # Create a task
        task_data = {
            "name": "Concurrent Cache Test",
            "description": "Testing concurrent cache access",
            "status": "pending"
        }
        response = await client.post(
            f"{BASE_URL}/api/v1/tasks",
            json=task_data,
            headers=AUTH_HEADER
        )
        task_id = response.json()['id']
        
        # Make 10 concurrent requests
        async def fetch_task():
            start = time.time()
            response = await client.get(
                f"{BASE_URL}/api/v1/tasks/{task_id}",
                headers=AUTH_HEADER
            )
            return time.time() - start
        
        # First round - cache miss
        tasks = [fetch_task() for _ in range(10)]
        first_times = await asyncio.gather(*tasks)
        
        # Second round - cache hits
        tasks = [fetch_task() for _ in range(10)]
        second_times = await asyncio.gather(*tasks)
        
        # Display results
        table = Table(title="Concurrent Request Performance")
        table.add_column("Request #", style="cyan")
        table.add_column("First Round (ms)", style="yellow")
        table.add_column("Second Round (ms)", style="green")
        table.add_column("Improvement", style="magenta")
        
        for i in range(10):
            improvement = ((first_times[i] - second_times[i]) / first_times[i]) * 100
            table.add_row(
                str(i + 1),
                f"{first_times[i]*1000:.2f}",
                f"{second_times[i]*1000:.2f}",
                f"{improvement:.1f}%"
            )
        
        console.print(table)
        
        # Clean up
        await client.delete(
            f"{BASE_URL}/api/v1/tasks/{task_id}",
            headers=AUTH_HEADER
        )


if __name__ == "__main__":
    console.print("[bold]AICtrlNet Redis Cache Testing[/bold]")
    console.print("=" * 50)
    
    asyncio.run(test_cache_functionality())
    asyncio.run(test_cache_with_multiple_requests())