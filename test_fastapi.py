#!/usr/bin/env python
"""Test script for FastAPI implementation."""

import requests
import json
from datetime import datetime

# Base URLs for each edition
EDITIONS = {
    "community": "http://localhost:8010",
    "business": "http://localhost:8001", 
    "enterprise": "http://localhost:8002",
}

# Auth header
AUTH_HEADER = {"Authorization": "Bearer dev-token-for-testing"}


def test_health(edition: str, base_url: str):
    """Test health endpoint."""
    print(f"\n🏥 Testing {edition} health...")
    
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed: {data}")
    else:
        print(f"❌ Health check failed: {response.status_code}")


def test_api_docs(edition: str, base_url: str):
    """Test API documentation."""
    print(f"\n📚 Testing {edition} API docs...")
    
    response = requests.get(f"{base_url}/api/v1/docs")
    if response.status_code == 200:
        print(f"✅ API docs available at {base_url}/api/v1/docs")
    else:
        print(f"❌ API docs not available: {response.status_code}")


def test_tasks(edition: str, base_url: str):
    """Test task endpoints."""
    print(f"\n📋 Testing {edition} tasks...")
    
    # List tasks
    response = requests.get(f"{base_url}/api/v1/tasks", headers=AUTH_HEADER)
    if response.status_code == 200:
        print(f"✅ List tasks: {len(response.json())} tasks")
    else:
        print(f"❌ List tasks failed: {response.status_code}")
    
    # Create task
    task_data = {
        "name": f"Test Task - {datetime.now().isoformat()}",
        "description": "Created by FastAPI test script",
    }
    response = requests.post(
        f"{base_url}/api/v1/tasks",
        json=task_data,
        headers=AUTH_HEADER
    )
    if response.status_code == 201:
        task = response.json()
        print(f"✅ Created task: {task['id']}")
        
        # Get task
        response = requests.get(
            f"{base_url}/api/v1/tasks/{task['id']}",
            headers=AUTH_HEADER
        )
        if response.status_code == 200:
            print(f"✅ Retrieved task: {response.json()['name']}")
        else:
            print(f"❌ Get task failed: {response.status_code}")
    else:
        print(f"❌ Create task failed: {response.status_code}")
        print(f"   Response: {response.text}")


def test_workflows(edition: str, base_url: str):
    """Test workflow endpoints."""
    print(f"\n🔄 Testing {edition} workflows...")
    
    # List workflows
    response = requests.get(f"{base_url}/api/v1/workflows", headers=AUTH_HEADER)
    if response.status_code == 200:
        print(f"✅ List workflows: {len(response.json())} workflows")
    else:
        print(f"❌ List workflows failed: {response.status_code}")
    
    # Create workflow
    workflow_data = {
        "name": f"Test Workflow - {datetime.now().isoformat()}",
        "description": "Created by FastAPI test script",
        "definition": {
            "nodes": [
                {"id": "start", "type": "input", "name": "Start"},
                {"id": "process", "type": "process", "name": "Process"},
                {"id": "end", "type": "output", "name": "End"},
            ],
            "edges": [
                {"from": "start", "to": "process"},
                {"from": "process", "to": "end"},
            ],
        },
    }
    response = requests.post(
        f"{base_url}/api/v1/workflows",
        json=workflow_data,
        headers=AUTH_HEADER
    )
    if response.status_code == 201:
        workflow = response.json()
        print(f"✅ Created workflow: {workflow['id']}")
    else:
        print(f"❌ Create workflow failed: {response.status_code}")
        print(f"   Response: {response.text}")


def test_templates(edition: str, base_url: str):
    """Test template endpoints."""
    print(f"\n📄 Testing {edition} templates...")
    
    # List templates
    response = requests.get(f"{base_url}/api/v1/templates", headers=AUTH_HEADER)
    if response.status_code == 200:
        data = response.json()
        templates = data.get("templates", [])
        print(f"✅ List templates: {len(templates)} templates")
        
        # Show categories
        categories = data.get("categories", [])
        if categories:
            print(f"   Categories: {', '.join(categories[:5])}...")
        
        # Test creating workflow from template
        if templates:
            template = templates[0]
            print(f"\n   Testing workflow creation from template '{template['name']}'...")
            
            workflow_data = {
                "name": f"Workflow from {template['name']}",
                "templateId": template["id"],
            }
            response = requests.post(
                f"{base_url}/api/v1/workflows/from-template",
                json=workflow_data,
                headers=AUTH_HEADER
            )
            if response.status_code == 201:
                print(f"   ✅ Created workflow from template")
            else:
                print(f"   ❌ Failed to create workflow: {response.status_code}")
    else:
        print(f"❌ List templates failed: {response.status_code}")


def compare_performance():
    """Compare performance between Flask and FastAPI."""
    print("\n⚡ Performance Comparison...")
    
    import time
    
    # Test Flask endpoint
    flask_url = "http://localhost:5010/api/tasks"
    start = time.time()
    for _ in range(10):
        requests.get(flask_url, headers=AUTH_HEADER)
    flask_time = time.time() - start
    
    # Test FastAPI endpoint
    fastapi_url = "http://localhost:8010/api/v1/tasks"
    start = time.time()
    for _ in range(10):
        requests.get(fastapi_url, headers=AUTH_HEADER)
    fastapi_time = time.time() - start
    
    print(f"\n   Flask (10 requests): {flask_time:.3f}s")
    print(f"   FastAPI (10 requests): {fastapi_time:.3f}s")
    
    if fastapi_time < flask_time:
        improvement = ((flask_time - fastapi_time) / flask_time) * 100
        print(f"   ✅ FastAPI is {improvement:.1f}% faster!")
    else:
        print(f"   ⚠️  Flask was faster in this test")


def main():
    """Run all tests."""
    print("🚀 Testing AICtrlNet FastAPI Implementation")
    print("=" * 50)
    
    for edition, base_url in EDITIONS.items():
        print(f"\n{'='*50}")
        print(f"Testing {edition.upper()} Edition")
        print(f"{'='*50}")
        
        try:
            test_health(edition, base_url)
            test_api_docs(edition, base_url)
            test_tasks(edition, base_url)
            test_workflows(edition, base_url)
            test_templates(edition, base_url)
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Could not connect to {edition} edition at {base_url}")
            print("   Make sure the FastAPI services are running:")
            print("   docker-compose up -d")
    
    # Performance comparison
    try:
        compare_performance()
    except:
        print("\n⚠️  Could not run performance comparison")
    
    print("\n" + "="*50)
    print("✅ FastAPI test complete!")
    print("="*50)


if __name__ == "__main__":
    main()