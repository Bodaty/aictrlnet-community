#!/usr/bin/env python3
"""Test FastAPI endpoints to verify they're working."""

import requests
import json
import sys
from typing import Dict, List, Tuple

# Base URLs for each edition
BASE_URLS = {
    "community": "http://localhost:8010",
    "business": "http://localhost:8001", 
    "enterprise": "http://localhost:8002"
}

# Headers with dev token
HEADERS = {
    "Authorization": "Bearer dev-token-for-testing",
    "Content-Type": "application/json"
}

# Test data
TASK_DATA = {
    "name": "Test Task",
    "description": "Test task description",
    "metadata": {"test": True}
}

WORKFLOW_DATA = {
    "name": "Test Workflow",
    "description": "Test workflow description",
    "definition": {
        "nodes": [
            {
                "id": "node1",
                "type": "start",
                "name": "Start Node"
            },
            {
                "id": "node2",
                "type": "task",
                "name": "Task Node"
            },
            {
                "id": "node3",
                "type": "end",
                "name": "End Node"
            }
        ],
        "edges": [
            {
                "source": "node1",
                "target": "node2"
            },
            {
                "source": "node2",
                "target": "node3"
            }
        ]
    }
}

def test_endpoint(method: str, url: str, data: Dict = None) -> Tuple[bool, str]:
    """Test a single endpoint."""
    try:
        if method == "GET":
            response = requests.get(url, headers=HEADERS, timeout=5)
        elif method == "POST":
            response = requests.post(url, headers=HEADERS, json=data, timeout=5)
        elif method == "PUT":
            response = requests.put(url, headers=HEADERS, json=data, timeout=5)
        elif method == "DELETE":
            response = requests.delete(url, headers=HEADERS, timeout=5)
        else:
            return False, f"Unknown method: {method}"
        
        if response.status_code in [200, 201, 204]:
            return True, f"{response.status_code} OK"
        else:
            return False, f"{response.status_code} {response.reason}"
    except Exception as e:
        return False, str(e)

def main():
    """Run endpoint tests."""
    print("Testing FastAPI Endpoints\n")
    
    total_tests = 0
    passed_tests = 0
    
    # Define endpoints to test
    endpoints = [
        # Health check
        ("GET", "/health", None),
        
        # Test minimal endpoint
        ("GET", "/api/v1/test-minimal/", None),
        ("POST", "/api/v1/test-minimal/", None),
        
        # Tasks endpoints
        ("GET", "/api/v1/tasks/", None),
        ("POST", "/api/v1/tasks/", TASK_DATA),
        
        # Workflows endpoints
        ("GET", "/api/v1/workflows/", None),
        ("POST", "/api/v1/workflows/", WORKFLOW_DATA),
        
        # Templates endpoints
        ("GET", "/api/v1/templates/", None),
        
        # MCP endpoints
        ("GET", "/api/v1/mcp/info", None),
        ("GET", "/api/v1/mcp/servers", None),
        
        # Bridge endpoints  
        ("GET", "/api/v1/bridge/sessions", None),
        ("GET", "/api/v1/bridge/metrics", None),
    ]
    
    # Test each edition
    for edition, base_url in BASE_URLS.items():
        print(f"\n{edition.upper()} Edition ({base_url}):")
        print("-" * 50)
        
        edition_passed = 0
        edition_total = 0
        
        for method, path, data in endpoints:
            url = f"{base_url}{path}"
            passed, message = test_endpoint(method, url, data)
            
            status = "✓" if passed else "✗"
            print(f"{status} {method:6} {path:40} {message}")
            
            edition_total += 1
            total_tests += 1
            
            if passed:
                edition_passed += 1
                passed_tests += 1
        
        print(f"\nEdition Summary: {edition_passed}/{edition_total} passed ({edition_passed/edition_total*100:.1f}%)")
    
    print(f"\n{'='*50}")
    print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    return 0 if passed_tests == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())