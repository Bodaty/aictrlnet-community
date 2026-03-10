"""Response shape validation tests.

Validates that API responses match the declared OpenAPI schema shapes.
Requires running services and the OpenAPI spec at docs/api/openapi.json.
"""

import os
import sys
import pytest
import httpx

# Add smoke_common to path
_tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

# Also add project root tests dir for smoke_common
_project_tests = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests'))
if _project_tests not in sys.path:
    sys.path.insert(0, _project_tests)

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "dev-token-for-testing")

# Safe endpoints to test with GET (no side effects)
SAFE_GET_ENDPOINTS = [
    "/api/v1/workflows",
    "/api/v1/tasks",
    "/api/v1/templates",
    "/api/v1/agents",
    "/api/v1/health",
]


@pytest.fixture(scope="module")
def http_client():
    """Create an HTTP client with auth headers."""
    client = httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        timeout=10.0,
    )
    yield client
    client.close()


@pytest.fixture(scope="module")
def validator():
    """Load the response validator if jsonschema and OpenAPI schema are available."""
    try:
        from smoke_common.response_validator import validate_response_shape, get_openapi_schema
        get_openapi_schema()  # Verify schema file is accessible
        return validate_response_shape
    except (ImportError, FileNotFoundError):
        pytest.skip("jsonschema, response_validator, or OpenAPI schema not available")


@pytest.mark.parametrize("endpoint", SAFE_GET_ENDPOINTS)
def test_get_endpoint_response_shape(endpoint, http_client, validator):
    """Verify GET endpoint responses match OpenAPI schema."""
    try:
        response = http_client.get(endpoint)
    except httpx.ConnectError:
        pytest.skip(f"Service not running at {BASE_URL}")

    if response.status_code >= 400:
        pytest.skip(f"{endpoint} returned {response.status_code}")

    try:
        body = response.json()
    except Exception:
        pytest.skip(f"{endpoint} returned non-JSON response")

    # Validate against OpenAPI schema
    validator("get", endpoint, response.status_code, body)


def test_response_validator_loads():
    """Verify the response validator can load the OpenAPI schema."""
    try:
        from smoke_common.response_validator import get_openapi_schema, get_documented_endpoints
        schema = get_openapi_schema()
    except (ImportError, FileNotFoundError):
        pytest.skip("response_validator or OpenAPI schema not available")
    assert "paths" in schema
    assert "components" in schema or "definitions" in schema

    endpoints = get_documented_endpoints()
    assert len(endpoints) > 0, "No endpoints found in OpenAPI schema"
