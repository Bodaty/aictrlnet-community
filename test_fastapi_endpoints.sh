#!/bin/bash

# Test script for FastAPI endpoints

echo "Testing FastAPI AICtrlNet Endpoints"
echo "==================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0

# Function to test endpoint
test_endpoint() {
    local method=$1
    local url=$2
    local description=$3
    local data=$4
    
    TOTAL=$((TOTAL + 1))
    
    echo -n "Testing: $description... "
    
    if [ -z "$data" ]; then
        response=$(curl -s -o /dev/null -w "%{http_code}" -X $method "$url" -H "Authorization: Bearer dev-token-for-testing")
    else
        response=$(curl -s -o /dev/null -w "%{http_code}" -X $method "$url" -H "Authorization: Bearer dev-token-for-testing" -H "Content-Type: application/json" -d "$data")
    fi
    
    # Check if response is 2xx or 3xx (success)
    if [[ $response =~ ^[23][0-9]{2}$ ]]; then
        echo -e "${GREEN}✓${NC} ($response)"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗${NC} ($response)"
        FAILED=$((FAILED + 1))
    fi
}

echo ""
echo "Community Edition (Port 8010)"
echo "-----------------------------"
test_endpoint "GET" "http://localhost:8010/api/v1/health" "Health check"
test_endpoint "GET" "http://localhost:8010/api/v1/tasks" "List tasks"
test_endpoint "GET" "http://localhost:8010/api/v1/workflows" "List workflows"
test_endpoint "GET" "http://localhost:8010/api/v1/templates" "List templates"
test_endpoint "GET" "http://localhost:8010/api/v1/adapters" "List adapters"
test_endpoint "GET" "http://localhost:8010/api/v1/mcp/servers" "List MCP servers"
test_endpoint "GET" "http://localhost:8010/api/v1/bridge/connections" "List bridge connections"

echo ""
echo "Business Edition (Port 8001)"
echo "----------------------------"
test_endpoint "GET" "http://localhost:8001/api/v1/health" "Health check"
test_endpoint "GET" "http://localhost:8001/api/v1/approvals/workflows" "List approval workflows"
test_endpoint "GET" "http://localhost:8001/api/v1/rbac/roles" "List roles"
test_endpoint "GET" "http://localhost:8001/api/v1/governance/policies" "List policies"
test_endpoint "GET" "http://localhost:8001/api/v1/governance/compliance-report" "Compliance report"
test_endpoint "GET" "http://localhost:8001/api/v1/state" "List state entries"
test_endpoint "GET" "http://localhost:8001/api/v1/validation/rules" "List validation rules"

echo ""
echo "Enterprise Edition (Port 8002)"
echo "------------------------------"
test_endpoint "GET" "http://localhost:8002/api/v1/health" "Health check"
test_endpoint "GET" "http://localhost:8002/api/v1/federation/instances" "List federated instances"
test_endpoint "GET" "http://localhost:8002/api/v1/tenants" "List tenants"
test_endpoint "GET" "http://localhost:8002/api/v1/tenants/1a2b65b7-8e03-402d-b008-a9a683c4f06f" "Get specific tenant"
test_endpoint "GET" "http://localhost:8002/api/v1/tenants/1a2b65b7-8e03-402d-b008-a9a683c4f06f/quotas" "Get tenant quotas"

echo ""
echo "Summary"
echo "-------"
echo "Total tests: $TOTAL"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed!${NC}"
    exit 1
fi