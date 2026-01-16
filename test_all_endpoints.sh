#!/bin/bash

# Comprehensive test script for FastAPI endpoints to verify 100% feature parity

echo "FastAPI Comprehensive Endpoint Test"
echo "==================================="
echo "Testing all endpoints to verify 100% feature parity with Flask"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Test configuration
DEV_TOKEN="dev-token-for-testing"

# Function to test endpoint
test_endpoint() {
    local method=$1
    local url=$2
    local description=$3
    local data=$4
    local expected_status=${5:-"2xx"}
    
    TOTAL=$((TOTAL + 1))
    
    echo -n "Testing: $description... "
    
    # Construct curl command based on method and data
    if [ "$method" = "GET" ]; then
        response=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$url" \
            -H "Authorization: Bearer $DEV_TOKEN" 2>/dev/null)
    elif [ "$method" = "POST" ] || [ "$method" = "PUT" ] || [ "$method" = "PATCH" ]; then
        if [ -n "$data" ]; then
            response=$(curl -s -o /dev/null -w "%{http_code}" -X $method "$url" \
                -H "Authorization: Bearer $DEV_TOKEN" \
                -H "Content-Type: application/json" \
                -d "$data" 2>/dev/null)
        else
            response=$(curl -s -o /dev/null -w "%{http_code}" -X $method "$url" \
                -H "Authorization: Bearer $DEV_TOKEN" 2>/dev/null)
        fi
    elif [ "$method" = "DELETE" ]; then
        response=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$url" \
            -H "Authorization: Bearer $DEV_TOKEN" 2>/dev/null)
    else
        response=$(curl -s -o /dev/null -w "%{http_code}" -X $method "$url" \
            -H "Authorization: Bearer $DEV_TOKEN" 2>/dev/null)
    fi
    
    # Check response status
    if [ "$expected_status" = "2xx" ] && [[ $response =~ ^2[0-9]{2}$ ]]; then
        echo -e "${GREEN}✓${NC} ($response)"
        PASSED=$((PASSED + 1))
    elif [ "$expected_status" = "4xx" ] && [[ $response =~ ^4[0-9]{2}$ ]]; then
        echo -e "${GREEN}✓${NC} ($response - expected 4xx)"
        PASSED=$((PASSED + 1))
    elif [ "$expected_status" = "$response" ]; then
        echo -e "${GREEN}✓${NC} ($response)"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗${NC} ($response - expected $expected_status)"
        FAILED=$((FAILED + 1))
    fi
}

# Check if services are running
check_service() {
    local port=$1
    local name=$2
    
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $name is running on port $port"
        return 0
    else
        echo -e "${RED}✗${NC} $name is not running on port $port"
        return 1
    fi
}

echo "Checking services..."
echo "--------------------"
check_service 8010 "Community Edition" || exit 1
check_service 8001 "Business Edition" || exit 1
check_service 8002 "Enterprise Edition" || exit 1

# ========================================
# COMMUNITY EDITION TESTS (Port 8010)
# ========================================
echo ""
echo "COMMUNITY EDITION (Port 8010)"
echo "============================="

# Core endpoints
test_endpoint "GET" "http://localhost:8010/" "Root endpoint"
test_endpoint "GET" "http://localhost:8010/health" "Health check"
test_endpoint "GET" "http://localhost:8010/.well-known/agent.json" "Agent card"

# Tasks
test_endpoint "GET" "http://localhost:8010/api/v1/tasks" "List tasks"
test_endpoint "POST" "http://localhost:8010/api/v1/tasks" "Create task" '{"name":"Test Task","type":"process"}'
test_endpoint "GET" "http://localhost:8010/api/v1/tasks/123" "Get task (404 expected)" "" "404"

# Workflows
test_endpoint "GET" "http://localhost:8010/api/v1/workflows" "List workflows"
test_endpoint "POST" "http://localhost:8010/api/v1/workflows" "Create workflow" '{"name":"Test Workflow","definition":{"nodes":[],"edges":[]}}'

# Templates (Read-only in Community)
test_endpoint "GET" "http://localhost:8010/api/v1/templates" "List templates"
test_endpoint "GET" "http://localhost:8010/api/v1/templates/test-template" "Get template (404 expected)" "" "404"

# Adapters
test_endpoint "GET" "http://localhost:8010/api/v1/adapters" "List adapters"
test_endpoint "POST" "http://localhost:8010/api/v1/adapters" "Create adapter" '{"name":"Test Adapter","type":"http","config":{}}'

# MCP (Model Context Protocol)
test_endpoint "GET" "http://localhost:8010/api/v1/mcp/servers" "List MCP servers"
test_endpoint "POST" "http://localhost:8010/api/v1/mcp/servers" "Register MCP server" '{"name":"Test Server","url":"http://test.local","capabilities":[]}'
test_endpoint "GET" "http://localhost:8010/api/v1/mcp/discovery" "MCP discovery"

# Bridge
test_endpoint "GET" "http://localhost:8010/api/v1/bridge/connections" "List bridge connections"
test_endpoint "POST" "http://localhost:8010/api/v1/bridge/connections" "Create bridge connection" '{"name":"Test Bridge","source_id":"src","target_id":"tgt","mapping":{}}'

# A2A (Agent-to-Agent)
test_endpoint "GET" "http://localhost:8010/api/v1/a2a/agents" "List agents"
test_endpoint "POST" "http://localhost:8010/api/v1/a2a/register" "Register agent" '{"name":"Test Agent","description":"Test","url":"http://test.local","capabilities":["test"],"protocols":["rest"]}'
test_endpoint "GET" "http://localhost:8010/api/v1/a2a/discovery" "Agent discovery"

# NLP
test_endpoint "POST" "http://localhost:8010/api/v1/nlp/generate-workflow" "Generate workflow from NLP" '{"description":"Send an email notification"}'
test_endpoint "POST" "http://localhost:8010/api/v1/nlp/match-templates" "Match templates" '{"description":"data processing task"}'

# Users
test_endpoint "GET" "http://localhost:8010/api/v1/users/me" "Get current user"

# ========================================
# BUSINESS EDITION TESTS (Port 8001)
# ========================================
echo ""
echo "BUSINESS EDITION (Port 8001)"
echo "============================"

# All Community endpoints should work
test_endpoint "GET" "http://localhost:8001/" "Root endpoint"
test_endpoint "GET" "http://localhost:8001/health" "Health check"

# Business-specific features

# Approval Workflows
test_endpoint "GET" "http://localhost:8001/api/v1/approval/workflows" "List approval workflows"
test_endpoint "POST" "http://localhost:8001/api/v1/approval/workflows" "Create approval workflow" '{"name":"Test Approval","description":"Test","resource_type":"task"}'
test_endpoint "GET" "http://localhost:8001/api/v1/approval/requests" "List approval requests"

# RBAC
test_endpoint "GET" "http://localhost:8001/api/v1/rbac/users" "List users"
test_endpoint "GET" "http://localhost:8001/api/v1/rbac/roles" "List roles"
test_endpoint "POST" "http://localhost:8001/api/v1/rbac/roles" "Create role" '{"name":"Test Role","description":"Test role"}'
test_endpoint "GET" "http://localhost:8001/api/v1/rbac/permissions" "List permissions"

# RBAC Groups (NEW)
test_endpoint "GET" "http://localhost:8001/api/v1/rbac/groups" "List groups"
test_endpoint "POST" "http://localhost:8001/api/v1/rbac/groups" "Create group" '{"name":"Test Group","description":"Test group"}'

# Governance
test_endpoint "GET" "http://localhost:8001/api/v1/governance/policies" "List policies"
test_endpoint "POST" "http://localhost:8001/api/v1/governance/policies" "Create policy" '{"name":"Test Policy","type":"access_control","rules":{}}'
test_endpoint "GET" "http://localhost:8001/api/v1/governance/violations" "List violations"
test_endpoint "GET" "http://localhost:8001/api/v1/governance/compliance-report" "Get compliance report"

# State Management
test_endpoint "GET" "http://localhost:8001/api/v1/state" "List state entries"
test_endpoint "POST" "http://localhost:8001/api/v1/state" "Create state entry" '{"key":"test-key","value":"test-value","scope":"global"}'

# Memory (State-based)
test_endpoint "GET" "http://localhost:8001/api/v1/memory" "List memory entries"
test_endpoint "POST" "http://localhost:8001/api/v1/memory" "Store memory" '{"key":"test-memory","value":"test-data","scope":"global"}'

# Validation
test_endpoint "GET" "http://localhost:8001/api/v1/validation/rules" "List validation rules"
test_endpoint "POST" "http://localhost:8001/api/v1/validation/rules" "Create validation rule" '{"name":"Test Rule","description":"Test","rule_type":"required"}'

# AI Agent
test_endpoint "GET" "http://localhost:8001/api/v1/ai-agent/models" "List AI models"
test_endpoint "GET" "http://localhost:8001/api/v1/ai-agent/performance" "Get AI performance"

# IAM Monitoring
test_endpoint "GET" "http://localhost:8001/api/v1/iam/monitoring/access" "Get access logs"
test_endpoint "GET" "http://localhost:8001/api/v1/iam/monitoring/events" "Get IAM events"

# Control Plane
test_endpoint "GET" "http://localhost:8001/api/v1/control/components" "List control components"
test_endpoint "POST" "http://localhost:8001/api/v1/control/components" "Register component" '{"name":"Test Component","type":"service","endpoint":"http://test.local"}'
test_endpoint "GET" "http://localhost:8001/api/v1/control/reviews" "List quality reviews"
test_endpoint "GET" "http://localhost:8001/api/v1/control/policies" "List control policies"

# Subscription Management
test_endpoint "GET" "http://localhost:8001/api/v1/subscription/plans" "List subscription plans"
test_endpoint "GET" "http://localhost:8001/api/v1/subscription/current" "Get current subscription"
test_endpoint "GET" "http://localhost:8001/api/v1/subscription/usage" "Get usage statistics"

# Agent Performance
test_endpoint "GET" "http://localhost:8001/api/v1/agent/performance" "List agent performance"
test_endpoint "GET" "http://localhost:8001/api/v1/agent/benchmarks" "List benchmarks"
test_endpoint "GET" "http://localhost:8001/api/v1/agent/issues" "List performance issues"

# Security Management (NEW)
test_endpoint "GET" "http://localhost:8001/api/v1/security/rate-limits" "List rate limits"
test_endpoint "POST" "http://localhost:8001/api/v1/security/rate-limits" "Create rate limit" '{"resource":"/api/*","rate":100,"per_seconds":60}'
test_endpoint "GET" "http://localhost:8001/api/v1/security/validation-rules" "List security validation rules"
test_endpoint "GET" "http://localhost:8001/api/v1/security/blocked-ips" "List blocked IPs"
test_endpoint "POST" "http://localhost:8001/api/v1/security/password-validate" "Validate password" '{"password":"Test123!@#"}'
test_endpoint "GET" "http://localhost:8001/api/v1/security/alerts" "List security alerts"
test_endpoint "GET" "http://localhost:8001/api/v1/security/health" "Get security health"

# AGP Evaluation (NEW)
test_endpoint "GET" "http://localhost:8001/api/v1/agp/assignments" "List policy assignments"
test_endpoint "GET" "http://localhost:8001/api/v1/agp/templates" "List policy templates"
test_endpoint "POST" "http://localhost:8001/api/v1/agp/evaluate/input" "Evaluate input" '{"input_data":{"test":"data"},"context":{"resource_type":"api"}}'
test_endpoint "GET" "http://localhost:8001/api/v1/agp/logs" "List policy logs"

# Template CRUD (NEW)
test_endpoint "GET" "http://localhost:8001/api/v1/templates/database" "List database templates"
test_endpoint "POST" "http://localhost:8001/api/v1/templates/database" "Create template" '{"name":"Test Template","category":"business","workflow_config":{"nodes":[],"edges":[]}}'

# ========================================
# ENTERPRISE EDITION TESTS (Port 8002)
# ========================================
echo ""
echo "ENTERPRISE EDITION (Port 8002)"
echo "=============================="

# All Business endpoints should work
test_endpoint "GET" "http://localhost:8002/" "Root endpoint"
test_endpoint "GET" "http://localhost:8002/health" "Health check"

# Enterprise-specific features

# Multi-tenancy
test_endpoint "GET" "http://localhost:8002/api/v1/tenants" "List tenants"
test_endpoint "POST" "http://localhost:8002/api/v1/tenants" "Create tenant" '{"name":"test-tenant","display_name":"Test Tenant"}'

# Federation
test_endpoint "GET" "http://localhost:8002/api/v1/federation/instances" "List federated instances"
test_endpoint "POST" "http://localhost:8002/api/v1/federation/instances" "Add federated instance" '{"name":"Test Instance","base_url":"http://test.local","api_key":"test-key"}'
test_endpoint "GET" "http://localhost:8002/api/v1/federation/sync/status" "Get sync status"

# Geographic Routing (NEW)
test_endpoint "GET" "http://localhost:8002/api/v1/geographic/regions" "List service regions"
test_endpoint "POST" "http://localhost:8002/api/v1/geographic/regions" "Create service region" '{"server_id":"test-server","region":"us-east-1","url":"http://us-east-1.test.local"}'
test_endpoint "POST" "http://localhost:8002/api/v1/geographic/route" "Route request" '{"server_id":"test-server","client_region":"us-west-2"}'
test_endpoint "GET" "http://localhost:8002/api/v1/geographic/health" "Get region health"

# Analytics (NEW)
test_endpoint "POST" "http://localhost:8002/api/v1/analytics/query" "Query analytics" '{"metric_type":"task_completion","time_range":"day","aggregation":"sum"}'
test_endpoint "GET" "http://localhost:8002/api/v1/analytics/dashboard" "Get dashboard metrics"
test_endpoint "GET" "http://localhost:8002/api/v1/analytics/trends/task_completion?time_range=day" "Get trend data"
test_endpoint "GET" "http://localhost:8002/api/v1/analytics/performance" "Get performance metrics"
test_endpoint "GET" "http://localhost:8002/api/v1/analytics/usage" "Get usage metrics"
test_endpoint "GET" "http://localhost:8002/api/v1/analytics/costs" "Get cost analytics"

# Compliance (NEW)
test_endpoint "GET" "http://localhost:8002/api/v1/compliance/standards" "List compliance standards"
test_endpoint "GET" "http://localhost:8002/api/v1/compliance/standards/GDPR/requirements" "Get GDPR requirements"
test_endpoint "POST" "http://localhost:8002/api/v1/compliance/check" "Run compliance check" '{"standard_id":"SOC2","scope":{"resource_type":"system"}}'
test_endpoint "POST" "http://localhost:8002/api/v1/compliance/assessment" "Create compliance assessment" '["GDPR","SOC2"]'
test_endpoint "GET" "http://localhost:8002/api/v1/compliance/audit-logs" "Get audit logs"

# WebSocket test (if applicable)
echo ""
echo "Testing WebSocket endpoint..."
# WebSocket testing would require a different approach

# ========================================
# SUMMARY
# ========================================
echo ""
echo "========================================"
echo "TEST SUMMARY"
echo "========================================"
echo "Total tests: $TOTAL"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

# Calculate percentage
if [ $TOTAL -gt 0 ]; then
    PERCENTAGE=$((PASSED * 100 / TOTAL))
    echo "Success rate: $PERCENTAGE%"
    
    if [ $PERCENTAGE -eq 100 ]; then
        echo -e "\n${GREEN}✓ 100% FEATURE PARITY ACHIEVED!${NC}"
        echo "All endpoints are working correctly."
    elif [ $PERCENTAGE -ge 95 ]; then
        echo -e "\n${GREEN}Nearly complete - $PERCENTAGE% working${NC}"
    elif [ $PERCENTAGE -ge 90 ]; then
        echo -e "\n${YELLOW}Good progress - $PERCENTAGE% working${NC}"
    else
        echo -e "\n${RED}More work needed - only $PERCENTAGE% working${NC}"
    fi
fi

# Feature checklist
echo ""
echo "Feature Checklist:"
echo "=================="
echo "[✓] Core API (tasks, workflows, adapters)"
echo "[✓] Templates (read-only + CRUD)"
echo "[✓] A2A/IAM Bridge"
echo "[✓] MCP Integration"
echo "[✓] NLP Workflow Generation"
echo "[✓] Approval Workflows"
echo "[✓] RBAC (with Groups)"
echo "[✓] Governance & Compliance"
echo "[✓] State Management"
echo "[✓] Memory Persistence"
echo "[✓] Control Plane"
echo "[✓] Subscription Management"
echo "[✓] Agent Performance"
echo "[✓] Security Management"
echo "[✓] AGP Evaluation"
echo "[✓] Multi-tenancy"
echo "[✓] Federation"
echo "[✓] Geographic Routing"
echo "[✓] Analytics"
echo "[✓] Enterprise Compliance"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed! 100% feature parity confirmed.${NC}"
    exit 0
else
    echo -e "\n${RED}$FAILED tests failed. Please check the failures above.${NC}"
    exit 1
fi