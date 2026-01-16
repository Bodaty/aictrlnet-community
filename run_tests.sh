#!/bin/bash

# AICtrlNet FastAPI Test Runner Script

echo "================================"
echo "AICtrlNet FastAPI Test Suite"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo "Activating virtual environment..."
    source venv/bin/activate 2>/dev/null || {
        echo -e "${RED}Failed to activate virtual environment${NC}"
        echo "Please ensure virtual environment exists at ./venv"
        exit 1
    }
fi

# Install test dependencies if needed
echo "Checking test dependencies..."
pip install -q pytest pytest-asyncio pytest-cov pytest-mock pytest-benchmark matplotlib numpy psutil

# Create test results directory
mkdir -p test_results

# Function to run test category
run_test_category() {
    local category=$1
    local marker=$2
    local description=$3
    
    echo ""
    echo -e "${GREEN}Running $description...${NC}"
    echo "================================"
    
    if [ "$marker" == "all" ]; then
        pytest src/tests -v --tb=short --junit-xml=test_results/${category}_results.xml
    else
        pytest src/tests -v -m "$marker" --tb=short --junit-xml=test_results/${category}_results.xml
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $description passed${NC}"
    else
        echo -e "${RED}✗ $description failed${NC}"
        FAILED_TESTS+=("$description")
    fi
}

# Track failed tests
FAILED_TESTS=()

# Run tests based on argument
case "${1:-all}" in
    unit)
        run_test_category "unit" "unit" "Unit Tests"
        ;;
    integration)
        run_test_category "integration" "integration" "Integration Tests"
        ;;
    e2e)
        run_test_category "e2e" "e2e" "End-to-End Tests"
        ;;
    performance)
        run_test_category "performance" "benchmark" "Performance Benchmarks"
        ;;
    quick)
        echo "Running quick test suite (unit tests only)..."
        run_test_category "unit" "unit" "Unit Tests"
        ;;
    all)
        echo "Running complete test suite..."
        
        # Run each category
        run_test_category "unit" "unit" "Unit Tests"
        run_test_category "integration" "integration" "Integration Tests"
        run_test_category "e2e" "e2e" "End-to-End Tests"
        
        # Performance tests are optional and slow
        if [ "${INCLUDE_BENCHMARKS:-false}" == "true" ]; then
            run_test_category "performance" "benchmark" "Performance Benchmarks"
        else
            echo ""
            echo -e "${YELLOW}Skipping performance benchmarks (set INCLUDE_BENCHMARKS=true to run)${NC}"
        fi
        ;;
    coverage)
        echo "Running tests with coverage report..."
        pytest src/tests --cov=src --cov-report=html --cov-report=term-missing
        echo ""
        echo "Coverage report generated at: htmlcov/index.html"
        ;;
    *)
        echo "Usage: $0 [unit|integration|e2e|performance|all|quick|coverage]"
        echo ""
        echo "Options:"
        echo "  unit         - Run unit tests only"
        echo "  integration  - Run integration tests only"
        echo "  e2e          - Run end-to-end tests only"
        echo "  performance  - Run performance benchmarks only"
        echo "  all          - Run all tests (default)"
        echo "  quick        - Run quick tests (unit only)"
        echo "  coverage     - Run tests with coverage report"
        echo ""
        echo "Environment variables:"
        echo "  INCLUDE_BENCHMARKS=true - Include performance benchmarks in 'all' run"
        exit 1
        ;;
esac

# Summary
echo ""
echo "================================"
echo "Test Summary"
echo "================================"

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}- $test${NC}"
    done
    exit 1
fi