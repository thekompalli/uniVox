#!/bin/bash

# PS-06 Competition System - Test Runner Script
# Comprehensive testing script with multiple test types and reporting

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
TEST_TYPE="${TEST_TYPE:-all}"
COVERAGE="${COVERAGE:-true}"
PARALLEL="${PARALLEL:-true}"
VERBOSE="${VERBOSE:-false}"
QUICK="${QUICK:-false}"
INTEGRATION="${INTEGRATION:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-test_results}"
PYTEST_ARGS="${PYTEST_ARGS:-}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_header() {
    echo -e "${BLUE}"
    echo "================================================================="
    echo "PS-06 Competition System - Test Runner"
    echo "================================================================="
    echo -e "${NC}"
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run tests for PS-06 Competition System

OPTIONS:
    -h, --help              Show this help message
    -t, --type TYPE         Test type: unit, integration, performance, all (default: all)
    -c, --coverage          Enable coverage reporting (default: true)
    -p, --parallel          Run tests in parallel (default: true)
    -v, --verbose           Verbose output
    -q, --quick             Quick test run (skip slow tests)
    --no-integration        Skip integration tests
    --output-dir DIR        Output directory for reports (default: test_results)
    --pytest-args ARGS     Additional pytest arguments

TEST TYPES:
    unit                    Unit tests only
    integration             Integration tests only
    performance             Performance/benchmark tests
    api                     API endpoint tests
    models                  Model inference tests
    services                Service layer tests
    utils                   Utility function tests
    all                     All test types

EXAMPLES:
    $0                                  # Run all tests with coverage
    $0 --type unit --quick             # Quick unit tests only
    $0 --type integration --verbose    # Verbose integration tests
    $0 --no-coverage --parallel        # Fast run without coverage

ENVIRONMENT VARIABLES:
    TEST_DATABASE_URL       Test database URL
    REDIS_TEST_URL         Test Redis URL
    PYTEST_WORKERS        Number of parallel workers
    TEST_TIMEOUT           Test timeout in seconds

EOF
}

check_requirements() {
    log_info "Checking test requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required"
        exit 1
    fi
    
    # Check pytest
    if ! python3 -c "import pytest" &> /dev/null; then
        log_error "pytest is required. Install with: pip install pytest"
        exit 1
    fi
    
    # Check coverage if enabled
    if [[ "$COVERAGE" == "true" ]]; then
        if ! python3 -c "import coverage" &> /dev/null; then
            log_warning "coverage not found. Installing..."
            pip install coverage pytest-cov
        fi
    fi
    
    # Check for parallel testing
    if [[ "$PARALLEL" == "true" ]]; then
        if ! python3 -c "import pytest_xdist" &> /dev/null; then
            log_warning "pytest-xdist not found. Installing..."
            pip install pytest-xdist
        fi
    fi
    
    log_success "Requirements check passed"
}

setup_test_environment() {
    log_info "Setting up test environment..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Set test environment variables
    export TESTING=true
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
    
    # Load test environment
    if [[ -f ".env.test" ]]; then
        set -a
        source .env.test
        set +a
        log_info "Loaded .env.test"
    elif [[ -f ".env" ]]; then
        set -a
        source .env
        set +a
        log_info "Loaded .env (no .env.test found)"
    fi
    
    # Override for testing
    export DATABASE_URL="${TEST_DATABASE_URL:-sqlite:///test.db}"
    export REDIS_URL="${REDIS_TEST_URL:-redis://localhost:6379/15}"
    export LOG_LEVEL="${LOG_LEVEL:-WARNING}"
    
    # Create test directories
    mkdir -p test_data/{audio,models,results}
    mkdir -p logs/test
    
    log_success "Test environment setup completed"
}

cleanup_test_environment() {
    log_info "Cleaning up test environment..."
    
    # Remove test databases
    if [[ -f "test.db" ]]; then
        rm -f test.db
    fi
    
    # Clean test cache
    if [[ -d ".pytest_cache" ]]; then
        rm -rf .pytest_cache
    fi
    
    # Clean coverage files
    if [[ -f ".coverage" ]]; then
        rm -f .coverage
    fi
    
    log_success "Test environment cleanup completed"
}

run_unit_tests() {
    log_info "Running unit tests..."
    
    local pytest_cmd="pytest tests/unit/"
    local args=()
    
    # Add basic arguments
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("-v")
    else
        args+=("-q")
    fi
    
    # Add coverage if enabled
    if [[ "$COVERAGE" == "true" ]]; then
        args+=("--cov=src" "--cov-report=html:$OUTPUT_DIR/coverage_unit" "--cov-report=xml:$OUTPUT_DIR/coverage_unit.xml")
    fi
    
    # Add parallel execution
    if [[ "$PARALLEL" == "true" ]]; then
        local workers="${PYTEST_WORKERS:-auto}"
        args+=("-n" "$workers")
    fi
    
    # Add quick mode
    if [[ "$QUICK" == "true" ]]; then
        args+=("-m" "not slow")
    fi
    
    # Add timeout
    local timeout="${TEST_TIMEOUT:-300}"
    args+=("--timeout=$timeout")
    
    # Add JUnit XML report
    args+=("--junitxml=$OUTPUT_DIR/junit_unit.xml")
    
    # Run tests
    if $pytest_cmd "${args[@]}" $PYTEST_ARGS; then
        log_success "Unit tests passed"
        return 0
    else
        log_error "Unit tests failed"
        return 1
    fi
}

run_integration_tests() {
    if [[ "$INTEGRATION" != "true" ]]; then
        log_info "Skipping integration tests"
        return 0
    fi
    
    log_info "Running integration tests..."
    
    # Check if test services are available
    check_test_services
    
    local pytest_cmd="pytest tests/integration/"
    local args=()
    
    # Add basic arguments
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("-v")
    else
        args+=("-q")
    fi
    
    # Add coverage if enabled
    if [[ "$COVERAGE" == "true" ]]; then
        args+=("--cov=src" "--cov-append" "--cov-report=html:$OUTPUT_DIR/coverage_integration" "--cov-report=xml:$OUTPUT_DIR/coverage_integration.xml")
    fi
    
    # Integration tests should not run in parallel by default
    if [[ "$PARALLEL" == "true" ]]; then
        log_warning "Running integration tests sequentially for stability"
    fi
    
    # Add timeout (longer for integration tests)
    local timeout="${INTEGRATION_TEST_TIMEOUT:-600}"
    args+=("--timeout=$timeout")
    
    # Add JUnit XML report
    args+=("--junitxml=$OUTPUT_DIR/junit_integration.xml")
    
    # Run tests
    if $pytest_cmd "${args[@]}" $PYTEST_ARGS; then
        log_success "Integration tests passed"
        return 0
    else
        log_error "Integration tests failed"
        return 1
    fi
}

run_api_tests() {
    log_info "Running API tests..."
    
    local pytest_cmd="pytest tests/test_api/"
    local args=()
    
    # Add basic arguments
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("-v")
    else
        args+=("-q")
    fi
    
    # Add coverage if enabled
    if [[ "$COVERAGE" == "true" ]]; then
        args+=("--cov=src.api" "--cov-append" "--cov-report=html:$OUTPUT_DIR/coverage_api")
    fi
    
    # Add JUnit XML report
    args+=("--junitxml=$OUTPUT_DIR/junit_api.xml")
    
    # Run tests
    if $pytest_cmd "${args[@]}" $PYTEST_ARGS; then
        log_success "API tests passed"
        return 0
    else
        log_error "API tests failed"
        return 1
    fi
}

run_model_tests() {
    log_info "Running model tests..."
    
    # Check if models are available
    if [[ ! -d "models" ]] && [[ "$QUICK" != "true" ]]; then
        log_warning "Models directory not found. Some tests may be skipped."
    fi
    
    local pytest_cmd="pytest tests/test_models/"
    local args=()
    
    # Add basic arguments
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("-v")
    else
        args+=("-q")
    fi
    
    # Add coverage if enabled
    if [[ "$COVERAGE" == "true" ]]; then
        args+=("--cov=src.models" "--cov-append" "--cov-report=html:$OUTPUT_DIR/coverage_models")
    fi
    
    # Model tests can be slow
    if [[ "$QUICK" == "true" ]]; then
        args+=("-m" "not slow")
    fi
    
    # Add timeout (longer for model tests)
    local timeout="${MODEL_TEST_TIMEOUT:-900}"
    args+=("--timeout=$timeout")
    
    # Add JUnit XML report
    args+=("--junitxml=$OUTPUT_DIR/junit_models.xml")
    
    # Run tests
    if $pytest_cmd "${args[@]}" $PYTEST_ARGS; then
        log_success "Model tests passed"
        return 0
    else
        log_error "Model tests failed"
        return 1
    fi
}

run_performance_tests() {
    log_info "Running performance tests..."
    
    # Check if benchmark plugin is available
    if ! python3 -c "import pytest_benchmark" &> /dev/null; then
        log_warning "pytest-benchmark not found. Installing..."
        pip install pytest-benchmark
    fi
    
    local pytest_cmd="pytest tests/performance/"
    local args=()
    
    # Add basic arguments
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("-v")
    else
        args+=("-q")
    fi
    
    # Add benchmark arguments
    args+=("--benchmark-only" "--benchmark-sort=mean" "--benchmark-json=$OUTPUT_DIR/benchmark.json")
    
    # Add JUnit XML report
    args+=("--junitxml=$OUTPUT_DIR/junit_performance.xml")
    
    # Run tests
    if $pytest_cmd "${args[@]}" $PYTEST_ARGS; then
        log_success "Performance tests passed"
        return 0
    else
        log_error "Performance tests failed"
        return 1
    fi
}

check_test_services() {
    log_info "Checking test services..."
    
    # Check test database
    if [[ "$DATABASE_URL" == *"postgresql"* ]]; then
        if ! python3 -c "
import asyncpg
import asyncio
async def check():
    try:
        conn = await asyncpg.connect('$DATABASE_URL')
        await conn.close()
        print('PostgreSQL connection OK')
    except Exception as e:
        print(f'PostgreSQL connection failed: {e}')
        exit(1)
asyncio.run(check())
"; then
            log_error "Test database connection failed"
            exit 1
        fi
    fi
    
    # Check test Redis
    if command -v redis-cli &> /dev/null; then
        if ! redis-cli -u "$REDIS_URL" ping &> /dev/null; then
            log_warning "Test Redis connection failed"
        fi
    fi
    
    log_success "Test services check completed"
}

run_linting() {
    log_info "Running code quality checks..."
    
    local errors=0
    
    # Run flake8
    if command -v flake8 &> /dev/null; then
        log_info "Running flake8..."
        if ! flake8 src/ tests/ --output-file="$OUTPUT_DIR/flake8.txt"; then
            log_error "flake8 found issues"
            ((errors++))
        fi
    fi
    
    # Run mypy
    if command -v mypy &> /dev/null; then
        log_info "Running mypy..."
        if ! mypy src/ --html-report "$OUTPUT_DIR/mypy" --txt-report "$OUTPUT_DIR"; then
            log_warning "mypy found type issues"
        fi
    fi
    
    # Run black check
    if command -v black &> /dev/null; then
        log_info "Running black check..."
        if ! black --check src/ tests/ --diff > "$OUTPUT_DIR/black.diff"; then
            log_warning "Code formatting issues found"
        fi
    fi
    
    # Run isort check
    if command -v isort &> /dev/null; then
        log_info "Running isort check..."
        if ! isort --check-only src/ tests/ --diff > "$OUTPUT_DIR/isort.diff"; then
            log_warning "Import sorting issues found"
        fi
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_success "Code quality checks passed"
        return 0
    else
        log_error "Code quality checks failed"
        return 1
    fi
}

generate_reports() {
    log_info "Generating test reports..."
    
    # Combine coverage reports if multiple test types were run
    if [[ "$COVERAGE" == "true" ]]; then
        if command -v coverage &> /dev/null; then
            coverage combine 2>/dev/null || true
            coverage html -d "$OUTPUT_DIR/coverage_combined"
            coverage xml -o "$OUTPUT_DIR/coverage_combined.xml"
            coverage report > "$OUTPUT_DIR/coverage_summary.txt"
            
            log_info "Coverage reports generated in $OUTPUT_DIR/"
        fi
    fi
    
    # Generate summary report
    cat > "$OUTPUT_DIR/test_summary.txt" << EOF
PS-06 Competition System - Test Summary
=======================================
Generated: $(date)
Test Type: $TEST_TYPE
Quick Mode: $QUICK
Coverage: $COVERAGE
Parallel: $PARALLEL

Test Results:
============
$(find "$OUTPUT_DIR" -name "junit_*.xml" -exec basename {} \; | sed 's/junit_//' | sed 's/.xml//' | sort)

Coverage Files:
==============
$(find "$OUTPUT_DIR" -name "*coverage*" -type f | sort)

Additional Reports:
==================
$(find "$OUTPUT_DIR" -name "*.txt" -o -name "*.json" -o -name "*.diff" | sort)

EOF
    
    log_success "Test reports generated in $OUTPUT_DIR/"
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -t|--type)
                TEST_TYPE="$2"
                shift 2
                ;;
            -c|--coverage)
                COVERAGE=true
                shift
                ;;
            --no-coverage)
                COVERAGE=false
                shift
                ;;
            -p|--parallel)
                PARALLEL=true
                shift
                ;;
            --no-parallel)
                PARALLEL=false
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -q|--quick)
                QUICK=true
                shift
                ;;
            --no-integration)
                INTEGRATION=false
                shift
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --pytest-args)
                PYTEST_ARGS="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    print_header
    
    # Setup
    check_requirements
    setup_test_environment
    
    # Cleanup on exit
    trap cleanup_test_environment EXIT
    
    # Run tests based on type
    local test_failures=0
    
    case $TEST_TYPE in
        unit)
            run_unit_tests || ((test_failures++))
            ;;
        integration)
            run_integration_tests || ((test_failures++))
            ;;
        api)
            run_api_tests || ((test_failures++))
            ;;
        models)
            run_model_tests || ((test_failures++))
            ;;
        performance)
            run_performance_tests || ((test_failures++))
            ;;
        services)
            pytest tests/test_services/ --junitxml="$OUTPUT_DIR/junit_services.xml" || ((test_failures++))
            ;;
        utils)
            pytest tests/test_utils/ --junitxml="$OUTPUT_DIR/junit_utils.xml" || ((test_failures++))
            ;;
        all)
            run_unit_tests || ((test_failures++))
            run_integration_tests || ((test_failures++))
            run_api_tests || ((test_failures++))
            
            if [[ "$QUICK" != "true" ]]; then
                run_model_tests || ((test_failures++))
                run_performance_tests || ((test_failures++))
            fi
            ;;
        *)
            log_error "Unknown test type: $TEST_TYPE"
            print_usage
            exit 1
            ;;
    esac
    
    # Run linting
    run_linting || ((test_failures++))
    
    # Generate reports
    generate_reports
    
    # Summary
    echo
    if [[ $test_failures -eq 0 ]]; then
        log_success "All tests passed! 🎉"
        echo "Test reports available in: $OUTPUT_DIR/"
        exit 0
    else
        log_error "$test_failures test suite(s) failed"
        echo "Check test reports in: $OUTPUT_DIR/"
        exit 1
    fi
}

# Run main function
main "$@"