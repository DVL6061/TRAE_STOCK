#!/bin/bash

# Deployment Validation Script for Stock Prediction System
# This script validates the entire AWS deployment and all services

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/stock-prediction/validation.log"
TIMEOUT=30
MAX_RETRIES=5
RETRY_DELAY=10

# Load environment variables
if [[ -f "/opt/stock-prediction/.env" ]]; then
    source "/opt/stock-prediction/.env"
fi

# Service endpoints
DOMAIN="${DOMAIN:-localhost}"
BACKEND_URL="https://$DOMAIN/api"
FRONTEND_URL="https://$DOMAIN"
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3001"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-stock_prediction}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNING_TESTS=0

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# Print test result
print_test_result() {
    local status=$1
    local test_name=$2
    local message=$3
    local details="${4:-}"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    case $status in
        "PASS")
            echo -e "${GREEN}✓ PASS${NC} $test_name: $message"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            log "INFO" "TEST PASS: $test_name - $message"
            ;;
        "FAIL")
            echo -e "${RED}✗ FAIL${NC} $test_name: $message"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            log "ERROR" "TEST FAIL: $test_name - $message"
            if [[ -n "$details" ]]; then
                echo -e "  ${RED}Details:${NC} $details"
                log "ERROR" "TEST FAIL Details: $details"
            fi
            ;;
        "WARN")
            echo -e "${YELLOW}⚠ WARN${NC} $test_name: $message"
            WARNING_TESTS=$((WARNING_TESTS + 1))
            log "WARNING" "TEST WARN: $test_name - $message"
            if [[ -n "$details" ]]; then
                echo -e "  ${YELLOW}Details:${NC} $details"
                log "WARNING" "TEST WARN Details: $details"
            fi
            ;;
        "INFO")
            echo -e "${BLUE}ℹ INFO${NC} $test_name: $message"
            log "INFO" "TEST INFO: $test_name - $message"
            ;;
    esac
}

# HTTP request with retry
http_request() {
    local url=$1
    local method=${2:-GET}
    local expected_status=${3:-200}
    local timeout=${4:-$TIMEOUT}
    local retries=${5:-$MAX_RETRIES}
    
    local attempt=1
    while [[ $attempt -le $retries ]]; do
        local response
        local http_code
        local response_time
        
        if response=$(curl -s -w "\n%{http_code}\n%{time_total}" \
                          --max-time "$timeout" \
                          --connect-timeout 5 \
                          -X "$method" \
                          -H "Accept: application/json" \
                          -H "User-Agent: ValidationScript/1.0" \
                          "$url" 2>/dev/null); then
            
            http_code=$(echo "$response" | tail -n 2 | head -n 1)
            response_time=$(echo "$response" | tail -n 1)
            
            if [[ "$http_code" == "$expected_status" ]]; then
                echo "SUCCESS|$http_code|$response_time"
                return 0
            else
                echo "HTTP_ERROR|$http_code|$response_time"
                if [[ $attempt -eq $retries ]]; then
                    return 1
                fi
            fi
        else
            echo "CONNECTION_ERROR|0|0"
            if [[ $attempt -eq $retries ]]; then
                return 1
            fi
        fi
        
        sleep $RETRY_DELAY
        attempt=$((attempt + 1))
    done
}

# Test SSL certificate
test_ssl_certificate() {
    local test_name="SSL Certificate"
    
    if [[ "$DOMAIN" == "localhost" ]]; then
        print_test_result "INFO" "$test_name" "Skipped (localhost)"
        return 0
    fi
    
    if command -v openssl >/dev/null 2>&1; then
        local cert_info
        if cert_info=$(echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:443" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null); then
            local not_after
            not_after=$(echo "$cert_info" | grep "notAfter" | cut -d= -f2)
            local expiry_epoch
            expiry_epoch=$(date -d "$not_after" +%s 2>/dev/null || echo "0")
            local current_epoch
            current_epoch=$(date +%s)
            local days_until_expiry
            days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
            
            if [[ "$days_until_expiry" -gt 30 ]]; then
                print_test_result "PASS" "$test_name" "Valid for $days_until_expiry days"
            elif [[ "$days_until_expiry" -gt 7 ]]; then
                print_test_result "WARN" "$test_name" "Expires in $days_until_expiry days"
            else
                print_test_result "FAIL" "$test_name" "Expires in $days_until_expiry days"
            fi
        else
            print_test_result "FAIL" "$test_name" "Cannot retrieve certificate"
        fi
    else
        print_test_result "WARN" "$test_name" "OpenSSL not available"
    fi
}

# Test DNS resolution
test_dns_resolution() {
    local test_name="DNS Resolution"
    
    if [[ "$DOMAIN" == "localhost" ]]; then
        print_test_result "INFO" "$test_name" "Skipped (localhost)"
        return 0
    fi
    
    if command -v nslookup >/dev/null 2>&1; then
        if nslookup "$DOMAIN" >/dev/null 2>&1; then
            local ip_address
            ip_address=$(nslookup "$DOMAIN" | grep -A1 "Name:" | tail -n1 | awk '{print $2}' || echo "unknown")
            print_test_result "PASS" "$test_name" "Resolves to $ip_address"
        else
            print_test_result "FAIL" "$test_name" "Cannot resolve domain"
        fi
    else
        print_test_result "WARN" "$test_name" "nslookup not available"
    fi
}

# Test frontend accessibility
test_frontend() {
    local test_name="Frontend Service"
    
    local result
    result=$(http_request "$FRONTEND_URL" "GET" "200")
    
    local status
    status=$(echo "$result" | cut -d'|' -f1)
    local http_code
    http_code=$(echo "$result" | cut -d'|' -f2)
    local response_time
    response_time=$(echo "$result" | cut -d'|' -f3)
    
    case $status in
        "SUCCESS")
            print_test_result "PASS" "$test_name" "Accessible (${response_time}s)"
            ;;
        "HTTP_ERROR")
            print_test_result "FAIL" "$test_name" "HTTP $http_code" "Expected 200"
            ;;
        "CONNECTION_ERROR")
            print_test_result "FAIL" "$test_name" "Connection failed"
            ;;
    esac
}

# Test backend API
test_backend_api() {
    local test_name="Backend API"
    
    # Test health endpoint
    local result
    result=$(http_request "$BACKEND_URL/health" "GET" "200")
    
    local status
    status=$(echo "$result" | cut -d'|' -f1)
    local http_code
    http_code=$(echo "$result" | cut -d'|' -f2)
    local response_time
    response_time=$(echo "$result" | cut -d'|' -f3)
    
    case $status in
        "SUCCESS")
            print_test_result "PASS" "$test_name" "Health check OK (${response_time}s)"
            
            # Test additional endpoints
            test_api_endpoints
            ;;
        "HTTP_ERROR")
            print_test_result "FAIL" "$test_name" "HTTP $http_code" "Expected 200"
            ;;
        "CONNECTION_ERROR")
            print_test_result "FAIL" "$test_name" "Connection failed"
            ;;
    esac
}

# Test API endpoints
test_api_endpoints() {
    local endpoints=(
        "/api/v1/stocks/list:GET:200"
        "/api/v1/predictions/status:GET:200"
        "/api/v1/health/detailed:GET:200"
        "/api/v1/metrics:GET:200"
    )
    
    for endpoint_config in "${endpoints[@]}"; do
        local endpoint
        endpoint=$(echo "$endpoint_config" | cut -d':' -f1)
        local method
        method=$(echo "$endpoint_config" | cut -d':' -f2)
        local expected_status
        expected_status=$(echo "$endpoint_config" | cut -d':' -f3)
        
        local test_name="API Endpoint $endpoint"
        local full_url="https://$DOMAIN$endpoint"
        
        local result
        result=$(http_request "$full_url" "$method" "$expected_status")
        
        local status
        status=$(echo "$result" | cut -d'|' -f1)
        local http_code
        http_code=$(echo "$result" | cut -d'|' -f2)
        local response_time
        response_time=$(echo "$result" | cut -d'|' -f3)
        
        case $status in
            "SUCCESS")
                print_test_result "PASS" "$test_name" "$method $http_code (${response_time}s)"
                ;;
            "HTTP_ERROR")
                print_test_result "WARN" "$test_name" "$method $http_code" "Expected $expected_status"
                ;;
            "CONNECTION_ERROR")
                print_test_result "FAIL" "$test_name" "Connection failed"
                ;;
        esac
    done
}

# Test database connectivity
test_database() {
    local test_name="PostgreSQL Database"
    
    if command -v pg_isready >/dev/null 2>&1; then
        if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -d "$POSTGRES_DB" -U "$POSTGRES_USER" >/dev/null 2>&1; then
            print_test_result "PASS" "$test_name" "Connection successful"
            
            # Test database operations
            test_database_operations
        else
            print_test_result "FAIL" "$test_name" "Connection failed"
        fi
    else
        print_test_result "WARN" "$test_name" "pg_isready not available"
    fi
}

# Test database operations
test_database_operations() {
    if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
        print_test_result "WARN" "Database Operations" "Password not available"
        return
    fi
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Test basic query
    local test_name="Database Query"
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -d "$POSTGRES_DB" -U "$POSTGRES_USER" \
            -c "SELECT 1;" >/dev/null 2>&1; then
        print_test_result "PASS" "$test_name" "Basic query successful"
    else
        print_test_result "FAIL" "$test_name" "Basic query failed"
    fi
    
    # Test table existence
    local test_name="Database Tables"
    local table_count
    table_count=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -d "$POSTGRES_DB" -U "$POSTGRES_USER" \
                      -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | xargs || echo "0")
    
    if [[ "$table_count" -gt 0 ]]; then
        print_test_result "PASS" "$test_name" "$table_count tables found"
    else
        print_test_result "WARN" "$test_name" "No tables found"
    fi
    
    unset PGPASSWORD
}

# Test Redis connectivity
test_redis() {
    local test_name="Redis Cache"
    
    if command -v redis-cli >/dev/null 2>&1; then
        local redis_cmd="redis-cli -h $REDIS_HOST -p $REDIS_PORT"
        
        if [[ -n "${REDIS_PASSWORD:-}" ]]; then
            redis_cmd="$redis_cmd -a $REDIS_PASSWORD"
        fi
        
        if $redis_cmd ping >/dev/null 2>&1; then
            local memory_usage
            memory_usage=$($redis_cmd info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r' || echo "unknown")
            print_test_result "PASS" "$test_name" "Connection successful (Memory: $memory_usage)"
            
            # Test Redis operations
            test_redis_operations
        else
            print_test_result "FAIL" "$test_name" "Connection failed"
        fi
    else
        print_test_result "WARN" "$test_name" "redis-cli not available"
    fi
}

# Test Redis operations
test_redis_operations() {
    local redis_cmd="redis-cli -h $REDIS_HOST -p $REDIS_PORT"
    
    if [[ -n "${REDIS_PASSWORD:-}" ]]; then
        redis_cmd="$redis_cmd -a $REDIS_PASSWORD"
    fi
    
    # Test set/get operations
    local test_name="Redis Operations"
    local test_key="validation_test_$(date +%s)"
    local test_value="test_value_$(date +%s)"
    
    if $redis_cmd set "$test_key" "$test_value" >/dev/null 2>&1; then
        local retrieved_value
        retrieved_value=$($redis_cmd get "$test_key" 2>/dev/null || echo "")
        
        if [[ "$retrieved_value" == "$test_value" ]]; then
            print_test_result "PASS" "$test_name" "Set/Get operations successful"
            # Cleanup
            $redis_cmd del "$test_key" >/dev/null 2>&1 || true
        else
            print_test_result "FAIL" "$test_name" "Set/Get operations failed"
        fi
    else
        print_test_result "FAIL" "$test_name" "Set operation failed"
    fi
}

# Test monitoring services
test_monitoring() {
    # Test Prometheus
    local test_name="Prometheus"
    local result
    result=$(http_request "$PROMETHEUS_URL/-/healthy" "GET" "200")
    
    local status
    status=$(echo "$result" | cut -d'|' -f1)
    
    case $status in
        "SUCCESS")
            print_test_result "PASS" "$test_name" "Service healthy"
            ;;
        *)
            print_test_result "FAIL" "$test_name" "Service unhealthy"
            ;;
    esac
    
    # Test Grafana
    local test_name="Grafana"
    local result
    result=$(http_request "$GRAFANA_URL/api/health" "GET" "200")
    
    local status
    status=$(echo "$result" | cut -d'|' -f1)
    
    case $status in
        "SUCCESS")
            print_test_result "PASS" "$test_name" "Service healthy"
            ;;
        *)
            print_test_result "FAIL" "$test_name" "Service unhealthy"
            ;;
    esac
}

# Test Docker containers
test_docker_containers() {
    local test_name="Docker Containers"
    
    if command -v docker >/dev/null 2>&1; then
        local running_containers
        running_containers=$(docker ps --format "{{.Names}}" 2>/dev/null || echo "")
        
        if [[ -n "$running_containers" ]]; then
            local container_count
            container_count=$(echo "$running_containers" | wc -l)
            print_test_result "PASS" "$test_name" "$container_count containers running"
            
            # Check for unhealthy containers
            local unhealthy_containers
            unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" 2>/dev/null || echo "")
            
            if [[ -n "$unhealthy_containers" ]]; then
                print_test_result "FAIL" "Container Health" "Unhealthy: $unhealthy_containers"
            else
                print_test_result "PASS" "Container Health" "All containers healthy"
            fi
        else
            print_test_result "FAIL" "$test_name" "No containers running"
        fi
    else
        print_test_result "WARN" "$test_name" "Docker not available"
    fi
}

# Test system resources
test_system_resources() {
    local test_name="System Resources"
    
    # Check disk space
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    
    if [[ "$disk_usage" -lt 80 ]]; then
        print_test_result "PASS" "Disk Usage" "${disk_usage}% used"
    elif [[ "$disk_usage" -lt 90 ]]; then
        print_test_result "WARN" "Disk Usage" "${disk_usage}% used"
    else
        print_test_result "FAIL" "Disk Usage" "${disk_usage}% used (critical)"
    fi
    
    # Check memory
    if command -v free >/dev/null 2>&1; then
        local memory_usage
        memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
        
        if [[ "$memory_usage" -lt 80 ]]; then
            print_test_result "PASS" "Memory Usage" "${memory_usage}% used"
        elif [[ "$memory_usage" -lt 90 ]]; then
            print_test_result "WARN" "Memory Usage" "${memory_usage}% used"
        else
            print_test_result "FAIL" "Memory Usage" "${memory_usage}% used (critical)"
        fi
    fi
    
    # Check load average
    if [[ -f /proc/loadavg ]]; then
        local load_avg
        load_avg=$(cat /proc/loadavg | cut -d' ' -f1)
        local cpu_count
        cpu_count=$(nproc 2>/dev/null || echo "1")
        local load_percentage
        load_percentage=$(echo "$load_avg * 100 / $cpu_count" | bc -l 2>/dev/null | cut -d'.' -f1 || echo "0")
        
        if [[ "$load_percentage" -lt 70 ]]; then
            print_test_result "PASS" "System Load" "${load_percentage}% (${load_avg})"
        elif [[ "$load_percentage" -lt 90 ]]; then
            print_test_result "WARN" "System Load" "${load_percentage}% (${load_avg})"
        else
            print_test_result "FAIL" "System Load" "${load_percentage}% (${load_avg}) (critical)"
        fi
    fi
}

# Test network connectivity
test_network_connectivity() {
    local test_name="Network Connectivity"
    
    # Test external connectivity
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        print_test_result "PASS" "External Network" "Internet connectivity OK"
    else
        print_test_result "FAIL" "External Network" "No internet connectivity"
    fi
    
    # Test internal services
    local internal_services=(
        "$POSTGRES_HOST:$POSTGRES_PORT"
        "$REDIS_HOST:$REDIS_PORT"
    )
    
    for service in "${internal_services[@]}"; do
        local host
        host=$(echo "$service" | cut -d':' -f1)
        local port
        port=$(echo "$service" | cut -d':' -f2)
        
        if command -v nc >/dev/null 2>&1; then
            if nc -z "$host" "$port" 2>/dev/null; then
                print_test_result "PASS" "Internal Network" "$service reachable"
            else
                print_test_result "FAIL" "Internal Network" "$service unreachable"
            fi
        elif command -v telnet >/dev/null 2>&1; then
            if timeout 5 telnet "$host" "$port" </dev/null >/dev/null 2>&1; then
                print_test_result "PASS" "Internal Network" "$service reachable"
            else
                print_test_result "FAIL" "Internal Network" "$service unreachable"
            fi
        else
            print_test_result "WARN" "Internal Network" "No network testing tools available"
            break
        fi
    done
}

# Test log files
test_log_files() {
    local test_name="Log Files"
    local log_dir="/var/log/stock-prediction"
    
    if [[ -d "$log_dir" ]]; then
        local log_files
        log_files=$(find "$log_dir" -name "*.log" -type f | wc -l)
        
        if [[ "$log_files" -gt 0 ]]; then
            print_test_result "PASS" "$test_name" "$log_files log files found"
            
            # Check for recent errors
            local recent_errors
            recent_errors=$(find "$log_dir" -name "*.log" -type f -exec grep -l "ERROR\|CRITICAL\|FATAL" {} \; 2>/dev/null | wc -l)
            
            if [[ "$recent_errors" -eq 0 ]]; then
                print_test_result "PASS" "Log Errors" "No recent errors found"
            else
                print_test_result "WARN" "Log Errors" "$recent_errors files with errors"
            fi
        else
            print_test_result "WARN" "$test_name" "No log files found"
        fi
    else
        print_test_result "WARN" "$test_name" "Log directory not found"
    fi
}

# Generate test report
generate_report() {
    local report_file="/var/log/stock-prediction/validation-report-$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" <<EOF
{
    "validation_report": {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "hostname": "$(hostname)",
        "domain": "$DOMAIN",
        "summary": {
            "total_tests": $TOTAL_TESTS,
            "passed": $PASSED_TESTS,
            "failed": $FAILED_TESTS,
            "warnings": $WARNING_TESTS,
            "success_rate": "$(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")%"
        },
        "environment": {
            "backend_url": "$BACKEND_URL",
            "frontend_url": "$FRONTEND_URL",
            "prometheus_url": "$PROMETHEUS_URL",
            "grafana_url": "$GRAFANA_URL",
            "postgres_host": "$POSTGRES_HOST",
            "redis_host": "$REDIS_HOST"
        }
    }
}
EOF
    
    echo "Detailed report saved to: $report_file"
}

# Main validation function
main() {
    local start_time
    start_time=$(date +%s)
    
    echo "=========================================="
    echo -e "${PURPLE}Stock Prediction System Validation${NC}"
    echo "Time: $(date)"
    echo "Host: $(hostname)"
    echo "Domain: $DOMAIN"
    echo "=========================================="
    
    log "INFO" "Starting deployment validation"
    
    # Infrastructure tests
    echo -e "\n${CYAN}=== Infrastructure Tests ===${NC}"
    test_dns_resolution
    test_ssl_certificate
    test_network_connectivity
    test_system_resources
    test_docker_containers
    
    # Service tests
    echo -e "\n${CYAN}=== Service Tests ===${NC}"
    test_frontend
    test_backend_api
    test_database
    test_redis
    test_monitoring
    
    # System tests
    echo -e "\n${CYAN}=== System Tests ===${NC}"
    test_log_files
    
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$((end_time - start_time))
    
    # Generate report
    generate_report
    
    # Summary
    echo -e "\n=========================================="
    echo -e "${PURPLE}Validation Summary${NC}"
    echo "=========================================="
    echo "Total Tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    echo -e "${YELLOW}Warnings: $WARNING_TESTS${NC}"
    
    local success_rate
    success_rate=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
    echo "Success Rate: ${success_rate}%"
    echo "Duration: ${duration}s"
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        echo -e "\n${GREEN}✓ Deployment validation completed successfully!${NC}"
        log "INFO" "Deployment validation completed successfully"
        exit 0
    else
        echo -e "\n${RED}✗ Deployment validation completed with failures!${NC}"
        log "ERROR" "Deployment validation completed with $FAILED_TESTS failures"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    "--help"|-h)
        echo "Usage: $0 [--help|--version|--quick]"
        echo "  --help     Show this help message"
        echo "  --version  Show version information"
        echo "  --quick    Run quick validation (skip detailed tests)"
        exit 0
        ;;
    "--version"|-v)
        echo "Stock Prediction System Validation v1.0.0"
        exit 0
        ;;
    "--quick")
        TIMEOUT=10
        MAX_RETRIES=2
        echo "Running quick validation..."
        ;;
esac

# Run main function
main "$@"