#!/bin/bash

# Performance Test Runner for Stock Prediction System
# This script runs comprehensive performance tests including load, stress, and WebSocket tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/test-reports/performance"
LOGS_DIR="$PROJECT_ROOT/logs/performance-tests"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_SESSION_ID="perf_test_${TIMESTAMP}"

# Default configuration
BASE_URL="${BASE_URL:-http://localhost:8000}"
WS_URL="${WS_URL:-ws://localhost:8000/ws}"
TEST_DURATION="${TEST_DURATION:-15m}"
MAX_VUS="${MAX_VUS:-200}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
EMAIL_RECIPIENT="${EMAIL_RECIPIENT:-}"
RUN_LOAD_TEST="${RUN_LOAD_TEST:-true}"
RUN_STRESS_TEST="${RUN_STRESS_TEST:-true}"
RUN_WEBSOCKET_TEST="${RUN_WEBSOCKET_TEST:-true}"
GENERATE_REPORT="${GENERATE_REPORT:-true}"
CLEANUP_OLD_REPORTS="${CLEANUP_OLD_REPORTS:-true}"

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${BLUE}[$timestamp] [INFO]${NC} $message" | tee -a "$LOGS_DIR/test-runner.log"
            ;;
        "WARN")
            echo -e "${YELLOW}[$timestamp] [WARN]${NC} $message" | tee -a "$LOGS_DIR/test-runner.log"
            ;;
        "ERROR")
            echo -e "${RED}[$timestamp] [ERROR]${NC} $message" | tee -a "$LOGS_DIR/test-runner.log"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[$timestamp] [SUCCESS]${NC} $message" | tee -a "$LOGS_DIR/test-runner.log"
            ;;
        "DEBUG")
            if [[ "${DEBUG:-false}" == "true" ]]; then
                echo -e "${PURPLE}[$timestamp] [DEBUG]${NC} $message" | tee -a "$LOGS_DIR/test-runner.log"
            fi
            ;;
    esac
}

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "                    STOCK PREDICTION SYSTEM - PERFORMANCE TESTS"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    echo "Test Session ID: $TEST_SESSION_ID"
    echo "Base URL: $BASE_URL"
    echo "WebSocket URL: $WS_URL"
    echo "Max VUs: $MAX_VUS"
    echo "Test Duration: $TEST_DURATION"
    echo "Reports Directory: $REPORTS_DIR"
    echo "Logs Directory: $LOGS_DIR"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    # Check if k6 is installed
    if ! command -v k6 &> /dev/null; then
        log "ERROR" "k6 is not installed. Please install k6 from https://k6.io/docs/getting-started/installation/"
        exit 1
    fi
    
    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        log "ERROR" "curl is not installed. Please install curl."
        exit 1
    fi
    
    # Check if jq is available for JSON processing
    if ! command -v jq &> /dev/null; then
        log "WARN" "jq is not installed. JSON processing will be limited."
    fi
    
    # Create necessary directories
    mkdir -p "$REPORTS_DIR" "$LOGS_DIR"
    
    log "SUCCESS" "Prerequisites check completed"
}

# Check system health
check_system_health() {
    log "INFO" "Checking system health before tests..."
    
    # Check if the application is running
    if ! curl -s -f "$BASE_URL/health" > /dev/null; then
        log "ERROR" "Application health check failed. Please ensure the application is running at $BASE_URL"
        exit 1
    fi
    
    # Check WebSocket endpoint
    if [[ "$RUN_WEBSOCKET_TEST" == "true" ]]; then
        # Simple WebSocket connectivity test
        timeout 10s curl -s -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Key: test" -H "Sec-WebSocket-Version: 13" "${WS_URL/ws:/http:}" > /dev/null 2>&1 || {
            log "WARN" "WebSocket endpoint check failed. WebSocket tests may not work properly."
        }
    fi
    
    # Check system resources
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}' || echo "unknown")
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}' || echo "unknown")
    local disk_usage=$(df -h . | awk 'NR==2{print $5}' | sed 's/%//' || echo "unknown")
    
    log "INFO" "System Resources - CPU: ${cpu_usage}%, Memory: ${memory_usage}%, Disk: ${disk_usage}%"
    
    # Warn if system resources are high
    if [[ "$cpu_usage" != "unknown" ]] && (( $(echo "$cpu_usage > 80" | bc -l 2>/dev/null || echo 0) )); then
        log "WARN" "High CPU usage detected (${cpu_usage}%). This may affect test results."
    fi
    
    if [[ "$memory_usage" != "unknown" ]] && (( $(echo "$memory_usage > 80" | bc -l 2>/dev/null || echo 0) )); then
        log "WARN" "High memory usage detected (${memory_usage}%). This may affect test results."
    fi
    
    log "SUCCESS" "System health check completed"
}

# Run load test
run_load_test() {
    if [[ "$RUN_LOAD_TEST" != "true" ]]; then
        log "INFO" "Skipping load test (disabled)"
        return 0
    fi
    
    log "INFO" "Starting load test..."
    
    local test_file="$SCRIPT_DIR/load-test.js"
    local report_file="$REPORTS_DIR/load-test-${TIMESTAMP}"
    
    if [[ ! -f "$test_file" ]]; then
        log "ERROR" "Load test file not found: $test_file"
        return 1
    fi
    
    # Run k6 load test
    k6 run \
        --out json="${report_file}.json" \
        --out csv="${report_file}.csv" \
        -e BASE_URL="$BASE_URL" \
        -e TEST_DURATION="$TEST_DURATION" \
        -e MAX_VUS="$MAX_VUS" \
        -e TEST_SESSION_ID="$TEST_SESSION_ID" \
        "$test_file" 2>&1 | tee "$LOGS_DIR/load-test-${TIMESTAMP}.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        log "SUCCESS" "Load test completed successfully"
        # Move generated HTML report if it exists
        [[ -f "load-test-report.html" ]] && mv "load-test-report.html" "${report_file}.html"
        [[ -f "load-test-summary.json" ]] && mv "load-test-summary.json" "${report_file}-summary.json"
    else
        log "ERROR" "Load test failed with exit code: $exit_code"
        return 1
    fi
}

# Run stress test
run_stress_test() {
    if [[ "$RUN_STRESS_TEST" != "true" ]]; then
        log "INFO" "Skipping stress test (disabled)"
        return 0
    fi
    
    log "INFO" "Starting stress test..."
    
    local test_file="$SCRIPT_DIR/stress-test.js"
    local report_file="$REPORTS_DIR/stress-test-${TIMESTAMP}"
    
    if [[ ! -f "$test_file" ]]; then
        log "ERROR" "Stress test file not found: $test_file"
        return 1
    fi
    
    # Run k6 stress test
    k6 run \
        --out json="${report_file}.json" \
        --out csv="${report_file}.csv" \
        -e BASE_URL="$BASE_URL" \
        -e MAX_VUS="$((MAX_VUS * 2))" \
        -e TEST_SESSION_ID="$TEST_SESSION_ID" \
        "$test_file" 2>&1 | tee "$LOGS_DIR/stress-test-${TIMESTAMP}.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        log "SUCCESS" "Stress test completed successfully"
        # Move generated HTML report if it exists
        [[ -f "stress-test-report.html" ]] && mv "stress-test-report.html" "${report_file}.html"
        [[ -f "stress-test-summary.json" ]] && mv "stress-test-summary.json" "${report_file}-summary.json"
    else
        log "ERROR" "Stress test failed with exit code: $exit_code"
        return 1
    fi
}

# Run WebSocket stress test
run_websocket_test() {
    if [[ "$RUN_WEBSOCKET_TEST" != "true" ]]; then
        log "INFO" "Skipping WebSocket test (disabled)"
        return 0
    fi
    
    log "INFO" "Starting WebSocket stress test..."
    
    local test_file="$SCRIPT_DIR/websocket-stress-test.js"
    local report_file="$REPORTS_DIR/websocket-test-${TIMESTAMP}"
    
    if [[ ! -f "$test_file" ]]; then
        log "ERROR" "WebSocket test file not found: $test_file"
        return 1
    fi
    
    # Run k6 WebSocket stress test
    k6 run \
        --out json="${report_file}.json" \
        --out csv="${report_file}.csv" \
        -e WS_URL="$WS_URL" \
        -e BASE_URL="$BASE_URL" \
        -e MAX_VUS="$MAX_VUS" \
        -e TEST_SESSION_ID="$TEST_SESSION_ID" \
        "$test_file" 2>&1 | tee "$LOGS_DIR/websocket-test-${TIMESTAMP}.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        log "SUCCESS" "WebSocket stress test completed successfully"
        # Move generated HTML report if it exists
        [[ -f "websocket-stress-test-report.html" ]] && mv "websocket-stress-test-report.html" "${report_file}.html"
        [[ -f "websocket-stress-test-summary.json" ]] && mv "websocket-stress-test-summary.json" "${report_file}-summary.json"
    else
        log "ERROR" "WebSocket stress test failed with exit code: $exit_code"
        return 1
    fi
}

# Generate consolidated report
generate_consolidated_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        log "INFO" "Skipping report generation (disabled)"
        return 0
    fi
    
    log "INFO" "Generating consolidated performance report..."
    
    local consolidated_report="$REPORTS_DIR/consolidated-report-${TIMESTAMP}.html"
    local summary_json="$REPORTS_DIR/test-summary-${TIMESTAMP}.json"
    
    # Create consolidated HTML report
    cat > "$consolidated_report" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Prediction System - Performance Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
        .test-section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .test-section h2 { color: #007bff; margin-top: 0; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
        .status-pass { color: #28a745; }
        .status-fail { color: #dc3545; }
        .status-warn { color: #ffc107; }
        .summary { background: #e9ecef; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .file-links { margin: 15px 0; }
        .file-links a { display: inline-block; margin: 5px 10px 5px 0; padding: 8px 15px; background: #007bff; color: white; text-decoration: none; border-radius: 3px; }
        .file-links a:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Stock Prediction System - Performance Test Report</h1>
            <p>Test Session: $TEST_SESSION_ID</p>
            <p>Generated: $(date)</p>
            <p>Base URL: $BASE_URL | WebSocket URL: $WS_URL</p>
        </div>
EOF
    
    # Add test results sections
    local overall_status="PASS"
    local test_count=0
    local passed_tests=0
    
    # Process load test results
    if [[ "$RUN_LOAD_TEST" == "true" ]] && [[ -f "$REPORTS_DIR/load-test-${TIMESTAMP}.json" ]]; then
        test_count=$((test_count + 1))
        echo "        <div class='test-section'>" >> "$consolidated_report"
        echo "            <h2>Load Test Results</h2>" >> "$consolidated_report"
        
        # Extract key metrics from JSON if jq is available
        if command -v jq &> /dev/null; then
            local avg_response_time=$(jq -r '.metrics.http_req_duration.avg // "N/A"' "$REPORTS_DIR/load-test-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
            local p95_response_time=$(jq -r '.metrics.http_req_duration."p(95)" // "N/A"' "$REPORTS_DIR/load-test-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
            local error_rate=$(jq -r '.metrics.http_req_failed.rate // "N/A"' "$REPORTS_DIR/load-test-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
            local total_requests=$(jq -r '.metrics.http_reqs.count // "N/A"' "$REPORTS_DIR/load-test-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
            
            echo "            <div class='metrics'>" >> "$consolidated_report"
            echo "                <div class='metric'><div class='metric-value'>$avg_response_time ms</div><div class='metric-label'>Avg Response Time</div></div>" >> "$consolidated_report"
            echo "                <div class='metric'><div class='metric-value'>$p95_response_time ms</div><div class='metric-label'>95th Percentile</div></div>" >> "$consolidated_report"
            echo "                <div class='metric'><div class='metric-value'>$error_rate%</div><div class='metric-label'>Error Rate</div></div>" >> "$consolidated_report"
            echo "                <div class='metric'><div class='metric-value'>$total_requests</div><div class='metric-label'>Total Requests</div></div>" >> "$consolidated_report"
            echo "            </div>" >> "$consolidated_report"
            
            # Determine if load test passed (error rate < 5%, p95 < 2000ms)
            if [[ "$error_rate" != "N/A" ]] && (( $(echo "$error_rate < 0.05" | bc -l 2>/dev/null || echo 0) )) && \
               [[ "$p95_response_time" != "N/A" ]] && (( $(echo "$p95_response_time < 2000" | bc -l 2>/dev/null || echo 1) )); then
                passed_tests=$((passed_tests + 1))
                echo "            <p class='status-pass'>✓ Load test PASSED</p>" >> "$consolidated_report"
            else
                overall_status="FAIL"
                echo "            <p class='status-fail'>✗ Load test FAILED</p>" >> "$consolidated_report"
            fi
        else
            echo "            <p>Detailed metrics require jq for JSON processing</p>" >> "$consolidated_report"
        fi
        
        echo "            <div class='file-links'>" >> "$consolidated_report"
        [[ -f "$REPORTS_DIR/load-test-${TIMESTAMP}.html" ]] && echo "                <a href='load-test-${TIMESTAMP}.html'>Detailed HTML Report</a>" >> "$consolidated_report"
        [[ -f "$REPORTS_DIR/load-test-${TIMESTAMP}.json" ]] && echo "                <a href='load-test-${TIMESTAMP}.json'>JSON Data</a>" >> "$consolidated_report"
        [[ -f "$REPORTS_DIR/load-test-${TIMESTAMP}.csv" ]] && echo "                <a href='load-test-${TIMESTAMP}.csv'>CSV Data</a>" >> "$consolidated_report"
        echo "            </div>" >> "$consolidated_report"
        echo "        </div>" >> "$consolidated_report"
    fi
    
    # Process stress test results (similar pattern)
    if [[ "$RUN_STRESS_TEST" == "true" ]] && [[ -f "$REPORTS_DIR/stress-test-${TIMESTAMP}.json" ]]; then
        test_count=$((test_count + 1))
        echo "        <div class='test-section'>" >> "$consolidated_report"
        echo "            <h2>Stress Test Results</h2>" >> "$consolidated_report"
        echo "            <p>Stress test evaluates system behavior under extreme load conditions.</p>" >> "$consolidated_report"
        
        if command -v jq &> /dev/null; then
            local max_response_time=$(jq -r '.metrics.http_req_duration.max // "N/A"' "$REPORTS_DIR/stress-test-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
            local stress_error_rate=$(jq -r '.metrics.http_req_failed.rate // "N/A"' "$REPORTS_DIR/stress-test-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
            
            echo "            <div class='metrics'>" >> "$consolidated_report"
            echo "                <div class='metric'><div class='metric-value'>$max_response_time ms</div><div class='metric-label'>Max Response Time</div></div>" >> "$consolidated_report"
            echo "                <div class='metric'><div class='metric-value'>$stress_error_rate%</div><div class='metric-label'>Error Rate Under Stress</div></div>" >> "$consolidated_report"
            echo "            </div>" >> "$consolidated_report"
            
            # Stress test passes if error rate < 15% and max response time < 10s
            if [[ "$stress_error_rate" != "N/A" ]] && (( $(echo "$stress_error_rate < 0.15" | bc -l 2>/dev/null || echo 0) )) && \
               [[ "$max_response_time" != "N/A" ]] && (( $(echo "$max_response_time < 10000" | bc -l 2>/dev/null || echo 1) )); then
                passed_tests=$((passed_tests + 1))
                echo "            <p class='status-pass'>✓ Stress test PASSED</p>" >> "$consolidated_report"
            else
                overall_status="FAIL"
                echo "            <p class='status-fail'>✗ Stress test FAILED</p>" >> "$consolidated_report"
            fi
        fi
        
        echo "            <div class='file-links'>" >> "$consolidated_report"
        [[ -f "$REPORTS_DIR/stress-test-${TIMESTAMP}.html" ]] && echo "                <a href='stress-test-${TIMESTAMP}.html'>Detailed HTML Report</a>" >> "$consolidated_report"
        [[ -f "$REPORTS_DIR/stress-test-${TIMESTAMP}.json" ]] && echo "                <a href='stress-test-${TIMESTAMP}.json'>JSON Data</a>" >> "$consolidated_report"
        echo "            </div>" >> "$consolidated_report"
        echo "        </div>" >> "$consolidated_report"
    fi
    
    # Process WebSocket test results
    if [[ "$RUN_WEBSOCKET_TEST" == "true" ]] && [[ -f "$REPORTS_DIR/websocket-test-${TIMESTAMP}.json" ]]; then
        test_count=$((test_count + 1))
        echo "        <div class='test-section'>" >> "$consolidated_report"
        echo "            <h2>WebSocket Stress Test Results</h2>" >> "$consolidated_report"
        echo "            <p>WebSocket test evaluates real-time communication performance under load.</p>" >> "$consolidated_report"
        
        if command -v jq &> /dev/null; then
            local ws_connection_rate=$(jq -r '.metrics.ws_connection_rate.rate // "N/A"' "$REPORTS_DIR/websocket-test-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
            local ws_message_rate=$(jq -r '.metrics.ws_message_rate.rate // "N/A"' "$REPORTS_DIR/websocket-test-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
            local ws_avg_latency=$(jq -r '.metrics.ws_latency.avg // "N/A"' "$REPORTS_DIR/websocket-test-${TIMESTAMP}.json" 2>/dev/null || echo "N/A")
            
            echo "            <div class='metrics'>" >> "$consolidated_report"
            echo "                <div class='metric'><div class='metric-value'>$ws_connection_rate%</div><div class='metric-label'>Connection Success Rate</div></div>" >> "$consolidated_report"
            echo "                <div class='metric'><div class='metric-value'>$ws_message_rate%</div><div class='metric-label'>Message Success Rate</div></div>" >> "$consolidated_report"
            echo "                <div class='metric'><div class='metric-value'>$ws_avg_latency ms</div><div class='metric-label'>Avg Message Latency</div></div>" >> "$consolidated_report"
            echo "            </div>" >> "$consolidated_report"
            
            # WebSocket test passes if connection rate > 95% and message rate > 98%
            if [[ "$ws_connection_rate" != "N/A" ]] && (( $(echo "$ws_connection_rate > 0.95" | bc -l 2>/dev/null || echo 0) )) && \
               [[ "$ws_message_rate" != "N/A" ]] && (( $(echo "$ws_message_rate > 0.98" | bc -l 2>/dev/null || echo 0) )); then
                passed_tests=$((passed_tests + 1))
                echo "            <p class='status-pass'>✓ WebSocket test PASSED</p>" >> "$consolidated_report"
            else
                overall_status="FAIL"
                echo "            <p class='status-fail'>✗ WebSocket test FAILED</p>" >> "$consolidated_report"
            fi
        fi
        
        echo "            <div class='file-links'>" >> "$consolidated_report"
        [[ -f "$REPORTS_DIR/websocket-test-${TIMESTAMP}.html" ]] && echo "                <a href='websocket-test-${TIMESTAMP}.html'>Detailed HTML Report</a>" >> "$consolidated_report"
        [[ -f "$REPORTS_DIR/websocket-test-${TIMESTAMP}.json" ]] && echo "                <a href='websocket-test-${TIMESTAMP}.json'>JSON Data</a>" >> "$consolidated_report"
        echo "            </div>" >> "$consolidated_report"
        echo "        </div>" >> "$consolidated_report"
    fi
    
    # Add summary section
    local status_class="status-pass"
    [[ "$overall_status" == "FAIL" ]] && status_class="status-fail"
    
    cat >> "$consolidated_report" << EOF
        <div class="summary">
            <h2>Test Summary</h2>
            <p class="$status_class">Overall Status: $overall_status</p>
            <p>Tests Passed: $passed_tests / $test_count</p>
            <p>Test Session ID: $TEST_SESSION_ID</p>
            <p>Test Duration: $TEST_DURATION</p>
            <p>Maximum VUs: $MAX_VUS</p>
            <p>Generated: $(date)</p>
        </div>
    </div>
</body>
</html>
EOF
    
    # Create JSON summary
    cat > "$summary_json" << EOF
{
    "test_session_id": "$TEST_SESSION_ID",
    "timestamp": "$(date -Iseconds)",
    "base_url": "$BASE_URL",
    "websocket_url": "$WS_URL",
    "test_duration": "$TEST_DURATION",
    "max_vus": $MAX_VUS,
    "overall_status": "$overall_status",
    "tests_run": $test_count,
    "tests_passed": $passed_tests,
    "tests_enabled": {
        "load_test": $([[ "$RUN_LOAD_TEST" == "true" ]] && echo "true" || echo "false"),
        "stress_test": $([[ "$RUN_STRESS_TEST" == "true" ]] && echo "true" || echo "false"),
        "websocket_test": $([[ "$RUN_WEBSOCKET_TEST" == "true" ]] && echo "true" || echo "false")
    },
    "reports_directory": "$REPORTS_DIR",
    "logs_directory": "$LOGS_DIR"
}
EOF
    
    log "SUCCESS" "Consolidated report generated: $consolidated_report"
    log "INFO" "Summary JSON: $summary_json"
}

# Send notifications
send_notifications() {
    local overall_status="$1"
    local test_summary="$2"
    
    # Send Slack notification if webhook is configured
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        log "INFO" "Sending Slack notification..."
        
        local color="good"
        [[ "$overall_status" == "FAIL" ]] && color="danger"
        
        local payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "Stock Prediction System - Performance Test Results",
            "fields": [
                {
                    "title": "Status",
                    "value": "$overall_status",
                    "short": true
                },
                {
                    "title": "Test Session",
                    "value": "$TEST_SESSION_ID",
                    "short": true
                },
                {
                    "title": "Summary",
                    "value": "$test_summary",
                    "short": false
                },
                {
                    "title": "Base URL",
                    "value": "$BASE_URL",
                    "short": true
                },
                {
                    "title": "Duration",
                    "value": "$TEST_DURATION",
                    "short": true
                }
            ],
            "footer": "Performance Test Runner",
            "ts": $(date +%s)
        }
    ]
}
EOF
        )
        
        if curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK" > /dev/null; then
            log "SUCCESS" "Slack notification sent"
        else
            log "ERROR" "Failed to send Slack notification"
        fi
    fi
    
    # Send email notification if recipient is configured
    if [[ -n "$EMAIL_RECIPIENT" ]] && command -v mail &> /dev/null; then
        log "INFO" "Sending email notification..."
        
        local subject="Stock Prediction System - Performance Test $overall_status"
        local body="Performance test completed with status: $overall_status\n\nTest Session: $TEST_SESSION_ID\nSummary: $test_summary\nBase URL: $BASE_URL\nDuration: $TEST_DURATION\n\nDetailed reports are available in: $REPORTS_DIR"
        
        if echo -e "$body" | mail -s "$subject" "$EMAIL_RECIPIENT"; then
            log "SUCCESS" "Email notification sent to $EMAIL_RECIPIENT"
        else
            log "ERROR" "Failed to send email notification"
        fi
    fi
}

# Cleanup old reports
cleanup_old_reports() {
    if [[ "$CLEANUP_OLD_REPORTS" != "true" ]]; then
        log "INFO" "Skipping cleanup of old reports (disabled)"
        return 0
    fi
    
    log "INFO" "Cleaning up old reports (keeping last 10)..."
    
    # Keep only the 10 most recent report files
    find "$REPORTS_DIR" -name "*-test-*.json" -type f -printf '%T@ %p\n' | sort -rn | tail -n +11 | cut -d' ' -f2- | xargs -r rm -f
    find "$REPORTS_DIR" -name "*-test-*.html" -type f -printf '%T@ %p\n' | sort -rn | tail -n +11 | cut -d' ' -f2- | xargs -r rm -f
    find "$REPORTS_DIR" -name "*-test-*.csv" -type f -printf '%T@ %p\n' | sort -rn | tail -n +11 | cut -d' ' -f2- | xargs -r rm -f
    
    # Keep only the 5 most recent log files
    find "$LOGS_DIR" -name "*-test-*.log" -type f -printf '%T@ %p\n' | sort -rn | tail -n +6 | cut -d' ' -f2- | xargs -r rm -f
    
    log "SUCCESS" "Cleanup completed"
}

# Main execution function
main() {
    local start_time=$(date +%s)
    
    print_banner
    check_prerequisites
    check_system_health
    
    # Run tests
    local failed_tests=0
    
    if ! run_load_test; then
        failed_tests=$((failed_tests + 1))
    fi
    
    if ! run_stress_test; then
        failed_tests=$((failed_tests + 1))
    fi
    
    if ! run_websocket_test; then
        failed_tests=$((failed_tests + 1))
    fi
    
    # Generate reports and notifications
    generate_consolidated_report
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local overall_status="PASS"
    [[ $failed_tests -gt 0 ]] && overall_status="FAIL"
    
    local test_summary="Total duration: ${total_duration}s, Failed tests: $failed_tests"
    
    send_notifications "$overall_status" "$test_summary"
    cleanup_old_reports
    
    # Final summary
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "                              PERFORMANCE TEST SUMMARY"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    echo "Overall Status: $overall_status"
    echo "Total Duration: ${total_duration}s"
    echo "Failed Tests: $failed_tests"
    echo "Reports Directory: $REPORTS_DIR"
    echo "Logs Directory: $LOGS_DIR"
    echo "Test Session ID: $TEST_SESSION_ID"
    echo ""
    
    if [[ $failed_tests -gt 0 ]]; then
        log "ERROR" "Performance tests completed with $failed_tests failures"
        exit 1
    else
        log "SUCCESS" "All performance tests completed successfully"
        exit 0
    fi
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --ws-url)
            WS_URL="$2"
            shift 2
            ;;
        --duration)
            TEST_DURATION="$2"
            shift 2
            ;;
        --max-vus)
            MAX_VUS="$2"
            shift 2
            ;;
        --no-load-test)
            RUN_LOAD_TEST="false"
            shift
            ;;
        --no-stress-test)
            RUN_STRESS_TEST="false"
            shift
            ;;
        --no-websocket-test)
            RUN_WEBSOCKET_TEST="false"
            shift
            ;;
        --no-report)
            GENERATE_REPORT="false"
            shift
            ;;
        --no-cleanup)
            CLEANUP_OLD_REPORTS="false"
            shift
            ;;
        --slack-webhook)
            SLACK_WEBHOOK="$2"
            shift 2
            ;;
        --email)
            EMAIL_RECIPIENT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --base-url URL          Base URL for HTTP tests (default: http://localhost:8000)"
            echo "  --ws-url URL            WebSocket URL (default: ws://localhost:8000/ws)"
            echo "  --duration DURATION     Test duration (default: 15m)"
            echo "  --max-vus NUMBER        Maximum virtual users (default: 200)"
            echo "  --no-load-test          Skip load test"
            echo "  --no-stress-test        Skip stress test"
            echo "  --no-websocket-test     Skip WebSocket test"
            echo "  --no-report             Skip report generation"
            echo "  --no-cleanup            Skip cleanup of old reports"
            echo "  --slack-webhook URL     Slack webhook for notifications"
            echo "  --email EMAIL           Email recipient for notifications"
            echo "  --debug                 Enable debug logging"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"