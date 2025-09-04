#!/bin/bash

# Security Testing Orchestrator for Stock Prediction System
# Runs comprehensive security testing suite and generates consolidated reports

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORTS_DIR="$SCRIPT_DIR/reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_SESSION="security_test_$TIMESTAMP"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging setup
LOG_FILE="$REPORTS_DIR/security_test_$TIMESTAMP.log"
mkdir -p "$REPORTS_DIR"

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${PURPLE}========================================${NC}" | tee -a "$LOG_FILE"
    log "${PURPLE}$1${NC}"
    echo -e "${PURPLE}========================================${NC}" | tee -a "$LOG_FILE"
}

# Configuration variables
TARGET_URL="${TARGET_URL:-http://localhost:8000}"
WS_URL="${WS_URL:-ws://localhost:8000}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
EMAIL_RECIPIENT="${EMAIL_RECIPIENT:-}"
SMTP_SERVER="${SMTP_SERVER:-}"
SMTP_PORT="${SMTP_PORT:-587}"
SMTP_USER="${SMTP_USER:-}"
SMTP_PASS="${SMTP_PASS:-}"

# Test configuration
RUN_DEPENDENCY_SCAN="${RUN_DEPENDENCY_SCAN:-true}"
RUN_STATIC_ANALYSIS="${RUN_STATIC_ANALYSIS:-true}"
RUN_DOCKER_SCAN="${RUN_DOCKER_SCAN:-true}"
RUN_CONFIG_SCAN="${RUN_CONFIG_SCAN:-true}"
RUN_API_SECURITY="${RUN_API_SECURITY:-true}"
RUN_PENETRATION_TEST="${RUN_PENETRATION_TEST:-true}"
RUN_NETWORK_SCAN="${RUN_NETWORK_SCAN:-false}"

# Test results tracking
declare -A TEST_RESULTS
declare -A TEST_REPORTS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
TOTAL_VULNERABILITIES=0

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add cleanup logic here if needed
}

trap cleanup EXIT

# Check prerequisites
check_prerequisites() {
    log_header "CHECKING PREREQUISITES"
    
    local missing_tools=()
    
    # Check required tools
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    command -v pip3 >/dev/null 2>&1 || missing_tools+=("pip3")
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install missing tools and try again"
        exit 1
    fi
    
    # Check Python packages
    local python_packages=("requests" "websocket-client" "pyjwt" "bandit" "safety" "semgrep")
    for package in "${python_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            log_warning "Python package '$package' not found, installing..."
            pip3 install "$package" || log_warning "Failed to install $package"
        fi
    done
    
    # Check if target is accessible
    if ! curl -s --connect-timeout 5 "$TARGET_URL/health" >/dev/null; then
        log_warning "Target URL $TARGET_URL is not accessible"
        log_warning "Some tests may fail or be skipped"
    fi
    
    log_success "Prerequisites check completed"
}

# Run dependency vulnerability scanning
run_dependency_scan() {
    if [ "$RUN_DEPENDENCY_SCAN" != "true" ]; then
        log_info "Dependency scan skipped"
        return 0
    fi
    
    log_header "DEPENDENCY VULNERABILITY SCANNING"
    
    local report_file="$REPORTS_DIR/dependency_scan_$TIMESTAMP.json"
    local exit_code=0
    
    # Run the security scan script
    if [ -f "$SCRIPT_DIR/security-scan.sh" ]; then
        bash "$SCRIPT_DIR/security-scan.sh" --output "$report_file" || exit_code=$?
    else
        log_error "security-scan.sh not found"
        exit_code=1
    fi
    
    TEST_RESULTS["dependency_scan"]=$exit_code
    TEST_REPORTS["dependency_scan"]="$report_file"
    
    if [ $exit_code -eq 0 ]; then
        log_success "Dependency scan completed successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log_error "Dependency scan failed with exit code $exit_code"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

# Run static code analysis
run_static_analysis() {
    if [ "$RUN_STATIC_ANALYSIS" != "true" ]; then
        log_info "Static analysis skipped"
        return 0
    fi
    
    log_header "STATIC CODE ANALYSIS"
    
    local report_file="$REPORTS_DIR/static_analysis_$TIMESTAMP.json"
    local exit_code=0
    
    # Run Bandit for Python security analysis
    log_info "Running Bandit security analysis..."
    if command -v bandit >/dev/null 2>&1; then
        bandit -r "$PROJECT_ROOT/backend" -f json -o "${report_file%.json}_bandit.json" || exit_code=$?
    else
        log_warning "Bandit not available, skipping Python security analysis"
    fi
    
    # Run Semgrep for additional static analysis
    log_info "Running Semgrep analysis..."
    if command -v semgrep >/dev/null 2>&1; then
        semgrep --config=auto --json --output="${report_file%.json}_semgrep.json" "$PROJECT_ROOT" || true
    else
        log_warning "Semgrep not available, skipping advanced static analysis"
    fi
    
    TEST_RESULTS["static_analysis"]=$exit_code
    TEST_REPORTS["static_analysis"]="$report_file"
    
    if [ $exit_code -eq 0 ]; then
        log_success "Static analysis completed successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log_error "Static analysis failed with exit code $exit_code"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

# Run Docker security scanning
run_docker_scan() {
    if [ "$RUN_DOCKER_SCAN" != "true" ]; then
        log_info "Docker scan skipped"
        return 0
    fi
    
    log_header "DOCKER SECURITY SCANNING"
    
    local report_file="$REPORTS_DIR/docker_scan_$TIMESTAMP.json"
    local exit_code=0
    
    # Scan Docker images with Trivy if available
    if command -v trivy >/dev/null 2>&1; then
        log_info "Scanning Docker images with Trivy..."
        
        # Find Docker images to scan
        local images=()
        if [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
            # Extract image names from docker-compose
            images+=($(grep -E "^\s*image:" "$PROJECT_ROOT/docker-compose.yml" | awk '{print $2}' | tr -d '"' || true))
        fi
        
        # Add common images
        images+=("stock-prediction-backend" "stock-prediction-frontend")
        
        for image in "${images[@]}"; do
            if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "$image"; then
                log_info "Scanning image: $image"
                trivy image --format json --output "${report_file%.json}_${image//[:\/]/_}.json" "$image" || true
            fi
        done
    else
        log_warning "Trivy not available, installing..."
        # Try to install Trivy
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y trivy || log_warning "Failed to install Trivy"
        elif command -v yum >/dev/null 2>&1; then
            sudo yum install -y trivy || log_warning "Failed to install Trivy"
        else
            log_warning "Cannot install Trivy automatically"
            exit_code=1
        fi
    fi
    
    TEST_RESULTS["docker_scan"]=$exit_code
    TEST_REPORTS["docker_scan"]="$report_file"
    
    if [ $exit_code -eq 0 ]; then
        log_success "Docker scan completed successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log_error "Docker scan failed with exit code $exit_code"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

# Run configuration security scanning
run_config_scan() {
    if [ "$RUN_CONFIG_SCAN" != "true" ]; then
        log_info "Configuration scan skipped"
        return 0
    fi
    
    log_header "CONFIGURATION SECURITY SCANNING"
    
    local report_file="$REPORTS_DIR/config_scan_$TIMESTAMP.json"
    local exit_code=0
    
    # Run the configuration scanner
    if [ -f "$SCRIPT_DIR/security-config-scanner.py" ]; then
        python3 "$SCRIPT_DIR/security-config-scanner.py" \
            --project-root "$PROJECT_ROOT" \
            --output "$report_file" || exit_code=$?
    else
        log_error "security-config-scanner.py not found"
        exit_code=1
    fi
    
    TEST_RESULTS["config_scan"]=$exit_code
    TEST_REPORTS["config_scan"]="$report_file"
    
    if [ $exit_code -eq 0 ]; then
        log_success "Configuration scan completed successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log_error "Configuration scan failed with exit code $exit_code"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

# Run API security testing
run_api_security() {
    if [ "$RUN_API_SECURITY" != "true" ]; then
        log_info "API security test skipped"
        return 0
    fi
    
    log_header "API SECURITY TESTING"
    
    local report_file="$REPORTS_DIR/api_security_$TIMESTAMP.json"
    local exit_code=0
    
    # Run the API security tester
    if [ -f "$SCRIPT_DIR/api-security-test.py" ]; then
        python3 "$SCRIPT_DIR/api-security-test.py" \
            --url "$TARGET_URL" \
            --ws-url "$WS_URL" \
            --output "$report_file" || exit_code=$?
    else
        log_error "api-security-test.py not found"
        exit_code=1
    fi
    
    TEST_RESULTS["api_security"]=$exit_code
    TEST_REPORTS["api_security"]="$report_file"
    
    if [ $exit_code -eq 0 ]; then
        log_success "API security test completed successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log_error "API security test failed with exit code $exit_code"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

# Run penetration testing
run_penetration_test() {
    if [ "$RUN_PENETRATION_TEST" != "true" ]; then
        log_info "Penetration test skipped"
        return 0
    fi
    
    log_header "PENETRATION TESTING"
    
    local report_file="$REPORTS_DIR/penetration_test_$TIMESTAMP.json"
    local exit_code=0
    
    # Run the penetration tester
    if [ -f "$SCRIPT_DIR/penetration-test.py" ]; then
        python3 "$SCRIPT_DIR/penetration-test.py" \
            --target "$TARGET_URL" \
            --output "$report_file" || exit_code=$?
    else
        log_error "penetration-test.py not found"
        exit_code=1
    fi
    
    TEST_RESULTS["penetration_test"]=$exit_code
    TEST_REPORTS["penetration_test"]="$report_file"
    
    if [ $exit_code -eq 0 ]; then
        log_success "Penetration test completed successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log_error "Penetration test failed with exit code $exit_code"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

# Run network security scanning
run_network_scan() {
    if [ "$RUN_NETWORK_SCAN" != "true" ]; then
        log_info "Network scan skipped"
        return 0
    fi
    
    log_header "NETWORK SECURITY SCANNING"
    
    local report_file="$REPORTS_DIR/network_scan_$TIMESTAMP.json"
    local exit_code=0
    
    # Extract host from URL
    local host=$(echo "$TARGET_URL" | sed -E 's|^https?://([^:/]+).*|\1|')
    
    # Run nmap if available
    if command -v nmap >/dev/null 2>&1; then
        log_info "Running nmap scan on $host..."
        nmap -sV -sC -O -A --script vuln "$host" -oX "${report_file%.json}.xml" || exit_code=$?
        
        # Convert XML to JSON if possible
        if command -v python3 >/dev/null 2>&1; then
            python3 -c "
import xml.etree.ElementTree as ET
import json
try:
    tree = ET.parse('${report_file%.json}.xml')
    root = tree.getroot()
    # Simple XML to JSON conversion
    result = {'nmap_scan': ET.tostring(root, encoding='unicode')}
    with open('$report_file', 'w') as f:
        json.dump(result, f, indent=2)
except Exception as e:
    print(f'Error converting XML to JSON: {e}')
" || true
        fi
    else
        log_warning "nmap not available, skipping network scan"
        exit_code=1
    fi
    
    TEST_RESULTS["network_scan"]=$exit_code
    TEST_REPORTS["network_scan"]="$report_file"
    
    if [ $exit_code -eq 0 ]; then
        log_success "Network scan completed successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log_error "Network scan failed with exit code $exit_code"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

# Generate consolidated report
generate_consolidated_report() {
    log_header "GENERATING CONSOLIDATED REPORT"
    
    local consolidated_report="$REPORTS_DIR/security_report_consolidated_$TIMESTAMP.json"
    local html_report="$REPORTS_DIR/security_report_$TIMESTAMP.html"
    
    # Count total vulnerabilities from all reports
    local total_vulnerabilities=0
    local critical_count=0
    local high_count=0
    local medium_count=0
    local low_count=0
    
    for report_file in "${TEST_REPORTS[@]}"; do
        if [ -f "$report_file" ] && command -v jq >/dev/null 2>&1; then
            # Try to extract vulnerability counts
            local vuln_count=$(jq -r '.vulnerabilities_found // .total_findings // 0' "$report_file" 2>/dev/null || echo "0")
            total_vulnerabilities=$((total_vulnerabilities + vuln_count))
            
            # Extract severity counts if available
            local crit=$(jq -r '.summary.critical // .critical // 0' "$report_file" 2>/dev/null || echo "0")
            local high=$(jq -r '.summary.high // .high // 0' "$report_file" 2>/dev/null || echo "0")
            local med=$(jq -r '.summary.medium // .medium // 0' "$report_file" 2>/dev/null || echo "0")
            local low=$(jq -r '.summary.low // .low // 0' "$report_file" 2>/dev/null || echo "0")
            
            critical_count=$((critical_count + crit))
            high_count=$((high_count + high))
            medium_count=$((medium_count + med))
            low_count=$((low_count + low))
        fi
    done
    
    # Create consolidated JSON report
    cat > "$consolidated_report" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "test_session": "$TEST_SESSION",
  "target_url": "$TARGET_URL",
  "project_root": "$PROJECT_ROOT",
  "summary": {
    "total_tests": $TOTAL_TESTS,
    "passed_tests": $PASSED_TESTS,
    "failed_tests": $FAILED_TESTS,
    "success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0"),
    "total_vulnerabilities": $total_vulnerabilities,
    "critical_vulnerabilities": $critical_count,
    "high_vulnerabilities": $high_count,
    "medium_vulnerabilities": $medium_count,
    "low_vulnerabilities": $low_count
  },
  "test_results": {
EOF
    
    # Add test results
    local first=true
    for test_name in "${!TEST_RESULTS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$consolidated_report"
        fi
        echo "    \"$test_name\": {" >> "$consolidated_report"
        echo "      \"exit_code\": ${TEST_RESULTS[$test_name]}," >> "$consolidated_report"
        echo "      \"status\": \"$([ ${TEST_RESULTS[$test_name]} -eq 0 ] && echo "PASS" || echo "FAIL")\"," >> "$consolidated_report"
        echo "      \"report_file\": \"${TEST_REPORTS[$test_name]}\"" >> "$consolidated_report"
        echo -n "    }" >> "$consolidated_report"
    done
    
    cat >> "$consolidated_report" << EOF

  },
  "individual_reports": [
EOF
    
    # Add individual report paths
    first=true
    for report_file in "${TEST_REPORTS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$consolidated_report"
        fi
        echo -n "    \"$report_file\"" >> "$consolidated_report"
    done
    
    cat >> "$consolidated_report" << EOF

  ]
}
EOF
    
    # Generate HTML report
    generate_html_report "$consolidated_report" "$html_report"
    
    log_success "Consolidated report generated: $consolidated_report"
    log_success "HTML report generated: $html_report"
    
    TOTAL_VULNERABILITIES=$total_vulnerabilities
}

# Generate HTML report
generate_html_report() {
    local json_report="$1"
    local html_report="$2"
    
    cat > "$html_report" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Testing Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
        .metric h3 { margin: 0 0 10px 0; color: #333; }
        .metric .value { font-size: 24px; font-weight: bold; }
        .critical { color: #dc3545; }
        .high { color: #fd7e14; }
        .medium { color: #ffc107; }
        .low { color: #28a745; }
        .pass { color: #28a745; }
        .fail { color: #dc3545; }
        .test-results { margin-top: 30px; }
        .test-item { background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
        .test-item.fail { border-left-color: #dc3545; }
        .test-item.pass { border-left-color: #28a745; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .footer { margin-top: 30px; text-align: center; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 Security Testing Report</h1>
            <p>Stock Prediction System Security Assessment</p>
            <p><strong>Generated:</strong> <span id="timestamp"></span></p>
        </div>
        
        <div class="summary">
            <div class="metric">
                <h3>Total Tests</h3>
                <div class="value" id="total-tests">-</div>
            </div>
            <div class="metric">
                <h3>Success Rate</h3>
                <div class="value" id="success-rate">-</div>
            </div>
            <div class="metric">
                <h3>Total Vulnerabilities</h3>
                <div class="value" id="total-vulns">-</div>
            </div>
            <div class="metric">
                <h3>Critical</h3>
                <div class="value critical" id="critical-vulns">-</div>
            </div>
            <div class="metric">
                <h3>High</h3>
                <div class="value high" id="high-vulns">-</div>
            </div>
            <div class="metric">
                <h3>Medium</h3>
                <div class="value medium" id="medium-vulns">-</div>
            </div>
            <div class="metric">
                <h3>Low</h3>
                <div class="value low" id="low-vulns">-</div>
            </div>
        </div>
        
        <div class="test-results">
            <h2>Test Results</h2>
            <div id="test-results-container"></div>
        </div>
        
        <div class="footer">
            <p>Security testing completed for Stock Prediction System</p>
            <p>Review individual reports for detailed findings and recommendations</p>
        </div>
    </div>
    
    <script>
        // Load and display report data
        const reportData = 
EOF
    
    # Append JSON data to HTML
    cat "$json_report" >> "$html_report"
    
    cat >> "$html_report" << 'EOF'
;
        
        // Populate the HTML with data
        document.getElementById('timestamp').textContent = new Date(reportData.timestamp).toLocaleString();
        document.getElementById('total-tests').textContent = reportData.summary.total_tests;
        document.getElementById('success-rate').textContent = reportData.summary.success_rate + '%';
        document.getElementById('total-vulns').textContent = reportData.summary.total_vulnerabilities;
        document.getElementById('critical-vulns').textContent = reportData.summary.critical_vulnerabilities;
        document.getElementById('high-vulns').textContent = reportData.summary.high_vulnerabilities;
        document.getElementById('medium-vulns').textContent = reportData.summary.medium_vulnerabilities;
        document.getElementById('low-vulns').textContent = reportData.summary.low_vulnerabilities;
        
        // Populate test results
        const container = document.getElementById('test-results-container');
        for (const [testName, result] of Object.entries(reportData.test_results)) {
            const div = document.createElement('div');
            div.className = `test-item ${result.status.toLowerCase()}`;
            div.innerHTML = `
                <h4>${testName.replace(/_/g, ' ').toUpperCase()}</h4>
                <p><strong>Status:</strong> <span class="${result.status.toLowerCase()}">${result.status}</span></p>
                <p><strong>Report:</strong> ${result.report_file}</p>
            `;
            container.appendChild(div);
        }
    </script>
</body>
</html>
EOF
}

# Send notifications
send_notifications() {
    log_header "SENDING NOTIFICATIONS"
    
    local message="Security Testing Completed\n\n"
    message+="📊 **Summary:**\n"
    message+="• Total Tests: $TOTAL_TESTS\n"
    message+="• Passed: $PASSED_TESTS\n"
    message+="• Failed: $FAILED_TESTS\n"
    message+="• Total Vulnerabilities: $TOTAL_VULNERABILITIES\n\n"
    
    if [ $TOTAL_VULNERABILITIES -gt 0 ]; then
        message+="⚠️ **Action Required:** $TOTAL_VULNERABILITIES vulnerabilities found\n"
    else
        message+="✅ **No vulnerabilities found**\n"
    fi
    
    message+="\n📋 **Reports:** $REPORTS_DIR/security_report_$TIMESTAMP.html"
    
    # Send Slack notification
    if [ -n "$SLACK_WEBHOOK" ]; then
        log_info "Sending Slack notification..."
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK" || log_warning "Failed to send Slack notification"
    fi
    
    # Send email notification
    if [ -n "$EMAIL_RECIPIENT" ] && [ -n "$SMTP_SERVER" ]; then
        log_info "Sending email notification..."
        # This would require a proper email sending setup
        log_warning "Email notification not implemented yet"
    fi
}

# Print final summary
print_summary() {
    echo -e "\n${CYAN}" | tee -a "$LOG_FILE"
    echo "=========================================================" | tee -a "$LOG_FILE"
    echo "                SECURITY TESTING SUMMARY" | tee -a "$LOG_FILE"
    echo "=========================================================" | tee -a "$LOG_FILE"
    echo -e "${NC}" | tee -a "$LOG_FILE"
    
    echo "📅 Test Session: $TEST_SESSION" | tee -a "$LOG_FILE"
    echo "🎯 Target: $TARGET_URL" | tee -a "$LOG_FILE"
    echo "📁 Project: $PROJECT_ROOT" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    echo "📊 Test Results:" | tee -a "$LOG_FILE"
    echo "   Total Tests: $TOTAL_TESTS" | tee -a "$LOG_FILE"
    echo "   Passed: $PASSED_TESTS" | tee -a "$LOG_FILE"
    echo "   Failed: $FAILED_TESTS" | tee -a "$LOG_FILE"
    
    if [ $TOTAL_TESTS -gt 0 ]; then
        local success_rate=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
        echo "   Success Rate: ${success_rate}%" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "🔍 Security Findings:" | tee -a "$LOG_FILE"
    echo "   Total Vulnerabilities: $TOTAL_VULNERABILITIES" | tee -a "$LOG_FILE"
    
    if [ $TOTAL_VULNERABILITIES -eq 0 ]; then
        echo -e "   ${GREEN}✅ No vulnerabilities found!${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "   ${RED}⚠️  Action required - vulnerabilities detected${NC}" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "📋 Reports Generated:" | tee -a "$LOG_FILE"
    echo "   📄 Consolidated JSON: $REPORTS_DIR/security_report_consolidated_$TIMESTAMP.json" | tee -a "$LOG_FILE"
    echo "   🌐 HTML Report: $REPORTS_DIR/security_report_$TIMESTAMP.html" | tee -a "$LOG_FILE"
    echo "   📝 Log File: $LOG_FILE" | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo -e "${CYAN}=========================================================${NC}" | tee -a "$LOG_FILE"
}

# Main execution
main() {
    log_header "STARTING SECURITY TESTING SUITE"
    log_info "Test session: $TEST_SESSION"
    log_info "Target URL: $TARGET_URL"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Reports directory: $REPORTS_DIR"
    
    # Run all security tests
    check_prerequisites
    run_dependency_scan
    run_static_analysis
    run_docker_scan
    run_config_scan
    run_api_security
    run_penetration_test
    run_network_scan
    
    # Generate reports and notifications
    generate_consolidated_report
    send_notifications
    print_summary
    
    # Exit with appropriate code
    if [ $FAILED_TESTS -gt 0 ] || [ $TOTAL_VULNERABILITIES -gt 0 ]; then
        log_warning "Security testing completed with issues"
        exit 1
    else
        log_success "Security testing completed successfully"
        exit 0
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target-url)
            TARGET_URL="$2"
            shift 2
            ;;
        --ws-url)
            WS_URL="$2"
            shift 2
            ;;
        --skip-dependency)
            RUN_DEPENDENCY_SCAN="false"
            shift
            ;;
        --skip-static)
            RUN_STATIC_ANALYSIS="false"
            shift
            ;;
        --skip-docker)
            RUN_DOCKER_SCAN="false"
            shift
            ;;
        --skip-config)
            RUN_CONFIG_SCAN="false"
            shift
            ;;
        --skip-api)
            RUN_API_SECURITY="false"
            shift
            ;;
        --skip-pentest)
            RUN_PENETRATION_TEST="false"
            shift
            ;;
        --enable-network)
            RUN_NETWORK_SCAN="true"
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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --target-url URL       Target application URL (default: http://localhost:8000)"
            echo "  --ws-url URL          WebSocket URL (default: ws://localhost:8000)"
            echo "  --skip-dependency     Skip dependency vulnerability scanning"
            echo "  --skip-static         Skip static code analysis"
            echo "  --skip-docker         Skip Docker security scanning"
            echo "  --skip-config         Skip configuration security scanning"
            echo "  --skip-api            Skip API security testing"
            echo "  --skip-pentest        Skip penetration testing"
            echo "  --enable-network      Enable network security scanning"
            echo "  --slack-webhook URL   Slack webhook for notifications"
            echo "  --email EMAIL         Email recipient for notifications"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main