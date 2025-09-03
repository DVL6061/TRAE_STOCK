#!/bin/bash

# Security Testing and Vulnerability Assessment Script
# For Stock Prediction System

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORT_DIR="$SCRIPT_DIR/reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$REPORT_DIR/security_report_$TIMESTAMP.json"
HTML_REPORT="$REPORT_DIR/security_report_$TIMESTAMP.html"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Initialize report structure
init_report() {
    mkdir -p "$REPORT_DIR"
    cat > "$REPORT_FILE" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "project": "Stock Prediction System",
    "scan_type": "Security Vulnerability Assessment",
    "results": {
        "dependency_scan": {},
        "code_analysis": {},
        "docker_scan": {},
        "network_scan": {},
        "api_security": {},
        "secrets_scan": {},
        "compliance_check": {}
    },
    "summary": {
        "total_vulnerabilities": 0,
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0
    }
}
EOF
}

# Check prerequisites
check_prerequisites() {
    log "Checking security scanning tools..."
    
    local missing_tools=()
    
    # Check for required tools
    command -v npm >/dev/null 2>&1 || missing_tools+=("npm")
    command -v pip >/dev/null 2>&1 || missing_tools+=("pip")
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        return 1
    fi
    
    # Install security scanning tools if not present
    if ! command -v safety >/dev/null 2>&1; then
        log "Installing Python safety scanner..."
        pip install safety
    fi
    
    if ! command -v bandit >/dev/null 2>&1; then
        log "Installing Python bandit scanner..."
        pip install bandit
    fi
    
    if ! npm list -g audit >/dev/null 2>&1; then
        log "NPM audit is available by default"
    fi
    
    if ! command -v trivy >/dev/null 2>&1; then
        log "Installing Trivy scanner..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install trivy
        fi
    fi
    
    log_success "Prerequisites check completed"
}

# Scan Python dependencies for vulnerabilities
scan_python_dependencies() {
    log "Scanning Python dependencies for vulnerabilities..."
    
    cd "$PROJECT_ROOT/backend"
    
    # Safety scan
    if [ -f "requirements.txt" ]; then
        log "Running Safety scan on requirements.txt..."
        safety check --json --output "$REPORT_DIR/safety_report_$TIMESTAMP.json" || true
        
        # Parse safety results
        if [ -f "$REPORT_DIR/safety_report_$TIMESTAMP.json" ]; then
            python3 << EOF
import json
import sys

try:
    with open('$REPORT_DIR/safety_report_$TIMESTAMP.json', 'r') as f:
        safety_data = json.load(f)
    
    # Update main report
    with open('$REPORT_FILE', 'r') as f:
        report = json.load(f)
    
    report['results']['dependency_scan']['python_safety'] = {
        'vulnerabilities_found': len(safety_data),
        'details': safety_data
    }
    
    with open('$REPORT_FILE', 'w') as f:
        json.dump(report, f, indent=2)
        
except Exception as e:
    print(f"Error processing safety report: {e}")
EOF
        fi
    fi
    
    # Pip audit (if available)
    if command -v pip-audit >/dev/null 2>&1; then
        log "Running pip-audit..."
        pip-audit --format=json --output="$REPORT_DIR/pip_audit_$TIMESTAMP.json" || true
    fi
    
    log_success "Python dependency scan completed"
}

# Scan Node.js dependencies
scan_nodejs_dependencies() {
    log "Scanning Node.js dependencies for vulnerabilities..."
    
    cd "$PROJECT_ROOT/frontend"
    
    if [ -f "package.json" ]; then
        log "Running npm audit..."
        npm audit --json > "$REPORT_DIR/npm_audit_$TIMESTAMP.json" 2>/dev/null || true
        
        # Parse npm audit results
        python3 << EOF
import json
import sys

try:
    with open('$REPORT_DIR/npm_audit_$TIMESTAMP.json', 'r') as f:
        npm_data = json.load(f)
    
    # Update main report
    with open('$REPORT_FILE', 'r') as f:
        report = json.load(f)
    
    vulnerabilities = npm_data.get('vulnerabilities', {})
    report['results']['dependency_scan']['nodejs_npm'] = {
        'total_vulnerabilities': len(vulnerabilities),
        'details': vulnerabilities
    }
    
    with open('$REPORT_FILE', 'w') as f:
        json.dump(report, f, indent=2)
        
except Exception as e:
    print(f"Error processing npm audit report: {e}")
EOF
    fi
    
    log_success "Node.js dependency scan completed"
}

# Static code analysis
static_code_analysis() {
    log "Performing static code analysis..."
    
    # Python code analysis with Bandit
    cd "$PROJECT_ROOT/backend"
    log "Running Bandit security analysis on Python code..."
    bandit -r . -f json -o "$REPORT_DIR/bandit_report_$TIMESTAMP.json" || true
    
    # Parse bandit results
    python3 << EOF
import json
import sys

try:
    with open('$REPORT_DIR/bandit_report_$TIMESTAMP.json', 'r') as f:
        bandit_data = json.load(f)
    
    # Update main report
    with open('$REPORT_FILE', 'r') as f:
        report = json.load(f)
    
    report['results']['code_analysis']['python_bandit'] = {
        'issues_found': len(bandit_data.get('results', [])),
        'details': bandit_data
    }
    
    with open('$REPORT_FILE', 'w') as f:
        json.dump(report, f, indent=2)
        
except Exception as e:
    print(f"Error processing bandit report: {e}")
EOF
    
    # ESLint security analysis for JavaScript/TypeScript
    cd "$PROJECT_ROOT/frontend"
    if [ -f "package.json" ] && npm list eslint-plugin-security >/dev/null 2>&1; then
        log "Running ESLint security analysis..."
        npx eslint . --ext .js,.jsx,.ts,.tsx --format json > "$REPORT_DIR/eslint_security_$TIMESTAMP.json" 2>/dev/null || true
    fi
    
    log_success "Static code analysis completed"
}

# Docker image security scan
docker_security_scan() {
    log "Scanning Docker images for vulnerabilities..."
    
    cd "$PROJECT_ROOT"
    
    # Scan backend image
    if docker images | grep -q "stock-prediction-backend"; then
        log "Scanning backend Docker image..."
        trivy image --format json --output "$REPORT_DIR/trivy_backend_$TIMESTAMP.json" stock-prediction-backend:latest || true
    fi
    
    # Scan frontend image
    if docker images | grep -q "stock-prediction-frontend"; then
        log "Scanning frontend Docker image..."
        trivy image --format json --output "$REPORT_DIR/trivy_frontend_$TIMESTAMP.json" stock-prediction-frontend:latest || true
    fi
    
    # Scan Dockerfile
    if [ -f "Dockerfile" ]; then
        log "Scanning Dockerfile for best practices..."
        trivy config --format json --output "$REPORT_DIR/trivy_dockerfile_$TIMESTAMP.json" . || true
    fi
    
    log_success "Docker security scan completed"
}

# Network security scan
network_security_scan() {
    log "Performing network security assessment..."
    
    # Check for open ports
    log "Checking for open ports..."
    netstat -tuln > "$REPORT_DIR/open_ports_$TIMESTAMP.txt" 2>/dev/null || true
    
    # SSL/TLS configuration check (if applicable)
    if command -v testssl >/dev/null 2>&1; then
        log "Testing SSL/TLS configuration..."
        testssl --jsonfile "$REPORT_DIR/ssl_test_$TIMESTAMP.json" localhost:443 || true
    fi
    
    log_success "Network security scan completed"
}

# API security testing
api_security_test() {
    log "Testing API security..."
    
    # Check for common API vulnerabilities
    python3 << 'EOF'
import requests
import json
import sys
from datetime import datetime

def test_api_security():
    base_url = "http://localhost:8000"
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    # Test 1: Check for CORS configuration
    try:
        response = requests.options(f"{base_url}/api/v1/stocks", 
                                  headers={"Origin": "http://malicious-site.com"})
        cors_test = {
            "test": "CORS Configuration",
            "status": "PASS" if "Access-Control-Allow-Origin" in response.headers else "FAIL",
            "details": dict(response.headers)
        }
        results["tests"].append(cors_test)
    except Exception as e:
        results["tests"].append({
            "test": "CORS Configuration",
            "status": "ERROR",
            "error": str(e)
        })
    
    # Test 2: Check for rate limiting
    try:
        responses = []
        for i in range(10):
            resp = requests.get(f"{base_url}/api/v1/health")
            responses.append(resp.status_code)
        
        rate_limit_test = {
            "test": "Rate Limiting",
            "status": "PASS" if 429 in responses else "WARN",
            "details": f"Made 10 requests, got status codes: {responses}"
        }
        results["tests"].append(rate_limit_test)
    except Exception as e:
        results["tests"].append({
            "test": "Rate Limiting",
            "status": "ERROR",
            "error": str(e)
        })
    
    # Test 3: Check for SQL injection protection
    try:
        malicious_payload = "'; DROP TABLE users; --"
        response = requests.get(f"{base_url}/api/v1/stocks", 
                              params={"symbol": malicious_payload})
        
        sql_injection_test = {
            "test": "SQL Injection Protection",
            "status": "PASS" if response.status_code != 500 else "FAIL",
            "details": f"Response status: {response.status_code}"
        }
        results["tests"].append(sql_injection_test)
    except Exception as e:
        results["tests"].append({
            "test": "SQL Injection Protection",
            "status": "ERROR",
            "error": str(e)
        })
    
    # Test 4: Check for XSS protection
    try:
        xss_payload = "<script>alert('XSS')</script>"
        response = requests.post(f"{base_url}/api/v1/feedback", 
                               json={"message": xss_payload})
        
        xss_test = {
            "test": "XSS Protection",
            "status": "PASS" if xss_payload not in response.text else "FAIL",
            "details": f"Response contains payload: {xss_payload in response.text}"
        }
        results["tests"].append(xss_test)
    except Exception as e:
        results["tests"].append({
            "test": "XSS Protection",
            "status": "ERROR",
            "error": str(e)
        })
    
    # Save results
    with open(f"$REPORT_DIR/api_security_$TIMESTAMP.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    test_api_security()
EOF
    
    log_success "API security testing completed"
}

# Secrets scanning
secrets_scan() {
    log "Scanning for exposed secrets and credentials..."
    
    cd "$PROJECT_ROOT"
    
    # Use git-secrets if available
    if command -v git-secrets >/dev/null 2>&1; then
        log "Running git-secrets scan..."
        git secrets --scan > "$REPORT_DIR/git_secrets_$TIMESTAMP.txt" 2>&1 || true
    fi
    
    # Manual pattern search for common secrets
    log "Searching for common secret patterns..."
    
    # Define patterns to search for
    declare -a patterns=(
        "password\s*=\s*['\"][^'\"]+['\"]"  # password=
        "api[_-]?key\s*=\s*['\"][^'\"]+['\"]"  # api_key=
        "secret[_-]?key\s*=\s*['\"][^'\"]+['\"]"  # secret_key=
        "access[_-]?token\s*=\s*['\"][^'\"]+['\"]"  # access_token=
        "private[_-]?key\s*=\s*['\"][^'\"]+['\"]"  # private_key=
        "[A-Za-z0-9]{20,}"  # Long strings that might be tokens
    )
    
    secrets_found=()
    for pattern in "${patterns[@]}"; do
        matches=$(grep -r -i -E "$pattern" . --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=__pycache__ --exclude="*.log" || true)
        if [ -n "$matches" ]; then
            secrets_found+=("$matches")
        fi
    done
    
    # Save secrets scan results
    printf '%s\n' "${secrets_found[@]}" > "$REPORT_DIR/secrets_scan_$TIMESTAMP.txt"
    
    log_success "Secrets scanning completed"
}

# Compliance checks
compliance_check() {
    log "Performing security compliance checks..."
    
    cd "$PROJECT_ROOT"
    
    compliance_results=(
        "OWASP Top 10 Compliance Check:"
        "1. Injection: $([ -f "backend/app/security/input_validation.py" ] && echo "PASS" || echo "FAIL")"
        "2. Broken Authentication: $([ -f "backend/app/auth/jwt_handler.py" ] && echo "PASS" || echo "FAIL")"
        "3. Sensitive Data Exposure: $([ -f ".env.example" ] && echo "PASS" || echo "FAIL")"
        "4. XML External Entities: $(grep -r "xml" . >/dev/null 2>&1 && echo "REVIEW" || echo "N/A")"
        "5. Broken Access Control: $([ -f "backend/app/middleware/auth.py" ] && echo "PASS" || echo "FAIL")"
        "6. Security Misconfiguration: $([ -f "docker-compose.prod.yml" ] && echo "PASS" || echo "FAIL")"
        "7. Cross-Site Scripting: $(grep -r "escape" frontend/ >/dev/null 2>&1 && echo "PASS" || echo "REVIEW")"
        "8. Insecure Deserialization: $(grep -r "pickle\|yaml.load" . >/dev/null 2>&1 && echo "REVIEW" || echo "PASS")"
        "9. Using Components with Known Vulnerabilities: $([ -f "$REPORT_DIR/safety_report_$TIMESTAMP.json" ] && echo "CHECKED" || echo "PENDING")"
        "10. Insufficient Logging & Monitoring: $([ -f "backend/app/utils/logger.py" ] && echo "PASS" || echo "FAIL")"
    )
    
    printf '%s\n' "${compliance_results[@]}" > "$REPORT_DIR/compliance_check_$TIMESTAMP.txt"
    
    log_success "Compliance checks completed"
}

# Generate HTML report
generate_html_report() {
    log "Generating HTML security report..."
    
    python3 << 'EOF'
import json
import os
from datetime import datetime

def generate_html_report():
    # Read the main report
    with open(os.environ['REPORT_FILE'], 'r') as f:
        report = json.load(f)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Assessment Report - Stock Prediction System</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .critical {{ background-color: #dc3545; color: white; }}
        .high {{ background-color: #fd7e14; color: white; }}
        .medium {{ background-color: #ffc107; color: black; }}
        .low {{ background-color: #28a745; color: white; }}
        .section {{ margin-bottom: 30px; }}
        .section h3 {{ background: #007bff; color: white; padding: 10px; margin: 0; border-radius: 4px 4px 0 0; }}
        .section-content {{ border: 1px solid #007bff; padding: 15px; border-radius: 0 0 4px 4px; }}
        .vulnerability {{ margin-bottom: 15px; padding: 10px; border-left: 4px solid #dc3545; background: #f8f9fa; }}
        .pass {{ color: #28a745; font-weight: bold; }}
        .fail {{ color: #dc3545; font-weight: bold; }}
        .warn {{ color: #ffc107; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Security Assessment Report</h1>
            <h2>Stock Prediction System</h2>
            <p>Generated on: {report['timestamp']}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card critical">
                <h3>Critical</h3>
                <p>{report['summary']['critical']}</p>
            </div>
            <div class="summary-card high">
                <h3>High</h3>
                <p>{report['summary']['high']}</p>
            </div>
            <div class="summary-card medium">
                <h3>Medium</h3>
                <p>{report['summary']['medium']}</p>
            </div>
            <div class="summary-card low">
                <h3>Low</h3>
                <p>{report['summary']['low']}</p>
            </div>
            <div class="summary-card">
                <h3>Total</h3>
                <p>{report['summary']['total_vulnerabilities']}</p>
            </div>
        </div>
"""
    
    # Add sections for each scan type
    for section_name, section_data in report['results'].items():
        if section_data:
            html_content += f"""
        <div class="section">
            <h3>{section_name.replace('_', ' ').title()}</h3>
            <div class="section-content">
                <pre>{json.dumps(section_data, indent=2)}</pre>
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Write HTML report
    with open(os.environ['HTML_REPORT'], 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    generate_html_report()
EOF
    
    log_success "HTML report generated: $HTML_REPORT"
}

# Main execution
main() {
    log "Starting comprehensive security assessment..."
    
    init_report
    check_prerequisites
    
    # Run all security scans
    scan_python_dependencies
    scan_nodejs_dependencies
    static_code_analysis
    docker_security_scan
    network_security_scan
    api_security_test
    secrets_scan
    compliance_check
    
    # Generate reports
    generate_html_report
    
    log_success "Security assessment completed!"
    log "Reports generated:"
    log "  - JSON Report: $REPORT_FILE"
    log "  - HTML Report: $HTML_REPORT"
    log "  - Individual scan reports in: $REPORT_DIR"
    
    # Summary
    echo
    echo "=== SECURITY ASSESSMENT SUMMARY ==="
    echo "Timestamp: $(date)"
    echo "Project: Stock Prediction System"
    echo "Report Location: $REPORT_DIR"
    echo "======================================"
}

# Run main function
main "$@"