#!/bin/bash

# Health Check Script for Stock Prediction System
# This script monitors all services and provides detailed health status

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/stock-prediction/health-check.log"
ALERT_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
MAX_LOG_SIZE=10485760  # 10MB

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service endpoints
BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3001"
REDIS_HOST="localhost"
REDIS_PORT="6379"
POSTGRES_HOST="localhost"
POSTGRES_PORT="5432"
POSTGRES_DB="${POSTGRES_DB:-stock_prediction}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Rotate log if too large
    if [[ -f "$LOG_FILE" ]] && [[ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0) -gt $MAX_LOG_SIZE ]]; then
        mv "$LOG_FILE" "${LOG_FILE}.old"
    fi
    
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# Print colored status
print_status() {
    local status=$1
    local service=$2
    local message=$3
    
    case $status in
        "HEALTHY")
            echo -e "${GREEN}✓${NC} $service: $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠${NC} $service: $message"
            ;;
        "CRITICAL")
            echo -e "${RED}✗${NC} $service: $message"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ${NC} $service: $message"
            ;;
    esac
}

# Send alert to Slack/webhook
send_alert() {
    local severity=$1
    local service=$2
    local message=$3
    
    if [[ -n "$ALERT_WEBHOOK" ]]; then
        local color="good"
        case $severity in
            "WARNING") color="warning" ;;
            "CRITICAL") color="danger" ;;
        esac
        
        local payload=$(cat <<EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "Stock Prediction System Alert",
            "fields": [
                {
                    "title": "Service",
                    "value": "$service",
                    "short": true
                },
                {
                    "title": "Severity",
                    "value": "$severity",
                    "short": true
                },
                {
                    "title": "Message",
                    "value": "$message",
                    "short": false
                },
                {
                    "title": "Timestamp",
                    "value": "$(date)",
                    "short": true
                },
                {
                    "title": "Host",
                    "value": "$(hostname)",
                    "short": true
                }
            ]
        }
    ]
}
EOF
        )
        
        curl -s -X POST -H 'Content-type: application/json' \
            --data "$payload" "$ALERT_WEBHOOK" > /dev/null || true
    fi
}

# Check HTTP endpoint
check_http_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    local timeout=${4:-10}
    
    local response
    local http_code
    local response_time
    
    if response=$(curl -s -w "\n%{http_code}\n%{time_total}" \
                      --max-time "$timeout" \
                      --connect-timeout 5 \
                      "$url" 2>/dev/null); then
        
        http_code=$(echo "$response" | tail -n 2 | head -n 1)
        response_time=$(echo "$response" | tail -n 1)
        
        if [[ "$http_code" == "$expected_status" ]]; then
            print_status "HEALTHY" "$name" "HTTP $http_code (${response_time}s)"
            log "INFO" "$name health check passed: HTTP $http_code in ${response_time}s"
            return 0
        else
            print_status "WARNING" "$name" "HTTP $http_code (expected $expected_status)"
            log "WARNING" "$name health check warning: HTTP $http_code, expected $expected_status"
            send_alert "WARNING" "$name" "HTTP $http_code (expected $expected_status)"
            return 1
        fi
    else
        print_status "CRITICAL" "$name" "Connection failed"
        log "ERROR" "$name health check failed: Connection timeout or error"
        send_alert "CRITICAL" "$name" "Connection failed or timeout"
        return 1
    fi
}

# Check Redis
check_redis() {
    local name="Redis"
    
    if command -v redis-cli >/dev/null 2>&1; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
            local memory_usage=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
            local connected_clients=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" info clients | grep connected_clients | cut -d: -f2 | tr -d '\r')
            
            print_status "HEALTHY" "$name" "Memory: $memory_usage, Clients: $connected_clients"
            log "INFO" "Redis health check passed: Memory $memory_usage, Clients $connected_clients"
            return 0
        else
            print_status "CRITICAL" "$name" "Connection failed"
            log "ERROR" "Redis health check failed: Cannot connect"
            send_alert "CRITICAL" "$name" "Connection failed"
            return 1
        fi
    else
        print_status "WARNING" "$name" "redis-cli not available"
        log "WARNING" "Redis health check skipped: redis-cli not installed"
        return 1
    fi
}

# Check PostgreSQL
check_postgres() {
    local name="PostgreSQL"
    
    if command -v pg_isready >/dev/null 2>&1; then
        if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -d "$POSTGRES_DB" -U "$POSTGRES_USER" >/dev/null 2>&1; then
            # Get database size and connection count
            local db_size="N/A"
            local connections="N/A"
            
            if command -v psql >/dev/null 2>&1 && [[ -n "${POSTGRES_PASSWORD:-}" ]]; then
                export PGPASSWORD="$POSTGRES_PASSWORD"
                db_size=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -d "$POSTGRES_DB" -U "$POSTGRES_USER" -t -c "SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'));" 2>/dev/null | xargs || echo "N/A")
                connections=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -d "$POSTGRES_DB" -U "$POSTGRES_USER" -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | xargs || echo "N/A")
                unset PGPASSWORD
            fi
            
            print_status "HEALTHY" "$name" "Size: $db_size, Connections: $connections"
            log "INFO" "PostgreSQL health check passed: Size $db_size, Connections $connections"
            return 0
        else
            print_status "CRITICAL" "$name" "Connection failed"
            log "ERROR" "PostgreSQL health check failed: Cannot connect"
            send_alert "CRITICAL" "$name" "Connection failed"
            return 1
        fi
    else
        print_status "WARNING" "$name" "pg_isready not available"
        log "WARNING" "PostgreSQL health check skipped: pg_isready not installed"
        return 1
    fi
}

# Check Docker containers
check_docker_containers() {
    local name="Docker Containers"
    
    if command -v docker >/dev/null 2>&1; then
        local containers
        containers=$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "")
        
        if [[ -n "$containers" ]]; then
            local running_count
            running_count=$(docker ps -q | wc -l)
            local total_count
            total_count=$(docker ps -a -q | wc -l)
            
            print_status "INFO" "$name" "Running: $running_count/$total_count containers"
            log "INFO" "Docker containers: $running_count/$total_count running"
            
            # Check for unhealthy containers
            local unhealthy
            unhealthy=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" 2>/dev/null || echo "")
            
            if [[ -n "$unhealthy" ]]; then
                print_status "WARNING" "$name" "Unhealthy containers: $unhealthy"
                log "WARNING" "Unhealthy Docker containers: $unhealthy"
                send_alert "WARNING" "$name" "Unhealthy containers: $unhealthy"
                return 1
            fi
            
            return 0
        else
            print_status "WARNING" "$name" "No containers found"
            log "WARNING" "No Docker containers found"
            return 1
        fi
    else
        print_status "WARNING" "$name" "Docker not available"
        log "WARNING" "Docker health check skipped: Docker not installed"
        return 1
    fi
}

# Check system resources
check_system_resources() {
    local name="System Resources"
    
    # CPU usage
    local cpu_usage
    if command -v top >/dev/null 2>&1; then
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
    else
        cpu_usage="N/A"
    fi
    
    # Memory usage
    local memory_usage
    if command -v free >/dev/null 2>&1; then
        memory_usage=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')
    else
        memory_usage="N/A"
    fi
    
    # Disk usage
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2{printf "%s", $5}')
    
    # Load average
    local load_avg
    if [[ -f /proc/loadavg ]]; then
        load_avg=$(cat /proc/loadavg | cut -d' ' -f1-3)
    else
        load_avg="N/A"
    fi
    
    print_status "INFO" "$name" "CPU: $cpu_usage, Memory: $memory_usage, Disk: $disk_usage, Load: $load_avg"
    log "INFO" "System resources - CPU: $cpu_usage, Memory: $memory_usage, Disk: $disk_usage, Load: $load_avg"
    
    # Check for critical resource usage
    local disk_percent
    disk_percent=$(echo "$disk_usage" | sed 's/%//')
    
    if [[ "$disk_percent" -gt 90 ]]; then
        print_status "CRITICAL" "$name" "Disk usage critical: $disk_usage"
        send_alert "CRITICAL" "$name" "Disk usage critical: $disk_usage"
        return 1
    elif [[ "$disk_percent" -gt 80 ]]; then
        print_status "WARNING" "$name" "Disk usage high: $disk_usage"
        send_alert "WARNING" "$name" "Disk usage high: $disk_usage"
        return 1
    fi
    
    return 0
}

# Check SSL certificates
check_ssl_certificates() {
    local name="SSL Certificates"
    local domain="${DOMAIN:-localhost}"
    
    if command -v openssl >/dev/null 2>&1; then
        local cert_file="/etc/letsencrypt/live/$domain/cert.pem"
        
        if [[ -f "$cert_file" ]]; then
            local expiry_date
            expiry_date=$(openssl x509 -enddate -noout -in "$cert_file" | cut -d= -f2)
            local expiry_epoch
            expiry_epoch=$(date -d "$expiry_date" +%s 2>/dev/null || echo "0")
            local current_epoch
            current_epoch=$(date +%s)
            local days_until_expiry
            days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
            
            if [[ "$days_until_expiry" -lt 7 ]]; then
                print_status "CRITICAL" "$name" "Certificate expires in $days_until_expiry days"
                send_alert "CRITICAL" "$name" "Certificate expires in $days_until_expiry days"
                return 1
            elif [[ "$days_until_expiry" -lt 30 ]]; then
                print_status "WARNING" "$name" "Certificate expires in $days_until_expiry days"
                send_alert "WARNING" "$name" "Certificate expires in $days_until_expiry days"
                return 1
            else
                print_status "HEALTHY" "$name" "Certificate valid for $days_until_expiry days"
                log "INFO" "SSL certificate valid for $days_until_expiry days"
                return 0
            fi
        else
            print_status "WARNING" "$name" "Certificate file not found"
            log "WARNING" "SSL certificate file not found: $cert_file"
            return 1
        fi
    else
        print_status "WARNING" "$name" "OpenSSL not available"
        log "WARNING" "SSL check skipped: OpenSSL not installed"
        return 1
    fi
}

# Check log files for errors
check_logs() {
    local name="Application Logs"
    local log_dir="/var/log/stock-prediction"
    local error_count=0
    
    if [[ -d "$log_dir" ]]; then
        # Check for recent errors (last 5 minutes)
        local recent_errors
        recent_errors=$(find "$log_dir" -name "*.log" -type f -exec grep -l "ERROR\|CRITICAL\|FATAL" {} \; 2>/dev/null | wc -l)
        
        if [[ "$recent_errors" -gt 0 ]]; then
            # Count actual error lines from the last 5 minutes
            local five_min_ago
            five_min_ago=$(date -d '5 minutes ago' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -v-5M '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "")
            
            if [[ -n "$five_min_ago" ]]; then
                error_count=$(find "$log_dir" -name "*.log" -type f -exec awk -v since="$five_min_ago" '$0 >= since && /ERROR|CRITICAL|FATAL/ {count++} END {print count+0}' {} \; 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
            fi
            
            if [[ "$error_count" -gt 10 ]]; then
                print_status "CRITICAL" "$name" "$error_count errors in last 5 minutes"
                send_alert "CRITICAL" "$name" "$error_count errors in last 5 minutes"
                return 1
            elif [[ "$error_count" -gt 0 ]]; then
                print_status "WARNING" "$name" "$error_count errors in last 5 minutes"
                send_alert "WARNING" "$name" "$error_count errors in last 5 minutes"
                return 1
            fi
        fi
        
        print_status "HEALTHY" "$name" "No recent errors detected"
        log "INFO" "Log analysis: No recent errors detected"
        return 0
    else
        print_status "WARNING" "$name" "Log directory not found"
        log "WARNING" "Log directory not found: $log_dir"
        return 1
    fi
}

# Main health check function
main() {
    local exit_code=0
    local start_time
    start_time=$(date +%s)
    
    echo "=========================================="
    echo "Stock Prediction System Health Check"
    echo "Time: $(date)"
    echo "Host: $(hostname)"
    echo "=========================================="
    
    log "INFO" "Starting health check"
    
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Run all health checks
    check_system_resources || exit_code=1
    echo
    
    check_docker_containers || exit_code=1
    echo
    
    check_http_endpoint "Backend API" "$BACKEND_URL/health" || exit_code=1
    check_http_endpoint "Frontend" "$FRONTEND_URL" || exit_code=1
    check_http_endpoint "Prometheus" "$PROMETHEUS_URL/-/healthy" || exit_code=1
    check_http_endpoint "Grafana" "$GRAFANA_URL/api/health" || exit_code=1
    echo
    
    check_redis || exit_code=1
    check_postgres || exit_code=1
    echo
    
    check_ssl_certificates || exit_code=1
    echo
    
    check_logs || exit_code=1
    echo
    
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$((end_time - start_time))
    
    echo "=========================================="
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}✓ All health checks passed${NC}"
        log "INFO" "Health check completed successfully in ${duration}s"
    else
        echo -e "${RED}✗ Some health checks failed${NC}"
        log "ERROR" "Health check completed with failures in ${duration}s"
    fi
    echo "Duration: ${duration}s"
    echo "=========================================="
    
    exit $exit_code
}

# Handle script arguments
case "${1:-}" in
    "--help"|-h)
        echo "Usage: $0 [--help|--version|--quiet]"
        echo "  --help     Show this help message"
        echo "  --version  Show version information"
        echo "  --quiet    Run in quiet mode (errors only)"
        exit 0
        ;;
    "--version"|-v)
        echo "Stock Prediction System Health Check v1.0.0"
        exit 0
        ;;
    "--quiet"|-q)
        exec > /dev/null
        ;;
esac

# Run main function
main "$@"