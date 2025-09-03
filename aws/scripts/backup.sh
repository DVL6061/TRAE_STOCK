#!/bin/bash

# Backup Script for Stock Prediction System
# This script creates backups of database, Redis data, and application files

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/stock-prediction}"
S3_BUCKET="${S3_BACKUP_BUCKET:-}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"
LOG_FILE="/var/log/stock-prediction/backup.log"
LOCK_FILE="/var/run/stock-prediction-backup.lock"
MAX_LOG_SIZE=10485760  # 10MB

# Database configuration
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-stock_prediction}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"

# Redis configuration
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

# Notification configuration
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
EMAIL_RECIPIENT="${BACKUP_EMAIL:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Print colored output
print_status() {
    local status=$1
    local message=$2
    
    case $status in
        "SUCCESS")
            echo -e "${GREEN}✓${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}✗${NC} $message"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ${NC} $message"
            ;;
    esac
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    local details="${3:-}"
    
    # Slack notification
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local color="good"
        case $status in
            "WARNING") color="warning" ;;
            "ERROR") color="danger" ;;
        esac
        
        local payload=$(cat <<EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "Stock Prediction System Backup",
            "fields": [
                {
                    "title": "Status",
                    "value": "$status",
                    "short": true
                },
                {
                    "title": "Message",
                    "value": "$message",
                    "short": false
                },
                {
                    "title": "Details",
                    "value": "$details",
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
            --data "$payload" "$SLACK_WEBHOOK" > /dev/null || true
    fi
    
    # Email notification
    if [[ -n "$EMAIL_RECIPIENT" ]] && command -v mail >/dev/null 2>&1; then
        local subject="Stock Prediction Backup - $status"
        local body="Backup Status: $status\n\nMessage: $message\n\nDetails: $details\n\nTimestamp: $(date)\nHost: $(hostname)"
        
        echo -e "$body" | mail -s "$subject" "$EMAIL_RECIPIENT" || true
    fi
}

# Check if another backup is running
check_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        local pid
        pid=$(cat "$LOCK_FILE")
        
        if kill -0 "$pid" 2>/dev/null; then
            log "ERROR" "Another backup process is already running (PID: $pid)"
            print_status "ERROR" "Another backup process is already running"
            exit 1
        else
            log "WARNING" "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    # Create lock file
    echo $$ > "$LOCK_FILE"
    
    # Remove lock file on exit
    trap 'rm -f "$LOCK_FILE"' EXIT
}

# Create backup directory structure
setup_backup_dir() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_path="$BACKUP_DIR/$timestamp"
    
    mkdir -p "$backup_path/database"
    mkdir -p "$backup_path/redis"
    mkdir -p "$backup_path/application"
    mkdir -p "$backup_path/logs"
    mkdir -p "$backup_path/config"
    
    echo "$backup_path"
}

# Backup PostgreSQL database
backup_postgres() {
    local backup_path=$1
    local db_backup_file="$backup_path/database/postgres_${POSTGRES_DB}.sql"
    
    print_status "INFO" "Starting PostgreSQL backup..."
    log "INFO" "Starting PostgreSQL backup to $db_backup_file"
    
    if [[ -z "$POSTGRES_PASSWORD" ]]; then
        log "ERROR" "PostgreSQL password not set"
        print_status "ERROR" "PostgreSQL password not configured"
        return 1
    fi
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Create database dump
    if pg_dump -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" \
               -d "$POSTGRES_DB" --verbose --no-password \
               --format=custom --compress="$COMPRESSION_LEVEL" \
               --file="$db_backup_file" 2>>"$LOG_FILE"; then
        
        local file_size
        file_size=$(du -h "$db_backup_file" | cut -f1)
        
        print_status "SUCCESS" "PostgreSQL backup completed ($file_size)"
        log "INFO" "PostgreSQL backup completed: $db_backup_file ($file_size)"
        
        # Create schema-only backup for quick restore testing
        local schema_file="$backup_path/database/postgres_${POSTGRES_DB}_schema.sql"
        pg_dump -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" \
                -d "$POSTGRES_DB" --schema-only --no-password \
                --file="$schema_file" 2>>"$LOG_FILE" || true
        
        unset PGPASSWORD
        return 0
    else
        unset PGPASSWORD
        print_status "ERROR" "PostgreSQL backup failed"
        log "ERROR" "PostgreSQL backup failed"
        return 1
    fi
}

# Backup Redis data
backup_redis() {
    local backup_path=$1
    local redis_backup_file="$backup_path/redis/redis_dump.rdb"
    
    print_status "INFO" "Starting Redis backup..."
    log "INFO" "Starting Redis backup to $redis_backup_file"
    
    # Try to get Redis dump
    local redis_cmd="redis-cli -h $REDIS_HOST -p $REDIS_PORT"
    
    if [[ -n "$REDIS_PASSWORD" ]]; then
        redis_cmd="$redis_cmd -a $REDIS_PASSWORD"
    fi
    
    # Force Redis to save current state
    if $redis_cmd BGSAVE >/dev/null 2>&1; then
        # Wait for background save to complete
        local save_in_progress=1
        local timeout=300  # 5 minutes timeout
        local elapsed=0
        
        while [[ $save_in_progress -eq 1 ]] && [[ $elapsed -lt $timeout ]]; do
            if $redis_cmd LASTSAVE >/dev/null 2>&1; then
                sleep 2
                elapsed=$((elapsed + 2))
                
                # Check if BGSAVE is still running
                if ! $redis_cmd INFO persistence | grep -q "rdb_bgsave_in_progress:1"; then
                    save_in_progress=0
                fi
            else
                break
            fi
        done
        
        # Copy the RDB file
        local redis_data_dir="/var/lib/redis"
        if [[ -f "$redis_data_dir/dump.rdb" ]]; then
            cp "$redis_data_dir/dump.rdb" "$redis_backup_file"
            
            local file_size
            file_size=$(du -h "$redis_backup_file" | cut -f1)
            
            print_status "SUCCESS" "Redis backup completed ($file_size)"
            log "INFO" "Redis backup completed: $redis_backup_file ($file_size)"
            return 0
        else
            print_status "WARNING" "Redis dump file not found"
            log "WARNING" "Redis dump file not found at $redis_data_dir/dump.rdb"
            return 1
        fi
    else
        print_status "ERROR" "Redis backup failed"
        log "ERROR" "Redis BGSAVE command failed"
        return 1
    fi
}

# Backup application files
backup_application() {
    local backup_path=$1
    local app_backup_file="$backup_path/application/application_files.tar.gz"
    
    print_status "INFO" "Starting application files backup..."
    log "INFO" "Starting application files backup to $app_backup_file"
    
    # Define directories to backup
    local app_dirs=(
        "/opt/stock-prediction"
        "/etc/stock-prediction"
        "/var/lib/stock-prediction"
    )
    
    local existing_dirs=()
    for dir in "${app_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            existing_dirs+=("$dir")
        fi
    done
    
    if [[ ${#existing_dirs[@]} -eq 0 ]]; then
        print_status "WARNING" "No application directories found to backup"
        log "WARNING" "No application directories found"
        return 1
    fi
    
    # Create tar archive
    if tar -czf "$app_backup_file" "${existing_dirs[@]}" 2>>"$LOG_FILE"; then
        local file_size
        file_size=$(du -h "$app_backup_file" | cut -f1)
        
        print_status "SUCCESS" "Application files backup completed ($file_size)"
        log "INFO" "Application files backup completed: $app_backup_file ($file_size)"
        return 0
    else
        print_status "ERROR" "Application files backup failed"
        log "ERROR" "Application files backup failed"
        return 1
    fi
}

# Backup logs
backup_logs() {
    local backup_path=$1
    local logs_backup_file="$backup_path/logs/application_logs.tar.gz"
    
    print_status "INFO" "Starting logs backup..."
    log "INFO" "Starting logs backup to $logs_backup_file"
    
    local log_dirs=(
        "/var/log/stock-prediction"
        "/var/log/nginx"
        "/var/log/docker"
    )
    
    local existing_log_dirs=()
    for dir in "${log_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            existing_log_dirs+=("$dir")
        fi
    done
    
    if [[ ${#existing_log_dirs[@]} -eq 0 ]]; then
        print_status "WARNING" "No log directories found to backup"
        log "WARNING" "No log directories found"
        return 1
    fi
    
    # Create tar archive (exclude current backup log)
    if tar -czf "$logs_backup_file" "${existing_log_dirs[@]}" \
           --exclude="$LOG_FILE" 2>>"$LOG_FILE"; then
        
        local file_size
        file_size=$(du -h "$logs_backup_file" | cut -f1)
        
        print_status "SUCCESS" "Logs backup completed ($file_size)"
        log "INFO" "Logs backup completed: $logs_backup_file ($file_size)"
        return 0
    else
        print_status "ERROR" "Logs backup failed"
        log "ERROR" "Logs backup failed"
        return 1
    fi
}

# Backup configuration files
backup_config() {
    local backup_path=$1
    local config_backup_file="$backup_path/config/configuration.tar.gz"
    
    print_status "INFO" "Starting configuration backup..."
    log "INFO" "Starting configuration backup to $config_backup_file"
    
    local config_files=(
        "/etc/nginx/nginx.conf"
        "/etc/letsencrypt"
        "/opt/stock-prediction/docker-compose.yml"
        "/opt/stock-prediction/.env"
        "/etc/prometheus"
        "/etc/grafana"
    )
    
    local existing_configs=()
    for item in "${config_files[@]}"; do
        if [[ -e "$item" ]]; then
            existing_configs+=("$item")
        fi
    done
    
    if [[ ${#existing_configs[@]} -eq 0 ]]; then
        print_status "WARNING" "No configuration files found to backup"
        log "WARNING" "No configuration files found"
        return 1
    fi
    
    # Create tar archive
    if tar -czf "$config_backup_file" "${existing_configs[@]}" 2>>"$LOG_FILE"; then
        local file_size
        file_size=$(du -h "$config_backup_file" | cut -f1)
        
        print_status "SUCCESS" "Configuration backup completed ($file_size)"
        log "INFO" "Configuration backup completed: $config_backup_file ($file_size)"
        return 0
    else
        print_status "ERROR" "Configuration backup failed"
        log "ERROR" "Configuration backup failed"
        return 1
    fi
}

# Upload to S3
upload_to_s3() {
    local backup_path=$1
    
    if [[ -z "$S3_BUCKET" ]]; then
        print_status "INFO" "S3 upload skipped (no bucket configured)"
        return 0
    fi
    
    if ! command -v aws >/dev/null 2>&1; then
        print_status "WARNING" "AWS CLI not installed, skipping S3 upload"
        log "WARNING" "AWS CLI not available for S3 upload"
        return 1
    fi
    
    print_status "INFO" "Starting S3 upload..."
    log "INFO" "Starting S3 upload to s3://$S3_BUCKET"
    
    local backup_name
    backup_name=$(basename "$backup_path")
    local s3_path="s3://$S3_BUCKET/stock-prediction-backups/$backup_name"
    
    # Create archive of entire backup
    local archive_file="${backup_path}.tar.gz"
    if tar -czf "$archive_file" -C "$(dirname "$backup_path")" "$backup_name"; then
        
        # Upload to S3
        if aws s3 cp "$archive_file" "$s3_path.tar.gz" \
               --storage-class STANDARD_IA 2>>"$LOG_FILE"; then
            
            local file_size
            file_size=$(du -h "$archive_file" | cut -f1)
            
            print_status "SUCCESS" "S3 upload completed ($file_size)"
            log "INFO" "S3 upload completed: $s3_path.tar.gz ($file_size)"
            
            # Remove local archive after successful upload
            rm -f "$archive_file"
            return 0
        else
            print_status "ERROR" "S3 upload failed"
            log "ERROR" "S3 upload failed"
            rm -f "$archive_file"
            return 1
        fi
    else
        print_status "ERROR" "Failed to create backup archive"
        log "ERROR" "Failed to create backup archive"
        return 1
    fi
}

# Clean old backups
cleanup_old_backups() {
    print_status "INFO" "Cleaning up old backups..."
    log "INFO" "Cleaning up backups older than $RETENTION_DAYS days"
    
    # Clean local backups
    local deleted_count=0
    if [[ -d "$BACKUP_DIR" ]]; then
        while IFS= read -r -d '' backup_dir; do
            rm -rf "$backup_dir"
            deleted_count=$((deleted_count + 1))
        done < <(find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +"$RETENTION_DAYS" -print0 2>/dev/null)
    fi
    
    # Clean S3 backups
    if [[ -n "$S3_BUCKET" ]] && command -v aws >/dev/null 2>&1; then
        local cutoff_date
        cutoff_date=$(date -d "$RETENTION_DAYS days ago" '+%Y-%m-%d' 2>/dev/null || date -v-"${RETENTION_DAYS}d" '+%Y-%m-%d' 2>/dev/null || echo "")
        
        if [[ -n "$cutoff_date" ]]; then
            aws s3 ls "s3://$S3_BUCKET/stock-prediction-backups/" 2>/dev/null | \
            while read -r line; do
                local file_date
                file_date=$(echo "$line" | awk '{print $1}')
                local file_name
                file_name=$(echo "$line" | awk '{print $4}')
                
                if [[ "$file_date" < "$cutoff_date" ]] && [[ -n "$file_name" ]]; then
                    aws s3 rm "s3://$S3_BUCKET/stock-prediction-backups/$file_name" 2>>"$LOG_FILE" || true
                    deleted_count=$((deleted_count + 1))
                fi
            done
        fi
    fi
    
    print_status "SUCCESS" "Cleanup completed ($deleted_count items removed)"
    log "INFO" "Cleanup completed: $deleted_count items removed"
}

# Create backup manifest
create_manifest() {
    local backup_path=$1
    local manifest_file="$backup_path/backup_manifest.json"
    
    local backup_name
    backup_name=$(basename "$backup_path")
    
    cat > "$manifest_file" <<EOF
{
    "backup_name": "$backup_name",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "hostname": "$(hostname)",
    "version": "1.0.0",
    "retention_days": $RETENTION_DAYS,
    "components": {
        "database": {
            "postgres_host": "$POSTGRES_HOST",
            "postgres_db": "$POSTGRES_DB",
            "postgres_user": "$POSTGRES_USER"
        },
        "redis": {
            "redis_host": "$REDIS_HOST",
            "redis_port": "$REDIS_PORT"
        },
        "s3_bucket": "$S3_BUCKET"
    },
    "files": {
EOF
    
    # Add file information
    find "$backup_path" -type f -not -name "backup_manifest.json" | while read -r file; do
        local rel_path
        rel_path=$(realpath --relative-to="$backup_path" "$file")
        local file_size
        file_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0")
        local file_hash
        file_hash=$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1 || echo "unknown")
        
        echo "        \"$rel_path\": {" >> "$manifest_file"
        echo "            \"size\": $file_size," >> "$manifest_file"
        echo "            \"sha256\": \"$file_hash\"" >> "$manifest_file"
        echo "        }," >> "$manifest_file"
    done
    
    # Remove trailing comma and close JSON
    sed -i '$ s/,$//' "$manifest_file"
    echo "    }" >> "$manifest_file"
    echo "}" >> "$manifest_file"
    
    log "INFO" "Backup manifest created: $manifest_file"
}

# Main backup function
main() {
    local start_time
    start_time=$(date +%s)
    local exit_code=0
    local backup_path
    
    echo "=========================================="
    echo "Stock Prediction System Backup"
    echo "Time: $(date)"
    echo "Host: $(hostname)"
    echo "=========================================="
    
    log "INFO" "Starting backup process"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Check for running backup
    check_lock
    
    # Setup backup directory
    backup_path=$(setup_backup_dir)
    log "INFO" "Backup directory: $backup_path"
    
    # Perform backups
    backup_postgres "$backup_path" || exit_code=1
    backup_redis "$backup_path" || exit_code=1
    backup_application "$backup_path" || exit_code=1
    backup_logs "$backup_path" || exit_code=1
    backup_config "$backup_path" || exit_code=1
    
    # Create manifest
    create_manifest "$backup_path"
    
    # Upload to S3
    upload_to_s3 "$backup_path" || exit_code=1
    
    # Cleanup old backups
    cleanup_old_backups
    
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$((end_time - start_time))
    
    local total_size
    total_size=$(du -sh "$backup_path" | cut -f1)
    
    echo "=========================================="
    if [[ $exit_code -eq 0 ]]; then
        print_status "SUCCESS" "Backup completed successfully"
        log "INFO" "Backup completed successfully in ${duration}s (Size: $total_size)"
        send_notification "SUCCESS" "Backup completed successfully" "Duration: ${duration}s, Size: $total_size, Path: $backup_path"
    else
        print_status "ERROR" "Backup completed with errors"
        log "ERROR" "Backup completed with errors in ${duration}s"
        send_notification "ERROR" "Backup completed with errors" "Duration: ${duration}s, Path: $backup_path"
    fi
    echo "Duration: ${duration}s"
    echo "Size: $total_size"
    echo "Path: $backup_path"
    echo "=========================================="
    
    exit $exit_code
}

# Handle script arguments
case "${1:-}" in
    "--help"|-h)
        echo "Usage: $0 [--help|--version|--dry-run]"
        echo "  --help     Show this help message"
        echo "  --version  Show version information"
        echo "  --dry-run  Show what would be backed up without doing it"
        exit 0
        ;;
    "--version"|-v)
        echo "Stock Prediction System Backup v1.0.0"
        exit 0
        ;;
    "--dry-run")
        echo "Dry run mode - showing what would be backed up:"
        echo "- PostgreSQL database: $POSTGRES_DB"
        echo "- Redis data from: $REDIS_HOST:$REDIS_PORT"
        echo "- Application files"
        echo "- System logs"
        echo "- Configuration files"
        if [[ -n "$S3_BUCKET" ]]; then
            echo "- Upload to S3: s3://$S3_BUCKET"
        fi
        echo "- Retention: $RETENTION_DAYS days"
        exit 0
        ;;
esac

# Run main function
main "$@"