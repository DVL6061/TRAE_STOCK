#!/bin/bash

# User Data Script for Stock Prediction System EC2 Instances
# This script automatically sets up the application on new instances

set -e

# Variables from Terraform
DOMAIN_NAME="${domain_name}"
EMAIL="${email}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a /var/log/stock-app-setup.log
}

log "Starting Stock Prediction System setup..."

# Update system packages
log "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install required packages
log "Installing required packages..."
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    unzip \
    htop \
    nginx \
    certbot \
    python3-certbot-nginx

# Install Docker
log "Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu
    rm get-docker.sh
else
    log "Docker already installed"
fi

# Install Docker Compose
log "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
else
    log "Docker Compose already installed"
fi

# Create application directory
log "Creating application directory..."
mkdir -p /opt/stock-prediction
cd /opt/stock-prediction

# Clone or download application code (assuming it's in a Git repository)
# For now, we'll create the necessary structure
log "Setting up application structure..."

# Create docker-compose.yml for production
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # Backend API Service
  backend:
    image: stock-prediction/backend:latest
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - ANGEL_API_KEY=${ANGEL_API_KEY}
      - ANGEL_CLIENT_ID=${ANGEL_CLIENT_ID}
      - ANGEL_PASSWORD=${ANGEL_PASSWORD}
      - ANGEL_TOTP_SECRET=${ANGEL_TOTP_SECRET}
      - ALPHAVANTAGE_API_KEY=${ALPHAVANTAGE_API_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - ENVIRONMENT=production
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
    networks:
      - stock-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Frontend React Service
  frontend:
    image: stock-prediction/frontend:latest
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=https://${DOMAIN_NAME}/api
      - REACT_APP_WS_URL=wss://${DOMAIN_NAME}/ws
    restart: unless-stopped
    networks:
      - stock-network
    depends_on:
      - backend

  # Redis for caching
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - stock-network
    command: redis-server --appendonly yes

  # Prometheus for monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - stock-network

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_SERVER_ROOT_URL=https://${DOMAIN_NAME}/grafana
      - GF_SERVER_SERVE_FROM_SUB_PATH=true
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    restart: unless-stopped
    networks:
      - stock-network
    depends_on:
      - prometheus

volumes:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  stock-network:
    driver: bridge
EOF

# Create monitoring configuration
log "Setting up monitoring configuration..."
mkdir -p monitoring/grafana/{dashboards,datasources}

# Prometheus configuration
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'stock-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF

# Grafana datasource configuration
cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Create environment file from AWS Systems Manager Parameter Store or environment variables
log "Creating environment configuration..."
cat > .env << EOF
# Database Configuration (will be populated by Terraform outputs)
DATABASE_URL=postgresql://\${DB_USERNAME}:\${DB_PASSWORD}@\${DB_ENDPOINT}:5432/\${DB_NAME}

# Redis Configuration (will be populated by Terraform outputs)
REDIS_URL=redis://\${REDIS_ENDPOINT}:6379/0

# API Keys (from environment variables or Parameter Store)
ALPHA_VANTAGE_API_KEY=\${ALPHA_VANTAGE_API_KEY}
ANGEL_ONE_API_KEY=\${ANGEL_ONE_API_KEY}
ANGEL_ONE_CLIENT_ID=\${ANGEL_ONE_CLIENT_ID}
ANGEL_ONE_PASSWORD=\${ANGEL_ONE_PASSWORD}
ANGEL_ONE_TOTP_SECRET=\${ANGEL_ONE_TOTP_SECRET}

# Application Configuration
JWT_SECRET_KEY=\${JWT_SECRET_KEY}
ENVIRONMENT=production
DEBUG=false
DOMAIN_NAME=${DOMAIN_NAME}

# Monitoring
GRAFANA_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD}
EOF

# Create data directories
log "Creating data directories..."
mkdir -p logs models data/{historical,news}

# Set proper permissions
chown -R ubuntu:ubuntu /opt/stock-prediction
chmod +x /opt/stock-prediction

# Configure Nginx as reverse proxy
log "Configuring Nginx..."
cat > /etc/nginx/sites-available/stock-prediction << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name _;
    
    # SSL configuration (will be updated after certificate generation)
    ssl_certificate /etc/letsencrypt/live/DOMAIN_PLACEHOLDER/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/DOMAIN_PLACEHOLDER/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API routes
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket routes
    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
    
    # Grafana
    location /grafana/ {
        proxy_pass http://localhost:3001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Health check
    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
EOF

# Enable the site
ln -sf /etc/nginx/sites-available/stock-prediction /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
nginx -t

# Create systemd service for the application
log "Creating systemd service..."
cat > /etc/systemd/system/stock-prediction.service << 'EOF'
[Unit]
Description=Stock Prediction System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/stock-prediction
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
systemctl daemon-reload
systemctl enable stock-prediction.service

# Create health check script
log "Creating health check script..."
cat > /opt/stock-prediction/health-check.sh << 'EOF'
#!/bin/bash

# Health check script for Stock Prediction System

log_file="/var/log/stock-app-health.log"

log_health() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $log_file
}

# Check if containers are running
if ! docker-compose ps | grep -q "Up"; then
    log_health "ERROR: Some containers are not running. Restarting..."
    docker-compose restart
else
    log_health "INFO: All containers are running normally"
fi

# Check backend health endpoint
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    log_health "ERROR: Backend health check failed"
else
    log_health "INFO: Backend health check passed"
fi

# Check frontend
if ! curl -f http://localhost:3000 > /dev/null 2>&1; then
    log_health "ERROR: Frontend health check failed"
else
    log_health "INFO: Frontend health check passed"
fi
EOF

chmod +x /opt/stock-prediction/health-check.sh

# Setup cron job for health checks
echo "*/5 * * * * /opt/stock-prediction/health-check.sh" | crontab -u ubuntu -

# Create backup script
log "Creating backup script..."
cat > /opt/stock-prediction/backup.sh << 'EOF'
#!/bin/bash

# Backup script for Stock Prediction System

BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup models
tar -czf $BACKUP_DIR/models_backup_$DATE.tar.gz -C /opt/stock-prediction models/

# Backup logs
tar -czf $BACKUP_DIR/logs_backup_$DATE.tar.gz -C /opt/stock-prediction logs/

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*backup*" -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x /opt/stock-prediction/backup.sh

# Setup daily backup cron job
echo "0 2 * * * /opt/stock-prediction/backup.sh" | crontab -u ubuntu -

# Configure firewall
log "Configuring firewall..."
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Install CloudWatch agent for monitoring
log "Installing CloudWatch agent..."
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

# Create CloudWatch agent configuration
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
    "metrics": {
        "namespace": "StockPrediction/EC2",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/stock-app-setup.log",
                        "log_group_name": "stock-prediction-setup",
                        "log_stream_name": "{instance_id}"
                    },
                    {
                        "file_path": "/var/log/stock-app-health.log",
                        "log_group_name": "stock-prediction-health",
                        "log_stream_name": "{instance_id}"
                    },
                    {
                        "file_path": "/opt/stock-prediction/logs/*.log",
                        "log_group_name": "stock-prediction-app",
                        "log_stream_name": "{instance_id}"
                    }
                ]
            }
        }
    }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s

# Signal that the instance is ready
log "Instance setup completed successfully!"

# Create a status file
echo "READY" > /opt/stock-prediction/instance-status
echo "Setup completed at: $(date)" >> /opt/stock-prediction/instance-status

log "Stock Prediction System setup completed successfully!"