#!/bin/bash

# Stock Prediction System Deployment Script
# This script helps deploy the application on AWS EC2 with SSL

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DOMAIN_NAME="STOCK_MARKET_PREDICTION.com"
EMAIL="pateldvldilipkumar@gmail.com"
APP_NAME="stock-prediction"

echo -e "${GREEN}🚀 Stock Prediction System Deployment Script${NC}"
echo "================================================"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
print_status "Installing Docker and Docker Compose..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
else
    print_status "Docker already installed"
fi

if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    print_status "Docker Compose already installed"
fi

# Install Certbot for SSL certificates
print_status "Installing Certbot for SSL certificates..."
if ! command -v certbot &> /dev/null; then
    sudo apt install -y certbot python3-certbot-nginx
else
    print_status "Certbot already installed"
fi

# Create SSL certificates directory
print_status "Creating SSL certificates directory..."
sudo mkdir -p /etc/nginx/ssl

# Generate self-signed certificates for initial setup
print_status "Generating self-signed SSL certificates for initial setup..."
if [ ! -f "/etc/nginx/ssl/cert.pem" ]; then
    sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/key.pem \
        -out /etc/nginx/ssl/cert.pem \
        -subj "/C=IN/ST=State/L=City/O=Organization/CN=${DOMAIN_NAME}"
else
    print_status "SSL certificates already exist"
fi

# Create environment file from template
print_status "Creating environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Database Configuration
POSTGRES_DB=stock_prediction
POSTGRES_USER=stockuser
POSTGRES_PASSWORD=$(openssl rand -base64 32)
DATABASE_URL=postgresql://stockuser:$(openssl rand -base64 32)@postgres:5432/stock_prediction

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# API Keys (REPLACE WITH YOUR ACTUAL KEYS)
ALPHA_VANTAGE_API_KEY=YQKILNW2LG827ITH
ANGEL_ONE_API_KEY=sWspiKMS
ANGEL_ONE_CLIENT_ID=P59977856
ANGEL_ONE_PASSWORD=3010
ANGEL_ONE_TOTP_SECRET=WE2VO3P5PCRPOACOVHUS7BKZFE

# Application Configuration
SECRET_KEY=$(openssl rand -base64 32)
ENVIRONMENT=production
DEBUG=false
ALLOWED_HOSTS=${DOMAIN_NAME},localhost,127.0.0.1

# Monitoring
PROMETHEUS_RETENTION_TIME=15d
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)

# SSL Configuration
DOMAIN_NAME=${DOMAIN_NAME}
EMAIL=${EMAIL}
EOF
    print_warning "Please edit .env file with your actual API keys and configuration"
else
    print_status "Environment file already exists"
fi

# Create data directories
print_status "Creating data directories..."
mkdir -p data/postgres data/redis data/grafana logs models

# Set proper permissions
print_status "Setting proper permissions..."
sudo chown -R $USER:$USER .
chmod +x deploy.sh

# Build and start services
print_status "Building and starting services..."
docker-compose build
docker-compose up -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Check service health
print_status "Checking service health..."
docker-compose ps

# Setup Let's Encrypt SSL (if domain is configured)
if [ "$DOMAIN_NAME" != "your-domain.com" ]; then
    print_status "Setting up Let's Encrypt SSL certificate..."
    sudo certbot certonly --webroot -w /var/www/html -d $DOMAIN_NAME --email $EMAIL --agree-tos --non-interactive
    
    # Copy certificates to nginx directory
    sudo cp /etc/letsencrypt/live/$DOMAIN_NAME/fullchain.pem /etc/nginx/ssl/cert.pem
    sudo cp /etc/letsencrypt/live/$DOMAIN_NAME/privkey.pem /etc/nginx/ssl/key.pem
    
    # Restart nginx to use new certificates
    docker-compose restart nginx
    
    # Setup automatic renewal
    echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
else
    print_warning "Domain not configured. Using self-signed certificates."
    print_warning "Please update DOMAIN_NAME in this script and re-run for Let's Encrypt SSL."
fi

# Setup firewall
print_status "Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# Create backup script
print_status "Creating backup script..."
cat > backup.sh << 'EOF'
#!/bin/bash
# Backup script for Stock Prediction System

BACKUP_DIR="/home/$USER/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
docker-compose exec -T postgres pg_dump -U stockuser stock_prediction > $BACKUP_DIR/db_backup_$DATE.sql

# Backup models
tar -czf $BACKUP_DIR/models_backup_$DATE.tar.gz models/

# Backup logs
tar -czf $BACKUP_DIR/logs_backup_$DATE.tar.gz logs/

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*backup*" -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x backup.sh

# Setup daily backup cron job
echo "0 2 * * * /home/$USER/$(basename $PWD)/backup.sh" | crontab -

print_status "Deployment completed successfully! 🎉"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your actual API keys"
echo "2. Update DOMAIN_NAME in deploy.sh if you have a domain"
echo "3. Run 'docker-compose restart' after updating .env"
echo "4. Access your application at: https://${DOMAIN_NAME} (or https://your-server-ip)"
echo "5. Access Grafana dashboard at: https://${DOMAIN_NAME}/grafana"
echo "6. Monitor logs with: docker-compose logs -f"
echo ""
print_status "Backup script created and scheduled to run daily at 2 AM"
print_status "SSL certificates will auto-renew via cron job"