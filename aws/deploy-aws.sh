#!/bin/bash

# AWS Deployment Script for Stock Prediction System
# This script automates the complete deployment process using Terraform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$SCRIPT_DIR/terraform"
APP_NAME="stock-prediction"

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

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is not installed. Please install Terraform first."
        print_status "Visit: https://www.terraform.io/downloads.html"
        exit 1
    fi
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install AWS CLI first."
        print_status "Visit: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        print_status "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    print_status "All prerequisites met!"
}

# Function to validate configuration
validate_config() {
    print_header "Validating Configuration"
    
    if [ ! -f "$TERRAFORM_DIR/terraform.tfvars" ]; then
        print_error "terraform.tfvars file not found!"
        print_status "Please copy terraform.tfvars.example to terraform.tfvars and update with your values."
        exit 1
    fi
    
    # Check if required variables are set
    required_vars=("domain_name" "email" "db_password" "public_key")
    for var in "${required_vars[@]}"; do
        if ! grep -q "^$var" "$TERRAFORM_DIR/terraform.tfvars"; then
            print_error "Required variable '$var' not found in terraform.tfvars"
            exit 1
        fi
    done
    
    print_status "Configuration validated!"
}

# Function to build Docker images
build_docker_images() {
    print_header "Building Docker Images"
    
    cd "$PROJECT_ROOT"
    
    # Build backend image
    print_status "Building backend Docker image..."
    docker build -t stock-prediction/backend:latest ./backend
    
    # Build frontend image
    print_status "Building frontend Docker image..."
    docker build -t stock-prediction/frontend:latest ./frontend
    
    print_status "Docker images built successfully!"
}

# Function to push images to ECR (optional)
push_to_ecr() {
    print_header "Setting up ECR Repositories"
    
    AWS_REGION=$(grep '^aws_region' "$TERRAFORM_DIR/terraform.tfvars" | cut -d'=' -f2 | tr -d ' "')
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    
    # Create ECR repositories if they don't exist
    aws ecr describe-repositories --repository-names "$APP_NAME/backend" --region "$AWS_REGION" 2>/dev/null || \
        aws ecr create-repository --repository-name "$APP_NAME/backend" --region "$AWS_REGION"
    
    aws ecr describe-repositories --repository-names "$APP_NAME/frontend" --region "$AWS_REGION" 2>/dev/null || \
        aws ecr create-repository --repository-name "$APP_NAME/frontend" --region "$AWS_REGION"
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    
    # Tag and push images
    docker tag stock-prediction/backend:latest "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$APP_NAME/backend:latest"
    docker tag stock-prediction/frontend:latest "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$APP_NAME/frontend:latest"
    
    docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$APP_NAME/backend:latest"
    docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$APP_NAME/frontend:latest"
    
    print_status "Images pushed to ECR successfully!"
}

# Function to deploy infrastructure
deploy_infrastructure() {
    print_header "Deploying Infrastructure with Terraform"
    
    cd "$TERRAFORM_DIR"
    
    # Initialize Terraform
    print_status "Initializing Terraform..."
    terraform init
    
    # Plan deployment
    print_status "Planning deployment..."
    terraform plan -out=tfplan
    
    # Ask for confirmation
    echo
    read -p "Do you want to proceed with the deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled."
        exit 0
    fi
    
    # Apply deployment
    print_status "Applying deployment..."
    terraform apply tfplan
    
    # Save outputs
    terraform output -json > terraform-outputs.json
    
    print_status "Infrastructure deployed successfully!"
}

# Function to configure DNS (if using Route 53)
configure_dns() {
    print_header "Configuring DNS (Optional)"
    
    if [ -f "$TERRAFORM_DIR/terraform-outputs.json" ]; then
        LOAD_BALANCER_DNS=$(jq -r '.load_balancer_dns.value' "$TERRAFORM_DIR/terraform-outputs.json")
        LOAD_BALANCER_ZONE_ID=$(jq -r '.load_balancer_zone_id.value' "$TERRAFORM_DIR/terraform-outputs.json")
        DOMAIN_NAME=$(grep '^domain_name' "$TERRAFORM_DIR/terraform.tfvars" | cut -d'=' -f2 | tr -d ' "')
        
        print_status "Load Balancer DNS: $LOAD_BALANCER_DNS"
        print_status "Load Balancer Zone ID: $LOAD_BALANCER_ZONE_ID"
        print_status "Domain Name: $DOMAIN_NAME"
        
        echo
        print_warning "Please configure your DNS to point $DOMAIN_NAME to $LOAD_BALANCER_DNS"
        print_warning "If using Route 53, create an A record (alias) pointing to the load balancer."
    fi
}

# Function to wait for deployment
wait_for_deployment() {
    print_header "Waiting for Deployment to Complete"
    
    if [ -f "$TERRAFORM_DIR/terraform-outputs.json" ]; then
        LOAD_BALANCER_DNS=$(jq -r '.load_balancer_dns.value' "$TERRAFORM_DIR/terraform-outputs.json")
        
        print_status "Waiting for load balancer to be ready..."
        
        # Wait for load balancer to respond
        for i in {1..30}; do
            if curl -f "http://$LOAD_BALANCER_DNS/health" &> /dev/null; then
                print_status "Application is ready!"
                break
            fi
            
            if [ $i -eq 30 ]; then
                print_warning "Application may still be starting up. Check the instances manually."
            else
                print_status "Waiting... ($i/30)"
                sleep 30
            fi
        done
    fi
}

# Function to setup SSL certificate
setup_ssl() {
    print_header "SSL Certificate Setup"
    
    print_warning "SSL certificate setup requires manual intervention on the EC2 instances."
    print_status "After DNS propagation, SSH into the instances and run:"
    print_status "sudo certbot --nginx -d your-domain.com"
    
    if [ -f "$TERRAFORM_DIR/terraform-outputs.json" ]; then
        ASG_NAME=$(jq -r '.autoscaling_group_name.value' "$TERRAFORM_DIR/terraform-outputs.json")
        print_status "Auto Scaling Group: $ASG_NAME"
        print_status "Use AWS Console to find instance IPs for SSH access."
    fi
}

# Function to display deployment summary
show_deployment_summary() {
    print_header "Deployment Summary"
    
    if [ -f "$TERRAFORM_DIR/terraform-outputs.json" ]; then
        echo
        print_status "Deployment completed successfully! 🎉"
        echo
        
        # Display key information
        LOAD_BALANCER_DNS=$(jq -r '.load_balancer_dns.value' "$TERRAFORM_DIR/terraform-outputs.json")
        DOMAIN_NAME=$(grep '^domain_name' "$TERRAFORM_DIR/terraform.tfvars" | cut -d'=' -f2 | tr -d ' "')
        
        echo "📊 Application URLs:"
        echo "   HTTP:  http://$LOAD_BALANCER_DNS"
        echo "   HTTPS: https://$DOMAIN_NAME (after SSL setup)"
        echo
        
        echo "🔧 Next Steps:"
        echo "   1. Configure DNS to point $DOMAIN_NAME to $LOAD_BALANCER_DNS"
        echo "   2. Wait for DNS propagation (5-30 minutes)"
        echo "   3. SSH into instances and setup SSL certificates"
        echo "   4. Monitor application health and performance"
        echo
        
        echo "📈 Monitoring:"
        echo "   Grafana: https://$DOMAIN_NAME/grafana (after SSL setup)"
        echo "   Prometheus: Available internally on port 9090"
        echo
        
        echo "🔍 Troubleshooting:"
        echo "   - Check Auto Scaling Group health in AWS Console"
        echo "   - View CloudWatch logs for application logs"
        echo "   - SSH into instances to check Docker containers"
        echo
        
        # Display Terraform outputs
        echo "📋 Terraform Outputs:"
        terraform -chdir="$TERRAFORM_DIR" output
    else
        print_error "Deployment outputs not found. Check for errors above."
    fi
}

# Function to cleanup resources
cleanup() {
    print_header "Cleanup Resources"
    
    cd "$TERRAFORM_DIR"
    
    echo
    read -p "Are you sure you want to destroy all resources? This cannot be undone! (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Destroying infrastructure..."
        terraform destroy -auto-approve
        print_status "Resources destroyed successfully!"
    else
        print_warning "Cleanup cancelled."
    fi
}

# Main function
main() {
    print_header "AWS Deployment for Stock Prediction System"
    
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            validate_config
            build_docker_images
            # push_to_ecr  # Uncomment if using ECR
            deploy_infrastructure
            configure_dns
            wait_for_deployment
            setup_ssl
            show_deployment_summary
            ;;
        "destroy")
            cleanup
            ;;
        "status")
            if [ -f "$TERRAFORM_DIR/terraform-outputs.json" ]; then
                show_deployment_summary
            else
                print_error "No deployment found. Run './deploy-aws.sh deploy' first."
            fi
            ;;
        "help")
            echo "Usage: $0 [deploy|destroy|status|help]"
            echo
            echo "Commands:"
            echo "  deploy   - Deploy the complete infrastructure (default)"
            echo "  destroy  - Destroy all resources"
            echo "  status   - Show deployment status and URLs"
            echo "  help     - Show this help message"
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Run '$0 help' for usage information."
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"