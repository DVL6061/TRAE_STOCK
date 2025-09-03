# AWS Deployment Guide

This guide provides step-by-step instructions for deploying the Stock Prediction System on AWS using Terraform and automated deployment scripts.

## 🏗️ Architecture Overview

The deployment creates a highly available, scalable infrastructure on AWS:

- **VPC**: Custom VPC with public and private subnets across multiple AZs
- **Load Balancer**: Application Load Balancer with SSL termination
- **Auto Scaling**: EC2 instances with auto-scaling based on CPU utilization
- **Database**: RDS PostgreSQL with Multi-AZ deployment
- **Cache**: ElastiCache Redis cluster
- **Monitoring**: Prometheus, Grafana, and CloudWatch integration
- **Security**: Security groups, NACLs, and encrypted storage

## 📋 Prerequisites

Before starting the deployment, ensure you have:

### 1. Required Tools
- **AWS CLI** (v2.0+): [Installation Guide](https://aws.amazon.com/cli/)
- **Terraform** (v1.0+): [Installation Guide](https://www.terraform.io/downloads.html)
- **Docker** (v20.0+): [Installation Guide](https://docs.docker.com/get-docker/)
- **Git**: For cloning the repository
- **jq**: For JSON processing (optional but recommended)

### 2. AWS Account Setup
- AWS account with appropriate permissions
- AWS CLI configured with access keys
- Domain name (optional, for custom domain)
- Route 53 hosted zone (if using custom domain)

### 3. Required AWS Permissions
Your AWS user/role needs permissions for:
- EC2 (instances, security groups, load balancers)
- VPC (subnets, route tables, internet gateways)
- RDS (database instances, parameter groups)
- ElastiCache (Redis clusters)
- IAM (roles, policies)
- CloudWatch (logs, metrics)
- Auto Scaling
- Certificate Manager (for SSL)

## 🚀 Quick Start

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd TRAE_STOCK/aws

# Make deployment script executable
chmod +x deploy-aws.sh
```

### Step 2: Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Verify credentials
aws sts get-caller-identity
```

### Step 3: Configure Terraform Variables

```bash
# Copy the example configuration
cp terraform/terraform.tfvars.example terraform/terraform.tfvars

# Edit the configuration file
nano terraform/terraform.tfvars
```

**Required Configuration:**

```hcl
# Basic Configuration
aws_region = "us-east-1"
project_name = "stock-prediction"
environment = "production"

# Domain Configuration (optional)
domain_name = "your-domain.com"
email = "your-email@domain.com"

# Infrastructure Settings
instance_type = "t3.medium"
min_size = 2
max_size = 10
desired_capacity = 2

# Database Configuration
db_instance_class = "db.t3.micro"
db_name = "stockprediction"
db_username = "stockuser"
db_password = "your-secure-password-here"

# SSH Access
public_key = "ssh-rsa AAAAB3NzaC1yc2E... your-public-key"
ssh_cidr_blocks = ["0.0.0.0/0"]  # Restrict this to your IP

# API Keys (get from respective providers)
alpha_vantage_api_key = "your-alpha-vantage-key"
angel_one_api_key = "your-angel-one-key"
angel_one_client_id = "your-angel-one-client-id"
angel_one_password = "your-angel-one-password"
angel_one_totp_secret = "your-angel-one-totp-secret"

# Application Secrets
jwt_secret_key = "your-jwt-secret-key"
redis_password = "your-redis-password"
grafana_admin_password = "your-grafana-password"
```

### Step 4: Deploy Infrastructure

```bash
# Run the deployment script
./deploy-aws.sh deploy
```

The script will:
1. ✅ Check prerequisites
2. ✅ Validate configuration
3. 🐳 Build Docker images
4. 🏗️ Deploy infrastructure with Terraform
5. 🌐 Configure DNS settings
6. ⏳ Wait for deployment completion
7. 📊 Display deployment summary

## 📖 Detailed Configuration

### Infrastructure Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `aws_region` | AWS region for deployment | `us-east-1` | Yes |
| `project_name` | Project name for resource naming | `stock-prediction` | Yes |
| `environment` | Environment name (prod/staging/dev) | `production` | Yes |
| `domain_name` | Custom domain name | `""` | No |
| `email` | Email for SSL certificates | `""` | No |

### Compute Configuration

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `instance_type` | EC2 instance type | `t3.medium` | Adjust based on load |
| `min_size` | Minimum instances | `2` | For high availability |
| `max_size` | Maximum instances | `10` | Scale limit |
| `desired_capacity` | Initial instance count | `2` | Starting capacity |

### Database Configuration

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `db_instance_class` | RDS instance type | `db.t3.micro` | Upgrade for production |
| `db_name` | Database name | `stockprediction` | Application database |
| `db_username` | Database username | `stockuser` | Admin user |
| `db_password` | Database password | - | **Required, use strong password** |

### Security Configuration

| Variable | Description | Notes |
|----------|-------------|-------|
| `public_key` | SSH public key | For EC2 access |
| `ssh_cidr_blocks` | SSH access IPs | Restrict to your IP |
| `jwt_secret_key` | JWT signing key | Generate secure key |
| `redis_password` | Redis password | For cache security |

## 🔧 Post-Deployment Setup

### 1. DNS Configuration

After deployment, configure your domain:

**Option A: Route 53 (Recommended)**
```bash
# Get load balancer DNS from Terraform output
terraform -chdir=terraform output load_balancer_dns

# Create Route 53 A record (alias) pointing to the load balancer
```

**Option B: External DNS Provider**
- Create CNAME record pointing your domain to the load balancer DNS
- Wait for DNS propagation (5-30 minutes)

### 2. SSL Certificate Setup

```bash
# SSH into EC2 instances
ssh -i your-key.pem ec2-user@<instance-ip>

# Setup Let's Encrypt SSL
sudo certbot --nginx -d your-domain.com

# Verify SSL renewal
sudo certbot renew --dry-run
```

### 3. Application Verification

```bash
# Check application health
curl http://<load-balancer-dns>/health

# Verify API endpoints
curl http://<load-balancer-dns>/api/v1/health

# Access monitoring
# Grafana: https://your-domain.com/grafana
# Username: admin
# Password: <grafana_admin_password>
```

## 📊 Monitoring and Maintenance

### CloudWatch Dashboards
- **Application Metrics**: CPU, memory, request rates
- **Database Metrics**: Connections, query performance
- **Infrastructure Metrics**: Load balancer health, auto-scaling events

### Grafana Dashboards
- **System Overview**: Infrastructure health
- **Application Performance**: Response times, error rates
- **Business Metrics**: Prediction accuracy, user activity

### Log Management
- **Application Logs**: `/var/log/stock-prediction/`
- **Nginx Logs**: `/var/log/nginx/`
- **CloudWatch Logs**: Centralized log aggregation

### Backup Strategy
- **Database**: Automated RDS snapshots (7-day retention)
- **Application Data**: Daily backups to S3
- **Configuration**: Terraform state backup

## 🔍 Troubleshooting

### Common Issues

**1. Deployment Fails**
```bash
# Check Terraform logs
terraform -chdir=terraform plan

# Verify AWS credentials
aws sts get-caller-identity

# Check resource limits
aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A
```

**2. Application Not Accessible**
```bash
# Check load balancer health
aws elbv2 describe-target-health --target-group-arn <target-group-arn>

# Check security groups
aws ec2 describe-security-groups --group-ids <security-group-id>

# SSH into instance and check Docker
ssh -i your-key.pem ec2-user@<instance-ip>
sudo docker ps
sudo docker logs stock-prediction-backend
```

**3. Database Connection Issues**
```bash
# Check RDS status
aws rds describe-db-instances --db-instance-identifier <db-identifier>

# Test database connectivity from EC2
psql -h <rds-endpoint> -U <username> -d <database>
```

**4. SSL Certificate Issues**
```bash
# Check certificate status
sudo certbot certificates

# Renew certificate manually
sudo certbot renew --force-renewal

# Check Nginx configuration
sudo nginx -t
sudo systemctl reload nginx
```

### Performance Optimization

**1. Auto Scaling Tuning**
```bash
# Adjust scaling policies based on metrics
aws autoscaling put-scaling-policy --policy-name scale-up \
  --auto-scaling-group-name <asg-name> \
  --scaling-adjustment 2 \
  --adjustment-type ChangeInCapacity
```

**2. Database Performance**
- Monitor slow queries in CloudWatch
- Adjust RDS instance class if needed
- Configure read replicas for read-heavy workloads

**3. Cache Optimization**
- Monitor Redis memory usage
- Adjust cache TTL values
- Scale Redis cluster if needed

## 🧹 Cleanup

To destroy all resources:

```bash
# Destroy infrastructure
./deploy-aws.sh destroy

# Verify all resources are deleted
aws ec2 describe-instances --filters "Name=tag:Project,Values=stock-prediction"
```

## 📚 Additional Resources

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Nginx Configuration Guide](https://nginx.org/en/docs/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review CloudWatch logs and metrics
3. Consult AWS documentation
4. Create an issue in the project repository

---

**Security Note**: Always follow AWS security best practices:
- Use IAM roles with minimal required permissions
- Enable MFA for AWS accounts
- Regularly rotate access keys and passwords
- Monitor AWS CloudTrail for suspicious activity
- Keep all software and dependencies updated