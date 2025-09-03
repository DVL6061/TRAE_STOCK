# Variables for AWS Infrastructure

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "stock-prediction"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "stockprediction.example.com"
}

variable "email" {
  description = "Email for SSL certificate registration"
  type        = string
  default     = "admin@example.com"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "min_instances" {
  description = "Minimum number of instances in ASG"
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum number of instances in ASG"
  type        = number
  default     = 3
}

variable "desired_instances" {
  description = "Desired number of instances in ASG"
  type        = number
  default     = 2
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "stockdb"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "stockuser"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "public_key" {
  description = "Public key for EC2 instances"
  type        = string
}

variable "ssh_cidr_block" {
  description = "CIDR block for SSH access"
  type        = string
  default     = "0.0.0.0/0"
}

# API Keys (should be provided via environment variables or terraform.tfvars)
variable "alpha_vantage_api_key" {
  description = "Alpha Vantage API Key"
  type        = string
  sensitive   = true
}

variable "angel_one_api_key" {
  description = "Angel One API Key"
  type        = string
  sensitive   = true
}

variable "angel_one_client_id" {
  description = "Angel One Client ID"
  type        = string
  sensitive   = true
}

variable "angel_one_password" {
  description = "Angel One Password"
  type        = string
  sensitive   = true
}

variable "angel_one_totp_secret" {
  description = "Angel One TOTP Secret"
  type        = string
  sensitive   = true
}

variable "jwt_secret_key" {
  description = "JWT Secret Key for authentication"
  type        = string
  sensitive   = true
  default     = ""
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
  default     = ""
}